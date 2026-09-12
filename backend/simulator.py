"""
Background simulation engine (spec sections 5-7). Runs in its own daemon
thread so it never blocks the API server. Every SIMULATION_INTERVAL_SECONDS
it either:
  - reports a normal (or occasionally fraudulent) generation reading, or
  - moves an existing REC between entities (normal trade, or a forced fraud
    pattern: circular trading, rapid chain, dense cluster, retired reuse),
and pushes the full pipeline result out over the WebSocket.
"""
from __future__ import annotations

import logging
import random
import threading
import time
import uuid

import config
import graph_service
import models
import pipeline
import verification_service
import websocket_service
from database import SessionLocal

logger = logging.getLogger("simulator")


def _new_run_id() -> str:
    return f"RUN-{uuid.uuid4().hex[:8]}"


# Tables cleared by "Reset Transactions", children before parents. Generators
# (seed config), ML model files, contracts, and everything under backend/chain
# are deliberately NOT touched -- see Simulator.reset_all's docstring.
_RESET_TABLES = [
    models.FraudAlert,
    models.GraphAlert,
    models.RECTransaction,
    models.GraphEdge,
    models.FraudCluster,
    models.GraphEntity,
    models.RECRecord,
    models.GenerationRecord,
]

# entities with real seeded Hardhat wallets -- transfers among these actually
# hit the chain. "Clean" pool entities are off-chain-only strings; the
# pipeline gracefully marks their blockchain sync as pending (see
# pipeline._resolve_address / blockchain_service.TxResult.pending_sync).
FRAUD_RING = ["Trader: Broker X", "Trader: Broker Y", "Trader: Company B", "Trader: Company C"]
CLEAN_TRADERS = ["GreenBuy Industries", "EcoRetail Ltd", "Fairtrade Broker Co", "SunTrust Energy Buyers"]

FRAUD_SCENARIOS = [
    "OVER_ISSUANCE", "DUPLICATE_GENERATION", "CAPACITY_VIOLATION", "RETIRED_REC_REUSE",
    "CIRCULAR_TRADING", "RAPID_TRANSFER_CHAIN", "SUSPICIOUS_DENSE_CLUSTER",
]


def _capacity_factor(plant_type: str) -> float:
    hour = time.localtime().tm_hour
    if plant_type == "Solar":
        return random.uniform(0.55, 0.9) if 7 <= hour <= 18 else random.uniform(0.0, 0.05)
    if plant_type == "Wind":
        return random.uniform(0.1, 0.85)
    return random.uniform(0.45, 0.8)  # Hydro: more stable


def _normal_generation(db) -> dict:
    generators = db.query(models.Generator).filter_by(registration_status="ACTIVE").all()
    if not generators:
        return {"skipped": "no registered generators"}
    gen = random.choice(generators)
    factor = _capacity_factor(gen.plant_type)
    energy = round(gen.capacity_mw * factor, 2)
    weather = round(random.uniform(0.6, 1.0), 2)
    return pipeline.process_generation_event(db, gen.generator_id, energy, weather)


def _normal_transfer(db) -> dict:
    active = db.query(models.RECRecord).filter_by(status="ACTIVE").all()
    if not active:
        return {"skipped": "no active RECs to transfer"}
    rec = random.choice(active)
    receiver = random.choice(CLEAN_TRADERS + FRAUD_RING)
    return pipeline.process_transfer(db, rec.rec_id, rec.current_owner, receiver, rec.quantity)


def _fraud_over_issuance(db) -> dict:
    generators = db.query(models.Generator).filter_by(registration_status="ACTIVE").all()
    if not generators:
        return {"skipped": "no registered generators"}
    gen = random.choice(generators)
    energy = round(gen.capacity_mw * random.uniform(0.3, 0.6), 2)
    inflated_quantity = round(energy * random.uniform(3, 6), 2)
    return pipeline.process_generation_event(db, gen.generator_id, energy, 0.8, rec_quantity=inflated_quantity)


def _fraud_duplicate_generation(db) -> dict:
    prior = db.query(models.RECRecord).order_by(models.RECRecord.id.desc()).first()
    if prior is None:
        return _normal_generation(db)
    gen = db.query(models.Generator).filter_by(generator_id=prior.generator_id).first()
    energy = round(gen.capacity_mw * random.uniform(0.3, 0.6), 2) if gen else 50
    return pipeline.process_generation_event(
        db, prior.generator_id, energy, 0.8, reuse_generation_id=prior.generation_id, force_duplicate_flag=True,
    )


def _fraud_capacity_violation(db) -> dict:
    generators = db.query(models.Generator).filter_by(registration_status="ACTIVE").all()
    if not generators:
        return {"skipped": "no registered generators"}
    gen = random.choice(generators)
    impossible_energy = round(gen.capacity_mw * random.uniform(12, 20), 2)
    return pipeline.process_generation_event(db, gen.generator_id, impossible_energy, 1.0)


def _fraud_retired_reuse(db) -> dict:
    retired = db.query(models.RECRecord).filter_by(status="RETIRED").all()
    if not retired:
        # nothing retired yet -- retire one now so the next tick can reuse it
        active = db.query(models.RECRecord).filter_by(status="ACTIVE").first()
        if active:
            pipeline.retire_rec(db, active.rec_id, active.current_owner)
        return {"skipped": "retired a REC this tick; reuse attempt will trigger next fraud tick"}
    rec = random.choice(retired)
    return pipeline.process_transfer(db, rec.rec_id, rec.current_owner, random.choice(CLEAN_TRADERS), rec.quantity)


def _fraud_circular_trading(db) -> dict:
    """Push a REC one hop further around the fraud ring; over several ticks
    this naturally closes a cycle (Broker X -> Company B -> Broker Y -> Company C -> Broker X)."""
    active = db.query(models.RECRecord).filter(
        models.RECRecord.status == "ACTIVE", models.RECRecord.current_owner.isnot(None)
    ).all()
    ring_owned = [r for r in active if pipeline._label_for(r.current_owner) in FRAUD_RING]
    if not ring_owned:
        # seed the ring: push a fresh REC from its generator into the ring
        result = _normal_generation(db)
        rec = result.get("rec")
        if not rec or rec.get("status") != "ACTIVE":
            return result
        return pipeline.process_transfer(db, rec["rec_id"], rec["current_owner"], FRAUD_RING[0], rec["quantity"])

    rec = random.choice(ring_owned)
    current_label = pipeline._label_for(rec.current_owner)
    idx = FRAUD_RING.index(current_label) if current_label in FRAUD_RING else -1
    next_hop = FRAUD_RING[(idx + 1) % len(FRAUD_RING)]
    return pipeline.process_transfer(db, rec.rec_id, rec.current_owner, next_hop, rec.quantity)


def _fraud_rapid_transfer_chain(db) -> dict:
    active = db.query(models.RECRecord).filter_by(status="ACTIVE").all()
    if not active:
        return {"skipped": "no active RECs"}
    rec = random.choice(active)
    chain_entities = random.sample(FRAUD_RING, k=min(3, len(FRAUD_RING)))
    last_result = None
    owner = rec.current_owner
    for hop in chain_entities:
        last_result = pipeline.process_transfer(db, rec.rec_id, owner, hop, rec.quantity)
        if "error" in last_result or last_result["decision"]["decision"] != "LEGITIMATE":
            break
        owner = hop
    return last_result or {"skipped": "no hops executed"}


def _fraud_dense_cluster(db) -> dict:
    active = db.query(models.RECRecord).filter_by(status="ACTIVE").all()
    if not active:
        return {"skipped": "no active RECs"}
    rec = random.choice(active)
    a, b = random.sample(FRAUD_RING, 2)
    return pipeline.process_transfer(db, rec.rec_id, rec.current_owner, random.choice([a, b]), rec.quantity)


def _simulate_tamper(db) -> dict | None:
    """SYNTHETIC TAMPERING SIMULATION -- demo/test mode only (spec section
    11). Directly edits energy_generated_mwh/rec_quantity on a copy of an
    already-issued, ACTIVE REC's rows, exactly like an attacker with raw DB
    access would -- WITHOUT touching rec.generation_hash or anything on
    -chain, which is what makes it something the REC Verification Portal
    can actually catch later. The act of tampering is itself logged (event
    type REC_UPDATED, source=tamper_simulation, changed_by=SYSTEM_TAMPER_SIM)
    so it's visible in the audit trail immediately -- but detection (a
    TAMPER_DETECTED/HASH_MISMATCH audit entry + a CRITICAL fraud alert) only
    happens the next time someone actually verifies this REC, which is the
    whole point of the demo."""
    candidates = db.query(models.RECRecord).filter(
        models.RECRecord.status == "ACTIVE", models.RECRecord.generation_hash.isnot(None)
    ).all()
    if not candidates:
        return {"skipped": "no eligible ACTIVE RECs with an anchored hash to tamper with"}
    rec = random.choice(candidates)
    generation = db.query(models.GenerationRecord).filter_by(generation_id=rec.generation_id).first()
    if generation is None:
        return {"skipped": "REC has no linked generation record"}

    old_energy = generation.energy_generated_mwh
    old_qty = rec.quantity
    factor = round(random.uniform(1.3, 1.8), 2)
    generation.energy_generated_mwh = round(old_energy * factor, 2)
    rec.quantity = round(old_qty * factor, 2)
    # rec.generation_hash and blockchain data are deliberately left alone.

    verification_service.audit_log(
        db, rec.rec_id, "REC_UPDATED",
        old_value={"energy_generated_mwh": old_energy, "rec_quantity": old_qty},
        new_value={"energy_generated_mwh": generation.energy_generated_mwh, "rec_quantity": rec.quantity},
        changed_by="SYSTEM_TAMPER_SIM", source="tamper_simulation",
    )
    db.commit()

    payload = {
        "type": "tamper_simulated",
        "rec_id": rec.rec_id,
        "old_energy_mwh": old_energy, "new_energy_mwh": generation.energy_generated_mwh,
        "old_rec_quantity": old_qty, "new_rec_quantity": rec.quantity,
        "note": "SYNTHETIC TAMPERING SIMULATION -- demo/test mode only, not a real intrusion",
    }
    websocket_service.broadcast_sync(payload)
    return {"skipped": f"tamper-simulated REC {rec.rec_id} ({old_energy}->{generation.energy_generated_mwh} MWh) -- run Verify REC to detect it"}


FRAUD_HANDLERS = {
    "OVER_ISSUANCE": _fraud_over_issuance,
    "DUPLICATE_GENERATION": _fraud_duplicate_generation,
    "CAPACITY_VIOLATION": _fraud_capacity_violation,
    "RETIRED_REC_REUSE": _fraud_retired_reuse,
    "CIRCULAR_TRADING": _fraud_circular_trading,
    "RAPID_TRANSFER_CHAIN": _fraud_rapid_transfer_chain,
    "SUSPICIOUS_DENSE_CLUSTER": _fraud_dense_cluster,
}


class Simulator:
    """Background tick loop plus the authoritative start/stop/reset/config
    state machine. All state-changing calls (start/stop/configure/reset_all)
    go through `self._lock` (a re-entrant lock, since reset_all calls stop
    from within the same lock) so rapid Start/Stop/Reset clicks from the UI
    can't interleave into an inconsistent state (e.g. a reset racing a tick
    that's mid-write)."""

    def __init__(self):
        self.interval = config.SIMULATION_INTERVAL_SECONDS
        self.fraud_probability = config.FRAUD_PROBABILITY
        self.tamper_enabled = False
        self.tamper_probability = 0.05
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self.running = False
        self.ticks = 0
        self.generated_count = 0
        self.fraud_injected_count = 0
        self.tamper_injected_count = 0
        self.last_result: dict | None = None
        self.simulation_run_id = _new_run_id()

    def configure(self, interval_seconds: float | None = None, fraud_probability: float | None = None,
                  tamper_enabled: bool | None = None, tamper_probability: float | None = None) -> None:
        with self._lock:
            if interval_seconds is not None:
                self.interval = max(1.0, interval_seconds)
            if fraud_probability is not None:
                self.fraud_probability = min(max(fraud_probability, 0.0), 1.0)
            if tamper_enabled is not None:
                self.tamper_enabled = bool(tamper_enabled)
            if tamper_probability is not None:
                self.tamper_probability = min(max(tamper_probability, 0.0), 1.0)

    def start(self, interval_seconds: float | None = None, fraud_probability: float | None = None,
              tamper_enabled: bool | None = None, tamper_probability: float | None = None) -> None:
        """Optionally applies config first, so "Start Simulation" always uses
        whatever the user currently has selected -- fixes the bug where the
        fraud-probability slider only took effect after a separate "Apply"
        click, so a fresh run silently kept using the previous value."""
        with self._lock:
            self.configure(interval_seconds, fraud_probability, tamper_enabled, tamper_probability)
            if self.running:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self.running = True
            self._thread.start()
            logger.info(
                "Simulator started (run=%s interval=%.1fs, fraud_probability=%.2f)",
                self.simulation_run_id, self.interval, self.fraud_probability,
            )

    def stop(self, wait: bool = False, timeout: float = 20.0) -> None:
        with self._lock:
            was_running = self.running
            self.running = False
            self._stop_event.set()
            thread = self._thread
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if was_running:
            logger.info("Simulator stopped")

    def status(self) -> dict:
        observed_fraud_rate = (
            round(self.fraud_injected_count / self.generated_count, 4) if self.generated_count else 0.0
        )
        return {
            "running": self.running,
            "interval_seconds": self.interval,
            "fraud_probability": self.fraud_probability,
            "tamper_enabled": self.tamper_enabled,
            "tamper_probability": self.tamper_probability,
            "ticks": self.ticks,
            "generated_count": self.generated_count,
            "fraud_injected_count": self.fraud_injected_count,
            "tamper_injected_count": self.tamper_injected_count,
            "observed_fraud_rate": observed_fraud_rate,
            "simulation_run_id": self.simulation_run_id,
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception("simulator tick failed")
            self._stop_event.wait(self.interval)

    def tick(self) -> dict:
        db = SessionLocal()
        try:
            self.ticks += 1
            is_fraud = random.random() < self.fraud_probability
            if is_fraud:
                scenario = random.choice(FRAUD_SCENARIOS)
                result = FRAUD_HANDLERS[scenario](db)
                result = result or {}
                result["injected_scenario"] = scenario
            else:
                result = _normal_generation(db) if random.random() < 0.65 else _normal_transfer(db)

            result["tick"] = self.ticks
            result["is_fraud_injected"] = is_fraud
            result["simulation_run_id"] = self.simulation_run_id
            self.last_result = result

            if "skipped" not in result:
                self.generated_count += 1
                if is_fraud:
                    self.fraud_injected_count += 1

            # Tampering is independent of the fraud-scenario roll above --
            # it edits an *already-issued* REC after the fact rather than
            # generating a new fraudulent event, so it's rolled separately
            # and doesn't count toward generated_count/fraud_injected_count.
            if self.tamper_enabled and random.random() < self.tamper_probability:
                tamper_result = _simulate_tamper(db)
                if tamper_result:
                    self.tamper_injected_count += 1
                    result.setdefault("tamper_simulation", tamper_result)

            graph_engine = graph_service.get_engine()
            graph_summary = graph_engine.analyze(db)

            websocket_service.broadcast_sync({"type": "pipeline_result", "data": result})
            _broadcast_transaction_created(result)
            _broadcast_graph_update(graph_summary)
            return result
        finally:
            db.close()

    def reset_all(self, db) -> dict:
        """Reset Transactions: stops the simulator (waiting for any in-flight
        tick to finish first, so we never delete rows out from under a
        write), clears every simulator-generated table, resets the
        in-memory graph and this simulator's counters, and starts a fresh
        `simulation_run_id`.

        Left untouched on purpose: `generators` (seed config, not simulated
        activity), backend/models/*.pkl (ML models), contracts/ (Solidity
        source), backend/chain/*.json (deployed contract address/ABI/wallets)
        and config.py. The local Hardhat chain itself can't be reset without
        restarting the node -- its old REC entries stay on that immutable
        ledger, but nothing in the (now-empty) SQLite DB references them
        anymore, so they never resurface in the UI; see README "Resetting
        the Simulation" for the full explanation.

        Runs the deletes in one DB transaction: if anything raises partway
        through, the whole reset rolls back rather than leaving some tables
        cleared and others not.
        """
        with self._lock:
            self.stop(wait=True)
            counts: dict[str, int] = {}
            try:
                for model in _RESET_TABLES:
                    counts[model.__tablename__] = db.query(model).count()
                    db.query(model).delete(synchronize_session=False)
                db.commit()
            except Exception:
                db.rollback()
                raise

            graph_service.get_engine().reset()
            self.ticks = 0
            self.generated_count = 0
            self.fraud_injected_count = 0
            self.tamper_injected_count = 0
            self.last_result = None
            self.simulation_run_id = _new_run_id()
            logger.info("Simulation data reset (new run=%s)", self.simulation_run_id)
            return counts


def _broadcast_transaction_created(result: dict) -> None:
    """Best-effort second broadcast in the shape the frontend spec calls
    for, alongside the richer "pipeline_result" event the dashboard already
    consumes -- additive only, never required for the UI to function."""
    if "skipped" in result or "error" in result:
        return
    rec = result.get("rec") or {}
    tx = result.get("transaction") or {}
    decision = result.get("decision") or {}
    ml = result.get("ml") or {}
    graph = result.get("graph") or {}
    websocket_service.broadcast_sync({
        "type": "transaction_created",
        "transaction_id": tx.get("transaction_id"),
        "rec_id": rec.get("rec_id") or tx.get("rec_id"),
        "ml_score": ml.get("ml_score"),
        "graph_score": graph.get("graph_score"),
        "final_risk_score": decision.get("final_risk_score"),
        "risk_level": decision.get("risk_level"),
        "blockchain_status": rec.get("blockchain_status") or tx.get("blockchain_status"),
        "blockchain_tx_hash": rec.get("blockchain_tx_hash") or tx.get("blockchain_tx_hash"),
        "fraud_type": result.get("injected_scenario") if result.get("is_fraud_injected") else None,
    })


_ALERT_KEY_EVENT = {
    "scc": "fraud_ring_detected",
    "community": "community_detected",
    "hub": "motif_detected",
    "dup": "motif_detected",
    "mismatch": "motif_detected",
    "temporal": "motif_detected",
}


def _broadcast_graph_update(summary: dict) -> None:
    """Pushes the aggregate `graph_update` event every tick (spec section
    13), plus one specific event per genuinely NEW graph_alerts row this
    analyze() call created (fraud_ring_detected / community_detected /
    motif_detected) -- not every tick a pattern merely *remains* true,
    since `_sync_graph_alerts` dedupes by key and only reports freshly
    created ones. Granular graph_node_added/graph_edge_added/
    graph_risk_updated events are intentionally not emitted per-node/edge --
    analyze() recomputes the whole graph rather than tracking incremental
    diffs, and the aggregate graph_update already covers what the dashboard
    needs to refresh live."""
    websocket_service.broadcast_sync({
        "type": "graph_update",
        "run_id": summary["run_id"],
        "node_count": summary["node_count"],
        "edge_count": summary["edge_count"],
        "suspicious_nodes": summary["suspicious_nodes"],
        "suspicious_edges": summary["suspicious_edges"],
        "fraud_rings": summary["fraud_rings"],
        "suspicious_communities": summary["suspicious_communities"],
    })

    seen_events = set()
    for key in summary.get("new_alert_keys", []):
        prefix = key.split(":", 1)[0]
        event = _ALERT_KEY_EVENT.get(prefix)
        if not event or event in seen_events:
            continue
        seen_events.add(event)
        websocket_service.broadcast_sync({
            "type": event, "run_id": summary["run_id"], "key": key,
        })


_simulator = Simulator()


def get_simulator() -> Simulator:
    return _simulator
