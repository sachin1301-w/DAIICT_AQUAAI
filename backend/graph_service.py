"""
Graph-based fraud detection over the REC ownership/transfer graph.

Nodes = entities (generators, brokers, traders, companies). Edges = REC
transfers. Rebuilt from `rec_transactions` on every call, since the graph
changes continuously as the simulator runs.

NOTE: ISSUE events are deliberately NOT turned into edges from a shared
"origin" node -- doing that would connect every unrelated plant through one
hub and merge clean, unrelated ownership chains into a single false
"cluster". Only TRANSFER edges count toward clustering/cycle/density
analysis (verified against a real bug hit during an earlier iteration of
this project).
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from collections import defaultdict

import networkx as nx
from sqlalchemy import func
from sqlalchemy.orm import Session

import config
import gnn_service
import models
import rules_engine

RAPID_TRANSFER_WINDOW_SECONDS = 60 * 60 * 6  # 6 hours
HIGH_DEGREE_THRESHOLD = 4

# Order matters -- ml_training/train_gnn.py and backend/gnn_service.py both
# read features in exactly this order (see GraphFraudEngine.gnn_node_features).
GNN_FEATURE_ORDER = [
    "degree", "in_degree", "out_degree", "total_volume",
    "transaction_count", "avg_quantity", "clustering_coefficient", "avg_time_gap_seconds",
]

GRAPH_SCORE_WEIGHTS = {
    "circular_ownership": 35,
    "dense_suspicious_cluster": 25,
    "rapid_transfer_chain": 20,
    "high_degree_hub": 10,
    "repeated_edges": 10,
}


def _new_run_id() -> str:
    return f"GRUN-{uuid.uuid4().hex[:8]}"


def _risk_level(score: float, thresholds: dict = config.GRAPH_RISK_LEVEL_THRESHOLDS) -> str:
    for level, (lo, hi) in thresholds.items():
        if lo <= score <= hi:
            return level
    return "LOW"


def _classify_entity(entity_id: str) -> str:
    """Best-effort node type classification for display/DB purposes -- entity
    ids are either a generator's synthetic id, a wallet label like
    'Trader: Broker X', a raw 0x address, or a simulator-invented company
    name. There's no strict registry for the last case, so this is a
    heuristic, not authoritative."""
    lowered = entity_id.lower()
    if entity_id.startswith("GEN-") or "generator" in lowered:
        return "GENERATOR"
    if "broker" in lowered:
        return "BROKER"
    if "trader" in lowered or "company" in lowered:
        return "TRADER"
    if entity_id.startswith("0x"):
        return "WALLET"
    return "COMPANY"


class GraphFraudEngine:
    """A module-level singleton (see get_engine() below), read and mutated
    from both the simulator's background thread (every tick) and FastAPI
    request-handler threads (every /api/graph/* call) -- genuinely
    concurrent, not just "later". `self._lock` (re-entrant, since analyze()
    calls build() on itself) serializes build()/analyze()/reset() so two
    threads can never interleave writes into the same graph object. build()
    additionally populates a local graph first and only swaps it into
    `self.graph` once fully built, rather than mutating the shared
    attribute in place -- verified directly against a real bug: without
    this, a concurrent build() call reassigning `self.graph` mid-loop could
    cause this thread's remaining add_edge() calls to land on the OTHER
    thread's new graph object, inserting the same transaction as a
    duplicate parallel edge (observed as vis-network's frontend DataSet
    rejecting a second edge with an id it had already seen)."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.run_id = _new_run_id()
        self._lock = threading.RLock()
        # Tracks what's already been broadcast as a "newly detected"
        # fraud-ring/community/motif so simulator.tick() doesn't re-announce
        # the same finding on every single tick while it remains true.
        self._announced: set[str] = set()

    # ---------------- build ----------------

    def reset(self, new_run_id: bool = True) -> None:
        """Drop all in-memory graph state. `build()` already rebuilds the
        graph from scratch on every call (see module docstring), so this is
        mostly belt-and-braces for the moment right after a data reset,
        before anything has called build() again -- callers that read
        `self.graph` directly (rather than through build()) still see an
        empty graph instead of the previous run's stale state."""
        with self._lock:
            self.graph = nx.MultiDiGraph()
            self._announced = set()
            if new_run_id:
                self.run_id = _new_run_id()

    def build(self, db: Session) -> None:
        g = nx.MultiDiGraph()  # built locally, swapped in atomically at the end -- see class docstring
        transactions = db.query(models.RECTransaction).order_by(models.RECTransaction.transaction_timestamp).all()
        for tx in transactions:
            if tx.transaction_type == "TRANSFER" and tx.sender and tx.receiver:
                g.add_node(tx.sender)
                g.add_node(tx.receiver)
                g.add_edge(
                    tx.sender, tx.receiver,
                    rec_id=tx.rec_id, timestamp=tx.transaction_timestamp, transaction_id=tx.transaction_id,
                    relationship_type="TRANSFERRED", quantity=tx.quantity,
                    blockchain_status=tx.blockchain_status, blockchain_tx_hash=tx.blockchain_tx_hash,
                )
            elif tx.transaction_type == "ISSUE" and tx.receiver:
                g.add_node(tx.receiver)
            # RETIRE transactions deliberately do NOT become edges in this
            # analytical graph -- same reasoning as ISSUE above, amplified:
            # a single shared "Retirement" node (or even one node per REC)
            # would sit in the entity/hub/community tables as a fake
            # high-degree participant unrelated to any real fraud pattern.
            # Retirement is still fully visible in per-REC timelines
            # (rec_transactions is the source of truth for that) and in
            # `to_network_json`'s optional display-only terminal nodes.
        with self._lock:
            self.graph = g

    # ---------------- entity-level metrics ----------------

    def entity_metrics(self) -> dict[str, dict]:
        if self.graph.number_of_nodes() == 0:
            return {}
        undirected = nx.Graph(self.graph)
        betweenness = nx.betweenness_centrality(undirected) if undirected.number_of_nodes() > 2 else {n: 0 for n in undirected}
        closeness = nx.closeness_centrality(undirected)
        clustering = nx.clustering(undirected)
        metrics = {}
        for n in self.graph.nodes():
            metrics[n] = {
                "degree": self.graph.in_degree(n) + self.graph.out_degree(n),
                "in_degree": self.graph.in_degree(n),
                "out_degree": self.graph.out_degree(n),
                "betweenness_centrality": round(betweenness.get(n, 0), 4),
                "closeness_centrality": round(closeness.get(n, 0), 4),
                "clustering_coefficient": round(clustering.get(n, 0), 4),
            }
        return metrics

    def sync_entities_to_db(self, db: Session) -> None:
        """Persists per-entity graph metrics into graph_entities (upsert by
        entity_id), same rationale as sync_clusters_to_db: the graph is
        rebuilt live on every call, this just keeps a queryable snapshot."""
        metrics = self.entity_metrics()
        for entity_id, m in metrics.items():
            row = db.query(models.GraphEntity).filter_by(entity_id=entity_id).first()
            if row is None:
                row = models.GraphEntity(entity_id=entity_id, entity_type=_classify_entity(entity_id))
                db.add(row)
            row.degree = m["degree"]
            row.betweenness = m["betweenness_centrality"]
            row.risk_score = min(m["degree"] * 10 + m["betweenness_centrality"] * 100, 100)
        db.commit()

    # ---------------- individual detectors ----------------

    def circular_ownership(self) -> list[list[str]]:
        simple = nx.DiGraph(self.graph)
        return [c for c in nx.simple_cycles(simple) if len(c) > 1]

    def repeated_transfer_pairs(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for u, v in self.graph.edges():
            counts[(u, v)] += 1
        return {pair: c for pair, c in counts.items() if c > 1}

    def high_degree_hubs(self, threshold: int = HIGH_DEGREE_THRESHOLD) -> list[str]:
        return [n for n in self.graph.nodes() if self.graph.in_degree(n) + self.graph.out_degree(n) >= threshold]

    def rapid_transfer_chains(self, window_seconds: float = RAPID_TRANSFER_WINDOW_SECONDS) -> list[list[str]]:
        simple = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if simple.has_edge(u, v):
                simple[u][v]["timestamps"].append(data["timestamp"])
            else:
                simple.add_edge(u, v, timestamps=[data["timestamp"]])

        chains = []
        for start in simple.nodes():
            path, node, visited = [start], start, set()
            while True:
                candidates = [(v, min(d["timestamps"])) for v, d in simple[node].items() if (node, v) not in visited]
                if not candidates:
                    break
                nxt, ts = min(candidates, key=lambda c: c[1])
                if len(path) > 1:
                    prev_ts = simple[path[-2]][path[-1]]["timestamps"][0]
                    if ts - prev_ts > window_seconds:
                        break
                visited.add((node, nxt))
                path.append(nxt)
                node = nxt
                if len(path) > 8:
                    break
            if len(path) >= 4:
                chains.append(path)
        return chains

    def communities(self) -> list[set[str]]:
        undirected = nx.Graph(self.graph)
        if undirected.number_of_edges() == 0:
            return []
        try:
            return [set(c) for c in nx.algorithms.community.greedy_modularity_communities(undirected)]
        except Exception:
            return [set(c) for c in nx.connected_components(undirected)]

    # ---------------- cluster-level scoring ----------------

    def suspicious_clusters(self) -> list[dict]:
        undirected = nx.Graph(self.graph)
        cycles = self.circular_ownership()
        hubs = set(self.high_degree_hubs())
        repeated_pairs = self.repeated_transfer_pairs()
        chains = self.rapid_transfer_chains()

        clusters = []
        for i, component in enumerate(nx.connected_components(undirected)):
            if len(component) < 2:
                continue
            sub = self.graph.subgraph(component)
            entities = len(component)
            transfers = sub.number_of_edges()
            rec_ids = {d["rec_id"] for _, _, d in sub.edges(data=True)}

            has_cycle = any(set(c).issubset(component) for c in cycles)
            has_repeats = any(u in component and v in component for (u, v) in repeated_pairs)
            has_hub = bool(hubs & component)
            has_rapid_chain = any(set(chain) & component for chain in chains)

            # A plain ownership chain (tree) needs exactly (entities-1) transfers;
            # extra transfers beyond that indicate real density, not an artifact
            # of small-chain topology (see module docstring).
            extra_edges = transfers - (entities - 1)
            has_dense_cluster = entities >= 4 and extra_edges >= 2
            density = transfers / (entities * (entities - 1)) if entities > 1 else 0

            if not (has_cycle or has_repeats or has_hub or has_dense_cluster or has_rapid_chain):
                continue

            score = 0
            patterns = []
            if has_cycle:
                score += GRAPH_SCORE_WEIGHTS["circular_ownership"]
                patterns.append("Circular ownership")
            if has_dense_cluster:
                score += GRAPH_SCORE_WEIGHTS["dense_suspicious_cluster"]
                patterns.append("Dense suspicious cluster")
            if has_rapid_chain:
                score += GRAPH_SCORE_WEIGHTS["rapid_transfer_chain"]
                patterns.append("Rapid transfer chain")
            if has_hub:
                score += GRAPH_SCORE_WEIGHTS["high_degree_hub"]
                patterns.append("Unusually high-degree broker")
            if has_repeats:
                score += GRAPH_SCORE_WEIGHTS["repeated_edges"]
                patterns.append("Repeated transfers")

            clusters.append({
                "cluster_id": f"FRAUD-{i:03d}",
                "entities_involved": entities,
                "entity_ids": sorted(component),
                "recs_involved": len(rec_ids),
                "transfers_involved": transfers,
                "graph_score": min(score, 100),
                "risk_score": min(score, 100),
                "cluster_density": round(density, 3),
                "patterns": patterns,
            })

        clusters.sort(key=lambda c: c["graph_score"], reverse=True)
        return clusters

    def sync_clusters_to_db(self, db: Session, clusters: list[dict] | None = None) -> None:
        """Persists the live-computed clusters into fraud_clusters (upsert by
        cluster_id), and closes out any previously-recorded cluster that's no
        longer suspicious. Clusters are still computed live on every request
        (the graph changes every tick) -- this just keeps a queryable,
        historical record matching the fraud_clusters schema."""
        import json

        if clusters is None:
            clusters = self.suspicious_clusters()

        current_ids = set()
        for c in clusters:
            current_ids.add(c["cluster_id"])
            row = db.query(models.FraudCluster).filter_by(cluster_id=c["cluster_id"]).first()
            if row is None:
                row = models.FraudCluster(cluster_id=c["cluster_id"])
                db.add(row)
            row.entity_count = c["entities_involved"]
            row.rec_count = c["recs_involved"]
            row.transfer_count = c["transfers_involved"]
            row.graph_score = c["graph_score"]
            row.risk_score = c["risk_score"]
            row.fraud_pattern = json.dumps(c["patterns"])
            row.status = "ACTIVE"

        stale = db.query(models.FraudCluster).filter(
            models.FraudCluster.status == "ACTIVE", ~models.FraudCluster.cluster_id.in_(current_ids or [""])
        ).all()
        for row in stale:
            row.status = "RESOLVED"

        db.commit()

    def transaction_graph_features(self, source: str, target: str) -> dict:
        """Per-transaction graph features fed into ml_service.score_transaction."""
        metrics = self.entity_metrics()
        src_m = metrics.get(source, {"degree": 0, "betweenness_centrality": 0})
        tgt_m = metrics.get(target, {"degree": 0, "betweenness_centrality": 0})

        cycles = self.circular_ownership()
        cycle_count = sum(1 for c in cycles if source in c and target in c)

        community_size = 1
        for community in self.communities():
            if source in community or target in community:
                community_size = max(community_size, len(community))

        clusters = self.suspicious_clusters()
        cluster_density = 0.0
        repeated_edge_count = 0
        for cluster in clusters:
            if source in cluster["entity_ids"] or target in cluster["entity_ids"]:
                cluster_density = max(cluster_density, cluster["cluster_density"])
        repeated_pairs = self.repeated_transfer_pairs()
        repeated_edge_count = repeated_pairs.get((source, target), 0)

        return {
            "source_degree": src_m["degree"],
            "target_degree": tgt_m["degree"],
            "source_betweenness": src_m.get("betweenness_centrality", 0),
            "target_betweenness": tgt_m.get("betweenness_centrality", 0),
            "cycle_count": cycle_count,
            "community_size": community_size,
            "cluster_density": cluster_density,
            "repeated_edge_count": repeated_edge_count,
        }

    # ==================================================================== Advanced Graph Fraud Detection
    #
    # Everything below this line is the extended analysis pipeline: Tarjan
    # SCC (circular trading), Louvain communities, PageRank, temporal burst
    # detection, and named motif detection, combined into one explainable
    # per-entity risk score and persisted (tagged with self.run_id) for the
    # /api/graph/* endpoints and the Graph Fraud Detection dashboard page.
    # The simpler methods above (circular_ownership, communities,
    # suspicious_clusters, ...) are untouched and still power the live
    # per-transaction scoring path in pipeline.py -- this section builds on
    # top of them rather than replacing them, so the already-tested
    # transaction pipeline can't be destabilized by this addition.

    def strongly_connected_components(self) -> list[list[str]]:
        """Tarjan's algorithm (networkx's `strongly_connected_components`
        is a Tarjan implementation) over the directed graph -- a proper SCC,
        not just the elementary cycles `circular_ownership()` enumerates.
        Every node in a >1-member SCC can reach every other node in it, i.e.
        a REC can flow in a closed loop through the whole group, even if no
        single simple cycle visits every member."""
        simple = nx.DiGraph(self.graph)
        return [sorted(c) for c in nx.strongly_connected_components(simple) if len(c) > 1]

    def pagerank(self) -> dict[str, float]:
        if self.graph.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(nx.DiGraph(self.graph))
        except Exception:
            return {n: 0.0 for n in self.graph.nodes()}

    def communities_louvain(self) -> list[set[str]]:
        """Louvain community detection (networkx's native `leiden_communities`
        requires an external backend that isn't installed in this project --
        see graph_service module notes -- so this uses Louvain, which the
        spec explicitly allows as the fallback and networkx implements
        natively)."""
        undirected = nx.Graph(self.graph)
        if undirected.number_of_edges() == 0:
            return []
        try:
            return [set(c) for c in nx.algorithms.community.louvain_communities(undirected, seed=42)]
        except Exception:
            return self.communities()

    def temporal_bursts(self) -> list[dict]:
        """Rapid/burst transfer detection (spec section 6): more than
        `TEMPORAL_MAX_TRANSFERS_IN_WINDOW` movements of the *same REC*, or
        touching the *same entity*, inside `TEMPORAL_RAPID_WINDOW_SECONDS`.
        Thresholds come from config.py, not hardcoded here."""
        window = config.TEMPORAL_RAPID_WINDOW_SECONDS
        max_transfers = config.TEMPORAL_MAX_TRANSFERS_IN_WINDOW
        bursts: list[dict] = []

        by_rec: dict[str, list[tuple]] = defaultdict(list)
        for u, v, d in self.graph.edges(data=True):
            if d.get("rec_id") and d.get("timestamp") is not None:
                by_rec[d["rec_id"]].append((d["timestamp"], u, v, d.get("transaction_id")))
        for rec_id, events in by_rec.items():
            events.sort(key=lambda e: e[0])
            if len(events) < max_transfers:
                continue
            for i in range(len(events) - max_transfers + 1):
                window_events = events[i:i + max_transfers]
                span = window_events[-1][0] - window_events[0][0]
                if span <= window:
                    bursts.append({
                        "type": "rec_velocity", "rec_id": rec_id,
                        "transfer_count": len(window_events), "window_seconds": round(span, 1),
                        "entities": sorted({e[1] for e in window_events} | {e[2] for e in window_events}),
                        "transaction_ids": [e[3] for e in window_events if e[3]],
                        "start": window_events[0][0], "end": window_events[-1][0],
                    })
                    break

        by_entity: dict[str, list[float]] = defaultdict(list)
        for u, v, d in self.graph.edges(data=True):
            if d.get("timestamp") is not None:
                by_entity[u].append(d["timestamp"])
                by_entity[v].append(d["timestamp"])
        for entity, timestamps in by_entity.items():
            timestamps.sort()
            if len(timestamps) < max_transfers:
                continue
            for i in range(len(timestamps) - max_transfers + 1):
                span = timestamps[i + max_transfers - 1] - timestamps[i]
                if span <= window:
                    bursts.append({
                        "type": "entity_velocity", "entity_id": entity,
                        "transfer_count": max_transfers, "window_seconds": round(span, 1),
                        "start": timestamps[i], "end": timestamps[i + max_transfers - 1],
                    })
                    break
        return bursts

    def motifs(self, db: Session) -> dict[str, list[dict]]:
        """Named suspicious subgraph patterns (spec section 7, motifs A-E)."""
        results: dict[str, list[dict]] = {
            "circular_trading": [], "duplicate_transfer": [], "rapid_relay": [],
            "suspicious_hub": [], "generator_claim_mismatch": [],
        }

        for cycle in self.circular_ownership():
            results["circular_trading"].append({
                "entities": cycle, "cycle_length": len(cycle),
                "reason": f"Closed transfer loop: {' -> '.join(cycle + [cycle[0]])}",
                "confidence": min(40 + len(cycle) * 10, 95),
            })

        dup_rows = (
            db.query(models.RECRecord.generation_id, func.count(models.RECRecord.id))
            .group_by(models.RECRecord.generation_id)
            .having(func.count(models.RECRecord.id) > 1)
            .all()
        )
        for generation_id, count in dup_rows:
            recs = db.query(models.RECRecord).filter_by(generation_id=generation_id).all()
            results["duplicate_transfer"].append({
                "generation_id": generation_id, "rec_ids": [r.rec_id for r in recs],
                "generator_id": recs[0].generator_id if recs else None,
                "reason": f"{count} RECs issued against the same generation record ({generation_id})",
                "confidence": min(50 + count * 15, 95),
            })

        for chain in self.rapid_transfer_chains():
            results["rapid_relay"].append({
                "entities": chain, "chain_length": len(chain),
                "reason": f"REC relayed through {len(chain)} entities in quick succession",
                "confidence": min(30 + len(chain) * 10, 90),
            })

        for n in self.graph.nodes():
            indeg, outdeg = self.graph.in_degree(n), self.graph.out_degree(n)
            if indeg >= 2 and outdeg >= 2 and (indeg + outdeg) >= config.GRAPH_HIGH_DEGREE_THRESHOLD:
                results["suspicious_hub"].append({
                    "entity_id": n, "in_degree": indeg, "out_degree": outdeg,
                    "reason": f"{n} both receives from {indeg} and sends to {outdeg} distinct entities -- a pass-through hub shape",
                    "confidence": min(40 + (indeg + outdeg) * 5, 90),
                })

        rec_gen_pairs = (
            db.query(models.RECRecord, models.GenerationRecord)
            .join(models.GenerationRecord, models.RECRecord.generation_id == models.GenerationRecord.generation_id)
            .all()
        )
        for rec, gen in rec_gen_pairs:
            if gen.energy_generated_mwh and rec.quantity > gen.energy_generated_mwh * rules_engine.OVER_ISSUANCE_TOLERANCE:
                results["generator_claim_mismatch"].append({
                    "rec_id": rec.rec_id, "generator_id": rec.generator_id,
                    "claimed_quantity": rec.quantity, "eligible_generation": gen.energy_generated_mwh,
                    "reason": f"REC quantity {rec.quantity} exceeds eligible generation {gen.energy_generated_mwh} MWh",
                    "confidence": min(50 + (rec.quantity / max(gen.energy_generated_mwh, 0.01)) * 10, 95),
                })

        return results

    # ---------------- community classification ----------------

    @staticmethod
    def _classify_community(density: float, alert_count: int, cycle_count: int, hub_count: int) -> str:
        score = 0
        if density > 0.3:
            score += 1
        if alert_count >= 2:
            score += 1
        if cycle_count >= 1:
            score += 2
        if hub_count >= 1:
            score += 1
        if score >= 4:
            return "HIGH_RISK_FRAUD_CLUSTER"
        if score >= 2:
            return "SUSPICIOUS"
        if score >= 1:
            return "WATCHLIST"
        return "NORMAL"

    @staticmethod
    def _community_reason(level: str, density: float, alert_count: int, cycle_count: int, hub_count: int) -> str:
        if level == "NORMAL":
            return "No suspicious patterns detected in this group of connected entities."
        bits = []
        if cycle_count:
            bits.append(f"{cycle_count} circular trading loop(s)")
        if alert_count:
            bits.append(f"{alert_count} prior fraud alert(s)")
        if hub_count:
            bits.append(f"{hub_count} high-degree hub(s)")
        if density > 0.3:
            bits.append(f"high internal density ({density:.0%})")
        return f"{level.replace('_', ' ').title()}: " + ", ".join(bits)

    def _alert_counts_by_entity(self, db: Session) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        tx_entities: dict[str, tuple[str, str]] = {}
        for u, v, d in self.graph.edges(data=True):
            if d.get("transaction_id"):
                tx_entities[d["transaction_id"]] = (u, v)
        rec_generator_cache: dict[str, str | None] = {}
        for alert in db.query(models.FraudAlert).all():
            if alert.transaction_id and alert.transaction_id in tx_entities:
                u, v = tx_entities[alert.transaction_id]
                counts[u] += 1
                counts[v] += 1
            elif alert.rec_id:
                if alert.rec_id not in rec_generator_cache:
                    rec = db.query(models.RECRecord).filter_by(rec_id=alert.rec_id).first()
                    rec_generator_cache[alert.rec_id] = rec.generator_id if rec else None
                gen_id = rec_generator_cache.get(alert.rec_id)
                if gen_id:
                    counts[gen_id] += 1
        return counts

    def _tampered_entities(self, db: Session) -> set[str]:
        tampered_rec_ids = {
            r.rec_id for r in db.query(models.RECVerificationRequest.rec_id).filter_by(result="TAMPERED").distinct()
        }
        if not tampered_rec_ids:
            return set()
        entities: set[str] = set()
        for u, v, d in self.graph.edges(data=True):
            if d.get("rec_id") in tampered_rec_ids:
                entities.add(u)
                entities.add(v)
        recs = db.query(models.RECRecord).filter(models.RECRecord.rec_id.in_(tampered_rec_ids)).all()
        for r in recs:
            if r.current_owner:
                entities.add(r.current_owner)
            entities.add(r.generator_id)
        return entities

    def community_rows(self, db: Session, comms: list[set[str]] | None = None,
                        sccs: list[list[str]] | None = None) -> list[dict]:
        comms = self.communities_louvain() if comms is None else comms
        sccs = self.strongly_connected_components() if sccs is None else sccs
        hubs = set(self.high_degree_hubs())
        alert_counts = self._alert_counts_by_entity(db)

        rows = []
        for i, community in enumerate(comms):
            if len(community) < 2:
                continue
            sub = self.graph.subgraph(community)
            edge_count = sub.number_of_edges()
            rec_ids = {d["rec_id"] for _, _, d in sub.edges(data=True) if d.get("rec_id")}
            volume = sum(d.get("quantity") or 0 for _, _, d in sub.edges(data=True))
            possible_edges = len(community) * (len(community) - 1)
            density = edge_count / possible_edges if possible_edges else 0
            community_cycles = [c for c in sccs if set(c).issubset(community)]
            suspicious_brokers = hubs & community
            community_alerts = sum(alert_counts.get(e, 0) for e in community)

            level = self._classify_community(density, community_alerts, len(community_cycles), len(suspicious_brokers))
            cluster_id = f"COMM-{i:03d}-{self.run_id[-6:]}"
            rows.append({
                "cluster_id": cluster_id, "cluster_type": "COMMUNITY", "entity_ids": sorted(community),
                "entity_count": len(community), "edge_count": edge_count, "rec_count": len(rec_ids),
                "transfer_count": edge_count, "total_volume": round(volume, 2), "density": round(density, 3),
                "cycle_count": len(community_cycles), "suspicious_broker_count": len(suspicious_brokers),
                "alert_count": community_alerts, "risk_level": level,
                "graph_score": {"NORMAL": 10, "WATCHLIST": 35, "SUSPICIOUS": 65, "HIGH_RISK_FRAUD_CLUSTER": 90}[level],
                "detection_reason": self._community_reason(level, density, community_alerts, len(community_cycles), len(suspicious_brokers)),
            })
        return rows

    def combined_entity_risk(self, entity_id: str, *, sccs: list[list[str]], community_label: str | None,
                              hub_set: set[str], betweenness: dict[str, float], motif_hits: dict[str, set],
                              alert_count: int) -> tuple[float, list[str]]:
        """Per-entity weighted graph risk score (spec section 8's table),
        capped at 100. Every contribution is independent and explainable --
        `reasons` is what the frontend shows verbatim in the investigation
        panel, never just a bare number."""
        weights = config.GRAPH_RISK_WEIGHTS
        score = 0.0
        reasons: list[str] = []

        if any(entity_id in c for c in sccs):
            score += weights["circular_trading"]
            reasons.append("Part of a circular trading loop (strongly connected component)")
        if entity_id in motif_hits.get("duplicate_transfer_entities", set()):
            score += weights["duplicate_transfer"]
            reasons.append("Involved in a duplicate REC transfer (reused generation record)")
        if entity_id in hub_set:
            score += weights["high_degree_centrality"]
            reasons.append("Unusually high number of connections")
        if betweenness.get(entity_id, 0) >= config.GRAPH_HIGH_BETWEENNESS_THRESHOLD:
            score += weights["high_betweenness_centrality"]
            reasons.append("Acts as a bridge between otherwise separate groups")
        if community_label in ("SUSPICIOUS", "HIGH_RISK_FRAUD_CLUSTER"):
            score += weights["suspicious_community"]
            reasons.append(f"Member of a {community_label.replace('_', ' ').lower()} community")
        if entity_id in motif_hits.get("temporal_entities", set()):
            score += weights["rapid_transfer_velocity"]
            reasons.append("Involved in rapid/burst transfer activity")
        if entity_id in motif_hits.get("motif_entities", set()):
            score += weights["motif_match"]
            reasons.append("Matched a known suspicious transaction pattern")
        if alert_count > 0:
            score += min(weights["previous_fraud_alerts"], alert_count * 5)
            reasons.append(f"{alert_count} prior fraud alert(s) involving this entity")
        if entity_id in motif_hits.get("tampered_entities", set()):
            score += weights["rec_tampering"]
            reasons.append("Associated with a confirmed REC tampering event")
        if entity_id in motif_hits.get("gnn_flagged_entities", set()):
            score += weights["gnn_anomaly"]
            gnn_prob = motif_hits.get("gnn_scores", {}).get(entity_id)
            reasons.append(
                f"GNN model flagged this entity as structurally anomalous (probability {gnn_prob:.0%})"
                if gnn_prob is not None else "GNN model flagged this entity as structurally anomalous"
            )

        return min(score, 100), reasons

    # ---------------- persistence ----------------

    def _sync_entities(self, db: Session, entity_rows: list[dict]) -> None:
        now = time.time()
        for r in entity_rows:
            row = db.query(models.GraphEntity).filter_by(entity_id=r["entity_id"]).first()
            if row is None:
                row = models.GraphEntity(entity_id=r["entity_id"], created_at=now)
                db.add(row)
            row.entity_type = r["entity_type"]
            row.entity_name = r["entity_id"]
            row.degree = r["degree"]
            row.in_degree = r["in_degree"]
            row.out_degree = r["out_degree"]
            row.betweenness = r["betweenness"]
            row.pagerank_score = r["pagerank"]
            row.gnn_risk_score = r["gnn_score"]
            row.community_id = r["community_id"]
            row.total_rec_volume = r["total_rec_volume"]
            row.transaction_count = r["transaction_count"]
            row.risk_score = r["risk_score"]
            row.risk_level = r["risk_level"]
            row.alert_count = r["alert_count"]
            row.run_id = self.run_id
            row.updated_at = now
        db.commit()

    def _sync_edges(self, db: Session) -> None:
        """Upserts by transaction_id (stable and unique), since the graph
        -- and therefore every edge -- is rebuilt from scratch on every
        analyze() call."""
        for u, v, d in self.graph.edges(data=True):
            tx_id = d.get("transaction_id")
            if not tx_id:
                continue
            row = db.query(models.GraphEdge).filter_by(transaction_id=tx_id).first()
            if row is None:
                row = models.GraphEdge(transaction_id=tx_id)
                db.add(row)
            row.source_entity = u
            row.target_entity = v
            row.relationship_type = d.get("relationship_type", "TRANSFERRED")
            row.rec_id = d.get("rec_id")
            row.quantity = d.get("quantity") or 0
            row.blockchain_status = d.get("blockchain_status")
            row.blockchain_tx_hash = d.get("blockchain_tx_hash")
            row.transaction_timestamp = d.get("timestamp")
            row.run_id = self.run_id

            reasons = []
            risk = 0.0
            if d.get("blockchain_status") == "FAILED":
                risk += config.GRAPH_RISK_WEIGHTS["rec_tampering"]
                reasons.append("Blockchain verification failed for this REC")
            row.risk_score = min(risk, 100)
            row.risk_level = _risk_level(row.risk_score)
            row.fraud_reason = json.dumps(reasons)
            row.status = "FLAGGED" if risk > 0 else "ACTIVE"
        db.commit()

    def _sync_clusters(self, db: Session, community_rows: list[dict], sccs: list[list[str]]) -> None:
        # A human can mark a cluster UNDER_INVESTIGATION/RESOLVED/
        # FALSE_POSITIVE (spec section 11's investigation-panel actions);
        # analyze() re-runs on every tick and must not silently clobber
        # that back to ACTIVE -- only ACTIVE/RESOLVED are auto-managed here.
        HUMAN_STATUSES = {"UNDER_INVESTIGATION", "FALSE_POSITIVE"}
        current_ids = set()
        for row in community_rows:
            current_ids.add(row["cluster_id"])
            rec = db.query(models.FraudCluster).filter_by(cluster_id=row["cluster_id"]).first()
            if rec is None:
                rec = models.FraudCluster(cluster_id=row["cluster_id"])
                db.add(rec)
            rec.cluster_type = "COMMUNITY"
            rec.entity_count = row["entity_count"]
            rec.rec_count = row["rec_count"]
            rec.transfer_count = row["transfer_count"]
            rec.graph_score = row["graph_score"]
            rec.risk_score = row["graph_score"]
            rec.risk_level = row["risk_level"]
            rec.fraud_pattern = json.dumps([row["detection_reason"]])
            rec.detection_reason = row["detection_reason"]
            if rec.status not in HUMAN_STATUSES:
                rec.status = "ACTIVE"
            rec.run_id = self.run_id

        for i, scc in enumerate(sccs):
            cluster_id = f"SCC-{i:03d}-{self.run_id[-6:]}"
            current_ids.add(cluster_id)
            sub = self.graph.subgraph(scc)
            rec_ids = {d["rec_id"] for _, _, d in sub.edges(data=True) if d.get("rec_id")}
            rec = db.query(models.FraudCluster).filter_by(cluster_id=cluster_id).first()
            if rec is None:
                rec = models.FraudCluster(cluster_id=cluster_id)
                db.add(rec)
            rec.cluster_type = "SCC"
            rec.entity_count = len(scc)
            rec.rec_count = len(rec_ids)
            rec.transfer_count = sub.number_of_edges()
            rec.graph_score = 90
            rec.risk_score = 90
            rec.risk_level = "CRITICAL"
            rec.fraud_pattern = json.dumps(["Circular trading (strongly connected component)"])
            rec.detection_reason = f"Strongly connected component of {len(scc)} entities -- RECs can flow in a closed loop"
            if rec.status not in HUMAN_STATUSES:
                rec.status = "ACTIVE"
            rec.run_id = self.run_id

        stale = db.query(models.FraudCluster).filter(
            models.FraudCluster.status == "ACTIVE", ~models.FraudCluster.cluster_id.in_(current_ids or [""])
        ).all()
        for row in stale:
            row.status = "RESOLVED"
        db.commit()

    def _sync_graph_alerts(self, db: Session, sccs: list[list[str]], community_rows: list[dict],
                            motif_results: dict[str, list[dict]], temporal: list[dict]) -> tuple[list[models.GraphAlert], list[str]]:
        """Idempotent: re-running analyze() on an unchanged graph creates no
        new alert rows (deduped by a deterministic `key` embedded in
        `evidence`), so alerts persist across ticks instead of being
        recreated (and re-broadcast) every few seconds while a pattern
        remains true. Returns (all currently-open alerts, keys created just
        now) so the caller can broadcast only genuinely new findings."""
        newly_created: list[str] = []

        def _ensure(alert_type, key, severity, reason, evidence, entity_id=None, cluster_id=None, transaction_id=None):
            # Exact match on dedup_key, regardless of status: once a human
            # has marked a finding UNDER_INVESTIGATION/RESOLVED/
            # FALSE_POSITIVE, the next analyze() call must not spawn a
            # duplicate OPEN alert for the exact same underlying pattern
            # just because the original row is no longer OPEN.
            existing = (
                db.query(models.GraphAlert)
                .filter_by(alert_type=alert_type, dedup_key=key)
                .first()
            )
            if existing:
                return
            row = models.GraphAlert(
                alert_id=f"GALERT-{uuid.uuid4().hex[:10]}", alert_type=alert_type, dedup_key=key,
                entity_id=entity_id, cluster_id=cluster_id, transaction_id=transaction_id,
                severity=severity, reason=reason, evidence=json.dumps(evidence, default=str),
                status="OPEN", run_id=self.run_id,
            )
            db.add(row)
            newly_created.append(key)

        for i, scc in enumerate(sccs):
            key = "scc:" + ",".join(scc)
            _ensure("CYCLE", key, "CRITICAL", f"Circular trading loop among {len(scc)} entities",
                    {"entities": scc}, cluster_id=f"SCC-{i:03d}-{self.run_id[-6:]}")

        for row in community_rows:
            if row["risk_level"] in ("SUSPICIOUS", "HIGH_RISK_FRAUD_CLUSTER"):
                key = "community:" + row["cluster_id"]
                _ensure("COMMUNITY", key, "HIGH" if row["risk_level"] == "SUSPICIOUS" else "CRITICAL",
                        row["detection_reason"], {"entities": row["entity_ids"]}, cluster_id=row["cluster_id"])

        for m in motif_results.get("suspicious_hub", []):
            key = "hub:" + m["entity_id"]
            _ensure("HUB", key, "MEDIUM", m["reason"], m, entity_id=m["entity_id"])

        for m in motif_results.get("duplicate_transfer", []):
            key = "dup:" + m["generation_id"]
            _ensure("MOTIF", key, "HIGH", m["reason"], m, entity_id=m.get("generator_id"))

        for m in motif_results.get("generator_claim_mismatch", []):
            key = "mismatch:" + m["rec_id"]
            _ensure("MOTIF", key, "HIGH", m["reason"], m, entity_id=m.get("generator_id"))

        for b in temporal:
            key = "temporal:" + json.dumps(sorted(b.get("transaction_ids") or [b.get("entity_id", b.get("rec_id", ""))]))
            _ensure("TEMPORAL", key, "MEDIUM",
                    f"Rapid transfer activity detected ({b.get('transfer_count')} transfers in {b.get('window_seconds', 0):.0f}s)",
                    b, entity_id=b.get("entity_id"))

        db.commit()
        open_alerts = db.query(models.GraphAlert).filter_by(status="OPEN").all()
        return open_alerts, newly_created

    # ---------------- orchestrator ----------------

    def analyze(self, db: Session) -> dict:
        """Full pipeline (spec section 2):
        Database Transactions -> Graph Construction -> Graph Analytics ->
        Fraud Detection -> Risk Scores and Alerts -> persisted for the
        frontend. Returns a summary dict used for /api/graph/overview and
        the `graph_update` WebSocket broadcast.

        The whole computation runs under `self._lock` (held for this
        method's entire duration, not just build()'s) so a concurrent
        build()/analyze() call from another thread can't swap `self.graph`
        out from under the SCC/community/motif/risk-scoring steps that all
        read it after the initial build()."""
        with self._lock:
            return self._analyze_locked(db)

    def _analyze_locked(self, db: Session) -> dict:
        self.build(db)

        metrics = self.entity_metrics()
        pagerank = self.pagerank()
        sccs = self.strongly_connected_components()
        comms = self.communities_louvain()
        community_rows = self.community_rows(db, comms, sccs)
        community_of: dict[str, str] = {}
        community_level: dict[str, str] = {row["cluster_id"]: row["risk_level"] for row in community_rows}
        for row in community_rows:
            for ent in row["entity_ids"]:
                community_of[ent] = row["cluster_id"]

        motif_results = self.motifs(db)
        temporal = self.temporal_bursts()
        hub_set = set(self.high_degree_hubs())
        alert_counts = self._alert_counts_by_entity(db)
        tampered_entities = self._tampered_entities(db)

        motif_entities: set[str] = set()
        duplicate_transfer_entities: set[str] = set()
        for m in motif_results["circular_trading"] + motif_results["rapid_relay"]:
            motif_entities.update(m["entities"])
        for m in motif_results["suspicious_hub"]:
            motif_entities.add(m["entity_id"])
        for m in motif_results["duplicate_transfer"] + motif_results["generator_claim_mismatch"]:
            if m.get("generator_id"):
                duplicate_transfer_entities.add(m["generator_id"])
        temporal_entities: set[str] = set()
        for b in temporal:
            temporal_entities.update(b.get("entities", []))
            if b.get("entity_id"):
                temporal_entities.add(b["entity_id"])

        gnn_scores = gnn_service.get_service().score_entities(self)  # entity_id -> fraud probability, {} if unavailable
        gnn_flagged = {e for e, p in gnn_scores.items() if p >= config.GNN_FRAUD_THRESHOLD}

        hits = {
            "motif_entities": motif_entities, "duplicate_transfer_entities": duplicate_transfer_entities,
            "temporal_entities": temporal_entities, "tampered_entities": tampered_entities,
            "gnn_flagged_entities": gnn_flagged, "gnn_scores": gnn_scores,
        }

        volume_by_entity: dict[str, float] = defaultdict(float)
        txcount_by_entity: dict[str, int] = defaultdict(int)
        for u, v, d in self.graph.edges(data=True):
            qty = d.get("quantity") or 0
            volume_by_entity[u] += qty
            volume_by_entity[v] += qty
            txcount_by_entity[u] += 1
            txcount_by_entity[v] += 1

        entity_rows = []
        for entity_id, m in metrics.items():
            score, reasons = self.combined_entity_risk(
                entity_id, sccs=sccs, community_label=community_level.get(community_of.get(entity_id), None),
                hub_set=hub_set, betweenness={k: v["betweenness_centrality"] for k, v in metrics.items()},
                motif_hits=hits, alert_count=alert_counts.get(entity_id, 0),
            )
            entity_rows.append({
                "entity_id": entity_id, "entity_type": _classify_entity(entity_id),
                "degree": m["degree"], "in_degree": m["in_degree"], "out_degree": m["out_degree"],
                "betweenness": m["betweenness_centrality"], "pagerank": round(pagerank.get(entity_id, 0), 5),
                "community_id": community_of.get(entity_id), "risk_score": round(score, 2),
                "risk_level": _risk_level(score), "alert_count": alert_counts.get(entity_id, 0),
                "total_rec_volume": round(volume_by_entity.get(entity_id, 0), 2),
                "transaction_count": txcount_by_entity.get(entity_id, 0),
                "gnn_score": gnn_scores.get(entity_id),
                "reasons": reasons,
            })

        self._sync_entities(db, entity_rows)
        self._sync_edges(db)
        self._sync_clusters(db, community_rows, sccs)
        graph_alerts, new_alert_keys = self._sync_graph_alerts(db, sccs, community_rows, motif_results, temporal)
        self._announced |= set(new_alert_keys)

        return {
            "run_id": self.run_id,
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "suspicious_nodes": sum(1 for r in entity_rows if r["risk_level"] in ("HIGH", "CRITICAL")),
            "suspicious_edges": sum(1 for _, _, d in self.graph.edges(data=True) if d.get("blockchain_status") == "FAILED"),
            "fraud_rings": len(sccs),
            "suspicious_communities": sum(1 for r in community_rows if r["risk_level"] in ("SUSPICIOUS", "HIGH_RISK_FRAUD_CLUSTER")),
            "high_risk_brokers": len(hub_set),
            "active_graph_alerts": len(graph_alerts),
            "new_alert_count": len(new_alert_keys),
            "new_alert_keys": new_alert_keys,
            "generated_at": time.time(),
            "entity_rows": entity_rows,
            "community_rows": community_rows,
            "sccs": sccs,
            "motifs": motif_results,
            "temporal": temporal,
        }

    def to_network_json(self, db: Session, *, node_type: str | None = None, risk_level: str | None = None,
                         status: str | None = None, since: float | None = None, until: float | None = None,
                         suspicious_only: bool = False, search: str | None = None) -> dict:
        """Frontend-friendly network payload (spec section 12) -- reads the
        persisted graph_entities/graph_edges rows (already scored by
        analyze()) rather than recomputing metrics on every request."""
        entity_rows = {r.entity_id: r for r in db.query(models.GraphEntity).all()}
        edge_rows = {r.transaction_id: r for r in db.query(models.GraphEdge).all() if r.transaction_id}
        hubs = set(self.high_degree_hubs())
        cycle_nodes = set(itertools.chain.from_iterable(self.circular_ownership()))

        nodes = []
        for n in self.graph.nodes():
            er = entity_rows.get(n)
            risk = er.risk_level if er else "LOW"
            etype = er.entity_type if er else _classify_entity(n)
            if node_type and etype != node_type:
                continue
            if risk_level and risk != risk_level:
                continue
            if suspicious_only and risk not in ("HIGH", "CRITICAL"):
                continue
            if search and search.lower() not in n.lower():
                continue
            nodes.append({
                "id": n, "label": n[:28], "type": etype,
                "risk_score": er.risk_score if er else 0, "risk_level": risk,
                "degree": er.degree if er else 0, "in_degree": er.in_degree if er else 0,
                "out_degree": er.out_degree if er else 0, "betweenness": er.betweenness if er else 0,
                "pagerank": er.pagerank_score if er else 0, "gnn_risk_score": er.gnn_risk_score if er else None,
                "community_id": er.community_id if er else None,
                "alert_count": er.alert_count if er else 0,
                "total_rec_volume": er.total_rec_volume if er else 0,
                "transaction_count": er.transaction_count if er else 0,
                "flagged": n in hubs or n in cycle_nodes or risk in ("HIGH", "CRITICAL"),
            })
        node_ids = {n["id"] for n in nodes}

        edges = []
        for u, v, d in self.graph.edges(data=True):
            if u not in node_ids or v not in node_ids:
                continue
            tx_id = d.get("transaction_id")
            er = edge_rows.get(tx_id)
            edge_status = er.status if er else "ACTIVE"
            if status and edge_status != status:
                continue
            ts = d.get("timestamp")
            if since is not None and ts is not None and ts < since:
                continue
            if until is not None and ts is not None and ts > until:
                continue
            edges.append({
                "id": tx_id or f"{u}->{v}:{ts}", "from": u, "to": v, "transaction_id": tx_id,
                "rec_id": d.get("rec_id"), "quantity": d.get("quantity"),
                "transaction_type": d.get("relationship_type"), "transaction_timestamp": ts,
                "status": edge_status,
                "blockchain_status": d.get("blockchain_status"), "blockchain_tx_hash": d.get("blockchain_tx_hash"),
                "risk_score": er.risk_score if er else 0, "risk_level": er.risk_level if er else "LOW",
                "fraud_reason": json.loads(er.fraud_reason) if er and er.fraud_reason else [],
            })

        return {"nodes": nodes, "edges": edges, "generated_at": time.time(), "run_id": self.run_id}

    def export_investigation(self, db: Session, entity_id: str | None = None, cluster_id: str | None = None) -> dict:
        """Everything the spec's export section (17) asks for, scoped to
        one entity or one cluster."""
        entities: set[str] = set()
        cluster_row = None
        if cluster_id:
            cluster_row = db.query(models.FraudCluster).filter_by(cluster_id=cluster_id).first()
            member_rows = db.query(models.GraphEntity).filter_by(community_id=cluster_id).all()
            entities = {r.entity_id for r in member_rows}
            if not entities:
                # SCC-type clusters aren't tagged with community_id -- fall
                # back to whatever edges/entities the live SCC/community
                # computation currently associates with this id.
                for scc in self.strongly_connected_components():
                    if f"SCC-" in cluster_id:
                        idx = cluster_id.split("-")[1]
                        try:
                            entities = set(self.strongly_connected_components()[int(idx)])
                        except (IndexError, ValueError):
                            pass
        elif entity_id:
            entities = {entity_id}

        edge_rows = db.query(models.GraphEdge).filter(
            (models.GraphEdge.source_entity.in_(entities)) | (models.GraphEdge.target_entity.in_(entities))
        ).all() if entities else []
        rec_ids = sorted({e.rec_id for e in edge_rows if e.rec_id})
        tx_ids = sorted({e.transaction_id for e in edge_rows if e.transaction_id})
        entity_rows = db.query(models.GraphEntity).filter(models.GraphEntity.entity_id.in_(entities)).all() if entities else []
        alerts = db.query(models.GraphAlert).filter(
            models.GraphAlert.entity_id.in_(entities) if entities else False
        ).all() if entities else []

        return {
            "exported_at": time.time(),
            "scope": {"entity_id": entity_id, "cluster_id": cluster_id},
            "cluster": {
                "cluster_id": cluster_row.cluster_id, "cluster_type": cluster_row.cluster_type,
                "risk_level": cluster_row.risk_level, "risk_score": cluster_row.risk_score,
                "detection_reason": cluster_row.detection_reason,
            } if cluster_row else None,
            "entities": [
                {
                    "entity_id": r.entity_id, "entity_type": r.entity_type, "risk_score": r.risk_score,
                    "risk_level": r.risk_level, "degree": r.degree, "betweenness": r.betweenness,
                    "pagerank": r.pagerank_score, "community_id": r.community_id, "alert_count": r.alert_count,
                }
                for r in entity_rows
            ],
            "edges": [
                {
                    "transaction_id": e.transaction_id, "rec_id": e.rec_id, "source_entity": e.source_entity,
                    "target_entity": e.target_entity, "quantity": e.quantity, "risk_score": e.risk_score,
                    "blockchain_status": e.blockchain_status, "blockchain_tx_hash": e.blockchain_tx_hash,
                    "transaction_timestamp": e.transaction_timestamp,
                }
                for e in edge_rows
            ],
            "rec_ids": rec_ids,
            "transaction_ids": tx_ids,
            "graph_alerts": [
                {"alert_id": a.alert_id, "alert_type": a.alert_type, "severity": a.severity, "reason": a.reason}
                for a in alerts
            ],
        }

    # ==================================================================== GNN feature extraction
    #
    # Shared by both ml_training/train_gnn.py (called on synthetic training
    # graphs) and backend/gnn_service.py (called on the live graph) so the
    # two can never silently drift apart -- the single most important
    # correctness property for any ML system that trains on one feature
    # definition and infers on another. Deliberately RAW/structural
    # features (not already-derived fraud signals like "is in an SCC" --
    # that would make the GNN just parrot the classical detector); the
    # point of message passing over edge_index is to let the model learn
    # structural patterns itself, on top of these.

    def gnn_node_features(self) -> tuple[list[str], dict[str, list[float]]]:
        if self.graph.number_of_nodes() == 0:
            return GNN_FEATURE_ORDER, {}
        undirected = nx.Graph(self.graph)
        clustering = nx.clustering(undirected)
        features: dict[str, list[float]] = {}
        for n in self.graph.nodes():
            in_d, out_d = self.graph.in_degree(n), self.graph.out_degree(n)
            incident = list(self.graph.in_edges(n, data=True)) + list(self.graph.out_edges(n, data=True))
            volume = sum(d.get("quantity") or 0 for _, _, d in incident)
            tx_count = len(incident)
            avg_qty = volume / tx_count if tx_count else 0.0
            timestamps = sorted(d["timestamp"] for _, _, d in incident if d.get("timestamp") is not None)
            if len(timestamps) >= 2:
                gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
                avg_gap = sum(gaps) / len(gaps)
            else:
                # No repeated activity to measure a gap from -- treat as
                # "slow" (a large gap), never 0, which would misleadingly
                # read as maximally rapid/suspicious.
                avg_gap = 1_000_000.0
            features[n] = [
                float(in_d + out_d), float(in_d), float(out_d), float(volume), float(tx_count),
                float(avg_qty), float(clustering.get(n, 0.0)), float(avg_gap),
            ]
        return GNN_FEATURE_ORDER, features

    def as_visjs(self) -> dict:
        hubs = set(self.high_degree_hubs())
        cycle_nodes = set(itertools.chain.from_iterable(self.circular_ownership()))
        nodes = [
            {"id": n, "label": n, "group": "flagged" if (n in hubs or n in cycle_nodes) else "normal"}
            for n in self.graph.nodes()
        ]
        edges = [
            {"from": u, "to": v, "arrows": "to", "title": d.get("rec_id", "")}
            for u, v, d in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}


_engine = GraphFraudEngine()


def get_engine() -> GraphFraudEngine:
    return _engine
