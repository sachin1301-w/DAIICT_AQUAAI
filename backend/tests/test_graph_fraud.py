"""
Backend test suite for the Advanced Graph Fraud Detection module (spec
section 18). Each test seeds exactly the SQLite rows it needs directly via
the SQLAlchemy models (fast, deterministic -- bypasses pipeline.py's
ML/blockchain calls, which is the right scope for testing graph_service.py's
own logic) and drives a fresh `GraphFraudEngine()` instance directly, so
tests never share state with the app's module-level singleton or with each
other.

Run with:  cd backend && pytest tests/ -v
"""
from __future__ import annotations

import time

import pytest

import gnn_service
import graph_service
import models
import pipeline
import verification_service

# ==================================================================== seed helpers


def make_generator(db, name="Gen", plant_type="Solar", state="Gujarat", capacity_mw=50.0):
    g = models.Generator(name=name, plant_type=plant_type, state=state, capacity_mw=capacity_mw)
    db.add(g)
    db.commit()
    return g


def make_generation(db, generator, energy_mwh=40.0, ts=None):
    gr = models.GenerationRecord(
        generator_id=generator.generator_id, generation_timestamp=ts or time.time(), energy_generated_mwh=energy_mwh,
    )
    db.add(gr)
    db.commit()
    return gr


def make_rec(db, generation, generator, quantity=None, owner=None, ts=None):
    quantity = generation.energy_generated_mwh if quantity is None else quantity
    now = ts if ts is not None else time.time()
    owner = owner or generator.generator_id
    rec = models.RECRecord(
        generation_id=generation.generation_id, generator_id=generator.generator_id, quantity=quantity,
        issue_timestamp=now, current_owner=owner, status="ACTIVE",
    )
    db.add(rec)
    db.flush()  # need the real rec_id (assigned on flush) before hashing
    rec.generation_hash = verification_service.canonical_rec_hash(rec, generation, generator)
    db.add(models.RECTransaction(
        rec_id=rec.rec_id, sender=None, receiver=owner, transaction_type="ISSUE",
        quantity=quantity, transaction_timestamp=now,
    ))
    db.commit()
    return rec


def make_transfer(db, rec, sender, receiver, quantity=None, ts=None):
    quantity = rec.quantity if quantity is None else quantity
    now = ts if ts is not None else time.time()
    rec.current_owner = receiver
    db.add(models.RECTransaction(
        rec_id=rec.rec_id, sender=sender, receiver=receiver, transaction_type="TRANSFER",
        quantity=quantity, transaction_timestamp=now,
    ))
    db.commit()


def fresh_engine() -> graph_service.GraphFraudEngine:
    return graph_service.GraphFraudEngine()


# ==================================================================== 1-4: basic graph tests


def test_empty_graph(db):
    engine = fresh_engine()
    summary = engine.analyze(db)
    assert summary["node_count"] == 0
    assert summary["edge_count"] == 0
    assert summary["fraud_rings"] == 0
    assert engine.strongly_connected_components() == []
    assert engine.communities_louvain() == []


def test_one_generator_one_buyer(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen)
    make_transfer(db, rec, gen.generator_id, "Buyer Co")

    engine = fresh_engine()
    engine.build(db)
    assert gen.generator_id in engine.graph.nodes
    assert "Buyer Co" in engine.graph.nodes
    assert engine.graph.number_of_edges() == 1


def test_normal_transfer_shape(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen)
    make_transfer(db, rec, gen.generator_id, "Buyer Co")

    engine = fresh_engine()
    summary = engine.analyze(db)
    assert summary["node_count"] == 2
    assert summary["edge_count"] == 1
    assert summary["suspicious_nodes"] == 0


def test_multiple_transfers_between_two_entities(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen)
    now = time.time()
    make_transfer(db, rec, gen.generator_id, "Buyer Co", ts=now)
    make_transfer(db, rec, "Buyer Co", gen.generator_id, ts=now + 5)
    make_transfer(db, rec, gen.generator_id, "Buyer Co", ts=now + 10)

    engine = fresh_engine()
    engine.build(db)
    repeated = engine.repeated_transfer_pairs()
    assert repeated.get((gen.generator_id, "Buyer Co")) == 2


# ==================================================================== 5-11: fraud detection tests


def test_duplicate_rec_transfer_motif(db):
    """Same generation record reused to issue two RECs (spec Rule/Motif B)."""
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec1 = make_rec(db, generation, gen)
    # Second REC issued against the SAME generation_id -- simulate duplicate issuance directly.
    now = time.time()
    rec2 = models.RECRecord(
        generation_id=generation.generation_id, generator_id=gen.generator_id, quantity=generation.energy_generated_mwh,
        issue_timestamp=now, current_owner=gen.generator_id, status="ACTIVE",
    )
    db.add(rec2)
    db.commit()

    engine = fresh_engine()
    motifs = engine.motifs(db)
    assert len(motifs["duplicate_transfer"]) == 1
    assert generation.generation_id == motifs["duplicate_transfer"][0]["generation_id"]
    assert {rec1.rec_id, rec2.rec_id} == set(motifs["duplicate_transfer"][0]["rec_ids"])


def test_circular_trading_a_b_c_a(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    now = time.time()
    make_transfer(db, rec, "A", "B", ts=now)
    make_transfer(db, rec, "B", "C", ts=now + 60)
    make_transfer(db, rec, "C", "A", ts=now + 120)

    engine = fresh_engine()
    engine.build(db)
    sccs = engine.strongly_connected_components()
    assert any(set(c) == {"A", "B", "C"} for c in sccs)

    motifs = engine.motifs(db)
    assert len(motifs["circular_trading"]) >= 1


def test_suspicious_hub(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    hub = "Hub Broker"
    now = time.time()
    for i, other in enumerate(["P1", "P2", "P3", "P4"]):
        rec = make_rec(db, generation, gen, owner=other, ts=now + i)
        make_transfer(db, rec, other, hub, ts=now + i + 1)
    for i, other in enumerate(["Q1", "Q2"]):
        rec2 = make_rec(db, generation, gen, owner=hub, ts=now + 10 + i)
        make_transfer(db, rec2, hub, other, ts=now + 10 + i + 1)

    engine = fresh_engine()
    engine.build(db)
    assert hub in engine.high_degree_hubs()
    motifs = engine.motifs(db)
    assert any(m["entity_id"] == hub for m in motifs["suspicious_hub"])


def test_dense_suspicious_community(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    members = ["M1", "M2", "M3", "M4"]
    now = time.time()
    rec = make_rec(db, generation, gen, owner=members[0], ts=now)
    # densely interconnect the group -- every member trades with every other
    hop = 0
    for a in members:
        for b in members:
            if a == b:
                continue
            hop += 1
            make_transfer(db, rec, a, b, ts=now + hop)

    engine = fresh_engine()
    engine.build(db)
    rows = engine.community_rows(db)
    assert any(set(r["entity_ids"]) >= set(members) and r["risk_level"] in ("SUSPICIOUS", "HIGH_RISK_FRAUD_CLUSTER") for r in rows)


def test_rapid_transfers_within_window(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="Fast Entity")
    now = time.time()
    # 4 transfers of the SAME rec within a few seconds -- well inside the
    # default 10-minute window and above the 3-transfer threshold.
    make_transfer(db, rec, "Fast Entity", "X1", ts=now)
    make_transfer(db, rec, "X1", "X2", ts=now + 2)
    make_transfer(db, rec, "X2", "X3", ts=now + 4)
    make_transfer(db, rec, "X3", "X4", ts=now + 6)

    engine = fresh_engine()
    engine.build(db)
    bursts = engine.temporal_bursts()
    assert any(b["type"] == "rec_velocity" and b["rec_id"] == rec.rec_id for b in bursts)


def test_known_motif_combination(db):
    """One scenario that should trip multiple motif types at once, exactly
    like the seeded FRAUD_RING in simulator.py does in the live app."""
    gen = make_generator(db)
    generation = make_generation(db, gen)
    now = time.time()
    rec = make_rec(db, generation, gen, owner="A", ts=now)
    make_transfer(db, rec, "A", "B", ts=now + 1)
    make_transfer(db, rec, "B", "C", ts=now + 2)
    make_transfer(db, rec, "C", "A", ts=now + 3)

    engine = fresh_engine()
    engine.build(db)
    motifs = engine.motifs(db)
    assert len(motifs["circular_trading"]) >= 1
    assert len(motifs["rapid_relay"]) >= 1


def test_generator_claim_mismatch(db):
    gen = make_generator(db, capacity_mw=50.0)
    generation = make_generation(db, gen, energy_mwh=100.0)
    # REC claims 150 while only 100 MWh was generated -- classic 100->150 tampering shape.
    rec = models.RECRecord(
        generation_id=generation.generation_id, generator_id=gen.generator_id, quantity=150.0,
        issue_timestamp=time.time(), current_owner=gen.generator_id, status="ACTIVE",
    )
    db.add(rec)
    db.commit()

    engine = fresh_engine()
    motifs = engine.motifs(db)
    assert len(motifs["generator_claim_mismatch"]) == 1
    assert motifs["generator_claim_mismatch"][0]["rec_id"] == rec.rec_id
    assert motifs["generator_claim_mismatch"][0]["claimed_quantity"] == 150.0


# ==================================================================== 12-20: integration tests


def test_ml_and_graph_score_combined(db):
    """Runs the REAL pipeline (rule + ML + graph), not a direct graph_service
    call -- confirms graph_service's output actually reaches the fraud
    decision engine end to end."""
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    result = pipeline.process_transfer(db, rec.rec_id, "A", "B", rec.quantity)
    assert "error" not in result
    assert "ml_score" in result["ml"]
    assert "graph_score" in result["graph"]
    assert "final_risk_score" in result["decision"]


def test_blockchain_status_reflected_in_graph(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    now = time.time()
    tx = models.RECTransaction(
        rec_id=rec.rec_id, sender="A", receiver="B", transaction_type="TRANSFER",
        quantity=rec.quantity, transaction_timestamp=now, blockchain_status="FAILED",
    )
    db.add(tx)
    rec.current_owner = "B"
    db.commit()

    engine = fresh_engine()
    engine.analyze(db)
    edge_row = db.query(models.GraphEdge).filter_by(transaction_id=tx.transaction_id).first()
    assert edge_row is not None
    assert edge_row.blockchain_status == "FAILED"
    assert edge_row.risk_score > 0
    assert edge_row.status == "FLAGGED"


def test_tampering_100_to_150_detected(db):
    """The exact demo scenario: energy_generated_mwh edited from 100 to 150
    directly in SQLite after issuance, without touching the anchored hash."""
    gen = make_generator(db, capacity_mw=50.0)
    generation = make_generation(db, gen, energy_mwh=100.0)
    rec = make_rec(db, generation, gen, quantity=100.0)
    original_hash = rec.generation_hash

    generation.energy_generated_mwh = 150.0  # tamper, bypassing the app entirely
    rec.quantity = 150.0
    db.commit()

    result = verification_service.verify_rec(db, rec.rec_id)
    assert result["result"] == "TAMPERED"
    assert result["hash_status"] == "MISMATCHED"
    assert result["original_hash"] == original_hash
    assert result["recomputed_hash"] != original_hash

    engine = fresh_engine()
    engine.build(db)
    tampered = engine._tampered_entities(db)
    assert gen.generator_id in tampered

    alert = db.query(models.FraudAlert).filter_by(rec_id=rec.rec_id).first()
    assert alert is not None
    assert alert.risk_level == "CRITICAL"


def test_reset_graph_clears_derived_tables(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    make_transfer(db, rec, "A", "B")

    engine = fresh_engine()
    engine.analyze(db)
    assert db.query(models.GraphEntity).count() > 0
    assert db.query(models.GraphEdge).count() > 0

    old_run_id = engine.run_id
    for model in (models.GraphAlert, models.GraphEdge, models.FraudCluster, models.GraphEntity):
        db.query(model).delete(synchronize_session=False)
    db.commit()
    engine.reset(new_run_id=True)

    assert db.query(models.GraphEntity).count() == 0
    assert db.query(models.GraphEdge).count() == 0
    assert engine.run_id != old_run_id
    assert engine.graph.number_of_nodes() == 0


def test_reset_transactions_clears_graph_data():
    """simulator._RESET_TABLES must include every graph-derived table --
    this is a static assertion that the list wasn't left out of sync with
    the models added for Advanced Graph Fraud Detection."""
    import simulator

    reset_table_names = {m.__tablename__ for m in simulator._RESET_TABLES}
    assert {"graph_entities", "graph_edges", "fraud_clusters", "graph_alerts"} <= reset_table_names
    # Verification/audit tables must NOT be wiped by a demo data reset.
    assert "rec_verification_requests" not in reset_table_names
    assert "rec_audit_log" not in reset_table_names


def test_new_run_has_no_old_graph_data(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    make_transfer(db, rec, "A", "B")

    engine = fresh_engine()
    engine.analyze(db)
    first_run_id = engine.run_id

    for model in (models.GraphAlert, models.GraphEdge, models.FraudCluster, models.GraphEntity,
                  models.RECTransaction, models.RECRecord, models.GenerationRecord):
        db.query(model).delete(synchronize_session=False)
    db.commit()
    engine.reset(new_run_id=True)

    summary = engine.analyze(db)
    assert summary["run_id"] != first_run_id
    assert summary["node_count"] == 0
    assert db.query(models.GraphEntity).filter_by(run_id=first_run_id).count() == 0


def test_rec_id_links_to_verification_portal(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    make_transfer(db, rec, "A", "B")

    engine = fresh_engine()
    engine.analyze(db)
    edge_row = db.query(models.GraphEdge).filter_by(rec_id=rec.rec_id).first()
    assert edge_row is not None

    detail = verification_service.get_rec_detail(db, edge_row.rec_id)
    assert detail is not None
    assert detail["rec_id"] == rec.rec_id
    assert detail["generator_id"] == gen.generator_id


def test_export_graph_investigation(db):
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    now = time.time()
    make_transfer(db, rec, "A", "B", ts=now)
    make_transfer(db, rec, "B", "C", ts=now + 1)
    make_transfer(db, rec, "C", "A", ts=now + 2)

    engine = fresh_engine()
    engine.analyze(db)
    export = engine.export_investigation(db, entity_id="A")
    assert export["scope"]["entity_id"] == "A"
    assert len(export["entities"]) >= 1
    assert rec.rec_id in export["rec_ids"]
    assert len(export["edges"]) >= 1


# ==================================================================== GNN-based entity risk scoring


def test_gnn_node_features_shape(db):
    """gnn_node_features() must return the exact feature vector length
    ml_training/train_gnn.py's saved model was trained on, regardless of
    whether a trained model is actually loaded right now."""
    gen = make_generator(db)
    generation = make_generation(db, gen)
    rec = make_rec(db, generation, gen, owner="A")
    make_transfer(db, rec, "A", "B")

    engine = fresh_engine()
    engine.build(db)
    order, features = engine.gnn_node_features()
    assert order == graph_service.GNN_FEATURE_ORDER
    assert set(features.keys()) == {"A", "B"}
    for vec in features.values():
        assert len(vec) == len(order)


def test_gnn_scoring_never_raises_on_empty_graph():
    """score_entities() must degrade gracefully (return {}), never throw,
    whether or not a trained model happens to be loaded in this environment."""
    engine = fresh_engine()  # empty graph, build() never called
    scores = gnn_service.get_service().score_entities(engine)
    assert scores == {}


def test_gnn_scores_fraud_ring_higher_than_normal_chain(db):
    """Skipped gracefully if no trained model is present (e.g. a fresh
    checkout before running ml_training/train_gnn.py) -- GNN scoring is an
    optional signal, so its own test must not fail the suite when the
    optional model simply isn't there."""
    svc = gnn_service.get_service()
    if not svc.available:
        pytest.skip("GNN model not trained/loaded -- run ml_training/train_gnn.py")

    gen = make_generator(db)
    generation = make_generation(db, gen)

    # A circular, rapid fraud ring...
    ring_rec = make_rec(db, generation, gen, owner="R1")
    now = time.time()
    make_transfer(db, ring_rec, "R1", "R2", ts=now)
    make_transfer(db, ring_rec, "R2", "R3", ts=now + 5)
    make_transfer(db, ring_rec, "R3", "R1", ts=now + 10)

    # ...alongside an unrelated, slow, non-circular chain.
    chain_rec = make_rec(db, generation, gen, owner="Gen2")
    make_transfer(db, chain_rec, "Gen2", "NormalBuyer", ts=now)

    engine = fresh_engine()
    engine.build(db)
    scores = svc.score_entities(engine)
    assert scores, "expected non-empty scores from a loaded model"

    ring_avg = sum(scores[e] for e in ("R1", "R2", "R3")) / 3
    normal_score = scores.get("NormalBuyer", 0.0)
    assert ring_avg > normal_score
