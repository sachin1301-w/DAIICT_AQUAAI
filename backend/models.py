from __future__ import annotations

import time
import uuid

from sqlalchemy import (
    Boolean, Column, Float, ForeignKey, Integer, String, Text,
)

from database import Base


def _uuid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class Generator(Base):
    __tablename__ = "generators"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generator_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("GEN"))
    name = Column(String(120), nullable=False)
    plant_type = Column(String(20), nullable=False)  # Solar | Wind | Hydro
    state = Column(String(60), nullable=False)
    capacity_mw = Column(Float, nullable=False)
    registration_status = Column(String(20), default="ACTIVE")
    wallet_address = Column(String(64), nullable=True)
    created_at = Column(Float, default=time.time)


class GenerationRecord(Base):
    __tablename__ = "generation_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("GENEV"))
    generator_id = Column(String(40), ForeignKey("generators.generator_id"), index=True)
    generation_timestamp = Column(Float, default=time.time)
    energy_generated_mwh = Column(Float, nullable=False)
    weather_factor = Column(Float, default=1.0)
    meter_data_hash = Column(String(66), nullable=True)
    is_synthetic = Column(Boolean, default=True)
    created_at = Column(Float, default=time.time)


class RECRecord(Base):
    __tablename__ = "rec_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rec_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("REC"))
    generation_id = Column(String(40), ForeignKey("generation_records.generation_id"), index=True)
    generator_id = Column(String(40), ForeignKey("generators.generator_id"), index=True)
    quantity = Column(Float, nullable=False)
    issue_timestamp = Column(Float, default=time.time)
    status = Column(String(20), default="ACTIVE")  # ACTIVE | RETIRED | REVOKED | HELD
    current_owner = Column(String(64), nullable=True)
    blockchain_tx_hash = Column(String(80), nullable=True)
    generation_hash = Column(String(66), nullable=True)
    # PASSED | FAILED | PENDING | NOT_CONNECTED -- see pipeline._blockchain_settlement.
    # Defaults to PENDING: true until some chain call actually classifies it,
    # never silently reads as PASSED just because nothing has run yet.
    blockchain_status = Column(String(20), default="PENDING")
    blockchain_record_hash = Column(String(66), nullable=True)
    blockchain_verification_message = Column(String(200), nullable=True)
    blockchain_verified_at = Column(Float, nullable=True)
    created_at = Column(Float, default=time.time)


class RECTransaction(Base):
    __tablename__ = "rec_transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("TXN"))
    rec_id = Column(String(40), ForeignKey("rec_records.rec_id"), index=True)
    sender = Column(String(64), nullable=True)
    receiver = Column(String(64), nullable=True)
    transaction_type = Column(String(20), nullable=False)  # ISSUE | TRANSFER | RETIRE | REVOKE
    quantity = Column(Float, nullable=False)
    transaction_timestamp = Column(Float, default=time.time)
    blockchain_tx_hash = Column(String(80), nullable=True)
    blockchain_status = Column(String(20), default="PENDING")
    blockchain_record_hash = Column(String(66), nullable=True)
    blockchain_verification_message = Column(String(200), nullable=True)
    blockchain_verified_at = Column(Float, nullable=True)
    created_at = Column(Float, default=time.time)


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("ALERT"))
    rec_id = Column(String(40), nullable=True, index=True)
    generation_id = Column(String(40), nullable=True, index=True)
    transaction_id = Column(String(40), nullable=True, index=True)
    rule_score = Column(Float, default=0)
    ml_score = Column(Float, default=0)
    graph_score = Column(Float, default=0)
    final_risk_score = Column(Float, default=0)
    risk_level = Column(String(20), default="LOW")  # LOW | MEDIUM | HIGH | CRITICAL
    fraud_type = Column(String(60), nullable=True)
    reasons = Column(Text, default="[]")  # JSON-encoded list[str]
    status = Column(String(20), default="OPEN")  # OPEN | REVIEWING | RESOLVED | DISMISSED
    created_at = Column(Float, default=time.time)


class GraphEntity(Base):
    __tablename__ = "graph_entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), unique=True, index=True)
    entity_type = Column(String(20), nullable=False)  # GENERATOR | ISSUER | BUYER | BROKER | TRADER | COMPANY | ACCOUNT
    entity_name = Column(String(120), nullable=True)
    risk_score = Column(Float, default=0)
    risk_level = Column(String(20), default="LOW")  # LOW | WATCHLIST | HIGH | CRITICAL
    degree = Column(Integer, default=0)
    in_degree = Column(Integer, default=0)
    out_degree = Column(Integer, default=0)
    betweenness = Column(Float, default=0)
    pagerank_score = Column(Float, default=0)
    # Fraud probability (0-1) from the trained GraphSAGE model, or None
    # when gnn_service.available is False (model not trained/loaded) --
    # distinct from `risk_score`, which already folds this in as one of
    # several weighted signals (see GraphFraudEngine.combined_entity_risk).
    gnn_risk_score = Column(Float, nullable=True)
    community_id = Column(String(40), nullable=True)
    total_rec_volume = Column(Float, default=0)
    transaction_count = Column(Integer, default=0)
    alert_count = Column(Integer, default=0)
    run_id = Column(String(40), nullable=True, index=True)
    created_at = Column(Float, default=time.time)
    updated_at = Column(Float, default=time.time)


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(40), nullable=True, index=True)
    source_entity = Column(String(64), index=True)
    target_entity = Column(String(64), index=True)
    relationship_type = Column(String(20), nullable=False)  # ISSUED | OWNED | TRANSFERRED | RETIRED
    rec_id = Column(String(40), nullable=True)
    quantity = Column(Float, default=0)
    risk_score = Column(Float, default=0)
    risk_level = Column(String(20), default="LOW")
    status = Column(String(20), default="ACTIVE")
    fraud_reason = Column(Text, default="[]")  # JSON-encoded list[str]
    blockchain_status = Column(String(20), nullable=True)
    blockchain_tx_hash = Column(String(80), nullable=True)
    transaction_timestamp = Column(Float, default=time.time)
    run_id = Column(String(40), nullable=True, index=True)


class FraudCluster(Base):
    __tablename__ = "fraud_clusters"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("FRAUD"))
    cluster_type = Column(String(30), default="COMMUNITY")  # COMMUNITY | SCC | MOTIF
    entity_count = Column(Integer, default=0)
    rec_count = Column(Integer, default=0)
    transfer_count = Column(Integer, default=0)
    graph_score = Column(Float, default=0)
    risk_score = Column(Float, default=0)
    risk_level = Column(String(20), default="LOW")
    fraud_pattern = Column(Text, default="[]")  # JSON-encoded list[str]
    detection_reason = Column(Text, nullable=True)
    status = Column(String(20), default="ACTIVE")
    run_id = Column(String(40), nullable=True, index=True)
    created_at = Column(Float, default=time.time)


class GraphAlert(Base):
    """Structure-level alerts (a cycle, a community, a motif, a hub) --
    distinct from FraudAlert, which is one transaction's rule/ML/graph
    decision. A single SCC or community can span many transactions, so it
    gets its own alert row rather than being force-fit onto one of them."""
    __tablename__ = "graph_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("GALERT"))
    alert_type = Column(String(30), nullable=False)  # CYCLE | COMMUNITY | HUB | MOTIF | TEMPORAL
    # Exact-match dedup key (see verification_service... no, graph_service._sync_graph_alerts):
    # a plain string, never JSON-nested inside itself, so an exact `==`
    # lookup is reliable -- a substring/LIKE match against the JSON
    # `evidence` blob broke whenever a key itself contained embedded quotes
    # (e.g. temporal alert keys), silently defeating dedup and spamming a
    # fresh alert every tick.
    dedup_key = Column(String(300), nullable=True, index=True)
    entity_id = Column(String(64), nullable=True, index=True)
    transaction_id = Column(String(40), nullable=True)
    cluster_id = Column(String(40), nullable=True, index=True)
    severity = Column(String(20), default="LOW")  # LOW | MEDIUM | HIGH | CRITICAL
    reason = Column(Text, nullable=True)
    evidence = Column(Text, default="{}")  # JSON blob: entities/recs/txs/timestamps involved
    status = Column(String(20), default="OPEN")  # OPEN | UNDER_INVESTIGATION | RESOLVED | FALSE_POSITIVE
    created_at = Column(Float, default=time.time)
    run_id = Column(String(40), nullable=True, index=True)


class RECVerificationRequest(Base):
    """One row per REC Verification Portal check -- every attempt, including
    failed/NOT_FOUND ones, per the append-only audit requirement. Deliberately
    NOT cleared by Reset Transactions (see simulator.py's _RESET_TABLES) --
    a verification/compliance trail should outlive a demo data reset."""
    __tablename__ = "rec_verification_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    verification_id = Column(String(40), unique=True, index=True, default=lambda: _uuid("VER"))
    rec_id = Column(String(40), index=True, nullable=False)
    verifier_company = Column(String(120), nullable=True)
    verifier_user = Column(String(120), nullable=True)
    requested_at = Column(Float, default=time.time)
    result = Column(String(20), nullable=False)  # VALID|SUSPICIOUS|INVALID|TAMPERED|NOT_FOUND|PENDING|UNCONFIRMED
    reason = Column(Text, default="[]")  # JSON-encoded list[str]
    blockchain_status = Column(String(20), nullable=True)
    hash_status = Column(String(20), nullable=True)
    lifecycle_status = Column(String(20), nullable=True)
    verified_blockchain = Column(Boolean, default=False)
    verified_hash = Column(Boolean, default=False)
    verified_lifecycle = Column(Boolean, default=False)
    risk_score = Column(Float, nullable=True)
    source_ip = Column(String(64), nullable=True)
    # Full computed response captured verbatim at verification time, so the
    # "Share Verification Report" feature always re-serves exactly what was
    # decided then -- never recomputed, never editable by the client.
    result_detail = Column(Text, nullable=True)


class RECAuditLog(Base):
    """Append-only, hash-chained history of everything that happens to a
    REC. Each row's record_hash covers its own content plus the previous
    row's hash (see verification_service.audit_log), so silently editing or
    deleting a row breaks the chain -- a cheap, real tamper-evidence property
    for what is otherwise a plain SQLite table."""
    __tablename__ = "rec_audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rec_id = Column(String(40), index=True, nullable=False)
    event_type = Column(String(30), nullable=False)
    old_value = Column(Text, nullable=True)  # JSON
    new_value = Column(Text, nullable=True)  # JSON
    changed_by = Column(String(80), default="system")
    changed_at = Column(Float, default=time.time)
    source = Column(String(40), nullable=True)
    record_hash = Column(String(66), nullable=True)
    previous_record_hash = Column(String(66), nullable=True)
