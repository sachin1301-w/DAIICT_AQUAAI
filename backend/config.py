"""
Central configuration, all overridable via environment variables (.env).
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------------- database ----------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'rec_fraud.db'}")

# ---------------- simulation ----------------
SIMULATION_INTERVAL_SECONDS = float(os.getenv("SIMULATION_INTERVAL_SECONDS", "5"))
FRAUD_PROBABILITY = float(os.getenv("FRAUD_PROBABILITY", "0.15"))
SIMULATION_AUTOSTART = os.getenv("SIMULATION_AUTOSTART", "true").lower() == "true"

# ---------------- ML ----------------
MODELS_DIR = BASE_DIR / "models"
ISOLATION_FOREST_PATH = MODELS_DIR / "tx_isolation_forest_model.pkl"
SCALER_PATH = MODELS_DIR / "tx_scaler.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "tx_feature_names.json"

# Legacy REC-issuance-time models supplied at project start (different,
# incompatible feature schema -- plant/vintage based, not per-transaction).
# Kept in service for the "Submit Generation Reading" REC-issuance flow.
LEGACY_ISOLATION_FOREST_PATH = MODELS_DIR / "isolation_forest_model.pkl"
LEGACY_XGBOOST_PATH = MODELS_DIR / "xgboost_risk_model.pkl"
LEGACY_FUEL_ENCODER_PATH = MODELS_DIR / "label_encoder_fuel.pkl"
LEGACY_STATE_ENCODER_PATH = MODELS_DIR / "label_encoder_state.pkl"
LEGACY_STATUS_ENCODER_PATH = MODELS_DIR / "label_encoder_status.pkl"

ML_SCORE_BANDS = {
    "NORMAL": (0, 30),
    "MILD_ANOMALY": (31, 60),
    "SUSPICIOUS": (61, 80),
    "HIGHLY_SUSPICIOUS": (81, 100),
}

# ---------------- fraud decision weights ----------------
RULE_SCORE_WEIGHT = float(os.getenv("RULE_SCORE_WEIGHT", "0.30"))
ML_SCORE_WEIGHT = float(os.getenv("ML_SCORE_WEIGHT", "0.30"))
GRAPH_SCORE_WEIGHT = float(os.getenv("GRAPH_SCORE_WEIGHT", "0.40"))

RISK_LEVEL_THRESHOLDS = {
    "LOW": (0, 40),
    "MEDIUM": (41, 60),
    "HIGH": (61, 80),
    "CRITICAL": (81, 100),
}

DECISION_RULE_SCORE_FRAUD_THRESHOLD = float(os.getenv("DECISION_RULE_SCORE_FRAUD_THRESHOLD", "80"))
DECISION_GRAPH_SCORE_FRAUD_THRESHOLD = float(os.getenv("DECISION_GRAPH_SCORE_FRAUD_THRESHOLD", "80"))
DECISION_ML_SCORE_SUSPICIOUS_THRESHOLD = float(os.getenv("DECISION_ML_SCORE_SUSPICIOUS_THRESHOLD", "80"))

# ---------------- Advanced Graph Fraud Detection ----------------
# Combined graph risk score weights (spec section 8) -- every factor below
# can independently push the score up; the total is capped at 100 in
# graph_service.GraphFraudEngine.combined_graph_risk_score().
GRAPH_RISK_WEIGHTS = {
    "circular_trading": 30,
    "duplicate_transfer": 35,
    "high_degree_centrality": 15,
    "high_betweenness_centrality": 10,
    "suspicious_community": 20,
    "rapid_transfer_velocity": 15,
    "motif_match": 20,
    "previous_fraud_alerts": 20,
    "rec_tampering": 40,
    "gnn_anomaly": 25,
}

# Temporal analysis thresholds (spec section 6) -- configurable rather than
# hardcoded at each call site.
TEMPORAL_RAPID_WINDOW_SECONDS = float(os.getenv("TEMPORAL_RAPID_WINDOW_SECONDS", str(60 * 10)))  # 10 min
TEMPORAL_MAX_TRANSFERS_IN_WINDOW = int(os.getenv("TEMPORAL_MAX_TRANSFERS_IN_WINDOW", "3"))
TEMPORAL_MIN_VELOCITY_MWH_PER_MIN = float(os.getenv("TEMPORAL_MIN_VELOCITY_MWH_PER_MIN", "5"))
TEMPORAL_MAX_INTERMEDIARIES = int(os.getenv("TEMPORAL_MAX_INTERMEDIARIES", "4"))
RAPID_TRANSFER_CHAIN_WINDOW_SECONDS = float(os.getenv("RAPID_TRANSFER_CHAIN_WINDOW_SECONDS", str(60 * 60 * 6)))

# Centrality / hub thresholds
GRAPH_HIGH_DEGREE_THRESHOLD = int(os.getenv("GRAPH_HIGH_DEGREE_THRESHOLD", "4"))
GRAPH_HIGH_BETWEENNESS_THRESHOLD = float(os.getenv("GRAPH_HIGH_BETWEENNESS_THRESHOLD", "0.25"))
GRAPH_HIGH_PAGERANK_MULTIPLE = float(os.getenv("GRAPH_HIGH_PAGERANK_MULTIPLE", "3.0"))  # x the mean

GRAPH_RISK_LEVEL_THRESHOLDS = {
    "LOW": (0, 30),
    "MEDIUM": (31, 60),
    "HIGH": (61, 80),
    "CRITICAL": (81, 100),
}

# ---------------- GNN-based entity risk scoring ----------------
GNN_MODEL_PATH = MODELS_DIR / "gnn_fraud_model.pt"
GNN_FEATURE_META_PATH = MODELS_DIR / "gnn_feature_meta.json"
# Probability (0-1) above which combined_entity_risk() counts an entity as
# "GNN-flagged" and applies GRAPH_RISK_WEIGHTS["gnn_anomaly"].
GNN_FRAUD_THRESHOLD = float(os.getenv("GNN_FRAUD_THRESHOLD", "0.6"))

# ---------------- blockchain ----------------
WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI", "http://127.0.0.1:8545")
CHAIN_DIR = BASE_DIR / "chain"
CONTRACT_ABI_PATH = CHAIN_DIR / "contract_abi.json"
CONTRACT_ADDRESS_PATH = CHAIN_DIR / "contract_address.json"
DEPLOYMENT_INFO_PATH = CHAIN_DIR / "deployment_info.json"

# Hardhat's deterministic default account #0 private key (well-known, local
# dev chain only -- never use this key anywhere but a local Hardhat node).
DEFAULT_ADMIN_PRIVATE_KEY = os.getenv(
    "ADMIN_PRIVATE_KEY",
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
)

# ---------------- CORS (for the separately-served Vite React frontend) ----------------
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
# Also allow any private-LAN IP on port 5173, so a phone on the same WiFi
# (opening the frontend via the dev machine's LAN IP to scan the REC
# verification QR code) isn't blocked by CORS.
CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^http://(localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}):5173$",
)

# ---------------- misc ----------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
