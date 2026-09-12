"""
GNN-based entity risk scoring: a real, trained Graph Neural Network
(2-layer GraphSAGE, PyTorch + PyTorch Geometric) that takes the live
transaction graph -- node features plus edge_index -- and predicts a
per-entity fraud probability via message passing across neighbors, rather
than a hand-coded rule.

This is one MORE independent signal folded into
`GraphFraudEngine.combined_entity_risk`, alongside (never replacing) the
classical SCC/Louvain/centrality/motif detectors already there and the
separate Isolation Forest in ml_service.py. See ml_training/train_gnn.py
for how the model is trained (on synthetic data -- honestly disclosed,
same as the project's other ML models) and README.md's "GNN-Based Entity
Risk Scoring" section for the full picture.

Design mirrors blockchain_service.py's fail-soft philosophy: torch/
torch_geometric are optional at import time, and a missing/unreadable
model file must never crash the app -- `available` just stays False and
`score_entities()` returns {} instead of raising.
"""
from __future__ import annotations

import json
import logging

import config

logger = logging.getLogger("gnn_service")

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    _TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover -- exercised only when torch/PyG aren't installed
    _TORCH_AVAILABLE = False
    _IMPORT_ERROR = e


if _TORCH_AVAILABLE:
    class FraudGNN(torch.nn.Module):
        """2-layer GraphSAGE node classifier: legitimate (0) vs.
        fraud-associated (1). SAGEConv is inductive by design -- it
        aggregates each node's live neighborhood at inference time rather
        than requiring the exact training graph, which is exactly what's
        needed here since the real transaction graph keeps changing."""

        def __init__(self, in_channels: int, hidden_channels: int = 32, out_channels: int = 2):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv2(x, edge_index).relu()
            return self.lin(x)


class GNNService:
    def __init__(self):
        self.available = False
        self.model = None
        self.feature_order: list[str] = []
        self.mean = None
        self.std = None
        self.metrics: dict = {}
        self.unavailable_reason: str | None = None
        self._load()

    def _load(self) -> None:
        if not _TORCH_AVAILABLE:
            self.unavailable_reason = f"torch/torch_geometric not installed ({_IMPORT_ERROR})"
            logger.warning("GNN scoring unavailable: %s", self.unavailable_reason)
            return
        try:
            with open(config.GNN_FEATURE_META_PATH) as f:
                meta = json.load(f)
            self.feature_order = meta["feature_order"]
            self.mean = torch.tensor(meta["mean"], dtype=torch.float)
            self.std = torch.tensor(meta["std"], dtype=torch.float)
            self.metrics = meta.get("metrics", {})
            model = FraudGNN(len(self.feature_order), meta.get("hidden_channels", 32))
            model.load_state_dict(torch.load(config.GNN_MODEL_PATH, map_location="cpu"))
            model.eval()
            self.model = model
            self.available = True
            logger.info(
                "GNN fraud model loaded (%d features, val_f1=%.3f, val_accuracy=%.3f)",
                len(self.feature_order), self.metrics.get("val_f1", 0.0), self.metrics.get("val_accuracy", 0.0),
            )
        except FileNotFoundError:
            self.unavailable_reason = "model not trained yet -- run ml_training/train_gnn.py and copy its output into backend/models/"
            logger.warning("GNN scoring unavailable: %s", self.unavailable_reason)
        except Exception:
            self.unavailable_reason = "failed to load GNN model (see logs)"
            logger.exception("GNN scoring unavailable: failed to load model")

    def score_entities(self, engine) -> dict[str, float]:
        """Returns entity_id -> fraud probability (0-1) for every node
        currently in `engine.graph`. Never raises -- an unloaded model, an
        empty graph, or an unexpected inference error all just produce {},
        so a GNN problem can never take down the rest of graph analysis."""
        if not self.available or engine.graph.number_of_nodes() == 0:
            return {}
        try:
            _order, features = engine.gnn_node_features()
            nodes = list(features.keys())
            if not nodes:
                return {}
            node_index = {n: i for i, n in enumerate(nodes)}
            x = torch.tensor([features[n] for n in nodes], dtype=torch.float)
            x = (x - self.mean) / self.std.clamp(min=1e-6)

            edges = [(node_index[u], node_index[v]) for u, v in engine.graph.edges() if u in node_index and v in node_index]
            edge_index = (
                torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
            )

            with torch.no_grad():
                logits = self.model(x, edge_index)
                probs = F.softmax(logits, dim=1)[:, 1]
            return {n: round(float(probs[i]), 4) for n, i in node_index.items()}
        except Exception:
            logger.exception("GNN inference failed -- returning no scores for this cycle")
            return {}

    def status(self) -> dict:
        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "feature_order": self.feature_order,
            "metrics": self.metrics,
        }


_service: GNNService | None = None


def get_service() -> GNNService:
    global _service
    if _service is None:
        _service = GNNService()
    return _service
