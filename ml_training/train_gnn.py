"""
Trains the GraphSAGE-based entity fraud classifier and saves the two files
backend/gnn_service.py looks for:

    gnn_fraud_model.pt, gnn_feature_meta.json

Run from ml_training/:
    python train_gnn.py

Then copy both output files into backend/models/ to activate GNN scoring
(without them, gnn_service.available stays False and the app keeps working
normally -- GNN scoring is one more optional signal, never a hard
dependency; see graph_service.combined_entity_risk).

Requires: torch, torch_geometric (see requirements.txt in this folder).

Architecture and node features are defined ONCE, in
backend/graph_service.py (GraphFraudEngine.gnn_node_features,
GNN_FEATURE_ORDER) and backend/gnn_service.py (FraudGNN) -- imported here
rather than redefined, so training and live inference can never compute
features or model shapes differently from each other.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Never touch the real dev database -- nothing here needs SQL at all, this
# is purely in-memory graph construction, but set this defensively before
# importing backend modules in case any import path ends up opening a
# session.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from generate_synthetic_graph_data import generate
from gnn_service import FraudGNN
from graph_service import GNN_FEATURE_ORDER, GraphFraudEngine

OUT_MODEL = "gnn_fraud_model.pt"
OUT_META = "gnn_feature_meta.json"


def _build_nx_graph(sample: dict) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for e in sample["edges"]:
        g.add_edge(
            e["source"], e["target"], rec_id=e["rec_id"], timestamp=e["timestamp"],
            transaction_id=e["transaction_id"], relationship_type="TRANSFERRED",
            quantity=e["quantity"], blockchain_status=None, blockchain_tx_hash=None,
        )
    for n in sample["labels"]:  # covers any isolated node not touched by an edge
        g.add_node(n)
    return g


def _sample_to_pyg(sample: dict) -> Data | None:
    """Builds a graph the exact same way GraphFraudEngine.build() would
    from real RECTransaction rows, then calls the SAME feature method
    gnn_service.py uses at inference time (see module docstring)."""
    g = _build_nx_graph(sample)
    engine = GraphFraudEngine()
    engine.graph = g
    _order, features = engine.gnn_node_features()
    nodes = list(features.keys())
    if not nodes:
        return None
    node_index = {n: i for i, n in enumerate(nodes)}
    x = torch.tensor([features[n] for n in nodes], dtype=torch.float)
    y = torch.tensor([sample["labels"].get(n, 0) for n in nodes], dtype=torch.long)
    edges = [(node_index[u], node_index[v]) for u, v, _ in g.edges(data=True) if u in node_index and v in node_index]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def _evaluate(model: FraudGNN, loader: DataLoader) -> dict:
    model.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.x, batch.edge_index)
            pred = logits.argmax(dim=1)
            tp += int(((pred == 1) & (batch.y == 1)).sum())
            fp += int(((pred == 1) & (batch.y == 0)).sum())
            fn += int(((pred == 0) & (batch.y == 1)).sum())
            tn += int(((pred == 0) & (batch.y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def main():
    samples = generate(n_samples=220, fraud_ratio=0.45, seed=42)
    data_list = [d for d in (_sample_to_pyg(s) for s in samples) if d is not None]
    random.Random(7).shuffle(data_list)

    n = len(data_list)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    train_data = data_list[:n_train]
    val_data = data_list[n_train:n_train + n_val]
    test_data = data_list[n_train + n_val:]
    print(f"{n} graphs -> train {len(train_data)} / val {len(val_data)} / test {len(test_data)}")

    # Standardization stats from the TRAIN split only -- applied identically
    # here and in gnn_service.py (saved to gnn_feature_meta.json).
    all_train_x = torch.cat([d.x for d in train_data], dim=0)
    mean = all_train_x.mean(dim=0)
    std = all_train_x.std(dim=0)
    std[std < 1e-6] = 1.0
    for split in (train_data, val_data, test_data):
        for d in split:
            d.x = (d.x - mean) / std

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)

    all_train_y = torch.cat([d.y for d in train_data])
    n_pos = int((all_train_y == 1).sum())
    n_neg = int((all_train_y == 0).sum())
    class_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float)
    print(f"train label balance: {n_neg} normal / {n_pos} fraud-associated -- class weight {class_weight.tolist()}")

    model = FraudGNN(in_channels=len(GNN_FEATURE_ORDER), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_f1 = -1.0
    best_state = None
    for epoch in range(1, 121):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(logits, batch.y, weight=class_weight)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch == 1 or epoch % 10 == 0:
            val_metrics = _evaluate(model, val_loader)
            print(
                f"epoch {epoch:3d}  loss={total_loss / len(train_data):.4f}  "
                f"val_f1={val_metrics['f1']:.3f}  val_acc={val_metrics['accuracy']:.3f}"
            )
            if val_metrics["f1"] >= best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_metrics = _evaluate(model, val_loader)
    test_metrics = _evaluate(model, test_loader)
    print(f"FINAL  val: {val_metrics}")
    print(f"FINAL  test: {test_metrics}")

    torch.save(model.state_dict(), OUT_MODEL)
    meta = {
        "feature_order": GNN_FEATURE_ORDER,
        "mean": mean.tolist(), "std": std.tolist(),
        "hidden_channels": 32,
        "metrics": {
            "val_accuracy": val_metrics["accuracy"], "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
            "test_accuracy": test_metrics["accuracy"], "test_f1": test_metrics["f1"],
            "trained_on": "synthetic data (ml_training/generate_synthetic_graph_data.py) -- see README",
        },
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {OUT_MODEL}, {OUT_META}")
    print("Copy both into backend/models/ to activate GNN scoring.")


if __name__ == "__main__":
    main()
