"""
Generates synthetic transaction graphs for training the GNN fraud
classifier (train_gnn.py). Mirrors the structural patterns
backend/simulator.py actually produces in the live app -- normal ownership
chains, plus occasional fraud rings with circular trading, dense
interconnection, and rapid transfers -- at larger scale and topology
variety than any single live demo run would produce on its own.

Each sample is one independent small graph:
    {"edges": [{"source", "target", "quantity", "timestamp", "rec_id", "transaction_id"}, ...],
     "labels": {entity_id: 0 or 1}}   -- 1 = part of an injected fraud ring

SYNTHETIC DATA DISCLOSURE: there is no public REC fraud dataset (see this
project's README and generate_synthetic_fraud.py for the same disclosure
about the tabular ML models) -- this is a structural simulation designed to
teach the GNN what circular/dense/rapid trading *shapes* look like, not a
claim of real transaction data.
"""
from __future__ import annotations

import random
import time
import uuid

NORMAL_TRADER_POOL = [
    "GreenBuy Industries", "EcoRetail Ltd", "Fairtrade Broker Co", "SunTrust Energy Buyers",
    "Clean Power Partners", "Solaris Holdings", "Verde Capital", "BrightGrid Traders",
]


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _make_normal_chain(rng: random.Random, base_ts: float) -> tuple[list[str], list[dict]]:
    """A sparse tree: generator -> a few hops -> done. No cycles, low
    density, hours apart -- a legitimate REC's ordinary lifecycle."""
    n_hops = rng.randint(1, 4)
    entities = [f"GEN-{uuid.uuid4().hex[:6]}"] + rng.sample(NORMAL_TRADER_POOL, k=min(n_hops, len(NORMAL_TRADER_POOL)))
    edges = []
    ts = base_ts
    for i in range(len(entities) - 1):
        ts += rng.uniform(3600, 6 * 3600)  # 1-6 hours apart
        edges.append({
            "source": entities[i], "target": entities[i + 1],
            "quantity": round(rng.uniform(5, 80), 2), "timestamp": ts,
            "rec_id": _uid("REC"), "transaction_id": _uid("TXN"),
        })
    return entities, edges


def _make_fraud_ring(rng: random.Random, base_ts: float) -> tuple[list[str], list[dict]]:
    """3-6 entities: a closed transfer loop plus extra dense cross-links,
    seconds-to-minutes apart -- the same structural shape
    backend/simulator.py's FRAUD_RING scenario produces, generalized with
    randomized size/membership for training diversity."""
    size = rng.randint(3, 6)
    ring = [f"Ring-{uuid.uuid4().hex[:6]}" for _ in range(size)]
    edges = []
    ts = base_ts
    for i in range(size):
        ts += rng.uniform(5, 300)  # rapid: seconds to a few minutes
        edges.append({
            "source": ring[i], "target": ring[(i + 1) % size],
            "quantity": round(rng.uniform(1, 40), 2), "timestamp": ts,
            "rec_id": _uid("REC"), "transaction_id": _uid("TXN"),
        })
    extra = rng.randint(2, size * 2)
    for _ in range(extra):
        a, b = rng.sample(ring, 2)
        ts += rng.uniform(5, 300)
        edges.append({
            "source": a, "target": b,
            "quantity": round(rng.uniform(1, 40), 2), "timestamp": ts,
            "rec_id": _uid("REC"), "transaction_id": _uid("TXN"),
        })
    return ring, edges


def generate(n_samples: int = 220, fraud_ratio: float = 0.45, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        base_ts = time.time() - rng.uniform(0, 30 * 86400)
        all_edges: list[dict] = []
        labels: dict[str, int] = {}

        for _ in range(rng.randint(1, 3)):  # 1-3 normal chains: background legitimate activity
            entities, edges = _make_normal_chain(rng, base_ts)
            all_edges.extend(edges)
            for e in entities:
                labels.setdefault(e, 0)

        if rng.random() < fraud_ratio:
            ring, edges = _make_fraud_ring(rng, base_ts)
            all_edges.extend(edges)
            for e in ring:
                labels[e] = 1  # ring membership always wins over any incidental normal-chain overlap

        samples.append({"edges": all_edges, "labels": labels})
    return samples


if __name__ == "__main__":
    import json

    data = generate()
    n_nodes = sum(len(s["labels"]) for s in data)
    n_fraud = sum(sum(s["labels"].values()) for s in data)
    print(f"{len(data)} graphs, {n_nodes} total node-labels, {n_fraud} fraud-labeled ({n_fraud / n_nodes:.1%})")
    with open("gnn_graphs.json", "w") as f:
        json.dump(data, f)
    print("wrote gnn_graphs.json")
