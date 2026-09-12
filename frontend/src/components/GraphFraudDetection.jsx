import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import RiskCard from "./RiskCard.jsx";
import BlockchainStatusBadge from "./BlockchainStatusBadge.jsx";

const NODE_TYPES = ["", "GENERATOR", "ISSUER", "BUYER", "BROKER", "TRADER", "COMPANY", "WALLET", "ACCOUNT"];
const RISK_LEVELS = ["", "LOW", "MEDIUM", "WATCHLIST", "HIGH", "SUSPICIOUS", "CRITICAL", "HIGH_RISK_FRAUD_CLUSTER"];
const EDGE_STATUSES = ["", "ACTIVE", "FLAGGED"];

const RISK_COLOR = {
  LOW: "#34d399", NORMAL: "#34d399",
  MEDIUM: "#fbbf24", WATCHLIST: "#fbbf24",
  HIGH: "#fb923c", SUSPICIOUS: "#fb923c",
  CRITICAL: "#f87171", HIGH_RISK_FRAUD_CLUSTER: "#f87171",
};

function riskColor(level) {
  return RISK_COLOR[level] || "#64748b";
}

function fmtTime(ts) {
  return ts ? new Date(ts * 1000).toLocaleString() : "—";
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function toCsv(rows) {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return [headers.join(","), ...rows.map((r) => headers.map((h) => esc(r[h])).join(","))].join("\n");
}

function downloadCsv(filename, rows) {
  const blob = new Blob([toCsv(rows)], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function GraphFraudDetection({ resetEpoch = 0, onOpenRecVerify }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  const [overview, setOverview] = useState(null);
  const [filters, setFilters] = useState({
    search: "", node_type: "", risk_level: "", status: "", suspicious_only: false, since: "", until: "",
  });
  const [stats, setStats] = useState({ nodes: 0, edges: 0 });
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);

  const [selected, setSelected] = useState(null); // { kind: "node"|"edge"|"cluster", data }
  const [tab, setTab] = useState("rings"); // rings | communities | motifs | temporal
  const [fraudRings, setFraudRings] = useState([]);
  const [communities, setCommunities] = useState([]);
  const [motifs, setMotifs] = useState(null);
  const [temporal, setTemporal] = useState(null);
  const [gnnStatus, setGnnStatus] = useState(null);

  function loadOverview() {
    api.graphOverview().then(setOverview).catch(() => {});
  }
  function loadSidePanels() {
    api.graphClusters({ cluster_type: "SCC" }).then(setFraudRings).catch(() => {});
    api.graphClusters({ cluster_type: "COMMUNITY" }).then(setCommunities).catch(() => {});
    api.graphMotifs().then(setMotifs).catch(() => {});
    api.graphTemporalAnalysis().then(setTemporal).catch(() => {});
  }

  useEffect(() => {
    loadOverview();
    loadSidePanels();
    const id = setInterval(() => {
      loadOverview();
      loadSidePanels();
    }, 8000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resetEpoch]);

  // GNN model status only needs checking once -- it doesn't change on its
  // own while the app is running (only a backend restart after re-running
  // ml_training/train_gnn.py would change it).
  useEffect(() => {
    api.graphGnnStatus().then(setGnnStatus).catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const params = {
        node_type: filters.node_type || undefined,
        risk_level: filters.risk_level || undefined,
        status: filters.status || undefined,
        suspicious_only: filters.suspicious_only || undefined,
        search: filters.search || undefined,
        since: filters.since ? new Date(filters.since).getTime() / 1000 : undefined,
        until: filters.until ? new Date(filters.until).getTime() / 1000 : undefined,
      };
      const data = await api.graphNetwork(params);
      if (cancelled) return;
      setStats({ nodes: data.nodes.length, edges: data.edges.length });
      setLoaded(true);

      if (!containerRef.current || !window.vis) return;
      if (networkRef.current) networkRef.current.destroy();
      if (data.nodes.length === 0) {
        networkRef.current = null;
        return;
      }

      const nodes = new window.vis.DataSet(
        data.nodes.map((n) => ({
          id: n.id,
          label: n.label,
          color: riskColor(n.risk_level),
          size: n.flagged ? 20 : 14,
          shape: n.type === "GENERATOR" ? "square" : "dot",
          // vis-network renders a string `title` as plain text (it escapes
          // HTML rather than interpreting it) -- newlines, not <br/> tags,
          // are what actually produce a multi-line tooltip here.
          title: `${n.id}\nType: ${n.type}\nRisk: ${n.risk_level} (${n.risk_score})\n` +
            `Volume: ${n.total_rec_volume} · Transactions: ${n.transaction_count}\n` +
            `Degree: ${n.degree} (in ${n.in_degree} / out ${n.out_degree})\n` +
            `Betweenness: ${n.betweenness} · PageRank: ${n.pagerank?.toFixed?.(4) ?? n.pagerank}\n` +
            `GNN score: ${n.gnn_risk_score != null ? (n.gnn_risk_score * 100).toFixed(1) + "%" : "n/a"}\n` +
            `Community: ${n.community_id || "—"} · Alerts: ${n.alert_count}`,
        }))
      );
      // Defensive: de-dupe by id before handing rows to vis-network's
      // DataSet, which throws (crashing the whole page) on a second item
      // with an id it's already seen. The backend already serializes
      // build()/analyze() under a lock so this shouldn't fire in practice,
      // but a malformed/overlapping API response should degrade to
      // "one edge drawn" rather than a blank crashed graph.
      const seenEdgeIds = new Set();
      const edgeRows = [];
      for (const e of data.edges) {
        if (seenEdgeIds.has(e.id)) continue;
        seenEdgeIds.add(e.id);
        edgeRows.push({
          id: e.id, from: e.from, to: e.to, arrows: "to",
          color: e.risk_level === "HIGH" || e.risk_level === "CRITICAL" ? "#f87171" : "#475569",
          width: e.risk_score > 0 ? 2.5 : 1,
          title: `${e.transaction_type}\nTx: ${e.transaction_id || "—"}\nREC: ${e.rec_id || "—"}\n` +
            `Qty: ${e.quantity ?? "—"} · Status: ${e.status}\n` +
            `Blockchain: ${e.blockchain_status || "—"}\nRisk: ${e.risk_level} (${e.risk_score})\n` +
            `${fmtTime(e.transaction_timestamp)}`,
        });
      }
      const edges = new window.vis.DataSet(edgeRows);

      networkRef.current = new window.vis.Network(
        containerRef.current,
        { nodes, edges },
        {
          nodes: { font: { color: "#e2e8f0", size: 11 }, borderWidth: 1 },
          edges: { smooth: true },
          physics: { stabilization: true },
          interaction: { hover: true },
        }
      );

      networkRef.current.on("click", async (params) => {
        if (params.nodes.length) {
          const entityId = params.nodes[0];
          try {
            const detail = await api.graphNode(entityId);
            setSelected({ kind: "node", data: detail });
          } catch {
            /* ignore */
          }
        } else if (params.edges.length) {
          const edgeId = params.edges[0];
          const edge = data.edges.find((e) => e.id === edgeId);
          if (edge?.transaction_id) {
            try {
              const detail = await api.graphEdge(edge.transaction_id);
              setSelected({ kind: "edge", data: detail });
            } catch {
              setSelected({ kind: "edge", data: edge });
            }
          } else if (edge) {
            setSelected({ kind: "edge", data: edge });
          }
        } else {
          setSelected(null);
        }
      });
    }

    load();
    const id = setInterval(load, 10000);
    return () => {
      cancelled = true;
      clearInterval(id);
      networkRef.current?.destroy();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, resetEpoch]);

  function resetFilters() {
    setFilters({ search: "", node_type: "", risk_level: "", status: "", suspicious_only: false, since: "", until: "" });
  }

  async function recalculate() {
    setBusy(true);
    try {
      await api.graphRecalculate();
      loadOverview();
      loadSidePanels();
    } catch (err) {
      alert(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function resetGraph() {
    if (!confirm("Reset Graph? This clears graph nodes, edges, clusters and alerts (not the underlying transactions), and rebuilds from scratch as new activity arrives.")) return;
    setBusy(true);
    try {
      await api.graphReset();
      setSelected(null);
      loadOverview();
      loadSidePanels();
    } catch (err) {
      alert(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function markStatus(status) {
    if (!selected) return;
    try {
      if (selected.kind === "node" && selected.data.graph_alerts?.[0]) {
        await api.graphAlertStatus(selected.data.graph_alerts[0].alert_id, status);
      } else if (selected.kind === "cluster") {
        await api.graphClusterStatus(selected.data.cluster_id, status);
      }
      loadSidePanels();
    } catch (err) {
      alert(err.message);
    }
  }

  async function exportSelection() {
    if (!selected) return;
    const params = selected.kind === "cluster"
      ? { cluster_id: selected.data.cluster_id }
      : { entity_id: selected.data.entity_id };
    try {
      const data = await api.graphExport(params);
      downloadJson(`graph-investigation-${Date.now()}.json`, data);
      if (data.entities?.length) downloadCsv(`graph-investigation-entities-${Date.now()}.csv`, data.entities);
    } catch (err) {
      alert(err.message);
    }
  }

  function openCluster(cluster) {
    setSelected({ kind: "cluster", data: cluster });
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-1">Graph Fraud Detection</h1>
      <p className="text-slate-400 text-sm mb-6">
        REC transactions analyzed as a directed graph -- strongly connected components (circular trading), Louvain
        communities, centrality, temporal bursts, and named suspicious motifs, combined into one explainable
        per-entity risk score.
      </p>

      {overview && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <RiskCard label="Total Nodes" value={overview.total_nodes} />
          <RiskCard label="Total Edges" value={overview.total_edges} />
          <RiskCard label="Suspicious Nodes" value={overview.suspicious_nodes} tone="warn" />
          <RiskCard label="Suspicious Edges" value={overview.suspicious_edges} tone="warn" />
          <RiskCard label="Fraud Rings" value={overview.fraud_rings} tone="bad" />
          <RiskCard label="Suspicious Communities" value={overview.suspicious_communities} tone="bad" />
          <RiskCard label="High-Risk Brokers" value={overview.high_risk_brokers} tone="warn" />
          <RiskCard label="Active Graph Alerts" value={overview.active_graph_alerts} tone="bad" />
        </div>
      )}

      {gnnStatus && (
        <div
          className={`mb-4 text-xs rounded-lg px-3 py-2 border ${
            gnnStatus.available
              ? "bg-emerald-950/30 border-emerald-900 text-emerald-400"
              : "bg-slate-900 border-slate-800 text-slate-500"
          }`}
        >
          {gnnStatus.available ? (
            <>
              GNN model active: 2-layer GraphSAGE trained on the transaction graph &middot; validation F1{" "}
              {gnnStatus.metrics.val_f1?.toFixed(3)} &middot; accuracy {(gnnStatus.metrics.val_accuracy * 100).toFixed(1)}%.
              Contributes to each entity's risk score below (see "GNN model flagged..." reasons) and to a node's{" "}
              GNN Score in its investigation panel.
            </>
          ) : (
            <>
              GNN model not loaded ({gnnStatus.unavailable_reason}). Entity risk scores still work from the classical
              graph algorithms -- run <code>ml_training/train_gnn.py</code> and copy its output into{" "}
              <code>backend/models/</code> to enable GNN scoring.
            </>
          )}
        </div>
      )}

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 mb-4">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 mb-3">
          <input
            value={filters.search}
            onChange={(e) => setFilters({ ...filters, search: e.target.value })}
            placeholder="Search entity / REC ID"
            className="col-span-2 bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs"
          />
          <select value={filters.node_type} onChange={(e) => setFilters({ ...filters, node_type: e.target.value })}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs">
            {NODE_TYPES.map((t) => <option key={t} value={t}>{t || "All types"}</option>)}
          </select>
          <select value={filters.risk_level} onChange={(e) => setFilters({ ...filters, risk_level: e.target.value })}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs">
            {RISK_LEVELS.map((r) => <option key={r} value={r}>{r || "All risk levels"}</option>)}
          </select>
          <select value={filters.status} onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                  className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs">
            {EDGE_STATUSES.map((s) => <option key={s} value={s}>{s || "All statuses"}</option>)}
          </select>
          <input type="datetime-local" value={filters.since} onChange={(e) => setFilters({ ...filters, since: e.target.value })}
                 className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs" title="Since" />
          <input type="datetime-local" value={filters.until} onChange={(e) => setFilters({ ...filters, until: e.target.value })}
                 className="bg-slate-950 border border-slate-800 rounded-lg px-2 py-1.5 text-xs" title="Until" />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs text-slate-400">
            <input type="checkbox" checked={filters.suspicious_only}
                   onChange={(e) => setFilters({ ...filters, suspicious_only: e.target.checked })} />
            Show only suspicious nodes
          </label>
          <button onClick={resetFilters} className="text-xs text-emerald-400 hover:underline">Reset Filters</button>
          <div className="ml-auto flex gap-2">
            <button onClick={recalculate} disabled={busy}
                    className="text-xs bg-slate-800 hover:bg-slate-700 disabled:opacity-40 rounded-lg px-3 py-1.5">
              Recalculate
            </button>
            <button onClick={resetGraph} disabled={busy}
                    className="text-xs bg-red-900/60 hover:bg-red-900 disabled:opacity-40 text-red-200 rounded-lg px-3 py-1.5">
              Reset Graph
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-2">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-500">{stats.nodes} nodes &middot; {stats.edges} edges shown</span>
            <span className="text-[11px] text-slate-600">Drag to pan &middot; scroll to zoom &middot; click a node or edge for details</span>
          </div>
          <div className="relative h-[560px] bg-slate-900 border border-slate-800 rounded-xl">
            <div ref={containerRef} className="absolute inset-0" />
            {loaded && stats.nodes === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-center text-slate-500 text-sm px-6">
                No transactions available. Start the simulation to build the graph.
              </div>
            )}
          </div>
          <div className="flex gap-4 mt-2 text-[11px] text-slate-500">
            <span><span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: RISK_COLOR.LOW }} />Low risk</span>
            <span><span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: RISK_COLOR.MEDIUM }} />Watchlist</span>
            <span><span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: RISK_COLOR.HIGH }} />High</span>
            <span><span className="inline-block w-2.5 h-2.5 rounded-full mr-1" style={{ background: RISK_COLOR.CRITICAL }} />Critical</span>
            <span>&#9632; = Generator &middot; &#9679; = other entity</span>
          </div>
        </div>

        <div>
          {selected ? (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs uppercase tracking-wide text-slate-400">
                  {selected.kind === "node" ? "Entity" : selected.kind === "edge" ? "Transaction" : "Cluster"} Investigation
                </h3>
                <button onClick={() => setSelected(null)} className="text-slate-500 hover:text-slate-300">close</button>
              </div>

              {selected.kind === "node" && (
                <>
                  <div className="font-mono font-semibold mb-1">{selected.data.entity_id}</div>
                  <div className="mb-3">
                    <span className="badge" style={{ background: `${riskColor(selected.data.risk_level)}22`, color: riskColor(selected.data.risk_level) }}>
                      {selected.data.risk_level}
                    </span>{" "}
                    <span className="text-slate-400">risk {selected.data.risk_score}</span>
                  </div>
                  <dl className="grid grid-cols-2 gap-y-1 mb-3">
                    <dt className="text-slate-500">Type</dt><dd>{selected.data.entity_type}</dd>
                    <dt className="text-slate-500">Degree</dt><dd>{selected.data.degree} (in {selected.data.in_degree} / out {selected.data.out_degree})</dd>
                    <dt className="text-slate-500">Betweenness</dt><dd>{selected.data.betweenness_centrality}</dd>
                    <dt className="text-slate-500">PageRank</dt><dd>{selected.data.pagerank_score}</dd>
                    <dt className="text-slate-500">GNN Score</dt>
                    <dd>
                      {selected.data.gnn_risk_score != null
                        ? `${(selected.data.gnn_risk_score * 100).toFixed(1)}% fraud probability`
                        : "n/a (model not loaded)"}
                    </dd>
                    <dt className="text-slate-500">Community</dt><dd>{selected.data.community_id || "—"}</dd>
                    <dt className="text-slate-500">Total Volume</dt><dd>{selected.data.total_rec_volume}</dd>
                    <dt className="text-slate-500">Transactions</dt><dd>{selected.data.transaction_count}</dd>
                    <dt className="text-slate-500">Alerts</dt><dd>{selected.data.alert_count}</dd>
                  </dl>
                  {selected.data.cycles?.length > 0 && (
                    <div className="mb-3">
                      <div className="text-slate-500 mb-1">Detected Cycles</div>
                      {selected.data.cycles.map((c, i) => (
                        <div key={i} className="text-red-300 font-mono text-[11px]">{c.join(" -> ")} -&gt; {c[0]}</div>
                      ))}
                    </div>
                  )}
                  {selected.data.related_rec_ids?.length > 0 && (
                    <div className="mb-3">
                      <div className="text-slate-500 mb-1">Related REC IDs</div>
                      <div className="flex flex-wrap gap-1">
                        {selected.data.related_rec_ids.map((r) => (
                          <button key={r} onClick={() => onOpenRecVerify?.(r)}
                                  className="font-mono text-[11px] bg-slate-800 hover:bg-emerald-800 rounded px-2 py-0.5">
                            {r}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                  {selected.data.graph_alerts?.length > 0 && (
                    <div className="mb-3">
                      <div className="text-slate-500 mb-1">Graph Alerts ({selected.data.graph_alerts.length})</div>
                      {selected.data.graph_alerts.slice(0, 5).map((a) => (
                        <div key={a.alert_id} className="text-slate-300 mb-1">
                          <span className="text-amber-400">{a.alert_type}</span> ({a.severity}): {a.reason}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}

              {selected.kind === "edge" && (
                <dl className="grid grid-cols-2 gap-y-1">
                  <dt className="text-slate-500">Transaction</dt><dd className="font-mono">{selected.data.transaction_id || "—"}</dd>
                  <dt className="text-slate-500">REC ID</dt>
                  <dd>
                    {selected.data.rec_id ? (
                      <button onClick={() => onOpenRecVerify?.(selected.data.rec_id)} className="font-mono text-emerald-400 hover:underline">
                        {selected.data.rec_id}
                      </button>
                    ) : "—"}
                  </dd>
                  <dt className="text-slate-500">From</dt><dd className="font-mono truncate">{selected.data.source_entity || selected.data.from}</dd>
                  <dt className="text-slate-500">To</dt><dd className="font-mono truncate">{selected.data.target_entity || selected.data.to}</dd>
                  <dt className="text-slate-500">Quantity</dt><dd>{selected.data.quantity}</dd>
                  <dt className="text-slate-500">Type</dt><dd>{selected.data.relationship_type || selected.data.transaction_type}</dd>
                  <dt className="text-slate-500">Status</dt><dd>{selected.data.status}</dd>
                  <dt className="text-slate-500">Blockchain</dt>
                  <dd><BlockchainStatusBadge status={selected.data.blockchain_status} /></dd>
                  <dt className="text-slate-500">Risk</dt><dd>{selected.data.risk_level} ({selected.data.risk_score})</dd>
                  <dt className="text-slate-500">Time</dt><dd>{fmtTime(selected.data.transaction_timestamp)}</dd>
                  {selected.data.fraud_reason?.length > 0 && (
                    <>
                      <dt className="text-slate-500">Reasons</dt>
                      <dd className="text-red-300">{selected.data.fraud_reason.join("; ")}</dd>
                    </>
                  )}
                </dl>
              )}

              {selected.kind === "cluster" && (
                <>
                  <div className="font-mono font-semibold mb-1">{selected.data.cluster_id}</div>
                  <div className="mb-3">
                    <span className="badge" style={{ background: `${riskColor(selected.data.risk_level)}22`, color: riskColor(selected.data.risk_level) }}>
                      {selected.data.risk_level}
                    </span>
                  </div>
                  <p className="text-slate-300 mb-3">{selected.data.detection_reason}</p>
                  <dl className="grid grid-cols-2 gap-y-1 mb-3">
                    <dt className="text-slate-500">Members</dt><dd>{selected.data.member_count}</dd>
                    <dt className="text-slate-500">RECs</dt><dd>{selected.data.rec_count}</dd>
                    <dt className="text-slate-500">Transfers</dt><dd>{selected.data.transfer_count}</dd>
                    <dt className="text-slate-500">Graph Score</dt><dd>{selected.data.graph_score}</dd>
                  </dl>
                </>
              )}

              <div className="pt-3 mt-3 border-t border-slate-800 flex flex-wrap gap-2">
                {selected.kind === "node" && selected.data.related_rec_ids?.[0] && (
                  <button onClick={() => onOpenRecVerify?.(selected.data.related_rec_ids[0])}
                          className="bg-emerald-800 hover:bg-emerald-700 rounded px-2 py-1">Open REC Verification Portal</button>
                )}
                <button onClick={() => markStatus("UNDER_INVESTIGATION")} className="bg-slate-800 hover:bg-slate-700 rounded px-2 py-1">Mark Under Investigation</button>
                <button onClick={() => markStatus("RESOLVED")} className="bg-slate-800 hover:bg-slate-700 rounded px-2 py-1">Mark Resolved</button>
                <button onClick={() => markStatus("FALSE_POSITIVE")} className="bg-slate-800 hover:bg-slate-700 rounded px-2 py-1">Mark False Positive</button>
                <button onClick={exportSelection} className="bg-slate-800 hover:bg-slate-700 rounded px-2 py-1">Export (JSON + CSV)</button>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs">
              <div className="flex gap-1 mb-3 flex-wrap">
                {[["rings", "Fraud Rings"], ["communities", "Communities"], ["motifs", "Motifs"], ["temporal", "Temporal"]].map(([k, label]) => (
                  <button key={k} onClick={() => setTab(k)}
                          className={`px-2.5 py-1 rounded-lg ${tab === k ? "bg-emerald-700 text-white" : "bg-slate-800 text-slate-400"}`}>
                    {label}
                  </button>
                ))}
              </div>

              {tab === "rings" && (
                fraudRings.length === 0 ? <div className="text-slate-500">No circular trading loops detected.</div> :
                fraudRings.map((r) => (
                  <div key={r.cluster_id} onClick={() => openCluster(r)}
                       className="border border-red-900/60 rounded-lg p-2.5 mb-2 cursor-pointer hover:border-red-600">
                    <div className="font-mono text-red-300">{r.cluster_id}</div>
                    <div className="text-slate-400">{r.member_count} entities &middot; {r.detection_reason}</div>
                  </div>
                ))
              )}

              {tab === "communities" && (
                communities.length === 0 ? <div className="text-slate-500">No communities formed yet.</div> :
                communities.map((c) => (
                  <div key={c.cluster_id} onClick={() => openCluster(c)}
                       className="border border-slate-800 rounded-lg p-2.5 mb-2 cursor-pointer hover:border-emerald-600">
                    <div className="flex items-center justify-between">
                      <span className="font-mono">{c.cluster_id}</span>
                      <span style={{ color: riskColor(c.risk_level) }}>{c.risk_level.replace(/_/g, " ")}</span>
                    </div>
                    <div className="text-slate-400">{c.member_count} members &middot; {c.detection_reason}</div>
                  </div>
                ))
              )}

              {tab === "motifs" && motifs && (
                <div className="space-y-2">
                  {Object.entries(motifs).map(([type, hits]) => (
                    <div key={type} className="flex items-center justify-between border-b border-slate-800 pb-1.5">
                      <span className="text-slate-300">{type.replace(/_/g, " ")}</span>
                      <span className="font-semibold">{hits.length}</span>
                    </div>
                  ))}
                </div>
              )}

              {tab === "temporal" && temporal && (
                <div>
                  <div className="text-slate-500 mb-2">
                    Window {temporal.window_seconds}s &middot; threshold {temporal.max_transfers_threshold} transfers
                  </div>
                  {temporal.bursts.length === 0 ? <div className="text-slate-500">No rapid transfer bursts detected.</div> :
                    temporal.bursts.slice(0, 15).map((b, i) => (
                      <div key={i} className="border border-amber-900/50 rounded-lg p-2 mb-2">
                        <div className="text-amber-400">{b.type === "rec_velocity" ? `REC ${b.rec_id}` : b.entity_id}</div>
                        <div className="text-slate-400">{b.transfer_count} transfers in {b.window_seconds}s</div>
                      </div>
                    ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
