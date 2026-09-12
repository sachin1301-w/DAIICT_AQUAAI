// Derive from the page's own hostname rather than hardcoding "localhost":
// when this page is opened from a phone via the dev machine's LAN IP
// (e.g. scanning the QR code on the Blockchain Verify page), "localhost"
// would resolve to the phone itself, not the dev machine.
const BASE = import.meta.env.VITE_API_BASE || `http://${window.location.hostname}:8000`;

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const isJson = res.headers.get("content-type")?.includes("application/json");
  const body = isJson ? await res.json() : await res.text();
  if (!res.ok) {
    const message = typeof body === "object" ? body.detail || JSON.stringify(body) : body;
    throw new Error(message);
  }
  return body;
}

export const api = {
  health: () => request("/api/health"),
  dashboardSummary: () => request("/api/dashboard/summary"),

  generators: () => request("/api/generators"),
  createGenerator: (payload) => request("/api/generators", { method: "POST", body: JSON.stringify(payload) }),

  transactions: (limit = 100) => request(`/api/transactions?limit=${limit}`),
  certificates: (status) => request(`/api/certificates${status ? `?status=${status}` : ""}`),
  retireCertificate: (recId) => request(`/api/certificates/${recId}/retire`, { method: "POST", body: "{}" }),
  revokeCertificate: (recId, reason) =>
    request(`/api/certificates/${recId}/revoke`, { method: "POST", body: JSON.stringify({ reason }) }),

  alerts: (riskLevel) => request(`/api/alerts${riskLevel ? `?risk_level=${riskLevel}` : ""}`),
  alert: (alertId) => request(`/api/alerts/${alertId}`),

  fraudClusters: () => request("/api/fraud-clusters"),
  networkGraph: () => request("/api/network-graph"),

  blockchainStatus: () => request("/api/blockchain/status"),
  verifyRec: (recId) => request(`/api/verify/${recId}`),

  // REC Verification Portal
  getRec: (recId) => request(`/api/rec/${recId}`),
  verifyRecPortal: (recId, payload) =>
    request(`/api/rec/${recId}/verify`, { method: "POST", body: JSON.stringify(payload || {}) }),
  recAuditHistory: (recId) => request(`/api/rec/${recId}/history`),
  recVerificationHistory: (recId) => request(`/api/rec/${recId}/verification-history`),
  verificationHistory: (params) => {
    const qs = new URLSearchParams(Object.entries(params || {}).filter(([, v]) => v !== undefined && v !== ""));
    const s = qs.toString();
    return request(`/api/verification/history${s ? `?${s}` : ""}`);
  },
  getVerification: (verificationId) => request(`/api/verification/${verificationId}`),
  verificationReport: (verificationId) =>
    request("/api/verification/report", { method: "POST", body: JSON.stringify({ verification_id: verificationId }) }),

  // `config` is optional -- when passed, the backend applies it before
  // starting, so the currently-selected fraud probability/interval always
  // takes effect immediately on a fresh run.
  simulationStart: (config) => request("/api/simulation/start", { method: "POST", body: JSON.stringify(config || {}) }),
  simulationStop: () => request("/api/simulation/stop", { method: "POST" }),
  simulationStatus: () => request("/api/simulation/status"),
  simulationConfig: (payload) => request("/api/simulation/config", { method: "POST", body: JSON.stringify(payload) }),
  simulationReset: () => request("/api/simulation/reset", { method: "POST" }),

  submitGeneration: (payload) => request("/api/generation", { method: "POST", body: JSON.stringify(payload) }),
  submitTransfer: (payload) => request("/api/transactions/transfer", { method: "POST", body: JSON.stringify(payload) }),

  // Advanced Graph Fraud Detection
  graphOverview: () => request("/api/graph/overview"),
  graphNodes: (params) => request(`/api/graph/nodes${qs(params)}`),
  graphEdges: (params) => request(`/api/graph/edges${qs(params)}`),
  graphNetwork: (params) => request(`/api/graph/network${qs(params)}`),
  graphNode: (entityId) => request(`/api/graph/node/${encodeURIComponent(entityId)}`),
  graphEdge: (transactionId) => request(`/api/graph/edge/${encodeURIComponent(transactionId)}`),
  graphClusters: (params) => request(`/api/graph/clusters${qs(params)}`),
  graphFraudRings: () => request("/api/graph/fraud-rings"),
  graphTemporalAnalysis: () => request("/api/graph/temporal-analysis"),
  graphCentrality: () => request("/api/graph/centrality"),
  graphMotifs: () => request("/api/graph/motifs"),
  graphGnnStatus: () => request("/api/graph/gnn-status"),
  graphRecalculate: () => request("/api/graph/recalculate", { method: "POST" }),
  graphReset: () => request("/api/graph/reset", { method: "POST" }),
  graphExport: (params) => request(`/api/graph/export${qs(params)}`),
  graphAlertStatus: (alertId, status) =>
    request(`/api/graph/alerts/${alertId}/status`, { method: "POST", body: JSON.stringify({ status }) }),
  graphClusterStatus: (clusterId, status) =>
    request(`/api/graph/clusters/${clusterId}/status`, { method: "POST", body: JSON.stringify({ status }) }),
};

function qs(params) {
  if (!params) return "";
  const q = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== "")
  );
  const s = q.toString();
  return s ? `?${s}` : "";
}

export function connectLiveFeed(onMessage) {
  const wsBase = BASE.replace(/^http/, "ws");
  let ws;
  let closedByClient = false;

  function connect() {
    ws = new WebSocket(`${wsBase}/ws`);
    ws.onmessage = (event) => {
      try {
        onMessage(JSON.parse(event.data));
      } catch {
        /* ignore malformed frame */
      }
    };
    ws.onclose = () => {
      if (!closedByClient) setTimeout(connect, 2000);
    };
  }
  connect();

  return () => {
    closedByClient = true;
    ws?.close();
  };
}
