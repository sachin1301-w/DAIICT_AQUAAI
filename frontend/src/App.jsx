import React, { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard.jsx";
import CertificateTable from "./components/CertificateTable.jsx";
import FraudAlerts from "./components/FraudAlerts.jsx";
import FraudClusters from "./components/FraudClusters.jsx";
import NetworkGraph from "./components/NetworkGraph.jsx";
import GenerateTransaction from "./components/GenerateTransaction.jsx";
import QRVerification from "./components/QRVerification.jsx";
import RecVerification from "./components/RecVerification.jsx";
import VerificationHistory from "./components/VerificationHistory.jsx";
import GraphFraudDetection from "./components/GraphFraudDetection.jsx";
import { api, connectLiveFeed } from "./api.js";

const NAV = [
  { id: "dashboard", label: "Dashboard" },
  { id: "certificates", label: "Certificates" },
  { id: "alerts", label: "Fraud Alerts" },
  { id: "clusters", label: "Fraud Clusters" },
  { id: "graph", label: "Network Graph" },
  { id: "graphfraud", label: "Graph Fraud Detection" },
  { id: "simulation", label: "Simulation Control" },
  { id: "recverify", label: "Verify REC" },
  { id: "verhistory", label: "Verification History" },
  { id: "verify", label: "Blockchain Verify" },
];

function initialViewFromUrl() {
  const recId = new URLSearchParams(window.location.search).get("verify");
  return recId ? "verify" : "dashboard";
}

function initialRecIdFromUrl() {
  return new URLSearchParams(window.location.search).get("verify") || "";
}

export default function App() {
  const [view, setView] = useState(initialViewFromUrl);
  const [feed, setFeed] = useState([]);
  const [chainStatus, setChainStatus] = useState(null);
  const [graphFocus, setGraphFocus] = useState(null);
  // Bumped on every "simulation_reset" broadcast. Passed down as a prop so
  // every live view (Dashboard, graph, alerts, clusters, certificates)
  // refetches immediately instead of waiting out its normal poll interval.
  const [resetEpoch, setResetEpoch] = useState(0);
  const [resetBanner, setResetBanner] = useState(false);
  // Bumped on every "rec_verified" broadcast so Dashboard's Verification
  // Summary card and the Verification History page refresh immediately --
  // separate from resetEpoch since a verification never clears anything.
  const [verificationEpoch, setVerificationEpoch] = useState(0);
  // Bumped on "graph_reset" broadcasts (from the dedicated Reset Graph
  // button, separate from a full Reset Transactions) so the Graph Fraud
  // Detection page's network view clears immediately too.
  const [graphEpoch, setGraphEpoch] = useState(0);
  const [deepLinkRecId, setDeepLinkRecId] = useState("");
  const [recVerifyNonce, setRecVerifyNonce] = useState(0);
  const [toasts, setToasts] = useState([]);

  function pushToast(toast) {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { id, ...toast }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 6000);
  }

  function openClusterGraph(entityIds) {
    setGraphFocus(entityIds);
    setView("graph");
  }

  function openRecVerify(recId) {
    setDeepLinkRecId(recId);
    setRecVerifyNonce((n) => n + 1);
    setView("recverify");
  }

  useEffect(() => {
    const disconnect = connectLiveFeed((msg) => {
      if (msg.type === "pipeline_result") {
        setFeed((prev) => [msg.data, ...prev].slice(0, 60));
      } else if (msg.type === "simulation_reset") {
        setFeed([]);
        setGraphFocus(null);
        setResetEpoch((e) => e + 1);
        setResetBanner(true);
        setTimeout(() => setResetBanner(false), 4000);
      } else if (msg.type === "rec_verified") {
        setVerificationEpoch((e) => e + 1);
        pushToast({
          tone: msg.result === "TAMPERED" ? "danger" : msg.result === "VALID" ? "ok" : "warn",
          text: `${msg.rec_id} verified: ${msg.result}`,
        });
      } else if (msg.type === "tamper_detected") {
        pushToast({ tone: "danger", text: `TAMPER DETECTED on ${msg.rec_id}: ${msg.reason}` });
      } else if (msg.type === "tamper_simulated") {
        pushToast({ tone: "warn", text: `[demo] Simulated tampering on ${msg.rec_id} -- run Verify REC to catch it` });
      } else if (msg.type === "graph_reset") {
        setGraphEpoch((e) => e + 1);
      } else if (msg.type === "fraud_ring_detected") {
        pushToast({ tone: "danger", text: "New circular trading loop (fraud ring) detected in the graph" });
      } else if (msg.type === "community_detected") {
        pushToast({ tone: "warn", text: "New suspicious community detected in the graph" });
      } else if (msg.type === "motif_detected") {
        pushToast({ tone: "warn", text: "New suspicious pattern (motif) detected in the graph" });
      }
    });
    return disconnect;
  }, []);

  useEffect(() => {
    const poll = () => api.blockchainStatus().then(setChainStatus).catch(() => setChainStatus({ status: "OFFLINE" }));
    poll();
    const id = setInterval(poll, 10000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="min-h-screen flex bg-slate-950">
      <aside className="w-60 shrink-0 border-r border-slate-800 flex flex-col p-4 gap-1">
        <div className="flex items-center gap-2 pb-4 mb-3 border-b border-slate-800">
          <span className="text-2xl">&#9889;</span>
          <div>
            <div className="font-bold text-sm">REC Guard</div>
            <div className="text-[11px] text-slate-400">Blockchain + AI Fraud Platform</div>
          </div>
        </div>
        {NAV.map((n) => (
          <button
            key={n.id}
            onClick={() => setView(n.id)}
            className={`text-left px-3 py-2 rounded-lg text-sm transition ${
              view === n.id ? "bg-emerald-700 text-white" : "text-slate-400 hover:bg-slate-900 hover:text-slate-100"
            }`}
          >
            {n.label}
          </button>
        ))}
        <div className="mt-auto pt-3 border-t border-slate-800 text-[11px]">
          <div
            className={`rounded-full px-3 py-1.5 text-center ${
              chainStatus?.status === "ONLINE" ? "bg-emerald-900/40 text-emerald-400" : "bg-red-900/40 text-red-400"
            }`}
          >
            Blockchain {chainStatus?.status || "checking..."}
          </div>
        </div>
      </aside>

      <main className="flex-1 p-8 overflow-y-auto max-h-screen">
        {resetBanner && (
          <div className="mb-4 text-xs bg-emerald-950/40 border border-emerald-900 text-emerald-400 rounded-lg px-3 py-2">
            Simulation data reset -- all transactions, alerts and graph data cleared.
          </div>
        )}
        {view === "dashboard" && <Dashboard feed={feed} resetEpoch={resetEpoch + verificationEpoch} />}
        {view === "certificates" && <CertificateTable resetEpoch={resetEpoch} />}
        {view === "alerts" && <FraudAlerts resetEpoch={resetEpoch} />}
        {view === "clusters" && <FraudClusters onOpenGraph={openClusterGraph} resetEpoch={resetEpoch} />}
        {view === "graph" && (
          <NetworkGraph focusEntities={graphFocus} onClearFocus={() => setGraphFocus(null)} resetEpoch={resetEpoch} />
        )}
        {view === "graphfraud" && (
          <GraphFraudDetection resetEpoch={resetEpoch + graphEpoch} onOpenRecVerify={openRecVerify} />
        )}
        {view === "simulation" && <GenerateTransaction resetEpoch={resetEpoch} />}
        {view === "recverify" && (
          <RecVerification key={`recverify-${recVerifyNonce}`} resetEpoch={resetEpoch} initialRecId={deepLinkRecId} />
        )}
        {view === "verhistory" && <VerificationHistory resetEpoch={verificationEpoch} />}
        {view === "verify" && <QRVerification initialRecId={initialRecIdFromUrl()} />}
      </main>

      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50 w-80">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`text-xs rounded-lg px-3 py-2 shadow-lg border ${
              t.tone === "danger"
                ? "bg-red-950/90 border-red-800 text-red-200"
                : t.tone === "ok"
                ? "bg-emerald-950/90 border-emerald-800 text-emerald-200"
                : "bg-amber-950/90 border-amber-800 text-amber-200"
            }`}
          >
            {t.text}
          </div>
        ))}
      </div>
    </div>
  );
}
