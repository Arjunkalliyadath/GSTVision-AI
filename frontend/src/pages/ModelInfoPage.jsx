// ModelInfoPage.jsx — Read-only view of model version + auto-training status
// Users can no longer trigger training manually.
// Training fires automatically in the background every AUTO_TRAIN_EVERY_N uploads.

import { useState, useEffect } from "react";
import { getModelVersion, getTrainingStats, getTrainingCurrentStatus } from "../services/api";

const CARD = {
  background: "white",
  borderRadius: "14px",
  padding: "24px",
  boxShadow: "0 2px 16px rgba(0,0,0,0.07)",
  marginBottom: "24px",
};

export default function ModelInfoPage() {
  const [version,  setVersion]  = useState(null);
  const [stats,    setStats]    = useState(null);
  const [live,     setLive]     = useState(null);
  const [error,    setError]    = useState("");

  const load = async () => {
    try {
      const [v, s, l] = await Promise.all([
        getModelVersion(),
        getTrainingStats(),
        getTrainingCurrentStatus(),
      ]);
      setVersion(v);
      setStats(s);
      setLive(l);
      setError("");
    } catch {
      setError("Could not reach the ML service. Make sure FastAPI is running on port 8001.");
    }
  };

  useEffect(() => {
    load();
    // Poll every 5 s so the training indicator updates in real time
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  const isTraining = live?.is_training ?? false;

  return (
    <div style={{ maxWidth: "760px", margin: "0 auto", padding: "32px 24px" }}>

      <h2 style={{ color: "#1a237e", margin: "0 0 24px", fontSize: "22px" }}>
        🧠 AI Model Information
      </h2>

      {error && (
        <div style={{
          background: "#fce4ec", border: "1px solid #ef9a9a",
          borderRadius: "10px", padding: "14px 18px",
          color: "#c62828", fontSize: "13px", marginBottom: "20px",
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* ── Current Model Version ─────────────────────────────────────────── */}
      <div style={CARD}>
        <h3 style={{ margin: "0 0 16px", color: "#1a237e" }}>📌 Current Model Version</h3>
        {version ? (
          <div>
            <div style={{
              display: "inline-block",
              background: version.version === "base" ? "#e3f2fd" : "#e8f5e9",
              color:      version.version === "base" ? "#1565c0" : "#2e7d32",
              border:     `1px solid ${version.version === "base" ? "#90caf9" : "#81c784"}`,
              padding: "8px 20px", borderRadius: "24px",
              fontSize: "18px", fontWeight: "700", marginBottom: "16px",
            }}>
              {version.version === "base" ? "🔵 Base Model" : `🟢 ${version.version}`}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
              {[
                ["Trained At",      version.trained_at
                                      ? new Date(version.trained_at).toLocaleString()
                                      : "Not yet trained"],
                ["Training Epochs", version.epochs || "—"],
                ["Training Files",  version.training_files || "—"],
                ["Accuracy",        version.accuracy
                                      ? `${(version.accuracy * 100).toFixed(1)}%`
                                      : "—"],
                ["Final Loss",      version.final_loss
                                      ? version.final_loss.toFixed(4)
                                      : "—"],
              ].map(([label, val]) => (
                <div key={label} style={{
                  background: "#f8faff", borderRadius: "8px", padding: "12px 16px",
                }}>
                  <div style={{ fontSize: "11px", color: "#888", marginBottom: "4px" }}>{label}</div>
                  <div style={{ fontSize: "15px", fontWeight: "600", color: "#1a237e" }}>{val}</div>
                </div>
              ))}
            </div>

            {version.note && (
              <p style={{ margin: "14px 0 0", fontSize: "13px", color: "#888" }}>
                ℹ️ {version.note}
              </p>
            )}
          </div>
        ) : (
          <Spinner />
        )}
      </div>

      {/* ── Auto-Training Status ──────────────────────────────────────────── */}
      <div style={CARD}>
        <h3 style={{ margin: "0 0 16px", color: "#1a237e" }}>⚙️ Auto-Training Status</h3>

        {/* Live indicator */}
        <div style={{
          display: "flex", alignItems: "center", gap: "12px",
          padding: "14px 18px", borderRadius: "10px",
          background: isTraining ? "#e8f5e9" : "#f5f5f5",
          border: `1px solid ${isTraining ? "#81c784" : "#e0e0e0"}`,
          marginBottom: "18px",
        }}>
          {isTraining ? (
            <>
              <span style={{ fontSize: "22px" }}>🔄</span>
              <div>
                <div style={{ fontWeight: "700", color: "#2e7d32", fontSize: "14px" }}>
                  Training in progress…
                </div>
                <div style={{ fontSize: "12px", color: "#555" }}>
                  The model is being fine-tuned in the background. You can continue uploading files.
                </div>
              </div>
            </>
          ) : (
            <>
              <span style={{ fontSize: "22px" }}>💤</span>
              <div>
                <div style={{ fontWeight: "700", color: "#555", fontSize: "14px" }}>
                  Idle — waiting for enough data
                </div>
                <div style={{ fontSize: "12px", color: "#888" }}>
                  {live?.message || "Training triggers automatically when enough files are uploaded."}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Schedule */}
        {stats && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px" }}>
            {[
              ["Documents Uploaded",    stats.raw_documents,       "📄"],
              ["Next Training Trigger", `${stats.next_auto_train_at} uploads`, "🎯"],
              ["Files Until Retrain",   stats.uploads_until_retrain, "⏳"],
              ["Epochs per Run",        stats.auto_train_epochs,   "🔁"],
              ["Train Every N Uploads", stats.auto_train_every_n,  "📈"],
              ["Annotated (Corrected)", stats.annotated_documents, "✏️"],
            ].map(([label, val, icon]) => (
              <div key={label} style={{
                background: "#f8faff", borderRadius: "8px",
                padding: "12px 14px", textAlign: "center",
              }}>
                <div style={{ fontSize: "18px", marginBottom: "4px" }}>{icon}</div>
                <div style={{ fontSize: "11px", color: "#888", marginBottom: "4px" }}>{label}</div>
                <div style={{ fontSize: "16px", fontWeight: "700", color: "#1a237e" }}>{val}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── How It Works ─────────────────────────────────────────────────── */}
      <div style={{ ...CARD, background: "#f8faff" }}>
        <h3 style={{ margin: "0 0 14px", color: "#1a237e" }}>💡 How Automatic Training Works</h3>
        <ol style={{ margin: 0, paddingLeft: "20px", lineHeight: "2", color: "#555", fontSize: "14px" }}>
          <li><strong>Upload</strong> GST 2A statements — system extracts data automatically.</li>
          <li>Every upload is saved as training data in the background.</li>
          <li>After every <strong>{stats?.auto_train_every_n ?? 5} uploads</strong>, fine-tuning starts automatically.</li>
          <li>Training runs for <strong>{stats?.auto_train_epochs ?? 10} epochs</strong> using your historical data.</li>
          <li>The model version updates and future extractions are more accurate.</li>
          <li><strong>No action needed from you</strong> — it all happens invisibly.</li>
        </ol>
      </div>

      <div style={{ textAlign: "center", color: "#bbb", fontSize: "12px" }}>
        Auto-refreshes every 5 seconds
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{ textAlign: "center", padding: "24px", color: "#90caf9" }}>
      ⚙️ Loading model information…
    </div>
  );
}
