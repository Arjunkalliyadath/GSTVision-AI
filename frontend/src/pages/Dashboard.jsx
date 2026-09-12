import { useState, useCallback, useRef, useEffect } from "react";
import { uploadFiles, getJobStatus, getDownloadUrl, getModelVersion } from "../services/api";

const CARD = {
  background: "white",
  borderRadius: "14px",
  padding: "24px",
  boxShadow: "0 2px 16px rgba(0,0,0,0.07)",
  marginBottom: "24px",
};

const STATUS_COLOR = {
  queued:     "#ff9800",
  processing: "#1976d2",
  complete:   "#2e7d32",
  failed:     "#c62828",
};

const STATUS_ICON = {
  queued:     "⏳",
  processing: "⚙️",
  complete:   "✅",
  failed:     "❌",
};

const ALLOWED = new Set([
  "pdf","jpg","jpeg","png","bmp","tiff","tif","webp","gif","heic","heif"
]);

export default function Dashboard() {
  const [dragging,     setDragging]     = useState(false);
  const [jobs,         setJobs]         = useState([]);
  const [uploading,    setUploading]    = useState(false);
  const [uploadError,  setUploadError]  = useState("");
  const [modelVersion, setModelVersion] = useState(null);
  const fileInputRef = useRef();

  // Load model version once on mount
  useEffect(() => {
    getModelVersion()
      .then(setModelVersion)
      .catch(() => setModelVersion({ version: "base" }));
  }, []);

  const handleFiles = useCallback(async (files) => {
    const list = Array.from(files);
    setUploadError("");

    const allowed  = list.filter(f => ALLOWED.has(f.name.split(".").pop()?.toLowerCase()));
    const rejected = list.length - allowed.length;

    if (allowed.length === 0) {
      setUploadError("No supported files. Upload PDF, JPG, PNG, HEIC, or TIFF.");
      return;
    }
    if (rejected > 0)
      setUploadError(`${rejected} file(s) skipped (unsupported). Processing ${allowed.length} valid file(s).`);

    setUploading(true);
    try {
      const response = await uploadFiles(allowed);
      const newJobs  = response.jobs.map(job => ({
        ml_job_id: job.ml_job_id,
        filename:  job.filename || "Unknown file",
        status:    "processing",
        percent:   0,
        message:   "Queued for processing…",
      }));
      setJobs(prev => [...newJobs, ...prev]);
      newJobs.forEach(job => { if (job.ml_job_id) pollJob(job.ml_job_id); });
    } catch (err) {
      setUploadError(err.response?.data?.error || err.message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }, []);

  const pollJob = (mlJobId) => {
    const poll = async () => {
      try {
        const data = await getJobStatus(mlJobId);
        setJobs(prev => prev.map(j => j.ml_job_id === mlJobId ? { ...j, ...data } : j));
        if (data.status !== "complete" && data.status !== "failed")
          setTimeout(poll, 2000);
        else if (data.status === "complete")
          // Refresh model version badge after each extraction completes
          getModelVersion().then(setModelVersion).catch(() => {});
      } catch {
        setTimeout(poll, 4000);
      }
    };
    poll();
  };

  return (
    <div style={{ maxWidth: "980px", margin: "0 auto", padding: "32px 24px" }}>

      {/* Model Version Badge */}
      {modelVersion && (
        <div style={{
          display: "flex", justifyContent: "flex-end", marginBottom: "12px"
        }}>
          <span style={{
            background: modelVersion.version === "base" ? "#e3f2fd" : "#e8f5e9",
            color:      modelVersion.version === "base" ? "#1565c0" : "#2e7d32",
            padding: "5px 14px", borderRadius: "20px",
            fontSize: "12px", fontWeight: "600",
            border: `1px solid ${modelVersion.version === "base" ? "#90caf9" : "#81c784"}`,
          }}>
            🧠 Model: {modelVersion.version}
            {modelVersion.accuracy
              ? ` · Accuracy ${(modelVersion.accuracy * 100).toFixed(1)}%`
              : ""}
          </span>
        </div>
      )}

      {/* Upload Zone */}
      <div style={CARD}>
        <h2 style={{ margin: "0 0 6px", color: "#1a237e", fontSize: "20px" }}>
          📤 Upload GST 2A Statements
        </h2>
        <p style={{ margin: "0 0 20px", color: "#777", fontSize: "13px" }}>
          Supports PDF, JPG, JPEG, PNG, HEIC, TIFF, BMP, WEBP — including phone photos of statements
        </p>

        <div
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={e => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
          style={{
            border: `2px dashed ${dragging ? "#1565c0" : "#90caf9"}`,
            borderRadius: "12px",
            padding: "52px 24px",
            textAlign: "center",
            cursor: "pointer",
            background: dragging ? "#e3f2fd" : "#f8faff",
            transition: "all 0.2s",
          }}
        >
          <div style={{ fontSize: "52px", marginBottom: "14px" }}>
            {dragging ? "📂" : "📁"}
          </div>
          <div style={{ fontSize: "18px", fontWeight: "700", color: "#1a237e", marginBottom: "8px" }}>
            {dragging ? "Drop files here!" : "Drag & drop or click to browse"}
          </div>
          <div style={{ fontSize: "13px", color: "#aaa" }}>
            {uploading ? "⚙️ Uploading…" : "Max 100 MB per file · Multiple files at once"}
          </div>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.tif,.webp,.gif,.heic,.heif"
          style={{ display: "none" }}
          onChange={e => handleFiles(e.target.files)}
        />

        {uploadError && (
          <div style={{
            marginTop: "12px", padding: "10px 14px",
            background: "#fff3e0", border: "1px solid #ffb74d",
            borderRadius: "8px", color: "#e65100", fontSize: "13px",
          }}>
            ⚠️ {uploadError}
          </div>
        )}
      </div>

      {/* Jobs List */}
      {jobs.length > 0 && (
        <div style={CARD}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
            <h3 style={{ margin: 0, color: "#1a237e" }}>📋 Processing Jobs</h3>
            <span style={{
              background: "#e3f2fd", color: "#1565c0",
              padding: "4px 12px", borderRadius: "20px",
              fontSize: "13px", fontWeight: "600"
            }}>
              {jobs.length} file{jobs.length !== 1 ? "s" : ""}
            </span>
          </div>
          {jobs.map((job, idx) => (
            <JobCard key={job.ml_job_id || idx} job={job} />
          ))}
        </div>
      )}

      {jobs.length === 0 && !uploading && (
        <div style={{ ...CARD, textAlign: "center", padding: "56px", color: "#bbb" }}>
          <div style={{ fontSize: "64px", marginBottom: "16px" }}>📊</div>
          <div style={{ fontSize: "15px", color: "#999" }}>
            No uploads yet — select your GST 2A files above to get started.
          </div>
        </div>
      )}
    </div>
  );
}

function JobCard({ job }) {
  const color = STATUS_COLOR[job.status] || "#999";

  return (
    <div style={{
      border: `1px solid ${color}30`,
      borderRadius: "10px",
      padding: "16px 18px",
      marginBottom: "12px",
      background: job.status === "complete" ? "#f1f8e9" :
                  job.status === "failed"   ? "#fce4ec" : "#fafafa",
    }}>
      {/* Top row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <span style={{ fontWeight: "700", color: "#1a237e", fontSize: "14px" }}>
            {STATUS_ICON[job.status]} {job.filename}
          </span>
          {job.model_version && (
            <span style={{
              marginLeft: "10px", fontSize: "11px",
              background: "#e8f5e9", color: "#2e7d32",
              padding: "2px 8px", borderRadius: "10px",
            }}>
              🧠 {job.model_version}
            </span>
          )}
        </div>
        <span style={{
          background: color, color: "white",
          padding: "3px 10px", borderRadius: "20px",
          fontSize: "11px", fontWeight: "700",
          textTransform: "uppercase", marginLeft: "12px",
        }}>
          {job.status}
        </span>
      </div>

      {/* Progress bar */}
      {(job.status === "processing" || job.status === "queued") && (
        <div style={{
          background: "#e3f2fd", borderRadius: "4px",
          height: "6px", overflow: "hidden", marginBottom: "8px",
        }}>
          <div style={{
            width: `${job.percent || 0}%`,
            height: "100%",
            background: "linear-gradient(90deg, #1565c0, #42a5f5)",
            borderRadius: "4px",
            transition: "width 0.5s ease",
            animation: "pulse 1.5s infinite",
          }} />
        </div>
      )}

      {/* Status message */}
      {job.message && (
        <div style={{ fontSize: "12px", color: "#666", marginBottom: "8px" }}>
          {job.message}
        </div>
      )}

      {/* Warning */}
      {job.warning && (
        <div style={{
          background: "#fff8e1", border: "1px solid #ffe082",
          borderRadius: "6px", padding: "8px 12px",
          fontSize: "12px", color: "#f57f17", marginBottom: "8px",
        }}>
          ⚠️ {job.warning}
        </div>
      )}

      {/* Complete stats + download */}
      {job.status === "complete" && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "8px" }}>
          <div style={{ display: "flex", gap: "16px", fontSize: "12px", color: "#555" }}>
            <span>📄 <strong>{job.records_extracted}</strong> records</span>
            {job.confidence > 0 && (
              <span>🎯 <strong>{(job.confidence * 100).toFixed(0)}%</strong> confidence</span>
            )}
            {job.processing_time && (
              <span>⏱ <strong>{job.processing_time}s</strong></span>
            )}
          </div>
          {job.excel_filename && (
            <a
              href={getDownloadUrl(job.excel_filename)}
              download
              style={{
                display: "inline-flex", alignItems: "center", gap: "6px",
                background: "linear-gradient(135deg, #1565c0, #0d47a1)",
                color: "white", padding: "8px 18px",
                borderRadius: "8px", textDecoration: "none",
                fontSize: "13px", fontWeight: "600",
                boxShadow: "0 2px 6px rgba(21,101,192,0.35)",
              }}
            >
              ⬇️ Download Excel
            </a>
          )}
        </div>
      )}

      {/* Failed */}
      {job.status === "failed" && job.error && (
        <div style={{ fontSize: "12px", color: "#c62828" }}>
          ❌ {job.error}
        </div>
      )}
    </div>
  );
}
