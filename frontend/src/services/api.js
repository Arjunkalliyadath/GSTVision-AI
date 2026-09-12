import axios from "axios";

const DJANGO_BASE  = "http://localhost:8000/api";
const FASTAPI_BASE = "http://localhost:8001";

const api = axios.create({
  baseURL: DJANGO_BASE,
  timeout: 30000,
});

// ── Upload & Extraction ───────────────────────────────────────────────────────

export const uploadFiles = async (files) => {
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  const response = await api.post("/documents/upload/", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
};

export const getJobStatus = async (mlJobId) => {
  const response = await api.get(`/documents/status/${mlJobId}/`);
  return response.data;
};

export const getDownloadUrl = (excelFilename) =>
  `${DJANGO_BASE}/documents/download/${excelFilename}/`;

export const saveCorrection = async (jobId, corrections, originalData) => {
  const response = await api.post("/documents/correction/", {
    job_id: jobId,
    corrections,
    original_data: originalData,
  });
  return response.data;
};

// ── Model & Training Info ─────────────────────────────────────────────────────

/** Current fine-tuned model version (auto-updated after each training run) */
export const getModelVersion = async () => {
  const response = await api.get("/documents/training/version/");
  return response.data;
};

/** Dataset stats + auto-train schedule */
export const getTrainingStats = async () => {
  const response = await api.get("/documents/training/stats/");
  return response.data;
};

/** Is auto-training currently running? */
export const getTrainingCurrentStatus = async () => {
  const response = await api.get("/documents/training/status/");
  return response.data;
};
