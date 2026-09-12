import torch  # Must stay first to prevent DLL conflicts
import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import extract, train
import uvicorn
from dotenv import load_dotenv

# ── Environment Setup ────────────────────────────────────────────────────────
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv("../../.env")

# ── Ensure Log Directory Exists ──────────────────────────────────────────────
log_dir = "../../logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# ── UTF-8 Logging for Windows ────────────────────────────────────────────────
file_handler   = logging.FileHandler(os.path.join(log_dir, "fastapi.log"), encoding="utf-8")
stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger("gst2_fastapi")

# ── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="GST2 Converter ML Service",
    description="OCR + Table Extraction + LLM Validation + Auto-Training",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000",
                   "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(extract.router, prefix="/api/extract", tags=["Extraction"])
app.include_router(train.router,   prefix="/api/train",   tags=["Training"])


@app.get("/health")
async def health_check():
    try:
        import paddle
        paddle_ok = True
    except Exception:
        paddle_ok = False

    return {
        "status":         "running",
        "gpu_available":  torch.cuda.is_available(),
        "gpu_name":       torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "vram_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
                          if torch.cuda.is_available() else "0",
        "paddle_ok":      paddle_ok,
        "service":        "GST2 ML Service v3.0",
    }


@app.get("/model/version")
async def model_version():
    """Return current fine-tuned model version (auto-updated after each training run)."""
    from services.pipeline import get_model_version
    return get_model_version()


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True, log_level="info")
