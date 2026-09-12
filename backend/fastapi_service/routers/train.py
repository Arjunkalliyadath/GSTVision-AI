# routers/train.py  (v4.0 — AUTO-TRAINING + CALIBRATION PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────
# CHANGES v4.0:
#   • Added /start-calibration — triggers image pipeline training on training2/
#   • Added /calibration/status — live calibration progress
#   • All v3.0 endpoints preserved (dataset/stats, status/current, model/version)
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import logging
import asyncio
from fastapi import APIRouter, BackgroundTasks
from pathlib import Path
from typing import Dict, Any

router = APIRouter()
logger = logging.getLogger("gst2_fastapi.train")

TRAINING_DATA_PATH    = os.getenv("TRAINING_DATA_PATH",    "../../training_data")
TRAINING2_DATA_PATH   = os.getenv("TRAINING2_DATA_PATH",   "../../training2/gstr2a_bw")
LAYOUTLMV3_MODEL_PATH = os.getenv("LAYOUTLMV3_MODEL_PATH", "../../ml_models/layoutlmv3")
AUTO_TRAIN_EVERY_N    = int(os.getenv("AUTO_TRAIN_EVERY_N", "5"))
AUTO_TRAIN_EPOCHS     = int(os.getenv("AUTO_TRAIN_EPOCHS",  "10"))

# Calibration job state
_calibration_status: Dict[str, Any] = {
    "status": "idle",
    "percent": 0,
    "message": "No calibration running.",
}
_calibration_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# DATASET STATS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/dataset/stats")
async def get_dataset_stats():
    """Training dataset counts + auto-train schedule info."""
    raw_dir       = Path(TRAINING_DATA_PATH) / "raw"
    annotated_dir = Path(TRAINING_DATA_PATH) / "annotated"

    raw_count       = len(list(raw_dir.glob("*.json")))       if raw_dir.exists()       else 0
    annotated_count = len(list(annotated_dir.glob("*.json"))) if annotated_dir.exists() else 0
    total_count     = raw_count + annotated_count

    next_train_at = ((raw_count // AUTO_TRAIN_EVERY_N) + 1) * AUTO_TRAIN_EVERY_N

    return {
        "raw_documents":        raw_count,
        "annotated_documents":  annotated_count,
        "total_documents":      total_count,
        "auto_train_every_n":   AUTO_TRAIN_EVERY_N,
        "auto_train_epochs":    AUTO_TRAIN_EPOCHS,
        "next_auto_train_at":   next_train_at,
        "uploads_until_retrain": max(0, next_train_at - raw_count),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CURRENT TRAINING STATUS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status/current")
async def get_current_training_status():
    """Is a training job running right now?"""
    from services.pipeline import _training_lock
    return {
        "is_training": _training_lock.locked(),
        "message":     "Training in progress…" if _training_lock.locked()
                       else "Idle — auto-training triggers automatically.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION PIPELINE (NEW in v4.0)
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/start-calibration")
async def start_calibration(
    background_tasks: BackgroundTasks,
    epochs: int = 30,
):
    """
    Start the image pipeline calibration/training on training2/ data.

    This runs ground truth extraction from PDFs, then evaluates
    the image OCR pipeline on JPG/PNG files for the specified number of epochs.
    """
    global _calibration_status

    if _calibration_lock.locked():
        return {
            "success": False,
            "message": "Calibration already running.",
            "status": _calibration_status,
        }

    _calibration_status = {
        "status": "starting",
        "percent": 0,
        "message": "Calibration queued…",
    }

    background_tasks.add_task(_run_calibration, epochs)

    return {
        "success": True,
        "message": f"Calibration started with {epochs} epochs on {TRAINING2_DATA_PATH}",
        "epochs": epochs,
    }


@router.get("/calibration/status")
async def get_calibration_status():
    """Get current calibration progress."""
    return _calibration_status


async def _run_calibration(epochs: int):
    """Background task: run the full calibration pipeline."""
    global _calibration_status

    async with _calibration_lock:
        _calibration_status = {
            "status": "running",
            "percent": 0,
            "message": "Calibration starting…",
        }

        async def progress_cb(update: dict):
            _calibration_status.update({
                "status": "running",
                "percent": update.get("percent", 0),
                "message": update.get("message", ""),
            })

        try:
            from training.training_pipeline import run_training
            result = await run_training(
                training_data_dir=TRAINING2_DATA_PATH,
                epochs=epochs,
                use_gpu=True,
                progress_callback=progress_cb,
            )

            if result.get("status") == "complete":
                _calibration_status = {
                    "status": "complete",
                    "percent": 100,
                    "message": (
                        f"Calibration complete! "
                        f"Best accuracy: {result.get('summary', {}).get('best_accuracy', 0):.1%}"
                    ),
                    "result": result.get("summary", {}),
                }
            else:
                _calibration_status = {
                    "status": "failed",
                    "percent": 0,
                    "message": f"Calibration failed: {result.get('error', 'Unknown')}",
                }

        except Exception as exc:
            logger.error(f"Calibration crashed: {exc}", exc_info=True)
            _calibration_status = {
                "status": "failed",
                "percent": 0,
                "message": f"Calibration crashed: {exc}",
            }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL VERSION
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/model/version")
async def get_model_version():
    """Current fine-tuned model version (updated after each auto-train run)."""
    from services.pipeline import get_model_version
    return get_model_version()
