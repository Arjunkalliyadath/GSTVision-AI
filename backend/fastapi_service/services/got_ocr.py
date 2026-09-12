# got_ocr.py — GOT-OCR 2.0 Fallback Service  (v2.0 — FIXED)
# ─────────────────────────────────────────────────────────────────────────────
# ROOT-CAUSE FIX (v2.0):
#   The original code called AutoModel.from_pretrained("stepfun-ai/GOT-OCR2_0")
#   which ALWAYS tries to download the model from HuggingFace even when the
#   local ml_models/got_ocr folder is empty.  This caused:
#     • Long hangs (HuggingFace download attempt)
#     • Auth errors if HuggingFace is blocked
#     • Pipeline failure even though GOT-OCR is OPTIONAL
#
#   FIX: GOTOCRService now checks the local model path FIRST.
#   If the folder is missing or empty → immediately mark as unavailable.
#   No network call is ever made.  The pipeline continues using PaddleOCR
#   results even at low confidence.
#
#   HOW TO ENABLE GOT-OCR (optional, gives better OCR on complex images):
#     1. huggingface-cli download stepfun-ai/GOT-OCR2_0 \
#            --local-dir ml_models/got_ocr
#     2. Set GOT_OCR_MODEL_PATH=<path> in your .env file
#        (defaults to ../../ml_models/got_ocr)
# ─────────────────────────────────────────────────────────────────────────────

import os
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.got_ocr")

CONFIDENCE_THRESHOLD = 0.80   # Use fallback if PaddleOCR avg confidence < 80%

# Where to look for the local model — can be overridden in .env
GOT_OCR_MODEL_PATH = os.getenv("GOT_OCR_MODEL_PATH", "../../ml_models/got_ocr")


def _model_is_available(model_path: str) -> bool:
    """
    Return True only if the local model directory exists AND contains files.
    This prevents any HuggingFace download attempts.
    """
    p = Path(model_path)
    if not p.exists() or not p.is_dir():
        return False
    # Directory must have at least one non-hidden file (model weights etc.)
    files = [f for f in p.iterdir() if not f.name.startswith(".")]
    return len(files) > 0


class GOTOCRService:
    """
    GOT-OCR 2.0 — optional fallback for low-confidence PaddleOCR pages.

    If the local model folder is empty or missing, this service immediately
    marks itself as unavailable and all calls return empty results.
    No network calls are ever made.
    """

    def __init__(self, model_path: str = GOT_OCR_MODEL_PATH):
        self.model_path = str(model_path)
        self.model      = None
        self.tokenizer  = None
        self._loaded    = False
        self._unavailable = False

        # ── Check availability IMMEDIATELY at construction time ──────────
        # This is the KEY FIX: we never try to load if model isn't local.
        if not _model_is_available(self.model_path):
            self._unavailable = True
            logger.info(
                f"GOT-OCR 2.0 not available (model folder empty or missing: "
                f"{self.model_path}). "
                "GOT-OCR fallback disabled — PaddleOCR results will be used as-is. "
                "To enable GOT-OCR: download stepfun-ai/GOT-OCR2_0 into that folder."
            )
        else:
            logger.info(f"GOT-OCR 2.0 model found at {self.model_path}. Will load on first use.")

    def load(self):
        """Lazy-load the model. Does nothing if unavailable or already loaded."""
        if self._loaded or self._unavailable:
            return

        logger.info(f"Loading GOT-OCR 2.0 from local path: {self.model_path} ...")
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,   # ← NEVER fetch from HuggingFace
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,   # ← NEVER fetch from HuggingFace
                low_cpu_mem_usage=True,
                device_map=device,
                use_safetensors=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            self.model = self.model.eval()
            if device == "cuda":
                self.model = self.model.cuda()
            self._loaded = True
            logger.info(f"GOT-OCR 2.0 loaded successfully from local files on {device}!")

        except Exception as e:
            logger.error(
                f"Failed to load GOT-OCR 2.0 from {self.model_path}: {e}. "
                "Marking as unavailable — PaddleOCR results will be used."
            )
            self._unavailable = True

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text using GOT-OCR 2.0.
        Returns empty result (not an error) if model is unavailable.
        """
        _EMPTY = {
            "lines":           [],
            "full_text":       "",
            "avg_confidence":  0.0,
            "line_count":      0,
            "source":          "got_ocr_unavailable",
        }

        # Fast path: unavailable → return empty immediately, no loading attempt
        if self._unavailable:
            return _EMPTY

        # Lazy load
        if not self._loaded:
            self.load()

        if not self._loaded:
            return _EMPTY

        try:
            result = self.model.chat(
                self.tokenizer,
                image_path,
                ocr_type="ocr",
            )
            lines_text = [l.strip() for l in result.split("\n") if l.strip()]
            lines      = [{"text": l, "confidence": 0.95, "bbox": None} for l in lines_text]
            return {
                "lines":          lines,
                "full_text":      result,
                "avg_confidence": 0.95,
                "line_count":     len(lines),
                "source":         "got_ocr",
            }

        except Exception as e:
            logger.error(f"GOT-OCR extraction failed for {image_path}: {e}")
            return {
                "lines":          [],
                "full_text":      "",
                "avg_confidence": 0.0,
                "line_count":     0,
                "source":         "got_ocr_failed",
            }


# ── Singleton ─────────────────────────────────────────────────────────────────
_got_ocr_instance = None


def get_got_ocr() -> GOTOCRService:
    global _got_ocr_instance
    if _got_ocr_instance is None:
        _got_ocr_instance = GOTOCRService()
    return _got_ocr_instance


def should_use_fallback(paddle_result: Dict) -> bool:
    """
    Returns True if PaddleOCR confidence is low enough to try GOT-OCR.
    Also returns False immediately if GOT-OCR is not available —
    avoids even creating the service for nothing.
    """
    service = get_got_ocr()
    if service._unavailable:
        return False   # ← KEY: skip fallback entirely when model not downloaded
    return paddle_result.get("avg_confidence", 1.0) < CONFIDENCE_THRESHOLD