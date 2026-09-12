# easyocr_service.py — EasyOCR Engine (v2.0 — FIXED \\n BUG + GPU)
# ─────────────────────────────────────────────────────────────────────────────
# FIX v2.0 (CRITICAL):
#   Previous version used "\\n" (literal backslash-n) in stdin writes instead
#   of "\n" (actual newline).  This broke the JSON line protocol — the worker's
#   readline() never saw a newline terminator so every OCR call hung forever.
#   Fixed in 3 locations: request send, quit command, full_text join.
#
# Interface matches PaddleOCR/Tesseract for drop-in use:
#   { lines, full_text, avg_confidence, line_count, image_path }
# ─────────────────────────────────────────────────────────────────────────────

import logging
import json
import subprocess
import sys
import os
import atexit
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.easyocr")


class EasyOCRService:
    """EasyOCR wrapper — runs in a daemon subprocess to isolate PyTorch GPU DLLs."""

    def __init__(self, use_gpu: bool = True, languages: list = None):
        self._available = False
        self._use_gpu   = use_gpu
        self._languages = languages or ["en"]
        self._proc      = None
        self._load()
        atexit.register(self._cleanup)

    def _load(self):
        try:
            # Strip Paddle nvidia DLLs from PATH so PyTorch finds its own CUDA
            env = os.environ.copy()
            if sys.platform == "win32" and "PATH" in env:
                paths = env["PATH"].split(";")
                clean = [p for p in paths if "nvidia" not in p.lower()]
                env["PATH"] = ";".join(clean)

            worker_script = str(Path(__file__).parent / "easyocr_worker.py")
            self._proc = subprocess.Popen(
                [sys.executable, worker_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,   # capture stderr; prevents console bleed
                text=True,
                env=env,
                bufsize=1,
            )

            # Read init status — worker writes ONE JSON line on startup
            line = self._proc.stdout.readline()
            if not line:
                raise Exception("Worker crashed immediately (no output)")

            status = json.loads(line)
            if "error" in status:
                raise Exception(status["error"])

            self._available = True
            device = "GPU" if status.get("gpu") else "CPU"
            logger.info(f"EasyOCR Worker loaded ({device})")

        except Exception as exc:
            logger.warning(f"EasyOCR worker load failed: {exc}")
            self._available = False
            self._cleanup()

    def _cleanup(self):
        if self._proc:
            try:
                # FIX: was "\\n" (backslash-n), now "\n" (actual newline)
                self._proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    @property
    def available(self) -> bool:
        return self._available

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        _EMPTY = {
            "lines": [], "full_text": "", "avg_confidence": 0.0,
            "line_count": 0, "image_path": image_path,
            "source": "easyocr_unavailable",
        }

        if not self._available or not self._proc:
            return _EMPTY

        try:
            req = {"image_path": str(image_path)}
            # FIX: was "\\n" (backslash-n) — now "\n" (actual newline)
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()

            line = self._proc.stdout.readline()
            if not line:
                self._available = False
                return {**_EMPTY, "source": "easyocr_crashed"}

            res = json.loads(line)
            if "error" in res:
                logger.warning(f"EasyOCR error: {res['error']}")
                return {**_EMPTY, "source": "easyocr_error"}

            results = res.get("results", [])
            if not results:
                return {**_EMPTY, "source": "easyocr_empty"}

            # Group word-level detections into text lines by Y-coordinate
            raw_items = []
            for item in results:
                bbox = item["bbox"]
                text = item["text"].strip()
                if not text:
                    continue
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                raw_items.append({
                    "text":       text,
                    "confidence": item["confidence"],
                    "bbox":       bbox,
                    "y":          y_center,
                    "x":          x_center,
                })

            if not raw_items:
                return {**_EMPTY, "source": "easyocr_empty"}

            raw_items.sort(key=lambda it: it["y"])
            current_line_items = [raw_items[0]]
            lines: List[Dict] = []
            confs: List[float] = []

            for item in raw_items[1:]:
                if abs(item["y"] - current_line_items[0]["y"]) < 20:
                    current_line_items.append(item)
                else:
                    current_line_items.sort(key=lambda it: it["x"])
                    merged_text = " ".join(it["text"] for it in current_line_items)
                    merged_conf = sum(it["confidence"] for it in current_line_items) / len(current_line_items)
                    lines.append({
                        "text":       merged_text,
                        "confidence": merged_conf,
                        "bbox":       self._merge_bboxes([it["bbox"] for it in current_line_items]),
                    })
                    confs.append(merged_conf)
                    current_line_items = [item]

            if current_line_items:
                current_line_items.sort(key=lambda it: it["x"])
                merged_text = " ".join(it["text"] for it in current_line_items)
                merged_conf = sum(it["confidence"] for it in current_line_items) / len(current_line_items)
                lines.append({
                    "text":       merged_text,
                    "confidence": merged_conf,
                    "bbox":       self._merge_bboxes([it["bbox"] for it in current_line_items]),
                })
                confs.append(merged_conf)

            return {
                "lines":          lines,
                # FIX: was "\\n".join(...) — now "\n".join(...)
                "full_text":      "\n".join(l["text"] for l in lines),
                "avg_confidence": sum(confs) / len(confs) if confs else 0.0,
                "line_count":     len(lines),
                "image_path":     image_path,
                "source":         "easyocr",
            }

        except Exception as exc:
            logger.warning(f"EasyOCR extraction failed for {Path(image_path).name}: {exc}")
            return {**_EMPTY, "source": "easyocr_error"}

    @staticmethod
    def _merge_bboxes(bboxes: list) -> list:
        if not bboxes:
            return [[0, 0], [0, 0], [0, 0], [0, 0]]
        all_x = [p[0] for b in bboxes for p in b]
        all_y = [p[1] for b in bboxes for p in b]
        return [
            [min(all_x), min(all_y)],
            [max(all_x), min(all_y)],
            [max(all_x), max(all_y)],
            [min(all_x), max(all_y)],
        ]


_easyocr_instance: Optional[EasyOCRService] = None


def get_easyocr(use_gpu: bool = True) -> EasyOCRService:
    global _easyocr_instance
    if _easyocr_instance is None:
        _easyocr_instance = EasyOCRService(use_gpu=use_gpu)
    return _easyocr_instance