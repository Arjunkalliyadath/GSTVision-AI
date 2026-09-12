# paddle_ocr.py  (v5.0 — SUBPROCESS WORKER + GPU + TIMEOUT)
# ─────────────────────────────────────────────────────────────────────────────
# ROOT-CAUSE FIX FOR PERMANENT HANG:
#   PaddleOCR's GPU model loading and predict() calls can hang indefinitely
#   on Windows when CUDA drivers are partially initialised or when the GPU is
#   busy.  Running PaddleOCR in a SUBPROCESS (paddle_worker.py) means:
#     • A hang in the worker NEVER blocks the FastAPI event loop
#     • We can enforce a per-call timeout and kill/restart the worker
#     • PaddlePaddle and PyTorch GPU DLLs are isolated in separate processes
#       (prevents DLL conflicts between PaddlePaddle + PyTorch on Windows)
#
# ARCHITECTURE (v5.0):
#   PaddleOCRService → spawns paddle_worker.py subprocess
#   Communication: newline-delimited JSON via stdin/stdout
#   Timeout: 120s per OCR call (large images on GPU can take ~30-60s)
#   Restart: worker is restarted automatically if it crashes
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import subprocess
import threading
import atexit
import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import asyncio

logger = logging.getLogger("gst2_fastapi.paddle_ocr")

OCR_TIMEOUT_SEC    = 120
MAX_WORKER_RESTARTS = 2


class PaddleOCRService:
    """PaddleOCR wrapper — subprocess-based, no hang, GPU-enforced."""

    def __init__(self, use_gpu: bool = True, gpu_id: int = 0):
        self.use_gpu   = use_gpu
        self.gpu_id    = gpu_id
        self._proc     = None
        self._lock     = threading.Lock()
        self._restarts = 0
        self._available = False
        logger.info(f"PaddleOCRService v5.0 — starting worker (GPU={use_gpu})…")
        self._start_worker()
        atexit.register(self._cleanup)

    # ── Worker lifecycle ───────────────────────────────────────────────────────

    def _start_worker(self):
        worker_path = str(Path(__file__).parent / "paddle_worker.py")
        env  = os.environ.copy()
        args = [sys.executable, worker_path]
        if not self.use_gpu:
            args.append("--cpu")
        try:
            self._proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                bufsize=1,
            )
            ready_line = self._read_line_timeout(timeout=90)
            if ready_line is None:
                stderr_out = ""
                try:
                    self._proc.kill()
                    _, stderr_out = self._proc.communicate(timeout=5)
                except Exception:
                    pass
                raise RuntimeError(
                    f"Worker did not send ready signal within 90s. stderr: {stderr_out[:300]}"
                )
            status = json.loads(ready_line)
            if "error" in status:
                raise RuntimeError(f"Worker error: {status['error']}")
            self._available = True
            logger.info(
                f"PaddleOCR worker ready "
                f"(GPU={status.get('gpu')}, api={status.get('api')}, pid={self._proc.pid})"
            )
        except Exception as exc:
            logger.error(f"PaddleOCR worker start failed: {exc}")
            self._available = False
            if self._proc:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._proc = None

    def _read_line_timeout(self, timeout: float) -> Optional[str]:
        result      = [None]
        exc_holder  = [None]
        def _reader():
            try:
                result[0] = self._proc.stdout.readline()
            except Exception as e:
                exc_holder[0] = e
        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            return None
        if exc_holder[0]:
            raise exc_holder[0]
        line = result[0]
        return line.strip() if line else None

    def _restart_worker(self):
        self._restarts += 1
        logger.warning(f"Restarting PaddleOCR worker (attempt {self._restarts}/{MAX_WORKER_RESTARTS})…")
        try:
            if self._proc:
                self._proc.kill()
                self._proc.communicate(timeout=5)
        except Exception:
            pass
        self._proc     = None
        self._available = False
        time.sleep(2)
        self._start_worker()

    def _cleanup(self):
        if self._proc:
            try:
                self._proc.stdin.write(json.dumps({"action": "quit"}) + "\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    # ── OCR call ──────────────────────────────────────────────────────────────

    def _send_ocr_request(self, image_path: str) -> Optional[Dict]:
        if not self._available or self._proc is None:
            return None
        with self._lock:
            try:
                self._proc.stdin.write(json.dumps({"image_path": str(image_path)}) + "\n")
                self._proc.stdin.flush()
                resp_line = self._read_line_timeout(timeout=OCR_TIMEOUT_SEC)
                if resp_line is None:
                    logger.error(
                        f"PaddleOCR worker timed out ({OCR_TIMEOUT_SEC}s) for "
                        f"{Path(image_path).name}"
                    )
                    if self._restarts < MAX_WORKER_RESTARTS:
                        self._restart_worker()
                    else:
                        logger.error("Max restarts reached — PaddleOCR disabled.")
                        self._available = False
                    return None
                return json.loads(resp_line)
            except (BrokenPipeError, OSError) as exc:
                logger.warning(f"Worker pipe broken: {exc}")
                if self._restarts < MAX_WORKER_RESTARTS:
                    self._restart_worker()
                else:
                    self._available = False
                return None
            except json.JSONDecodeError as exc:
                logger.warning(f"Worker returned invalid JSON: {exc}")
                return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        _EMPTY = {
            "lines": [], "full_text": "", "avg_confidence": 0.0,
            "line_count": 0, "image_path": image_path,
        }
        if not self._available:
            return {**_EMPTY, "source": "paddle_unavailable"}
        resp = self._send_ocr_request(image_path)
        if resp is None:
            return {**_EMPTY, "source": "paddle_timeout"}
        if "error" in resp:
            logger.warning(f"PaddleOCR error: {resp['error']}")
            return {**_EMPTY, "source": "paddle_error"}

        lines: List[Dict] = []
        confs: List[float] = []
        for item in resp.get("results", []):
            text = str(item.get("text", "")).strip()
            conf = float(item.get("confidence", 0.9))
            if text:
                lines.append({"text": text, "confidence": conf, "bbox": item.get("bbox")})
                confs.append(conf)

        avg_conf  = sum(confs) / len(confs) if confs else 0.0
        full_text = "\n".join(l["text"] for l in lines)
        return {
            "lines":          lines,
            "full_text":      full_text,
            "avg_confidence": avg_conf,
            "line_count":     len(lines),
            "image_path":     image_path,
            "source":         "paddleocr",
        }

    def extract_best_from_variants(self, image_paths: List[str]) -> Dict[str, Any]:
        best_result: Optional[Dict] = None
        best_score = -1.0
        for path in image_paths:
            try:
                res   = self.extract_text(path)
                score = res["avg_confidence"] * (res["line_count"] ** 0.5)
                logger.info(
                    f"OCR {Path(path).name}: lines={res['line_count']} "
                    f"conf={res['avg_confidence']:.2f} score={score:.2f}"
                )
                if score > best_score:
                    best_score  = score
                    best_result = res
            except Exception as exc:
                logger.warning(f"OCR attempt failed for {path}: {exc}")
        return best_result or {
            "lines": [], "full_text": "", "avg_confidence": 0.0,
            "line_count": 0, "image_path": image_paths[0] if image_paths else "",
        }

    async def extract_from_image_async(self, image_path: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.extract_text, image_path)

    async def extract_best_from_variants_async(self, image_paths: List[str]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.extract_best_from_variants, image_paths)

    async def extract_from_multiple_images_async(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        tasks = [self.extract_from_image_async(p) for p in image_paths]
        return list(await asyncio.gather(*tasks))

    def extract_from_multiple_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        return [self.extract_text(p) for p in image_paths]

    @property
    def available(self) -> bool:
        return self._available


# ── Singleton ──────────────────────────────────────────────────────────────────
_ocr_instance: Optional[PaddleOCRService] = None


def get_paddle_ocr(use_gpu: bool = True, gpu_id: int = 0) -> PaddleOCRService:
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCRService(use_gpu=use_gpu, gpu_id=gpu_id)
    return _ocr_instance