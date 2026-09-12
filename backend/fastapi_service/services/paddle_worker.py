# paddle_worker.py — PaddleOCR Subprocess Worker (v1.0)
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Runs PaddleOCR in an ISOLATED SUBPROCESS so a GPU hang never blocks
#   the FastAPI event loop or the main process.
#
# PROTOCOL:
#   Parent → Worker: newline-delimited JSON  {"image_path": "..."}
#   Worker → Parent: newline-delimited JSON  {"results": [...]} or {"error":"..."}
#   Init:   Worker writes {"status":"ready","gpu":true/false} on startup.
#
# This mirrors the EasyOCR worker pattern that is already proven stable.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import json
import os
import site
from pathlib import Path

# ── Inject CUDA DLL paths BEFORE importing paddle (Windows only) ──────────────
if sys.platform == "win32":
    for sp in site.getsitepackages():
        nvidia_dir = Path(sp) / "nvidia"
        if nvidia_dir.exists():
            for bin_dir in nvidia_dir.rglob("bin"):
                os.environ["PATH"] = f"{bin_dir};{os.environ['PATH']}"
                try:
                    os.add_dll_directory(str(bin_dir))
                except Exception:
                    pass

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# Suppress paddle's verbose output so only our JSON goes to stdout
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_call_stack_level"] = "0"

import logging
logging.disable(logging.CRITICAL)  # Silence all logging in worker subprocess


def _init_paddle(use_gpu: bool):
    """Try to load PaddleOCR, return (ocr_instance, api_version, actual_gpu)."""
    from paddleocr import PaddleOCR
    import paddle

    actual_gpu = use_gpu and paddle.device.cuda.device_count() > 0

    # Try v3.x API first
    for kwargs in [
        {"device": "gpu:0" if actual_gpu else "cpu"},
        {"lang": "en", "device": "gpu:0" if actual_gpu else "cpu"},
    ]:
        try:
            ocr = PaddleOCR(**kwargs)
            return ocr, "v3", actual_gpu
        except TypeError:
            continue
        except Exception:
            continue

    # Fall back to v2.x API
    try:
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=actual_gpu,
            gpu_id=0,
            show_log=False,
        )
        return ocr, "v2", actual_gpu
    except Exception as e:
        raise RuntimeError(f"All PaddleOCR init attempts failed: {e}")


def _run_ocr(ocr, api_ver: str, image_path: str):
    """Run OCR and return list of {bbox, text, confidence} dicts."""
    results = []

    if api_ver == "v3":
        try:
            raw = ocr.predict(input=image_path)
        except Exception:
            raw = ocr.ocr(image_path)
    else:
        raw = ocr.ocr(image_path, cls=True)

    if raw is None:
        return results

    for page in raw:
        if page is None:
            continue

        # v3 object-style result
        if hasattr(page, "rec_texts"):
            texts  = page.rec_texts  or []
            scores = page.rec_scores or []
            boxes  = page.boxes      or [None] * len(texts)
            for text, score, box in zip(texts, scores, boxes):
                if text and text.strip():
                    bbox_serializable = None
                    if box is not None:
                        try:
                            bbox_serializable = [[float(x), float(y)] for x, y in box]
                        except Exception:
                            pass
                    results.append({
                        "bbox": bbox_serializable,
                        "text": str(text).strip(),
                        "confidence": float(score) if score is not None else 0.9,
                    })
            continue

        # v2/v3 list-style result
        if isinstance(page, list):
            for item in page:
                try:
                    bbox = item[0]
                    text_info = item[1]
                    text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
                    conf = float(text_info[1]) if isinstance(text_info, (list, tuple)) else 0.9
                    if text and text.strip():
                        bbox_serializable = None
                        if bbox is not None:
                            try:
                                bbox_serializable = [[float(x), float(y)] for x, y in bbox]
                            except Exception:
                                pass
                        results.append({
                            "bbox": bbox_serializable,
                            "text": str(text).strip(),
                            "confidence": conf,
                        })
                except Exception:
                    continue

    return results


def main():
    use_gpu = "--cpu" not in sys.argv

    try:
        ocr, api_ver, actual_gpu = _init_paddle(use_gpu)
    except Exception as e:
        sys.stdout.write(json.dumps({"error": f"PaddleOCR init failed: {e}"}) + "\n")
        sys.stdout.flush()
        sys.exit(1)

    sys.stdout.write(json.dumps({"status": "ready", "gpu": actual_gpu, "api": api_ver}) + "\n")
    sys.stdout.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        try:
            req = json.loads(line)
        except Exception:
            continue

        if req.get("action") == "quit":
            break

        path = req.get("image_path")
        if not path:
            sys.stdout.write(json.dumps({"error": "No image_path provided"}) + "\n")
            sys.stdout.flush()
            continue

        try:
            results = _run_ocr(ocr, api_ver, path)
            sys.stdout.write(json.dumps({"results": results}) + "\n")
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")

        sys.stdout.flush()


if __name__ == "__main__":
    main()