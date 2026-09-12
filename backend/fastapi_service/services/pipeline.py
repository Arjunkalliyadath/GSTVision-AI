# pipeline.py  (v12.0 — DUAL PIPELINE + TRIPLE OCR + EASYOCR)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT'S NEW IN v12.0
# ───────────────────
#
#  1. DUAL PIPELINE ARCHITECTURE:
#     PATH 1 — Digital PDF → pdfplumber direct (UNCHANGED, highest accuracy)
#     PATH 2 — Image/Photo PDF → NEW dedicated image_pipeline.py
#              with triple-OCR (PaddleOCR + Tesseract + EasyOCR)
#
#  2. FILE CLASSIFIER:
#     file_classifier.py detects digital vs photo PDF vs image at upload.
#     Routes each file to the optimal pipeline automatically.
#
#  3. GOT-OCR REPLACED BY EASYOCR:
#     GOT-OCR removed (slow, 4GB, CUDA-only, never usable).
#     EasyOCR added (~100MB, CPU+GPU, excellent on printed tables).
#
#  4. TRIPLE OCR on image pipeline:
#     PaddleOCR + Tesseract + EasyOCR per variant.
#     Best-of-3 scoring per variant, best variant selected overall.
#
#  5. All previous features preserved:
#     • PATH 1 digital PDF (ZERO changes)
#     • Auto-rotation, adaptive preprocessing
#     • E-Way Bill + GST 2A support
#     • LayoutLMv3 optional post-processing
#     • Auto-training trigger
#
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import asyncio
import logging
import time
import json
import math
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv("../../.env")

logger = logging.getLogger("gst2_fastapi.pipeline")

USE_GPU                = os.getenv("USE_GPU",               "True").lower() == "true"
LAYOUTLMV3_MODEL_PATH  = os.getenv("LAYOUTLMV3_MODEL_PATH", "../../ml_models/layoutlmv3")
LLAMA_MODEL_PATH       = os.getenv("LLAMA_MODEL_PATH",      "../../ml_models/llama/llama.gguf")
GOT_OCR_MODEL_PATH     = os.getenv("GOT_OCR_MODEL_PATH",    "../../ml_models/got_ocr")
MEDIA_ROOT             = os.getenv("MEDIA_ROOT",            "../../media")
TRAINING_DATA_PATH     = os.getenv("TRAINING_DATA_PATH",    "../../training_data")

MIN_DIRECT_RECORDS     = 3
IMAGE_EXTENSIONS       = {".jpg", ".jpeg", ".png", ".heic", ".heif",
                           ".bmp", ".tiff", ".tif", ".webp"}

AUTO_TRAIN_EVERY_N = int(os.getenv("AUTO_TRAIN_EVERY_N", "5"))
AUTO_TRAIN_EPOCHS  = int(os.getenv("AUTO_TRAIN_EPOCHS",  "10"))
_training_lock     = asyncio.Lock()


def _model_dir_has_files(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and any(f for f in p.iterdir() if not f.name.startswith("."))


def _ocr_score(result: Dict) -> float:
    """
    Composite variant-selection score.
    score = avg_confidence × √(line_count)
    """
    conf  = result.get("avg_confidence", 0.0)
    lines = result.get("line_count", 0)
    return conf * math.sqrt(max(lines, 1))


# ─────────────────────────────────────────────────────────────────────────────
# DUAL OCR HELPER — Run both PaddleOCR and Tesseract on one image
# ─────────────────────────────────────────────────────────────────────────────

def _dual_ocr_extract(
    image_path: str,
    ocr_service,
    tesseract_service,
) -> Dict[str, Any]:
    """
    Run both PaddleOCR and Tesseract on a single image.
    Return the result with the higher composite score.
    """
    best_result = None
    best_score  = -1.0

    # PaddleOCR
    try:
        paddle_result = ocr_service.extract_text(image_path)
        paddle_score  = _ocr_score(paddle_result)
        paddle_result["_ocr_engine"] = "paddleocr"
        if paddle_score > best_score:
            best_score  = paddle_score
            best_result = paddle_result
        logger.debug(
            f"  PaddleOCR: lines={paddle_result['line_count']} "
            f"conf={paddle_result['avg_confidence']:.2f} score={paddle_score:.2f}"
        )
    except Exception as exc:
        logger.warning(f"  PaddleOCR failed: {exc}")

    # Tesseract
    if tesseract_service and tesseract_service.available:
        try:
            tess_result = tesseract_service.extract_text_with_boxes(image_path)
            tess_score  = _ocr_score(tess_result)
            tess_result["_ocr_engine"] = "tesseract"
            if tess_score > best_score:
                best_score  = tess_score
                best_result = tess_result
            logger.debug(
                f"  Tesseract: lines={tess_result['line_count']} "
                f"conf={tess_result['avg_confidence']:.2f} score={tess_score:.2f}"
            )
        except Exception as exc:
            logger.warning(f"  Tesseract failed: {exc}")

    if best_result is None:
        best_result = {
            "lines": [], "full_text": "", "avg_confidence": 0.0,
            "line_count": 0, "source": "ocr_failed",
        }

    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPER — preprocessing + multi-variant dual-OCR for one image page
# ─────────────────────────────────────────────────────────────────────────────

async def _ocr_with_preprocessing(
    page_num:     int,
    total_pages:  int,
    orig_path:    str,
    temp_dir:     Path,
    ocr_service,
    tesseract_service,
    preprocessor,
    job_id:       str,
    progress_fn,
) -> Dict[str, Any]:
    """
    Run the full preprocessing + best-variant dual-OCR pipeline on a single image.

    Steps:
      1. Call image_preprocessor.preprocess_variants() → up to 7 enhanced images
         (includes auto-rotation + adaptive intensity)
      2. Dual-OCR (Paddle + Tesseract) every variant; keep highest _ocr_score()
      3. Tag result with page_number and image_path
    """
    page_pct = 20 + int(page_num * 40 / max(total_pages, 1))
    await progress_fn(
        "preprocessing",
        page_pct,
        f"Page {page_num+1}/{total_pages}: auto-rotate → preprocess → dual-OCR…",
    )

    # Step 1 — Generate enhanced variants (includes auto-rotation)
    variants = [orig_path]
    if preprocessor:
        try:
            extra = await asyncio.to_thread(
                preprocessor.preprocess_variants,
                orig_path,
                str(temp_dir / f"pre_{page_num}"),
            )
            processed = [v for v in extra if v != orig_path]
            variants  = processed + [orig_path]
            logger.info(
                f"[{job_id}] p{page_num+1}: {len(variants)} variants "
                f"({len(processed)} processed + original)"
            )
        except Exception as exc:
            logger.warning(f"[{job_id}] Variant generation p{page_num+1}: {exc}")

    # Step 2 — Dual-OCR all variants, keep the highest-scoring result
    best_result = None
    best_score  = -1.0

    for vi, vpath in enumerate(variants):
        try:
            result = await asyncio.to_thread(
                _dual_ocr_extract, vpath, ocr_service, tesseract_service
            )
            score = _ocr_score(result)
            engine = result.get("_ocr_engine", "unknown")
            logger.info(
                f"[{job_id}] p{page_num+1} v{vi+1}/{len(variants)} "
                f"({Path(vpath).stem[-15:]}): "
                f"lines={result['line_count']} "
                f"conf={result['avg_confidence']:.2f} "
                f"score={score:.2f} engine={engine}"
            )
            if score > best_score:
                best_score  = score
                best_result = result
        except Exception as exc:
            logger.warning(f"[{job_id}] OCR variant {vi+1} failed: {exc}")
        await asyncio.sleep(0)

    if best_result is None:
        best_result = {
            "lines": [], "full_text": "", "avg_confidence": 0.0,
            "line_count": 0, "source": "ocr_failed",
        }

    best_result["page_number"] = page_num + 1
    best_result["image_path"]  = orig_path
    logger.info(
        f"[{job_id}] Page {page_num+1} final: "
        f"lines={best_result['line_count']} "
        f"conf={best_result['avg_confidence']:.2f} "
        f"score={best_score:.2f} "
        f"engine={best_result.get('_ocr_engine', 'unknown')}"
    )
    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def process_file(
    file_path: str,
    job_id:    str,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Full extraction pipeline.  Always returns a result dict — never raises.

    Pipeline routing
    ────────────────
    PATH 1  — Digital PDF (GSTN portal text-layer PDFs)
                pdfplumber → parse_gst2a_pdf → ≥ MIN_DIRECT_RECORDS  → Excel
                Skips OCR entirely.  Highest accuracy and speed.

    PATH 2a — Image file (.jpg / .png / .jpeg / .heic / etc.)
                → Auto-rotate (90°/180°/270°)
                → Adaptive preprocessing (2/4/7 variants based on quality)
                → Dual OCR (PaddleOCR + Tesseract) per variant
                → Best-score selection
                → Table extractor + LayoutLMv3 + image_gst_parser
                → Excel

    PATH 2b — Scanned / image PDF (PDF where direct extraction yielded 0–2 records)
                SAME pipeline as Path 2a.
    """
    start_time = time.time()
    file_path  = Path(file_path)
    file_name  = file_path.name
    ext        = file_path.suffix.lower()
    output_dir = Path(MEDIA_ROOT) / "outputs"
    temp_dir   = Path(MEDIA_ROOT) / "temp" / job_id

    logger.info(f"[{job_id}] ══ Pipeline v12 start: {file_name} (ext={ext})")

    async def progress(step: str, pct: int, msg: str):
        logger.info(f"[{job_id}] {pct:3d}% [{step}] {msg}")
        if progress_callback:
            try:
                await progress_callback({"step": step, "percent": pct, "message": msg})
            except Exception:
                pass
        await asyncio.sleep(0)

    try:
        # ════════════════════════════════════════════════════════════════════
        # STEP 0 — CLASSIFY FILE TYPE
        # ════════════════════════════════════════════════════════════════════
        from .file_classifier import classify_file
        file_type = await asyncio.to_thread(classify_file, str(file_path))
        logger.info(f"[{job_id}] File classified as: {file_type}")

        # ════════════════════════════════════════════════════════════════════
        # PATH 1 — Direct PDF text extraction (GSTN portal digital PDFs)
        # ════════════════════════════════════════════════════════════════════
        if file_type == "digital_pdf":
            await progress("detecting", 5, "Digital PDF detected — direct extraction…")
            direct_records = []
            try:
                from .gst2a_pdf_parser import parse_gst2a_pdf, records_to_excel_format
                direct_records = await asyncio.to_thread(parse_gst2a_pdf, str(file_path))
            except Exception as exc:
                logger.warning(
                    f"[{job_id}] Direct PDF parser error: {exc}. "
                    "Falling back to image pipeline."
                )

            if len(direct_records) >= MIN_DIRECT_RECORDS:
                await progress(
                    "pdf_parse", 40,
                    f"Digital PDF: {len(direct_records)} records extracted directly."
                )
                excel_records = await asyncio.to_thread(records_to_excel_format, direct_records)

                if Path(LLAMA_MODEL_PATH).exists():
                    await progress("validating", 75, "LLaMA 3 validation…")
                    try:
                        from .llm_validator import get_llama_validator
                        validator     = get_llama_validator(LLAMA_MODEL_PATH)
                        sample_text   = _build_sample_text(direct_records[:10])
                        excel_records = await asyncio.to_thread(
                            _validate_records_sync, validator, excel_records, sample_text)
                    except Exception as exc:
                        logger.info(f"[{job_id}] LLaMA skipped: {exc}")

                await progress("excel", 90, "Generating Excel…")
                excel_path = await asyncio.to_thread(
                    _generate_excel_sync, excel_records, output_dir, file_name, job_id)
                await asyncio.to_thread(
                    _save_training_data, str(file_path), direct_records, [], job_id)
                t = time.time() - start_time
                await progress("complete", 100,
                    f"Done! {len(excel_records)} records in {t:.1f}s.")
                asyncio.create_task(_maybe_auto_train())
                return _ok(
                    excel_path, excel_records, t, 0,
                    "digital_pdf_extraction", 0.98, job_id
                )

            else:
                logger.info(
                    f"[{job_id}] Digital PDF had only {len(direct_records)} records. "
                    "Reclassifying as photo_pdf → routing to image pipeline."
                )
                file_type = "photo_pdf"  # Fall through to PATH 2

        # ════════════════════════════════════════════════════════════════════
        # PATH 2 — Dedicated Image Pipeline (images + photo/scanned PDFs)
        # ════════════════════════════════════════════════════════════════════
        if file_type in ("image", "photo_pdf"):
            pipeline_label = "image_file_ocr" if file_type == "image" else "scanned_pdf_ocr"
            await progress(
                "image_pipeline", 10,
                f"Routing to dedicated image pipeline [{pipeline_label}]…"
            )

            from .image_pipeline import process_image_file
            img_result = await process_image_file(
                file_path=str(file_path),
                job_id=job_id,
                output_dir=output_dir,
                temp_dir=temp_dir,
                use_gpu=USE_GPU,
                progress_callback=progress_callback,
            )

            records = img_result.get("records", [])
            ocr_results = img_result.get("ocr_results", [])
            image_paths = img_result.get("image_paths", [])

            # LLaMA validation (optional)
            if records and Path(LLAMA_MODEL_PATH).exists():
                await progress("validating", 85, "LLaMA validation…")
                try:
                    from .llm_validator import get_llama_validator
                    validator = get_llama_validator(LLAMA_MODEL_PATH)
                    full_text = " ".join(r.get("full_text", "") for r in ocr_results)
                    records = await asyncio.to_thread(
                        _validate_records_sync, validator, records, full_text)
                except Exception as exc:
                    logger.info(f"[{job_id}] LLaMA skipped: {exc}")

            # Generate Excel
            await progress("excel", 92, "Generating Excel file…")
            excel_path = await asyncio.to_thread(
                _generate_excel_sync, records, output_dir, file_name, job_id)
            await asyncio.to_thread(
                _save_training_data, str(file_path), records, ocr_results, job_id)

            t = time.time() - start_time
            final_conf = (
                sum(r.get("_confidence", 0) for r in records) / len(records)
            ) if records else 0.0

            await progress("complete", 100,
                f"Done! {len(records)} records in {t:.1f}s [{pipeline_label}].")

            asyncio.create_task(_maybe_auto_train())
            return _ok(
                excel_path, records, t, len(image_paths),
                pipeline_label, final_conf, job_id
            )

        # If we reach here, something unexpected happened
        raise RuntimeError(f"Unhandled file_type: {file_type}")

    except Exception as exc:
        logger.error(f"[{job_id}] Pipeline crashed: {exc}", exc_info=True)
        return {
            "success":         False,
            "error":           str(exc),
            "job_id":          job_id,
            "processing_time": time.time() - start_time,
        }
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir), ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-TRAINING
# ─────────────────────────────────────────────────────────────────────────────

async def _maybe_auto_train():
    if not _model_dir_has_files(LAYOUTLMV3_MODEL_PATH):
        return
    raw_dir = Path(TRAINING_DATA_PATH) / "raw"
    if not raw_dir.exists():
        return
    count = len(list(raw_dir.glob("*.json")))
    if count == 0 or count % AUTO_TRAIN_EVERY_N != 0:
        return
    if _training_lock.locked():
        logger.info("Auto-train: already running.")
        return
    async with _training_lock:
        logger.info(f"Auto-train triggered: {count} files (every {AUTO_TRAIN_EVERY_N}).")
        try:
            from training.layoutlm_trainer import train_layoutlmv3
            result = await train_layoutlmv3(
                data_dir   = str(Path(TRAINING_DATA_PATH) / "raw"),
                output_dir = str(Path(LAYOUTLMV3_MODEL_PATH)),
                epochs     = AUTO_TRAIN_EPOCHS,
                batch_size = 4,
            )
            if result.get("success"):
                version_info = {
                    "version":        datetime.now().strftime("v%Y%m%d_%H%M"),
                    "trained_at":     datetime.now().isoformat(),
                    "epochs":         AUTO_TRAIN_EPOCHS,
                    "training_files": count,
                    "final_loss":     result.get("final_loss", 0),
                    "accuracy":       result.get("accuracy",   0),
                }
                with open(Path(LAYOUTLMV3_MODEL_PATH) / "version.json", "w") as f:
                    json.dump(version_info, f, indent=2)
                logger.info(f"Auto-train complete: {version_info['version']}")
            else:
                logger.warning(f"Auto-train failed: {result.get('error')}")
        except Exception as exc:
            logger.error(f"Auto-train crashed: {exc}", exc_info=True)


def get_model_version() -> Dict[str, Any]:
    version_path = Path(LAYOUTLMV3_MODEL_PATH) / "version.json"
    if version_path.exists():
        try:
            with open(version_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "version":        "base",
        "trained_at":     None,
        "epochs":         0,
        "training_files": 0,
        "note":           "Using pre-trained LayoutLMv3 base (no fine-tuning yet)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYNC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _validate_records_sync(validator, records, full_text):
    return [validator.validate_and_clean(r, full_text) for r in records]


def _generate_excel_sync(records, output_dir, file_name, job_id) -> str:
    from .excel_generator import get_excel_generator
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return get_excel_generator().generate(
        extracted_data=records,
        output_path=str(output_dir),
        file_name=file_name,
        job_id=job_id,
    )


def _ok(
    excel_path, records, t, pages, method, conf, job_id,
    warning: Optional[str] = None,
) -> Dict:
    r = {
        "success":           True,
        "excel_path":        excel_path,
        "excel_filename":    Path(excel_path).name if excel_path else "",
        "records_extracted": len(records),
        "confidence":        round(conf, 4),
        "processing_time":   round(t, 2),
        "pages_processed":   pages,
        "method_used":       method,
        "job_id":            job_id,
    }
    if warning:
        r["warning"] = warning
    return r


def _build_sample_text(records: List[Dict]) -> str:
    return "\n".join(
        f"GSTIN:{r.get('GSTIN','')} Name:{r.get('TRADE_NAME','')} "
        f"Inv:{r.get('INVOICE_NO','')} Date:{r.get('INVOICE_DATE','')}"
        for r in records
    )


def _parse_ocr_text_fallback(text: str) -> List[Dict]:
    import re
    records  = []
    gstin_re = re.compile(r'\b(\d{2}[A-Z0-9]{13})\b')
    ewb_re   = re.compile(r'\b(\d{12})\b')
    date_re  = re.compile(r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b')
    amt_re   = re.compile(r'\b(\d+\.\d{1,2}|\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?)\b')

    gstins   = gstin_re.findall(text)
    ewbs     = ewb_re.findall(text)
    dates    = date_re.findall(text)
    amounts  = amt_re.findall(text)

    # Process GSTINs
    for i, gstin in enumerate(gstins):
        records.append({
            "GSTIN": gstin, "TRADE_NAME": "", "INVOICE_NO": f"UNKNOWN-{i+1}",
            "INVOICE_TYPE": "R",
            "INVOICE_DATE":  dates[i]     if i     < len(dates)    else "",
            "INVOICE_VALUE": amounts[i*2].replace(",","")   if i*2   < len(amounts) else "",
            "TAXABLE_VALUE": amounts[i*2+1].replace(",","") if i*2+1 < len(amounts) else "",
            "IGST": "", "CGST": "", "SGST": "", "CESS": "",
            "PLACE_OF_SUPPLY": "", "REVERSE_CHARGE": "N", "TAX_RATE": "",
            "GSTR1_STATUS": "", "GSTR1_FILING_DATE": "", "GSTR1_PERIOD": "",
            "GSTR3B_STATUS": "", "AMENDMENT": "N",
            "TAX_PERIOD_AMENDED": "N/A", "CANCEL_DATE": "N/A",
            "SOURCE": "regex_fallback", "IRN": "N/A", "IRN_DATE": "N/A",
            "_confidence": 0.3, "_source": "emergency_regex_fallback",
        })

    # Process EWB numbers if no GSTINs found
    if not records:
        for i, ewb in enumerate(ewbs):
            records.append({
                "GSTIN": "", "TRADE_NAME": "", "INVOICE_NO": ewb,
                "INVOICE_TYPE": "R",
                "INVOICE_DATE":  dates[i]     if i     < len(dates)    else "",
                "INVOICE_VALUE": amounts[i].replace(",","") if i < len(amounts) else "",
                "TAXABLE_VALUE": "",
                "IGST": "", "CGST": "", "SGST": "", "CESS": "",
                "PLACE_OF_SUPPLY": "", "REVERSE_CHARGE": "N", "TAX_RATE": "",
                "GSTR1_STATUS": "", "GSTR1_FILING_DATE": "", "GSTR1_PERIOD": "",
                "GSTR3B_STATUS": "", "AMENDMENT": "N",
                "TAX_PERIOD_AMENDED": "N/A", "CANCEL_DATE": "N/A",
                "SOURCE": "eway_bill_fallback", "IRN": "N/A", "IRN_DATE": "N/A",
                "_confidence": 0.25, "_source": "emergency_ewb_fallback",
            })

    return records


def _save_training_data(file_path, records, ocr_results, job_id):
    try:
        tp = Path(TRAINING_DATA_PATH) / "raw"
        tp.mkdir(parents=True, exist_ok=True)
        entry = {
            "job_id":       job_id,
            "source_file":  Path(file_path).name,
            "timestamp":    datetime.now().isoformat(),
            "records":      records,
            "ocr_text":     [r.get("full_text", "") for r in ocr_results],
            "record_count": len(records),
        }
        with open(tp / f"{job_id}_training_data.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Training data saved: {job_id}")
    except Exception as exc:
        logger.warning(f"Training data save failed: {exc}")