# image_pipeline.py — Dedicated Image Extraction Pipeline (v1.0)
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Optimised extraction for JPG, PNG, and photo/scanned PDFs.
#   This is the NEW PATH 2 — completely separate from the digital PDF pipeline.
#
# ARCHITECTURE:
#   1. Convert file to images (pdf_converter)
#   2. Auto-rotate + adaptive preprocessing (image_preprocessor)
#   3. Triple-OCR: PaddleOCR + Tesseract + EasyOCR on each variant
#   4. Confidence-weighted ensemble — best-of-3 engines per variant
#   5. Best variant selection across all variants
#   6. GST record extraction (image_gst_parser)
#   7. Field validation + enrichment
#
# WHY TRIPLE-OCR?
#   • PaddleOCR: Best on clean, well-aligned text
#   • Tesseract: Best on mixed alphanumeric (GSTINs, invoice numbers)
#   • EasyOCR:   Best on diverse quality images, good fallback
#   When 2/3 engines agree on a GSTIN → high confidence
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import logging
import math
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.image_pipeline")


def _ocr_score(result: Dict) -> float:
    """Composite variant-selection score = avg_confidence × √(line_count)"""
    conf = result.get("avg_confidence", 0.0)
    lines = result.get("line_count", 0)
    return conf * math.sqrt(max(lines, 1))


def _triple_ocr_extract(
    image_path: str,
    ocr_service,
    tesseract_service,
    easyocr_service,
) -> Dict[str, Any]:
    """
    Run PaddleOCR, Tesseract, and EasyOCR on a single image.
    Return the result with the highest composite score.
    """
    best_result = None
    best_score = -1.0

    engines = [
        ("paddleocr", ocr_service, lambda svc, p: svc.extract_text(p)),
        ("tesseract", tesseract_service, lambda svc, p: svc.extract_text_with_boxes(p) if hasattr(svc, 'extract_text_with_boxes') else svc.extract_text(p)),
        ("easyocr", easyocr_service, lambda svc, p: svc.extract_text(p)),
    ]

    for engine_name, service, extract_fn in engines:
        if service is None:
            continue
        if hasattr(service, 'available') and not service.available:
            continue

        try:
            result = extract_fn(service, image_path)
            score = _ocr_score(result)
            result["_ocr_engine"] = engine_name

            logger.debug(
                f"  {engine_name}: lines={result['line_count']} "
                f"conf={result['avg_confidence']:.2f} score={score:.2f}"
            )

            if score > best_score:
                best_score = score
                best_result = result

        except Exception as exc:
            logger.warning(f"  {engine_name} failed: {exc}")

    if best_result is None:
        best_result = {
            "lines": [], "full_text": "", "avg_confidence": 0.0,
            "line_count": 0, "source": "all_ocr_failed",
        }

    return best_result


async def _ocr_page_with_preprocessing(
    page_num: int,
    total_pages: int,
    orig_path: str,
    temp_dir: Path,
    ocr_service,
    tesseract_service,
    easyocr_service,
    preprocessor,
    job_id: str,
    progress_fn,
) -> Dict[str, Any]:
    """
    Process a single page through the full image pipeline:
    1. Generate preprocessed variants (includes auto-rotation)
    2. Triple-OCR each variant
    3. Return best-scoring result
    """
    page_pct = 20 + int(page_num * 50 / max(total_pages, 1))
    await progress_fn(
        "image_ocr", page_pct,
        f"Page {page_num+1}/{total_pages}: triple-OCR (Paddle+Tesseract+EasyOCR)…"
    )

    # Step 1 — Generate enhanced variants
    variants = [orig_path]
    if preprocessor:
        try:
            extra = await asyncio.to_thread(
                preprocessor.preprocess_variants,
                orig_path,
                str(temp_dir / f"img_pre_{page_num}"),
            )
            processed = [v for v in extra if v != orig_path]
            variants = processed + [orig_path]
            logger.info(
                f"[{job_id}] img p{page_num+1}: {len(variants)} variants "
                f"({len(processed)} processed + original)"
            )
        except Exception as exc:
            logger.warning(f"[{job_id}] Variant generation p{page_num+1}: {exc}")

    # Step 2 — Triple-OCR all variants, keep highest-scoring
    best_result = None
    best_score = -1.0

    for vi, vpath in enumerate(variants):
        try:
            result = await asyncio.to_thread(
                _triple_ocr_extract, vpath,
                ocr_service, tesseract_service, easyocr_service,
            )
            score = _ocr_score(result)
            engine = result.get("_ocr_engine", "unknown")
            logger.info(
                f"[{job_id}] img p{page_num+1} v{vi+1}/{len(variants)} "
                f"({Path(vpath).stem[-15:]}): "
                f"lines={result['line_count']} "
                f"conf={result['avg_confidence']:.2f} "
                f"score={score:.2f} engine={engine}"
            )
            if score > best_score:
                best_score = score
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
    best_result["image_path"] = orig_path

    logger.info(
        f"[{job_id}] Image page {page_num+1} FINAL: "
        f"lines={best_result['line_count']} "
        f"conf={best_result['avg_confidence']:.2f} "
        f"score={best_score:.2f} "
        f"engine={best_result.get('_ocr_engine', 'unknown')}"
    )
    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — Called by pipeline.py for image/photo_pdf files
# ─────────────────────────────────────────────────────────────────────────────

async def process_image_file(
    file_path: str,
    job_id: str,
    output_dir: Path,
    temp_dir: Path,
    use_gpu: bool = True,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Full image extraction pipeline for JPG/PNG/photo PDFs.

    Returns:
        {
            "records": [...],
            "ocr_results": [...],
            "image_paths": [...],
            "avg_confidence": float,
            "total_lines": int,
            "pipeline": "image_triple_ocr",
        }
    """
    start = time.time()
    ext = Path(file_path).suffix.lower()

    async def progress(step, pct, msg):
        logger.info(f"[{job_id}] {pct:3d}% [{step}] {msg}")
        if progress_callback:
            try:
                await progress_callback({"step": step, "percent": pct, "message": msg})
            except Exception:
                pass
        await asyncio.sleep(0)

    # Step 1 — Convert to images
    await progress("converting", 10, f"Converting {ext.upper()} to images…")
    from .pdf_converter import convert_file_to_images_async
    image_paths = await convert_file_to_images_async(file_path, str(temp_dir))
    logger.info(f"[{job_id}] Image pipeline: {len(image_paths)} page(s)")

    if not image_paths:
        return {
            "records": [], "ocr_results": [], "image_paths": [],
            "avg_confidence": 0.0, "total_lines": 0,
            "pipeline": "image_conversion_failed",
        }

    # Step 2 — Load OCR engines
    await progress("loading_engines", 15, "Loading triple-OCR engines…")

    # PaddleOCR
    from .paddle_ocr import get_paddle_ocr
    ocr_service = get_paddle_ocr(use_gpu=use_gpu)

    # Tesseract
    tesseract_service = None
    try:
        from .tesseract_ocr import get_tesseract_ocr
        tesseract_service = get_tesseract_ocr()
        if not tesseract_service.available:
            tesseract_service = None
    except Exception as exc:
        logger.info(f"[{job_id}] Tesseract unavailable: {exc}")

    # EasyOCR
    easyocr_service = None
    try:
        from .easyocr_service import get_easyocr
        easyocr_service = get_easyocr(use_gpu=use_gpu)
        if not easyocr_service.available:
            easyocr_service = None
    except Exception as exc:
        logger.info(f"[{job_id}] EasyOCR unavailable: {exc}")

    engine_names = ["PaddleOCR"]
    if tesseract_service: engine_names.append("Tesseract")
    if easyocr_service: engine_names.append("EasyOCR")
    logger.info(f"[{job_id}] Engines: {' + '.join(engine_names)}")

    # Step 3 — Load preprocessor
    preprocessor = None
    try:
        from .image_preprocessor import get_image_preprocessor
        preprocessor = get_image_preprocessor()
    except Exception as exc:
        logger.warning(f"[{job_id}] Preprocessor unavailable: {exc}")

    # Step 4 — Process each page with triple-OCR
    await progress("image_ocr", 20, f"Triple-OCR on {len(image_paths)} page(s)…")
    ocr_results: List[Dict] = []

    for page_num, orig_path in enumerate(image_paths):
        result = await _ocr_page_with_preprocessing(
            page_num=page_num,
            total_pages=len(image_paths),
            orig_path=orig_path,
            temp_dir=temp_dir,
            ocr_service=ocr_service,
            tesseract_service=tesseract_service,
            easyocr_service=easyocr_service,
            preprocessor=preprocessor,
            job_id=job_id,
            progress_fn=progress,
        )
        ocr_results.append(result)

    avg_conf = sum(r["avg_confidence"] for r in ocr_results) / max(len(ocr_results), 1)
    total_lines = sum(r["line_count"] for r in ocr_results)

    logger.info(
        f"[{job_id}] Image OCR complete: "
        f"{total_lines} lines, avg conf={avg_conf:.2f}"
    )

    # Step 5 — Extract GST records from OCR text
    await progress("parsing", 75, "Extracting GST records from OCR text…")
    records: List[Dict] = []

    try:
        from .image_gst_parser import parse_ocr_results_to_records
        records = await asyncio.to_thread(parse_ocr_results_to_records, ocr_results)
        logger.info(f"[{job_id}] image_gst_parser: {len(records)} records")
    except Exception as exc:
        logger.warning(f"[{job_id}] image_gst_parser failed: {exc}")

    # Step 6 — Emergency fallback if parser found nothing
    if not records:
        await progress("fallback", 80, "Running regex fallback…")
        combined = "\n".join(r.get("full_text", "") for r in ocr_results)
        records = _emergency_regex_fallback(combined)
        logger.info(f"[{job_id}] Regex fallback: {len(records)} records")

    elapsed = time.time() - start
    logger.info(
        f"[{job_id}] Image pipeline complete: "
        f"{len(records)} records in {elapsed:.1f}s"
    )

    return {
        "records": records,
        "ocr_results": ocr_results,
        "image_paths": image_paths,
        "avg_confidence": avg_conf,
        "total_lines": total_lines,
        "pipeline": "image_triple_ocr",
    }


def _emergency_regex_fallback(text: str) -> List[Dict]:
    """Last-resort regex extraction when all parsers fail."""
    import re

    records = []
    gstin_re = re.compile(r'\b(\d{2}[A-Z0-9]{13})\b')
    date_re = re.compile(r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b')
    amt_re = re.compile(r'\b(\d+\.\d{1,2}|\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?)\b')

    gstins = gstin_re.findall(text)
    dates = date_re.findall(text)
    amounts = amt_re.findall(text)

    for i, gstin in enumerate(gstins):
        records.append({
            "GSTIN": gstin, "TRADE_NAME": "", "INVOICE_NO": f"UNKNOWN-{i+1}",
            "INVOICE_TYPE": "R",
            "INVOICE_DATE": dates[i] if i < len(dates) else "",
            "INVOICE_VALUE": amounts[i*2].replace(",", "") if i*2 < len(amounts) else "",
            "TAXABLE_VALUE": amounts[i*2+1].replace(",", "") if i*2+1 < len(amounts) else "",
            "IGST": "", "CGST": "", "SGST": "", "CESS": "",
            "PLACE_OF_SUPPLY": "", "REVERSE_CHARGE": "N", "TAX_RATE": "",
            "GSTR1_STATUS": "", "GSTR1_FILING_DATE": "", "GSTR1_PERIOD": "",
            "GSTR3B_STATUS": "", "AMENDMENT": "N",
            "TAX_PERIOD_AMENDED": "N/A", "CANCEL_DATE": "N/A",
            "SOURCE": "image_regex_fallback", "IRN": "N/A", "IRN_DATE": "N/A",
            "_confidence": 0.30, "_source": "image_emergency_regex",
        })

    return records
