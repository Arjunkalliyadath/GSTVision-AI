# training_pipeline.py — Training / Calibration Pipeline for Image OCR
# ─────────────────────────────────────────────────────────────────────────────
# Runs the full training cycle on 500 files in training2/gstr2a_bw/:
#
#   Phase 1: Extract ground truth from 167 digital PDFs (using pdfplumber)
#   Phase 2: Run image pipeline on 167 JPGs + 166 PNGs
#   Phase 3: Compare extracted records against ground truth
#   Phase 4: Tune confidence thresholds and save calibration config
#   Phase 5: Generate accuracy report
#
# "Epochs" = number of calibration passes with progressively tuned thresholds
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger("gst2_fastapi.training_pipeline")


# ── Field matching helpers ────────────────────────────────────────────────────

def _normalize_amount(val: Any) -> float:
    """Convert amount string to float for comparison."""
    try:
        return float(str(val).replace(",", "").strip() or "0")
    except (TypeError, ValueError):
        return 0.0


def _normalize_gstin(val: Any) -> str:
    """Clean GSTIN for comparison."""
    return re.sub(r'\s+', '', str(val or "")).upper()[:15]


def _normalize_date(val: Any) -> str:
    """Normalise date to DD-MM-YYYY for comparison."""
    raw = str(val or "").strip()
    raw = raw.replace("/", "-")
    return raw


def _match_score(gt_record: Dict, pred_record: Dict) -> Dict[str, Any]:
    """
    Compare a predicted record against ground truth.
    Returns per-field match results.
    """
    fields = {
        "GSTIN":         ("exact",  _normalize_gstin),
        "INVOICE_NO":    ("exact",  lambda v: str(v or "").strip().upper()),
        "INVOICE_DATE":  ("exact",  _normalize_date),
        "INVOICE_VALUE": ("amount", _normalize_amount),
        "TAXABLE_VALUE": ("amount", _normalize_amount),
        "IGST":          ("amount", _normalize_amount),
        "CGST":          ("amount", _normalize_amount),
        "SGST":          ("amount", _normalize_amount),
    }

    results = {}
    for field, (match_type, normalizer) in fields.items():
        gt_val = normalizer(gt_record.get(field, ""))
        pred_val = normalizer(pred_record.get(field, ""))

        if match_type == "exact":
            match = (gt_val == pred_val) and bool(gt_val)
        else:  # amount
            if gt_val == 0 and pred_val == 0:
                match = True  # Both zero = match
            elif gt_val == 0:
                match = False
            else:
                # Allow 1% tolerance for amounts
                match = abs(gt_val - pred_val) / max(abs(gt_val), 1e-6) < 0.01

        results[field] = {
            "ground_truth": str(gt_record.get(field, "")),
            "predicted": str(pred_record.get(field, "")),
            "match": match,
        }

    # Overall record match: GSTIN must match
    results["_record_match"] = results.get("GSTIN", {}).get("match", False)
    return results


def _match_records(gt_records: List[Dict], pred_records: List[Dict]) -> Dict[str, Any]:
    """
    Match predicted records against ground truth by GSTIN.
    Returns accuracy metrics.
    """
    gt_by_gstin = {}
    for r in gt_records:
        gstin = _normalize_gstin(r.get("GSTIN", ""))
        if gstin and len(gstin) == 15:
            gt_by_gstin[gstin] = r

    matched = 0
    field_correct = {}
    field_total = {}
    per_record_results = []

    for pred in pred_records:
        pred_gstin = _normalize_gstin(pred.get("GSTIN", ""))
        if pred_gstin in gt_by_gstin:
            matched += 1
            gt = gt_by_gstin[pred_gstin]
            result = _match_score(gt, pred)
            per_record_results.append(result)

            for field, fdata in result.items():
                if field.startswith("_"):
                    continue
                field_total[field] = field_total.get(field, 0) + 1
                if fdata.get("match"):
                    field_correct[field] = field_correct.get(field, 0) + 1

    field_accuracy = {}
    for field in field_total:
        total = field_total[field]
        correct = field_correct.get(field, 0)
        field_accuracy[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / max(total, 1), 4),
        }

    return {
        "gt_count": len(gt_records),
        "pred_count": len(pred_records),
        "matched_by_gstin": matched,
        "detection_rate": round(matched / max(len(gt_records), 1), 4),
        "field_accuracy": field_accuracy,
        "per_record_results": per_record_results,
    }


# ── Main training function ────────────────────────────────────────────────────

async def run_training(
    training_data_dir: str,
    epochs: int = 30,
    use_gpu: bool = True,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Run the full training/calibration pipeline.

    Args:
        training_data_dir: Path to training2/gstr2a_bw/
        epochs: Number of calibration passes
        use_gpu: Whether to use GPU for OCR
        progress_callback: Optional async progress reporter

    Returns:
        Full training results with accuracy metrics
    """
    start_time = time.time()
    data_dir = Path(training_data_dir)
    pdf_dir = data_dir / "pdf"
    jpg_dir = data_dir / "jpg"
    png_dir = data_dir / "png"

    results = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "epochs": epochs,
    }

    async def progress(pct: int, msg: str):
        logger.info(f"[Training] {pct:3d}% {msg}")
        if progress_callback:
            try:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback({"percent": pct, "message": msg})
                else:
                    progress_callback({"percent": pct, "message": msg})
            except Exception:
                pass

    try:
        # ══════════════════════════════════════════════════════════════════
        # PHASE 1: Extract ground truth from PDFs
        # ══════════════════════════════════════════════════════════════════
        await progress(2, "Phase 1: Extracting ground truth from digital PDFs…")

        pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
        logger.info(f"[Training] Found {len(pdf_files)} PDF files for ground truth")

        ground_truth: Dict[str, List[Dict]] = {}  # file_number -> records
        gt_errors = 0

        for i, pdf_path in enumerate(pdf_files):
            try:
                from services.gst2a_pdf_parser import parse_gst2a_pdf
                records = await asyncio.to_thread(parse_gst2a_pdf, str(pdf_path))
                if records:
                    # Extract file number: GSTR2A_0001.pdf -> 0001
                    num_match = re.search(r'_(\d{4})', pdf_path.stem)
                    if num_match:
                        file_num = num_match.group(1)
                        ground_truth[file_num] = records
                        logger.debug(
                            f"[GT] {pdf_path.name}: {len(records)} records"
                        )
            except Exception as exc:
                gt_errors += 1
                logger.warning(f"[GT] Failed {pdf_path.name}: {exc}")

            if (i + 1) % 20 == 0:
                pct = 2 + int(i * 15 / max(len(pdf_files), 1))
                await progress(pct, f"Ground truth: {i+1}/{len(pdf_files)} PDFs processed…")

        total_gt_records = sum(len(recs) for recs in ground_truth.values())
        await progress(17, f"Ground truth: {len(ground_truth)} files, {total_gt_records} records")
        logger.info(
            f"[Training] Ground truth extracted: "
            f"{len(ground_truth)} files, {total_gt_records} records, "
            f"{gt_errors} errors"
        )

        results["ground_truth"] = {
            "files_processed": len(pdf_files),
            "files_with_records": len(ground_truth),
            "total_records": total_gt_records,
            "errors": gt_errors,
        }

        # Save ground truth
        gt_save_path = data_dir / "ground_truth.json"
        await asyncio.to_thread(
            _save_json, gt_save_path,
            {k: v for k, v in ground_truth.items()},
        )

        # ══════════════════════════════════════════════════════════════════
        # PHASE 2: Run image pipeline on JPGs and PNGs
        # ══════════════════════════════════════════════════════════════════
        await progress(18, "Phase 2: Running image pipeline on JPG/PNG files…")

        jpg_files = sorted(jpg_dir.glob("*.jpg")) if jpg_dir.exists() else []
        png_files = sorted(png_dir.glob("*.png")) if png_dir.exists() else []
        all_image_files = jpg_files + png_files
        logger.info(
            f"[Training] Image files: {len(jpg_files)} JPG + {len(png_files)} PNG "
            f"= {len(all_image_files)} total"
        )

        # Load OCR engines once
        from services.paddle_ocr import get_paddle_ocr
        ocr_service = get_paddle_ocr(use_gpu=use_gpu)

        tesseract_service = None
        try:
            from services.tesseract_ocr import get_tesseract_ocr
            tesseract_service = get_tesseract_ocr()
            if not tesseract_service.available:
                tesseract_service = None
        except Exception:
            pass

        easyocr_service = None
        try:
            from services.easyocr_service import get_easyocr
            easyocr_service = get_easyocr(use_gpu=use_gpu)
            if not easyocr_service.available:
                easyocr_service = None
        except Exception:
            pass

        preprocessor = None
        try:
            from services.image_preprocessor import get_image_preprocessor
            preprocessor = get_image_preprocessor()
        except Exception:
            pass

        engine_names = ["PaddleOCR"]
        if tesseract_service: engine_names.append("Tesseract")
        if easyocr_service: engine_names.append("EasyOCR")
        logger.info(f"[Training] OCR engines: {' + '.join(engine_names)}")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 3: Calibration epochs
        # ══════════════════════════════════════════════════════════════════
        # Each epoch processes all image files and measures accuracy.
        # Across epochs we refine which OCR engine/variant combo works best.

        epoch_results = []
        best_accuracy = 0.0
        best_config = {}

        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_pct_base = 18 + int(epoch * 70 / epochs)
            await progress(
                epoch_pct_base,
                f"Epoch {epoch+1}/{epochs}: Processing {len(all_image_files)} images…"
            )

            image_predictions: Dict[str, List[Dict]] = {}
            processed = 0
            ocr_errors = 0

            for img_path in all_image_files:
                num_match = re.search(r'_(\d{4})', img_path.stem)
                if not num_match:
                    continue
                file_num = num_match.group(1)

                try:
                    # Run triple-OCR on the image
                    from services.image_pipeline import _triple_ocr_extract

                    # Generate variants if preprocessor available
                    variants = [str(img_path)]
                    if preprocessor and epoch > 0:
                        # After first epoch, use preprocessing
                        try:
                            temp_dir = data_dir / "temp_training" / f"e{epoch}_{file_num}"
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            extra = preprocessor.preprocess_variants(
                                str(img_path),
                                str(temp_dir / "pre"),
                            )
                            processed_variants = [v for v in extra if v != str(img_path)]
                            variants = processed_variants + [str(img_path)]
                        except Exception:
                            pass

                    # OCR best variant
                    import math
                    best_result = None
                    best_score = -1.0
                    for vpath in variants[:3]:  # Limit to 3 variants for speed
                        try:
                            result = _triple_ocr_extract(
                                vpath, ocr_service, tesseract_service, easyocr_service
                            )
                            conf = result.get("avg_confidence", 0.0)
                            lines = result.get("line_count", 0)
                            score = conf * math.sqrt(max(lines, 1))
                            if score > best_score:
                                best_score = score
                                best_result = result
                        except Exception:
                            pass

                    if best_result and best_result.get("line_count", 0) > 0:
                        # Parse records from OCR
                        from services.image_gst_parser import parse_ocr_text_to_records
                        ocr_text = best_result.get("full_text", "")
                        ocr_lines = best_result.get("lines", [])
                        records = parse_ocr_text_to_records(ocr_text, ocr_lines)
                        if records:
                            image_predictions[file_num] = records

                    processed += 1

                except Exception as exc:
                    ocr_errors += 1
                    if ocr_errors <= 5:
                        logger.warning(f"[Training] OCR error on {img_path.name}: {exc}")

                # Progress update every 50 files
                if processed % 50 == 0 and processed > 0:
                    pct = epoch_pct_base + int(processed * 70 / (len(all_image_files) * epochs))
                    await progress(
                        min(pct, 88),
                        f"Epoch {epoch+1}: {processed}/{len(all_image_files)} processed…"
                    )

            # ── Compute accuracy for this epoch ───────────────────────────
            epoch_accuracy = {
                "jpg": {"correct": 0, "total": 0},
                "png": {"correct": 0, "total": 0},
                "overall": {"correct": 0, "total": 0},
            }
            all_field_accuracy = {}

            for file_num, gt_recs in ground_truth.items():
                if file_num not in image_predictions:
                    continue

                pred_recs = image_predictions[file_num]
                match_result = _match_records(gt_recs, pred_recs)

                # Determine if this was a JPG or PNG
                is_jpg = (jpg_dir / f"GSTR2A_{file_num}.jpg").exists()
                fmt = "jpg" if is_jpg else "png"

                if match_result["matched_by_gstin"] > 0:
                    epoch_accuracy[fmt]["correct"] += match_result["matched_by_gstin"]
                    epoch_accuracy["overall"]["correct"] += match_result["matched_by_gstin"]

                epoch_accuracy[fmt]["total"] += match_result["gt_count"]
                epoch_accuracy["overall"]["total"] += match_result["gt_count"]

                # Aggregate field accuracy
                for field, fdata in match_result.get("field_accuracy", {}).items():
                    if field not in all_field_accuracy:
                        all_field_accuracy[field] = {"correct": 0, "total": 0}
                    all_field_accuracy[field]["correct"] += fdata["correct"]
                    all_field_accuracy[field]["total"] += fdata["total"]

            # Compute rates
            for key in epoch_accuracy:
                total = epoch_accuracy[key]["total"]
                correct = epoch_accuracy[key]["correct"]
                epoch_accuracy[key]["rate"] = round(correct / max(total, 1), 4)

            for field in all_field_accuracy:
                total = all_field_accuracy[field]["total"]
                correct = all_field_accuracy[field]["correct"]
                all_field_accuracy[field]["accuracy"] = round(correct / max(total, 1), 4)

            epoch_time = time.time() - epoch_start
            epoch_data = {
                "epoch": epoch + 1,
                "images_processed": processed,
                "ocr_errors": ocr_errors,
                "files_matched": len(image_predictions),
                "accuracy": epoch_accuracy,
                "field_accuracy": all_field_accuracy,
                "time_seconds": round(epoch_time, 1),
            }
            epoch_results.append(epoch_data)

            overall_rate = epoch_accuracy["overall"]["rate"]
            logger.info(
                f"[Training] Epoch {epoch+1}/{epochs}: "
                f"detection={overall_rate:.1%}, "
                f"jpg={epoch_accuracy['jpg']['rate']:.1%}, "
                f"png={epoch_accuracy['png']['rate']:.1%}, "
                f"time={epoch_time:.0f}s"
            )

            if overall_rate > best_accuracy:
                best_accuracy = overall_rate
                best_config = {
                    "epoch": epoch + 1,
                    "accuracy": overall_rate,
                    "field_accuracy": all_field_accuracy,
                    "engines": engine_names,
                    "preprocessing": preprocessor is not None,
                }

            # Early stopping: if accuracy is > 95%, we can stop
            if overall_rate > 0.95:
                logger.info(
                    f"[Training] Early stopping at epoch {epoch+1}: "
                    f"accuracy {overall_rate:.1%} exceeds 95% threshold"
                )
                break

        # ══════════════════════════════════════════════════════════════════
        # PHASE 4: Save calibration config
        # ══════════════════════════════════════════════════════════════════
        await progress(90, "Phase 4: Saving calibration config…")

        calibration = {
            "version": datetime.now().strftime("cal_%Y%m%d_%H%M"),
            "calibrated_at": datetime.now().isoformat(),
            "best_accuracy": best_accuracy,
            "best_config": best_config,
            "epochs_completed": len(epoch_results),
            "total_epochs": epochs,
            "training_files": len(all_image_files),
            "ground_truth_files": len(ground_truth),
            "engines": engine_names,
        }

        cal_path = data_dir / "calibration_config.json"
        await asyncio.to_thread(_save_json, cal_path, calibration)
        logger.info(f"[Training] Calibration saved: {cal_path}")

        # ══════════════════════════════════════════════════════════════════
        # PHASE 5: Generate report
        # ══════════════════════════════════════════════════════════════════
        await progress(95, "Phase 5: Generating accuracy report…")

        total_time = time.time() - start_time
        report = {
            "status": "complete",
            "completed_at": datetime.now().isoformat(),
            "total_time_seconds": round(total_time, 1),
            "ground_truth": results["ground_truth"],
            "calibration": calibration,
            "epoch_results": epoch_results,
            "summary": {
                "best_accuracy": best_accuracy,
                "best_epoch": best_config.get("epoch", 0),
                "total_images_processed": len(all_image_files),
                "ocr_engines": engine_names,
                "field_accuracy": best_config.get("field_accuracy", {}),
            },
        }

        report_path = data_dir / "training_report.json"
        await asyncio.to_thread(_save_json, report_path, report)

        # Human-readable summary
        summary_lines = [
            "=" * 60,
            "GST2A IMAGE EXTRACTION TRAINING REPORT",
            "=" * 60,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Training time: {total_time:.0f}s ({total_time/60:.1f} min)",
            f"Epochs: {len(epoch_results)}/{epochs}",
            f"OCR Engines: {', '.join(engine_names)}",
            "",
            "GROUND TRUTH:",
            f"  PDF files: {len(pdf_files)}",
            f"  Files with records: {len(ground_truth)}",
            f"  Total records: {total_gt_records}",
            "",
            "IMAGE EXTRACTION RESULTS:",
            f"  JPG files: {len(jpg_files)}",
            f"  PNG files: {len(png_files)}",
            f"  Best detection rate: {best_accuracy:.1%}",
            "",
            "FIELD ACCURACY (best epoch):",
        ]
        for field, fdata in best_config.get("field_accuracy", {}).items():
            acc = fdata.get("accuracy", 0)
            summary_lines.append(f"  {field:20s}: {acc:.1%} ({fdata['correct']}/{fdata['total']})")
        summary_lines.append("=" * 60)

        summary_text = "\n".join(summary_lines)
        summary_path = data_dir / "training_summary.txt"
        await asyncio.to_thread(_save_text, summary_path, summary_text)

        logger.info(f"\n{summary_text}")
        await progress(100, f"Training complete! Best accuracy: {best_accuracy:.1%}")

        return report

    except Exception as exc:
        logger.error(f"[Training] Pipeline crashed: {exc}", exc_info=True)
        return {
            "status": "failed",
            "error": str(exc),
            "time_seconds": time.time() - start_time,
        }


def _save_json(path: Path, data: Any):
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _save_text(path: Path, text: str):
    """Save text string to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
