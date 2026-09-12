import asyncio
import os
import sys
import json
import time

sys.path.append(os.path.abspath("c:/DSA/Intern2/GST2_Converter/backend/fastapi_service"))
os.environ["MEDIA_ROOT"] = "../../training2"

from services.paddle_ocr import get_paddle_ocr
from services.tesseract_ocr import get_tesseract_ocr
from services.easyocr_service import get_easyocr
from services.image_pipeline import _triple_ocr_extract
import math
from pathlib import Path

# field matcher
def _normalize_gstin(v):
    return str(v or "").replace(" ", "").upper()[:15]

def _normalize_amount(v):
    try:
        return float(str(v).replace(",", "").strip() or "0")
    except:
        return 0.0

async def measure_now():
    print("Loading OCR engines...")
    paddle = get_paddle_ocr(use_gpu=True)
    tess = get_tesseract_ocr()
    easy = get_easyocr(use_gpu=True)
    print("Engines loaded.\n")

    data_dir = Path("../../training2/gstr2a_bw")
    with open(data_dir / "ground_truth.json", "r") as f:
        gt = json.load(f)

    # Pick first 5 files that have jpgs
    test_files = []
    for file_num in gt.keys():
        jpg_path = data_dir / "jpg" / f"GSTR2A_{file_num}.jpg"
        if jpg_path.exists():
            test_files.append((file_num, str(jpg_path), gt[file_num]))
            if len(test_files) >= 5:
                break

    print(f"Testing on {len(test_files)} sample files for live metrics...")

    engine_wins = {"PaddleOCR": 0, "Tesseract": 0, "EasyOCR": 0}
    total_conf = 0.0
    
    # We measure simple precision / recall for fields
    tp, fp, fn = 0, 0, 0

    for num, path, truth_recs in test_files:
        print(f"-> Processing {Path(path).name}...")
        res = _triple_ocr_extract(path, paddle, tess, easy)
        
        # Which engine won?
        best_engine = res.get("engine", "Unknown")
        if best_engine in engine_wins:
            engine_wins[best_engine] += 1
        
        total_conf += res.get("avg_confidence", 0.0)

        # Parse the records
        from services.image_gst_parser import parse_ocr_text_to_records
        pred_recs = parse_ocr_text_to_records(res.get("full_text", ""), res.get("lines", []))

        # Ground truth GSTINs
        gt_gstins = set(_normalize_gstin(r.get("GSTIN", "")) for r in truth_recs if r.get("GSTIN"))
        pred_gstins = set(_normalize_gstin(r.get("GSTIN", "")) for r in pred_recs if r.get("GSTIN"))

        for g in pred_gstins:
            if g in gt_gstins:
                tp += 1
            else:
                fp += 1
        for g in gt_gstins:
            if g not in pred_gstins:
                fn += 1

    avg_conf = total_conf / max(len(test_files), 1)
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

    print("\n--- LIVE ESTIMATED METRICS ---")
    print(f"Average System Confidence: {avg_conf:.2%}")
    print(f"Detection Precision:       {precision:.2%}")
    print(f"Detection Recall:          {recall:.2%}")
    print(f"F1 Score:                  {f1:.2%}")
    print("\n--- ENGINE PERFORMANCE ---")
    for eng, wins in engine_wins.items():
        print(f"{eng} selected as BEST: {wins} times")

if __name__ == "__main__":
    asyncio.run(measure_now())
