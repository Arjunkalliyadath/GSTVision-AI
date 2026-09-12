"""Debug: compare GT vs OCR predictions on overlapping file numbers."""
import json, re, sys, os
from pathlib import Path

sys.path.insert(0, ".")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_DIR = Path("../../training2/gstr2a_bw")

with open(DATA_DIR / "ground_truth.json") as f:
    gt = json.load(f)

jpg_dir = DATA_DIR / "jpg"
png_dir = DATA_DIR / "png"

# Find overlapping file numbers (GT has it AND an image exists)
overlaps = []
for fnum in sorted(gt.keys()):
    jpg = jpg_dir / f"GSTR2A_{fnum}.jpg"
    png = png_dir / f"GSTR2A_{fnum}.png"
    if jpg.exists() or png.exists():
        img_path = str(jpg if jpg.exists() else png)
        overlaps.append((fnum, img_path))

print(f"Ground truth files: {len(gt)}")
print(f"Files with BOTH GT and image: {len(overlaps)}")
if not overlaps:
    print("NO OVERLAP — GT file numbers don't match image file numbers!")
    print(f"GT file numbers: {sorted(gt.keys())[:10]}...")

    # List image file numbers
    jpg_nums = set()
    for f in jpg_dir.glob("*.jpg"):
        m = re.search(r'_(\d{4})', f.stem)
        if m: jpg_nums.add(m.group(1))
    png_nums = set()
    for f in png_dir.glob("*.png"):
        m = re.search(r'_(\d{4})', f.stem)
        if m: png_nums.add(m.group(1))
    all_img_nums = jpg_nums | png_nums
    print(f"Image file numbers: {sorted(all_img_nums)[:10]}...")
    print(f"GT nums not in images: {sorted(set(gt.keys()) - all_img_nums)[:10]}...")
    print(f"Image nums not in GT: {sorted(all_img_nums - set(gt.keys()))[:10]}...")
    print(f"Intersection: {len(set(gt.keys()) & all_img_nums)}")
else:
    # Test first 3 overlapping files
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    from services.image_gst_parser import parse_ocr_text_to_records

    for fnum, img_path in overlaps[:3]:
        print(f"\n--- File {fnum}: {Path(img_path).name} ---")
        gt_recs = gt[fnum]
        gt_gstins = [r.get("GSTIN", "") for r in gt_recs if r.get("GSTIN")]
        print(f"  GT GSTINs ({len(gt_gstins)}): {gt_gstins[:3]}")

        # OCR
        img = Image.open(img_path)
        data = pytesseract.image_to_data(img, lang="eng", config="--oem 3 --psm 6",
                                          output_type=pytesseract.Output.DICT)
        lines = []
        cur_line, cur_text = -1, ""
        for i in range(len(data["text"])):
            t = data["text"][i].strip()
            c = int(data["conf"][i])
            ln = data["line_num"][i]
            if not t or c < 0: continue
            if ln != cur_line:
                if cur_text.strip(): lines.append(cur_text.strip())
                cur_line = ln
                cur_text = t
            else:
                cur_text += " " + t
        if cur_text.strip(): lines.append(cur_text.strip())

        full_text = "\n".join(lines)
        pred_recs = parse_ocr_text_to_records(full_text, None)
        pred_gstins = [r.get("GSTIN", "") for r in pred_recs if r.get("GSTIN")]
        print(f"  Pred GSTINs ({len(pred_gstins)}): {pred_gstins[:3]}")

        # Check matches
        gt_set = set(g.upper().replace(" ", "")[:15] for g in gt_gstins)
        pred_set = set(g.upper().replace(" ", "")[:15] for g in pred_gstins)
        overlap_set = gt_set & pred_set
        print(f"  Match: {len(overlap_set)}/{len(gt_set)}")
        if gt_set and not overlap_set:
            print(f"  GT sample:   {list(gt_set)[:2]}")
            print(f"  Pred sample: {list(pred_set)[:2]}")
