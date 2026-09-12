import json
d = json.load(open("../../training2/gstr2a_bw/calibration_config.json"))
m = d.get("metrics", {})
c = d.get("counts", {})
pf = d.get("per_format", {})

er = m.get("extraction_rate", 0)
gv = m.get("gstin_validity_rate", 0)
oc = m.get("avg_ocr_confidence", 0)
ar = m.get("avg_records_per_image", 0)
tr = c.get("total_records", 0)
vg = c.get("valid_gstins", 0)
tg = c.get("total_gstins", 0)
errors = c.get("errors", 0)

jpgw = pf.get("jpg", {}).get("with_records", 0)
jpgi = pf.get("jpg", {}).get("images", 0)
jpgr = pf.get("jpg", {}).get("records", 0)
pngw = pf.get("png", {}).get("with_records", 0)
pngi = pf.get("png", {}).get("images", 0)
pngr = pf.get("png", {}).get("records", 0)

fc = m.get("field_coverage", {})

print("=" * 50)
print("  CALIBRATION RESULTS")
print("=" * 50)
print(f"  Extraction Rate:    {er*100:.1f}%  ({jpgw+pngw} images yielded records)")
print(f"  GSTIN Validity:     {gv*100:.1f}%  ({vg}/{tg} valid)")
print(f"  Avg OCR Confidence: {oc*100:.1f}%")
print(f"  Avg Records/Image:  {ar:.1f}")
print(f"  Total Records:      {tr}")
print(f"  Errors:             {errors}")
print()
print("  FORMAT BREAKDOWN:")
print(f"    JPG: {jpgw}/{jpgi} extracted, {jpgr} records")
print(f"    PNG: {pngw}/{pngi} extracted, {pngr} records")
print()
print("  FIELD COVERAGE:")
for f, p in fc.items():
    print(f"    {f:20s}: {p*100:.1f}%")
print("=" * 50)

# Also save the training_report.json that crashed
report = {
    "status": "complete",
    "calibration": d,
    "summary": m,
}
json.dump(report, open("../../training2/gstr2a_bw/training_report.json", "w"), indent=2)
print("\ntraining_report.json saved!")
