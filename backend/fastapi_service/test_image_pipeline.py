"""Quick end-to-end test of the image OCR pipeline."""
import os, sys, asyncio, glob, time
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, ".")

async def test():
    # Find a test image
    upload_dir = "../../media/uploads"
    images = glob.glob(os.path.join(upload_dir, "*.jpeg")) + \
             glob.glob(os.path.join(upload_dir, "*.jpg")) + \
             glob.glob(os.path.join(upload_dir, "*.png"))
    
    if not images:
        print("No test images found in uploads directory")
        return
    
    test_file = images[0]
    print(f"\n{'='*60}")
    print(f"Testing with: {os.path.basename(test_file)}")
    print(f"{'='*60}\n")
    
    # Test 1: Auto-rotation
    print("[1] Testing auto-rotation detection...")
    from services.auto_rotate import detect_and_correct_rotation
    rotated_path, angle = detect_and_correct_rotation(test_file)
    print(f"    Rotation detected: {angle}°")
    print(f"    Output: {os.path.basename(rotated_path)}")
    
    # Test 2: Tesseract OCR
    print("\n[2] Testing Tesseract OCR...")
    from services.tesseract_ocr import get_tesseract_ocr
    tess = get_tesseract_ocr()
    if tess.available:
        tess_result = tess.extract_text(rotated_path)
        print(f"    Lines: {tess_result['line_count']}")
        print(f"    Confidence: {tess_result['avg_confidence']:.2f}")
        if tess_result['line_count'] > 0:
            print(f"    Sample: {tess_result['lines'][0]['text'][:80]}...")
    else:
        print("    Tesseract not available")
    
    # Test 3: PaddleOCR
    print("\n[3] Testing PaddleOCR...")
    from services.paddle_ocr import get_paddle_ocr
    paddle = get_paddle_ocr(use_gpu=True)
    paddle_result = paddle.extract_text(rotated_path)
    print(f"    Lines: {paddle_result['line_count']}")
    print(f"    Confidence: {paddle_result['avg_confidence']:.2f}")
    if paddle_result['line_count'] > 0:
        print(f"    Sample: {paddle_result['lines'][0]['text'][:80]}...")
    
    # Test 4: Image GST Parser
    print("\n[4] Testing image_gst_parser (format detection + parsing)...")
    from services.image_gst_parser import parse_ocr_text_to_records, _detect_document_format
    
    # Use whichever OCR got more text
    best_ocr = tess_result if tess.available and tess_result['line_count'] > paddle_result['line_count'] else paddle_result
    print(f"    Using OCR engine: {'Tesseract' if best_ocr == tess_result else 'PaddleOCR'}")
    
    doc_format = _detect_document_format(best_ocr['full_text'])
    print(f"    Detected format: {doc_format}")
    
    records = parse_ocr_text_to_records(best_ocr['full_text'], best_ocr.get('lines', []))
    print(f"    Records extracted: {len(records)}")
    
    if records:
        for i, rec in enumerate(records[:3]):
            gstin = rec.get('GSTIN', '')
            inv = rec.get('INVOICE_NO', '')
            val = rec.get('INVOICE_VALUE', '')
            date = rec.get('INVOICE_DATE', '')
            print(f"    Record {i+1}: GSTIN={gstin[:15]}, INV={inv[:15]}, VAL={val}, DATE={date}")
    
    # Test 5: Full pipeline
    print(f"\n[5] Testing full pipeline on {os.path.basename(test_file)}...")
    from services.pipeline import process_file
    
    start = time.time()
    result = await process_file(test_file, "test-job-001")
    elapsed = time.time() - start
    
    print(f"    Success: {result.get('success')}")
    print(f"    Records: {result.get('records_extracted', 0)}")
    print(f"    Method: {result.get('method_used', 'unknown')}")
    print(f"    Confidence: {result.get('confidence', 0):.2f}")
    print(f"    Time: {elapsed:.1f}s")
    
    if result.get('excel_path'):
        print(f"    Excel: {os.path.basename(result['excel_path'])}")
    
    if result.get('error'):
        print(f"    ERROR: {result['error']}")
    
    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    asyncio.run(test())
