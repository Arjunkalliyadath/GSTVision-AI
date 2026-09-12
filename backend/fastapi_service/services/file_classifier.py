# file_classifier.py — Detect file type for pipeline routing
# ─────────────────────────────────────────────────────────────────────────────
# Routes files to the correct pipeline:
#   "digital_pdf"  → PATH 1 (pdfplumber direct extraction — existing pipeline)
#   "photo_pdf"    → PATH 2 (image OCR pipeline — new dedicated pipeline)
#   "image"        → PATH 2 (image OCR pipeline — new dedicated pipeline)
#
# Detection logic for PDFs:
#   1. Open with pdfplumber
#   2. Count extractable text characters per page
#   3. If avg chars/page > threshold (100) → digital PDF
#   4. Otherwise → photo/scanned PDF
# ─────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger("gst2_fastapi.file_classifier")

FileType = Literal["digital_pdf", "photo_pdf", "image"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".heic", ".heif",
    ".bmp", ".tiff", ".tif", ".webp", ".gif",
}

# Minimum average characters per page to consider a PDF as "digital"
# Digital GSTR-2A PDFs typically have 500-5000 chars/page
# Scanned/photo PDFs have 0-50 chars/page (from embedded metadata only)
DIGITAL_PDF_CHARS_THRESHOLD = 100


def classify_file(file_path: str) -> FileType:
    """
    Classify a file to determine which extraction pipeline to use.
    
    Returns:
        "digital_pdf"  — text-layer PDF (use pdfplumber direct extraction)
        "photo_pdf"    — scanned/photo PDF (use image OCR pipeline)
        "image"        — JPG/PNG/etc (use image OCR pipeline)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # ── Image files: always route to image pipeline ───────────────────────
    if ext in IMAGE_EXTENSIONS:
        logger.info(f"[Classifier] {path.name} → IMAGE (ext={ext})")
        return "image"

    # ── PDF files: check for text layer ───────────────────────────────────
    if ext == ".pdf":
        return _classify_pdf(path)

    # ── Unknown: treat as image ───────────────────────────────────────────
    logger.warning(f"[Classifier] Unknown ext '{ext}' for {path.name} — treating as image")
    return "image"


def _classify_pdf(path: Path) -> FileType:
    """
    Determine if a PDF is digital (has text layer) or scanned/photo.
    
    Strategy:
    - Sample first 3 pages
    - Extract text with pdfplumber
    - If average character count per page exceeds threshold → digital
    - Otherwise → photo/scanned
    """
    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                logger.warning(f"[Classifier] {path.name} has 0 pages — treating as photo_pdf")
                return "photo_pdf"

            sample_pages = min(3, total_pages)
            total_chars = 0

            for i in range(sample_pages):
                text = pdf.pages[i].extract_text(x_tolerance=3, y_tolerance=3) or ""
                char_count = len(text.strip())
                total_chars += char_count
                logger.debug(f"[Classifier] Page {i+1}: {char_count} chars")

            avg_chars = total_chars / sample_pages

            if avg_chars >= DIGITAL_PDF_CHARS_THRESHOLD:
                logger.info(
                    f"[Classifier] {path.name} → DIGITAL_PDF "
                    f"(avg {avg_chars:.0f} chars/page across {sample_pages} pages)"
                )
                return "digital_pdf"
            else:
                logger.info(
                    f"[Classifier] {path.name} → PHOTO_PDF "
                    f"(avg {avg_chars:.0f} chars/page — below threshold {DIGITAL_PDF_CHARS_THRESHOLD})"
                )
                return "photo_pdf"

    except ImportError:
        logger.error("[Classifier] pdfplumber not installed — defaulting to photo_pdf")
        return "photo_pdf"
    except Exception as exc:
        logger.warning(f"[Classifier] PDF analysis failed for {path.name}: {exc} — defaulting to photo_pdf")
        return "photo_pdf"
