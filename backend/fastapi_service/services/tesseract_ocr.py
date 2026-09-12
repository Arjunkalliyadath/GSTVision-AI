# tesseract_ocr.py — Tesseract OCR Engine (Industry-Grade Secondary OCR)
# ─────────────────────────────────────────────────────────────────────────────
# Provides a secondary OCR engine to complement PaddleOCR.
# Tesseract excels at:
#   - Mixed alphanumeric strings (GSTINs, invoice numbers)
#   - Documents where PaddleOCR confidence is low
#   - Orientation-agnostic text extraction
#
# Returns the same format as PaddleOCR service:
#   { lines, full_text, avg_confidence, line_count, image_path }
# ─────────────────────────────────────────────────────────────────────────────

import logging
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.tesseract_ocr")

# Tesseract binary path (Windows default)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractOCRService:
    """Tesseract OCR wrapper — returns same format as PaddleOCRService."""

    def __init__(self):
        self._available = False
        self._load()

    def _load(self):
        try:
            import pytesseract
            from pathlib import Path as P

            # Set path on Windows
            if P(TESSERACT_CMD).exists():
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

            # Quick test
            version = pytesseract.get_tesseract_version()
            self._available = True
            logger.info(f"Tesseract OCR loaded (version {version})")
        except Exception as exc:
            logger.warning(f"Tesseract OCR not available: {exc}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from image using Tesseract.
        Returns same format as PaddleOCRService.extract_text().
        """
        if not self._available:
            return {
                "lines": [], "full_text": "", "avg_confidence": 0.0,
                "line_count": 0, "image_path": image_path,
                "source": "tesseract_unavailable",
            }

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)

            # Run OCR with detailed output (includes confidence per word)
            data = pytesseract.image_to_data(
                img,
                lang="eng",
                config="--oem 3 --psm 6",
                output_type=pytesseract.Output.DICT,
            )

            lines, confs = self._parse_tesseract_data(data)
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            full_text = "\n".join(l["text"] for l in lines)

            return {
                "lines": lines,
                "full_text": full_text,
                "avg_confidence": avg_conf / 100.0,  # Tesseract returns 0-100
                "line_count": len(lines),
                "image_path": image_path,
                "source": "tesseract",
            }

        except Exception as exc:
            logger.warning(f"Tesseract extraction failed for {Path(image_path).name}: {exc}")
            return {
                "lines": [], "full_text": "", "avg_confidence": 0.0,
                "line_count": 0, "image_path": image_path,
                "source": "tesseract_error",
            }

    def extract_text_with_boxes(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text with bounding boxes — useful for table structure detection.
        """
        if not self._available:
            return self.extract_text(image_path)

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(image_path)

            data = pytesseract.image_to_data(
                img,
                lang="eng",
                config="--oem 3 --psm 6",
                output_type=pytesseract.Output.DICT,
            )

            lines_with_boxes = []
            confs = []

            n_boxes = len(data["text"])
            current_line_num = -1
            current_line_text = ""
            current_line_conf = []
            current_bbox = None

            for i in range(n_boxes):
                text = data["text"][i].strip()
                conf = int(data["conf"][i])
                line_num = data["line_num"][i]

                if not text or conf < 0:
                    continue

                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

                if line_num != current_line_num:
                    # Save previous line
                    if current_line_text.strip():
                        avg_line_conf = sum(current_line_conf) / len(current_line_conf) if current_line_conf else 0
                        lines_with_boxes.append({
                            "text": current_line_text.strip(),
                            "confidence": avg_line_conf / 100.0,
                            "bbox": current_bbox,
                        })
                        confs.append(avg_line_conf)

                    current_line_num = line_num
                    current_line_text = text
                    current_line_conf = [conf]
                    current_bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                else:
                    current_line_text += " " + text
                    current_line_conf.append(conf)
                    # Expand bbox
                    if current_bbox:
                        current_bbox[1][0] = max(current_bbox[1][0], x + w)
                        current_bbox[2][0] = max(current_bbox[2][0], x + w)
                        current_bbox[2][1] = max(current_bbox[2][1], y + h)
                        current_bbox[3][1] = max(current_bbox[3][1], y + h)

            # Save last line
            if current_line_text.strip():
                avg_line_conf = sum(current_line_conf) / len(current_line_conf) if current_line_conf else 0
                lines_with_boxes.append({
                    "text": current_line_text.strip(),
                    "confidence": avg_line_conf / 100.0,
                    "bbox": current_bbox,
                })
                confs.append(avg_line_conf)

            avg_conf = sum(confs) / len(confs) if confs else 0.0
            full_text = "\n".join(l["text"] for l in lines_with_boxes)

            return {
                "lines": lines_with_boxes,
                "full_text": full_text,
                "avg_confidence": avg_conf / 100.0,
                "line_count": len(lines_with_boxes),
                "image_path": image_path,
                "source": "tesseract",
            }

        except Exception as exc:
            logger.warning(f"Tesseract box extraction failed: {exc}")
            return self.extract_text(image_path)

    def _parse_tesseract_data(self, data: dict) -> tuple:
        """Parse Tesseract word-level output into line-level output."""
        lines: List[Dict] = []
        confs: List[float] = []

        n_boxes = len(data["text"])
        current_line = -1
        current_text = ""
        current_confs = []

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            line_num = data["line_num"][i]

            if not text or conf < 0:
                continue

            if line_num != current_line:
                if current_text.strip():
                    avg = sum(current_confs) / len(current_confs) if current_confs else 0
                    lines.append({"text": current_text.strip(), "confidence": avg / 100.0})
                    confs.append(avg)
                current_line = line_num
                current_text = text
                current_confs = [conf]
            else:
                current_text += " " + text
                current_confs.append(conf)

        # Last line
        if current_text.strip():
            avg = sum(current_confs) / len(current_confs) if current_confs else 0
            lines.append({"text": current_text.strip(), "confidence": avg / 100.0})
            confs.append(avg)

        return lines, confs


# ── Singleton ──────────────────────────────────────────────────────────────────
_tesseract_instance: Optional[TesseractOCRService] = None


def get_tesseract_ocr() -> TesseractOCRService:
    """Return the module-level singleton TesseractOCRService."""
    global _tesseract_instance
    if _tesseract_instance is None:
        _tesseract_instance = TesseractOCRService()
    return _tesseract_instance
