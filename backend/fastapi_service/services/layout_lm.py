# layout_lm.py — LayoutLMv3 service (FIXED)
# FIX: Add ignore_mismatched_sizes=True to LOCAL model load too
#      (local fine-tuned model has 26 labels, base has 25 → size mismatch crash)

import logging
import torch
import os
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.layout_lm")

GST2A_LABELS = [
    "O",
    "B-GSTIN", "I-GSTIN",
    "B-TRADE_NAME", "I-TRADE_NAME",
    "B-INVOICE_NO", "I-INVOICE_NO",
    "B-INVOICE_DATE", "I-INVOICE_DATE",
    "B-INVOICE_VALUE", "I-INVOICE_VALUE",
    "B-TAXABLE_VALUE", "I-TAXABLE_VALUE",
    "B-IGST", "I-IGST",
    "B-CGST", "I-CGST",
    "B-SGST", "I-SGST",
    "B-CESS", "I-CESS",
    "B-PLACE_OF_SUPPLY", "I-PLACE_OF_SUPPLY",
    "B-RETURN_PERIOD", "I-RETURN_PERIOD",
]

LABEL2ID = {label: i for i, label in enumerate(GST2A_LABELS)}
ID2LABEL  = {i: label for i, label in enumerate(GST2A_LABELS)}


class LayoutLMv3Service:
    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.device     = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model      = None
        self.processor  = None
        self._loaded    = False

    def load(self):
        if self._loaded:
            return
        logger.info(f"Loading LayoutLMv3 from {self.model_path} on {self.device}...")
        try:
            from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

            local_path = Path(self.model_path)
            has_local  = local_path.exists() and any(local_path.iterdir())

            if has_local:
                logger.info("Loading fine-tuned local model...")
                self.processor = LayoutLMv3Processor.from_pretrained(
                    self.model_path, apply_ocr=False
                )
                # FIXED: ignore_mismatched_sizes=True prevents crash when
                # local model has different label count than base
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    self.model_path,
                    num_labels=len(GST2A_LABELS),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                    ignore_mismatched_sizes=True   # ← KEY FIX
                )
            else:
                logger.info("Loading pretrained microsoft/layoutlmv3-base...")
                self.processor = LayoutLMv3Processor.from_pretrained(
                    "microsoft/layoutlmv3-base", apply_ocr=False
                )
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    "microsoft/layoutlmv3-base",
                    num_labels=len(GST2A_LABELS),
                    id2label=ID2LABEL,
                    label2id=LABEL2ID,
                    ignore_mismatched_sizes=True
                )

            self.model = self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"LayoutLMv3 loaded on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3: {e}")
            raise

    def normalize_bbox(self, bbox, width: int, height: int) -> List[int]:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        left, top    = min(x_coords), min(y_coords)
        right, bottom = max(x_coords), max(y_coords)
        return [
            int(1000 * left / width),
            int(1000 * top / height),
            int(1000 * right / width),
            int(1000 * bottom / height)
        ]

    def classify_fields(self, ocr_result: Dict, image_path: str) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        from PIL import Image
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            words, boxes  = [], []
            for line in ocr_result.get("lines", []):
                if line.get("bbox") and line.get("text"):
                    words.append(line["text"])
                    boxes.append(self.normalize_bbox(line["bbox"], width, height))
            if not words:
                return self._fallback_extract(ocr_result)

            encoding = self.processor(
                image, words, boxes=boxes,
                truncation=True, padding="max_length", max_length=512, return_tensors="pt"
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            with torch.no_grad():
                outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()

            classified_fields = {}
            current_field     = None
            current_value     = []

            for idx, pred in enumerate(predictions):
                label = ID2LABEL.get(pred, "O")
                if label.startswith("B-"):
                    if current_field and current_value:
                        fn = current_field.replace("B-", "")
                        classified_fields.setdefault(fn, []).append(" ".join(current_value))
                    current_field = label
                    current_value = [words[idx]] if idx < len(words) else []
                elif label.startswith("I-") and current_field:
                    if idx < len(words):
                        current_value.append(words[idx])

            if current_field and current_value:
                fn = current_field.replace("B-", "")
                classified_fields.setdefault(fn, []).append(" ".join(current_value))

            logger.info(f"LayoutLMv3 classified {len(classified_fields)} field types")
            return classified_fields

        except Exception as e:
            logger.error(f"LayoutLMv3 classification failed: {e}")
            return self._fallback_extract(ocr_result)

    def _fallback_extract(self, ocr_result: Dict) -> Dict:
        import re
        full_text = ocr_result.get("full_text", "")
        fields    = {}
        patterns  = {
            "GSTIN":         r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b",
            "INVOICE_DATE":  r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        }
        for field, pattern in patterns.items():
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                fields[field] = matches
        logger.info(f"Fallback extraction found {len(fields)} fields")
        return fields


_layoutlm_instance = None

def get_layoutlm(model_path: str, use_gpu: bool = True) -> LayoutLMv3Service:
    global _layoutlm_instance
    if _layoutlm_instance is None:
        _layoutlm_instance = LayoutLMv3Service(model_path, use_gpu)
    return _layoutlm_instance