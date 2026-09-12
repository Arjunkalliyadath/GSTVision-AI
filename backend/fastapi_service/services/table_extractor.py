# table_extractor.py — Paddle Table Module (FIXED for v3.x)
# FIX: PPStructure removed from PaddleOCR v3.x → catch ImportError, return []

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.table_extractor")


class PaddleTableExtractor:
    """Paddle PPStructure table recognition — gracefully degrades if unavailable."""

    def __init__(self, use_gpu: bool = True):
        self.use_gpu      = use_gpu
        self.table_engine = None
        self._loaded      = False
        self._unavailable = False   # set True if PPStructure not available in this version

    def load(self):
        if self._loaded or self._unavailable:
            return
        logger.info("Loading Paddle Table Recognition Model...")

        # ── Try ImportError first (PPStructure removed in v3.x) ──
        try:
            from paddleocr import PPStructure
        except ImportError:
            logger.warning("PPStructure not available in this PaddleOCR version (v3.x). Table extraction disabled.")
            self._unavailable = True
            return

        # ── PPStructure exists (v2.x) — try v3.x API first ──
        try:
            device = "gpu:0" if self.use_gpu else "cpu"
            self.table_engine = PPStructure(
                table=True, ocr=True, show_log=False, device=device, lang="en"
            )
            self._loaded = True
            logger.info("Paddle Table Module loaded (v3.x API)!")
            return
        except TypeError:
            pass
        except Exception as e:
            logger.error(f"PPStructure v3 load failed: {e}")
            self._unavailable = True
            return

        # ── Fallback to v2.x API ──
        try:
            from paddleocr import PPStructure
            self.table_engine = PPStructure(
                table=True, ocr=True, show_log=False,
                use_gpu=self.use_gpu, lang="en"
            )
            self._loaded = True
            logger.info("Paddle Table Module loaded (v2.x API)!")
        except Exception as e:
            logger.error(f"PPStructure v2 load also failed: {e}")
            self._unavailable = True

    def extract_tables(self, image_path: str) -> List[Dict[str, Any]]:
        if self._unavailable:
            return []
        if not self._loaded:
            self.load()
        if not self._loaded:
            logger.warning("Table extractor unavailable, returning empty.")
            return []
        try:
            from PIL import Image
            import numpy as np
            img       = Image.open(image_path)
            img_array = np.array(img)
            result    = self.table_engine(img_array)
            tables    = []
            for i, region in enumerate(result):
                if region.get("type") == "table":
                    html = region.get("res", {}).get("html", "")
                    rows = self._parse_html_table(html)
                    tables.append({
                        "table_index": i,
                        "html":        html,
                        "rows":        rows,
                        "bbox":        region.get("bbox", [])
                    })
                    logger.info(f"Extracted table {i} with {len(rows)} rows")
            logger.info(f"Total tables found: {len(tables)}")
            return tables
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []

    def _parse_html_table(self, html: str) -> List[Dict]:
        try:
            from html.parser import HTMLParser

            class TableParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.rows         = []
                    self.current_row  = []
                    self.current_cell = ""
                    self.in_cell      = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("td", "th"):
                        self.in_cell      = True
                        self.current_cell = ""
                    elif tag == "tr":
                        self.current_row = []

                def handle_endtag(self, tag):
                    if tag in ("td", "th"):
                        self.current_row.append(self.current_cell.strip())
                        self.in_cell = False
                    elif tag == "tr" and self.current_row:
                        self.rows.append({"cells": self.current_row})

                def handle_data(self, data):
                    if self.in_cell:
                        self.current_cell += data

            parser = TableParser()
            parser.feed(html)
            return parser.rows
        except Exception as e:
            logger.error(f"HTML table parsing failed: {e}")
            return []


_table_extractor_instance = None

def get_table_extractor(use_gpu: bool = True) -> PaddleTableExtractor:
    global _table_extractor_instance
    if _table_extractor_instance is None:
        _table_extractor_instance = PaddleTableExtractor(use_gpu=use_gpu)
    return _table_extractor_instance