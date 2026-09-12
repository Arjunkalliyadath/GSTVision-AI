# gst2a_pdf_parser.py (v6.0 — FULLY FIXED)
# ─────────────────────────────────────────────────────────────────────────────
# Supports:
#  FORMAT A  — Split-group (invoice/tax/IRN on separate page groups)
#  FORMAT B  — Wide single-row (all columns on every page) — gst_2a_2
#  FORMAT C  — ITC Purchase Register — gst_2a_5
#  FORMAT D  — Interleaved chars (columns merged by pdfplumber) — gst_2a_4
#
# FIXES v6.0:
#  1. AMOUNT_RE — Supports Indian number format (1,23,456.78) by allowing
#     2-3 digit comma groups. Reordered alternatives so decimal-first wins.
#  2. INV_NOFMT — Post-processing strips embedded dates (INV-55962R14-02-2024
#     → INV-55962, date=14-02-2024) and trailing invoice-type codes (R, DE).
#  3. _parse_group_a_line — Recovers date from embedded invoice-number string;
#     skips leading small integers (row numbers) when looking for invoice value.
#  4. _parse_format_b_line — Strips type-code suffix from invoice number;
#     skips serial/row numbers in "before" section that masquerade as INVOICE_NO.
#  5. records_to_excel_format — Smart INVOICE_VALUE recalculation: when the
#     extracted value is clearly a row-number (< 1% of tax components sum),
#     it is replaced with taxable + IGST + CGST + SGST + CESS.
#  6. Header/summary line detection — Lines that are column headers or totals
#     are silently skipped in all format parsers.
#  7. Place-of-Supply fallback — derive from GSTIN state code when not found.
# ─────────────────────────────────────────────────────────────────────────────

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("gst2_fastapi.gst2a_parser")

# ── Shared regexes ────────────────────────────────────────────────────────────
GROUP_A_MARKERS = ["gstn", "gstin", "invoice number", "invoice no", "invoice date"]
GROUP_B_MARKERS = ["taxable value", "integrated tax", "central tax", "state/ut tax",
                   "gstr-1", "gstr1", "filing status", "filing date"]
GROUP_C_MARKERS = ["amendment", "irn", "irn date", "source", "e-invoice", "cancellation"]
ITC_MARKERS     = ["itc purchase register", "purchase register", "supplier details",
                   "taxable tax breakdown", "total transaction value"]

HEADER_KEYWORDS = [
    "gstin of supplier", "trade name", "invoice number", "invoice no",
    "invoice date", "invoice value", "place of supply", "reverse charge",
    "tax rate", "taxable value", "integrated tax", "central tax",
    "state/ut tax", "cess", "gstr-1", "gstr1", "filing status",
    "amendment", "irn", "sl.no", "sr.no", "s.no", "sl no",
    "total", "grand total", "sub total",
]

POS_TOKENS = [
    "Tamil Nadu","Andhra Pradesh","Arunachal Pradesh","Himachal Pradesh",
    "Jammu and Kashmir","Madhya Pradesh","Uttar Pradesh","West Bengal",
    "Karnataka","Maharashtra","Rajasthan","Uttarakhand","Chhattisgarh",
    "Jharkhand","Kerala","Gujarat","Haryana","Punjab","Bihar","Odisha",
    "Assam","Telangana","Goa","Tripura","Meghalaya","Manipur",
    "Mizoram","Nagaland","Sikkim","Puducherry","Chandigarh",
    "Lakshadweep","Andaman","Ladakh","Delhi",
]
POS_PATTERN = re.compile(
    r'(' + '|'.join(re.escape(p) for p in sorted(POS_TOKENS, key=len, reverse=True)) + r')',
    re.IGNORECASE
)

# State code → name (for fallback POS from GSTIN)
STATE_CODES = {
    "01":"Jammu and Kashmir","02":"Himachal Pradesh","03":"Punjab",
    "04":"Chandigarh","05":"Uttarakhand","06":"Haryana","07":"Delhi",
    "08":"Rajasthan","09":"Uttar Pradesh","10":"Bihar","11":"Sikkim",
    "12":"Arunachal Pradesh","13":"Nagaland","14":"Manipur","15":"Mizoram",
    "16":"Tripura","17":"Meghalaya","18":"Assam","19":"West Bengal",
    "20":"Jharkhand","21":"Odisha","22":"Chhattisgarh","23":"Madhya Pradesh",
    "24":"Gujarat","27":"Maharashtra","28":"Andhra Pradesh","29":"Karnataka",
    "30":"Goa","31":"Lakshadweep","32":"Kerala","33":"Tamil Nadu",
    "34":"Puducherry","35":"Andaman","36":"Telangana","37":"Andhra Pradesh",
    "38":"Ladakh",
}

IRN_RE    = re.compile(r'\b([0-9a-f]{64})\b', re.IGNORECASE)
DATE_RE   = re.compile(r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b')
DATE_NOBOUND = re.compile(r'(\d{2}[-/]\d{2}[-/]\d{4})')  # for embedded-date extraction
MON_DATE  = re.compile(r'\b(\d{2}[-/][A-Za-z]{3}[-/]\d{2,4})\b')
PERIOD    = re.compile(r'\b([A-Za-z]{3}[-/]\d{2,4})\b')

# FIX #1: Indian number format support.
# Order matters: try decimal-number first, then comma-grouped, then plain int.
AMOUNT_RE = re.compile(
    r'(\d+\.\d{1,2}'           # 12345.67  or  100000.00
    r'|\d{1,3}(?:,\d{2,3})+'   # 1,23,456  or  1,234,567  (requires ≥1 comma group)
    r'(?:\.\d{1,2})?'          #   with optional decimal
    r'|\d+)'                    # plain integer fallback
)

GSTIN_STRICT = re.compile(r'\b(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d])\b')
GSTIN_RELAX  = re.compile(r'(\d{2}[A-Z0-9]{10,13})')
RATE_RE   = re.compile(r'\b(0|5|12|18|28)\b')
TYPE_RE   = re.compile(r'\b(R|DE|SEWP|SEZWP|SEZWOP|CBW|RE)\b')
SOURCE_RE = re.compile(r'\b(E-Invoice|E-invoice|GSTR-1A|GSTR-1|GSTR-5|IFF|Manual)\b', re.IGNORECASE)
INV_NOFMT = re.compile(r'(INV[-/][A-Z0-9-]+)', re.IGNORECASE)

# Trailing type-code stripper for invoice numbers
_INV_TYPE_SUFFIX = re.compile(r'(?i)(?<=[A-Z0-9])(R|DE|SEWP|SEZWP|SEZWOP|CBW|RE)$')
# Header line detection (returns True for lines that are column headers/totals)
_HEADER_RE = re.compile(
    r'(?i)\b(' + '|'.join(re.escape(k) for k in HEADER_KEYWORDS) + r')\b'
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_header_or_summary_line(line: str) -> bool:
    """True for column-header rows, grand-total rows, page-number lines, etc."""
    lw = line.lower().strip()
    if not lw:
        return True
    # Column headers have multiple GST keywords
    kw_count = sum(1 for k in HEADER_KEYWORDS if k in lw)
    if kw_count >= 2:
        return True
    # Standalone "total" or "page N" lines
    if re.match(r'^(total|grand total|sub total|page\s+\d+|sl\.?\s*no\.?)$', lw):
        return True
    return False


def _clean_invoice_no(raw: str, existing_date: str = "") -> tuple:
    """
    FIX #2: Clean invoice number extracted by INV_NOFMT.
    - Extracts embedded date (INV-55962R14-02-2024... → date=14-02-2024)
    - Strips trailing invoice-type code (INV-44734R → INV-44734)
    Returns (cleaned_invoice_no, extracted_date_or_empty)
    """
    extracted_date = existing_date
    # Extract embedded date (no word-boundary needed inside invoice string)
    dm = DATE_NOBOUND.search(raw)
    if dm:
        if not extracted_date:
            extracted_date = dm.group(1)
        raw = raw[:dm.start()]
    # Strip trailing type code
    raw = _INV_TYPE_SUFFIX.sub('', raw).rstrip('-/ ')
    return raw.strip().upper(), extracted_date


def _pos_from_gstin(gstin: str) -> str:
    """Derive Place of Supply from GSTIN state code when not found in text."""
    if gstin and len(gstin) >= 2:
        return STATE_CODES.get(gstin[:2], "")
    return ""


def _recalc_invoice_value(inv_val: Any, taxable: Any, igst: Any,
                          cgst: Any, sgst: Any, cess: Any) -> Any:
    """
    FIX #5: Recalculate invoice value from tax components when it looks
    like a row-number or serial-number instead of a real rupee amount.
    """
    def to_float(v):
        try:
            return float(str(v).replace(",", "") or 0)
        except (TypeError, ValueError):
            return 0.0

    inv     = to_float(inv_val)
    tax_sum = to_float(taxable) + to_float(igst) + to_float(cgst) + to_float(sgst) + to_float(cess)

    if tax_sum > 100 and (inv == 0 or (inv > 0 and inv * 100 < tax_sum)):
        # Inv value is clearly wrong (less than 1 % of tax-components sum)
        return round(tax_sum, 2)
    return inv_val   # original value looks fine


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_gst2a_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return []

    logger.info(f"Parsing GST 2A PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Total pages: {total_pages}")
            fmt = _detect_format(pdf)
            logger.info(f"Detected format: {fmt}")
            if fmt == "ITC":
                return _parse_format_itc(pdf)
            elif fmt == "B":
                return _parse_format_b(pdf)
            elif fmt == "A":
                return _parse_format_a(pdf)
            else:
                # Format D or unknown — try B on stream-order text
                return _parse_format_b_stream(pdf)
    except Exception as e:
        logger.error(f"GST 2A PDF parsing failed: {e}", exc_info=True)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_format(pdf) -> str:
    pages = list(pdf.pages)
    sample_pages = pages[:min(3, len(pages))]
    combined_text = ""
    for p in sample_pages:
        combined_text += (p.extract_text(x_tolerance=3, y_tolerance=3) or "").lower()

    if any(m in combined_text for m in ITC_MARKERS):
        return "ITC"

    has_a = any(m in combined_text for m in GROUP_A_MARKERS)
    has_b = any(m in combined_text for m in GROUP_B_MARKERS)

    if has_a and has_b:
        return "B"
    if len(pages) <= 2:
        return "B"
    return "A"


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT ITC — ITC Purchase Register
# ─────────────────────────────────────────────────────────────────────────────

def _parse_format_itc(pdf) -> List[Dict]:
    records = []
    for page in pdf.pages:
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        for line in text.splitlines():
            line = line.strip()
            if _is_header_or_summary_line(line):
                continue
            rec = _parse_itc_line(line)
            if rec:
                records.append(rec)
    logger.info(f"ITC Register: extracted {len(records)} records")
    return records


def _parse_itc_line(line: str) -> Optional[Dict]:
    rec = {
        "GSTIN": "", "TRADE_NAME": "", "INVOICE_NO": "", "INVOICE_TYPE": "R",
        "INVOICE_DATE": "", "INVOICE_VALUE": "", "PLACE_OF_SUPPLY": "",
        "REVERSE_CHARGE": "N", "TAX_RATE": "",
        "TAXABLE_VALUE": "", "IGST": "", "CGST": "", "SGST": "", "CESS": "",
        "GSTR1_STATUS": "", "GSTR1_FILING_DATE": "", "GSTR1_PERIOD": "", "GSTR3B_STATUS": "",
        "AMENDMENT": "N", "TAX_PERIOD_AMENDED": "N/A", "CANCEL_DATE": "N/A",
        "SOURCE": "ITC Register", "IRN": "N/A", "IRN_DATE": "N/A",
        "_confidence": 0.95, "_source": "direct_pdf_itc"
    }

    dm = DATE_RE.match(line.strip())
    if not dm:
        return None
    rec["INVOICE_DATE"] = dm.group(1)
    remainder = line[dm.end():].strip()

    parts = remainder.split()
    inv_idx = -1
    for i, p in enumerate(parts):
        cleaned = p.replace(",", "")
        if re.match(r'^\d{3,8}$', cleaned):
            rec["INVOICE_NO"] = cleaned
            inv_idx = i
            break
    if inv_idx < 0:
        return None

    after_inv = " ".join(parts[inv_idx + 1:])
    gstin_m = GSTIN_STRICT.search(after_inv) or GSTIN_RELAX.search(after_inv)
    if not gstin_m:
        return None
    rec["GSTIN"] = gstin_m.group(1)
    rec["TRADE_NAME"] = after_inv[:gstin_m.start()].strip()

    after_gstin = after_inv[gstin_m.end():].strip()
    amounts = re.findall(r'[\d,]+\.\d{2}', after_gstin)
    amounts_clean = [a.replace(",", "") for a in amounts]

    def safe(lst, i): return lst[i] if i < len(lst) else ""

    rec["TAXABLE_VALUE"] = safe(amounts_clean, 0)
    rec["IGST"]          = safe(amounts_clean, 1)
    rec["CGST"]          = safe(amounts_clean, 2)
    rec["SGST"]          = safe(amounts_clean, 3)
    rec["CESS"]          = safe(amounts_clean, 4)
    rec["INVOICE_VALUE"] = safe(amounts_clean, 5)

    # Derive POS from GSTIN state code
    if not rec["PLACE_OF_SUPPLY"] and rec["GSTIN"]:
        rec["PLACE_OF_SUPPLY"] = _pos_from_gstin(rec["GSTIN"])

    if rec["GSTIN"] and rec["INVOICE_NO"]:
        return rec
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT A — SPLIT-GROUP PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_stream_order(page) -> str:
    chars = page.chars
    if not chars:
        return ""
    rows: Dict[int, List[str]] = {}
    for c in chars:
        y_key = round(c["top"] / 3) * 3
        rows.setdefault(y_key, []).append(c["text"])
    return "\n".join("".join(rows[y]) for y in sorted(rows.keys()))


def _parse_format_a(pdf) -> List[Dict]:
    pages_sorted = []
    pages_stream = []
    for page in pdf.pages:
        pages_sorted.append(page.extract_text(x_tolerance=3, y_tolerance=3) or "")
        pages_stream.append(_extract_text_stream_order(page))

    n = len(pages_sorted)
    first_a = first_b = first_c = None
    for i, text in enumerate(pages_sorted):
        lower = text.lower()
        if first_a is None and any(m in lower for m in GROUP_A_MARKERS):
            first_a = i
        if first_b is None and any(m in lower for m in GROUP_B_MARKERS):
            if first_a is not None and i > first_a:
                first_b = i
        if first_c is None and any(m in lower for m in GROUP_C_MARKERS):
            if first_b is not None and i > first_b:
                first_c = i

    first_a = first_a if first_a is not None else 0
    first_b = first_b if first_b is not None else max(1, n // 3)
    first_c = first_c if first_c is not None else max(2, (2 * n) // 3)
    first_b = min(first_b, n)
    first_c = min(first_c, n)

    group_a = list(range(first_a, first_b))
    group_b = list(range(first_b, first_c))
    group_c = list(range(first_c, n))

    logger.info(f"Format A groups — A:{group_a} B:{group_b} C:{group_c}")

    records_a = _parse_group_a(pages_stream, group_a, n)
    records_b = _parse_group_b(pages_sorted, group_b, n)
    records_c = _parse_group_c(pages_sorted, group_c, n)

    logger.info(f"Parsed rows — A:{len(records_a)} B:{len(records_b)} C:{len(records_c)}")
    return _merge_groups(records_a, records_b, records_c)


def _parse_group_a(pages_text, page_indices, total_pages):
    records = []
    for pi in page_indices:
        if pi >= total_pages:
            continue
        for line in pages_text[pi].splitlines():
            line = line.strip()
            if not line or _is_header_or_summary_line(line):
                continue
            gm = GSTIN_STRICT.search(line) or GSTIN_RELAX.search(line)
            if not gm:
                continue
            rec = _parse_group_a_line(line, gm)
            if rec:
                records.append(rec)
    return records


def _parse_group_a_line(line: str, gstin_match) -> Optional[Dict]:
    rec = {
        "GSTIN": gstin_match.group(1), "TRADE_NAME": "", "INVOICE_NO": "", "INVOICE_TYPE": "R",
        "INVOICE_DATE": "", "INVOICE_VALUE": "", "PLACE_OF_SUPPLY": "",
        "REVERSE_CHARGE": "N", "TAX_RATE": "", "_confidence": 0.98, "_source": "direct_pdf"
    }

    before_gstin = line[:gstin_match.start()].strip()
    after_gstin  = line[gstin_match.end():].strip()

    # FIX #2/#3: Clean invoice number and extract embedded date
    inv_m = INV_NOFMT.search(line)
    if inv_m:
        raw_inv = inv_m.group(1)
        cleaned_inv, extracted_date = _clean_invoice_no(raw_inv)
        rec["INVOICE_NO"] = cleaned_inv
        if extracted_date:
            rec["INVOICE_DATE"] = extracted_date
        name_part = line[:inv_m.start()].replace(rec["GSTIN"], "").strip()
        rec["TRADE_NAME"] = re.sub(GSTIN_RELAX, "", name_part).strip()
    else:
        parts = after_gstin.split()
        for p in parts:
            if re.match(r'^\d{3,8}$', p):
                rec["INVOICE_NO"] = p
                break
        rec["TRADE_NAME"] = before_gstin

    remainder = after_gstin

    # Invoice type
    tm = TYPE_RE.search(remainder)
    if tm:
        rec["INVOICE_TYPE"] = tm.group(1)

    # Date — use existing extracted date from invoice number if available
    if not rec["INVOICE_DATE"]:
        dm = DATE_RE.search(remainder)
        if dm:
            rec["INVOICE_DATE"] = dm.group(1)
            remainder = remainder[dm.end():]
    else:
        # Skip past any standalone date in remainder
        dm = DATE_RE.search(remainder)
        if dm:
            remainder = remainder[dm.end():]

    # FIX #3: Skip small leading integers (row numbers) before the real invoice value
    real_amount = None
    search_str = remainder
    am = AMOUNT_RE.search(search_str)
    while am:
        val_str = am.group(1).replace(",", "")
        try:
            val = float(val_str)
            if val >= 100:          # Real amounts are ≥ ₹100; row numbers are tiny
                real_amount = am.group(1)
                remainder = search_str[am.end():]
                break
        except ValueError:
            pass
        search_str = search_str[am.end():]
        am = AMOUNT_RE.search(search_str)
    if real_amount:
        rec["INVOICE_VALUE"] = real_amount.replace(",", "")

    # Place of supply
    pm = POS_PATTERN.search(remainder)
    if pm:
        rec["PLACE_OF_SUPPLY"] = pm.group(1)
        remainder = remainder[pm.end():]
    elif rec["GSTIN"]:
        rec["PLACE_OF_SUPPLY"] = _pos_from_gstin(rec["GSTIN"])

    # Reverse charge
    rc = re.search(r'\b([NY])\b', remainder)
    if rc:
        rec["REVERSE_CHARGE"] = rc.group(1)

    # Rate
    rm = RATE_RE.search(remainder)
    if rm:
        rec["TAX_RATE"] = rm.group(1)

    return rec if rec["GSTIN"] else None


def _parse_group_b(pages_text, page_indices, total_pages):
    records = []
    for pi in page_indices:
        if pi >= total_pages:
            continue
        for line in pages_text[pi].splitlines():
            line = line.strip()
            if not line or _is_header_or_summary_line(line):
                continue
            if not re.match(r'^\d', line):
                continue
            rec = _parse_group_b_line(line)
            if rec:
                records.append(rec)
    return records


def _parse_group_b_line(line: str) -> Optional[Dict]:
    tokens  = line.split()
    numbers = []
    rest_idx = 0
    for i, t in enumerate(tokens):
        if re.match(r'^[\d,.]+$', t):
            numbers.append(t.replace(",", ""))
            rest_idx = i + 1
        else:
            break
    if len(numbers) < 3:
        return None

    def safe(lst, i): return lst[i] if i < len(lst) else ""

    rec = {
        "TAXABLE_VALUE": safe(numbers, 0), "IGST": safe(numbers, 1),
        "CGST": safe(numbers, 2), "SGST": safe(numbers, 3), "CESS": safe(numbers, 4),
    }
    rest = " ".join(tokens[rest_idx:])
    g1 = re.search(r'\b([YN])\b', rest)
    rec["GSTR1_STATUS"] = g1.group(1) if g1 else ""
    if g1: rest = rest[g1.end():]
    dm = MON_DATE.search(rest)
    rec["GSTR1_FILING_DATE"] = dm.group(1) if dm else ""
    if dm: rest = rest[dm.end():]
    pm = PERIOD.search(rest)
    rec["GSTR1_PERIOD"] = pm.group(1) if pm else ""
    if pm: rest = rest[pm.end():]
    g3 = re.search(r'\b([YN])\b', rest)
    rec["GSTR3B_STATUS"] = g3.group(1) if g3 else ""
    return rec


def _parse_group_c(pages_text, page_indices, total_pages):
    records = []
    for pi in page_indices:
        if pi >= total_pages:
            continue
        for line in pages_text[pi].splitlines():
            line = line.strip()
            if not line or _is_header_or_summary_line(line):
                continue
            if not re.match(r'^[YN]\b', line, re.IGNORECASE):
                continue
            rec = _parse_group_c_line(line)
            if rec:
                records.append(rec)
    return records


def _parse_group_c_line(line: str) -> Optional[Dict]:
    tokens = line.split()
    if len(tokens) < 3:
        return None
    rec = {
        "AMENDMENT": tokens[0].upper(), "TAX_PERIOD_AMENDED": "N/A",
        "CANCEL_DATE": "N/A", "SOURCE": "Unknown", "IRN": "N/A", "IRN_DATE": "N/A"
    }
    if len(tokens) > 1 and tokens[1].upper() != "N/A":
        rec["TAX_PERIOD_AMENDED"] = tokens[1]
    if len(tokens) > 2 and tokens[2].upper() != "N/A":
        rec["CANCEL_DATE"] = tokens[2]
    src = SOURCE_RE.search(line)
    if src:
        rec["SOURCE"] = src.group(1)
    irn = IRN_RE.search(line)
    if irn:
        rec["IRN"] = irn.group(1)
        after = line[irn.end():].strip()
        dm = DATE_RE.search(after)
        if dm:
            rec["IRN_DATE"] = dm.group(1)
    return rec


def _merge_groups(records_a, records_b, records_c):
    merged   = []
    max_rows = max(len(records_a), len(records_b), len(records_c), 0)
    b_def    = {"TAXABLE_VALUE":"","IGST":"","CGST":"","SGST":"","CESS":"",
                "GSTR1_STATUS":"","GSTR1_FILING_DATE":"","GSTR1_PERIOD":"","GSTR3B_STATUS":""}
    c_def    = {"AMENDMENT":"N","TAX_PERIOD_AMENDED":"N/A","CANCEL_DATE":"N/A",
                "SOURCE":"Unknown","IRN":"N/A","IRN_DATE":"N/A"}
    for i in range(max_rows):
        a = records_a[i] if i < len(records_a) else {}
        b = records_b[i] if i < len(records_b) else b_def.copy()
        c = records_c[i] if i < len(records_c) else c_def.copy()
        record = {**a, **b, **c}
        record["_confidence"] = 0.98
        record["_source"]     = "direct_pdf"
        merged.append(record)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT B — WIDE SINGLE-ROW PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _parse_format_b(pdf) -> List[Dict]:
    records = []
    for page in pdf.pages:
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        for line in text.splitlines():
            line = line.strip()
            if not line or _is_header_or_summary_line(line):
                continue
            gm = GSTIN_STRICT.search(line) or GSTIN_RELAX.search(line)
            if not gm:
                continue
            rec = _parse_format_b_line(line, gm)
            if rec:
                records.append(rec)
    logger.info(f"Format B: extracted {len(records)} records")
    return records


def _parse_format_b_stream(pdf) -> List[Dict]:
    """Format D fallback: parse using stream-order (character-position) text."""
    records = []
    for page in pdf.pages:
        text = _extract_text_stream_order(page)
        for line in text.splitlines():
            line = line.strip()
            if not line or _is_header_or_summary_line(line):
                continue
            gm = GSTIN_STRICT.search(line) or GSTIN_RELAX.search(line)
            if not gm:
                continue
            rec = _parse_format_b_line(line, gm)
            if rec:
                records.append(rec)
    logger.info(f"Format B (stream-order): extracted {len(records)} records")
    return records


def _parse_format_b_line(line: str, gstin_match) -> Optional[Dict]:
    rec = {
        "GSTIN": gstin_match.group(1), "TRADE_NAME": "", "INVOICE_NO": "", "INVOICE_TYPE": "R",
        "INVOICE_DATE": "", "INVOICE_VALUE": "", "PLACE_OF_SUPPLY": "",
        "REVERSE_CHARGE": "N", "TAX_RATE": "",
        "TAXABLE_VALUE": "", "IGST": "", "CGST": "", "SGST": "", "CESS": "",
        "GSTR1_STATUS": "", "GSTR1_FILING_DATE": "", "GSTR1_PERIOD": "", "GSTR3B_STATUS": "",
        "AMENDMENT": "N", "TAX_PERIOD_AMENDED": "N/A", "CANCEL_DATE": "N/A",
        "SOURCE": "Unknown", "IRN": "N/A", "IRN_DATE": "N/A",
        "_confidence": 0.97, "_source": "direct_pdf"
    }

    before = line[:gstin_match.start()].strip()
    after  = line[gstin_match.end():].strip()

    # FIX #2: Clean invoice number — strip embedded dates and trailing type codes
    inv_m = INV_NOFMT.search(before + " " + after)
    if inv_m:
        raw_inv = inv_m.group(1)
        cleaned_inv, extracted_date = _clean_invoice_no(raw_inv)
        rec["INVOICE_NO"] = cleaned_inv
        if extracted_date:
            rec["INVOICE_DATE"] = extracted_date
        if inv_m.start() < len(before):
            rec["TRADE_NAME"] = re.sub(GSTIN_RELAX, "", before[:inv_m.start()]).strip()
        else:
            rec["TRADE_NAME"] = before
    else:
        # FIX #4: Skip leading serial/row numbers (1–6 digits at start of `before`)
        before_parts = before.split()
        skip = 0
        for p in before_parts:
            if re.match(r'^\d{1,6}$', p):
                skip += 1
            else:
                break
        meaningful_before = " ".join(before_parts[skip:])

        # Look for plain integer invoice number in meaningful_before
        found_inv = False
        for p in reversed(meaningful_before.split()):
            if re.match(r'^\d{3,8}$', p):
                rec["INVOICE_NO"] = p
                idx = meaningful_before.rfind(p)
                rec["TRADE_NAME"] = meaningful_before[:idx].strip()
                found_inv = True
                break
        if not found_inv:
            # Try after GSTIN
            for p in after.split():
                if re.match(r'^\d{3,8}$', p) and not re.match(r'^\d{4}$', p):
                    rec["INVOICE_NO"] = p
                    break
            rec["TRADE_NAME"] = meaningful_before

    remainder = after

    # Invoice type
    tm = TYPE_RE.search(remainder)
    if tm:
        rec["INVOICE_TYPE"] = tm.group(1)

    # Date
    if not rec["INVOICE_DATE"]:
        dm = DATE_RE.search(remainder)
        if dm:
            rec["INVOICE_DATE"] = dm.group(1)
            remainder = remainder[dm.end():].strip()
    else:
        dm = DATE_RE.search(remainder)
        if dm:
            remainder = remainder[dm.end():].strip()

    # FIX #3: Invoice value — skip small row-number-like integers
    real_amount = None
    search_str = remainder
    am = AMOUNT_RE.search(search_str)
    while am:
        val_str = am.group(1).replace(",", "")
        try:
            val = float(val_str)
            if val >= 100:
                real_amount = am.group(1)
                remainder = search_str[am.end():].strip()
                break
        except ValueError:
            pass
        search_str = search_str[am.end():]
        am = AMOUNT_RE.search(search_str)
    if real_amount:
        rec["INVOICE_VALUE"] = real_amount.replace(",", "")

    # Place of supply — mandatory for format B; fall back to GSTIN state code
    pm = POS_PATTERN.search(remainder)
    if pm:
        rec["PLACE_OF_SUPPLY"] = pm.group(1)
        remainder = remainder[pm.end():].strip()
    else:
        pos_fb = _pos_from_gstin(rec["GSTIN"])
        if pos_fb:
            rec["PLACE_OF_SUPPLY"] = pos_fb
        # Don't return None — allow records without explicit POS

    # Reverse charge
    rc = re.match(r'([NY])\s*', remainder)
    if rc:
        rec["REVERSE_CHARGE"] = rc.group(1)
        remainder = remainder[rc.end():].strip()

    # Tax rate
    rm = RATE_RE.match(remainder)
    if rm:
        rec["TAX_RATE"] = rm.group(1)
        remainder = remainder[rm.end():].strip()

    # Tax amounts: taxable, igst, cgst, sgst, cess
    tax_nums = []
    for _ in range(5):
        nm = AMOUNT_RE.match(remainder)
        if nm:
            tax_nums.append(nm.group(1).replace(",", ""))
            remainder = remainder[nm.end():].strip()
        else:
            break

    if len(tax_nums) >= 1:
        rec["TAXABLE_VALUE"] = tax_nums[0] if len(tax_nums) > 0 else ""
        rec["IGST"]          = tax_nums[1] if len(tax_nums) > 1 else ""
        rec["CGST"]          = tax_nums[2] if len(tax_nums) > 2 else ""
        rec["SGST"]          = tax_nums[3] if len(tax_nums) > 3 else ""
        rec["CESS"]          = tax_nums[4] if len(tax_nums) > 4 else ""

    # Filing status, dates, amendment, IRN …
    g1 = re.match(r'([YN])\s+', remainder)
    if g1:
        rec["GSTR1_STATUS"] = g1.group(1)
        remainder = remainder[g1.end():].strip()
    fmd = MON_DATE.match(remainder)
    if fmd:
        rec["GSTR1_FILING_DATE"] = fmd.group(1)
        remainder = remainder[fmd.end():].strip()
    fpm = PERIOD.match(remainder)
    if fpm:
        rec["GSTR1_PERIOD"] = fpm.group(1)
        remainder = remainder[fpm.end():].strip()
    g3 = re.match(r'([YN])\s*', remainder)
    if g3:
        rec["GSTR3B_STATUS"] = g3.group(1)
        remainder = remainder[g3.end():].strip()
    am_flag = re.match(r'([YN])\s+', remainder)
    if am_flag:
        rec["AMENDMENT"] = am_flag.group(1)
        remainder = remainder[am_flag.end():].strip()
        apm = PERIOD.match(remainder)
        if apm:
            rec["TAX_PERIOD_AMENDED"] = apm.group(1)
            remainder = remainder[apm.end():].strip()
    cd = DATE_RE.match(remainder)
    if cd:
        rec["CANCEL_DATE"] = cd.group(1)
        remainder = remainder[cd.end():].strip()
    src = SOURCE_RE.search(remainder)
    if src:
        rec["SOURCE"] = src.group(1)
        remainder = remainder[src.end():].strip()
    irn = IRN_RE.search(remainder)
    if irn:
        rec["IRN"] = irn.group(1)
        after2 = remainder[irn.end():].strip()
        dm2 = DATE_RE.search(after2)
        if dm2:
            rec["IRN_DATE"] = dm2.group(1)

    return rec if rec["GSTIN"] else None


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL FORMAT CONVERTER
# ─────────────────────────────────────────────────────────────────────────────

def records_to_excel_format(records: List[Dict]) -> List[Dict]:
    """
    Convert raw parser output to flat dict for ExcelGenerator.
    FIX #5: Applies smart invoice-value recalculation and POS fallback.
    """
    out = []
    for r in records:
        # FIX #5: Recalculate INVOICE_VALUE when it looks like a row number
        inv_val = _recalc_invoice_value(
            r.get("INVOICE_VALUE", ""),
            r.get("TAXABLE_VALUE", ""),
            r.get("IGST", ""),
            r.get("CGST", ""),
            r.get("SGST", ""),
            r.get("CESS", ""),
        )

        # POS fallback from GSTIN state code
        pos = str(r.get("PLACE_OF_SUPPLY", "") or "")
        if not pos:
            pos = _pos_from_gstin(str(r.get("GSTIN", "") or ""))

        row = {
            "GSTIN":             str(r.get("GSTIN", "")),
            "TRADE_NAME":        str(r.get("TRADE_NAME", "")),
            "INVOICE_NO":        str(r.get("INVOICE_NO", "")),
            "INVOICE_TYPE":      str(r.get("INVOICE_TYPE", "R")),
            "INVOICE_DATE":      str(r.get("INVOICE_DATE", "")),
            "INVOICE_VALUE":     inv_val,
            "PLACE_OF_SUPPLY":   pos,
            "REVERSE_CHARGE":    str(r.get("REVERSE_CHARGE", "N")),
            "TAX_RATE":          r.get("TAX_RATE", ""),
            "TAXABLE_VALUE":     r.get("TAXABLE_VALUE", ""),
            "IGST":              r.get("IGST", ""),
            "CGST":              r.get("CGST", ""),
            "SGST":              r.get("SGST", ""),
            "CESS":              r.get("CESS", ""),
            "GSTR1_STATUS":      str(r.get("GSTR1_STATUS", "")),
            "GSTR1_FILING_DATE": str(r.get("GSTR1_FILING_DATE", "")),
            "GSTR1_PERIOD":      str(r.get("GSTR1_PERIOD", "")),
            "GSTR3B_STATUS":     str(r.get("GSTR3B_STATUS", "")),
            "AMENDMENT":         str(r.get("AMENDMENT", "N")),
            "TAX_PERIOD_AMENDED":str(r.get("TAX_PERIOD_AMENDED", "N/A")),
            "CANCEL_DATE":       str(r.get("CANCEL_DATE", "N/A")),
            "SOURCE":            str(r.get("SOURCE", "Unknown")),
            "IRN":               str(r.get("IRN", "N/A")),
            "IRN_DATE":          str(r.get("IRN_DATE", "N/A")),
            "_confidence":       r.get("_confidence", 0.97),
            "_source":           str(r.get("_source", "direct_pdf")),
        }
        out.append(row)
    return out