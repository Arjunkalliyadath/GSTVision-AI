# pdf_converter.py — File → Image Converter  (v3.0 — FIXED)
# ─────────────────────────────────────────────────────────────────────────────
# FIXES v3.0:
#   1. CRITICAL (VS Code error): `from PIL import Image` moved to MODULE LEVEL.
#      Previously it was inside _to_rgb() — the `-> Image` return annotation
#      was unresolvable at class/module scope → red underline in VS Code +
#      potential issues with some Python type-checking environments.
#   2. _to_rgb() now has a correct `-> Image` annotation (not a string).
#   3. All converter functions use the module-level Image import consistently.
#   4. Added async-compatible version: convert_file_to_images_async() wraps
#      the sync function in asyncio.to_thread() so the event loop stays free.
# ─────────────────────────────────────────────────────────────────────────────

import os
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import List

# ── PIL imports at MODULE LEVEL — fixes VS Code annotation error ──────────────
from PIL import Image, ImageFile, ImageOps

logger = logging.getLogger("gst2_fastapi.pdf_converter")

# Path to poppler — update this to where you installed poppler on Windows
POPPLER_PATH = r"C:\poppler\Library\bin"

PDF_EXTENSIONS   = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
HEIC_EXTENSIONS  = {".heic", ".heif"}
ALL_SUPPORTED    = PDF_EXTENSIONS | IMAGE_EXTENSIONS | HEIC_EXTENSIONS

LARGE_FILE_WARNING_BYTES = 50 * 1024 * 1024  # 50 MB


# ── Public API ────────────────────────────────────────────────────────────────

def convert_file_to_images(file_path: str, output_dir: str) -> List[str]:
    """
    Convert any supported file to PNG images. SYNCHRONOUS version.
    For use in async pipeline: call convert_file_to_images_async() instead.
    """
    file_path  = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = file_path.suffix.lower()

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    if ext not in ALL_SUPPORTED:
        logger.error(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED))}"
        )
        return []

    size_kb = file_path.stat().st_size / 1024
    logger.info(f"Converting: {file_path.name} ({ext}, {size_kb:.0f} KB)")

    if file_path.stat().st_size > LARGE_FILE_WARNING_BYTES:
        logger.warning(f"Large file ({size_kb/1024:.1f} MB) — processing may be slow")

    try:
        if ext == ".pdf":
            return _convert_pdf(file_path, output_dir)
        elif ext in {".tiff", ".tif"}:
            return _convert_tiff(file_path, output_dir)
        elif ext == ".gif":
            return _convert_gif(file_path, output_dir)
        elif ext in HEIC_EXTENSIONS:
            return _convert_heic(file_path, output_dir)
        else:
            return _convert_standard_image(file_path, output_dir)
    except Exception as e:
        logger.error(f"Conversion failed for {file_path.name}: {e}", exc_info=True)
        return []


async def convert_file_to_images_async(file_path: str, output_dir: str) -> List[str]:
    """
    NON-BLOCKING version of convert_file_to_images().
    Runs in thread pool so FastAPI event loop stays free.
    """
    return await asyncio.to_thread(convert_file_to_images, file_path, output_dir)


# ── Converter functions ───────────────────────────────────────────────────────

def _convert_pdf(file_path: Path, output_dir: Path) -> List[str]:
    from pdf2image import convert_from_path

    pages = convert_from_path(
        str(file_path), dpi=300,
        poppler_path=POPPLER_PATH, fmt="PNG"
    )
    image_paths = []
    for i, page in enumerate(pages):
        img_path = output_dir / f"{file_path.stem}_page_{i+1}.png"
        _to_rgb(page).save(str(img_path), "PNG")
        image_paths.append(str(img_path))
        logger.info(f"PDF page {i+1}/{len(pages)} → {img_path.name}")
    return image_paths


def _convert_standard_image(file_path: Path, output_dir: Path) -> List[str]:
    """Convert JPG/PNG/BMP/WEBP to clean RGB PNG."""
    img = _safe_open(file_path)
    if img is None:
        return []
    img_path = output_dir / f"{file_path.stem}_page_1.png"
    _to_rgb(img).save(str(img_path), "PNG")
    logger.info(f"{file_path.suffix.upper()} → {img_path.name}")
    return [str(img_path)]


def _convert_tiff(file_path: Path, output_dir: Path) -> List[str]:
    img = _safe_open(file_path)
    if img is None:
        return []
    image_paths = []
    page_num    = 0
    while True:
        try:
            img.seek(page_num)
        except EOFError:
            break
        img_path = output_dir / f"{file_path.stem}_page_{page_num+1}.png"
        _to_rgb(img.copy()).save(str(img_path), "PNG")
        image_paths.append(str(img_path))
        page_num += 1
    return image_paths or _convert_standard_image(file_path, output_dir)


def _convert_gif(file_path: Path, output_dir: Path) -> List[str]:
    img = _safe_open(file_path)
    if img is None:
        return []
    img_path = output_dir / f"{file_path.stem}_page_1.png"
    _to_rgb(img.copy()).save(str(img_path), "PNG")
    logger.info(f"GIF (frame 1) → {img_path.name}")
    return [str(img_path)]


def _convert_heic(file_path: Path, output_dir: Path) -> List[str]:
    try:
        try:
            from pillow_heif import register_heif_opener  # type: ignore
            register_heif_opener()
        except ImportError:
            # pillow-heif not installed — fall back to treating as standard image
            logger.warning(
                f"HEIC support disabled: pillow-heif not installed. "
                f"File {file_path.name} will be opened with PIL directly (may fail)."
            )
        
        img = _safe_open(file_path)
        if img is None:
            logger.warning(f"Cannot open HEIC file {file_path.name}")
            return []
        
        img_path = output_dir / f"{file_path.stem}_page_1.png"
        _to_rgb(img).save(str(img_path), "PNG")
        logger.info(f"HEIC → {img_path.name}")
        return [str(img_path)]
        
    except Exception as e:
        logger.error(f"HEIC conversion failed for {file_path.name}: {e}")
        # Return empty list instead of raising — allows pipeline to continue
        return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_open(file_path: Path) -> Image.Image | None:
    """Open image with PIL, allowing truncated files. Returns None on failure."""
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        img = Image.open(str(file_path))
        img.load()  # Force full decode now so errors surface here
        return img
    except Exception as e:
        logger.error(f"Cannot open image {file_path.name}: {e}")
        return None


def _to_rgb(img: Image.Image) -> Image.Image:
    """
    Convert ANY PIL image mode to clean RGB.
    This handles every mode PIL can produce.
    """
    mode = img.mode

    if mode == "RGB":
        return img

    if mode == "RGBA":
        # Flatten alpha onto white background
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg

    if mode == "LA":
        # Grayscale + alpha → flatten onto white
        bg = Image.new("L", img.size, 255)
        bg.paste(img.split()[0], mask=img.split()[1])
        return bg.convert("RGB")

    if mode == "P":
        # Palette — convert via RGBA to handle transparency
        img_rgba = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img_rgba, mask=img_rgba.split()[3])
        return bg

    if mode == "L":
        return img.convert("RGB")

    if mode == "CMYK":
        return img.convert("RGB")

    if mode == "1":
        return img.convert("L").convert("RGB")

    if mode in ("I", "F"):
        # 32-bit int or float → normalize to 8-bit
        arr = np.array(img, dtype=np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)
        return Image.fromarray(arr).convert("RGB")

    # Unknown mode — try direct conversion
    logger.warning(f"Unknown PIL mode '{mode}' — trying direct RGB conversion")
    try:
        return img.convert("RGB")
    except Exception as e:
        logger.error(f"Cannot convert mode {mode} to RGB: {e}")
        return Image.new("RGB", img.size, (255, 255, 255))