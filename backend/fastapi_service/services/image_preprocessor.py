# image_preprocessor.py  (v6.0 — AUTO-ROTATE + ADAPTIVE PREPROCESSING)
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT'S NEW IN v6.0
# ──────────────────
#
#  1. STAGE 0 — AUTO-ROTATION (90°/180°/270°)
#     Before ANY preprocessing, detect if the image is rotated sideways.
#     Uses Tesseract OSD + Hough line heuristic. Critical for phone photos
#     taken in landscape that appear as portrait (or vice versa).
#
#  2. ADAPTIVE PREPROCESSING INTENSITY
#     Quality score determines how many variants to generate:
#     - Clean images (score ≥ 0.75): 2 variants (fast path)
#     - Medium quality (0.45–0.75): 4 variants (standard path)
#     - Poor quality (< 0.45): all 7 variants (heavy path)
#
#  3. VARIANT v7 — TESSERACT-OPTIMISED (NEW)
#     High-contrast Otsu binary with morphological noise cleanup.
#     Tesseract performs best on clean black-on-white binary images.
#
#  All v5.0 features preserved.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import threading
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger("gst2_fastapi.image_preprocessor")

# ── Tunable limits ─────────────────────────────────────────────────────────────
TARGET_WIDTH           = 3000    # pixels — upscale images below this width
MAX_WIDTH              = 6000    # pixels — hard cap to keep RAM safe
PREPROCESS_TIMEOUT_SEC = 180     # seconds — increased for 7-variant pipeline


class ImagePreprocessor:
    """
    Professional document scanner + enhancement engine for GST 2A statements.

    v6.0 architecture:
      • Stage 0  — Auto-rotation detection (90°/180°/270°)
      • Stage 1  — Load + normalise (EXIF, resize)
      • Stage 2  — Document detection + perspective warp
      • Stage 2.5 — Shadow removal + background normalisation
      • Stage 2.6 — Fine skew correction (±5°)
      • Stage 3  — Multi-variant generation (adaptive: 2/4/7 variants)
    """

    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi
        self._enhancer: Optional[Any] = None

    def _get_enhancer(self):
        if self._enhancer is None:
            from .gst_image_enhancer import get_gst_enhancer
            self._enhancer = get_gst_enhancer()
        return self._enhancer

    # ── Public API ─────────────────────────────────────────────────────────────

    def preprocess(self, image_path: str, output_path: Optional[str] = None) -> str:
        """Single-variant API (backward compatible). Returns best variant path."""
        variants = self.preprocess_variants(image_path, output_path)
        return variants[0] if variants else image_path

    def preprocess_variants(
        self,
        image_path: str,
        output_base: Optional[str] = None,
    ) -> List[str]:
        """
        Generate preprocessed variants with auto-rotation and adaptive intensity.

        Always returns at least [image_path] (original) as the final fallback.
        """
        output_base   = output_base or image_path
        result_holder: Dict[str, Any] = {"paths": [], "error": None}

        def _run():
            try:
                result_holder["paths"] = self._create_variants(image_path, output_base)
            except Exception as exc:
                result_holder["error"] = exc

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=PREPROCESS_TIMEOUT_SEC)

        if t.is_alive():
            logger.warning(
                f"Preprocessing timed out ({PREPROCESS_TIMEOUT_SEC}s) for "
                f"{Path(image_path).name}. Falling back to original."
            )
            return [image_path]

        if result_holder["error"]:
            logger.warning(
                f"Preprocessing failed for {Path(image_path).name}: "
                f"{result_holder['error']}. Falling back to original."
            )
            return [image_path]

        paths = result_holder["paths"]
        if not paths:
            return [image_path]

        # Always keep original as absolute last fallback
        if image_path not in paths:
            paths.append(image_path)
        return paths

    # ── Core pipeline ──────────────────────────────────────────────────────────

    def _create_variants(self, image_path: str, output_base: str) -> List[str]:
        import cv2
        from PIL import Image, ImageOps

        base   = Path(output_base)
        parent = base.parent
        parent.mkdir(parents=True, exist_ok=True)
        stem   = base.stem

        # ── STAGE 0: AUTO-ROTATION DETECTION ─────────────────────────────────
        actual_path = image_path
        try:
            from .auto_rotate import detect_and_correct_rotation
            rotated_path, angle = detect_and_correct_rotation(image_path)
            if angle != 0:
                actual_path = rotated_path
                logger.info(f"[Stage0] Auto-rotated {angle}°: {Path(image_path).name}")
        except Exception as exc:
            logger.warning(f"[Stage0] Auto-rotation failed: {exc}")

        # ── STAGE 1: LOAD + NORMALISE ────────────────────────────────────────
        pil_img = Image.open(actual_path)
        pil_img = ImageOps.exif_transpose(pil_img)          # EXIF rotation fix

        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
        if pil_img.mode == "RGBA":
            bg = Image.new("RGB", pil_img.size, (255, 255, 255))
            bg.paste(pil_img, mask=pil_img.split()[3])
            pil_img = bg

        w, h = pil_img.size
        logger.info(f"[Stage1] Loaded: {w}×{h} — {Path(image_path).name}")

        # Hard cap — prevent OOM on very large phone photos
        if w > MAX_WIDTH or h > MAX_WIDTH * 1.5:
            scale   = min(MAX_WIDTH / w, (MAX_WIDTH * 1.5) / h)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h    = pil_img.size
            logger.info(f"[Stage1] Downscaled → {w}×{h}")

        # Upscale small images — OCR needs ≥300 DPI equivalent
        if w < TARGET_WIDTH:
            scale   = min(TARGET_WIDTH / w, 4.0)
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h    = pil_img.size
            logger.info(f"[Stage1] Upscaled → {w}×{h}")

        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # ── QUALITY ASSESSMENT → ADAPTIVE INTENSITY ──────────────────────────
        enhancer = self._get_enhancer()
        quality = enhancer.score_image_quality(img_bgr)
        quality_score = quality["overall"]
        logger.info(
            f"[QualityCheck] score={quality_score:.2f} "
            f"sharp={quality['sharpness']:.2f} "
            f"contrast={quality['contrast']:.2f} "
            f"brightness={quality['brightness']:.2f}"
        )

        # ── STAGE 2: DOCUMENT DETECTION + PERSPECTIVE WARP ────────────────────
        warped_bgr, doc_found = self._detect_and_warp(img_bgr)
        logger.info(
            f"[Stage2] Document detection: "
            f"{'quad found → perspective warped' if doc_found else 'no quad → full image'}"
        )

        # ── STAGE 2.5: SHADOW REMOVAL + BACKGROUND NORMALISATION ─────────────
        try:
            photo_corrected, correction_meta = enhancer.full_photo_correction(
                warped_bgr,
                apply_shadow_removal=True,
                apply_background_norm=True,
                apply_skew_correction=False,
                apply_moire_removal=False,
            )
            logger.info(
                f"[Stage2.5] Shadow/BG correction done — "
                f"quality={correction_meta.get('overall', '?'):.2f}"
            )
        except Exception as exc:
            logger.warning(f"[Stage2.5] Shadow removal failed: {exc} — using warped image.")
            photo_corrected = warped_bgr

        # ── STAGE 2.6: FINE SKEW CORRECTION ──────────────────────────────────
        try:
            deskewed, angle = enhancer.correct_skew(photo_corrected, max_angle_deg=5.0)
            if abs(angle) >= 0.3:
                logger.info(f"[Stage2.6] Deskew applied: {angle:+.2f}°")
                photo_corrected = deskewed
            else:
                logger.debug(f"[Stage2.6] No deskew needed (angle={angle:.3f}°)")
        except Exception as exc:
            logger.warning(f"[Stage2.6] Deskew failed: {exc}")

        # ── STAGE 3: ADAPTIVE MULTI-VARIANT GENERATION ────────────────────────
        variants: List[str] = []

        def _save(img: np.ndarray, label: str) -> str:
            path = str(parent / f"{stem}_{label}.png")
            cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            try:
                q = enhancer.score_image_quality(img)
                logger.info(
                    f"  Variant saved: {label} | "
                    f"sharp={q['sharpness']:.2f} "
                    f"contrast={q['contrast']:.2f} "
                    f"overall={q['overall']:.2f}"
                )
            except Exception:
                logger.info(f"  Variant saved: {label}")
            return path

        if quality_score >= 0.75:
            # ── CLEAN IMAGE — Light path (2 variants) ─────────────────────────
            logger.info("[Stage3] Clean image → Light path (2 variants)")
            try:
                v1 = self._variant_colour_enhanced(photo_corrected)
                variants.append(_save(v1, "v1_colour"))
            except Exception as exc:
                logger.debug(f"v1 skipped: {exc}")

            try:
                v4 = self._variant_clahe_only(photo_corrected)
                variants.append(_save(v4, "v4_clahe"))
            except Exception as exc:
                logger.debug(f"v4 skipped: {exc}")

        elif quality_score >= 0.45:
            # ── MEDIUM QUALITY — Standard path (4 variants) ───────────────────
            logger.info("[Stage3] Medium quality → Standard path (4 variants)")
            try:
                v1 = self._variant_colour_enhanced(photo_corrected)
                variants.append(_save(v1, "v1_colour"))
            except Exception as exc:
                logger.debug(f"v1 skipped: {exc}")

            try:
                v2 = self._variant_bw_binary(photo_corrected)
                variants.append(_save(v2, "v2_binary"))
            except Exception as exc:
                logger.debug(f"v2 skipped: {exc}")

            try:
                v4 = self._variant_clahe_only(photo_corrected)
                variants.append(_save(v4, "v4_clahe"))
            except Exception as exc:
                logger.debug(f"v4 skipped: {exc}")

            try:
                v6 = self._variant_table_enhanced(photo_corrected, enhancer)
                variants.append(_save(v6, "v6_table"))
            except Exception as exc:
                logger.debug(f"v6 skipped: {exc}")

        else:
            # ── POOR QUALITY — Heavy path (all 7 variants) ────────────────────
            logger.info("[Stage3] Poor quality → Heavy path (7 variants)")

            try:
                v1 = self._variant_colour_enhanced(photo_corrected)
                variants.append(_save(v1, "v1_colour"))
            except Exception as exc:
                logger.debug(f"v1 skipped: {exc}")

            try:
                v2 = self._variant_bw_binary(photo_corrected)
                variants.append(_save(v2, "v2_binary"))
            except Exception as exc:
                logger.debug(f"v2 skipped: {exc}")

            try:
                v3 = self._variant_denoised_morph(photo_corrected)
                variants.append(_save(v3, "v3_denoise"))
            except Exception as exc:
                logger.debug(f"v3 skipped: {exc}")

            try:
                v4 = self._variant_clahe_only(photo_corrected)
                variants.append(_save(v4, "v4_clahe"))
            except Exception as exc:
                logger.debug(f"v4 skipped: {exc}")

            try:
                v5 = self._variant_shadow_binary(warped_bgr, enhancer)
                variants.append(_save(v5, "v5_shadow_binary"))
            except Exception as exc:
                logger.debug(f"v5 skipped: {exc}")

            try:
                v6 = self._variant_table_enhanced(photo_corrected, enhancer)
                variants.append(_save(v6, "v6_table"))
            except Exception as exc:
                logger.debug(f"v6 skipped: {exc}")

            try:
                v7 = self._variant_tesseract_optimised(photo_corrected)
                variants.append(_save(v7, "v7_tess_binary"))
            except Exception as exc:
                logger.debug(f"v7 skipped: {exc}")

        logger.info(
            f"Preprocessing complete: {len(variants)} variants for "
            f"{Path(image_path).name} (quality={quality_score:.2f})"
        )
        return variants

    # ── STAGE 2: Document Detection + Perspective Warp ─────────────────────────

    def _detect_and_warp(
        self, img_bgr: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        import cv2

        h, w = img_bgr.shape[:2]
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        median = float(np.median(blurred))
        sigma  = 0.33
        lo     = max(0,   int((1.0 - sigma) * median))
        hi     = min(255, int((1.0 + sigma) * median))
        edges  = cv2.Canny(blurred, lo, hi)

        kernel  = np.ones((9, 9), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return img_bgr, False

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        doc_quad = None
        for cnt in contours[:5]:
            area = cv2.contourArea(cnt)
            if area < w * h * 0.15:
                continue
            peri   = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                doc_quad = approx
                break
            if 4 <= len(approx) <= 6:
                doc_quad = self._force_quad(approx)
                if doc_quad is not None:
                    break

        if doc_quad is None:
            return img_bgr, False

        pts_src = doc_quad.reshape(4, 2).astype(np.float32)
        pts_src = self._order_points(pts_src)

        (tl, tr, br, bl) = pts_src
        width_top    = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left  = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)

        out_w = int(max(width_top,    width_bottom))
        out_h = int(max(height_left,  height_right))

        if out_w < 200 or out_h < 200 or out_w > w * 2 or out_h > h * 2:
            return img_bgr, False

        pts_dst = np.array([
            [0,         0        ],
            [out_w - 1, 0        ],
            [out_w - 1, out_h - 1],
            [0,         out_h - 1],
        ], dtype=np.float32)

        M      = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
        logger.info(f"[Stage2] Perspective warp: {w}×{h} → {out_w}×{out_h}")
        return warped, True

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _force_quad(approx: np.ndarray) -> Optional[np.ndarray]:
        pts = approx.reshape(-1, 2).astype(np.float32)
        if len(pts) < 4:
            return None
        min_dist = float("inf")
        min_idx  = 0
        for i in range(len(pts)):
            j    = (i + 1) % len(pts)
            dist = float(np.linalg.norm(pts[i] - pts[j]))
            if dist < min_dist:
                min_dist = dist
                min_idx  = i
        j       = (min_idx + 1) % len(pts)
        merged  = ((pts[min_idx] + pts[j]) / 2).astype(np.float32)
        new_pts = np.delete(pts, j, axis=0)
        new_pts[min_idx] = merged
        if len(new_pts) == 4:
            return new_pts.reshape(4, 1, 2).astype(np.int32)
        return None

    # ── Enhancement variants ───────────────────────────────────────────────────

    def _variant_colour_enhanced(self, img_bgr: np.ndarray) -> np.ndarray:
        """v1 — Colour image with CLAHE (L channel) + unsharp mask."""
        import cv2

        lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l     = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        blur      = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        return sharpened

    def _variant_bw_binary(self, img_bgr: np.ndarray) -> np.ndarray:
        """v2 — B&W adaptive binary."""
        import cv2

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharpened = cv2.filter2D(gray, -1, kernel)
        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21, C=8,
        )
        k_small = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_small, iterations=1)
        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    def _variant_denoised_morph(self, img_bgr: np.ndarray) -> np.ndarray:
        """v3 — Heavy denoise + morphological thickening."""
        import cv2

        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        clahe    = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k_thin    = np.ones((2, 2), np.uint8)
        thickened = cv2.dilate(binary, k_thin, iterations=1)
        return cv2.cvtColor(thickened, cv2.COLOR_GRAY2BGR)

    def _variant_clahe_only(self, img_bgr: np.ndarray) -> np.ndarray:
        """v4 — Grayscale + CLAHE + bilateral filter."""
        import cv2

        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _variant_shadow_binary(
        self, warped_bgr: np.ndarray, enhancer
    ) -> np.ndarray:
        """v5 — Shadow-normalised high-contrast binary."""
        import cv2

        try:
            shadow_free = enhancer.remove_shadows(warped_bgr)
        except Exception:
            shadow_free = warped_bgr

        gray  = cv2.cvtColor(shadow_free, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, None, h=6, templateWindowSize=7, searchWindowSize=21)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=6,
        )
        k = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _variant_table_enhanced(
        self, photo_corrected: np.ndarray, enhancer
    ) -> np.ndarray:
        """v6 — Table-line-enhanced binary."""
        try:
            return enhancer.enhance_table_lines(photo_corrected)
        except Exception as exc:
            logger.debug(f"v6 table enhancement internal error: {exc}")
            import cv2
            gray  = cv2.cvtColor(photo_corrected, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray  = clahe.apply(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _variant_tesseract_optimised(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        v7 — Tesseract-optimised binary (NEW in v6.0).
        Clean Otsu threshold after aggressive CLAHE + Gaussian blur.
        Tesseract performs best on high-contrast black-on-white images.
        """
        import cv2

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Extra aggressive CLAHE for maximum text contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12, 12))
        gray = clahe.apply(gray)

        # Light Gaussian blur to reduce noise before Otsu
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Otsu threshold — works very well on clean document images
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup — remove speckle noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Slight dilation to thicken thin text strokes
        binary = cv2.dilate(binary, kernel, iterations=1)

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ── Singleton ──────────────────────────────────────────────────────────────────

_preprocessor_instance: Optional[ImagePreprocessor] = None


def get_image_preprocessor() -> ImagePreprocessor:
    """Return the module-level singleton ImagePreprocessor."""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = ImagePreprocessor()
        logger.info("ImagePreprocessor v6.0 singleton created.")
    return _preprocessor_instance