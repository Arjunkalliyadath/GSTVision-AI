# gst_image_enhancer.py  (v1.0 — PROFESSIONAL GST-SPECIFIC IMAGE ENHANCEMENT)
# ─────────────────────────────────────────────────────────────────────────────
#
# PURPOSE
# -------
# Dedicated enhancement engine for phone photos and scanned images of GST 2A
# statements.  Addresses the five root causes of poor OCR accuracy on these
# documents:
#
#   Problem 1 — Uneven shadows from ambient/desk lighting
#     Fix: Morphological background estimation + illumination normalisation
#
#   Problem 2 — Fine rotational skew not caught by perspective warp
#     Fix: Hough line transform on near-horizontal text rows → micro-rotate
#
#   Problem 3 — Table grid lines lost during contrast enhancement
#     Fix: Morphological line detection → reinforce H/V grid before threshold
#
#   Problem 4 — Off-white / yellow / coloured paper background
#     Fix: LAB-space white-balance to force paper → neutral white
#
#   Problem 5 — Moiré interference from photographing printed documents
#     Fix: Band-limited Gaussian smoothing + unsharp mask to restore text
#
#   Problem 6 — Cannot choose the best preprocessing automatically
#     Fix: Laplacian/contrast/brightness quality scorer for each variant
#
# PUBLIC API
# ----------
#   enhancer = GSTImageEnhancer()
#   result   = enhancer.remove_shadows(img_bgr)        → np.ndarray
#   result   = enhancer.correct_skew(img_bgr)          → (np.ndarray, angle)
#   result   = enhancer.enhance_table_lines(img_bgr)   → np.ndarray
#   result   = enhancer.normalize_background(img_bgr)  → np.ndarray
#   result   = enhancer.remove_moire(img_bgr)          → np.ndarray
#   metrics  = enhancer.score_image_quality(img_bgr)   → Dict[str, float]
#
# All methods are stateless — safe to call concurrently.
# All methods are pure NumPy/OpenCV — no heavy ML dependencies.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Tuple, Optional

logger = logging.getLogger("gst2_fastapi.gst_image_enhancer")


class GSTImageEnhancer:
    """
    Professional image enhancement engine tailored for GST 2A statement photos.

    All methods accept a BGR NumPy array (as returned by cv2.imread / cv2.cvtColor)
    and return a BGR NumPy array unless noted otherwise.

    Methods are stateless and thread-safe — instantiate once and reuse across
    all pipeline calls.
    """

    # ─── 1. SHADOW REMOVAL ────────────────────────────────────────────────────

    def remove_shadows(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Remove shadows and normalise uneven illumination.

        Algorithm
        ---------
        For each channel:
          1. Estimate the background (slow-varying illumination) via morphological
             dilation + large Gaussian blur.  The large structuring element spans
             ~1/30 of the image, which is far bigger than any text glyph so the
             dilation fills over dark text regions.
          2. Normalise: channel_out = channel / background * TARGET_BRIGHTNESS.
             Division cancels the shadow gradient; the multiply rescales to the
             desired white-point.
        Result: paper background → near-uniform ~220 grey; text stays dark.

        Why 220 (not 255)?
            Clamping to 220 leaves a tiny headroom so that legitimate faint ink
            marks (low-contrast stamps, light print) are not clipped white.
        """
        import cv2

        h, w = img_bgr.shape[:2]
        # Kernel covers ~1/30 of image so it spans over individual glyphs
        kh = max(img_bgr.shape[0] // 30, 21) | 1   # ensure odd
        kw = max(img_bgr.shape[1] // 30, 21) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        result = np.zeros_like(rgb)

        for ch in range(3):
            channel = rgb[:, :, ch]
            # Background: dilate (fills dark text gaps) + smooth
            bg = cv2.dilate(channel, kernel)
            bg = cv2.GaussianBlur(bg, (21, 21), 0).astype(np.float32)
            # Normalise — avoid zero-division with small epsilon
            norm = channel / (bg + 1e-6) * 220.0
            result[:, :, ch] = np.clip(norm, 0, 255)

        normalised = result.astype(np.uint8)
        logger.debug("Shadow removal applied.")
        return cv2.cvtColor(normalised, cv2.COLOR_RGB2BGR)

    # ─── 2. FINE SKEW CORRECTION ──────────────────────────────────────────────

    def correct_skew(
        self,
        img_bgr: np.ndarray,
        max_angle_deg: float = 5.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Detect and correct fine rotational skew (up to ±max_angle_deg degrees).

        Algorithm
        ---------
        1. Convert to grayscale, apply Canny edge detection.
        2. Run Hough Probabilistic Line Transform to find all significant line
           segments in the image.
        3. Keep only near-horizontal lines (|angle| ≤ max_angle_deg) — these
           correspond to text baseline rows and table rules.
        4. Take the median angle (robust to a handful of false positives).
        5. If |median| ≥ 0.3° rotate the image by −median degrees with a white
           fill so no black border appears.

        Returns (corrected_image, angle_corrected_degrees).
        Returns (original_image, 0.0) if no correction is needed.

        Why only ≤5°?
            Perspective warp in Stage 2 already handles large tilts.  Hough
            lines on a strongly tilted document pick up wrong baselines and
            produce a worse result.  The 5° guard prevents over-correction.
        """
        import cv2

        h, w = img_bgr.shape[:2]
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Canny edges with auto thresholds
        median  = float(np.median(gray))
        sigma   = 0.33
        lo      = max(0,   int((1.0 - sigma) * median))
        hi      = min(255, int((1.0 + sigma) * median))
        edges   = cv2.Canny(gray, lo, hi, apertureSize=3)

        # Hough probabilistic lines
        min_line = max(w // 8, 80)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(w // 8, 80),
            minLineLength=min_line,
            maxLineGap=20,
        )

        if lines is None or len(lines) < 3:
            logger.debug("Skew correction: insufficient lines detected, skipped.")
            return img_bgr, 0.0

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            if dx == 0:
                continue
            angle = float(np.degrees(np.arctan2(y2 - y1, dx)))
            if abs(angle) <= max_angle_deg:
                angles.append(angle)

        if len(angles) < 3:
            logger.debug("Skew correction: too few near-horizontal lines, skipped.")
            return img_bgr, 0.0

        median_angle = float(np.median(angles))

        if abs(median_angle) < 0.3:
            logger.debug(f"Skew correction: angle={median_angle:.3f}° below threshold, skipped.")
            return img_bgr, 0.0

        # Rotate with expanded canvas to avoid clipping corners
        center = (w / 2.0, h / 2.0)
        M      = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        cos_a  = abs(M[0, 0])
        sin_a  = abs(M[0, 1])
        new_w  = int(h * sin_a + w * cos_a)
        new_h  = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0

        corrected = cv2.warpAffine(
            img_bgr, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        logger.info(f"Skew corrected: {median_angle:+.2f}° ({len(angles)} lines used)")
        return corrected, median_angle

    # ─── 3. TABLE LINE ENHANCEMENT ────────────────────────────────────────────

    def enhance_table_lines(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Detect and reinforce horizontal/vertical table grid lines.

        GST 2A statements are dense tabular data.  When photographed or scanned
        at low quality, table borders fade to light grey and adaptive thresholding
        treats them the same as background — causing columns to blur together and
        OCR to concatenate adjacent cell values.

        Algorithm
        ---------
        1. CLAHE on grayscale to boost local contrast.
        2. Morphological opening with a long horizontal kernel → detects
           horizontal rules (H-lines).
        3. Morphological opening with a long vertical kernel → detects
           vertical rules (V-lines).
        4. Combine H + V masks.  Dilate slightly (2×2) to ensure lines survive
           subsequent adaptive thresholding.
        5. Apply adaptive threshold to the CLAHE-enhanced gray.
        6. Force all detected grid pixels to black in the binary image, making
           table structure maximally distinct regardless of original line darkness.

        Returns a 3-channel BGR image (gray2bgr) ready for OCR.
        """
        import cv2

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Step 1 — CLAHE contrast boost
        clahe          = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray  = clahe.apply(gray)

        # Step 2 — Detect horizontal rules
        h_kernel_len = max(w // 12, 50)
        h_kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
        h_lines      = cv2.morphologyEx(enhanced_gray, cv2.MORPH_OPEN, h_kernel, iterations=2)
        _, h_thresh  = cv2.threshold(h_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 3 — Detect vertical rules
        v_kernel_len = max(h // 12, 50)
        v_kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
        v_lines      = cv2.morphologyEx(enhanced_gray, cv2.MORPH_OPEN, v_kernel, iterations=2)
        _, v_thresh  = cv2.threshold(v_lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 4 — Combine and dilate grid mask
        grid_mask   = cv2.add(h_thresh, v_thresh)
        dilate_kern = np.ones((2, 2), np.uint8)
        grid_mask   = cv2.dilate(grid_mask, dilate_kern, iterations=1)

        # Step 5 — Adaptive threshold on enhanced gray
        binary = cv2.adaptiveThreshold(
            enhanced_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,
            C=8,
        )

        # Step 6 — Re-stamp detected grid lines as solid black
        final = np.where(grid_mask > 0, np.uint8(0), binary).astype(np.uint8)

        n_h_px = int((h_thresh > 0).sum())
        n_v_px = int((v_thresh > 0).sum())
        logger.debug(f"Table line enhancement: H={n_h_px}px, V={n_v_px}px reinforced.")
        return cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)

    # ─── 4. BACKGROUND NORMALISATION ──────────────────────────────────────────

    def normalize_background(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Normalise off-white / yellowish / tinted paper to neutral white.

        Algorithm
        ---------
        1. Convert to CIE L*a*b* colour space.
        2. Identify "paper" pixels: L* > 75 (bright), |a*| < 15, |b*| < 25.
        3. Measure the median colour cast in the paper region (a*, b* bias).
        4. Subtract 80% of that bias from ALL pixels (soft correction —
           100% would over-correct on documents with intentional coloured areas).
        5. Convert back to BGR.

        Falls back gracefully if the paper region cannot be identified
        (too little bright area — e.g. very dark scan).
        """
        import cv2

        # Work in float32 LAB for precision
        img_f32 = img_bgr.astype(np.float32)
        lab     = cv2.cvtColor(img_f32, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab)

        # OpenCV LAB range: L 0-100, a/b -127 to +127 (scaled from float)
        paper_mask = (l > 75.0) & (np.abs(a) < 15.0) & (np.abs(b) < 25.0)
        n_paper    = int(paper_mask.sum())

        if n_paper > int(paper_mask.size * 0.10):
            # Median colour cast of paper region
            a_cast = float(np.median(a[paper_mask]))
            b_cast = float(np.median(b[paper_mask]))

            # Soft correction: remove 80% of the cast
            a = np.clip(a - a_cast * 0.80, -127.0, 127.0)
            b = np.clip(b - b_cast * 0.80, -127.0, 127.0)

            logger.debug(
                f"Background normalisation: a_cast={a_cast:.1f}, b_cast={b_cast:.1f}, "
                f"paper_px={n_paper}"
            )
        else:
            logger.debug(
                f"Background normalisation skipped: paper region too small "
                f"({n_paper} px / {paper_mask.size} total)"
            )

        lab_corrected = cv2.merge([l, a, b])
        bgr_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_Lab2BGR)
        return np.clip(bgr_corrected, 0, 255).astype(np.uint8)

    # ─── 5. MOIRÉ REMOVAL ─────────────────────────────────────────────────────

    def remove_moire(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Suppress moiré interference patterns from photographed printed documents.

        Moiré arises when the camera sensor grid aliases with the halftone dot
        pattern of a printed document.  It appears as a wavy colour/luminance
        pattern at a spatial frequency between that of individual dots and the
        full page — typically 50–150 cycles/image.

        Algorithm
        ---------
        1. Apply a mild Gaussian blur (σ=0.8) which attenuates the moiré
           frequency band while keeping long-range contrast.
        2. Apply an unsharp mask (weight 1.3 / −0.3) to restore the sharp
           edges of text glyphs which have much higher spatial frequency than
           moiré — so the sharpen step recovers them without bringing back moiré.

        Parameters are intentionally conservative.  Aggressive filtering removes
        fine numeric characters (esp. "1" and ".") that are barely wider than
        a moiré fringe.
        """
        import cv2

        blurred  = cv2.GaussianBlur(img_bgr, (3, 3), 0.8)
        sharpened = cv2.addWeighted(img_bgr, 1.3, blurred, -0.3, 0)
        logger.debug("Moiré removal applied.")
        return sharpened

    # ─── 6. IMAGE QUALITY SCORER ──────────────────────────────────────────────

    def score_image_quality(self, img_bgr: np.ndarray) -> Dict[str, float]:
        """
        Compute quality metrics for a preprocessed image.

        Metrics
        -------
        sharpness  — Laplacian variance (high = sharp text).
                     Normalised to [0, 1] by dividing by 500 (empirical cap).
        contrast   — Grayscale standard deviation.
                     Normalised to [0, 1] by dividing by 60.
        brightness — Deviation of mean gray from 185 (ideal paper brightness).
                     Score 1.0 at mean=185, drops toward 0 at extremes.
        noise      — Mean absolute difference between image and Gaussian-blurred
                     version.  Lower noise → higher score.
        overall    — Weighted composite: sharpness×0.35 + contrast×0.30 +
                     brightness×0.20 + noise×0.15.

        Returns a dict with keys: sharpness, contrast, brightness, noise, overall.
        All values in [0.0, 1.0].
        """
        import cv2

        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Sharpness: Laplacian variance (high = sharp edges)
        lap       = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness = float(min(lap.var() / 500.0, 1.0))

        # Contrast: standard deviation across pixels
        contrast  = float(min(gray.std() / 60.0, 1.0))

        # Brightness: penalty for too dark or too bright
        mean_gray   = float(gray.mean())
        brightness  = float(max(0.0, 1.0 - abs(mean_gray - 185.0) / 185.0))

        # Noise: mean-absolute residual after Gaussian blur
        blurred    = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_raw  = float(np.abs(gray - blurred).mean())
        noise      = float(max(0.0, 1.0 - noise_raw / 20.0))

        overall = (
            sharpness  * 0.35
            + contrast * 0.30
            + brightness * 0.20
            + noise    * 0.15
        )
        overall = float(min(overall, 1.0))

        return {
            "sharpness":  round(sharpness,  3),
            "contrast":   round(contrast,   3),
            "brightness": round(brightness, 3),
            "noise":      round(noise,      3),
            "overall":    round(overall,    3),
        }

    # ─── 7. COMPOSITE: FULL PHOTO CORRECTION PIPELINE ─────────────────────────

    def full_photo_correction(
        self,
        img_bgr: np.ndarray,
        apply_shadow_removal:     bool = True,
        apply_background_norm:    bool = True,
        apply_skew_correction:    bool = True,
        apply_moire_removal:      bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Run the complete photo correction pipeline in a single call.

        Stages (in recommended order):
          1. Shadow removal         — normalises illumination gradients
          2. Background normalisation — removes colour cast
          3. Skew correction        — fine-tunes rotation
          4. Moiré removal          — suppresses printing artefacts

        Returns (corrected_image, quality_metrics_dict).

        Note: table line enhancement is NOT included here because it binarises
        the image.  It is called separately in image_preprocessor.py when
        generating the table-enhanced variant.
        """
        result = img_bgr.copy()
        meta: Dict[str, float] = {}

        if apply_shadow_removal:
            result = self.remove_shadows(result)

        if apply_background_norm:
            result = self.normalize_background(result)

        if apply_skew_correction:
            result, angle = self.correct_skew(result)
            meta["skew_angle"] = angle

        if apply_moire_removal:
            result = self.remove_moire(result)

        quality = self.score_image_quality(result)
        meta.update(quality)

        logger.info(
            f"Full photo correction complete — "
            f"overall_quality={quality['overall']:.2f} "
            f"sharpness={quality['sharpness']:.2f} "
            f"contrast={quality['contrast']:.2f}"
        )
        return result, meta


# ── Singleton ──────────────────────────────────────────────────────────────────

_enhancer_instance: Optional[GSTImageEnhancer] = None


def get_gst_enhancer() -> GSTImageEnhancer:
    """Return the module-level singleton GSTImageEnhancer."""
    global _enhancer_instance
    if _enhancer_instance is None:
        _enhancer_instance = GSTImageEnhancer()
        logger.info("GSTImageEnhancer singleton created.")
    return _enhancer_instance