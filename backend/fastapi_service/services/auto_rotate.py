# auto_rotate.py — Auto-Rotation Detector for Document Images
# ─────────────────────────────────────────────────────────────────────────────
# Detects and corrects 90°/180°/270° rotation in document photos.
# Uses Tesseract OSD (Orientation & Script Detection) as primary method.
# Falls back to Hough-line angle analysis if OSD is unavailable.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger("gst2_fastapi.auto_rotate")


def detect_and_correct_rotation(image_path: str) -> Tuple[str, int]:
    """
    Detect if an image is rotated 90°/180°/270° and correct it.
    
    Returns:
        (corrected_image_path, rotation_angle_applied)
        If no rotation needed, returns (original_path, 0).
    """
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Cannot read image for rotation check: {image_path}")
        return image_path, 0
    
    angle = _detect_rotation_angle(img, image_path)
    
    if angle == 0:
        logger.info(f"[AutoRotate] No rotation needed: {Path(image_path).name}")
        return image_path, 0
    
    # Apply rotation
    rotated = _rotate_image(img, angle)
    
    # Save rotated image (overwrite or save alongside)
    p = Path(image_path)
    rotated_path = str(p.parent / f"{p.stem}_rotated{p.suffix}")
    cv2.imwrite(rotated_path, rotated)
    
    logger.info(f"[AutoRotate] Rotated {angle}°: {p.name} → {Path(rotated_path).name}")
    return rotated_path, angle


def _detect_rotation_angle(img: np.ndarray, image_path: str) -> int:
    """
    Detect rotation using Tesseract OSD first, then Hough-line fallback.
    Returns one of: 0, 90, 180, 270
    """
    # Method 1: Tesseract OSD
    angle = _detect_via_tesseract_osd(image_path)
    if angle is not None:
        return angle
    
    # Method 2: Aspect ratio + text line analysis
    angle = _detect_via_text_orientation(img)
    if angle is not None:
        return angle
    
    return 0


def _detect_via_tesseract_osd(image_path: str) -> Optional[int]:
    """Use Tesseract's OSD to detect page orientation."""
    try:
        import pytesseract
        from PIL import Image
        
        img = Image.open(image_path)
        # Resize for faster OSD (doesn't need full resolution)
        max_dim = 2000
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)))
        
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        orientation = osd.get("orientation", 0)
        confidence = osd.get("orientation_conf", 0)
        
        logger.info(
            f"[AutoRotate] Tesseract OSD: orientation={orientation}°, "
            f"confidence={confidence}"
        )
        
        # Only trust high-confidence detections
        if confidence >= 1.0 and orientation in (90, 180, 270):
            return orientation
        
        return 0  # No rotation needed or low confidence
        
    except Exception as exc:
        logger.debug(f"[AutoRotate] Tesseract OSD failed: {exc}")
        return None


def _detect_via_text_orientation(img: np.ndarray) -> Optional[int]:
    """
    Heuristic: If the image is taller than it is wide by a significant margin,
    AND horizontal text lines are detected when rotated 90°, then it's likely
    a landscape document photographed in portrait orientation.
    """
    import cv2
    
    h, w = img.shape[:2]
    aspect = w / h
    
    # If roughly square or landscape, probably correct already
    if aspect >= 0.7:
        return 0
    
    # Image is significantly portrait (aspect < 0.7) — test if rotating helps
    # Try rotating 90° counterclockwise and check for horizontal text lines
    rotated_90 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    h_lines_original = _count_horizontal_lines(img)
    h_lines_rotated = _count_horizontal_lines(rotated_90)
    
    logger.info(
        f"[AutoRotate] Heuristic: aspect={aspect:.2f}, "
        f"h_lines_orig={h_lines_original}, h_lines_rot90={h_lines_rotated}"
    )
    
    # If rotating 90° produces significantly more horizontal lines, rotate
    if h_lines_rotated > h_lines_original * 1.5 and h_lines_rotated >= 5:
        return 90
    
    # Also try 270° (90° clockwise)
    rotated_270 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h_lines_270 = _count_horizontal_lines(rotated_270)
    
    if h_lines_270 > h_lines_original * 1.5 and h_lines_270 >= 5:
        return 270
    
    return 0


def _count_horizontal_lines(img: np.ndarray) -> int:
    """Count near-horizontal line segments in an image using Hough transform."""
    import cv2
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Resize for speed
    h, w = gray.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=80, minLineLength=gray.shape[1] // 8, maxLineGap=20
    )
    
    if lines is None:
        return 0
    
    count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx == 0:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, dx)))
        if angle <= 5:  # Near-horizontal
            count += 1
    
    return count


def _rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by exact 90° increments."""
    import cv2
    
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img
