"""
Edge Detection Utilities
========================
Provides Canny-based edge detection and document contour extraction.
"""

import cv2
import numpy as np


def detect_edges(gray: np.ndarray,
                 low_threshold: int = 75,
                 high_threshold: int = 200,
                 dilation_iter: int = 1) -> np.ndarray:
    """
    Apply Canny edge detection with optional dilation to close gaps.

    Args:
        gray: Grayscale input image (already blurred recommended).
        low_threshold: Canny low hysteresis threshold.
        high_threshold: Canny high hysteresis threshold.
        dilation_iter: Number of dilation passes to close edge gaps.

    Returns:
        Binary edge map (uint8, 0/255).
    """
    edged = cv2.Canny(gray, low_threshold, high_threshold)

    if dilation_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=dilation_iter)

    return edged


def find_document_contour(edged: np.ndarray) -> np.ndarray | None:
    """
    Find the largest quadrilateral contour in an edge map.

    Strategy:
        1. Find all external contours sorted by area (largest first).
        2. For the top-5 largest, check whether the approximated
           polygon has exactly 4 vertices (a quadrilateral).
        3. Return the first match as a (4, 2) float32 array.

    Args:
        edged: Binary edge map from detect_edges().

    Returns:
        Array of shape (4, 2) with corner coordinates, or None if not found.
    """
    contours, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screen_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screen_contour = approx
            break

    if screen_contour is None:
        return None

    # Reshape to (4, 2)
    return screen_contour.reshape(4, 2).astype(np.float32)


def draw_contour(image: np.ndarray,
                 corners: np.ndarray,
                 color: tuple = (0, 255, 0),
                 thickness: int = 3) -> np.ndarray:
    """
    Draw detected corners on a copy of the image for visualisation.

    Args:
        image: BGR image.
        corners: (4, 2) array of corner points.
        color: BGR color tuple.
        thickness: Line thickness in pixels.

    Returns:
        Image with contour drawn.
    """
    vis = image.copy()
    pts = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)

    # Draw corner circles
    for pt in corners.astype(int):
        cv2.circle(vis, tuple(pt), 8, (0, 0, 255), -1)

    return vis


def auto_threshold_canny(gray: np.ndarray) -> tuple[int, int]:
    """
    Compute automatic Canny thresholds using Otsu's binarisation value.

    Returns:
        (low_threshold, high_threshold)
    """
    otsu_val, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    high = otsu_val
    low = max(0, int(0.4 * otsu_val))
    return low, high
