"""
Perspective Correction
======================
Four-point perspective transform to produce a top-down "bird's-eye" view
of a detected document region.
"""

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sort four corner points into a consistent order:
        [top-left, top-right, bottom-right, bottom-left]

    Args:
        pts: (4, 2) array of unordered corner points.

    Returns:
        (4, 2) array in TL, TR, BR, BL order.
    """
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum and difference along axes
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]   # Top-left:     smallest x+y
    rect[2] = pts[np.argmax(s)]   # Bottom-right: largest  x+y
    rect[1] = pts[np.argmin(d)]   # Top-right:    smallest x-y
    rect[3] = pts[np.argmax(d)]   # Bottom-left:  largest  x-y

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply a perspective warp using four corner points so that the document
    fills the output image as a flat rectangle.

    Args:
        image: BGR source image (original resolution).
        pts:   (4, 2) array of corner coordinates in the source image.

    Returns:
        Warped BGR image with width/height computed from the detected corners.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute output width = max of top edge and bottom edge
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Compute output height = max of left edge and right edge
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Destination rectangle (top-down view)
    dst = np.array([
        [0,             0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0,             max_height - 1],
    ], dtype=np.float32)

    # Perspective matrix and warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


def rotate_if_landscape(image: np.ndarray) -> np.ndarray:
    """
    Rotate a warped image 90° clockwise when it is wider than tall,
    so the final document is always in portrait orientation.
    """
    h, w = image.shape[:2]
    if w > h:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def crop_borders(image: np.ndarray, margin: int = 5) -> np.ndarray:
    """
    Remove a thin border (artefacts from perspective warp) from all sides.

    Args:
        image: Warped image.
        margin: Pixels to strip from each edge.

    Returns:
        Cropped image.
    """
    h, w = image.shape[:2]
    return image[margin: h - margin, margin: w - margin]
