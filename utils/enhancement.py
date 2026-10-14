"""
Image Enhancement
=================
Post-warp enhancement to maximise document readability.

Modes
-----
adaptive  : Adaptive Gaussian thresholding — best for uneven lighting.
otsu      : Otsu global thresholding — best for clean, uniform lighting.
clahe     : CLAHE equalisation — best when colour / grey levels must be preserved.
none      : Return as-is (no enhancement).
"""

import cv2
import numpy as np


def enhance_document(image: np.ndarray, mode: str = "adaptive") -> np.ndarray:
    """
    Convert a warped BGR document image to an enhanced, readable form.

    Args:
        image: BGR or grayscale numpy array.
        mode: One of 'adaptive', 'otsu', 'clahe', 'none'.

    Returns:
        Enhanced image (grayscale for threshold modes, BGR for clahe/none).
    """
    mode = mode.lower()

    if mode == "none":
        return image

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if mode == "adaptive":
        return _adaptive_threshold(gray)
    elif mode == "otsu":
        return _otsu_threshold(gray)
    elif mode == "clahe":
        return _clahe_enhance(gray)
    else:
        raise ValueError(f"Unknown enhancement mode: '{mode}'. "
                         f"Choose from: adaptive, otsu, clahe, none.")


# ------------------------------------------------------------------
# Individual enhancement functions
# ------------------------------------------------------------------

def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding.

    Splits the image into small tiles and thresholds each locally,
    handling shadows and uneven lighting very well.
    """
    # Slight blur removes noise before thresholding
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    result = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=7
    )
    return result


def _otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Otsu's automatic global thresholding.

    Finds the single intensity value that best separates foreground
    (text/lines) from background — fast and effective under uniform light.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, result = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    return result


def _clahe_enhance(gray: np.ndarray) -> np.ndarray:
    """
    CLAHE — Contrast Limited Adaptive Histogram Equalisation.

    Boosts local contrast without oversaturation; preserves grey levels.
    Returns BGR so it can be saved as a colour-compatible image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Convert back to BGR so all modes return same channel count
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    """
    Apply an unsharp-mask sharpening filter.

    Useful as an optional post-processing step for scanned text.
    """
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    return cv2.filter2D(image, -1, kernel)


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Fast Non-Local Means denoising.

    Args:
        image: Grayscale image.
        strength: Filter strength (h parameter). Higher = smoother.
    """
    return cv2.fastNlMeansDenoising(image, h=strength)


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct minor rotational skew using Hough line analysis.

    Note: Perspective correction handles large tilts; this corrects
    small residual skew (±10°) in the warped output.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return image

    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if -10 < angle < 10:          # Ignore lines that are not near-horizontal
            angles.append(angle)

    if not angles:
        return image

    median_angle = float(np.median(angles))
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated
