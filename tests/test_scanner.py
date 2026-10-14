"""
Unit & Integration Tests — Document Scanner
============================================
Run with:  python -m pytest tests/ -v
"""

import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.edge_detection import detect_edges, find_document_contour, auto_threshold_canny
from utils.perspective import order_points, four_point_transform, crop_borders
from utils.enhancement import enhance_document, sharpen, denoise


# ===========================================================
# Fixtures
# ===========================================================

@pytest.fixture
def blank_gray():
    """Plain 300x400 grayscale image."""
    return np.ones((400, 300), dtype=np.uint8) * 200


@pytest.fixture
def rect_image():
    """
    White 500x600 image with a dark grey rectangle drawn on it
    so that edge detection can find a quadrilateral.
    """
    img = np.ones((600, 500), dtype=np.uint8) * 240
    cv2.rectangle(img, (60, 80), (440, 520), 30, -1)
    return img


@pytest.fixture
def bgr_image():
    """Simple BGR test image."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (200, 210, 220)
    cv2.rectangle(img, (50, 50), (550, 350), (20, 20, 20), -1)
    return img


# ===========================================================
# Edge Detection Tests
# ===========================================================

class TestEdgeDetection:

    def test_output_shape(self, rect_image):
        blurred = cv2.GaussianBlur(rect_image, (5, 5), 0)
        edges = detect_edges(blurred)
        assert edges.shape == rect_image.shape, "Edge map must match input shape"

    def test_output_dtype(self, rect_image):
        blurred = cv2.GaussianBlur(rect_image, (5, 5), 0)
        edges = detect_edges(blurred)
        assert edges.dtype == np.uint8

    def test_edges_are_binary(self, rect_image):
        blurred = cv2.GaussianBlur(rect_image, (5, 5), 0)
        edges = detect_edges(blurred)
        unique = np.unique(edges)
        assert set(unique).issubset({0, 255}), "Edge map should be binary (0 or 255)"

    def test_edges_detected_on_rectangle(self, rect_image):
        blurred = cv2.GaussianBlur(rect_image, (5, 5), 0)
        edges = detect_edges(blurred)
        assert edges.sum() > 0, "Should detect edges on rectangle"

    def test_blank_image_few_edges(self, blank_gray):
        blurred = cv2.GaussianBlur(blank_gray, (5, 5), 0)
        edges = detect_edges(blurred)
        assert edges.sum() == 0, "Blank uniform image should produce no edges"

    def test_auto_threshold(self, rect_image):
        low, high = auto_threshold_canny(rect_image)
        assert 0 <= low < high <= 255


class TestContourDetection:

    def test_finds_quad_on_rect_image(self, rect_image):
        blurred = cv2.GaussianBlur(rect_image, (5, 5), 0)
        edges = detect_edges(blurred)
        corners = find_document_contour(edges)
        assert corners is not None, "Should find a quadrilateral on rect image"
        assert corners.shape == (4, 2), "Corners must be shape (4, 2)"

    def test_returns_float32(self, rect_image):
        blurred = cv2.GaussianBlur(rect_image, (5, 5), 0)
        edges = detect_edges(blurred)
        corners = find_document_contour(edges)
        if corners is not None:
            assert corners.dtype == np.float32

    def test_blank_returns_none(self, blank_gray):
        blurred = cv2.GaussianBlur(blank_gray, (5, 5), 0)
        edges = detect_edges(blurred)
        corners = find_document_contour(edges)
        assert corners is None, "Blank image should yield no corners"


# ===========================================================
# Perspective Tests
# ===========================================================

class TestPerspective:

    def test_order_points_tl_min_sum(self):
        pts = np.array([[100, 200], [300, 50], [320, 280], [80, 310]], dtype=np.float32)
        ordered = order_points(pts)
        # Top-left must have smallest x+y
        s = ordered.sum(axis=1)
        assert np.argmin(s) == 0, "First point should have smallest sum"

    def test_order_points_br_max_sum(self):
        pts = np.array([[100, 200], [300, 50], [320, 280], [80, 310]], dtype=np.float32)
        ordered = order_points(pts)
        s = ordered.sum(axis=1)
        assert np.argmax(s) == 2, "Third point should have largest sum"

    def test_four_point_transform_shape(self, bgr_image):
        pts = np.array([[50, 50], [549, 50], [549, 349], [50, 349]], dtype=np.float32)
        warped = four_point_transform(bgr_image, pts)
        assert warped.ndim == 3, "Warped output should be 3-channel BGR"
        assert warped.shape[0] > 0 and warped.shape[1] > 0

    def test_crop_borders_reduces_size(self, bgr_image):
        cropped = crop_borders(bgr_image, margin=10)
        assert cropped.shape[0] < bgr_image.shape[0]
        assert cropped.shape[1] < bgr_image.shape[1]


# ===========================================================
# Enhancement Tests
# ===========================================================

class TestEnhancement:

    @pytest.mark.parametrize("mode", ["adaptive", "otsu", "clahe", "none"])
    def test_modes_run_without_error(self, bgr_image, mode):
        result = enhance_document(bgr_image, mode=mode)
        assert result is not None
        assert result.size > 0

    def test_adaptive_returns_grayscale(self, bgr_image):
        result = enhance_document(bgr_image, mode="adaptive")
        assert result.ndim == 2, "Adaptive mode should return 2D grayscale"

    def test_otsu_returns_binary(self, bgr_image):
        result = enhance_document(bgr_image, mode="otsu")
        unique = np.unique(result)
        assert set(unique).issubset({0, 255}), "Otsu must produce binary output"

    def test_clahe_returns_bgr(self, bgr_image):
        result = enhance_document(bgr_image, mode="clahe")
        assert result.ndim == 3, "CLAHE should return 3-channel image"

    def test_none_returns_unchanged(self, bgr_image):
        result = enhance_document(bgr_image, mode="none")
        assert np.array_equal(result, bgr_image)

    def test_invalid_mode_raises(self, bgr_image):
        with pytest.raises(ValueError):
            enhance_document(bgr_image, mode="invalid_mode")

    def test_sharpen_same_shape(self, bgr_image):
        result = sharpen(bgr_image)
        assert result.shape == bgr_image.shape

    def test_denoise_same_shape(self):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = denoise(gray)
        assert result.shape == gray.shape


# ===========================================================
# Run
# ===========================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
