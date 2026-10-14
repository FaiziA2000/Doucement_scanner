"""
Document Scanner - Main Pipeline
=================================
Detects document edges, applies perspective correction,
and enhances readability using OpenCV.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.edge_detection import detect_edges, find_document_contour
from utils.perspective import four_point_transform
from utils.enhancement import enhance_document


class DocumentScanner:
    """
    End-to-end document scanning pipeline.

    Steps:
        1. Load image
        2. Preprocess (resize + grayscale + blur)
        3. Edge detection
        4. Contour / corner finding
        5. Perspective correction
        6. Image enhancement
        7. Save / return result
    """

    def __init__(self,
                 target_width: int = 800,
                 enhance_mode: str = "adaptive",
                 debug: bool = False):
        """
        Args:
            target_width: Width used for internal processing (keeps aspect ratio).
            enhance_mode: 'adaptive' | 'otsu' | 'clahe' | 'none'
            debug: If True, saves intermediate images to output/debug/.
        """
        self.target_width = target_width
        self.enhance_mode = enhance_mode
        self.debug = debug

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, image_path: str, output_path: str | None = None) -> np.ndarray:
        """
        Full scan pipeline.

        Args:
            image_path: Path to the input image.
            output_path: Optional path to save the scanned result.

        Returns:
            Scanned (perspective-corrected + enhanced) image as numpy array.
        """
        image_path = str(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # --- Step 1: Load -----------------------------------------------
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image: {image_path}")
        print(f"[Scanner] Loaded image  : {image_path}  {original.shape[1]}x{original.shape[0]}")

        # --- Step 2: Preprocess -----------------------------------------
        ratio = original.shape[0] / self.target_width
        resized = self._resize(original, width=self.target_width)
        self._debug_save(resized, "1_resized", image_path)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self._debug_save(blurred, "2_blurred", image_path)

        # --- Step 3: Edge detection -------------------------------------
        edged = detect_edges(blurred)
        self._debug_save(edged, "3_edges", image_path)
        print(f"[Scanner] Edge detection : done")

        # --- Step 4: Find document contour / corners --------------------
        corners = find_document_contour(edged)

        if corners is None:
            print("[Scanner] WARNING: Could not detect document corners — returning enhanced original.")
            result = self._enhance(original)
        else:
            # Scale corners back to original resolution
            corners_orig = corners * ratio
            print(f"[Scanner] Corners found  : {corners_orig.astype(int).tolist()}")

            # --- Step 5: Perspective correction -------------------------
            warped = four_point_transform(original, corners_orig)
            self._debug_save(warped, "4_warped", image_path)
            print(f"[Scanner] Warped size    : {warped.shape[1]}x{warped.shape[0]}")

            # --- Step 6: Enhancement ------------------------------------
            result = self._enhance(warped)

        self._debug_save(result, "5_final", image_path)
        print(f"[Scanner] Enhancement    : {self.enhance_mode}")

        # --- Step 7: Save -----------------------------------------------
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result)
            print(f"[Scanner] Saved result   : {output_path}")

        return result

    def scan_from_array(self, image: np.ndarray) -> np.ndarray:
        """Scan directly from a numpy array (useful for webcam frames)."""
        import tempfile, uuid
        tmp = os.path.join(tempfile.gettempdir(), f"scan_{uuid.uuid4().hex}.jpg")
        cv2.imwrite(tmp, image)
        result = self.scan(tmp)
        os.remove(tmp)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resize(self, image: np.ndarray, width: int) -> np.ndarray:
        h, w = image.shape[:2]
        new_h = int(h * width / w)
        return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        return enhance_document(image, mode=self.enhance_mode)

    def _debug_save(self, image: np.ndarray, name: str, source_path: str):
        if not self.debug:
            return
        debug_dir = os.path.join(
            os.path.dirname(source_path), "..", "output", "debug"
        )
        os.makedirs(debug_dir, exist_ok=True)
        stem = Path(source_path).stem
        path = os.path.join(debug_dir, f"{stem}_{name}.jpg")
        cv2.imwrite(path, image)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document Scanner CLI")
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("-o", "--output", default=None, help="Path to save output")
    parser.add_argument("--mode", default="adaptive",
                        choices=["adaptive", "otsu", "clahe", "none"],
                        help="Enhancement mode")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    args = parser.parse_args()

    scanner = DocumentScanner(enhance_mode=args.mode, debug=args.debug)

    if args.output is None:
        stem = Path(args.input).stem
        args.output = f"output/{stem}_scanned.jpg"

    scanner.scan(args.input, args.output)
    print("[Scanner] Done!")
