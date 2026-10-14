"""
demo.py — Generate a synthetic test document image and run the full pipeline.

Usage:
    python demo.py
    python demo.py --mode clahe
    python demo.py --mode otsu --debug
"""

import cv2
import numpy as np
import os
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.scanner import DocumentScanner


def create_synthetic_document(output_path: str, noise: bool = True) -> np.ndarray:
    """
    Create a synthetic 'document on a desk' image for testing when no
    real scan is available.
    """
    # Dark desktop background
    bg = np.full((720, 960, 3), (45, 38, 30), dtype=np.uint8)

    # White document (slightly rotated) placed on background
    doc_pts_src = np.array(
        [[150, 80], [810, 120], [830, 640], [130, 610]], dtype=np.float32
    )
    doc_pts_dst = np.array(
        [[0, 0], [600, 0], [600, 800], [0, 800]], dtype=np.float32
    )

    # Draw text lines on a white document
    doc = np.full((800, 600, 3), 250, dtype=np.uint8)
    cv2.putText(doc, "DOCUMENT SCANNER", (60, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 2)
    cv2.putText(doc, "Computer Vision Project", (80, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
    for i, y in enumerate(range(180, 700, 45)):
        line = f"Sample text line {i + 1} — lorem ipsum dolor sit amet."
        cv2.putText(doc, line, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

    # Draw a table
    cv2.rectangle(doc, (30, 720), (570, 780), (100, 100, 100), 1)
    for x in [200, 390]:
        cv2.line(doc, (x, 720), (x, 780), (100, 100, 100), 1)
    cv2.putText(doc, "Column A", (50, 756), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
    cv2.putText(doc, "Column B", (225, 756), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
    cv2.putText(doc, "Column C", (410, 756), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)

    # Warp document onto background using inverse perspective
    M = cv2.getPerspectiveTransform(doc_pts_dst, doc_pts_src)
    warped_doc = cv2.warpPerspective(doc, M, (960, 720))

    # Composite: overlay document on background
    mask = cv2.warpPerspective(
        np.full((800, 600), 255, dtype=np.uint8),
        M, (960, 720)
    )
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    composite = np.where(mask_3 > 0, warped_doc, bg)

    if noise:
        noise_arr = np.random.randint(0, 20, composite.shape, dtype=np.uint8)
        composite = cv2.add(composite, noise_arr)
        composite = cv2.GaussianBlur(composite, (3, 3), 0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, composite)
    print(f"[Demo] Synthetic document saved: {output_path}")
    return composite


def run_demo(mode: str = "adaptive", debug: bool = False):
    base = Path(__file__).parent

    input_path = str(base / "samples" / "synthetic_doc.jpg")
    output_path = str(base / "output" / f"synthetic_doc_scanned_{mode}.jpg")

    # Generate synthetic input
    create_synthetic_document(input_path)

    # Run scanner
    scanner = DocumentScanner(enhance_mode=mode, debug=debug)
    result = scanner.scan(input_path, output_path)

    print(f"\n[Demo] Pipeline complete!")
    print(f"       Input : {input_path}")
    print(f"       Output: {output_path}")
    print(f"       Size  : {result.shape[1]}x{result.shape[0]}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="adaptive",
                        choices=["adaptive", "otsu", "clahe", "none"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_demo(args.mode, args.debug)
