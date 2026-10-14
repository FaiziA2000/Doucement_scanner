# Document Scanner — Computer Vision Project

Detects document edges, applies perspective correction, and enhances readability using OpenCV.

## Features
- Automatic document edge detection (Canny + contour analysis)
- Four-point perspective transform (top-down correction)
- Multiple enhancement modes: Adaptive, Otsu, CLAHE
- Debug visualisation pipeline
- Full unit + integration test suite

## Project Structure
```
document_scanner/
├── src/
│   └── scanner.py          # Main DocumentScanner class
├── utils/
│   ├── edge_detection.py   # Canny edges + contour finding
│   ├── perspective.py      # Perspective warp
│   └── enhancement.py      # Adaptive / Otsu / CLAHE
├── tests/
│   └── test_scanner.py     # pytest suite
├── samples/                # Input images
├── output/                 # Scanned outputs
├── demo.py                 # Demo with synthetic image
└── requirements.txt
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Python API
```python
from src.scanner import DocumentScanner

scanner = DocumentScanner(enhance_mode="adaptive")
result = scanner.scan("samples/photo.jpg", "output/scanned.jpg")
```

### CLI
```bash
python src/scanner.py samples/photo.jpg -o output/scanned.jpg --mode adaptive
python src/scanner.py samples/photo.jpg --mode clahe --debug
```

### Demo (no image required)
```bash
python demo.py --mode adaptive
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Enhancement Modes
| Mode     | Best For                        |
|----------|---------------------------------|
| adaptive | Uneven lighting, shadows        |
| otsu     | Clean, uniform lighting         |
| clahe    | Preserving grey/colour levels   |
| none     | Raw perspective output          |
# Doucement_scanner
