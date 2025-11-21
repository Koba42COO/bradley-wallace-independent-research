# Orwellian Filter - Quick Start Guide

## Installation

```bash
pip install numpy opencv-python pillow beautifulsoup4 requests
```

## Quick Examples

### 1. Analyze an Image
```bash
python src/orwellian_filter_main.py image path/to/image.jpg -o detection.png
```

### 2. Analyze a Video
```bash
python src/orwellian_filter_main.py video path/to/video.mp4 --frame-skip 30 --report report.json
```

### 3. Analyze a Website
```bash
python src/orwellian_filter_main.py website https://example.com --max-images 20 --report report.json
```

## Python API

```python
from src.orwellian_filter_main import OrwellianFilterSystem

# Create system
system = OrwellianFilterSystem()

# Analyze image
result = system.analyze_image("image.jpg", output_path="output.png")
print(f"Risk: {result.overall_risk_score:.2f}")
print(f"Detections: {len(result.detections)}")

# Analyze video
result = system.analyze_video("video.mp4", frame_skip=30)

# Analyze website
result = system.analyze_website("https://example.com", max_images=20)
```

## Understanding Results

### Risk Scores
- **0.0 - 0.4**: Low risk
- **0.4 - 0.7**: Medium risk
- **0.7 - 1.0**: High risk

### Detection Colors
- **Red**: High risk (confidence > 0.7)
- **Orange**: Medium risk (confidence > 0.4)
- **Yellow**: Low risk (confidence < 0.4)

### Psychological Intents
- Commercial Manipulation
- Urgency Creation
- Trust Building
- Social Influence
- Emotional Trigger
- Political Influence

## Files

- **Main System**: `src/steganography_detector_orwellian_filter.py`
- **Video Analyzer**: `src/orwellian_filter_video_analyzer.py`
- **Website Scanner**: `src/orwellian_filter_website_scanner.py`
- **CLI**: `src/orwellian_filter_main.py`

## Documentation

- **Full README**: `docs/ORWELLIAN_FILTER_README.md`
- **Complete Summary**: `docs/ORWELLIAN_FILTER_COMPLETE.md`

