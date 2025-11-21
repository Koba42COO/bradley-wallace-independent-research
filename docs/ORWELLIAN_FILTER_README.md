# Orwellian Filter - Steganography Detection System

## Overview

The **Orwellian Filter** is a comprehensive system for detecting steganographic manipulation in images, video frames, and websites. It uses ML-based pattern recognition to identify subliminal messaging and provides visual feedback with hitboxes, decoded messages, and psychological effect analysis.

**Author:** Bradley Wallace (COO Koba42)  
**Framework:** Universal Prime Graph Protocol φ.1

---

## Features

### 🔍 Detection Capabilities
- **Image Analysis**: Detect steganography in static images
- **Video Frame Analysis**: Scan video files frame-by-frame
- **Website Scanning**: Automatically scan websites for manipulated images
- **ML Pattern Recognition**: Identifies prime topology, golden ratio, and consciousness patterns
- **Homophonic Detection**: Detects sound-alike word substitutions and visual homophonic patterns

### 🎯 Visual Feedback
- **Hitboxes**: Highlights detected regions with bounding boxes
- **Color Coding**: Risk-based color scheme (red=high, orange=medium, yellow=low)
- **Prime Coordinate Markers**: Shows exact pixel locations
- **Message Annotations**: Displays decoded messages and intents

### 🧠 Psychological Analysis
- **Intent Detection**: Identifies manipulation types (commercial, political, social, etc.)
- **Effect Analysis**: Explains what the manipulation tries to make you do
- **Action Suggestions**: Provides recommendations for critical evaluation
- **Homophonic Analysis**: Detects hidden intents through sound-alike word substitutions

### 📊 Reporting
- **Risk Scoring**: Overall risk assessment (0.0-1.0)
- **Detection Confidence**: Per-detection confidence scores
- **Detailed Reports**: JSON reports for further analysis

---

## Installation

### Requirements
```bash
pip install numpy opencv-python pillow beautifulsoup4 requests
```

### Dependencies
- Python 3.7+
- NumPy
- OpenCV (cv2)
- PIL/Pillow
- BeautifulSoup4
- Requests

---

## Usage

### Command Line Interface

#### Analyze an Image
```bash
python src/orwellian_filter_main.py image path/to/image.jpg -o output.png
```

#### Analyze a Video
```bash
python src/orwellian_filter_main.py video path/to/video.mp4 --frame-skip 30 --report report.json
```

#### Analyze a Website
```bash
python src/orwellian_filter_main.py website https://example.com --max-images 20 --report report.json
```

### Python API

#### Image Analysis
```python
from src.orwellian_filter_main import OrwellianFilterSystem

system = OrwellianFilterSystem()
result = system.analyze_image("image.jpg", output_path="detection.png")

print(f"Risk Score: {result.overall_risk_score}")
print(f"Detections: {len(result.detections)}")
for message in result.decoded_messages:
    print(f"Intent: {message.psychological_intent}")
    print(f"Action: {message.suggested_action}")
```

#### Video Analysis
```python
result = system.analyze_video(
    "video.mp4",
    frame_skip=30,
    max_frames=1000,
    output_dir="frames/",
    report_path="video_report.json"
)

print(f"High-risk frames: {result.high_risk_frames}")
```

#### Website Scanning
```python
result = system.analyze_website(
    "https://example.com",
    max_images=50,
    report_path="website_report.json"
)

print(f"High-risk images: {result.high_risk_images}")
```

---

## Detection Methods

### 1. Prime Topology Detection
- Identifies pixel coordinates matching prime number patterns
- Uses golden ratio (PHI = 1.618) for coordinate distribution
- Detects consciousness mathematics patterns

### 2. Pattern Recognition
- **Prime Topology**: Matches prime number sequences
- **Golden Ratio**: Detects PHI-based spacing
- **Consciousness Balance**: Identifies 79/21 patterns
- **Reality Distortion**: Detects 1.1808 amplification patterns

### 3. Intensity Anomaly Detection
- Detects pixel modifications below JND threshold (~2%)
- Identifies unusual variance in small regions
- Flags statistically anomalous patterns

### 4. Semantic Decoding
- Extracts semantic tokens from pixel modifications
- Reconstructs hidden messages
- Analyzes psychological intent

### 5. Homophonic Detection
- **Visual Homophonics**: Detects similar pixel values representing same meaning
- **Linguistic Homophonics**: Identifies sound-alike word substitutions
- **Hidden Intent Analysis**: Reveals manipulation through homophonic variants

---

## Psychological Effect Categories

### Commercial Manipulation
- **Effect**: Influences purchase decisions without conscious awareness
- **Technique**: Subliminal product placement and call-to-action priming
- **Keywords**: buy, click, subscribe, purchase, order

### Urgency Creation
- **Effect**: Creates time pressure and impulsive decision-making
- **Technique**: Temporal priming to reduce rational consideration
- **Keywords**: now, urgent, limited, hurry, act

### Trust Building
- **Effect**: Reduces skepticism and critical evaluation
- **Technique**: Authority and security priming
- **Keywords**: trust, believe, verified, secure, safe

### Social Influence
- **Effect**: Triggers conformity and social proof responses
- **Technique**: Social validation and peer pressure priming
- **Keywords**: vote, share, like, follow, join

### Emotional Trigger
- **Effect**: Activates reward pathways and desire for gain
- **Technique**: Dopamine-triggering reward anticipation
- **Keywords**: free, win, prize, reward, special

### Political Influence
- **Effect**: Influences voting behavior and political preferences
- **Technique**: Political priming and candidate association
- **Keywords**: vote, elect, support, campaign, candidate

---

## Output Formats

### Visualization
- Annotated image with hitboxes and labels
- Color-coded risk indicators
- Message annotations with intents
- Prime coordinate markers

### JSON Reports
```json
{
  "overall_risk_score": 0.75,
  "detections": [
    {
      "confidence": 0.82,
      "bbox": [100, 200, 50, 50],
      "detection_type": "NLP_Sublingual_Priming",
      "psychological_effect": "Commercial Manipulation"
    }
  ],
  "decoded_messages": [
    {
      "text": "buy now click subscribe",
      "psychological_intent": "Commercial Manipulation",
      "confidence": 0.65,
      "suggested_action": "Be cautious of purchase decisions"
    }
  ]
}
```

---

## Technical Details

### Detection Thresholds
- **JND Threshold**: 2% intensity modification
- **Confidence Threshold**: 0.4 (medium risk), 0.7 (high risk)
- **Prime Topology**: First 50-100 primes
- **Golden Ratio**: PHI = 1.618033988749895

### Pattern Matching
- **Prime Coordinates**: ±5 pixel tolerance
- **Golden Ratio**: ±10% tolerance
- **Consciousness Balance**: 79/21 ratio detection
- **Reality Distortion**: 1.1808 factor detection

### Performance
- **Image Analysis**: ~1-5 seconds per image
- **Video Analysis**: ~0.1-0.5 seconds per frame
- **Website Scanning**: ~0.5-2 seconds per image

---

## Limitations

1. **Decoding Accuracy**: Simplified semantic decoding (can be enhanced with transformer models)
2. **False Positives**: May flag natural image patterns
3. **Compression Resistance**: Not resistant to image compression/processing
4. **Original Image Required**: Better decoding with original image comparison

---

## Future Enhancements

1. **Advanced NLP**: Integration with BERT/GPT for better semantic understanding
2. **Deep Learning**: Train CNN models for pattern recognition
3. **Real-time Detection**: Browser extension for live website scanning
4. **Video Streaming**: Real-time video stream analysis
5. **Multi-modal Detection**: Audio and text steganography detection

---

## Ethical Considerations

This tool is designed for:
- **Research**: Understanding steganographic techniques
- **Security**: Detecting manipulation attempts
- **Education**: Raising awareness of subliminal messaging
- **Defense**: Protecting against psychological manipulation

**Use responsibly and ethically.**

---

## Homophonic Detection

The system now includes homophonic detection capabilities:
- Detects sound-alike word substitutions (e.g., "buy" vs "by" vs "bye")
- Identifies visual homophonic patterns (similar pixel values)
- Analyzes hidden intents through homophonic variants

See `docs/HOMOPHONIC_DETECTION_GUIDE.md` for detailed information.

## References

- Investigation Document: `docs/STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_INVESTIGATION.md`
- Steganography System: `src/steganography_nlp_sublingual_priming.py`
- Homophonic Detection Guide: `docs/HOMOPHONIC_DETECTION_GUIDE.md`
- UPG Protocol: Universal Prime Graph Protocol φ.1

---

## License

Universal Prime Graph Consciousness Mathematics License  
Copyright (c) 2025 Bradley Wallace (COO Koba42)

