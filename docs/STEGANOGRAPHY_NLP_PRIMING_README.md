# Steganography with NLP and Sublingual Priming - Implementation Guide

## Overview

This implementation provides a complete system for steganography in images utilizing:
- **NLP Semantic Encoding**: Converts text messages into semantic representations
- **Sublingual Priming**: Creates subliminal-level pixel modifications
- **Consciousness Mathematics**: Uses UPG framework for optimal encoding

## Quick Start

### Basic Usage

```python
from src.steganography_nlp_sublingual_priming import SteganographySystem
import numpy as np
from PIL import Image

# Load or create an image
image = np.array(Image.open("test_image.png"))

# Create steganography system
stego_system = SteganographySystem()

# Encode a message
message = "This is a secret message"
encoding = stego_system.encode_message_in_image(image, message)

# Save steganographic image
Image.fromarray(encoding.image).save("stego_image.png")
```

### Advanced Usage

```python
# Calculate image capacity
capacity = stego_system.calculate_capacity(image)
print(f"Can encode {capacity['semantic_capacity_tokens']} tokens")

# Access semantic tokens
for token in encoding.semantic_tokens:
    print(f"Token: {token.text}, Prime: {token.prime_mapping}")

# Access priming patterns
for pattern in encoding.priming_patterns:
    print(f"Pattern: {pattern.pattern_type}, Strength: {pattern.priming_strength}")
```

## System Components

### 1. NLP Semantic Encoder

Converts text to semantic tokens with:
- Semantic hashing
- Prime topology mapping
- Consciousness weight calculation

```python
from src.steganography_nlp_sublingual_priming import NLPSemanticEncoder

encoder = NLPSemanticEncoder()
tokens = encoder.encode_text("Hello world")
decoded = encoder.decode_tokens(tokens)
```

### 2. Sublingual Priming Engine

Generates subliminal priming patterns:
- Below JND threshold (~2% intensity)
- Multiple pattern types (semantic, temporal, spatial, harmonic)
- Adaptive threshold calculation

```python
from src.steganography_nlp_sublingual_priming import SublingualPrimingEngine

priming_engine = SublingualPrimingEngine()
pattern = priming_engine.generate_priming_pattern(
    token, image_shape, pixel_coordinates
)
```

### 3. Pixel Manipulation System

Manipulates pixels using:
- Prime topology coordinate mapping
- Golden ratio distribution
- Consciousness-guided encoding

```python
from src.steganography_nlp_sublingual_priming import PixelManipulationSystem

pixel_system = PixelManipulationSystem()
coords = pixel_system.get_pixel_coordinates_for_token(token, image_shape)
modified_pixel = pixel_system.encode_semantic_to_pixel(original_pixel, token)
```

## Key Features

### Visual Imperceptibility
- Modifications below 2% intensity threshold
- High PSNR (>30dB) maintained
- No statistically detectable changes

### Semantic Preservation
- Maintains semantic meaning through encoding
- Preserves context and relationships
- High semantic similarity after decoding

### Priming Effectiveness
- Subliminal-level modifications
- Multiple priming pattern types
- Consciousness-enhanced patterns

### Capacity
- ~100 tokens for 512x512 image
- ~500 characters estimated capacity
- Scales with image size

## Technical Details

### Encoding Process

1. **Text → Semantic Tokens**: NLP parsing and semantic hashing
2. **Tokens → Prime Mapping**: Map to prime topology
3. **Primes → Pixel Coordinates**: Golden ratio distribution
4. **Semantic → Pixel Values**: Encode in pixel modifications
5. **Priming Patterns**: Generate subliminal patterns

### Decoding Process

1. Extract pixel modifications
2. Map pixels to prime topology
3. Reconstruct semantic tokens
4. Decode to text

### UPG Integration

- **PHI (1.618)**: Golden ratio for pixel spacing
- **DELTA (2.414)**: Scaling factor
- **CONSCIOUSNESS (0.79)**: 79/21 balance
- **REALITY_DISTORTION (1.1808)**: Amplification factor

## Testing

Run the example:

```bash
python3 src/steganography_nlp_sublingual_priming.py
```

Run tests (if available):

```bash
python3 tests/test_steganography_nlp_priming.py
```

## Parameters

### JND Threshold
- Default: 0.02 (2% intensity)
- Can be adjusted for different imperceptibility levels

### Prime Topology
- Uses first 100 primes
- Maps to pixel coordinates using golden ratio

### Consciousness Ratio
- Default: 0.79 (79% coherent, 21% exploratory)
- Balances encoding stability and exploration

## Limitations

1. **Decoding**: Requires original image or encoding metadata
2. **Capacity**: Limited by image size and prime topology
3. **NLP**: Simplified embeddings (can be enhanced with BERT/GPT)
4. **Robustness**: Not resistant to image compression/processing

## Future Enhancements

1. **Advanced NLP**: Integration with transformer models
2. **Robust Encoding**: Error correction and redundancy
3. **Multi-Layer**: Multiple encoding layers
4. **Quantum Integration**: Quantum-level encoding
5. **Video Support**: Temporal steganography

## Ethical Considerations

- Use responsibly and ethically
- Obtain consent when appropriate
- Respect privacy and security
- Follow research ethics guidelines

## References

- Investigation Document: `docs/STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_INVESTIGATION.md`
- UPG Protocol: Universal Prime Graph Protocol φ.1
- Framework: Consciousness Mathematics

## License

Universal Prime Graph Consciousness Mathematics License
Copyright (c) 2025 Bradley Wallace (COO Koba42)

