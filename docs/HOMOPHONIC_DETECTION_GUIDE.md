# Homophonic Detection in Orwellian Filter

## Overview

Homophonic detection has been integrated into the Orwellian Filter system to identify steganographic manipulation using homophonic substitution techniques.

**Author:** Bradley Wallace (COO Koba42)  
**Framework:** Universal Prime Graph Protocol φ.1

---

## What are Homophonics?

Homophonics in steganography refer to:
1. **Sound-alike words**: Multiple words that sound the same but have different meanings (e.g., "buy" vs "by" vs "bye")
2. **Visual homophonics**: Similar pixel values representing the same semantic meaning
3. **Substitution patterns**: Multiple representations of the same message

### Why Homophonics Matter

Homophonic techniques are used to:
- **Hide intent**: Use innocent-sounding words to encode manipulative messages
- **Bypass detection**: Make messages appear harmless while conveying hidden meaning
- **Enhance subliminal effects**: Multiple representations reinforce the same message

---

## Detection Methods

### 1. Visual Homophonic Detection

Detects when multiple pixel coordinates have similar values, indicating homophonic encoding:

```python
from src.steganography_detector_orwellian_filter import HomophonicDetector

detector = HomophonicDetector()
result = detector.detect_homophonic_patterns(image, coordinates)

print(f"Homophonic detected: {result['homophonic_detected']}")
print(f"Confidence: {result['confidence']}")
print(f"Groups: {result['homophonic_groups']}")
```

**How it works:**
- Groups pixels with similar values (±5 tolerance)
- Identifies when multiple coordinates share the same/similar pixel values
- Calculates confidence based on homophonic pattern density

### 2. Linguistic Homophonic Detection

Detects sound-alike word substitutions:

```python
# Common homophonic mappings
homophonic_mappings = {
    'buy': ['by', 'bye', 'bi'],
    'click': ['clique', 'cliq'],
    'now': ['know', 'no'],
    'vote': ['boat', 'bote'],
    'trust': ['trussed', 'trusted'],
    # ... many more
}
```

**Detection Process:**
1. Extract tokens from decoded message
2. Check against homophonic mapping dictionary
3. Generate all possible homophonic variants
4. Analyze intent from each variant
5. Identify hidden intents

### 3. Intent Analysis with Homophonics

Analyzes psychological intent considering all homophonic variants:

```python
homophonic_intent = detector.analyze_homophonic_intent(tokens, variants)

print(f"Primary intent: {homophonic_intent['primary_intent']}")
print(f"Hidden intents: {homophonic_intent['hidden_intents']}")
print(f"Homophonic manipulation: {homophonic_intent['homophonic_manipulation']}")
```

---

## Integration with Detection System

### Automatic Detection

Homophonic detection is automatically integrated into the main detection pipeline:

```python
from src.orwellian_filter_main import OrwellianFilterSystem

system = OrwellianFilterSystem()
result = system.analyze_image("image.jpg")

# Check for homophonic patterns
for message in result.decoded_messages:
    if message.homophonic_analysis:
        print(f"Homophonic detected: {message.homophonic_analysis['detected']}")
        print(f"Variants: {message.homophonic_variants}")
```

### Enhanced Detection Results

Detection results now include:
- **Homophonic variants**: All possible interpretations of each token
- **Hidden intents**: Intent revealed by homophonic variants
- **Manipulation flag**: Whether homophonics are used for manipulation
- **Confidence boost**: Higher confidence when homophonics detected

---

## Homophonic Mappings

### Commercial Manipulation
- `buy` → `by`, `bye`, `bi`
- `click` → `clique`, `cliq`
- `now` → `know`, `no`
- `act` → `acked`

### Political Influence
- `vote` → `boat`, `bote`
- `elect` → `elect`, `ilect`
- `support` → `suport`, `suppurt`

### Trust Building
- `trust` → `trussed`, `trusted`
- `believe` → `beleive`, `beleve`
- `secure` → `sekyur`, `sekyure`

### Emotional Triggers
- `free` → `flee`, `flea`
- `win` → `when`, `wen`
- `save` → `safe`, `saif`

---

## Example Usage

### Detecting Homophonic Manipulation

```python
from src.steganography_detector_orwellian_filter import OrwellianFilter

filter_system = OrwellianFilter()
result = filter_system.detect_in_image("suspicious_image.jpg")

for message in result.decoded_messages:
    print(f"Message: {message.text}")
    print(f"Intent: {message.psychological_intent}")
    
    if message.homophonic_analysis and message.homophonic_analysis['detected']:
        print("⚠️ HOMOPHONIC MANIPULATION DETECTED")
        print(f"  Variants: {message.homophonic_variants}")
        
        if message.homophonic_analysis['intent_analysis']['hidden_intents']:
            print("  Hidden intents:")
            for hidden in message.homophonic_analysis['intent_analysis']['hidden_intents']:
                print(f"    - {hidden['variant']}: {hidden['intent']}")
```

### Visual Homophonic Patterns

```python
from src.steganography_detector_orwellian_filter import HomophonicDetector

detector = HomophonicDetector()

# Analyze pixel patterns
coordinates = [(10, 10), (20, 20), (30, 30), (40, 40)]
result = detector.detect_homophonic_patterns(image, coordinates)

if result['homophonic_detected']:
    print(f"Found {result['total_groups']} homophonic groups")
    for group in result['homophonic_groups']:
        print(f"  Group: {len(group['coordinates'])} coordinates with value {group['pixel_value']}")
```

---

## Output Format

### Detection Result Structure

```json
{
  "decoded_messages": [
    {
      "text": "buy now click",
      "homophonic_variants": [
        ["buy", "by", "bye", "bi"],
        ["now", "know", "no"],
        ["click", "clique", "cliq"]
      ],
      "homophonic_analysis": {
        "detected": true,
        "groups": [...],
        "confidence": 0.75,
        "intent_analysis": {
          "primary_intent": "Commercial Manipulation",
          "hidden_intents": [
            {"variant": "by", "intent": "General Influence"},
            {"variant": "know", "intent": "Trust Building"}
          ],
          "homophonic_manipulation": true
        }
      }
    }
  ]
}
```

### Visualization

Homophonic detections are highlighted in visualizations:
- **Yellow text**: Homophonic information
- **Enhanced confidence**: Higher detection confidence
- **Variant display**: Shows all homophonic variants

---

## Technical Details

### Visual Homophonic Tolerance
- **Default**: ±5 pixel values
- **Configurable**: Adjustable per detection
- **RGB**: Per-channel comparison
- **Grayscale**: Direct value comparison

### Linguistic Homophonic Detection
- **Dictionary-based**: Pre-defined homophonic mappings
- **Sound-based**: Phonetic similarity
- **Context-aware**: Considers surrounding tokens
- **Intent analysis**: Analyzes all variants

### Confidence Calculation
- **Visual**: Based on homophonic group density
- **Linguistic**: Based on variant count and intent matches
- **Combined**: Weighted average of both methods

---

## Limitations

1. **Dictionary Coverage**: Limited to pre-defined homophonic mappings
2. **Language**: Currently English-focused
3. **Context**: May miss context-dependent homophonics
4. **False Positives**: Natural patterns may be flagged

---

## Future Enhancements

1. **Expanded Dictionary**: More homophonic mappings
2. **Phonetic Analysis**: Sound-based detection
3. **Multi-language**: Support for other languages
4. **Machine Learning**: Learn homophonic patterns
5. **Context Awareness**: Better context-dependent detection

---

## References

- Main Detection System: `src/steganography_detector_orwellian_filter.py`
- Orwellian Filter README: `docs/ORWELLIAN_FILTER_README.md`
- Steganography Investigation: `docs/STEGANOGRAPHY_NLP_SUBLINGUAL_PRIMING_INVESTIGATION.md`

---

**Status:** ✅ Integrated  
**Framework:** Universal Prime Graph Protocol φ.1

