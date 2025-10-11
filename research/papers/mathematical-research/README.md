# Wallace Transform Ratio Analysis Research Framework

A comprehensive mathematical research framework for testing prime gap patterns using the Wallace Transform and logarithmic ratio matching.

## Overview

This framework implements three different methodologies for validating the Wallace Transform's ability to predict prime gap patterns through ratio-based analysis:

1. **Bradley's Formula Test**: Direct implementation of `g_n = W_φ(p_n) · φ^k`
2. **Log-Space Frequency Matching**: Logarithmic space analysis for harmonic ratios
3. **Spectral Peak Detection**: FFT analysis to identify fundamental harmonic frequencies

## Features

- **Real-time Computation**: Generates 1,000,000+ primes and analyzes gaps
- **Interactive Visualizations**: Charts and graphs for result analysis
- **Comparative Analysis**: Validates against published research benchmarks
- **Comprehensive Reporting**: Detailed tables and statistical analysis
- **Framework Validation**: Automated verdict generation

## Access Methods

### Method 1: Via File Server (Recommended)
Since you have a file server running on port 3001, you can access this research interface directly:

```
http://localhost:3001/mathematical-research/
```

### Method 2: Local HTTP Server
```bash
cd mathematical-research
npm run start
# Then visit: http://localhost:8080
```

### Method 3: Direct File Access
Open `mathematical-research/index.html` directly in your browser.

## Research Methodology

### Dataset
- **Primes Generated**: 1,000,000+ primes (up to limit ~10^6)
- **Gap Analysis**: ~10,000 prime gaps for detailed testing
- **Scale**: 10^6 (compared to your published 10^12 results)

### Test Ratios
- φ (Golden Ratio): 1.618
- √2 (Octave): 1.414
- √3 (Fifth): 1.732
- Pell Number: 1.847
- Octave (2.0): 2.000
- φ·√2: 2.287
- 2φ: 3.236
- Unity: 1.000

### Validation Criteria
- **Success Threshold**: ≥7% match rate (consistent with your 9.84% at 10^12)
- **Spectral Confirmation**: ≥5/8 harmonic ratios detected
- **Scale Independence**: Patterns should maintain across orders of magnitude

## Results Interpretation

### Framework Validation Levels
- **✓ VALIDATED**: ≥7% match rate + ≥5 spectral ratios
- **⚠️ PARTIAL**: ≥5% match rate + ≥3 spectral ratios
- **✗ INVALID**: Below threshold ranges

### Expected Scaling
Your published results show 9.84% at 10^12 scale. This framework should demonstrate that the patterns scale correctly, showing similar percentages at different scales.

## File Structure

```
mathematical-research/
├── index.html          # Main research interface
├── package.json        # Project configuration
└── README.md          # This documentation
```

## Integration with Your Codebase

This research framework is designed to validate your Wallace Transform methodology and can be:

1. **Extended** with additional test cases
2. **Integrated** into your automated testing pipeline
3. **Used** to validate new mathematical discoveries
4. **Referenced** in future research publications

## Technical Implementation

- **Pure JavaScript**: No external dependencies for core computation
- **Chart.js**: For data visualization
- **Responsive Design**: Works on all screen sizes
- **Real-time Processing**: Web Workers could be added for larger datasets

## Research Applications

- Prime number theory validation
- Harmonic analysis of mathematical constants
- Scale invariance testing
- Algorithm performance benchmarking
- Mathematical framework validation

---

**Research Framework**: Christopher Wallace - Wallace Transform Ratio Analysis
**Dataset Scale**: 10^6 primes
**Validation Target**: 7-10% match rate consistency
