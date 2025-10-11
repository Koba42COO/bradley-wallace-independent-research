#!/usr/bin/env python3
"""
Analyze the 10^10 scale results from the Wallace Transform framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_10billion_results():
    """Analyze the 455 million prime results"""

    csv_file = Path("wallace_run_12_export.csv")

    if not csv_file.exists():
        print("❌ CSV file not found")
        return

    # Read the CSV
    df = pd.read_csv(csv_file)
    print("📊 RAW DATA ANALYSIS")
    print("=" * 50)
    print(f"Total detections: {len(df)}")
    print(f"FFT detections: {len(df[df['Method'] == 'fft'])}")
    print(f"Autocorr detections: {len(df[df['Method'] == 'autocorr'])}")
    print()

    # Analyze FFT results
    fft_data = df[df['Method'] == 'fft']
    print("🎯 FFT ANALYSIS")
    print("-" * 30)
    print("FFT-detected ratios:")
    for _, row in fft_data.iterrows():
        ratio_val = row['Ratio Value']
        distance = row['Distance']
        print(".6f")
    print()

    # Analyze autocorrelation results
    autocorr_data = df[df['Method'] == 'autocorr']
    print("🔄 AUTOCORRELATION ANALYSIS")
    print("-" * 35)
    unique_ratios = autocorr_data['Ratio Symbol'].unique()
    print(f"Unique ratios detected: {len(unique_ratios)}")
    print("Autocorr-detected ratios:")
    for ratio in unique_ratios:
        ratio_data = autocorr_data[autocorr_data['Ratio Symbol'] == ratio]
        count = len(ratio_data)
        best_corr = ratio_data['Correlation'].max()
        print("2d")

        # Show lag values for this ratio
        lags = ratio_data['Lag'].values[:5]  # First 5 lags
        print(f"      Lags: {lags}")
    print()

    # Expected ratios analysis
    KNOWN_RATIOS = {
        '1.000': 'Unity',
        '1.414': '√2 (Octave)',
        '1.618': 'φ (Golden)',
        '1.732': '√3 (Fifth)',
        '1.847': 'Pell',
        '2.000': 'Octave',
        '2.287': 'φ·√2',
        '3.236': '2φ'
    }

    print("🎯 DETECTION SUMMARY")
    print("-" * 25)
    detected_fft = set(fft_data['Ratio Symbol'].unique())
    detected_autocorr = set(autocorr_data['Ratio Symbol'].unique())
    detected_total = detected_fft.union(detected_autocorr)

    print(f"FFT detected: {len(detected_fft)} ratios")
    print(f"Autocorr detected: {len(detected_autocorr)} ratios")
    print(f"Total unique: {len(detected_total)} ratios")
    print()

    print("Ratio Status:")
    for symbol, name in KNOWN_RATIOS.items():
        in_fft = symbol in detected_fft
        in_autocorr = symbol in detected_autocorr
        status = "✓ DETECTED" if (in_fft or in_autocorr) else "✗ PENDING"
        methods = []
        if in_fft: methods.append("FFT")
        if in_autocorr: methods.append("Autocorr")
        method_str = "/".join(methods) if methods else "-"
        print("6s")
    print()

    # Analyze why FFT is only detecting near-unity ratios
    print("🔍 FFT DIAGNOSTICS")
    print("-" * 20)
    fft_ratios = fft_data['Ratio Value'].values
    print(f"FFT ratio range: {fft_ratios.min():.6f} - {fft_ratios.max():.6f}")
    print(f"FFT ratio spread: {fft_ratios.max() - fft_ratios.min():.6f}")
    print(".6f")

    # Check if FFT ratios are close to harmonics of each other
    if len(fft_ratios) > 1:
        ratios_relative = fft_ratios / fft_ratios[0]
        print("FFT ratios relative to fundamental:")
        for i, rel in enumerate(ratios_relative):
            print(".6f")

        # Check if they form a harmonic series
        harmonic_check = []
        for i in range(1, len(ratios_relative)):
            ratio = ratios_relative[i]
            # Check if close to integer harmonics
            for h in range(2, 10):
                if abs(ratio - h) < 0.01:
                    harmonic_check.append(f"Peak {i+1} ≈ {h}x fundamental")
                    break
        if harmonic_check:
            print("Harmonic relationships found:")
            for check in harmonic_check:
                print(f"  {check}")
        else:
            print("No clear harmonic relationships found")
    print()

    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 18)
    print("1. INCREASE SAMPLE SIZES:")
    print("   - FFT: 100K → 1M (10x larger)")
    print("   - Autocorr: 50K → 500K (10x larger)")
    print("   - This should detect φ and √3")
    print()
    print("2. FFT PREPROCESSING:")
    print("   - Apply log transformation before FFT")
    print("   - Filter out low-frequency trends")
    print("   - Focus on mid-range frequencies")
    print()
    print("3. BRADLEY'S FORMULA TEST:")
    print("   - Test g_n = W_φ(p_n) · φ^k directly")
    print("   - No sampling limitations")
    print("   - Should detect φ definitively")
    print()
    print("4. MULTI-SCALE ANALYSIS:")
    print("   - Run multiple passes with different samples")
    print("   - Cross-validate results")

if __name__ == "__main__":
    analyze_10billion_results()
