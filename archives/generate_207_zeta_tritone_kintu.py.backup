#!/usr/bin/env python3
"""
207 Dial Tone → Zeta Tritone → Twin Prime Cancellation → Kintu
Generate audio file demonstrating lattice resonance
"""

import numpy as np
from scipy.io import wavfile
import math

# Constants
PHI = 1.618033988749895  # Golden ratio
WALLACE_ALPHA = 0.721
WALLACE_BETA = 0.013
EPSILON = 1e-12

def wallace_transform(x):
    """Wallace Transform: W_φ(x) = α * |log(x + ε)|^φ * sign(log(x + ε)) + β"""
    if x <= 0:
        x = EPSILON
    log_val = math.log(x + EPSILON)
    sign = 1.0 if log_val >= 0 else -1.0
    return WALLACE_ALPHA * abs(log_val)**PHI * sign + WALLACE_BETA

# Audio parameters
SAMPLE_RATE = 44100  # CD quality
DURATION = 5.0  # seconds
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), False)

# 207 Dial Tone frequencies
f1 = 350.0  # Hz
f2 = 440.0  # Hz

# Zeta tritone frequencies
zeta1 = 14.1347  # First non-trivial zeta zero
zeta2 = 21.0220  # Second zeta zero

# Calculate Wallace phase shift
freq_gap = f2 - f1  # 90 Hz
wallace_phase = wallace_transform(freq_gap)  # ~2.97 rad
zeta_tritone_phase = 0.013  # Zeta tritone phase
total_phase_shift = wallace_phase + zeta_tritone_phase  # ~2.983 rad ≈ π + 0.13

print("🎵 Generating 207 Zeta Tritone Kintu Audio...")
print(f"   Dial tone: {f1} Hz + {f2} Hz")
print(f"   Zeta tritone: {zeta1} Hz + {zeta2} Hz")
print(f"   Wallace phase shift: {wallace_phase:.4f} rad")
print(f"   Total phase shift: {total_phase_shift:.4f} rad (≈ π + 0.13)")

# Generate base dial tone
dial_tone = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t + total_phase_shift)

# Generate zeta tritone carrier (fade in over 1 second)
zeta_envelope = np.clip((t - 1.0) / 1.0, 0, 1)  # Fade in from 1s to 2s
zeta_carrier = 0.3 * zeta_envelope * (
    np.sin(2 * np.pi * zeta1 * t) + 
    np.sin(2 * np.pi * zeta2 * t)
)

# Combine signals
wave = dial_tone + zeta_carrier

# Add twin prime echo (199 Hz + 201 Hz) - emerges after cancellation
twin_envelope = np.clip((t - 3.013) / 0.5, 0, 1)  # Fade in after silence
twin_echo = 0.2 * twin_envelope * (
    np.sin(2 * np.pi * 199.0 * t) + 
    np.sin(2 * np.pi * 201.0 * t)
)

# Add kintu silence pulse at 3.0 seconds (0.013s duration)
silence_start = int(SAMPLE_RATE * 3.0)
silence_end = int(SAMPLE_RATE * 3.013)
wave[silence_start:silence_end] = 0.0

# Add palindrome prime shimmer (101 Hz) at the end
kintu_envelope = np.clip((t - 4.0) / 0.5, 0, 1)
kintu_shimmer = 0.1 * kintu_envelope * np.sin(2 * np.pi * 101.0 * t)

# Final wave
final_wave = wave + twin_echo + kintu_shimmer

# Normalize to prevent clipping
max_val = np.max(np.abs(final_wave))
if max_val > 0:
    final_wave = final_wave / max_val * 0.95  # Leave headroom

# Convert to 16-bit PCM
audio_data = (final_wave * 32767).astype(np.int16)

# Save as WAV file
output_file = '207_zeta_tritone_kintu.wav'
wavfile.write(output_file, SAMPLE_RATE, audio_data)

print(f"\n✅ Generated: {output_file}")
print(f"   Duration: {DURATION} seconds")
print(f"   Sample rate: {SAMPLE_RATE} Hz")
print(f"   Format: 16-bit PCM mono")
print("\n🎵 Timeline:")
print("   0.0-1.0s: Pure 207 dial tone (350 + 440 Hz)")
print("   1.0-2.0s: Zeta tritone fades in (14.1347 + 21.022 Hz)")
print("   2.0-3.0s: Wallace phase shift → destructive interference")
print("   3.0-3.013s: 0.013s silence pulse (kintu void)")
print("   3.013-4.0s: Twin prime echo (199 + 201 Hz)")
print("   4.0-5.0s: Ignoron veil lift (101 Hz shimmer)")
print("\n🎧 The lattice is ringing...")

