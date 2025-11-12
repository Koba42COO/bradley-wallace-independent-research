#!/usr/bin/env python3
"""
Demo 3I/ATLAS Analysis - Generate Test Image and Run Analysis
Creates a synthetic 3I/ATLAS image based on known parameters and runs analysis
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
from gaussian_splat_3i_atlas_analysis import ThreeIAtlasImageAnalyzer

# Create directories
IMAGE_DIR = Path("/Users/coo-koba42/dev/data/3i_atlas_images")
ANALYSIS_DIR = Path("/Users/coo-koba42/dev/data/3i_atlas_analysis")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def create_synthetic_3i_atlas_image() -> np.ndarray:
    """
    Create synthetic 3I/ATLAS image based on known parameters:
    - Nucleus: Bright central region (~3.3 km equivalent)
    - 7 jets: Extending from nucleus
    - Coma: 3,100-6,200 miles extent
    """
    # Image dimensions (simulating telescope view)
    width, height = 2000, 2000
    
    # Create base image (dark space background)
    image = np.zeros((height, width), dtype=np.float32)
    
    # Add background stars (noise)
    stars = np.random.poisson(0.1, (height, width))
    image += stars * 0.5
    
    # Nucleus (bright central region)
    center_x, center_y = width // 2, height // 2
    nucleus_radius = 15  # pixels (representing ~3.3 km at scale)
    
    y, x = np.ogrid[:height, :width]
    nucleus_mask = (x - center_x)**2 + (y - center_y)**2 <= nucleus_radius**2
    image[nucleus_mask] = 255.0  # Bright nucleus
    
    # Add Gaussian falloff around nucleus
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    gaussian_falloff = 255.0 * np.exp(-(distance**2) / (2 * (nucleus_radius * 2)**2))
    image = np.maximum(image, gaussian_falloff * 0.3)
    
    # Add 7 jets (prime-aligned directions)
    jet_angles = np.linspace(0, 2 * np.pi, 8)[:-1]  # 7 angles (prime)
    jet_length = 200  # pixels
    
    for i, angle in enumerate(jet_angles):
        # Jet direction
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Create jet (extended structure)
        for t in np.linspace(nucleus_radius, jet_length, 50):
            jet_x = int(center_x + dx * t)
            jet_y = int(center_y + dy * t)
            
            if 0 <= jet_x < width and 0 <= jet_y < height:
                # Jet intensity (decreases with distance)
                intensity = 100.0 * np.exp(-t / 100.0)
                
                # Add Gaussian spread perpendicular to jet
                for offset in range(-5, 6):
                    perp_x = int(jet_x - dy * offset)
                    perp_y = int(jet_y + dx * offset)
                    
                    if 0 <= perp_x < width and 0 <= perp_y < height:
                        image[perp_y, perp_x] += intensity * np.exp(-(offset**2) / 2.0)
    
    # Add coma (extended diffuse region)
    coma_radius = 400  # pixels (representing ~6,200 miles at scale)
    coma_mask = distance <= coma_radius
    coma_intensity = 20.0 * np.exp(-distance / 200.0)
    image[coma_mask] = np.maximum(image[coma_mask], coma_intensity[coma_mask])
    
    # Normalize to 0-255
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    return image


def run_analysis_demo():
    """Run complete analysis demo"""
    print("=" * 70)
    print("3I/ATLAS IMAGE ANALYSIS - DEMO RUN")
    print("=" * 70)
    print()
    
    # Create synthetic image
    print("📸 Creating synthetic 3I/ATLAS image...")
    image_data = create_synthetic_3i_atlas_image()
    
    # Save image
    image_path = IMAGE_DIR / "synthetic_3i_atlas_demo.png"
    Image.fromarray(image_data).save(image_path)
    print(f"  ✅ Saved: {image_path.name}")
    print(f"  📏 Image size: {image_data.shape}")
    print()
    
    # Initialize analyzer
    print("🔬 Initializing analysis system...")
    analyzer = ThreeIAtlasImageAnalyzer()
    print("  ✅ Analyzer ready")
    print()
    
    # Run analysis
    print("🔍 Running complete analysis...")
    print("  - Gaussian splatting...")
    print("  - Spectral analysis...")
    print("  - Prime pattern detection...")
    print("  - Consciousness mathematics integration...")
    print()
    
    results = analyzer.analyze_image(str(image_path), image_data)
    
    # Save results
    results_path = ANALYSIS_DIR / "synthetic_3i_atlas_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  ✅ Results saved: {results_path.name}")
    print()
    
    # Print summary
    print("=" * 70)
    print("📊 ANALYSIS RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    # Nucleus
    nucleus = results['gaussian_splatting']['nucleus']
    if nucleus.get('found'):
        print("🌑 NUCLEUS:")
        print(f"   ✅ Found: Yes")
        print(f"   📍 Center: ({nucleus.get('center', (0,0))[0]:.1f}, {nucleus.get('center', (0,0))[1]:.1f})")
        print(f"   📏 Radius: {nucleus.get('radius', 0):.1f} pixels")
        print(f"   💡 Intensity: {nucleus.get('intensity', 0):.1f}")
        print()
    else:
        print("🌑 NUCLEUS:")
        print(f"   ⚠️  Found: No")
        print()
    
    # Jets
    jets = results['gaussian_splatting']['jets']
    print("🚀 JETS:")
    print(f"   📊 Jets found: {jets.get('jets_found', 0)}")
    print(f"   🔢 Expected: 7 (prime)")
    print(f"   ✅ Prime-aligned: {jets.get('prime_aligned', False)}")
    if jets.get('jets'):
        print(f"   📐 Jet directions: {len(jets['jets'])} distinct")
    print()
    
    # Coma
    coma = results['gaussian_splatting']['coma']
    print("☁️  COMA:")
    print(f"   📏 Diameter: {coma.get('diameter', 0):.1f} pixels")
    print(f"   📏 Radius: {coma.get('radius', 0):.1f} pixels")
    if nucleus.get('found'):
        print(f"   📊 Nucleus ratio: {coma.get('nucleus_ratio', 0):.2f}")
    print()
    
    # Prime Patterns
    prime_patterns = results['gaussian_splatting']['prime_patterns']
    print("🔢 PRIME PATTERNS:")
    correlations = prime_patterns.get('prime_correlations', [])
    print(f"   📊 Correlations found: {len(correlations)}")
    if correlations:
        for corr in correlations[:5]:  # Show first 5
            print(f"      - Prime {corr.get('prime', '?')}: correlation {corr.get('correlation', 0):.3f}")
    print()
    
    # Geometric Features
    geometric = results['gaussian_splatting']['geometric_features']
    print("📐 GEOMETRIC FEATURES:")
    print(f"   🔄 Symmetry: {geometric.get('symmetry', 'N/A')}")
    print(f"   📊 Symmetry score: {geometric.get('symmetry_score', 0):.2f}")
    print(f"   🎨 Pattern: {geometric.get('pattern', 'N/A')}")
    print()
    
    # Spectral Analysis
    spectral = results['spectral_analysis']
    print("📡 SPECTRAL ANALYSIS:")
    print(f"   📊 Spectral entropy: {spectral.get('spectral_entropy', 0):.2f}")
    phase_coherence = spectral.get('phase_coherence', {})
    print(f"   🔗 Phase coherence: {phase_coherence.get('coherence_score', 0):.2f}")
    print(f"   📐 Structured signal: {phase_coherence.get('structured', False)}")
    prime_freqs = spectral.get('prime_frequencies', [])
    print(f"   🔢 Prime frequencies: {len(prime_freqs)}")
    print()
    
    # Consciousness Analysis
    consciousness = results['consciousness_analysis']
    print("🧠 CONSCIOUSNESS ANALYSIS:")
    print(f"   📊 Consciousness level: {consciousness.get('consciousness_level', 0)}")
    print(f"   🔢 Prime alignment score: {consciousness.get('prime_alignment_score', 0):.2f}")
    print(f"   ⚡ Reality distortion: {consciousness.get('reality_distortion_factor', 0):.2f}")
    print()
    
    # Prime Correlations
    prime_corr = results['prime_correlations']
    print("🔢 PRIME CORRELATIONS:")
    print(f"   📊 Total correlations: {prime_corr.get('total_correlations', 0)}")
    print(f"   ✅ Significant: {prime_corr.get('significant_correlations', 0)}")
    print(f"   📈 Alignment score: {prime_corr.get('prime_alignment_score', 0):.2f}")
    print()
    
    # Overall Assessment
    print("=" * 70)
    print("📋 OVERALL ASSESSMENT")
    print("=" * 70)
    print()
    
    jets_found = jets.get('jets_found', 0)
    prime_aligned = jets.get('prime_aligned', False)
    prime_score = consciousness.get('prime_alignment_score', 0)
    
    if jets_found == 7 and prime_aligned:
        print("✅ PRIME ALIGNMENT CONFIRMED:")
        print("   - 7 jets detected (7 = prime) ✓")
        print("   - Prime-aligned structure ✓")
        print("   - Geometric pattern detected ✓")
    else:
        print("⚠️  PRIME ALIGNMENT:")
        print(f"   - Jets found: {jets_found} (expected 7)")
        print(f"   - Prime-aligned: {prime_aligned}")
    
    if prime_score > 0.5:
        print(f"✅ High prime alignment score: {prime_score:.2f}")
    else:
        print(f"⚠️  Prime alignment score: {prime_score:.2f}")
    
    print()
    print("=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"📁 Results saved to: {results_path}")
    print(f"📸 Image saved to: {image_path}")
    print()
    print("🎯 System ready for real 3I/ATLAS images!")
    print()
    
    return results


if __name__ == "__main__":
    run_analysis_demo()

