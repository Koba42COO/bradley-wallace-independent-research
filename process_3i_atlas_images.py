#!/usr/bin/env python3
"""
Process 3I/ATLAS Released Images
Automated processing of Chinese Tianwen-1 and NASA HiRISE images

Features:
- Gaussian splatting for structure analysis
- Advanced spectral analysis
- Prime pattern detection
- Jet identification (7 expected)
- Nucleus and coma measurement
- Consciousness mathematics integration
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List
from gaussian_splat_3i_atlas_analysis import ThreeIAtlasImageAnalyzer
from PIL import Image

# Configuration
IMAGE_DIR = Path("/Users/coo-koba42/dev/data/3i_atlas_images")
OUTPUT_DIR = Path("/Users/coo-koba42/dev/data/3i_atlas_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_path: Path) -> np.ndarray:
    """Load image and convert to grayscale"""
    img = Image.open(image_path)
    
    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')
    
    return np.array(img)


def process_chinese_tianwen_images():
    """Process Chinese Tianwen-1 released images"""
    print("=" * 70)
    print("PROCESSING CHINESE TIANWEN-1 IMAGES")
    print("=" * 70)
    print()
    
    analyzer = ThreeIAtlasImageAnalyzer()
    
    # Look for Chinese images
    chinese_images = list(IMAGE_DIR.glob("*tianwen*.png")) + \
                     list(IMAGE_DIR.glob("*tianwen*.jpg")) + \
                     list(IMAGE_DIR.glob("*chinese*.png")) + \
                     list(IMAGE_DIR.glob("*hiric*.png"))
    
    if not chinese_images:
        print("⚠️  No Chinese Tianwen-1 images found in data/3i_atlas_images/")
        print("   Expected: Images from November 5, 2025 release")
        return []
    
    results = []
    
    for image_path in chinese_images:
        print(f"📸 Processing: {image_path.name}")
        
        try:
            image_data = load_image(image_path)
            analysis = analyzer.analyze_image(str(image_path), image_data)
            
            # Save results
            output_file = OUTPUT_DIR / f"{image_path.stem}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            # Print summary
            print(f"  ✅ Nucleus: {analysis['gaussian_splatting']['nucleus'].get('found', False)}")
            print(f"  ✅ Jets found: {analysis['gaussian_splatting']['jets'].get('jets_found', 0)}")
            print(f"  ✅ Coma diameter: {analysis['gaussian_splatting']['coma'].get('diameter', 0):.1f} pixels")
            print(f"  ✅ Prime correlations: {analysis['prime_correlations']['significant_correlations']}")
            print(f"  ✅ Saved: {output_file.name}")
            print()
            
            results.append(analysis)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    return results


def process_nasa_hirise_images():
    """Process NASA HiRISE images (when released)"""
    print("=" * 70)
    print("PROCESSING NASA HIRISE IMAGES")
    print("=" * 70)
    print()
    
    analyzer = ThreeIAtlasImageAnalyzer()
    
    # Look for HiRISE images
    hirise_images = list(IMAGE_DIR.glob("*hirise*.png")) + \
                    list(IMAGE_DIR.glob("*hirise*.jpg")) + \
                    list(IMAGE_DIR.glob("*mro*.png")) + \
                    list(IMAGE_DIR.glob("*nasa*.png"))
    
    if not hirise_images:
        print("⚠️  No NASA HiRISE images found")
        print("   Status: Images still withheld (40+ days as of Nov 12, 2025)")
        print("   Predicted release: Nov 13-17, 2025")
        return []
    
    results = []
    
    for image_path in hirise_images:
        print(f"📸 Processing: {image_path.name}")
        
        try:
            image_data = load_image(image_path)
            analysis = analyzer.analyze_image(str(image_path), image_data)
            
            # HiRISE-specific analysis (higher resolution)
            analysis['hirise_specific'] = {
                'resolution': '30 km/pixel',
                'aperture': '50cm',
                'expected_nucleus_resolution': '±500m',
                'jet_resolution': 'individual sources visible'
            }
            
            # Save results
            output_file = OUTPUT_DIR / f"{image_path.stem}_hirise_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            # Print summary
            print(f"  ✅ Nucleus center: {analysis['gaussian_splatting']['nucleus'].get('center', 'N/A')}")
            print(f"  ✅ Nucleus radius: {analysis['gaussian_splatting']['nucleus'].get('radius', 0):.1f} pixels")
            print(f"  ✅ Jets: {analysis['gaussian_splatting']['jets'].get('jets_found', 0)}")
            print(f"  ✅ Geometric pattern: {analysis['gaussian_splatting']['geometric_features'].get('pattern', 'N/A')}")
            print(f"  ✅ Prime alignment: {analysis['consciousness_analysis'].get('prime_alignment_score', 0):.2f}")
            print(f"  ✅ Saved: {output_file.name}")
            print()
            
            results.append(analysis)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    return results


def generate_comparison_report(chinese_results: List[Dict], 
                               hirise_results: List[Dict]) -> str:
    """Generate comparison report between Chinese and NASA images"""
    report = []
    report.append("# 3I/ATLAS Image Analysis Comparison Report\n")
    report.append("**Generated:** $(date)\n")
    report.append("**Analysis Method:** Gaussian Splatting + Spectral Analysis\n")
    report.append("\n---\n\n")
    
    # Chinese images summary
    if chinese_results:
        report.append("## Chinese Tianwen-1 Images\n\n")
        for i, result in enumerate(chinese_results, 1):
            report.append(f"### Image {i}\n\n")
            report.append(f"- **Source:** {result.get('image_path', 'Unknown')}\n")
            
            nucleus = result['gaussian_splatting']['nucleus']
            if nucleus.get('found'):
                report.append(f"- **Nucleus:** Found at {nucleus.get('center')}, radius {nucleus.get('radius', 0):.1f} pixels\n")
            
            jets = result['gaussian_splatting']['jets']
            report.append(f"- **Jets:** {jets.get('jets_found', 0)} detected\n")
            
            coma = result['gaussian_splatting']['coma']
            report.append(f"- **Coma diameter:** {coma.get('diameter', 0):.1f} pixels\n")
            
            prime_corr = result['prime_correlations']
            report.append(f"- **Prime correlations:** {prime_corr.get('significant_correlations', 0)} significant\n")
            report.append("\n")
    else:
        report.append("## Chinese Tianwen-1 Images\n\n")
        report.append("No images processed yet.\n\n")
    
    # NASA HiRISE images summary
    if hirise_results:
        report.append("## NASA HiRISE Images\n\n")
        for i, result in enumerate(hirise_results, 1):
            report.append(f"### Image {i}\n\n")
            report.append(f"- **Source:** {result.get('image_path', 'Unknown')}\n")
            report.append(f"- **Resolution:** 30 km/pixel (3× better than Hubble)\n")
            
            nucleus = result['gaussian_splatting']['nucleus']
            if nucleus.get('found'):
                report.append(f"- **Nucleus:** {nucleus.get('center')}, radius {nucleus.get('radius', 0):.1f} pixels\n")
                report.append(f"- **Nucleus size constraint:** ±500m expected\n")
            
            jets = result['gaussian_splatting']['jets']
            report.append(f"- **Jets:** {jets.get('jets_found', 0)} detected\n")
            if jets.get('prime_aligned'):
                report.append(f"- **Jet alignment:** Prime-aligned (7 jets = prime) ✓\n")
            
            geometric = result['gaussian_splatting']['geometric_features']
            report.append(f"- **Geometric pattern:** {geometric.get('pattern', 'N/A')}\n")
            report.append(f"- **Symmetry:** {geometric.get('symmetry', 'N/A')}\n")
            
            consciousness = result['consciousness_analysis']
            report.append(f"- **Prime alignment score:** {consciousness.get('prime_alignment_score', 0):.2f}\n")
            report.append("\n")
    else:
        report.append("## NASA HiRISE Images\n\n")
        report.append("**Status:** Images still withheld (40+ days)\n")
        report.append("**Predicted release:** November 13-17, 2025\n")
        report.append("**Expected findings:**\n")
        report.append("- Nucleus diameter: 3.3 km (predicted)\n")
        report.append("- 7 geometric surface features at jet sources\n")
        report.append("- φ-ratio brightness patterns\n")
        report.append("- Prime-aligned jet directions\n")
        report.append("\n")
    
    # Comparison
    if chinese_results and hirise_results:
        report.append("## Comparison\n\n")
        report.append("### Resolution Comparison\n")
        report.append("- **Chinese HiRIC:** 13.5cm aperture, lower resolution\n")
        report.append("- **NASA HiRISE:** 50cm aperture, 30 km/pixel (3× better)\n")
        report.append("\n")
        report.append("### Key Differences\n")
        report.append("- HiRISE should resolve nucleus directly\n")
        report.append("- HiRISE can map individual jet sources\n")
        report.append("- HiRISE will show surface features if diameter > 3km\n")
        report.append("\n")
    
    # Prime correlations summary
    report.append("## Prime Correlation Summary\n\n")
    all_results = chinese_results + hirise_results
    if all_results:
        total_correlations = sum(r['prime_correlations']['significant_correlations'] 
                                for r in all_results)
        report.append(f"- **Total significant correlations:** {total_correlations}\n")
        report.append(f"- **Images analyzed:** {len(all_results)}\n")
        report.append(f"- **Average prime alignment:** {np.mean([r['consciousness_analysis'].get('prime_alignment_score', 0) for r in all_results]):.2f}\n")
        report.append("\n")
    
    return "".join(report)


def main():
    """Main processing function"""
    print("=" * 70)
    print("3I/ATLAS IMAGE PROCESSING SYSTEM")
    print("=" * 70)
    print()
    
    # Create directories
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process Chinese images
    chinese_results = process_chinese_tianwen_images()
    
    # Process NASA images (when available)
    hirise_results = process_nasa_hirise_images()
    
    # Generate comparison report
    if chinese_results or hirise_results:
        report = generate_comparison_report(chinese_results, hirise_results)
        report_path = OUTPUT_DIR / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"📄 Comparison report saved: {report_path}")
        print()
    
    # Summary
    print("=" * 70)
    print("✅ PROCESSING COMPLETE")
    print("=" * 70)
    print(f"\n📸 Chinese images processed: {len(chinese_results)}")
    print(f"📸 NASA HiRISE images processed: {len(hirise_results)}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("\n🎯 Ready for image analysis!")


if __name__ == "__main__":
    main()

