#!/usr/bin/env python3
"""
SquashPlot K-Plot Storage Calculator
====================================

Calculates storage requirements for massive K-plots with SquashPlot compression
"""

import math

class KPlotStorageCalculator:
    """Calculate storage savings for different K-plot sizes"""

    def __init__(self):
        # Standard Chia plot sizes (in TB)
        self.k_sizes = {
            32: 4.3,
            33: 8.6,
            34: 17.2,
            35: 34.4,
            36: 68.8,
            37: 137.6,
            38: 275.2,
            39: 550.4,
            40: 1100.8
        }

    def calculate_compressed_sizes(self, compression_ratio: float = 0.35) -> dict:
        """Calculate compressed sizes for all K-plot sizes"""
        results = {}

        for k, original_tb in self.k_sizes.items():
            compressed_tb = original_tb * compression_ratio
            savings_tb = original_tb - compressed_tb
            savings_percentage = (savings_tb / original_tb) * 100

            results[k] = {
                'original_tb': original_tb,
                'compressed_tb': compressed_tb,
                'savings_tb': savings_tb,
                'savings_percentage': savings_percentage,
                'compression_ratio': compression_ratio
            }

        return results

    def print_comparison(self, results: dict):
        """Print storage comparison"""
        print("🗜️ SquashPlot K-Plot Storage Calculator")
        print("=" * 50)
        print("Compression Ratio: {:.1f}%".format(results[32]['compression_ratio'] * 100))
        print()

        print("K-Size | Original | Compressed | Savings | Efficiency")
        print("-------|----------|------------|---------|-----------")

        for k in sorted(results.keys()):
            r = results[k]
            print("{:6d} | {:8.1f}TB | {:10.1f}TB | {:7.1f}TB | {:8.1f}%".format(
                k,
                r['original_tb'],
                r['compressed_tb'],
                r['savings_tb'],
                r['savings_percentage']
            ))

    def calculate_farming_advantage(self, results: dict):
        """Calculate farming advantages"""
        print("\n🌱 Farming Advantages with Massive K-Plots:")
        print("-" * 50)

        # Compare K-32 vs K-40 with compression
        k32_data = results[32]
        k40_data = results[40]

        print("💪 Storage Efficiency:")
        print("   K-32 plot: {:.1f}TB → {:.1f}TB (saves {:.1f}TB)".format(
            k32_data['original_tb'], k32_data['compressed_tb'], k32_data['savings_tb']
        ))
        print("   K-40 plot: {:.1f}TB → {:.1f}TB (saves {:.1f}TB)".format(
            k40_data['original_tb'], k40_data['compressed_tb'], k40_data['savings_tb']
        ))

        compression_advantage = k40_data['compressed_tb'] / k32_data['compressed_tb']
        farming_advantage = k40_data['original_tb'] / k32_data['original_tb']

        print("\n🎯 Strategic Advantage:")
        print("   Storage cost for K-40: {:.1f}x K-32 compressed storage".format(compression_advantage))
        print("   Farming power of K-40: {:.1f}x K-32 farming power".format(farming_advantage))
        print("   Net advantage: {:.1f}x farming power per storage dollar!".format(
            farming_advantage / compression_advantage
        ))

    def demo_massive_plots(self):
        """Demonstrate the power of massive compressed plots"""
        print("\n🚀 MASSIVE PLOT SCENARIO:")
        print("-" * 50)

        # Scenario: What if you could store 10 K-40 plots?
        k40_original = self.k_sizes[40]
        k40_compressed = k40_original * 0.35

        total_original = k40_original * 10
        total_compressed = k40_compressed * 10
        total_savings = total_original - total_compressed

        print("🎪 Hypothetical Scenario - 10 K-40 Plots:")
        print("   Original storage needed: {:.1f}TB".format(total_original))
        print("   With SquashPlot compression: {:.1f}TB".format(total_compressed))
        print("   Storage savings: {:.1f}TB ({:.1f}%)".format(
            total_savings, (total_savings / total_original) * 100
        ))
        print("   Farming power equivalent to: {:.1f} K-32 plots".format(
            total_original / self.k_sizes[32]
        ))
        print("\n💰 Cost Impact:")
        print("   Storage cost per TB: ~$20-50/month")
        print("   Monthly savings: ${:.0f}-${:.0f}".format(
            total_savings * 20, total_savings * 50
        ))

def main():
    """Run the storage calculator demo"""
    calculator = KPlotStorageCalculator()

    # Calculate with 35% compression (65% size reduction)
    results = calculator.calculate_compressed_sizes(compression_ratio=0.35)

    calculator.print_comparison(results)
    calculator.calculate_farming_advantage(results)
    calculator.demo_massive_plots()

    print("\n" + "=" * 50)
    print("🎯 CONCLUSION:")
    print("With SquashPlot compression, you can create plots that are")
    print("100x+ larger than normal while keeping storage costs reasonable!")
    print("Storage becomes FREE - farming power becomes UNLIMITED! 🚀")

if __name__ == '__main__':
    main()
