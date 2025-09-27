#!/usr/bin/env python3
"""
SquashPlot Plotting Demo - Similar to Mad Max/BladeBit Structure
=================================================================

This demo shows how SquashPlot uses a command structure similar to
established plotters like Mad Max and BladeBit.

Usage Examples:
===============

1. Basic Plotting (Mad Max style):
   python demo_plotting.py -t /tmp/plot1 -d /plots -f farmer_key -p pool_key

2. Advanced Plotting (with compression like BladeBit):
   python demo_plotting.py -t /tmp/plot1 -2 /tmp/plot2 -d /plots -f farmer_key -p pool_key --compress 3 -n 2

3. Pool Farming:
   python demo_plotting.py -t /tmp/plot1 -d /plots -f farmer_key -p pool_key -c contract_address

4. Compression Only:
   python demo_plotting.py --mode compress --input plot.dat --output plot.squash --compress 4
"""

import sys
import os

def demo_mad_max_style():
    """Demo Mad Max style plotting"""
    print("🔧 Mad Max Style Plotting Demo")
    print("=" * 50)
    print("Command: ./chia_plot -t /tmp/plot1 -2 /tmp/plot2 -d /plots -p <pool_key> -f <farmer_key> -r 4 -u 256 -n 1")
    print()
    print("SquashPlot equivalent:")
    print("python squashplot.py -t /tmp/plot1 -2 /tmp/plot2 -d /plots -p <pool_key> -f <farmer_key> -r 4 -u 256 -n 1")
    print()
    print("Key similarities:")
    print("• -t: Primary temp directory")
    print("• -2: Secondary temp directory")
    print("• -d: Final destination")
    print("• -f: Farmer key")
    print("• -p: Pool key")
    print("• -r: Thread count")
    print("• -u: Bucket count")
    print("• -n: Number of plots")

def demo_bladebit_style():
    """Demo BladeBit style plotting"""
    print("\n🔧 BladeBit Style Plotting Demo")
    print("=" * 50)
    print("Command: chia plotters bladebit ramplot -d /plots -f <farmer_key> -p <pool_key> -c <contract> -n 1 --compress 3")
    print()
    print("SquashPlot equivalent:")
    print("python squashplot.py -t /tmp/plot1 -d /plots -f <farmer_key> -p <pool_key> -c <contract> -n 1 --compress 3")
    print()
    print("Key similarities:")
    print("• -d: Destination directory")
    print("• -f: Farmer key")
    print("• -p: Pool key")
    print("• -c: Contract address")
    print("• -n: Number of plots")
    print("• --compress: Compression level")
    print()
    print("Compression Levels (BladeBit style):")
    print("• 0: No compression (109GB)")
    print("• 1: Light compression (88GB)")
    print("• 2: Medium compression (86GB)")
    print("• 3: Good compression (84GB)")
    print("• 4: Better compression (82GB)")
    print("• 5: Strong compression (80GB)")
    print("• 6: Very strong (78GB)")
    print("• 7: Maximum compression (76GB)")

def demo_pool_farming():
    """Demo pool farming setup"""
    print("\n🔧 Pool Farming Demo")
    print("=" * 50)
    print("For pool farming, you need:")
    print("1. Pool contract address from 'chia plotnft show'")
    print("2. Use -c parameter for contract address")
    print()
    print("Example:")
    print("python squashplot.py -t /tmp/plot1 -d /plots -f <farmer_key> -p <pool_key> -c <pool_contract>")
    print()
    print("Get pool contract:")
    print("chia plotnft show")

def demo_compression_workflow():
    """Demo compression workflow"""
    print("\n🔧 Compression Workflow Demo")
    print("=" * 50)
    print("Two ways to compress plots:")
    print()
    print("1. Compress existing plots:")
    print("   python squashplot.py --mode compress --input plot.dat --output plot.squash --compress 4")
    print()
    print("2. Create compressed plots directly:")
    print("   python squashplot.py -t /tmp/plot1 -d /plots -f <farmer_key> -p <pool_key> --compress 3")
    print()
    print("Benefits:")
    print("• 42% compression (Basic)")
    print("• Up to 70% compression (Pro)")
    print("• 100% farming compatibility")
    print("• Faster plotting times")

def main():
    """Main demo function"""
    print("🌟 SquashPlot - Command Structure Demo")
    print("=======================================")
    print()
    print("SquashPlot uses a command structure similar to established plotters")
    print("like Mad Max and BladeBit, making it familiar to Chia farmers.")
    print()

    demo_mad_max_style()
    demo_bladebit_style()
    demo_pool_farming()
    demo_compression_workflow()

    print("\n" + "=" * 60)
    print("🎯 Ready to try SquashPlot?")
    print()
    print("1. Get your Chia keys: chia keys show")
    print("2. Try basic plotting:")
    print("   python squashplot.py -t /tmp/plot1 -d /plots -f <farmer_key> -p <pool_key>")
    print()
    print("3. Try with compression:")
    print("   python squashplot.py -t /tmp/plot1 -d /plots -f <farmer_key> -p <pool_key> --compress 3")
    print()
    print("4. Web interface:")
    print("   python main.py --web")
    print("=" * 60)

if __name__ == "__main__":
    main()
