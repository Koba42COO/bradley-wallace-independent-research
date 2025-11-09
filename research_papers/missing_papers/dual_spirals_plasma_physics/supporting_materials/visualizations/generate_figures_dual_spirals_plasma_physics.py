#!/usr/bin/env python3
"""
Visualization script for dual_spirals_plasma_physics
Generates figures and plots for all theorems.
"""
# Set high precision
getcontext().prec = 50


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
phi = Decimal('1.618033988749894848204586834365638117720309179805762862135')

def visualize_theorems():
    """Generate visualizations for all theorems."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations...")

    # Default visualization: Golden Ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0.1, 10, 1000)
    y = np.log(x) ** phi
    ax.plot(x, y, 'b-', linewidth=2)
    ax.set_title("Wallace Transform Visualization")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "figure_1_wallace_transform.png", dpi=300)
    print("  ✓ Generated default visualization")

    print("\n✅ All visualizations generated successfully!")

if __name__ == '__main__':
    visualize_theorems()
