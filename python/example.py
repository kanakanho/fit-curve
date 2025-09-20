#!/usr/bin/env python3
"""
Simple example demonstrating the Python fit-curve implementation with visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from fit_curve import fit_curve
from visualize import visualize_fit_curve


def main():
    """Run a simple example."""
    print("Python Fit-Curve Example")
    print("=" * 25)
    
    # Example 1: Simple curve
    points1 = [[0, 0], [10, 10], [10, 0], [20, 0]]
    curves1 = fit_curve(points1, 50)
    
    print(f"Example 1: {len(points1)} points -> {len(curves1)} curve(s)")
    print(f"Result: {curves1[0]}")
    
    # Example 2: More complex curve
    t = np.linspace(0, 2*np.pi, 20)
    points2 = [[20*np.cos(t_val), 15*np.sin(t_val)] for t_val in t]
    curves2 = fit_curve(points2, 10)
    
    print(f"\nExample 2: {len(points2)} points -> {len(curves2)} curve(s)")
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot example 1
    plt.sca(ax1)
    visualize_fit_curve(points1, 50, title="Simple Curve")
    
    # Plot example 2  
    plt.sca(ax2)
    visualize_fit_curve(points2, 10, title="Ellipse Approximation")
    
    plt.tight_layout()
    
    # Save the plot
    output_file = '/tmp/fit_curve_example.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Try to display if running interactively
    try:
        plt.show()
    except:
        print("(Display not available - plot saved to file)")


if __name__ == "__main__":
    main()