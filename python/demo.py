#!/usr/bin/env python3
"""
Demo script for the Python fit-curve implementation.
Shows basic usage and visualization capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from fit_curve import fit_curve
from visualize import visualize_fit_curve, compare_errors


def basic_example():
    """Basic usage example."""
    print("Basic Fit-Curve Example")
    print("=" * 30)
    
    # Simple curve
    points = [[0, 0], [10, 10], [10, 0], [20, 0]]
    error_tolerance = 50
    
    print(f"Original points: {points}")
    print(f"Error tolerance: {error_tolerance}")
    
    # Fit the curve
    bezier_curves = fit_curve(points, error_tolerance)
    
    print(f"Result: {len(bezier_curves)} Bezier curve(s)")
    for i, curve in enumerate(bezier_curves):
        print(f"  Curve {i+1}: {curve}")
    
    # Visualize
    fig = visualize_fit_curve(points, error_tolerance, 
                             title="Basic Example", show_control_points=True)
    plt.show()
    
    return bezier_curves


def complex_example():
    """More complex curve fitting example."""
    print("\nComplex Curve Example")
    print("=" * 30)
    
    # Create a more complex path (spiral)
    t = np.linspace(0, 4 * np.pi, 50)
    radius_decay = np.exp(-t / 10)
    points = []
    
    for i, t_val in enumerate(t):
        x = radius_decay[i] * 20 * np.cos(t_val) + 50
        y = radius_decay[i] * 20 * np.sin(t_val) + 50
        points.append([x, y])
    
    print(f"Generated {len(points)} points along a spiral")
    
    # Try different error tolerances
    error_values = [1, 5, 20, 50]
    
    for error in error_values:
        curves = fit_curve(points, error)
        print(f"Error {error:2d}: {len(curves)} curve(s)")
    
    # Show comparison
    fig = compare_errors(points, error_values)
    fig.suptitle("Spiral Curve - Different Error Tolerances", fontsize=16)
    plt.show()


def real_world_example():
    """Real-world-like example with noisy data."""
    print("\nReal-World Example (with noise)")
    print("=" * 35)
    
    # Create a smooth curve and add some noise
    t = np.linspace(0, 2 * np.pi, 30)
    clean_points = [[20 * np.cos(t_val) + 50, 15 * np.sin(t_val) + 30] for t_val in t]
    
    # Add noise
    np.random.seed(42)  # For reproducible results
    noisy_points = []
    for point in clean_points:
        noise = np.random.normal(0, 1, 2)  # Small amount of noise
        noisy_points.append([point[0] + noise[0], point[1] + noise[1]])
    
    print(f"Original clean points: {len(clean_points)}")
    print(f"Added Gaussian noise (σ=1)")
    
    # Fit curves to both clean and noisy data
    clean_curves = fit_curve(clean_points, 10)
    noisy_curves = fit_curve(noisy_points, 10)
    
    print(f"Clean data: {len(clean_curves)} curve(s)")
    print(f"Noisy data: {len(noisy_curves)} curve(s)")
    
    # Visualize both
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    plt.sca(ax1)
    visualize_fit_curve(clean_points, 10, title="Clean Data")
    
    plt.sca(ax2)
    visualize_fit_curve(noisy_points, 10, title="Noisy Data")
    
    plt.tight_layout()
    plt.show()


def interactive_drawing_simulation():
    """Simulate drawing points interactively."""
    print("\nInteractive Drawing Simulation")
    print("=" * 35)
    
    # Simulate mouse drawing by creating points along a path
    def simulate_drawing(func, t_range, num_points):
        t_values = np.linspace(t_range[0], t_range[1], num_points)
        return [func(t) for t in t_values]
    
    # Different "drawn" shapes
    shapes = {
        "Heart": lambda t: [16 * np.sin(t)**3, 
                           13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)],
        "Flower": lambda t: [10 * (2 + np.cos(5*t)) * np.cos(t), 
                            10 * (2 + np.cos(5*t)) * np.sin(t)],
        "Star": lambda t: [(1 + 0.5 * np.cos(5*t)) * 20 * np.cos(t), 
                          (1 + 0.5 * np.cos(5*t)) * 20 * np.sin(t)]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (name, func) in enumerate(shapes.items()):
        points = simulate_drawing(func, (0, 2*np.pi), 60)
        curves = fit_curve(points, 15)
        
        print(f"{name}: {len(points)} points → {len(curves)} curves")
        
        plt.sca(axes[i])
        visualize_fit_curve(points, 15, title=f"{name} ({len(curves)} curves)",
                           show_control_points=False)
    
    plt.tight_layout()
    plt.show()


def main():
    """Run all demo examples."""
    print("Python Fit-Curve Demo")
    print("=" * 50)
    
    try:
        # Run examples
        basic_example()
        complex_example()
        real_world_example()
        interactive_drawing_simulation()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("Try running 'python visualize.py' for an interactive demo.")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()