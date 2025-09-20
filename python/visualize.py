"""
Visualization utilities for fit-curve using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from fit_curve import fit_curve, Bezier


def plot_bezier_curve(bezier: List[List[float]], num_points: int = 100, 
                     color: str = 'blue', linewidth: float = 2.0, 
                     label: Optional[str] = None) -> None:
    """Plot a single Bezier curve."""
    ctrl_points = [np.array(p) for p in bezier]
    
    # Generate points along the curve
    t_values = np.linspace(0, 1, num_points)
    curve_points = []
    
    for t in t_values:
        point = Bezier.q(ctrl_points, t)
        curve_points.append(point)
    
    curve_points = np.array(curve_points)
    plt.plot(curve_points[:, 0], curve_points[:, 1], 
             color=color, linewidth=linewidth, label=label)


def plot_control_points(bezier: List[List[float]], color: str = 'red', 
                       size: float = 50, alpha: float = 0.7) -> None:
    """Plot control points of a Bezier curve."""
    points = np.array(bezier)
    plt.scatter(points[:, 0], points[:, 1], c=color, s=size, alpha=alpha, zorder=5)
    
    # Draw control polygon
    plt.plot(points[:, 0], points[:, 1], '--', color=color, alpha=0.5, linewidth=1)


def visualize_fit_curve(original_points: List[List[float]], max_error: float, 
                       title: Optional[str] = None, figsize: Tuple[int, int] = (10, 8),
                       show_control_points: bool = True, 
                       show_original_points: bool = True) -> plt.Figure:
    """
    Visualize the curve fitting process.
    
    Args:
        original_points: Original digitized points
        max_error: Error tolerance for fitting
        title: Plot title
        figsize: Figure size
        show_control_points: Whether to show Bezier control points
        show_original_points: Whether to show original points
        
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Fit curves
    bezier_curves = fit_curve(original_points, max_error)
    
    # Plot original points
    if show_original_points:
        orig_array = np.array(original_points)
        plt.scatter(orig_array[:, 0], orig_array[:, 1], 
                   c='black', s=30, alpha=0.8, zorder=10, label='Original Points')
        plt.plot(orig_array[:, 0], orig_array[:, 1], 
                'k--', alpha=0.3, linewidth=1, label='Original Path')
    
    # Plot fitted curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(bezier_curves)))
    
    for i, (bezier, color) in enumerate(zip(bezier_curves, colors)):
        label = f'Fitted Curve {i+1}' if len(bezier_curves) > 1 else 'Fitted Curve'
        plot_bezier_curve(bezier, color=color, label=label)
        
        if show_control_points:
            plot_control_points(bezier, color=color)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if title:
        plt.title(title)
    else:
        plt.title(f'Curve Fitting (Error tolerance: {max_error}, {len(bezier_curves)} curves)')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    return fig


def compare_errors(original_points: List[List[float]], 
                  error_values: List[float] = [1, 5, 10, 50, 100]) -> plt.Figure:
    """
    Compare curve fitting with different error tolerances.
    
    Args:
        original_points: Original digitized points
        error_values: List of error tolerances to compare
        
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    orig_array = np.array(original_points)
    
    for i, error in enumerate(error_values):
        if i >= len(axes):
            break
            
        ax = axes[i]
        plt.sca(ax)
        
        # Fit curves
        bezier_curves = fit_curve(original_points, error)
        
        # Plot original points
        plt.scatter(orig_array[:, 0], orig_array[:, 1], 
                   c='black', s=20, alpha=0.6, zorder=10)
        plt.plot(orig_array[:, 0], orig_array[:, 1], 
                'k--', alpha=0.3, linewidth=1)
        
        # Plot fitted curves
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(bezier_curves))))
        
        for j, (bezier, color) in enumerate(zip(bezier_curves, colors)):
            plot_bezier_curve(bezier, color=color, num_points=50)
        
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.title(f'Error: {error} ({len(bezier_curves)} curves)')
    
    # Hide unused subplots
    for i in range(len(error_values), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def interactive_demo():
    """
    Interactive demo with predefined examples.
    """
    examples = {
        "Simple Curve": [[0, 0], [10, 10], [10, 0], [20, 0]],
        "Complex Path": [
            [244, 92], [247, 93], [251, 95], [254, 96], [258, 97], [261, 97], [265, 97], 
            [267, 97], [270, 97], [273, 97], [281, 97], [284, 95], [286, 94], [289, 92], 
            [291, 90], [292, 88], [294, 86], [295, 85], [296, 85], [297, 85]
        ],
        "S-Curve": [[0, 0], [10, 0], [20, 10], [30, 10], [40, 0], [50, 0]],
        "Circle Approximation": [
            [100, 0], [93, 25], [75, 43], [50, 50], [25, 43], [7, 25], 
            [0, 0], [7, -25], [25, -43], [50, -50], [75, -43], [93, -25], [100, 0]
        ]
    }
    
    print("Fit Curve Interactive Demo")
    print("=" * 30)
    
    for name, points in examples.items():
        print(f"\n{name}:")
        print(f"Points: {len(points)}")
        
        # Show different error tolerances
        fig = compare_errors(points, [1, 10, 50])
        fig.suptitle(f'{name} - Error Comparison', fontsize=16)
        plt.show()
        
        # Detailed view with best fit
        fig = visualize_fit_curve(points, 10, title=f'{name} - Detailed View')
        plt.show()


if __name__ == "__main__":
    # Run the interactive demo
    interactive_demo()