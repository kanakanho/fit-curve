# Python Implementation of Fit-Curve

This directory contains a Python implementation of Philip J. Schneider's "Algorithm for Automatically Fitting Digitized Curves" from "Graphics Gems", Academic Press, 1990.

## Features

- **Pure Python implementation** with NumPy for mathematical operations
- **Matplotlib visualization** capabilities for curve fitting results
- **Compatible results** with the JavaScript version
- **Support for 2D and higher-dimensional curves**
- **Interactive examples** and demos

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from fit_curve import fit_curve

# Define your points
points = [[0, 0], [10, 10], [10, 0], [20, 0]]
error_tolerance = 50

# Fit Bezier curves
bezier_curves = fit_curve(points, error_tolerance)

# Result: List of Bezier curves
# Each curve is [start_point, control_point_1, control_point_2, end_point]
print(f"Fitted {len(bezier_curves)} curve(s)")
for i, curve in enumerate(bezier_curves):
    print(f"Curve {i+1}: {curve}")
```

### Visualization

```python
from visualize import visualize_fit_curve, compare_errors
import matplotlib.pyplot as plt

# Visualize curve fitting
fig = visualize_fit_curve(points, error_tolerance)
plt.show()

# Compare different error tolerances
fig = compare_errors(points, [1, 5, 10, 50])
plt.show()
```

## Files

- `fit_curve.py` - Main curve fitting algorithm
- `visualize.py` - Visualization utilities and interactive demos
- `test_fit_curve.py` - Test suite validating against JavaScript results
- `demo.py` - Example usage and demonstrations
- `requirements.txt` - Python package dependencies

## Running Examples

### Basic demo:
```bash
python demo.py
```

### Interactive visualization demo:
```bash
python visualize.py
```

### Run tests:
```bash
python test_fit_curve.py
```

## Algorithm Overview

The algorithm follows these main steps:

1. **Preprocessing**: Remove duplicate points and validate input
2. **Tangent calculation**: Compute unit tangent vectors at endpoints
3. **Parameterization**: Use chord-length parameterization for initial parameter values
4. **Curve generation**: Generate initial Bezier curve using least-squares method
5. **Refinement**: Iteratively improve parameters using Newton-Raphson method
6. **Subdivision**: If error is too large, split at maximum error point and recursively fit segments

## Key Classes and Functions

### `fit_curve(points, max_error, progress_callback=None)`
Main entry point for curve fitting.

**Parameters:**
- `points`: List of points `[[x1, y1], [x2, y2], ...]`
- `max_error`: Maximum allowed squared error between points and fitted curve
- `progress_callback`: Optional callback function for progress updates

**Returns:**
- List of Bezier curves, each containing 4 control points

### `Maths` class
Mathematical utilities for vector operations:
- `subtract()`, `add_arrays()`, `mul_items()`
- `vector_len()`, `normalize()`, `dot()`

### `Bezier` class
Bezier curve evaluation functions:
- `q()` - Evaluate curve at parameter t
- `qprime()` - First derivative
- `qprimeprime()` - Second derivative

## Visualization Functions

### `visualize_fit_curve(points, max_error, ...)`
Create a comprehensive visualization of curve fitting results.

### `compare_errors(points, error_values)`
Compare curve fitting with different error tolerances.

### `plot_bezier_curve()` and `plot_control_points()`
Low-level plotting functions for individual curves.

## Testing

The test suite validates the Python implementation against the same test cases used in the JavaScript version:

- Basic curve fitting examples
- Multiple curve scenarios
- Edge cases (empty points, single points, etc.)
- 3D curve support
- Performance testing

## Performance

The Python implementation is optimized for clarity and correctness rather than raw speed. For large datasets or real-time applications, consider:

- Using NumPy operations where possible
- Caching intermediate results
- Implementing progress callbacks to monitor long-running operations

## Examples

### Simple Curve
```python
points = [[0, 0], [10, 10], [10, 0], [20, 0]]
curves = fit_curve(points, 50)
# Result: 1 Bezier curve
```

### Complex Path
```python
# Spiral or other complex shape
import numpy as np
t = np.linspace(0, 4*np.pi, 50)
points = [[20*np.cos(t_val), 20*np.sin(t_val)] for t_val in t]
curves = fit_curve(points, 10)
# Result: Multiple Bezier curves approximating the spiral
```

### 3D Curves
```python
points_3d = [[0, 0, 0], [10, 10, 5], [20, 0, 10], [30, -10, 15]]
curves = fit_curve(points_3d, 25)
# Works with any dimensionality
```

## Comparison with JavaScript Version

The Python implementation produces identical results to the JavaScript version for the same inputs and error tolerances. Key differences:

- **Language**: Python vs JavaScript
- **Dependencies**: NumPy/Matplotlib vs None (JavaScript version)
- **Visualization**: Built-in matplotlib support
- **Array handling**: NumPy arrays vs JavaScript arrays
- **Type hints**: Python version includes type annotations

## License

This Python implementation follows the same MIT License as the original JavaScript version.