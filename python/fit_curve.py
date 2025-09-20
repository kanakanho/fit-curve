"""
Python implementation of Philip J. Schneider's "Algorithm for Automatically Fitting Digitized Curves"
from the book "Graphics Gems", Academic Press, 1990

Converted from JavaScript implementation.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable


class Maths:
    """Mathematical utilities for vector operations."""
    
    @staticmethod
    def subtract(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Subtract two arrays."""
        return arr1 - arr2
    
    @staticmethod
    def add_arrays(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
        """Add two arrays."""
        return arr1 + arr2
    
    @staticmethod
    def mul_items(items: np.ndarray, multiplier: float) -> np.ndarray:
        """Multiply array elements by a scalar."""
        return items * multiplier
    
    @staticmethod
    def vector_len(v: np.ndarray) -> float:
        """Calculate the length (norm) of a vector."""
        return np.linalg.norm(v)
    
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        length = Maths.vector_len(v)
        if length == 0:
            return v
        return v / length
    
    @staticmethod
    def dot(m1: np.ndarray, m2: np.ndarray) -> float:
        """Calculate dot product of two vectors."""
        return np.dot(m1, m2)


class Bezier:
    """Bezier curve utilities."""
    
    @staticmethod
    def q(ctrl_poly: List[np.ndarray], t: float) -> np.ndarray:
        """Evaluate a Bezier curve at parameter t."""
        tx = 1.0 - t
        return (ctrl_poly[0] * (tx * tx * tx) +
                ctrl_poly[1] * (3 * tx * tx * t) +
                ctrl_poly[2] * (3 * tx * t * t) +
                ctrl_poly[3] * (t * t * t))
    
    @staticmethod
    def qprime(ctrl_poly: List[np.ndarray], t: float) -> np.ndarray:
        """Evaluate the first derivative of a Bezier curve at parameter t."""
        tx = 1.0 - t
        return (Maths.subtract(ctrl_poly[1], ctrl_poly[0]) * (3 * tx * tx) +
                Maths.subtract(ctrl_poly[2], ctrl_poly[1]) * (6 * tx * t) +
                Maths.subtract(ctrl_poly[3], ctrl_poly[2]) * (3 * t * t))
    
    @staticmethod
    def qprimeprime(ctrl_poly: List[np.ndarray], t: float) -> np.ndarray:
        """Evaluate the second derivative of a Bezier curve at parameter t."""
        return ((Maths.subtract(Maths.subtract(ctrl_poly[2], Maths.mul_items(ctrl_poly[1], 2)), ctrl_poly[0]) * (6 * (1.0 - t))) +
                (Maths.subtract(Maths.subtract(ctrl_poly[3], Maths.mul_items(ctrl_poly[2], 2)), ctrl_poly[1]) * (6 * t)))


def create_tangent(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """Create a unit tangent vector from point B to point A."""
    return Maths.normalize(Maths.subtract(point_a, point_b))


def chord_length_parameterize(points: List[np.ndarray]) -> List[float]:
    """Assign parameter values to digitized points using relative distances between points."""
    u = []
    prev_u = 0.0
    prev_p = None
    
    for i, p in enumerate(points):
        if i == 0:
            curr_u = 0.0
        else:
            curr_u = prev_u + Maths.vector_len(Maths.subtract(p, prev_p))
        u.append(curr_u)
        prev_u = curr_u
        prev_p = p
    
    # Normalize to [0, 1]
    total_length = u[-1]
    if total_length > 0:
        u = [x / total_length for x in u]
    
    return u


def generate_bezier(points: List[np.ndarray], parameters: List[float], 
                   left_tangent: np.ndarray, right_tangent: np.ndarray) -> List[np.ndarray]:
    """Generate Bezier curve using least-squares method."""
    first_point = points[0]
    last_point = points[-1]
    n = len(parameters)
    
    # Create the coefficient matrix A
    A = []
    for i, u in enumerate(parameters):
        ux = 1 - u
        a_left = Maths.mul_items(left_tangent, 3 * u * (ux * ux))
        a_right = Maths.mul_items(right_tangent, 3 * ux * (u * u))
        A.append([a_left, a_right])
    
    # Create matrices C and X
    C = np.zeros((2, 2))
    X = np.zeros(2)
    
    for i in range(n):
        u = parameters[i]
        a = A[i]
        
        # Compute C matrix entries
        C[0][0] += Maths.dot(a[0], a[0])
        C[0][1] += Maths.dot(a[0], a[1])
        C[1][0] += Maths.dot(a[0], a[1])  # Same as C[0][1]
        C[1][1] += Maths.dot(a[1], a[1])
        
        # Compute right hand side vector
        # This is the key difference: use Bezier evaluation with control points at endpoints
        bez_point = Bezier.q([first_point, first_point, last_point, last_point], u)
        tmp = Maths.subtract(points[i], bez_point)
        
        X[0] += Maths.dot(a[0], tmp)
        X[1] += Maths.dot(a[1], tmp)
    
    # Compute determinants
    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]
    
    # Compute alpha values
    alpha_l = det_X_C1 / det_C0_C1 if det_C0_C1 != 0 else 0
    alpha_r = det_C0_X / det_C0_C1 if det_C0_C1 != 0 else 0
    
    # Use heuristic if alpha values are problematic
    seg_length = Maths.vector_len(Maths.subtract(first_point, last_point))
    epsilon = 1.0e-6 * seg_length
    
    if alpha_l < epsilon or alpha_r < epsilon:
        # Fall back on standard formula
        bezier_curve = [
            first_point,
            Maths.add_arrays(first_point, Maths.mul_items(left_tangent, seg_length / 3.0)),
            Maths.add_arrays(last_point, Maths.mul_items(right_tangent, seg_length / 3.0)),
            last_point
        ]
    else:
        # Use computed alpha values
        bezier_curve = [
            first_point,
            Maths.add_arrays(first_point, Maths.mul_items(left_tangent, alpha_l)),
            Maths.add_arrays(last_point, Maths.mul_items(right_tangent, alpha_r)),
            last_point
        ]
    
    return bezier_curve


def compute_max_error(points: List[np.ndarray], bezier: List[np.ndarray], 
                     parameters: List[float]) -> Tuple[float, int]:
    """Find the maximum squared distance of digitized points to fitted curve."""
    max_dist = 0.0
    split_point = len(points) // 2
    
    # Create distance mapping for better t parameter finding
    t_dist_map = map_t_to_relative_distances(bezier, 10)
    
    for i, point in enumerate(points):
        # Find optimal t parameter
        t = find_t(bezier, parameters[i], t_dist_map, 10)
        
        # Calculate distance
        curve_point = Bezier.q(bezier, t)
        v = Maths.subtract(curve_point, point)
        dist = v[0] * v[0] + v[1] * v[1]  # Squared distance
        
        if dist > max_dist:
            max_dist = dist
            split_point = i
    
    return max_dist, split_point


def map_t_to_relative_distances(bezier: List[np.ndarray], num_parts: int) -> List[float]:
    """Sample t values and map them to relative distances along the curve."""
    distances = [0.0]
    prev_point = bezier[0]
    total_length = 0.0
    
    for i in range(1, num_parts + 1):
        t = i / num_parts
        curr_point = Bezier.q(bezier, t)
        length = Maths.vector_len(Maths.subtract(curr_point, prev_point))
        total_length += length
        distances.append(total_length)
        prev_point = curr_point
    
    # Normalize to [0, 1]
    if total_length > 0:
        distances = [d / total_length for d in distances]
    
    return distances


def find_t(bezier: List[np.ndarray], param: float, t_dist_map: List[float], 
          num_parts: int) -> float:
    """Find optimal t parameter for curve evaluation."""
    if param < 0:
        return 0.0
    if param > 1:
        return 1.0
    
    # Find the closest distance in the mapping
    closest_index = 0
    min_diff = abs(t_dist_map[0] - param)
    
    for i, dist in enumerate(t_dist_map):
        diff = abs(dist - param)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    
    return closest_index / num_parts


def newton_raphson_root_find(bezier: List[np.ndarray], point: np.ndarray, u: float) -> float:
    """Use Newton-Raphson iteration to find better parameter value."""
    curve_point = Bezier.q(bezier, u)
    curve_prime = Bezier.qprime(bezier, u)
    curve_prime_prime = Bezier.qprimeprime(bezier, u)
    
    diff = Maths.subtract(curve_point, point)
    numerator = Maths.dot(diff, curve_prime)
    denominator = (Maths.dot(curve_prime, curve_prime) + 
                  Maths.dot(diff, curve_prime_prime))
    
    if abs(denominator) < 1e-10:
        return u
    
    new_u = u - (numerator / denominator)
    return max(0.0, min(1.0, new_u))  # Clamp to [0, 1]


def reparameterize(bezier: List[np.ndarray], points: List[np.ndarray], 
                  parameters: List[float]) -> List[float]:
    """Given set of points and their parameterization, try to find better parameterization."""
    return [newton_raphson_root_find(bezier, point, u) 
            for point, u in zip(points, parameters)]


def generate_and_report(points: List[np.ndarray], params_orig: List[float], 
                       params_prime: List[float], left_tangent: np.ndarray, 
                       right_tangent: np.ndarray, 
                       progress_callback: Optional[Callable] = None) -> Tuple[List[np.ndarray], float, int]:
    """Generate Bezier curve and report error."""
    bezier_curve = generate_bezier(points, params_prime, left_tangent, right_tangent)
    max_error, split_point = compute_max_error(points, bezier_curve, params_prime)
    
    if progress_callback:
        progress_callback(bezier_curve, max_error, split_point)
    
    return bezier_curve, max_error, split_point


def fit_cubic(points: List[np.ndarray], left_tangent: np.ndarray, 
              right_tangent: np.ndarray, error: float, 
              progress_callback: Optional[Callable] = None) -> List[List[np.ndarray]]:
    """
    Fit a Bezier curve to a (sub)set of digitized points.
    
    Args:
        points: Array of digitized points
        left_tangent: Unit tangent vector at start point
        right_tangent: Unit tangent vector at end point
        error: Tolerance, squared error between points and fitted curve
        progress_callback: Optional callback for progress updates
        
    Returns:
        Array of Bezier curves
    """
    MAX_ITERATIONS = 20
    
    # Use heuristic if region only has two points
    if len(points) == 2:
        dist = Maths.vector_len(Maths.subtract(points[0], points[1])) / 3.0
        bezier_curve = [
            points[0],
            Maths.add_arrays(points[0], Maths.mul_items(left_tangent, dist)),
            Maths.add_arrays(points[1], Maths.mul_items(right_tangent, dist)),
            points[1]
        ]
        return [bezier_curve]
    
    # Parameterize points and attempt to fit curve
    u = chord_length_parameterize(points)
    bezier_curve, max_error, split_point = generate_and_report(
        points, u, u, left_tangent, right_tangent, progress_callback)
    
    if max_error == 0 or max_error < error:
        return [bezier_curve]
    
    # If error not too large, try reparameterization and iteration
    if max_error < (error * error):
        u_prime = u[:]
        prev_err = max_error
        prev_split = split_point
        
        for i in range(MAX_ITERATIONS):
            u_prime = reparameterize(bezier_curve, points, u_prime)
            bezier_curve, max_error, split_point = generate_and_report(
                points, u, u_prime, left_tangent, right_tangent, progress_callback)
            
            if max_error < error:
                return [bezier_curve]
            
            # If development grinds to a halt, abort this attempt
            if split_point == prev_split:
                err_change = max_error / prev_err if prev_err != 0 else 1.0
                if 0.9999 < err_change < 1.0001:
                    break
            
            prev_err = max_error
            prev_split = split_point
    
    # Fitting failed -- split at max error point and fit recursively
    beziers = []
    
    # Calculate tangent at split point
    if split_point - 1 >= 0 and split_point + 1 < len(points):
        center_vector = Maths.subtract(points[split_point - 1], points[split_point + 1])
    else:
        center_vector = Maths.subtract(points[split_point - 1], points[split_point])
    
    # Handle zero vector case
    if np.allclose(center_vector, 0):
        center_vector = Maths.subtract(points[split_point - 1], points[split_point])
        # Rotate 90 degrees: [x, y] -> [-y, x]
        center_vector = np.array([-center_vector[1], center_vector[0]])
    
    to_center_tangent = Maths.normalize(center_vector)
    from_center_tangent = Maths.mul_items(to_center_tangent, -1)
    
    # Recursively fit curves
    beziers.extend(fit_cubic(points[:split_point + 1], left_tangent, to_center_tangent, 
                            error, progress_callback))
    beziers.extend(fit_cubic(points[split_point:], from_center_tangent, right_tangent, 
                            error, progress_callback))
    
    return beziers


def fit_curve(points: List[List[float]], max_error: float, 
              progress_callback: Optional[Callable] = None) -> List[List[List[float]]]:
    """
    Fit one or more Bezier curves to a set of points.
    
    Args:
        points: Array of digitized points, e.g. [[5,5],[5,50],[110,140],[210,160],[320,110]]
        max_error: Tolerance, squared error between points and fitted curve
        progress_callback: Optional callback for progress updates
        
    Returns:
        Array of Bezier curves, where each element is 
        [first-point, control-point-1, control-point-2, second-point]
    """
    if not isinstance(points, list):
        raise TypeError("First argument should be a list")
    
    # Convert to numpy arrays and validate
    np_points = []
    for i, point in enumerate(points):
        if not isinstance(point, list) or not all(isinstance(x, (int, float)) for x in point):
            raise ValueError("Each point should be a list of numbers")
        if i == 0:
            expected_dim = len(point)
        elif len(point) != expected_dim:
            raise ValueError("Each point should have the same number of dimensions")
        np_points.append(np.array(point, dtype=float))
    
    # Remove duplicate points
    unique_points = []
    for point in np_points:
        if not unique_points or not np.allclose(point, unique_points[-1]):
            unique_points.append(point)
    
    if len(unique_points) < 2:
        return []
    
    # Calculate tangents
    left_tangent = create_tangent(unique_points[1], unique_points[0])
    right_tangent = create_tangent(unique_points[-2], unique_points[-1])
    
    # Fit curves
    bezier_curves = fit_cubic(unique_points, left_tangent, right_tangent, 
                             max_error, progress_callback)
    
    # Convert back to lists
    result = []
    for curve in bezier_curves:
        result.append([point.tolist() for point in curve])
    
    return result


if __name__ == "__main__":
    # Simple test
    test_points = [[0, 0], [10, 10], [10, 0], [20, 0]]
    result = fit_curve(test_points, 50)
    print("Test result:", result)