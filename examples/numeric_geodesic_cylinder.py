"""Numerical geodesic on a cylinder."""

import numpy as np
import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

theta, z = sp.symbols("theta z")
R = 2

cylinder = ParametrizedMap(
    name="Cylinder",
    parameters=[theta, z],
    components=[
        R * sp.cos(theta),
        R * sp.sin(theta),
        z,
    ],
)

metric = cylinder.pullback_metric(simplify=True)
solution = metric.solve_geodesic([0, 0], [0.5, 0.25], (0, 8), num_points=300)

print("success:", solution.success)
print("final state:", solution.final_state())
print("theta linear drift:", float(np.max(np.abs(solution.x[:, 0] - 0.5 * solution.t))))
print("z linear drift:", float(np.max(np.abs(solution.x[:, 1] - 0.25 * solution.t))))
print("energy drift:", float(np.max(np.abs(solution.energy() - solution.energy()[0]))))
