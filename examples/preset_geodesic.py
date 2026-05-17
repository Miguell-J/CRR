"""Numerical geodesic using a preset sphere metric."""

import numpy as np

import _bootstrap  # noqa: F401
from crr.presets import sphere_metric

metric = sphere_metric(radius=1)
solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 6), num_points=200)

print("success:", solution.success)
print("final state:", solution.final_state())
print("theta drift:", float(np.max(np.abs(solution.x[:, 0] - np.pi / 2))))
