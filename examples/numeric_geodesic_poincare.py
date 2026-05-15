"""Numerical geodesic in the Poincare half-plane."""

import numpy as np
import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

x, y = sp.symbols("x y", positive=True)
metric = Metric(Manifold("H2", 2, [x, y]), [[1 / y**2, 0], [0, 1 / y**2]])

solution = metric.solve_geodesic(
    x0=[0, 1],
    v0=[0.5, 0.2],
    t_span=(0, 2),
    num_points=300,
)

energy = solution.energy()
print("success:", solution.success)
print("final state:", solution.final_state())
print("energy drift:", float(np.max(np.abs(energy - energy[0]))))
