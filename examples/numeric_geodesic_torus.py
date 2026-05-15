"""Numerical geodesic on a torus."""

import numpy as np
import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

u, v = sp.symbols("u v")
R = 2.0
r = 0.5

torus = ParametrizedMap(
    name="Torus",
    parameters=[u, v],
    components=[
        (R + r * sp.cos(v)) * sp.cos(u),
        (R + r * sp.cos(v)) * sp.sin(u),
        r * sp.sin(v),
    ],
)

metric = torus.pullback_metric(simplify=True)
solution = metric.solve_geodesic([0, 0.4], [0.6, 0.3], (0, 12), num_points=600)
energy = solution.energy()

print("success:", solution.success)
print("final state:", solution.final_state())
print("energy drift:", float(np.max(np.abs(energy - energy[0]))))
