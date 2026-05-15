"""Numerical geodesic on the unit sphere."""

import numpy as np
import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

theta, phi = sp.symbols("theta phi")

sphere = ParametrizedMap(
    name="UnitSphere",
    parameters=[theta, phi],
    components=[
        sp.sin(theta) * sp.cos(phi),
        sp.sin(theta) * sp.sin(phi),
        sp.cos(theta),
    ],
)

metric = sphere.pullback_metric(simplify=True)
solution = metric.solve_geodesic(
    x0=[np.pi / 2, 0],
    v0=[0, 1],
    t_span=(0, 10),
    num_points=500,
)

energy = solution.energy()
print("success:", solution.success)
print("final state:", solution.final_state())
print("energy drift:", float(np.max(np.abs(energy - energy[0]))))

try:
    from crr.visualization import plot_embedded_geodesic

    plot_embedded_geodesic(sphere, solution)
except ImportError:
    pass
