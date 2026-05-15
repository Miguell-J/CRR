"""Cylinder as an embedded surface in Euclidean R3."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

theta, z, R = sp.symbols("theta z R", positive=True)

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

print("Induced metric:")
print(metric.components)
print("Scalar curvature:")
print(metric.scalar_curvature(simplify=True))
