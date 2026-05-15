"""Unit sphere as an embedded surface in Euclidean R3."""

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

print("Induced metric:")
print(metric.components)
print("Scalar curvature:")
print(metric.scalar_curvature(simplify=True))
