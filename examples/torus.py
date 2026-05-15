"""Torus as an embedded surface in Euclidean R3."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

u, v, R, r = sp.symbols("u v R r", positive=True)

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

print("Induced metric:")
print(metric.components)
print("Nonzero Christoffel symbols:")
print(metric.christoffel_symbols(simplify=True).nonzero_components())
print("Scalar curvature:")
print(metric.scalar_curvature(simplify=True))
