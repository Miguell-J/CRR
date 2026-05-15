"""Euclidean plane example."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

x, y = sp.symbols("x y")

M = Manifold("R2", 2, [x, y])
g = Metric(M, [[1, 0], [0, 1]])

print("Nonzero Christoffel symbols:", g.christoffel_symbols().nonzero_components())
print("Scalar curvature:", g.scalar_curvature(simplify=True))
