"""Poincare half-plane example."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

x, y = sp.symbols("x y", positive=True)

M = Manifold("H2", 2, [x, y])
g = Metric(M, [[1 / y**2, 0], [0, 1 / y**2]])

print("Scalar curvature:", g.scalar_curvature(simplify=True))
