"""Unit 2-sphere example."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

theta, phi = sp.symbols("theta phi")

M = Manifold("S2", 2, [theta, phi])
g = Metric(M, [[1, 0], [0, sp.sin(theta) ** 2]])

print(g.scalar_curvature(simplify=True))
