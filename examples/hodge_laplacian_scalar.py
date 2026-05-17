"""Hodge Laplacian on a scalar in Euclidean R2."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import DifferentialForm, Manifold
from crr.presets import euclidean_metric

x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
g = euclidean_metric(dim=2, coordinates=[x, y])
f = DifferentialForm.scalar(M, x**2 + y**2)

print(f.hodge_laplacian(g, simplify=True))
