"""Integrate a scalar field against a metric density."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

x, y = sp.symbols("x y")
metric = Metric(Manifold("UnitSquare", 2, [x, y]), [[1, 0], [0, 1]])

value = metric.integrate_scalar(x + y, [(x, 0, 1), (y, 0, 1)], simplify=True)
print(value)
