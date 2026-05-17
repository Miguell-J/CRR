"""Yang-Mills action density F wedge *F."""

import sympy as sp

import _bootstrap  # noqa: F401

from crr import DifferentialForm
from crr.gauge import abelian_gauge_potential, yang_mills_action_density
from crr.presets import euclidean_metric


x, y = sp.symbols("x y")
metric = euclidean_metric(dim=2, coordinates=[x, y])
M = metric.manifold
dy = DifferentialForm.basis(M, 1)

A = abelian_gauge_potential(M, x * dy)
F = A.curvature()
density = yang_mills_action_density(F, metric, simplify=True)

print("F =", F.to_latex())
print("F wedge *F =", density.to_latex())
print("density components =", density.components)
