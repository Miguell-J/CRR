"""Covariant derivative and metric compatibility on the unit sphere."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

theta, phi = sp.symbols("theta phi")

M = Manifold("S2", 2, [theta, phi])
g = Metric(M, [[1, 0], [0, sp.sin(theta) ** 2]])

nabla_g = g.covariant_derivative_covariant_2tensor(g.components, simplify=True)
print("Nonzero components of nabla_k g_ij:")
print(nabla_g.nonzero_components())

f = sp.cos(theta)
df = g.covariant_derivative_scalar(f, simplify=True)
print("nabla_i cos(theta):")
print(df.tolist())
