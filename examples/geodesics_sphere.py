"""Geodesic acceleration on the unit sphere."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

theta, phi = sp.symbols("theta phi")
v_theta, v_phi = sp.symbols("v_theta v_phi")

M = Manifold("S2", 2, [theta, phi])
g = Metric(M, [[1, 0], [0, sp.sin(theta) ** 2]])

acceleration = g.geodesic_acceleration([v_theta, v_phi], simplify=True)

print("Using a^k = -Gamma^k_ij v^i v^j")
print("a_theta =", acceleration[0])
print("a_phi =", acceleration[1])
