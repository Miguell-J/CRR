"""U(1) gauge field and Maxwell field strength."""

import sympy as sp

import _bootstrap  # noqa: F401

from crr import DifferentialForm, Manifold
from crr.gauge import abelian_gauge_potential


x, y, z = sp.symbols("x y z")
M = Manifold("R3", 3, [x, y, z])
dy = DifferentialForm.basis(M, 1)

A = abelian_gauge_potential(M, x * dy)
F = A.curvature()
bianchi = A.bianchi_identity()

print("A =", A.to_latex())
print("F = dA =", F.to_latex())
print("dF =", bianchi.to_latex())
print("F components =", F.components[0].components)
