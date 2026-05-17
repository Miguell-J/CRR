"""Bianchi identity for a manageable non-abelian connection."""

import sympy as sp

import _bootstrap  # noqa: F401

from crr import DifferentialForm, Manifold
from crr.gauge import su2_connection
from crr.lie import AlgebraValuedForm


x, y, z = sp.symbols("x y z")
M = Manifold("R3", 3, [x, y, z])
dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)
zero = DifferentialForm.zero(M, 1)

A = su2_connection(M, [dx, dy, zero])
F = A.curvature()
dAF = A.bianchi_identity()

print("F =", F.to_latex())
print("d_A F =", dAF.to_latex())
print("Bianchi identity holds?", dAF.equals(AlgebraValuedForm.zero(A.algebra, M, 3)))
