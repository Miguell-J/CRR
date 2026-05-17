"""A simple su(2) connection with nonzero bracket curvature."""

import sympy as sp

import _bootstrap  # noqa: F401

from crr import DifferentialForm, Manifold
from crr.gauge import su2_connection


x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)
zero = DifferentialForm.zero(M, 1)

A = su2_connection(M, [dx, dy, zero])
F = A.curvature()

print("A =", A.to_latex())
print("F = dA + 1/2[A wedge A] =", F.to_latex())
print("F components =", [component.components for component in F.components])
