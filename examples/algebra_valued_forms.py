"""Algebra-valued differential forms."""

import sympy as sp

import _bootstrap  # noqa: F401

from crr import DifferentialForm, Manifold
from crr.lie import AlgebraValuedForm, su2_algebra


x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)

su2 = su2_algebra()
A = AlgebraValuedForm(
    su2,
    [
        x * dx,
        y * dy,
        DifferentialForm.zero(M, 1),
    ],
)

print("A =", A.to_latex())
print("dA =", A.exterior_derivative().to_latex())
print("[A wedge A] =", A.bracket_wedge(A).simplify().to_latex())
