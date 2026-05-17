"""Gauge curvature preview with algebra-valued forms."""

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
        (x + y) * dx,
    ],
)

F = A.exterior_derivative() + sp.Rational(1, 2) * A.bracket_wedge(A)

print("Connection preview A:")
print(A.to_latex())
print("Curvature preview F = dA + 1/2 [A wedge A]:")
print(F.simplify().to_latex())
