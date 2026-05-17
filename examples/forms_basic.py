"""Basic differential form operations."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import DifferentialForm, Manifold

x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])

dx = DifferentialForm.basis(M, 0)
dy = DifferentialForm.basis(M, 1)
f = DifferentialForm.scalar(M, x * y)

print("dx wedge dy:", dx.wedge(dy))
print("dy wedge dx:", dy.wedge(dx))
print("d(x*y):", f.exterior_derivative())
print("d(d(x*y)):", f.exterior_derivative().exterior_derivative())
