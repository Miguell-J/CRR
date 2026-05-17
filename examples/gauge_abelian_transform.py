"""Abelian gauge transformation A -> A + d chi."""

import sympy as sp

import _bootstrap  # noqa: F401

from crr import DifferentialForm, Manifold
from crr.gauge import abelian_gauge_potential, abelian_gauge_transform


x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
dy = DifferentialForm.basis(M, 1)

A = abelian_gauge_potential(M, x * dy)
A_prime = abelian_gauge_transform(A, x * y)

print("A =", A.to_latex())
print("A' =", A_prime.to_latex())
print("F =", A.curvature().to_latex())
print("F' =", A_prime.curvature().to_latex())
print("F' = F?", A_prime.curvature().equals(A.curvature()))
