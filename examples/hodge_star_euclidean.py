"""Hodge star in Euclidean R2 and R3."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import DifferentialForm, Manifold
from crr.presets import euclidean_metric

x, y, z = sp.symbols("x y z")

R2 = Manifold("R2", 2, [x, y])
g2 = euclidean_metric(dim=2, coordinates=[x, y])
dx2 = DifferentialForm.basis(R2, 0)
dy2 = DifferentialForm.basis(R2, 1)

print("R2 *dx:", dx2.hodge_star(g2, simplify=True))
print("R2 *dy:", dy2.hodge_star(g2, simplify=True))
print("R2 *(dx wedge dy):", dx2.wedge(dy2).hodge_star(g2, simplify=True))

R3 = Manifold("R3", 3, [x, y, z])
g3 = euclidean_metric(dim=3, coordinates=[x, y, z])
dx = DifferentialForm.basis(R3, 0)
dy = DifferentialForm.basis(R3, 1)
dz = DifferentialForm.basis(R3, 2)

print("R3 *dx:", dx.hodge_star(g3, simplify=True))
print("R3 *dy:", dy.hodge_star(g3, simplify=True))
print("R3 *dz:", dz.hodge_star(g3, simplify=True))
print("R3 volume:", dx.wedge(dy).wedge(dz))
