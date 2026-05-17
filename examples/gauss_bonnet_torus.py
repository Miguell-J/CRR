"""Gauss-Bonnet check for a standard torus."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap, gauss_bonnet_check

u, v = sp.symbols("u v")

torus = ParametrizedMap(
    name="Torus",
    parameters=[u, v],
    components=[
        (2 + sp.cos(v)) * sp.cos(u),
        (2 + sp.cos(v)) * sp.sin(u),
        sp.sin(v),
    ],
)

result = gauss_bonnet_check(
    torus,
    [(u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi)],
    euler_characteristic=0,
)

print("curvature integral:", result.curvature_integral)
print("expected:", result.expected)
print("passed:", result.passed)
