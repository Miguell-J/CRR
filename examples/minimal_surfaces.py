"""Catenoid and helicoid mean curvature examples."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

u, v = sp.symbols("u v")

catenoid = ParametrizedMap(
    name="Catenoid",
    parameters=[u, v],
    components=[
        sp.cosh(v) * sp.cos(u),
        sp.cosh(v) * sp.sin(u),
        v,
    ],
)

helicoid = ParametrizedMap(
    name="Helicoid",
    parameters=[u, v],
    components=[
        v * sp.cos(u),
        v * sp.sin(u),
        u,
    ],
)

print("Catenoid mean curvature:")
print(catenoid.mean_curvature(simplify=True))
print("Helicoid mean curvature:")
print(helicoid.mean_curvature(simplify=True))
