"""Integrate the area of the unit sphere."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

theta, phi = sp.symbols("theta phi")

sphere = ParametrizedMap(
    name="UnitSphere",
    parameters=[theta, phi],
    components=[
        sp.sin(theta) * sp.cos(phi),
        sp.sin(theta) * sp.sin(phi),
        sp.cos(theta),
    ],
)

area = sphere.integrate_area([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True)
print(area)
