"""Integrate the area of a cylinder."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

theta, z, R, h = sp.symbols("theta z R h", positive=True)

cylinder = ParametrizedMap(
    name="Cylinder",
    parameters=[theta, z],
    components=[
        R * sp.cos(theta),
        R * sp.sin(theta),
        z,
    ],
)

area = cylinder.integrate_area([(theta, 0, 2 * sp.pi), (z, 0, h)], simplify=True)
print(area)
