"""Gauss-Bonnet check for the unit sphere."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap, gauss_bonnet_check

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

result = gauss_bonnet_check(
    sphere,
    [(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)],
    euler_characteristic=2,
)

print("curvature integral:", result.curvature_integral)
print("expected:", result.expected)
print("passed:", result.passed)
