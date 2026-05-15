"""Extrinsic geometry of a cylinder in Euclidean R3."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

theta, z, R = sp.symbols("theta z R", positive=True)

cylinder = ParametrizedMap(
    name="Cylinder",
    parameters=[theta, z],
    components=[
        R * sp.cos(theta),
        R * sp.sin(theta),
        z,
    ],
)

metric = cylinder.pullback_metric(simplify=True)

print("Gaussian curvature K:")
print(cylinder.gaussian_curvature_extrinsic(simplify=True))
print("Mean curvature H:")
print(cylinder.mean_curvature(simplify=True))
print("Intrinsic scalar curvature R:")
print(metric.scalar_curvature(simplify=True))
