"""Extrinsic geometry of the unit sphere in Euclidean R3."""

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

metric = sphere.pullback_metric(simplify=True)
K = sphere.gaussian_curvature_extrinsic(simplify=True)
R = metric.scalar_curvature(simplify=True)

print("Normal vector:")
print(sphere.normal_vector(simplify=True))
print("Second fundamental form:")
print(sphere.second_fundamental_form(simplify=True))
print("Gaussian curvature K:")
print(K)
print("Mean curvature H:")
print(sphere.mean_curvature(simplify=True))
print("Intrinsic scalar curvature R:")
print(R)
print("K == R/2:")
print(sp.simplify(K - R / 2) == 0)
