"""Extrinsic geometry of a torus in Euclidean R3."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import ParametrizedMap

u, v, R, r = sp.symbols("u v R r", positive=True)

torus = ParametrizedMap(
    name="Torus",
    parameters=[u, v],
    components=[
        (R + r * sp.cos(v)) * sp.cos(u),
        (R + r * sp.cos(v)) * sp.sin(u),
        r * sp.sin(v),
    ],
)

metric = torus.pullback_metric(simplify=True)
K = torus.gaussian_curvature_extrinsic(simplify=True)
scalar = metric.scalar_curvature(simplify=True)

print("Gaussian curvature K:")
print(K)
print("Mean curvature H:")
print(torus.mean_curvature(simplify=True))
print("Intrinsic scalar curvature R:")
print(scalar)
print("K == R/2:")
print(sp.simplify(K - scalar / 2) == 0)
