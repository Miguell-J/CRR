"""Preset torus surface and metric."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr.presets import torus_metric, torus_surface

u, v = sp.symbols("u v")
R = 2
r = 1

surface = torus_surface(major_radius=R, minor_radius=r, coordinates=[u, v])
metric = torus_metric(major_radius=R, minor_radius=r, coordinates=[u, v])

print("Scalar curvature:")
print(metric.scalar_curvature(simplify=True))
print("Extrinsic Gaussian curvature:")
print(surface.gaussian_curvature_extrinsic(simplify=True))
print("Total Gaussian curvature:")
print(surface.integrate_gaussian_curvature([(u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi)], simplify=True))
