"""Preset sphere surface and metric."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr.presets import sphere_metric, sphere_surface

R = sp.symbols("R", positive=True)

sphere = sphere_surface(radius=R)
theta, phi = sphere.parameters
surface_metric = sphere.pullback_metric(simplify=True)
intrinsic_metric = sphere_metric(radius=R, coordinates=[theta, phi])

print("Induced metric:")
print(surface_metric.components)
print("Preset intrinsic metric:")
print(intrinsic_metric.components)
print("Scalar curvature:")
print(intrinsic_metric.scalar_curvature(simplify=True))
print("Area:")
print(sphere.integrate_area([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True))
print("Total Gaussian curvature:")
print(sphere.integrate_gaussian_curvature([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True))
