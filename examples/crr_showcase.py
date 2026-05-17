"""Compact CRR showcase for v1.0.1."""

import numpy as np
import sympy as sp

import _bootstrap  # noqa: F401

from crr import DifferentialForm, Manifold
from crr.gauge import abelian_gauge_potential
from crr.lie import su2_algebra
from crr.presets import sphere_metric, sphere_surface, torus_surface
from crr import gauss_bonnet_check


theta, phi = sp.symbols("theta phi")
sphere_metric_ = sphere_metric(radius=1, coordinates=[theta, phi])
print("Sphere scalar curvature:", sphere_metric_.scalar_curvature(simplify=True))

u, v = sp.symbols("u v")
torus = torus_surface(major_radius=2, minor_radius=1, coordinates=[u, v])
print("Torus Gaussian curvature:", torus.gaussian_curvature_extrinsic(simplify=True))

sphere = sphere_surface(radius=1, coordinates=[theta, phi])
gb = gauss_bonnet_check(sphere, [(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], euler_characteristic=2)
print("Gauss-Bonnet sphere:", gb)

x, y = sp.symbols("x y")
M = Manifold("R2", 2, [x, y])
f = DifferentialForm.scalar(M, x * y)
print("d^2 f = 0?", f.exterior_derivative().exterior_derivative().equals(DifferentialForm.zero(M, 2)))

su2 = su2_algebra()
print("su2 [T1,T2] =", su2.basis_element(0).bracket(su2.basis_element(1)))

dy = DifferentialForm.basis(M, 1)
A = abelian_gauge_potential(M, x * dy)
print("U(1) F = dA:", A.curvature().to_latex())

solution = sphere_metric_.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 1), num_points=30)
print("Sphere geodesic success:", solution.success)
