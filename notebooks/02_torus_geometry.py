# %% [markdown]
# # Torus Geometry

# %%
import _bootstrap  # noqa: F401

import sympy as sp

from crr.presets import torus_metric, torus_surface

u, v = sp.symbols("u v")
torus = torus_surface(major_radius=2, minor_radius=1, coordinates=[u, v])
g = torus_metric(major_radius=2, minor_radius=1, coordinates=[u, v])

print("Scalar curvature")
print(g.scalar_curvature(simplify=True))
print("Extrinsic Gaussian curvature")
print(torus.gaussian_curvature_extrinsic(simplify=True))
print("Total Gaussian curvature")
print(torus.integrate_gaussian_curvature([(u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi)], simplify=True))
