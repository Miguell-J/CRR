# %% [markdown]
# # Geometry of the 2-Sphere

# %%
import sympy as sp

from crr.presets import sphere_metric, sphere_surface

theta, phi = sp.symbols("theta phi")
sphere = sphere_surface(radius=1, coordinates=[theta, phi])
g = sphere_metric(radius=1, coordinates=[theta, phi])

print("Parametrization")
print(sphere)
print("Metric")
print(g.components)
print("Scalar curvature")
print(g.scalar_curvature(simplify=True))


# %%
print("Area")
print(sphere.integrate_area([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True))
