# %% [markdown]
# # Gauss-Bonnet Checks

# %%
import sympy as sp

from crr import gauss_bonnet_check
from crr.presets import sphere_surface, torus_surface

theta, phi = sp.symbols("theta phi")
sphere = sphere_surface(coordinates=[theta, phi])
print(gauss_bonnet_check(sphere, [(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], 2))


# %%
u, v = sp.symbols("u v")
torus = torus_surface(major_radius=2, minor_radius=1, coordinates=[u, v])
print(gauss_bonnet_check(torus, [(u, 0, 2 * sp.pi), (v, 0, 2 * sp.pi)], 0))
