# %% [markdown]
# # Visualizing Parametrized Surfaces

# %%
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from crr.presets import sphere_surface, torus_surface

# %%
sphere = sphere_surface(radius=1)
sphere.plot_surface((0, np.pi), (0, 2 * np.pi), resolution=30, show=False)
print("Sphere surface plot created")

# %%
torus = torus_surface(major_radius=2, minor_radius=0.7)
torus.plot_surface((0, 2 * np.pi), (0, 2 * np.pi), resolution=30, show=False)
print("Torus surface plot created")
