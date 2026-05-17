# %% [markdown]
# # Visualizing Curvature

# %%
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")

from crr.presets import torus_surface

# %%
torus = torus_surface(major_radius=2, minor_radius=0.7)
torus.plot_curvature(
    u_range=(0, 2 * np.pi),
    v_range=(0, 2 * np.pi),
    resolution=30,
    show=False,
)
print("Torus Gaussian curvature plot created")

# %%
torus.plot_curvature(
    curvature="mean",
    u_range=(0, 2 * np.pi),
    v_range=(0, 2 * np.pi),
    resolution=30,
    show=False,
)
print("Torus mean curvature plot created")
