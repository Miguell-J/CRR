# %% [markdown]
# # Visualizing Parametrized Surfaces

# %%
import _bootstrap  # noqa: F401

import importlib.util
import os

if importlib.util.find_spec("matplotlib") is None:
    print("Skipping visualization notebook: matplotlib is not installed.")
    raise SystemExit(0)

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
