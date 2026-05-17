# %% [markdown]
# # Visualizing Curvature

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
