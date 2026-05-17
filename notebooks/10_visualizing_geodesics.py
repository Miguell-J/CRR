# %% [markdown]
# # Visualizing Numerical Geodesics

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

from crr.presets import sphere_metric, sphere_surface

# %%
metric = sphere_metric(radius=1)
surface = sphere_surface(radius=1)
solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 2 * np.pi), num_points=120)

# %%
solution.plot_coordinates(show=False)
solution.plot_energy(show=False)
surface.plot_surface_with_geodesic(solution, (0, np.pi), (0, 2 * np.pi), resolution=30, show=False)

print("Geodesic success:", solution.success)
print("Final state:", solution.final_state())
