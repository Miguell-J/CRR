"""Plot an equator geodesic on a sphere."""

from pathlib import Path
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib

matplotlib.use("Agg")

import _bootstrap  # noqa: F401

from crr.presets import sphere_metric, sphere_surface


OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

surface = sphere_surface(radius=1)
metric = sphere_metric(radius=1)
solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 2 * np.pi), num_points=300)

surface.plot_surface_with_geodesic(
    solution,
    (0, np.pi),
    (0, 2 * np.pi),
    resolution=60,
    show=False,
    title="Equator geodesic",
    save_path=OUTPUT / "sphere_geodesic.png",
)

print("Success:", solution.success)
print("Saved", OUTPUT / "sphere_geodesic.png")
