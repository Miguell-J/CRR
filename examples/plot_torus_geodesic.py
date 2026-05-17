"""Plot a numerical geodesic on a torus."""

from pathlib import Path
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib

matplotlib.use("Agg")

import _bootstrap  # noqa: F401

from crr.presets import torus_metric, torus_surface


OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

surface = torus_surface(major_radius=2, minor_radius=0.7)
metric = torus_metric(major_radius=2, minor_radius=0.7)
solution = metric.solve_geodesic([0, 0.4], [0.6, 0.2], (0, 8), num_points=300)

surface.plot_surface_with_geodesic(
    solution,
    (0, 2 * np.pi),
    (0, 2 * np.pi),
    resolution=60,
    show=False,
    title="Torus geodesic",
    save_path=OUTPUT / "torus_geodesic.png",
)

print("Success:", solution.success)
print("Saved", OUTPUT / "torus_geodesic.png")
