"""Plot numerical geodesic energy along a sphere geodesic."""

from pathlib import Path
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib

matplotlib.use("Agg")

import _bootstrap  # noqa: F401

from crr.presets import sphere_metric


OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

metric = sphere_metric(radius=1)
solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 10), num_points=300)
solution.plot_energy(show=False, title="Sphere geodesic energy", save_path=OUTPUT / "geodesic_energy.png")

energy = solution.energy()
print("Success:", solution.success)
print("Energy drift:", float(abs(energy[-1] - energy[0])))
print("Saved", OUTPUT / "geodesic_energy.png")
