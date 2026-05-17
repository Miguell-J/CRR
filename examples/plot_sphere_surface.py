"""Plot a preset sphere surface."""

from pathlib import Path
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib

matplotlib.use("Agg")

import _bootstrap  # noqa: F401

from crr.presets import sphere_surface


OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

sphere = sphere_surface(radius=1)
fig, ax = sphere.plot_surface(
    (0, np.pi),
    (0, 2 * np.pi),
    resolution=60,
    show=False,
    title="Unit sphere",
    save_path=OUTPUT / "sphere_surface.png",
)

print("Saved", OUTPUT / "sphere_surface.png")
