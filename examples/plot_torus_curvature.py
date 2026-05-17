"""Plot Gaussian curvature on a torus."""

from pathlib import Path
import os

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib

matplotlib.use("Agg")

import _bootstrap  # noqa: F401

from crr.presets import torus_surface


OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

torus = torus_surface(major_radius=2, minor_radius=0.7)
torus.plot_curvature(
    u_range=(0, 2 * np.pi),
    v_range=(0, 2 * np.pi),
    resolution=60,
    show=False,
    title="Torus Gaussian curvature",
    save_path=OUTPUT / "torus_curvature.png",
)

print("Saved", OUTPUT / "torus_curvature.png")
