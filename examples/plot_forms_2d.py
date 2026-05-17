"""Plot 0-, 1-, and 2-forms on a 2D chart."""

from pathlib import Path
import os

import sympy as sp

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib

matplotlib.use("Agg")

import _bootstrap  # noqa: F401

from crr import DifferentialForm, Manifold


OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

x, y = sp.symbols("x y")
manifold = Manifold("R2", 2, [x, y])
dx = DifferentialForm.basis(manifold, 0)
dy = DifferentialForm.basis(manifold, 1)

scalar = DifferentialForm.scalar(manifold, x**2 - y**2)
one_form = -y * dx + x * dy
two_form = sp.sin(x * y) * dx.wedge(dy)

scalar.plot_2d((-2, 2), (-2, 2), show=False, save_path=OUTPUT / "form_scalar.png")
one_form.plot_2d((-2, 2), (-2, 2), show=False, save_path=OUTPUT / "form_one_form.png")
two_form.plot_2d((-2, 2), (-2, 2), show=False, save_path=OUTPUT / "form_density.png")

print("Saved 2D form plots in", OUTPUT)
