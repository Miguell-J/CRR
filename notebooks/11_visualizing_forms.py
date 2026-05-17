# %% [markdown]
# # Visualizing Differential Forms

# %%
import _bootstrap  # noqa: F401

import importlib.util
import os

if importlib.util.find_spec("matplotlib") is None:
    print("Skipping visualization notebook: matplotlib is not installed.")
    raise SystemExit(0)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/crr-matplotlib")

import matplotlib
import sympy as sp

matplotlib.use("Agg")

from crr import DifferentialForm, Manifold

# %%
x, y = sp.symbols("x y")
manifold = Manifold("R2", 2, [x, y])
dx = DifferentialForm.basis(manifold, 0)
dy = DifferentialForm.basis(manifold, 1)

# %%
DifferentialForm.scalar(manifold, x**2 - y**2).plot_2d((-2, 2), (-2, 2), show=False)
(-y * dx + x * dy).plot_2d((-2, 2), (-2, 2), show=False)
(sp.sin(x * y) * dx.wedge(dy)).plot_2d((-2, 2), (-2, 2), show=False)

print("0-, 1-, and 2-form plots created")
