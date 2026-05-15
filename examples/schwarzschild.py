"""Schwarzschild spacetime example."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import Manifold, Metric

t, r, theta, phi, M_sym = sp.symbols("t r theta phi M", positive=True)
f = 1 - 2 * M_sym / r

manifold = Manifold("Schwarzschild", 4, [t, r, theta, phi])
metric = Metric(
    manifold,
    [
        [-f, 0, 0, 0],
        [0, 1 / f, 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta) ** 2],
    ],
)

ricci = metric.ricci_tensor(simplify=True)
print("Representative Ricci components:")
for index in [(0, 0), (1, 1), (2, 2), (3, 3)]:
    print(f"R{index} =", sp.simplify(ricci[index]))
