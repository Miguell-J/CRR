"""de Sitter and Anti-de Sitter scalar curvatures."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr.presets import anti_de_sitter_metric, de_sitter_metric

L = sp.symbols("L", positive=True)

dS = de_sitter_metric(radius=L)
AdS = anti_de_sitter_metric(radius=L)

print("de Sitter scalar curvature:")
print(dS.scalar_curvature(simplify=True))
print("Anti-de Sitter scalar curvature:")
print(AdS.scalar_curvature(simplify=True))
