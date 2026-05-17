"""Preset relativity metrics."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr.presets import (
    alcubierre_metric,
    anti_de_sitter_metric,
    de_sitter_metric,
    eddington_finkelstein_metric,
    flrw_metric,
    godel_metric,
    kerr_metric,
    kruskal_szekeres_metric,
    minkowski_metric,
    reissner_nordstrom_metric,
    schwarzschild_metric,
)

minkowski = minkowski_metric()
print("Minkowski scalar curvature:")
print(minkowski.scalar_curvature(simplify=True))

schwarzschild = schwarzschild_metric()
ricci = schwarzschild.ricci_tensor(simplify=True)
print("Schwarzschild representative Ricci components:")
for index in [(0, 0), (1, 1), (2, 2), (3, 3)]:
    print(f"R{index} =", sp.simplify(ricci[index]))

flrw = flrw_metric()
print("FLRW scalar curvature:")
print(flrw.scalar_curvature())

print("Additional relativity presets:")
for factory in [
    kerr_metric,
    godel_metric,
    kruskal_szekeres_metric,
    eddington_finkelstein_metric,
    reissner_nordstrom_metric,
    de_sitter_metric,
    anti_de_sitter_metric,
    alcubierre_metric,
]:
    metric = factory()
    print(factory.__name__, metric.components.shape)
