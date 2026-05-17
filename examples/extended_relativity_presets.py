"""Build the extended relativity preset metrics."""

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
    robertson_walker_metric,
    schwarzschild_metric,
)

for factory in [
    minkowski_metric,
    schwarzschild_metric,
    flrw_metric,
    robertson_walker_metric,
    reissner_nordstrom_metric,
    kerr_metric,
    kruskal_szekeres_metric,
    eddington_finkelstein_metric,
    de_sitter_metric,
    anti_de_sitter_metric,
    godel_metric,
    alcubierre_metric,
]:
    metric = factory()
    print(factory.__name__, metric.components.shape)
