"""Alcubierre metric preset."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr.presets import alcubierre_metric, minkowski_metric

v_s, f_s = sp.symbols("v_s f_s", real=True)

metric = alcubierre_metric(velocity=v_s, shape_function=f_s)
flat_limit = alcubierre_metric(velocity=0)

print("Alcubierre metric:")
print(metric.components)
print("Cross term g_tx:")
print(metric.components[0, 1])
print("v_s=0 gives Minkowski:")
print(flat_limit.components.equals(minkowski_metric().components))
