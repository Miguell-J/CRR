"""Kerr and Reissner-Nordstrom limiting cases."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr.presets import kerr_metric, reissner_nordstrom_metric, schwarzschild_metric

t, r, theta, phi = sp.symbols("t r theta phi")
M = sp.symbols("M", positive=True)

sch = schwarzschild_metric(mass=M, coordinates=[t, r, theta, phi])
kerr_a0 = kerr_metric(mass=M, spin=0, coordinates=[t, r, theta, phi])
rn_q0 = reissner_nordstrom_metric(mass=M, charge=0, coordinates=[t, r, theta, phi])

print("Kerr a=0 matches Schwarzschild g_tt:")
print(sp.simplify(kerr_a0.components[0, 0] - sch.components[0, 0]) == 0)
print("Kerr a=0 has no t-phi cross term:")
print(sp.simplify(kerr_a0.components[0, 3]) == 0)
print("Reissner-Nordstrom Q=0 matches Schwarzschild:")
print(rn_q0.components.equals(sch.components))
