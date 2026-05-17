"""Sphere area form from the Hodge star."""

import sympy as sp

import _bootstrap  # noqa: F401
from crr import DifferentialForm
from crr.presets import sphere_metric

theta, phi = sp.symbols("theta phi")
g = sphere_metric(radius=1, coordinates=[theta, phi])
one = DifferentialForm.scalar(g.manifold, 1)
area_form = one.hodge_star(g, simplify=True)

print("Area form:", area_form)
print("Area:", area_form.integrate([(theta, 0, sp.pi), (phi, 0, 2 * sp.pi)], simplify=True))
