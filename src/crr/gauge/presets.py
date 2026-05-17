"""Gauge-field presets."""

from __future__ import annotations

import sympy as sp

from crr.forms import DifferentialForm
from crr.gauge.connection import GaugePotential
from crr.lie import abelian_lie_algebra, su2_algebra


def abelian_gauge_potential(manifold, potential_form: DifferentialForm | None = None, name: str = "A") -> GaugePotential:
    """Return a one-dimensional abelian gauge potential."""

    if potential_form is None:
        potential_form = DifferentialForm.zero(manifold, 1)
    if not isinstance(potential_form, DifferentialForm):
        raise TypeError("potential_form must be a DifferentialForm.")
    if potential_form.manifold != manifold or potential_form.degree != 1:
        raise ValueError("potential_form must be a 1-form on the given manifold.")
    return GaugePotential(abelian_lie_algebra(1, name="u(1)"), [potential_form], name=name)


def u1_connection_from_potential(manifold, components, coordinates=None, name: str = "A") -> GaugePotential:
    """Build ``A = A_i dx^i`` from coordinate components and return a U(1) connection."""

    if coordinates is not None and tuple(coordinates) != tuple(manifold.coordinates):
        raise ValueError("coordinates, when provided, must match manifold.coordinates.")
    return abelian_gauge_potential(manifold, DifferentialForm.one_form(manifold, components), name=name)


def su2_connection(manifold, component_forms, name: str = "A") -> GaugePotential:
    """Return an ``su(2)`` gauge potential from three 1-form components."""

    if len(component_forms) != 3:
        raise ValueError("su2_connection expects exactly three component 1-forms.")
    for component in component_forms:
        if not isinstance(component, DifferentialForm):
            raise TypeError("su2 component_forms must be DifferentialForm instances.")
        if component.manifold != manifold or component.degree != 1:
            raise ValueError("Each su2 component must be a 1-form on the given manifold.")
    return GaugePotential(su2_algebra(), list(component_forms), name=name)


def pure_gauge_abelian(manifold, chi, name: str = "A") -> GaugePotential:
    """Return the pure abelian gauge ``A = d chi``, whose curvature is zero."""

    scalar = chi if isinstance(chi, DifferentialForm) else DifferentialForm.scalar(manifold, sp.sympify(chi))
    if scalar.manifold != manifold or scalar.degree != 0:
        raise ValueError("chi must be a scalar expression or a 0-form on the given manifold.")
    return abelian_gauge_potential(manifold, scalar.exterior_derivative(), name=name)
