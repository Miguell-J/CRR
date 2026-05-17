"""Utility functions for local symbolic gauge theory."""

from __future__ import annotations

from crr.gauge.field_strength import GaugeCurvature
from crr.lie import AlgebraValuedForm


def covariant_exterior_derivative(connection, omega) -> AlgebraValuedForm:
    """Return ``d_A omega = d omega + [A wedge omega]``.

    ``omega`` may be a plain ``AlgebraValuedForm`` or a ``GaugeCurvature``.
    """

    if isinstance(omega, GaugeCurvature):
        form = omega.as_algebra_valued_form()
    elif isinstance(omega, AlgebraValuedForm):
        form = omega
    else:
        raise TypeError("omega must be an AlgebraValuedForm or GaugeCurvature.")
    if connection.algebra != form.algebra:
        raise ValueError("Connection and form must use the same Lie algebra.")
    if connection.manifold != form.manifold:
        raise ValueError("Connection and form must live on the same manifold.")
    return form.exterior_derivative() + connection.as_algebra_valued_form().bracket_wedge(form)
