"""Basic local gauge transformations."""

from __future__ import annotations

import sympy as sp

from crr.forms import DifferentialForm
from crr.gauge.connection import GaugePotential
from crr.lie import AlgebraValuedForm


def abelian_gauge_transform(connection: GaugePotential, chi) -> GaugePotential:
    """Return the abelian transform ``A -> A + d chi``."""

    if not connection.algebra.is_abelian():
        raise ValueError("abelian_gauge_transform expects an abelian connection.")
    if connection.algebra.dimension != 1:
        raise ValueError("abelian_gauge_transform expects a one-dimensional abelian connection.")

    scalar = chi if isinstance(chi, DifferentialForm) else DifferentialForm.scalar(connection.manifold, sp.sympify(chi))
    if scalar.degree != 0 or scalar.manifold != connection.manifold:
        raise ValueError("chi must be a scalar expression or a 0-form on the connection manifold.")
    return GaugePotential(connection.algebra, [connection.components[0] + scalar.exterior_derivative()], name=connection.name)


def infinitesimal_gauge_transform(connection: GaugePotential, epsilon: AlgebraValuedForm) -> GaugePotential:
    """Return the infinitesimal non-abelian transform ``A -> A + d_A epsilon``.

    Finite non-abelian gauge transformations are not implemented.
    """

    if not isinstance(epsilon, AlgebraValuedForm):
        raise TypeError("epsilon must be an AlgebraValuedForm.")
    if epsilon.degree != 0:
        raise ValueError("epsilon must be an algebra-valued 0-form.")
    delta = connection.covariant_exterior_derivative(epsilon)
    return GaugePotential(connection.algebra, [a + da for a, da in zip(connection.components, delta.components, strict=True)], name=connection.name)
