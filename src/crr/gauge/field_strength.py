"""Gauge curvature and field-strength helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from crr.forms import DifferentialForm
from crr.lie import AlgebraValuedForm, LieAlgebra


@dataclass(frozen=True)
class GaugeCurvature:
    """A Lie-algebra-valued curvature 2-form.

    Components are stored as ``F^a`` in ``F = sum_a F^a e_a``.
    """

    algebra: LieAlgebra
    components: tuple[DifferentialForm, ...]
    connection: "GaugePotential | None" = None
    name: str = "F"

    def __init__(
        self,
        algebra: LieAlgebra,
        components: list[DifferentialForm] | tuple[DifferentialForm, ...],
        connection: "GaugePotential | None" = None,
        name: str = "F",
    ) -> None:
        if len(components) != algebra.dimension:
            raise ValueError("One curvature component is required for each Lie algebra basis element.")
        if not components:
            raise ValueError("At least one curvature component is required.")
        for component in components:
            if not isinstance(component, DifferentialForm):
                raise TypeError("Gauge curvature components must be DifferentialForm instances.")
        manifold = components[0].manifold
        for component in components:
            if component.degree != 2:
                raise ValueError("Gauge curvature components must be 2-forms.")
            if component.manifold != manifold:
                raise ValueError("All curvature components must live on the same manifold.")
        if connection is not None:
            if connection.algebra != algebra or connection.manifold != manifold:
                raise ValueError("Curvature connection must use the same algebra and manifold.")

        object.__setattr__(self, "algebra", algebra)
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "connection", connection)
        object.__setattr__(self, "name", name)

    @property
    def degree(self) -> int:
        """Form degree, always 2."""

        return 2

    @property
    def manifold(self):
        """Underlying manifold."""

        return self.components[0].manifold

    def as_algebra_valued_form(self) -> AlgebraValuedForm:
        """Return this curvature as a plain algebra-valued form."""

        return AlgebraValuedForm(self.algebra, self.components)

    def bianchi(self, connection: "GaugePotential | None" = None, simplify: bool = True) -> AlgebraValuedForm:
        """Return ``d_A F = dF + [A wedge F]`` for the given connection."""

        from crr.gauge.utils import covariant_exterior_derivative

        active_connection = connection or self.connection
        if active_connection is None:
            raise ValueError("A connection is required to compute the Bianchi expression.")
        result = covariant_exterior_derivative(active_connection, self)
        return result.simplify() if simplify else result

    def yang_mills_current(self, metric, simplify: bool = False) -> AlgebraValuedForm:
        """Return a simple source-free Yang-Mills current ``d_A *F``."""

        from crr.gauge.utils import covariant_exterior_derivative

        if self.connection is None:
            raise ValueError("A connection is required to compute d_A *F.")
        starred = AlgebraValuedForm(
            self.algebra,
            [component.hodge_star(metric, simplify=simplify) for component in self.components],
        )
        result = covariant_exterior_derivative(self.connection, starred)
        return result.simplify() if simplify else result

    def action_density(self, metric, inner_product=None, simplify: bool = False) -> DifferentialForm:
        """Return ``sum_ab <e_a,e_b> F^a wedge *F^b`` as a top-degree form."""

        return yang_mills_action_density(self, metric, inner_product=inner_product, simplify=simplify)

    def simplify(self) -> "GaugeCurvature":
        """Return a componentwise simplified curvature."""

        return GaugeCurvature(
            self.algebra,
            [component.simplify() for component in self.components],
            connection=self.connection,
            name=self.name,
        )

    def equals(self, other: object, simplify: bool = True) -> bool:
        """Return whether two curvature forms are symbolically equal."""

        other_form = _as_algebra_valued_form(other)
        if other_form is None:
            return False
        return self.as_algebra_valued_form().equals(other_form, simplify=simplify)

    def to_latex(self) -> str:
        """Return a LaTeX expression."""

        return self.as_algebra_valued_form().to_latex()

    def to_markdown(self) -> str:
        """Return a Markdown table of curvature components."""

        return self.as_algebra_valued_form().to_markdown()

    def display(self) -> str:
        """Display or return this gauge curvature as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def __repr__(self) -> str:
        return f"GaugeCurvature(algebra={self.algebra.name!r}, degree=2, name={self.name!r})"

    def __str__(self) -> str:
        return f"{self.name}: {self.algebra.name}-valued 2-form on {self.manifold.name}"


def yang_mills_action_density(
    curvature: GaugeCurvature | AlgebraValuedForm,
    metric,
    inner_product=None,
    simplify: bool = False,
) -> DifferentialForm:
    """Return the local Yang-Mills action density ``<F, *F>``.

    The default Lie-algebra inner product is the identity matrix. This is a
    practical positive convention for compact examples such as the real
    ``su(2)`` preset; callers can pass an invariant form explicitly when a
    different normalization or signature is desired.
    """

    form = _require_algebra_valued_form(curvature)
    if form.degree != 2:
        raise ValueError("Yang-Mills action density expects a curvature 2-form.")
    if metric.manifold != form.manifold:
        raise ValueError("Metric must live on the same manifold as the curvature.")

    ip = identity_inner_product(form.algebra) if inner_product is None else sp.Matrix(inner_product)
    if ip.shape != (form.algebra.dimension, form.algebra.dimension):
        raise ValueError("inner_product must have shape algebra.dimension x algebra.dimension.")

    top = DifferentialForm.zero(form.manifold, form.manifold.dimension)
    stars = [component.hodge_star(metric, simplify=simplify) for component in form.components]
    for a, left in enumerate(form.components):
        for b, right_star in enumerate(stars):
            coefficient = ip[a, b]
            if coefficient == 0:
                continue
            top = top + coefficient * left.wedge(right_star)
    return top.simplify() if simplify else top


def maxwell_field_strength(connection) -> GaugeCurvature:
    """Return the abelian Maxwell field strength ``F = dA``."""

    if not connection.algebra.is_abelian():
        raise ValueError("maxwell_field_strength expects an abelian gauge potential.")
    return connection.curvature()


def maxwell_lagrangian_density(curvature: GaugeCurvature | AlgebraValuedForm, metric, simplify: bool = False) -> DifferentialForm:
    """Return the simple Maxwell density ``F wedge *F`` for a U(1) curvature."""

    return yang_mills_action_density(curvature, metric, inner_product=sp.Matrix([[1]]), simplify=simplify)


def identity_inner_product(algebra: LieAlgebra) -> sp.Matrix:
    """Return the identity Lie-algebra inner product matrix.

    This is CRR's default practical convention for compact examples such as
    the real ``su(2)`` preset. The Killing form can be indefinite or depend on
    normalization/sign conventions, so callers should pass an explicit
    ``inner_product`` when that structure is mathematically important.
    """

    return sp.eye(algebra.dimension)


def _require_algebra_valued_form(value: GaugeCurvature | AlgebraValuedForm) -> AlgebraValuedForm:
    form = _as_algebra_valued_form(value)
    if form is None:
        raise TypeError("Expected GaugeCurvature or AlgebraValuedForm.")
    return form


def _as_algebra_valued_form(value: object) -> AlgebraValuedForm | None:
    if isinstance(value, GaugeCurvature):
        return value.as_algebra_valued_form()
    if isinstance(value, AlgebraValuedForm):
        return value
    return None


if TYPE_CHECKING:
    from crr.gauge.connection import GaugePotential
