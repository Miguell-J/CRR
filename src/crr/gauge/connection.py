"""Gauge potentials as Lie-algebra-valued connection 1-forms."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from crr.forms import DifferentialForm
from crr.gauge.field_strength import GaugeCurvature
from crr.lie import AlgebraValuedForm, LieAlgebra


@dataclass(frozen=True)
class GaugePotential:
    """A local gauge potential / connection 1-form ``A in Omega^1(M, g)``.

    ``curvature(convention="half")`` computes ``F = dA + 1/2 [A wedge A]``.
    ``curvature(convention="matrix")`` computes ``F = dA + A wedge A`` for
    matrix-valued conventions where the bracket factor is already absorbed.
    """

    algebra: LieAlgebra
    components: tuple[DifferentialForm, ...]
    name: str = "A"

    def __init__(
        self,
        algebra: LieAlgebra,
        components: list[DifferentialForm] | tuple[DifferentialForm, ...],
        name: str = "A",
    ) -> None:
        if len(components) != algebra.dimension:
            raise ValueError("One connection component is required for each Lie algebra basis element.")
        if not components:
            raise ValueError("At least one connection component is required.")
        for component in components:
            if not isinstance(component, DifferentialForm):
                raise TypeError("Gauge potential components must be DifferentialForm instances.")
        manifold = components[0].manifold
        for component in components:
            if component.degree != 1:
                raise ValueError("Gauge potential components must be 1-forms.")
            if component.manifold != manifold:
                raise ValueError("All gauge potential components must live on the same manifold.")
        object.__setattr__(self, "algebra", algebra)
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "name", name)

    @property
    def degree(self) -> int:
        """Form degree, always 1."""

        return 1

    @property
    def manifold(self):
        """Underlying manifold."""

        return self.components[0].manifold

    def as_algebra_valued_form(self) -> AlgebraValuedForm:
        """Return this connection as a plain algebra-valued form."""

        return AlgebraValuedForm(self.algebra, self.components)

    def exterior_derivative(self) -> AlgebraValuedForm:
        """Return the componentwise exterior derivative ``dA``."""

        return self.as_algebra_valued_form().exterior_derivative()

    def bracket_wedge(self, other) -> AlgebraValuedForm:
        """Return ``[A wedge other]`` with no extra numerical factor."""

        return self.as_algebra_valued_form().bracket_wedge(_require_algebra_valued_form(other))

    def curvature(self, convention: str = "half") -> GaugeCurvature:
        """Return the curvature 2-form.

        ``"half"`` uses ``F = dA + 1/2 [A wedge A]``. ``"matrix"`` uses
        ``F = dA + [A wedge A]`` for callers using matrix-style conventions.
        """

        if convention not in {"half", "matrix"}:
            raise ValueError('convention must be "half" or "matrix".')
        dA = self.exterior_derivative()
        bracket = self.bracket_wedge(self)
        factor = sp.Rational(1, 2) if convention == "half" else sp.S.One
        result = dA + factor * bracket
        return GaugeCurvature(self.algebra, list(result.components), connection=self, name="F")

    def field_strength(self, convention: str = "half") -> GaugeCurvature:
        """Alias for :meth:`curvature`."""

        return self.curvature(convention=convention)

    def covariant_exterior_derivative(self, form) -> AlgebraValuedForm:
        """Return ``d_A form = d form + [A wedge form]``."""

        from crr.gauge.utils import covariant_exterior_derivative

        return covariant_exterior_derivative(self, form)

    def bianchi_identity(self, simplify: bool = True) -> AlgebraValuedForm:
        """Return the Bianchi expression ``d_A F``."""

        result = self.covariant_exterior_derivative(self.curvature())
        return result.simplify() if simplify else result

    def simplify(self) -> "GaugePotential":
        """Return a componentwise simplified connection."""

        return GaugePotential(self.algebra, [component.simplify() for component in self.components], name=self.name)

    def equals(self, other: object, simplify: bool = True) -> bool:
        """Return whether two gauge potentials are symbolically equal."""

        if not isinstance(other, GaugePotential):
            return False
        return self.as_algebra_valued_form().equals(other.as_algebra_valued_form(), simplify=simplify)

    def to_latex(self) -> str:
        """Return a LaTeX expression."""

        return self.as_algebra_valued_form().to_latex()

    def to_markdown(self) -> str:
        """Return a Markdown table of connection components."""

        return self.as_algebra_valued_form().to_markdown()

    def display(self) -> str:
        """Display or return this gauge potential as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def __repr__(self) -> str:
        return f"GaugePotential(algebra={self.algebra.name!r}, degree=1, name={self.name!r})"

    def __str__(self) -> str:
        return f"{self.name}: {self.algebra.name}-valued 1-form on {self.manifold.name}"


def _require_algebra_valued_form(value) -> AlgebraValuedForm:
    if isinstance(value, GaugePotential):
        return value.as_algebra_valued_form()
    if isinstance(value, GaugeCurvature):
        return value.as_algebra_valued_form()
    if isinstance(value, AlgebraValuedForm):
        return value
    raise TypeError("Expected GaugePotential, GaugeCurvature, or AlgebraValuedForm.")
