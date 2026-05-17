"""Lie-algebra-valued differential forms."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from crr.forms import DifferentialForm


@dataclass(frozen=True)
class AlgebraValuedForm:
    """A differential form with coefficients in a Lie algebra.

    Components are stored as ``A^a`` forms in ``A = sum_a A^a e_a``.
    """

    algebra: "LieAlgebra"
    components: tuple[DifferentialForm, ...]

    def __init__(self, algebra: "LieAlgebra", components: list[DifferentialForm] | tuple[DifferentialForm, ...]) -> None:
        if len(components) != algebra.dimension:
            raise ValueError("One differential-form component is required for each Lie algebra basis element.")
        if not components:
            raise ValueError("At least one component is required.")
        manifold = components[0].manifold
        degree = components[0].degree
        for component in components:
            if component.manifold != manifold or component.degree != degree:
                raise ValueError("All algebra-valued form components must have the same manifold and degree.")
        object.__setattr__(self, "algebra", algebra)
        object.__setattr__(self, "components", tuple(components))

    @property
    def degree(self) -> int:
        """Form degree."""

        return self.components[0].degree

    @property
    def manifold(self):
        """Underlying manifold."""

        return self.components[0].manifold

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"AlgebraValuedForm(algebra={self.algebra.name!r}, degree={self.degree})"

    def __add__(self, other: "AlgebraValuedForm") -> "AlgebraValuedForm":
        """Add algebra-valued forms."""

        self._validate_compatible(other)
        return AlgebraValuedForm(self.algebra, [a + b for a, b in zip(self.components, other.components, strict=True)])

    def __sub__(self, other: "AlgebraValuedForm") -> "AlgebraValuedForm":
        """Subtract algebra-valued forms."""

        return self + (-other)

    def __neg__(self) -> "AlgebraValuedForm":
        """Negate this algebra-valued form."""

        return AlgebraValuedForm(self.algebra, [-component for component in self.components])

    def __mul__(self, scalar) -> "AlgebraValuedForm":
        """Multiply by a scalar."""

        value = sp.sympify(scalar)
        return AlgebraValuedForm(self.algebra, [value * component for component in self.components])

    def __rmul__(self, scalar) -> "AlgebraValuedForm":
        """Multiply by a scalar."""

        return self * scalar

    @classmethod
    def zero(cls, algebra: "LieAlgebra", manifold, degree: int) -> "AlgebraValuedForm":
        """Return the zero algebra-valued form."""

        return cls(algebra, [DifferentialForm.zero(manifold, degree) for _ in range(algebra.dimension)])

    def exterior_derivative(self) -> "AlgebraValuedForm":
        """Return the componentwise exterior derivative."""

        return AlgebraValuedForm(self.algebra, [component.exterior_derivative() for component in self.components])

    def wedge(self, other: "AlgebraValuedForm"):
        """Plain wedge is intentionally not defined for Lie-algebra-valued forms."""

        raise NotImplementedError("Use bracket_wedge for Lie-bracket-valued wedge products.")

    def bracket_wedge(self, other: "AlgebraValuedForm") -> "AlgebraValuedForm":
        """Return ``[A wedge B]^k = c^k_ij A^i wedge B^j`` without a factor of 1/2."""

        self._validate_same_algebra(other)
        if self.manifold != other.manifold:
            raise ValueError("Algebra-valued forms must live on the same manifold.")
        out_degree = self.degree + other.degree
        if out_degree > self.manifold.dimension:
            return AlgebraValuedForm.zero(self.algebra, self.manifold, self.manifold.dimension)

        components = [DifferentialForm.zero(self.manifold, out_degree) for _ in range(self.algebra.dimension)]
        for k in range(self.algebra.dimension):
            value = DifferentialForm.zero(self.manifold, out_degree)
            for i in range(self.algebra.dimension):
                for j in range(self.algebra.dimension):
                    constant = self.algebra.structure_constant(k, i, j)
                    if constant == 0:
                        continue
                    value = value + constant * self.components[i].wedge(other.components[j])
            components[k] = value
        return AlgebraValuedForm(self.algebra, components)

    def simplify(self) -> "AlgebraValuedForm":
        """Return a componentwise simplified algebra-valued form."""

        return AlgebraValuedForm(self.algebra, [component.simplify() for component in self.components])

    def equals(self, other: "AlgebraValuedForm", simplify: bool = True) -> bool:
        """Return whether two algebra-valued forms are equal."""

        if self.algebra != other.algebra or self.manifold != other.manifold or self.degree != other.degree:
            return False
        return all(a.equals(b, simplify=simplify) for a, b in zip(self.components, other.components, strict=True))

    def to_latex(self) -> str:
        """Return a LaTeX expression."""

        terms = []
        for basis_name, component in zip(self.algebra.basis_names, self.components, strict=True):
            if component.nonzero_components():
                terms.append(rf"\left({component.to_latex()}\right){basis_name}")
        return " + ".join(terms) if terms else "0"

    def to_markdown(self) -> str:
        """Return a Markdown table of algebra-valued form components."""

        from crr.display.pretty import markdown_table

        rows = [(basis_name, str(component.nonzero_components())) for basis_name, component in zip(self.algebra.basis_names, self.components, strict=True)]
        return markdown_table(rows, headers=("Basis", "Form component"))

    def display(self) -> str:
        """Display or return this algebra-valued form as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def _validate_same_algebra(self, other: "AlgebraValuedForm") -> None:
        if self.algebra != other.algebra:
            raise ValueError("Algebra-valued forms must use the same Lie algebra.")

    def _validate_compatible(self, other: "AlgebraValuedForm") -> None:
        self._validate_same_algebra(other)
        if self.manifold != other.manifold or self.degree != other.degree:
            raise ValueError("Algebra-valued forms must have the same manifold and degree.")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.lie.algebra import LieAlgebra
