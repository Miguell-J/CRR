"""Elements of finite-dimensional Lie algebras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class LieAlgebraElement:
    """A coordinate vector in a fixed Lie algebra basis."""

    algebra: "LieAlgebra"
    components: tuple[sp.Expr, ...]
    name: str | None = None

    def __init__(self, algebra: "LieAlgebra", components: list[Any] | tuple[Any, ...], name: str | None = None) -> None:
        if len(components) != algebra.dimension:
            raise ValueError("Element components must match the Lie algebra dimension.")
        object.__setattr__(self, "algebra", algebra)
        object.__setattr__(self, "components", tuple(sp.sympify(component) for component in components))
        object.__setattr__(self, "name", name)

    def __add__(self, other: "LieAlgebraElement") -> "LieAlgebraElement":
        """Add two elements."""

        self._validate_compatible(other)
        return self.algebra.element([a + b for a, b in zip(self.components, other.components, strict=True)])

    def __sub__(self, other: "LieAlgebraElement") -> "LieAlgebraElement":
        """Subtract two elements."""

        return self + (-other)

    def __neg__(self) -> "LieAlgebraElement":
        """Negate this element."""

        return self.algebra.element([-component for component in self.components])

    def __mul__(self, scalar: Any) -> "LieAlgebraElement":
        """Multiply by a scalar."""

        value = sp.sympify(scalar)
        return self.algebra.element([value * component for component in self.components])

    def __rmul__(self, scalar: Any) -> "LieAlgebraElement":
        """Multiply by a scalar."""

        return self * scalar

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"LieAlgebraElement(algebra={self.algebra.name!r}, components={self.components})"

    def __str__(self) -> str:
        """Return a human-readable linear combination."""

        if self.name is not None:
            return self.name
        terms = []
        for component, basis_name in zip(self.components, self.algebra.basis_names, strict=True):
            if component != 0:
                terms.append(f"({component}) {basis_name}")
        return " + ".join(terms) if terms else "0"

    def bracket(self, other: "LieAlgebraElement") -> "LieAlgebraElement":
        """Return the Lie bracket with another element."""

        return self.algebra.bracket(self, other)

    def simplify(self) -> "LieAlgebraElement":
        """Return a simplified element."""

        return self.algebra.element([sp.simplify(component) for component in self.components], name=self.name)

    def equals(self, other: "LieAlgebraElement", simplify: bool = True) -> bool:
        """Return whether two elements are symbolically equal."""

        if self.algebra != other.algebra:
            return False
        for left, right in zip(self.components, other.components, strict=True):
            difference = left - right
            if simplify:
                difference = sp.simplify(difference)
            if difference != 0:
                return False
        return True

    def to_vector(self) -> sp.Matrix:
        """Return this element as a SymPy column vector."""

        return sp.Matrix(self.components)

    def to_latex(self) -> str:
        """Return a LaTeX linear combination."""

        terms = []
        for component, basis_name in zip(self.components, self.algebra.basis_names, strict=True):
            if sp.simplify(component) != 0:
                terms.append(rf"{sp.latex(component)} {basis_name}")
        return " + ".join(terms) if terms else "0"

    def to_markdown(self) -> str:
        """Return a Markdown table of components."""

        from crr.display.pretty import markdown_table

        rows = [(basis_name, str(component)) for component, basis_name in zip(self.components, self.algebra.basis_names, strict=True)]
        return markdown_table(rows, headers=("Basis", "Coefficient"))

    def display(self) -> str:
        """Display or return this element as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def _validate_compatible(self, other: "LieAlgebraElement") -> None:
        if self.algebra != other.algebra:
            raise ValueError("Lie algebra elements must belong to the same algebra.")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.lie.algebra import LieAlgebra
