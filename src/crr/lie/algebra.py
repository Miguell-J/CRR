"""Finite-dimensional symbolic Lie algebras."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class LieAlgebra:
    """A finite-dimensional Lie algebra given by structure constants.

    The convention is ``[e_i, e_j] = sum_k c^k_{ij} e_k``.
    """

    name: str
    dimension: int
    basis_names: tuple[str, ...]
    structure_constants: dict[tuple[int, int, int], sp.Expr]

    def __init__(
        self,
        name: str,
        dimension: int,
        basis_names: list[str] | tuple[str, ...],
        structure_constants: Any,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Lie algebra dimension must be positive.")
        if len(basis_names) != dimension:
            raise ValueError("basis_names must have one entry per dimension.")

        constants = _normalize_structure_constants(structure_constants, dimension)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "basis_names", tuple(str(name) for name in basis_names))
        object.__setattr__(self, "structure_constants", constants)

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"LieAlgebra(name={self.name!r}, dimension={self.dimension})"

    def __str__(self) -> str:
        """Return a concise human-readable representation."""

        return f"{self.name} Lie algebra, dim={self.dimension}"

    def structure_constant(self, k: int, i: int, j: int) -> sp.Expr:
        """Return ``c^k_{ij}``."""

        self._validate_index(k)
        self._validate_index(i)
        self._validate_index(j)
        return self.structure_constants.get((k, i, j), sp.S.Zero)

    def element(self, components: list[Any] | tuple[Any, ...], name: str | None = None) -> "LieAlgebraElement":
        """Return an element of this algebra."""

        from crr.lie.element import LieAlgebraElement

        return LieAlgebraElement(self, components, name=name)

    def basis_element(self, i: int) -> "LieAlgebraElement":
        """Return the basis element ``e_i``."""

        self._validate_index(i)
        components = [sp.S.Zero] * self.dimension
        components[i] = sp.S.One
        return self.element(components, name=self.basis_names[i])

    def bracket_basis(self, i: int, j: int) -> "LieAlgebraElement":
        """Return ``[e_i, e_j]``."""

        return self.element([self.structure_constant(k, i, j) for k in range(self.dimension)])

    def bracket(self, x: "LieAlgebraElement", y: "LieAlgebraElement") -> "LieAlgebraElement":
        """Return the Lie bracket of two elements."""

        self._validate_element(x)
        self._validate_element(y)
        components = []
        for k in range(self.dimension):
            value = sp.S.Zero
            for i in range(self.dimension):
                for j in range(self.dimension):
                    value += self.structure_constant(k, i, j) * x.components[i] * y.components[j]
            components.append(value)
        return self.element(components)

    def adjoint_matrix(self, i: int) -> sp.Matrix:
        """Return the matrix of ``ad_{e_i}``, with columns ``[e_i,e_j]``."""

        self._validate_index(i)
        return sp.Matrix(
            self.dimension,
            self.dimension,
            lambda row, col: self.structure_constant(row, i, col),
        )

    def killing_form_matrix(self, simplify: bool = False) -> sp.Matrix:
        """Return the Killing form matrix ``B_ij = Tr(ad_i ad_j)``."""

        matrices = [self.adjoint_matrix(i) for i in range(self.dimension)]
        form = sp.Matrix(
            self.dimension,
            self.dimension,
            lambda i, j: sp.trace(matrices[i] * matrices[j]),
        )
        return form.applyfunc(sp.simplify) if simplify else form

    def killing_form(self, simplify: bool = False) -> sp.Matrix:
        """Alias for :meth:`killing_form_matrix`."""

        return self.killing_form_matrix(simplify=simplify)

    def is_abelian(self, simplify: bool = True) -> bool:
        """Return whether all brackets vanish."""

        for value in self.structure_constants.values():
            checked = sp.simplify(value) if simplify else value
            if checked != 0:
                return False
        return True

    def check_antisymmetry(self, simplify: bool = True) -> bool:
        """Verify ``c^k_{ij} = -c^k_{ji}``."""

        for k in range(self.dimension):
            for i in range(self.dimension):
                for j in range(self.dimension):
                    value = self.structure_constant(k, i, j) + self.structure_constant(k, j, i)
                    if simplify:
                        value = sp.simplify(value)
                    if value != 0:
                        return False
        return True

    def check_jacobi(self, simplify: bool = True) -> bool:
        """Verify the Jacobi identity in structure constants."""

        for l in range(self.dimension):
            for i in range(self.dimension):
                for j in range(self.dimension):
                    for k in range(self.dimension):
                        value = sp.S.Zero
                        for m in range(self.dimension):
                            value += self.structure_constant(m, i, j) * self.structure_constant(l, m, k)
                            value += self.structure_constant(m, j, k) * self.structure_constant(l, m, i)
                            value += self.structure_constant(m, k, i) * self.structure_constant(l, m, j)
                        if simplify:
                            value = sp.simplify(value)
                        if value != 0:
                            return False
        return True

    def nonzero_structure_constants(self, simplify: bool = True) -> dict[tuple[int, int, int], sp.Expr]:
        """Return nonzero structure constants."""

        constants = {}
        for index, value in self.structure_constants.items():
            checked = sp.simplify(value) if simplify else value
            if checked != 0:
                constants[index] = checked
        return constants

    def display_structure_constants(self) -> str:
        """Return a readable structure-constant table."""

        return self.to_markdown_table()

    def to_markdown_table(self) -> str:
        """Return nonzero structure constants as a Markdown table."""

        rows = ["| Bracket | Value |", "| --- | --- |"]
        for (k, i, j), value in sorted(self.nonzero_structure_constants().items()):
            rows.append(f"| [{self.basis_names[i]}, {self.basis_names[j]}] component {self.basis_names[k]} | {value} |")
        if len(rows) == 2:
            rows.append("| all | 0 |")
        return "\n".join(rows)

    def to_markdown(self) -> str:
        """Alias for :meth:`to_markdown_table`."""

        return self.to_markdown_table()

    def to_latex(self) -> str:
        """Return nonzero brackets in LaTeX."""

        terms = []
        for i in range(self.dimension):
            for j in range(i + 1, self.dimension):
                bracket = self.bracket_basis(i, j).simplify()
                if not bracket.equals(self.element([0] * self.dimension)):
                    terms.append(rf"[{self.basis_names[i]}, {self.basis_names[j]}] = {bracket.to_latex()}")
        return r",\quad ".join(terms) if terms else "0"

    def display(self) -> str:
        """Display or return the nonzero brackets as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def _validate_index(self, i: int) -> None:
        if i < 0 or i >= self.dimension:
            raise IndexError("Lie algebra basis index out of range.")

    def _validate_element(self, element: "LieAlgebraElement") -> None:
        if element.algebra != self:
            raise ValueError("Lie algebra elements must belong to the same algebra.")


def _normalize_structure_constants(structure_constants: Any, dimension: int) -> dict[tuple[int, int, int], sp.Expr]:
    constants: dict[tuple[int, int, int], sp.Expr] = {}
    if isinstance(structure_constants, dict):
        items = structure_constants.items()
        for key, value in items:
            if len(key) != 3:
                raise ValueError("Structure constant dictionary keys must be (k, i, j).")
            k, i, j = key
            _check_indices((k, i, j), dimension)
            sym_value = sp.sympify(value)
            if sym_value != 0:
                constants[(int(k), int(i), int(j))] = sym_value
        return constants

    for k in range(dimension):
        for i in range(dimension):
            for j in range(dimension):
                value = sp.sympify(structure_constants[k][i][j])
                if value != 0:
                    constants[(k, i, j)] = value
    return constants


def _check_indices(indices: tuple[int, int, int], dimension: int) -> None:
    if any(index < 0 or index >= dimension for index in indices):
        raise IndexError("Structure constant index out of range.")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.lie.element import LieAlgebraElement
