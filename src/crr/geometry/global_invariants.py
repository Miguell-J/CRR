"""Global invariant helpers for coordinate-domain integrations."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class GaussBonnetResult:
    """Result of a Gauss-Bonnet comparison."""

    curvature_integral: sp.Expr
    expected: sp.Expr | None
    difference: sp.Expr | None
    passed: bool | None

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return (
            "GaussBonnetResult("
            f"curvature_integral={self.curvature_integral}, "
            f"expected={self.expected}, difference={self.difference}, passed={self.passed})"
        )

    def __str__(self) -> str:
        """Return a concise human-readable representation."""

        return f"Gauss-Bonnet check: integral={self.curvature_integral}, expected={self.expected}, passed={self.passed}"

    def to_latex(self) -> str:
        """Return a LaTeX summary."""

        pieces = [rf"\int K\,dA = {sp.latex(self.curvature_integral)}"]
        if self.expected is not None:
            pieces.append(rf"2\pi\chi = {sp.latex(self.expected)}")
        if self.passed is not None:
            pieces.append(rf"\mathrm{{passed}} = {self.passed}")
        return r",\quad ".join(pieces)

    def to_markdown(self) -> str:
        """Return a Markdown summary."""

        from crr.display.pretty import markdown_table

        rows = [
            ("Curvature integral", str(self.curvature_integral)),
            ("Expected", str(self.expected)),
            ("Difference", str(self.difference)),
            ("Passed", str(self.passed)),
        ]
        return markdown_table(rows, headers=("Field", "Value"))

    def display(self) -> str:
        """Display or return a LaTeX summary."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())


def gauss_bonnet_check(
    surface: "ParametrizedMap",
    ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
    euler_characteristic: int | None = None,
    simplify: bool = True,
) -> GaussBonnetResult:
    """Integrate Gaussian curvature and optionally compare to 2*pi*chi."""

    curvature_integral = surface.integrate_gaussian_curvature(ranges, intrinsic=True, simplify=simplify)
    if euler_characteristic is None:
        return GaussBonnetResult(curvature_integral, None, None, None)

    expected = 2 * sp.pi * euler_characteristic
    difference = curvature_integral - expected
    if simplify:
        difference = sp.simplify(difference)
    return GaussBonnetResult(
        curvature_integral=curvature_integral,
        expected=expected,
        difference=difference,
        passed=difference == 0,
    )


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.parametrized import ParametrizedMap
