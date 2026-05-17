"""LaTeX formatting helpers."""

from __future__ import annotations

from typing import Any

import sympy as sp


def latex_matrix(matrix: Any) -> str:
    """Return LaTeX for a matrix-like object."""

    return sp.latex(matrix)


def latex_component_equations(components: dict[tuple[int, ...], Any], label: str = "T") -> str:
    """Return LaTeX component equations."""

    if not components:
        return "0"
    terms = []
    for index, value in components.items():
        suffix = "".join(str(i) for i in index)
        terms.append(rf"{label}_{{{suffix}}} = {sp.latex(value)}")
    return r"\\ ".join(terms)
