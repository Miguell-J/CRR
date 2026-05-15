"""LaTeX export helpers."""

from __future__ import annotations

from typing import Any

import sympy as sp


def to_latex(obj: Any) -> str:
    """Return a LaTeX string for a SymPy object or CRR tensor-like object."""

    if hasattr(obj, "to_latex"):
        return obj.to_latex()
    return sp.latex(obj)
