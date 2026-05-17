"""Pretty printing and notebook display utilities."""

from __future__ import annotations

from typing import Any

import sympy as sp


def in_ipython() -> bool:
    """Return whether an IPython shell appears to be active."""

    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def display_latex(latex: str) -> str:
    """Display LaTeX in IPython when available, otherwise return the string."""

    if in_ipython():
        try:
            from IPython.display import Math, display

            display(Math(latex))
        except ImportError:
            pass
    return latex


def display_markdown(markdown: str) -> str:
    """Display Markdown in IPython when available, otherwise return the string."""

    if in_ipython():
        try:
            from IPython.display import Markdown, display

            display(Markdown(markdown))
        except ImportError:
            pass
    return markdown


def object_to_latex(obj: Any) -> str:
    """Return an object's LaTeX representation, falling back to ``str(obj)``."""

    if hasattr(obj, "to_latex"):
        return str(obj.to_latex())
    return str(obj)


def object_to_markdown(obj: Any) -> str:
    """Return an object's Markdown representation, falling back to ``str(obj)``."""

    if hasattr(obj, "to_markdown"):
        return str(obj.to_markdown())
    if hasattr(obj, "to_markdown_table"):
        return str(obj.to_markdown_table())
    return str(obj)


def display_object(obj: Any, prefer: str = "latex") -> str:
    """Display an object in IPython when available, otherwise return a string.

    ``prefer`` may be ``"latex"`` or ``"markdown"``. The function does not
    require IPython unless an IPython shell is active.
    """

    if prefer == "markdown":
        return display_markdown(object_to_markdown(obj))
    if prefer != "latex":
        raise ValueError('prefer must be "latex" or "markdown".')
    return display_latex(object_to_latex(obj))


def markdown_table(rows: list[tuple[str, str]], headers: tuple[str, str] = ("Component", "Value")) -> str:
    """Return a simple two-column Markdown table."""

    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | --- |"]
    lines.extend(f"| {left} | {right} |" for left, right in rows)
    return "\n".join(lines)


def format_nonzero_components(obj: Any, basis_names: list[str] | None = None, latex: bool = True) -> str:
    """Format nonzero tensor or differential-form components."""

    if hasattr(obj, "degree") and hasattr(obj, "components") and hasattr(obj, "manifold"):
        return _format_form_components(obj, basis_names=basis_names, latex=latex)
    if hasattr(obj, "nonzero_components"):
        return _format_tensor_components(obj, latex=latex)
    raise TypeError("Object does not expose recognizable components.")


def _format_tensor_components(tensor: Any, latex: bool = True) -> str:
    components = tensor.nonzero_components()
    if not components:
        return "0"
    label = _tensor_label(tensor)
    lines = []
    for index, value in components.items():
        left = _tensor_component_label(label, index, getattr(tensor, "index_signature", None), latex=latex)
        right = sp.latex(value) if latex else str(value)
        lines.append(rf"{left} = {right}")
    return (r"\\ " if latex else "\n").join(lines)


def _format_form_components(form: Any, basis_names: list[str] | None = None, latex: bool = True) -> str:
    if not form.components:
        return "0"
    coords = form.manifold.coordinates
    names = basis_names or [str(coord) for coord in coords]
    terms = []
    for index, value in form.components.items():
        coefficient = sp.latex(value) if latex else str(value)
        if index == ():
            terms.append(coefficient)
            continue
        if latex:
            basis = r"\wedge ".join(rf"d{sp.latex(coords[i])}" for i in index)
            terms.append(rf"{coefficient}\,{basis}")
        else:
            basis = " ∧ ".join(f"d{names[i]}" for i in index)
            terms.append(f"{coefficient} {basis}")
    return " + ".join(terms)


def _tensor_label(tensor: Any) -> str:
    name = getattr(tensor, "name", None)
    if name in {r"\Gamma", "Gamma"}:
        return r"\Gamma"
    if name == "R" and getattr(tensor, "rank", None) == 4:
        return "R"
    if name == "R":
        return "R"
    if name == "G":
        return "G"
    return name or "T"


def _tensor_component_label(label: str, index: tuple[int, ...], signature: str | None, latex: bool = True) -> str:
    if label == r"\Gamma" and len(index) == 3:
        return rf"\Gamma^{index[0]}_{{{index[1]}{index[2]}}}" if latex else f"Gamma^{index[0]}_{index[1]}{index[2]}"
    if label == "R" and len(index) == 4:
        return rf"R^{index[0]}_{{{index[1]}{index[2]}{index[3]}}}" if latex else f"R^{index[0]}_{index[1]}{index[2]}{index[3]}"
    if label in {"R", "G"} and len(index) == 2:
        return rf"{label}_{{{index[0]}{index[1]}}}" if latex else f"{label}_{index[0]}{index[1]}"
    suffix = "".join(str(i) for i in index)
    return rf"{label}_{{{suffix}}}" if latex else f"{label}_{suffix}"
