"""Display helpers for notebooks and plain terminals."""

from crr.display.latex import latex_component_equations, latex_matrix
from crr.display.pretty import (
    display_object,
    display_latex,
    display_markdown,
    format_nonzero_components,
    in_ipython,
    markdown_table,
    object_to_latex,
    object_to_markdown,
)

__all__ = [
    "in_ipython",
    "display_object",
    "object_to_latex",
    "object_to_markdown",
    "display_latex",
    "display_markdown",
    "format_nonzero_components",
    "markdown_table",
    "latex_matrix",
    "latex_component_equations",
]
