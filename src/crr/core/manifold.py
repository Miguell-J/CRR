"""Manifold definitions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import sympy as sp

from crr.core.coordinates import CoordinateChart


@dataclass(frozen=True)
class Manifold:
    """A smooth manifold represented by one coordinate chart."""

    name: str
    dimension: int
    coordinates: tuple[sp.Symbol, ...]

    def __init__(
        self,
        name: str,
        dimension: int,
        coordinates: Sequence[sp.Symbol] | CoordinateChart,
    ) -> None:
        chart = coordinates if isinstance(coordinates, CoordinateChart) else CoordinateChart(coordinates)
        if dimension != chart.dimension:
            raise ValueError("Manifold dimension must match the coordinate chart dimension.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "coordinates", chart.coordinates)

    @property
    def chart(self) -> CoordinateChart:
        """Return the default coordinate chart."""

        return CoordinateChart(self.coordinates, name=self.name)

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        coords = ", ".join(str(coord) for coord in self.coordinates)
        return f"Manifold(name={self.name!r}, dimension={self.dimension}, coordinates=({coords}))"

    def __str__(self) -> str:
        """Return a concise human-readable representation."""

        coords = ", ".join(str(coord) for coord in self.coordinates)
        return f"{self.name} ({self.dimension}D; {coords})"

    def to_latex(self) -> str:
        """Return a compact LaTeX representation."""

        coords = ", ".join(sp.latex(coord) for coord in self.coordinates)
        return rf"{self.name}\;({self.dimension}\mathrm{{D}};\ {coords})"

    def to_markdown(self) -> str:
        """Return a Markdown summary."""

        rows = [
            ("Name", self.name),
            ("Dimension", str(self.dimension)),
            ("Coordinates", ", ".join(str(coord) for coord in self.coordinates)),
        ]
        from crr.display.pretty import markdown_table

        return markdown_table(rows, headers=("Field", "Value"))

    def display(self) -> str:
        """Display or return a LaTeX summary."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())
