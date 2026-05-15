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
