"""Coordinate chart definitions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import sympy as sp


@dataclass(frozen=True)
class CoordinateChart:
    """A coordinate chart on a finite-dimensional manifold."""

    coordinates: tuple[sp.Symbol, ...]
    name: str | None = None

    def __init__(self, coordinates: Sequence[sp.Symbol], name: str | None = None) -> None:
        object.__setattr__(self, "coordinates", tuple(coordinates))
        object.__setattr__(self, "name", name)
        if not self.coordinates:
            raise ValueError("A coordinate chart must contain at least one coordinate.")
        if not all(isinstance(coord, sp.Symbol) for coord in self.coordinates):
            raise TypeError("Coordinates must be SymPy symbols.")

    @property
    def dimension(self) -> int:
        """Dimension of the coordinate chart."""

        return len(self.coordinates)
