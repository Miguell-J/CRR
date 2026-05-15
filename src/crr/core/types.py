"""Shared type aliases for CRR."""

from collections.abc import Sequence
from typing import TypeAlias

import sympy as sp

Coordinate: TypeAlias = sp.Symbol
Expression: TypeAlias = sp.Expr
MatrixLike: TypeAlias = Sequence[Sequence[object]]
