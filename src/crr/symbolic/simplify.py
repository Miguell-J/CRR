"""Controlled simplification helpers."""

from __future__ import annotations

from itertools import product
from typing import Any

import sympy as sp


def maybe_simplify(expression: Any, simplify: bool = False) -> Any:
    """Simplify an expression only when requested."""

    return sp.simplify(expression) if simplify else expression


def simplify_array(array: sp.MutableDenseNDimArray) -> sp.MutableDenseNDimArray:
    """Return a new dense array with each component simplified."""

    result = sp.MutableDenseNDimArray.zeros(*array.shape)
    for index in product(*(range(size) for size in array.shape)):
        result[index] = sp.simplify(array[index])
    return result
