"""Common finite-dimensional Lie algebra presets."""

from __future__ import annotations

import sympy as sp

from crr.lie.algebra import LieAlgebra


def abelian_lie_algebra(dim: int, name: str | None = None) -> LieAlgebra:
    """Return the abelian Lie algebra of dimension ``dim``."""

    basis_names = [f"e{i + 1}" for i in range(dim)]
    return LieAlgebra(name or f"R^{dim}_abelian", dim, basis_names, {})


def so2_algebra() -> LieAlgebra:
    """Return ``so(2)``, a one-dimensional abelian Lie algebra."""

    return LieAlgebra("so(2)", 1, ["J"], {})


def so3_algebra() -> LieAlgebra:
    """Return ``so(3)`` with ``[e1,e2]=e3`` and cyclic permutations."""

    constants = _cyclic_so3_constants()
    return LieAlgebra("so(3)", 3, ["e1", "e2", "e3"], constants)


def su2_algebra(convention: str = "physics") -> LieAlgebra:
    """Return a real compact ``su(2)`` convention structurally equivalent to ``so(3)``.

    The ``physics`` and ``math`` labels currently share the same real basis
    convention ``[T1,T2]=T3`` without explicit matrix ``i`` factors.
    """

    if convention not in {"physics", "math"}:
        raise ValueError('convention must be "physics" or "math".')
    return LieAlgebra("su(2)", 3, ["T1", "T2", "T3"], _cyclic_so3_constants())


def sl2r_algebra() -> LieAlgebra:
    """Return ``sl(2,R)`` with basis ``H,E,F``."""

    constants = {
        (1, 0, 1): 2,
        (1, 1, 0): -2,
        (2, 0, 2): -2,
        (2, 2, 0): 2,
        (0, 1, 2): 1,
        (0, 2, 1): -1,
    }
    return LieAlgebra("sl(2,R)", 3, ["H", "E", "F"], constants)


def heisenberg_algebra() -> LieAlgebra:
    """Return the three-dimensional Heisenberg algebra ``[X,Y]=Z``."""

    constants = {
        (2, 0, 1): sp.S.One,
        (2, 1, 0): -sp.S.One,
    }
    return LieAlgebra("heisenberg", 3, ["X", "Y", "Z"], constants)


def _cyclic_so3_constants() -> dict[tuple[int, int, int], sp.Expr]:
    constants: dict[tuple[int, int, int], sp.Expr] = {}
    cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for i, j, k in cyclic:
        constants[(k, i, j)] = sp.S.One
        constants[(k, j, i)] = -sp.S.One
    return constants
