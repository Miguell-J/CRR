"""Utility functions for matrix Lie algebras."""

from __future__ import annotations

from typing import Sequence

import sympy as sp

from crr.lie.algebra import LieAlgebra


def matrix_commutator(A, B) -> sp.Matrix:
    """Return the matrix commutator ``A*B - B*A``."""

    left = sp.Matrix(A)
    right = sp.Matrix(B)
    if left.shape != right.shape:
        raise ValueError("Matrix commutator requires matrices with the same shape.")
    return left * right - right * left


def matrix_lie_algebra_from_basis(
    name: str,
    matrices: Sequence[sp.Matrix],
    basis_names: Sequence[str] | None = None,
) -> LieAlgebra:
    """Construct a Lie algebra by expressing matrix commutators in a basis."""

    if not matrices:
        raise ValueError("At least one basis matrix is required.")
    basis = [sp.Matrix(matrix) for matrix in matrices]
    shape = basis[0].shape
    if any(matrix.shape != shape for matrix in basis):
        raise ValueError("All basis matrices must have the same shape.")

    dimension = len(basis)
    names = list(basis_names) if basis_names is not None else [f"e{i + 1}" for i in range(dimension)]
    if len(names) != dimension:
        raise ValueError("basis_names must match the number of matrices.")

    basis_columns = sp.Matrix.hstack(*[sp.Matrix(matrix).reshape(shape[0] * shape[1], 1) for matrix in basis])
    if basis_columns.rank() < dimension:
        raise ValueError("Basis matrices must be linearly independent.")
    constants: dict[tuple[int, int, int], sp.Expr] = {}
    for i in range(dimension):
        for j in range(dimension):
            commutator_column = matrix_commutator(basis[i], basis[j]).reshape(shape[0] * shape[1], 1)
            solution = sp.linsolve((basis_columns, commutator_column))
            if not solution:
                raise ValueError("Commutator could not be expressed in the supplied basis.")
            solution_tuple = next(iter(solution))
            for k, value in enumerate(solution_tuple):
                value = sp.simplify(value)
                if value != 0:
                    constants[(k, i, j)] = value
    return LieAlgebra(name, dimension, names, constants)
