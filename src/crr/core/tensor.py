"""Tensor container utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class Tensor:
    """A simple dense tensor wrapper backed by SymPy arrays."""

    components: sp.MutableDenseNDimArray
    name: str | None = None
    index_signature: str | None = None

    def __init__(
        self,
        components: Any,
        name: str | None = None,
        index_signature: str | None = None,
    ) -> None:
        array = (
            components
            if isinstance(components, sp.MutableDenseNDimArray)
            else sp.MutableDenseNDimArray(components)
        )
        object.__setattr__(self, "components", array)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "index_signature", index_signature)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor component shape."""

        return tuple(self.components.shape)

    @property
    def rank(self) -> int:
        """Tensor rank."""

        return len(self.shape)

    def __getitem__(self, indices: int | tuple[int, ...]) -> sp.Expr:
        """Return a tensor component."""

        return self.components[indices]

    def __iter__(self):
        """Iterate over the first tensor axis."""

        return iter(self.components)

    def tolist(self) -> list[Any]:
        """Return components as nested Python lists."""

        return self.components.tolist()

    def simplify(self) -> "Tensor":
        """Return a tensor with each component simplified."""

        result = sp.MutableDenseNDimArray.zeros(*self.shape)
        for index in product(*(range(size) for size in self.shape)):
            result[index] = sp.simplify(self.components[index])
        return Tensor(result, name=self.name, index_signature=self.index_signature)

    def equals(self, other: "Tensor", simplify: bool = True) -> bool:
        """Return whether two tensors have symbolically equal components."""

        if self.shape != other.shape:
            return False
        for index in product(*(range(size) for size in self.shape)):
            difference = self.components[index] - other.components[index]
            if simplify:
                difference = sp.simplify(difference)
            if difference != 0:
                return False
        return True

    def raise_index(self, metric: "Metric", index: int = 0, simplify: bool = False) -> "Tensor":
        """Raise one covariant index using the inverse metric."""

        self._validate_index(index)
        n = metric.dimension
        self._validate_metric_dimension(n)
        g_inv = metric.inverse()
        result = sp.MutableDenseNDimArray.zeros(*self.shape)

        for output_index in product(*(range(size) for size in self.shape)):
            value = sp.S.Zero
            for contracted in range(n):
                source_index = list(output_index)
                source_index[index] = contracted
                value += g_inv[output_index[index], contracted] * self.components[tuple(source_index)]
            result[output_index] = sp.simplify(value) if simplify else value

        return Tensor(result, name=self.name, index_signature=_raise_signature(self.index_signature, index))

    def lower_index(self, metric: "Metric", index: int = 0, simplify: bool = False) -> "Tensor":
        """Lower one contravariant index using the metric."""

        self._validate_index(index)
        n = metric.dimension
        self._validate_metric_dimension(n)
        result = sp.MutableDenseNDimArray.zeros(*self.shape)

        for output_index in product(*(range(size) for size in self.shape)):
            value = sp.S.Zero
            for contracted in range(n):
                source_index = list(output_index)
                source_index[index] = contracted
                value += metric.components[output_index[index], contracted] * self.components[tuple(source_index)]
            result[output_index] = sp.simplify(value) if simplify else value

        return Tensor(result, name=self.name, index_signature=_lower_signature(self.index_signature, index))

    def contract(self, axis1: int, axis2: int, simplify: bool = False) -> "Tensor | sp.Expr":
        """Contract two tensor axes using the Kronecker delta."""

        self._validate_index(axis1)
        self._validate_index(axis2)
        if axis1 == axis2:
            raise ValueError("Cannot contract an axis with itself.")
        if self.shape[axis1] != self.shape[axis2]:
            raise ValueError("Contracted axes must have the same dimension.")

        first, second = sorted((axis1, axis2))
        output_shape = tuple(size for axis, size in enumerate(self.shape) if axis not in (first, second))
        output_indices = list(product(*(range(size) for size in output_shape))) if output_shape else [()]

        if output_shape:
            result = sp.MutableDenseNDimArray.zeros(*output_shape)
            for output_index in output_indices:
                value = self._contracted_value(first, second, output_index)
                result[output_index] = sp.simplify(value) if simplify else value
            return Tensor(result, name=self.name, index_signature=_remove_signature_axes(self.index_signature, first, second))

        value = self._contracted_value(first, second, ())
        return sp.simplify(value) if simplify else value

    def trace(self, metric: "Metric | None" = None, simplify: bool = False) -> sp.Expr:
        """Return a trace.

        For rank-2 tensors, a metric computes g^ij T_ij. Without a metric,
        the ordinary matrix trace T^i_i is returned.
        """

        if self.rank != 2:
            raise ValueError("trace currently supports rank-2 tensors.")
        if self.shape[0] != self.shape[1]:
            raise ValueError("trace requires a square rank-2 tensor.")

        value = sp.S.Zero
        if metric is None:
            for i in range(self.shape[0]):
                value += self.components[i, i]
        else:
            self._validate_metric_dimension(metric.dimension)
            g_inv = metric.inverse()
            for i in range(metric.dimension):
                for j in range(metric.dimension):
                    value += g_inv[i, j] * self.components[i, j]
        return sp.simplify(value) if simplify else value

    def nonzero_components(self, simplify: bool = True) -> dict[tuple[int, ...], sp.Expr]:
        """Return a mapping of index tuples to nonzero components."""

        result: dict[tuple[int, ...], sp.Expr] = {}
        for index in product(*(range(size) for size in self.shape)):
            value = self.components[index]
            test_value = sp.simplify(value) if simplify else value
            if test_value != 0:
                result[index] = test_value
        return result

    def to_latex(self, only_nonzero: bool = True) -> str:
        """Return a basic LaTeX representation of tensor components."""

        if only_nonzero:
            components = self.nonzero_components()
            if not components:
                return "0"
            terms = []
            label = self.name or "T"
            for index, value in components.items():
                suffix = "".join(str(i) for i in index)
                terms.append(rf"{label}_{{{suffix}}} = {sp.latex(value)}")
            return r",\quad ".join(terms)
        return sp.latex(self.components)

    def _validate_index(self, index: int) -> None:
        if index < 0 or index >= self.rank:
            raise IndexError("Tensor index axis out of range.")

    def _validate_metric_dimension(self, dimension: int) -> None:
        if any(size != dimension for size in self.shape):
            raise ValueError("All tensor axes must match the metric dimension.")

    def _contracted_value(self, axis1: int, axis2: int, output_index: tuple[int, ...]) -> sp.Expr:
        value = sp.S.Zero
        n = self.shape[axis1]
        for contracted in range(n):
            source_index: list[int] = []
            output_iter = iter(output_index)
            for axis in range(self.rank):
                if axis in (axis1, axis2):
                    source_index.append(contracted)
                else:
                    source_index.append(next(output_iter))
            value += self.components[tuple(source_index)]
        return value


def _signature_axes(index_signature: str | None) -> list[str]:
    if index_signature is None:
        return []
    return [char for char in index_signature if char in {"^", "_"}]


def _replace_signature_axis(index_signature: str | None, index: int, marker: str) -> str | None:
    axes = _signature_axes(index_signature)
    if not axes or index >= len(axes):
        return index_signature
    axes[index] = marker
    return "".join(axes)


def _raise_signature(index_signature: str | None, index: int) -> str | None:
    return _replace_signature_axis(index_signature, index, "^")


def _lower_signature(index_signature: str | None, index: int) -> str | None:
    return _replace_signature_axis(index_signature, index, "_")


def _remove_signature_axes(index_signature: str | None, axis1: int, axis2: int) -> str | None:
    axes = _signature_axes(index_signature)
    if not axes:
        return index_signature
    return "".join(marker for axis, marker in enumerate(axes) if axis not in (axis1, axis2))


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
