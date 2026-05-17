"""Coordinate-basis differential forms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import sympy as sp

from crr.core.manifold import Manifold


@dataclass(frozen=True)
class DifferentialForm:
    """A sparse coordinate-basis differential form."""

    manifold: Manifold
    degree: int
    components: dict[tuple[int, ...], sp.Expr]

    def __init__(self, manifold: Manifold, degree: int, components: dict[tuple[int, ...], Any] | None = None) -> None:
        if degree < 0 or degree > manifold.dimension:
            raise ValueError("Form degree must be between 0 and manifold dimension.")
        canonical: dict[tuple[int, ...], sp.Expr] = {}
        for raw_index, raw_value in (components or {}).items():
            index, sign = _canonicalize(raw_index, degree, manifold.dimension)
            if sign == 0:
                continue
            value = sign * sp.sympify(raw_value)
            canonical[index] = canonical.get(index, sp.S.Zero) + value

        cleaned = {
            index: value
            for index, value in canonical.items()
            if sp.simplify(value) != 0
        }
        object.__setattr__(self, "manifold", manifold)
        object.__setattr__(self, "degree", degree)
        object.__setattr__(self, "components", cleaned)

    @classmethod
    def zero(cls, manifold: Manifold, degree: int) -> "DifferentialForm":
        """Return the zero form of a given degree."""

        return cls(manifold, degree, {})

    @classmethod
    def scalar(cls, manifold: Manifold, expr: Any) -> "DifferentialForm":
        """Return a 0-form."""

        return cls(manifold, 0, {(): sp.sympify(expr)})

    @classmethod
    def one_form(cls, manifold: Manifold, components: dict[int | tuple[int], Any] | list[Any] | tuple[Any, ...]) -> "DifferentialForm":
        """Return a 1-form from a sequence or mapping of components."""

        if isinstance(components, dict):
            data = {
                key if isinstance(key, tuple) else (key,): value
                for key, value in components.items()
            }
        else:
            if len(components) != manifold.dimension:
                raise ValueError("One component is required for each coordinate.")
            data = {(index,): value for index, value in enumerate(components)}
        return cls(manifold, 1, data)

    @classmethod
    def basis(cls, manifold: Manifold, index: int) -> "DifferentialForm":
        """Return the coordinate basis 1-form dx^index."""

        return cls(manifold, 1, {(index,): sp.S.One})

    def __getitem__(self, indices: int | tuple[int, ...]) -> sp.Expr:
        """Return a component, accounting for antisymmetry."""

        if isinstance(indices, int):
            indices = (indices,)
        index, sign = _canonicalize(indices, self.degree, self.manifold.dimension)
        if sign == 0:
            return sp.S.Zero
        return sign * self.components.get(index, sp.S.Zero)

    def __add__(self, other: "DifferentialForm") -> "DifferentialForm":
        """Add forms of the same degree on the same manifold."""

        self._validate_compatible(other)
        components = dict(self.components)
        for index, value in other.components.items():
            components[index] = components.get(index, sp.S.Zero) + value
        return DifferentialForm(self.manifold, self.degree, components)

    def __sub__(self, other: "DifferentialForm") -> "DifferentialForm":
        """Subtract forms of the same degree on the same manifold."""

        return self + (-other)

    def __neg__(self) -> "DifferentialForm":
        """Negate a form."""

        return DifferentialForm(self.manifold, self.degree, {index: -value for index, value in self.components.items()})

    def __mul__(self, scalar: Any) -> "DifferentialForm":
        """Multiply by a scalar expression."""

        value = sp.sympify(scalar)
        return DifferentialForm(self.manifold, self.degree, {index: value * component for index, component in self.components.items()})

    def __rmul__(self, scalar: Any) -> "DifferentialForm":
        """Multiply by a scalar expression."""

        return self * scalar

    def __repr__(self) -> str:
        """Return a readable representation."""

        return f"DifferentialForm(manifold={self.manifold.name!r}, degree={self.degree}, components={self.components})"

    def __str__(self) -> str:
        """Return a concise human-readable representation."""

        return f"{self.degree}-form on {self.manifold.name}: {self.components}"

    def copy(self) -> "DifferentialForm":
        """Return a copy of this form."""

        return DifferentialForm(self.manifold, self.degree, dict(self.components))

    def simplify(self) -> "DifferentialForm":
        """Return a form with simplified components."""

        return DifferentialForm(self.manifold, self.degree, {index: sp.simplify(value) for index, value in self.components.items()})

    def equals(self, other: "DifferentialForm", simplify: bool = True) -> bool:
        """Return whether two forms are symbolically equal."""

        if self.manifold != other.manifold or self.degree != other.degree:
            return False
        all_indices = set(self.components) | set(other.components)
        for index in all_indices:
            difference = self.components.get(index, sp.S.Zero) - other.components.get(index, sp.S.Zero)
            if simplify:
                difference = sp.simplify(difference)
            if difference != 0:
                return False
        return True

    def nonzero_components(self, simplify: bool = True) -> dict[tuple[int, ...], sp.Expr]:
        """Return nonzero components."""

        if simplify:
            return {
                index: sp.simplify(value)
                for index, value in self.components.items()
                if sp.simplify(value) != 0
            }
        return dict(self.components)

    def to_latex(self) -> str:
        """Return a basic LaTeX representation."""

        if not self.components:
            return "0"
        terms = []
        coords = self.manifold.coordinates
        for index, value in self.components.items():
            if index == ():
                terms.append(sp.latex(value))
                continue
            basis = r"\wedge ".join(rf"d{sp.latex(coords[i])}" for i in index)
            terms.append(rf"{sp.latex(value)}\,{basis}")
        return " + ".join(terms)

    def to_markdown(self) -> str:
        """Return a Markdown table of components."""

        from crr.display.pretty import markdown_table

        rows = [(str(index), str(value)) for index, value in self.nonzero_components().items()]
        return markdown_table(rows)

    def display(self) -> str:
        """Display or return this form as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def display_components(self) -> str:
        """Display or return a Markdown table of components."""

        from crr.display.pretty import display_markdown

        return display_markdown(self.to_markdown())

    def plot_2d(
        self,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        resolution: int = 30,
        params=None,
        ax=None,
        show: bool = True,
        title: str | None = None,
        save_path: str | None = None,
    ):
        """Plot this 0-, 1-, or 2-form on a 2D chart."""

        from crr.visualization.forms import plot_form_2d

        return plot_form_2d(
            self,
            x_range,
            y_range,
            resolution=resolution,
            params=params,
            ax=ax,
            show=show,
            title=title,
            save_path=save_path,
        )

    def wedge(self, other: "DifferentialForm") -> "DifferentialForm":
        """Return the wedge product of two forms."""

        if self.manifold != other.manifold:
            raise ValueError("Forms must live on the same manifold.")
        degree = self.degree + other.degree
        if degree > self.manifold.dimension:
            return DifferentialForm.zero(self.manifold, self.manifold.dimension)
        components: dict[tuple[int, ...], sp.Expr] = {}
        for left_index, left_value in self.components.items():
            for right_index, right_value in other.components.items():
                index, sign = _canonicalize(left_index + right_index, degree, self.manifold.dimension)
                if sign == 0:
                    continue
                components[index] = components.get(index, sp.S.Zero) + sign * left_value * right_value
        return DifferentialForm(self.manifold, degree, components)

    def exterior_derivative(self) -> "DifferentialForm":
        """Return the exterior derivative."""

        if self.degree == self.manifold.dimension:
            return DifferentialForm.zero(self.manifold, self.degree)
        components: dict[tuple[int, ...], sp.Expr] = {}
        for index, value in self.components.items():
            for coordinate_index, coordinate in enumerate(self.manifold.coordinates):
                derivative = sp.diff(value, coordinate)
                if derivative == 0:
                    continue
                new_index, sign = _canonicalize((coordinate_index,) + index, self.degree + 1, self.manifold.dimension)
                if sign == 0:
                    continue
                components[new_index] = components.get(new_index, sp.S.Zero) + sign * derivative
        return DifferentialForm(self.manifold, self.degree + 1, components)

    def hodge_star(self, metric: "Metric", simplify: bool = False) -> "DifferentialForm":
        """Return the Hodge star with coordinate-order orientation."""

        self._validate_metric(metric)
        n = self.manifold.dimension
        k = self.degree
        out_degree = n - k
        inverse = metric.inverse()
        density = sp.sqrt(metric.determinant())
        if simplify:
            density = sp.powdenest(sp.simplify(density), force=True)
        components: dict[tuple[int, ...], sp.Expr] = {}

        for out_index in combinations(range(n), out_degree):
            value = sp.S.Zero
            for full_index in product(range(n), repeat=k):
                if len(set(full_index)) != k:
                    continue
                epsilon = sp.LeviCivita(*(full_index + out_index))
                if epsilon == 0:
                    continue
                value += self._raised_component(full_index, inverse) * epsilon
            value = density * value / sp.factorial(k)
            components[out_index] = sp.simplify(value) if simplify else value
        return DifferentialForm(self.manifold, out_degree, components)

    def codifferential(self, metric: "Metric", simplify: bool = False) -> "DifferentialForm":
        """Return delta alpha = (-1)^(n*k+n+1) * d * alpha."""

        if self.degree == 0:
            return DifferentialForm.zero(self.manifold, 0)
        n = self.manifold.dimension
        sign = -1 if (n * self.degree + n + 1) % 2 else 1
        result = sign * self.hodge_star(metric, simplify=simplify).exterior_derivative().hodge_star(metric, simplify=simplify)
        return result.simplify() if simplify else result

    def hodge_laplacian(self, metric: "Metric", simplify: bool = False) -> "DifferentialForm":
        """Return Delta alpha = d delta alpha + delta d alpha."""

        if self.degree == 0:
            d_delta = DifferentialForm.zero(self.manifold, self.degree)
        else:
            delta_alpha = self.codifferential(metric, simplify=simplify)
            d_delta = delta_alpha.exterior_derivative()

        if self.degree == self.manifold.dimension:
            delta_d = DifferentialForm.zero(self.manifold, self.degree)
        else:
            d_alpha = self.exterior_derivative()
            delta_d = d_alpha.codifferential(metric, simplify=simplify)
        result = d_delta + delta_d
        return result.simplify() if simplify else result

    def integrate(
        self,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        simplify: bool = False,
    ) -> sp.Expr:
        """Integrate a top-degree coordinate form over coordinate ranges."""

        if self.degree != self.manifold.dimension:
            raise ValueError("Only top-degree forms can be integrated.")
        if len(ranges) != self.manifold.dimension:
            raise ValueError("One range is required for each coordinate.")
        top_index = tuple(range(self.manifold.dimension))
        value = self.components.get(top_index, sp.S.Zero)
        for integration_range in ranges:
            value = sp.integrate(value, integration_range)
        return sp.simplify(value) if simplify else value

    def _validate_compatible(self, other: "DifferentialForm") -> None:
        if self.manifold != other.manifold or self.degree != other.degree:
            raise ValueError("Forms must have the same manifold and degree.")

    def _validate_metric(self, metric: "Metric") -> None:
        if metric.manifold != self.manifold:
            raise ValueError("Metric must be defined on the same manifold.")

    def _raised_component(self, index: tuple[int, ...], inverse_metric: sp.Matrix) -> sp.Expr:
        if self.degree == 0:
            return self.components.get((), sp.S.Zero)
        value = sp.S.Zero
        n = self.manifold.dimension
        for lower_index in product(range(n), repeat=self.degree):
            component = self[lower_index]
            if component == 0:
                continue
            factor = sp.S.One
            for upper_axis, lower_axis in zip(index, lower_index, strict=True):
                factor *= inverse_metric[upper_axis, lower_axis]
            value += factor * component
        return value


def _canonicalize(indices: tuple[int, ...], degree: int, dimension: int) -> tuple[tuple[int, ...], int]:
    if len(indices) != degree:
        raise ValueError("Index tuple length must match form degree.")
    if any(index < 0 or index >= dimension for index in indices):
        raise IndexError("Form index out of range.")
    if len(set(indices)) != len(indices):
        return tuple(sorted(indices)), 0
    inversions = 0
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            if indices[left] > indices[right]:
                inversions += 1
    return tuple(sorted(indices)), -1 if inversions % 2 else 1


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
