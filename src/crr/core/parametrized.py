"""Parametrized maps and induced metrics."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import sympy as sp

from crr.core.manifold import Manifold
from crr.core.metric import Metric
from crr.core.types import MatrixLike

if TYPE_CHECKING:
    from crr.numeric.geodesic_solver import GeodesicSolution


@dataclass(frozen=True)
class ParametrizedMap:
    """A parametrized map from a coordinate domain into an ambient space."""

    name: str
    parameters: tuple[sp.Symbol, ...]
    components: tuple[sp.Expr, ...]
    ambient_coordinates: tuple[sp.Symbol, ...] | None = None
    ambient_metric: sp.Matrix | None = None

    def __init__(
        self,
        name: str,
        parameters: Sequence[sp.Symbol],
        components: Sequence[sp.Expr],
        ambient_coordinates: Sequence[sp.Symbol] | None = None,
        ambient_metric: MatrixLike | None = None,
    ) -> None:
        parameters_tuple = tuple(parameters)
        components_tuple = tuple(sp.sympify(component) for component in components)
        ambient_coordinates_tuple = tuple(ambient_coordinates) if ambient_coordinates is not None else None

        if not parameters_tuple:
            raise ValueError("A parametrized map must have at least one parameter.")
        if not components_tuple:
            raise ValueError("A parametrized map must have at least one component.")
        if not all(isinstance(parameter, sp.Symbol) for parameter in parameters_tuple):
            raise TypeError("Parameters must be SymPy symbols.")
        if ambient_coordinates_tuple is not None:
            if len(ambient_coordinates_tuple) != len(components_tuple):
                raise ValueError("Ambient coordinates must match the number of map components.")
            if not all(isinstance(coord, sp.Symbol) for coord in ambient_coordinates_tuple):
                raise TypeError("Ambient coordinates must be SymPy symbols.")

        ambient_metric_matrix = None
        if ambient_metric is not None:
            ambient_metric_matrix = sp.Matrix(ambient_metric)
            ambient_dimension = len(components_tuple)
            if ambient_metric_matrix.shape != (ambient_dimension, ambient_dimension):
                raise ValueError("Ambient metric must be square and match the ambient dimension.")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parameters", parameters_tuple)
        object.__setattr__(self, "components", components_tuple)
        object.__setattr__(self, "ambient_coordinates", ambient_coordinates_tuple)
        object.__setattr__(self, "ambient_metric", ambient_metric_matrix)

    @property
    def dimension(self) -> int:
        """Dimension of the parameter domain."""

        return len(self.parameters)

    @property
    def ambient_dimension(self) -> int:
        """Dimension of the ambient coordinate space."""

        return len(self.components)

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"ParametrizedMap(name={self.name!r}, dimension={self.dimension}, ambient_dimension={self.ambient_dimension})"

    def __str__(self) -> str:
        """Return a concise human-readable representation."""

        return f"{self.name}: R^{self.dimension} -> R^{self.ambient_dimension}"

    def jacobian(self, simplify: bool = False) -> sp.Matrix:
        """Return the Jacobian matrix dF^i/du^a."""

        jacobian = sp.Matrix(self.components).jacobian(self.parameters)
        return jacobian.applyfunc(sp.simplify) if simplify else jacobian

    def induced_metric_components(self, simplify: bool = False) -> sp.Matrix:
        """Return the induced or pullback metric components on the parameter domain."""

        jacobian = self.jacobian()
        ambient_metric = self._ambient_metric_on_domain()
        components = jacobian.T * ambient_metric * jacobian
        return components.applyfunc(sp.simplify) if simplify else components

    def first_fundamental_form(self, simplify: bool = False) -> sp.Matrix:
        """Alias for induced metric components."""

        return self.induced_metric_components(simplify=simplify)

    def pullback_metric(self, simplify: bool = False, manifold_name: str | None = None) -> Metric:
        """Return the pullback metric as a CRR Metric on the parameter domain."""

        manifold = Manifold(
            name=manifold_name or f"{self.name}_domain",
            dimension=self.dimension,
            coordinates=self.parameters,
        )
        return Metric(manifold, self.induced_metric_components(simplify=simplify))

    def volume_density(self, simplify: bool = False) -> sp.Expr:
        """Return sqrt(det(g)) for the induced metric."""

        determinant = self.induced_metric_components().det()
        density = sp.sqrt(determinant)
        return sp.simplify(density) if simplify else density

    def area_density(self, simplify: bool = False) -> sp.Expr:
        """Alias for volume density, commonly used for 2D surfaces."""

        return self.volume_density(simplify=simplify)

    def integrate_scalar(
        self,
        scalar: sp.Expr,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        simplify: bool = False,
    ) -> sp.Expr:
        """Integrate a scalar field against the induced volume density."""

        value = sp.sympify(scalar) * self.volume_density()
        for integration_range in ranges:
            value = sp.integrate(value, integration_range)
        return sp.simplify(value) if simplify else value

    def integrate_area(
        self,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        simplify: bool = False,
    ) -> sp.Expr:
        """Integrate the induced area density over a parameter domain."""

        return self.integrate_scalar(sp.S.One, ranges, simplify=simplify)

    def integrate_gaussian_curvature(
        self,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        intrinsic: bool = True,
        simplify: bool = False,
    ) -> sp.Expr:
        """Integrate Gaussian curvature against the induced area density."""

        if intrinsic:
            curvature = sp.Rational(1, 2) * self.pullback_metric(simplify=simplify).scalar_curvature(simplify=simplify)
        else:
            curvature = self.gaussian_curvature_extrinsic(simplify=simplify)
        return self.integrate_scalar(curvature, ranges, simplify=simplify)

    def integrate_mean_curvature(
        self,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        simplify: bool = False,
    ) -> sp.Expr:
        """Integrate mean curvature against the induced area density."""

        return self.integrate_scalar(self.mean_curvature(simplify=simplify), ranges, simplify=simplify)

    def evaluate(self, points: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Evaluate the parametrization at parameter points.

        ``points`` must have shape ``(m, domain_dimension)``. A single point
        with shape ``(domain_dimension,)`` is also accepted.
        """

        point_array = np.asarray(points, dtype=float)
        if point_array.ndim == 1:
            point_array = point_array.reshape(1, -1)
        if point_array.ndim != 2 or point_array.shape[1] != self.dimension:
            raise ValueError("points must have shape (m, domain_dimension).")

        evaluator = sp.lambdify(self.parameters, self.components, modules="numpy")
        values = evaluator(*[point_array[:, axis] for axis in range(self.dimension)])
        if not isinstance(values, (list, tuple)):
            values = [values]
        columns = []
        for value in values:
            column = np.asarray(value, dtype=float)
            if column.ndim == 0:
                column = np.full(point_array.shape[0], float(column))
            columns.append(column)
        return np.column_stack(columns).astype(float)

    def embed_geodesic(self, solution: "GeodesicSolution") -> np.ndarray:
        """Evaluate this parametrization along a numerical geodesic solution."""

        return self.evaluate(solution.x)

    def display(self) -> str:
        """Display or return the parametrization."""

        return self.display_parametrization()

    def to_latex(self) -> str:
        """Return the parametrization as LaTeX equations."""

        equations = []
        for i, component in enumerate(self.components):
            equations.append(rf"F^{i} = {sp.latex(component)}")
        return r"\\ ".join(equations)

    def to_markdown(self) -> str:
        """Return a Markdown table of parametrization components."""

        from crr.display.pretty import markdown_table

        rows = [(f"F^{i}", str(component)) for i, component in enumerate(self.components)]
        return markdown_table(rows, headers=("Component", "Expression"))

    def display_parametrization(self) -> str:
        """Display or return the parametrization as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def display_induced_metric(self, simplify: bool = False) -> str:
        """Display or return the induced metric matrix."""

        from crr.display.pretty import display_latex

        return display_latex(sp.latex(self.induced_metric_components(simplify=simplify)))

    def display_curvatures(self, simplify: bool = False) -> str:
        """Display or return available surface curvature expressions."""

        from crr.display.pretty import display_latex

        metric = self.pullback_metric(simplify=simplify)
        items = [rf"R = {sp.latex(metric.scalar_curvature(simplify=simplify))}"]
        if self.dimension == 2 and self.ambient_dimension == 3:
            items.append(rf"K = {sp.latex(self.gaussian_curvature_extrinsic(simplify=simplify))}")
            items.append(rf"H = {sp.latex(self.mean_curvature(simplify=simplify))}")
        return display_latex(r"\\ ".join(items))

    def plot_surface(
        self,
        u_range: tuple[float, float],
        v_range: tuple[float, float],
        resolution: int = 80,
        params: dict[sp.Symbol, object] | None = None,
        ax=None,
        show: bool = True,
        color=None,
        alpha: float = 1.0,
        wireframe: bool = False,
        title: str | None = None,
        save_path: str | None = None,
    ):
        """Plot this 2D R3 parametrized surface."""

        from crr.visualization.surfaces import plot_surface

        return plot_surface(
            self,
            u_range,
            v_range,
            resolution=resolution,
            params=params,
            ax=ax,
            show=show,
            color=color,
            alpha=alpha,
            wireframe=wireframe,
            title=title,
            save_path=save_path,
        )

    def plot_curvature(
        self,
        curvature: str = "gaussian",
        u_range: tuple[float, float] | None = None,
        v_range: tuple[float, float] | None = None,
        resolution: int = 80,
        params: dict[sp.Symbol, object] | None = None,
        intrinsic: bool = False,
        ax=None,
        show: bool = True,
        title: str | None = None,
        save_path: str | None = None,
    ):
        """Plot Gaussian or mean curvature on this surface."""

        from crr.visualization.surfaces import plot_curvature

        return plot_curvature(
            self,
            curvature=curvature,
            u_range=u_range,
            v_range=v_range,
            resolution=resolution,
            params=params,
            intrinsic=intrinsic,
            ax=ax,
            show=show,
            title=title,
            save_path=save_path,
        )

    def plot_geodesic(
        self,
        solution: "GeodesicSolution",
        ax=None,
        show: bool = True,
        color=None,
        linewidth: float = 2,
        title: str | None = None,
        save_path: str | None = None,
    ):
        """Plot an embedded geodesic on this surface."""

        from crr.visualization.surfaces import plot_geodesic

        return plot_geodesic(
            self,
            solution,
            ax=ax,
            show=show,
            color=color,
            linewidth=linewidth,
            title=title,
            save_path=save_path,
        )

    def plot_surface_with_geodesic(
        self,
        solution: "GeodesicSolution",
        u_range: tuple[float, float],
        v_range: tuple[float, float],
        resolution: int = 80,
        params: dict[sp.Symbol, object] | None = None,
        show: bool = True,
        surface_alpha: float = 0.6,
        geodesic_color=None,
        title: str | None = None,
        save_path: str | None = None,
    ):
        """Plot this surface with an embedded geodesic."""

        from crr.visualization.surfaces import plot_surface_with_geodesic

        return plot_surface_with_geodesic(
            self,
            solution,
            u_range,
            v_range,
            resolution=resolution,
            params=params,
            show=show,
            surface_alpha=surface_alpha,
            geodesic_color=geodesic_color,
            title=title,
            save_path=save_path,
        )

    def tangent_vectors(self, simplify: bool = False) -> tuple[sp.Matrix, sp.Matrix]:
        """Return tangent vectors F_u and F_v for a surface in Euclidean R3."""

        self._validate_euclidean_surface()
        tangents = tuple(sp.Matrix([sp.diff(component, parameter) for component in self.components]) for parameter in self.parameters)
        if simplify:
            tangents = tuple(vector.applyfunc(sp.simplify) for vector in tangents)
        return tangents  # type: ignore[return-value]

    def normal_vector(self, simplify: bool = False, unit: bool = True) -> sp.Matrix:
        """Return the oriented normal vector.

        The orientation is determined by F_u x F_v, where ``u`` and ``v`` are
        the first and second parameters.
        """

        self._validate_euclidean_surface()
        tangent_u, tangent_v = self.tangent_vectors()
        normal = tangent_u.cross(tangent_v)
        if unit:
            norm = sp.sqrt(normal.dot(normal))
            normal = normal / norm
        return normal.applyfunc(sp.simplify) if simplify else normal

    def second_fundamental_form(self, simplify: bool = False) -> sp.Matrix:
        """Return the second fundamental form matrix II for a Euclidean R3 surface."""

        self._validate_euclidean_surface()
        normal = self.normal_vector(unit=True)
        components = sp.zeros(2, 2)
        for a, parameter_a in enumerate(self.parameters):
            for b, parameter_b in enumerate(self.parameters):
                second_derivative = sp.Matrix(
                    [sp.diff(component, parameter_a, parameter_b) for component in self.components]
                )
                components[a, b] = second_derivative.dot(normal)
        return components.applyfunc(sp.simplify) if simplify else components

    def shape_operator(self, simplify: bool = False) -> sp.Matrix:
        """Return the Weingarten map S = I^{-1} II."""

        self._validate_euclidean_surface()
        first_form = self.first_fundamental_form()
        second_form = self.second_fundamental_form()
        shape = first_form.inv() * second_form
        return shape.applyfunc(sp.simplify) if simplify else shape

    def gaussian_curvature_extrinsic(self, simplify: bool = False) -> sp.Expr:
        """Return extrinsic Gaussian curvature K = det(S)."""

        curvature = self.shape_operator().det()
        return sp.simplify(curvature) if simplify else curvature

    def mean_curvature(self, simplify: bool = False) -> sp.Expr:
        """Return mean curvature H = trace(S) / 2.

        The sign depends on the normal orientation determined by F_u x F_v.
        """

        curvature = sp.Rational(1, 2) * self.shape_operator().trace()
        return sp.simplify(curvature) if simplify else curvature

    def principal_curvatures(self, simplify: bool = False) -> tuple[sp.Expr, ...]:
        """Return symbolic principal curvatures when SymPy can compute them."""

        shape = self.shape_operator(simplify=simplify)
        eigenvalues = shape.eigenvals()
        curvatures: list[sp.Expr] = []
        for eigenvalue, multiplicity in eigenvalues.items():
            value = sp.simplify(eigenvalue) if simplify else eigenvalue
            curvatures.extend([value] * multiplicity)
        return tuple(curvatures)

    def _ambient_metric_on_domain(self) -> sp.Matrix:
        if self.ambient_metric is None:
            return sp.eye(self.ambient_dimension)

        if self.ambient_coordinates is None:
            return self.ambient_metric

        substitutions = {
            coord: component
            for coord, component in zip(self.ambient_coordinates, self.components, strict=True)
        }
        return self.ambient_metric.subs(substitutions)

    def _validate_euclidean_surface(self) -> None:
        if self.dimension != 2:
            raise ValueError("Extrinsic surface methods require a 2-dimensional parameter domain.")
        if self.ambient_dimension != 3:
            raise ValueError("Extrinsic surface methods require an embedding into Euclidean R3.")
        if self.ambient_metric is not None and self.ambient_metric != sp.eye(3):
            raise NotImplementedError("Extrinsic surface methods currently support only Euclidean ambient metric.")
