"""Metric tensor implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from crr.core.manifold import Manifold
from crr.core.tensor import Tensor
from crr.core.types import MatrixLike
from crr.symbolic.simplify import simplify_array


@dataclass
class Metric:
    """A covariant metric tensor on a coordinate chart."""

    manifold: Manifold
    components: sp.Matrix
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __init__(self, manifold: Manifold, components: MatrixLike) -> None:
        matrix = sp.Matrix(components)
        if matrix.shape != (manifold.dimension, manifold.dimension):
            raise ValueError("Metric components must be a square matrix matching manifold dimension.")
        if matrix.det() == 0:
            raise ValueError("Metric must be non-degenerate.")
        self.manifold = manifold
        self.components = matrix
        self._cache = {}

    @property
    def dimension(self) -> int:
        """Dimension of the underlying manifold."""

        return self.manifold.dimension

    @property
    def coordinates(self) -> tuple[sp.Symbol, ...]:
        """Coordinate symbols for the metric chart."""

        return self.manifold.coordinates

    def __repr__(self) -> str:
        """Return a concise debug representation."""

        return f"Metric(manifold={self.manifold.name!r}, dimension={self.dimension})"

    def __str__(self) -> str:
        """Return a concise human-readable representation."""

        return f"Metric on {self.manifold.name} ({self.dimension}D)"

    def inverse(self, simplify: bool = False) -> sp.Matrix:
        """Return the inverse metric matrix g^ij."""

        key = "inverse_simplified" if simplify else "inverse"
        if key not in self._cache:
            inverse = self.components.inv()
            self._cache[key] = inverse.applyfunc(sp.simplify) if simplify else inverse
        return self._cache[key]

    def determinant(self, simplify: bool = False) -> sp.Expr:
        """Return det(g)."""

        key = "determinant_simplified" if simplify else "determinant"
        if key not in self._cache:
            det = self.components.det()
            self._cache[key] = sp.simplify(det) if simplify else det
        return self._cache[key]

    def volume_density(self, simplify: bool = False, absolute: bool = True) -> sp.Expr:
        """Return the Riemannian volume density sqrt(abs(det(g)))."""

        determinant = self.determinant()
        density = sp.sqrt(sp.Abs(determinant)) if absolute else sp.sqrt(determinant)
        return sp.simplify(density) if simplify else density

    def integrate_scalar(
        self,
        scalar: sp.Expr,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        simplify: bool = False,
        absolute_density: bool = True,
    ) -> sp.Expr:
        """Integrate a scalar field against the metric volume density."""

        self._validate_integration_ranges(ranges)
        value = sp.sympify(scalar) * self.volume_density(absolute=absolute_density)
        for integration_range in ranges:
            value = sp.integrate(value, integration_range)
        return sp.simplify(value) if simplify else value

    def integrate_volume(
        self,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        simplify: bool = False,
        absolute_density: bool = True,
    ) -> sp.Expr:
        """Integrate the metric volume density over a coordinate domain."""

        return self.integrate_scalar(
            sp.S.One,
            ranges,
            simplify=simplify,
            absolute_density=absolute_density,
        )

    def integrate_scalar_numeric(
        self,
        scalar: sp.Expr,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
        params: dict[sp.Symbol, object] | None = None,
        nquad_options: dict[str, object] | None = None,
        absolute_density: bool = True,
    ) -> float:
        """Numerically integrate a scalar field against the metric volume density."""

        from crr.numeric.integration import integrate_scalar_numeric

        return integrate_scalar_numeric(
            self,
            scalar,
            ranges,
            params=params,
            nquad_options=nquad_options,
            absolute_density=absolute_density,
        )

    def christoffel_symbols(self, simplify: bool = False) -> Tensor:
        """Return Christoffel symbols Gamma^k_ij."""

        from crr.geometry.christoffel import christoffel_symbols

        key = "christoffel_simplified" if simplify else "christoffel"
        if key not in self._cache:
            self._cache[key] = christoffel_symbols(self, simplify=simplify)
        return self._cache[key]

    def riemann_tensor(self, simplify: bool = False) -> Tensor:
        """Return Riemann tensor R^i_jkl."""

        from crr.geometry.riemann import riemann_tensor

        key = "riemann_simplified" if simplify else "riemann"
        if key not in self._cache:
            tensor = riemann_tensor(self, simplify=simplify)
            self._cache[key] = Tensor(simplify_array(tensor.components), tensor.name, tensor.index_signature) if simplify else tensor
        return self._cache[key]

    def ricci_tensor(self, simplify: bool = False) -> Tensor:
        """Return Ricci tensor R_ij."""

        from crr.geometry.ricci import ricci_tensor

        key = "ricci_simplified" if simplify else "ricci"
        if key not in self._cache:
            tensor = ricci_tensor(self, simplify=simplify)
            self._cache[key] = Tensor(simplify_array(tensor.components), tensor.name, tensor.index_signature) if simplify else tensor
        return self._cache[key]

    def scalar_curvature(self, simplify: bool = False) -> sp.Expr:
        """Return scalar curvature R."""

        from crr.geometry.scalar_curvature import scalar_curvature

        key = "scalar_curvature_simplified" if simplify else "scalar_curvature"
        if key not in self._cache:
            scalar = scalar_curvature(self, simplify=simplify)
            self._cache[key] = sp.simplify(scalar) if simplify else scalar
        return self._cache[key]

    def einstein_tensor(self, simplify: bool = False) -> Tensor:
        """Return Einstein tensor G_ij."""

        from crr.geometry.einstein import einstein_tensor

        key = "einstein_simplified" if simplify else "einstein"
        if key not in self._cache:
            tensor = einstein_tensor(self, simplify=simplify)
            self._cache[key] = Tensor(simplify_array(tensor.components), tensor.name, tensor.index_signature) if simplify else tensor
        return self._cache[key]

    def covariant_derivative_scalar(self, scalar_field: sp.Expr, simplify: bool = False) -> Tensor:
        """Return nabla_i f for a scalar field."""

        from crr.geometry.covariant_derivative import covariant_derivative_scalar

        return covariant_derivative_scalar(self, scalar_field, simplify=simplify)

    def covariant_derivative_vector(self, vector_field: Any, simplify: bool = False) -> Tensor:
        """Return nabla_j V^i for a contravariant vector field."""

        from crr.geometry.covariant_derivative import covariant_derivative_vector

        return covariant_derivative_vector(self, vector_field, simplify=simplify)

    def covariant_derivative_covector(self, covector_field: Any, simplify: bool = False) -> Tensor:
        """Return nabla_j omega_i for a covector field."""

        from crr.geometry.covariant_derivative import covariant_derivative_covector

        return covariant_derivative_covector(self, covector_field, simplify=simplify)

    def covariant_derivative_covariant_2tensor(self, tensor_field: Any, simplify: bool = False) -> Tensor:
        """Return nabla_k T_ij for a covariant rank-2 tensor field."""

        from crr.geometry.covariant_derivative import covariant_derivative_covariant_2tensor

        return covariant_derivative_covariant_2tensor(self, tensor_field, simplify=simplify)

    def flat(self, vector: Any, simplify: bool = False) -> "DifferentialForm":
        """Lower a vector field index and return the associated 1-form."""

        from crr.forms.musical import flat

        return flat(self, vector, simplify=simplify)

    def sharp(self, one_form: "DifferentialForm", simplify: bool = False) -> list[sp.Expr]:
        """Raise a 1-form index and return vector components."""

        from crr.forms.musical import sharp

        return sharp(self, one_form, simplify=simplify)

    def geodesic_acceleration(
        self,
        velocity_symbols: list[sp.Symbol] | tuple[sp.Symbol, ...] | None = None,
        simplify: bool = False,
    ) -> list[sp.Expr]:
        """Return a^k = - Gamma^k_ij v^i v^j."""

        from crr.geometry.geodesic import geodesic_acceleration

        return geodesic_acceleration(self, velocity_symbols=velocity_symbols, simplify=simplify)

    def geodesic_equations(
        self,
        parameter_symbol: sp.Symbol | None = None,
        coordinate_functions: Any = None,
        simplify: bool = False,
    ) -> list[sp.Expr]:
        """Return d2x^k/dlambda^2 + Gamma^k_ij dx^i/dlambda dx^j/dlambda = 0 left sides."""

        from crr.geometry.geodesic import geodesic_equations

        return geodesic_equations(
            self,
            parameter_symbol=parameter_symbol,
            coordinate_functions=coordinate_functions,
            simplify=simplify,
        )

    def solve_geodesic(
        self,
        x0: Any,
        v0: Any,
        t_span: tuple[float, float],
        num_points: int = 1000,
        method: str = "RK45",
        rtol: float = 1e-9,
        atol: float = 1e-9,
        simplify: bool = False,
        **solve_ivp_kwargs: Any,
    ) -> "GeodesicSolution":
        """Numerically solve the geodesic equation for initial position and velocity."""

        from crr.numeric.geodesic_solver import solve_geodesic

        return solve_geodesic(
            self,
            x0=x0,
            v0=v0,
            t_span=t_span,
            num_points=num_points,
            method=method,
            rtol=rtol,
            atol=atol,
            simplify=simplify,
            **solve_ivp_kwargs,
        )

    def display(self) -> str:
        """Display or return the metric matrix."""

        return self.display_matrix()

    def to_latex(self) -> str:
        """Return the metric matrix as LaTeX."""

        return sp.latex(self.components)

    def to_markdown(self) -> str:
        """Return a Markdown summary of the metric matrix."""

        from crr.display.pretty import markdown_table

        rows = [("Manifold", self.manifold.name), ("Dimension", str(self.dimension)), ("Matrix", str(self.components))]
        return markdown_table(rows, headers=("Field", "Value"))

    def display_matrix(self) -> str:
        """Display or return the metric matrix as LaTeX."""

        from crr.display.pretty import display_latex

        return display_latex(self.to_latex())

    def display_christoffel_nonzero(self, simplify: bool = False) -> str:
        """Display or return nonzero Christoffel symbols."""

        return self.christoffel_symbols(simplify=simplify).display_nonzero()

    def display_ricci(self, simplify: bool = False) -> str:
        """Display or return Ricci tensor components."""

        return self.ricci_tensor(simplify=simplify).display_nonzero()

    def display_scalar_curvature(self, simplify: bool = False) -> str:
        """Display or return scalar curvature."""

        from crr.display.pretty import display_latex

        return display_latex(sp.latex(self.scalar_curvature(simplify=simplify)))

    def display_einstein(self, simplify: bool = False) -> str:
        """Display or return Einstein tensor components."""

        return self.einstein_tensor(simplify=simplify).display_nonzero()

    def _validate_integration_ranges(
        self,
        ranges: list[tuple[sp.Symbol, object, object]] | tuple[tuple[sp.Symbol, object, object], ...],
    ) -> None:
        if len(ranges) != self.dimension:
            raise ValueError("One integration range is required for each coordinate.")
        range_coordinates = tuple(item[0] for item in ranges)
        if range_coordinates != self.coordinates:
            raise ValueError("Integration ranges must be ordered like the metric coordinates.")


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.forms import DifferentialForm
    from crr.numeric.geodesic_solver import GeodesicSolution
