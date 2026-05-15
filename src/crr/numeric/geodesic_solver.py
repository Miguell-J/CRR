"""Numerical geodesic solving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from crr.numeric.lambdify import lambdify_christoffel, lambdify_metric


@dataclass(frozen=True)
class GeodesicSolution:
    """Container for a numerical geodesic solution."""

    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    metric: "Metric"
    success: bool
    message: str
    raw_solution: Any = None

    def coordinates(self) -> np.ndarray:
        """Return coordinate samples with shape ``(num_points, dimension)``."""

        return self.x

    def velocities(self) -> np.ndarray:
        """Return velocity samples with shape ``(num_points, dimension)``."""

        return self.v

    def final_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return final coordinate and velocity vectors."""

        return self.x[-1].copy(), self.v[-1].copy()

    def speed_squared(self) -> np.ndarray:
        """Return ``g_ij(x) v^i v^j`` along the solution."""

        metric_func = lambdify_metric(self.metric)
        speeds = np.empty(len(self.t), dtype=float)
        for row, (point, velocity) in enumerate(zip(self.x, self.v, strict=True)):
            metric_components = metric_func(point)
            speeds[row] = float(velocity @ metric_components @ velocity)
        return speeds

    def energy(self) -> np.ndarray:
        """Return kinetic energy ``1/2 g_ij v^i v^j`` along the solution."""

        return 0.5 * self.speed_squared()

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(t, x, v)`` NumPy arrays."""

        return self.t, self.x, self.v

    def plot_coordinates(self):
        """Plot coordinate components against the integration parameter."""

        from crr.visualization.geodesics import plot_geodesic_coordinates

        return plot_geodesic_coordinates(self)

    def plot_phase_component(self, i: int):
        """Plot coordinate ``x_i`` against velocity ``v_i``."""

        from crr.visualization.geodesics import plot_phase_component

        return plot_phase_component(self, i)


def solve_geodesic(
    metric: "Metric",
    x0: Any,
    v0: Any,
    t_span: tuple[float, float],
    num_points: int = 1000,
    method: str = "RK45",
    rtol: float = 1e-9,
    atol: float = 1e-9,
    simplify: bool = False,
    **solve_ivp_kwargs: Any,
) -> GeodesicSolution:
    """Numerically solve the geodesic equation for a metric."""

    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    n = metric.dimension
    initial_x = _as_float_vector(x0, n, "x0")
    initial_v = _as_float_vector(v0, n, "v0")
    initial_state = np.concatenate([initial_x, initial_v])
    t_eval = np.linspace(float(t_span[0]), float(t_span[1]), num_points)

    christoffel_func = lambdify_christoffel(
        metric.christoffel_symbols(simplify=simplify),
        metric.coordinates,
    )

    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        position = state[:n]
        velocity = state[n:]
        gamma = christoffel_func(position)
        acceleration = -np.einsum("kij,i,j->k", gamma, velocity, velocity)
        return np.concatenate([velocity, acceleration])

    raw_solution = solve_ivp(
        rhs,
        (float(t_span[0]), float(t_span[1])),
        initial_state,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        **solve_ivp_kwargs,
    )

    y = raw_solution.y.T
    return GeodesicSolution(
        t=raw_solution.t,
        x=y[:, :n],
        v=y[:, n:],
        metric=metric,
        success=bool(raw_solution.success),
        message=str(raw_solution.message),
        raw_solution=raw_solution,
    )


def _as_float_vector(values: Any, dimension: int, name: str) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=float)
    if array.shape != (dimension,):
        raise ValueError(f"{name} must have shape (dimension,).")
    return array


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crr.core.metric import Metric
