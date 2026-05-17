"""Geometry computations for CRR."""

from crr.geometry.christoffel import christoffel_symbols
from crr.geometry.covariant_derivative import (
    covariant_derivative_covariant_2tensor,
    covariant_derivative_covector,
    covariant_derivative_scalar,
    covariant_derivative_vector,
)
from crr.geometry.einstein import einstein_tensor
from crr.geometry.geodesic import geodesic_acceleration, geodesic_equations
from crr.geometry.global_invariants import GaussBonnetResult, gauss_bonnet_check
from crr.geometry.ricci import ricci_tensor
from crr.geometry.riemann import riemann_tensor
from crr.geometry.scalar_curvature import scalar_curvature

__all__ = [
    "christoffel_symbols",
    "covariant_derivative_scalar",
    "covariant_derivative_vector",
    "covariant_derivative_covector",
    "covariant_derivative_covariant_2tensor",
    "riemann_tensor",
    "ricci_tensor",
    "scalar_curvature",
    "einstein_tensor",
    "geodesic_acceleration",
    "geodesic_equations",
    "GaussBonnetResult",
    "gauss_bonnet_check",
]
