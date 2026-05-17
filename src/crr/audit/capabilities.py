"""Structured capability reporting for CRR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Capability:
    """A documented CRR capability and its verification status."""

    name: str
    category: str
    status: str
    description: str
    example: str | None = None
    tested: bool = False
    experimental: bool = False
    limitations: list[str] = field(default_factory=list)


def get_capabilities() -> list[Capability]:
    """Return CRR's current capabilities as structured records."""

    return [
        Capability(
            "Manifolds, coordinate charts, and metrics",
            "Core geometry",
            "stable",
            "Construct local coordinate manifolds and nondegenerate metric tensors.",
            "examples/euclidean_plane.py",
            tested=True,
            limitations=["Local coordinate charts only; no atlas or transition maps."],
        ),
        Capability(
            "Curvature tensors",
            "Core geometry",
            "stable",
            "Compute Christoffel symbols, Riemann tensor, Ricci tensor, scalar curvature, and Einstein tensor.",
            "examples/sphere.py",
            tested=True,
            limitations=["Large symbolic metrics can be expensive; use simplification selectively."],
        ),
        Capability(
            "Tensor containers",
            "Tensor calculus",
            "stable",
            "Store dense coordinate tensor components with index-signature metadata.",
            tested=True,
            limitations=["No full abstract tensor algebra or sparse tensor engine yet."],
        ),
        Capability(
            "Covariant derivatives",
            "Tensor calculus",
            "stable",
            "Compute covariant derivatives of scalars, vectors, covectors, and covariant 2-tensors.",
            "examples/covariant_derivative_sphere.py",
            tested=True,
        ),
        Capability(
            "Symbolic geodesic equations",
            "Geodesics",
            "stable",
            "Derive geodesic equations from a metric.",
            "examples/geodesics_sphere.py",
            tested=True,
        ),
        Capability(
            "Numerical geodesic solving",
            "Geodesics",
            "stable",
            "Integrate geodesic initial-value problems and inspect solutions.",
            "examples/numeric_geodesic_sphere.py",
            tested=True,
            limitations=["Local coordinates can hit singularities or chart boundaries."],
        ),
        Capability(
            "Parametrized embedded surfaces",
            "Embedded surfaces",
            "stable",
            "Represent parametrized maps, pull back ambient metrics, and compute induced geometry.",
            "examples/embedded_sphere.py",
            tested=True,
        ),
        Capability(
            "Extrinsic surface geometry",
            "Embedded surfaces",
            "stable",
            "Compute normals, second fundamental form, Gaussian curvature, and mean curvature for surfaces in R3.",
            "examples/extrinsic_torus.py",
            tested=True,
            limitations=["Focused on 2D surfaces in 3D Euclidean ambient space."],
        ),
        Capability(
            "Coordinate-domain integration",
            "Global integration",
            "stable",
            "Integrate scalar densities, surface areas, and curvature densities over explicit coordinate ranges.",
            "examples/integrate_sphere_area.py",
            tested=True,
            limitations=["Does not glue charts or handle global identifications automatically."],
        ),
        Capability(
            "Gauss-Bonnet checks",
            "Global integration",
            "stable",
            "Compare integrated Gaussian curvature against expected Euler characteristic values.",
            "examples/gauss_bonnet_sphere.py",
            tested=True,
        ),
        Capability(
            "Relativity metric presets",
            "Relativity",
            "stable",
            "Provide standard local-coordinate metrics including Minkowski, Schwarzschild, FLRW, Kerr, and related examples.",
            "examples/extended_relativity_presets.py",
            tested=True,
            limitations=["Presets are local charts and inherit coordinate singularities."],
        ),
        Capability(
            "Differential forms",
            "Differential forms",
            "stable",
            "Construct sparse coordinate-basis forms, wedge products, exterior derivatives, and top-form integrals.",
            "examples/forms_basic.py",
            tested=True,
        ),
        Capability(
            "Hodge theory helpers",
            "Differential forms",
            "stable",
            "Compute Hodge star, codifferential, and Hodge Laplacian for coordinate forms.",
            "examples/hodge_star_euclidean.py",
            tested=True,
            limitations=["Pseudo-Riemannian sign conventions are experimental."],
        ),
        Capability(
            "Finite-dimensional Lie algebras",
            "Lie algebras",
            "stable",
            "Represent symbolic structure constants, brackets, Jacobi checks, and Killing forms.",
            "examples/lie_so3.py",
            tested=True,
        ),
        Capability(
            "Algebra-valued forms",
            "Lie algebras",
            "stable",
            "Represent Lie-algebra-valued differential forms and bracket-wedge products.",
            "examples/algebra_valued_forms.py",
            tested=True,
        ),
        Capability(
            "Gauge potentials and curvature",
            "Gauge fields",
            "stable",
            "Represent local Lie-algebra-valued connection 1-forms and compute curvature 2-forms.",
            "examples/gauge_su2_curvature.py",
            tested=True,
            limitations=["Local trivializations only; no principal bundle topology."],
        ),
        Capability(
            "Gauge transformations",
            "Gauge fields",
            "stable",
            "Apply abelian gauge transformations and infinitesimal non-abelian transformations.",
            "examples/gauge_abelian_transform.py",
            tested=True,
            limitations=["Finite non-abelian transformations are not implemented."],
        ),
        Capability(
            "Yang-Mills action density",
            "Gauge fields",
            "experimental",
            "Compute local symbolic densities of the form sum_ab <e_a,e_b> F^a wedge *F^b.",
            "examples/yang_mills_action_density.py",
            tested=True,
            experimental=True,
            limitations=["Uses identity inner product by default; pseudo-Riemannian signs are not fully handled."],
        ),
        Capability(
            "Matplotlib visualization",
            "Visualization",
            "stable",
            "Plot surfaces, curvature maps, numerical geodesics, scalar fields, vector fields, and 2D forms.",
            "examples/plot_sphere_surface.py",
            tested=True,
            limitations=["Optional matplotlib dependency; chart-based rendering only."],
        ),
        Capability(
            "Notebook-friendly display",
            "Jupyter/display",
            "stable",
            "Render LaTeX and Markdown summaries where IPython is available, with string fallbacks elsewhere.",
            "notebooks/01_sphere_geometry.py",
            tested=True,
            limitations=["Display helpers are lightweight, not a full notebook UI framework."],
        ),
    ]


def capability_report_markdown() -> str:
    """Return a Markdown capability report."""

    capabilities = get_capabilities()
    categories = []
    for capability in capabilities:
        if capability.category not in categories:
            categories.append(capability.category)

    lines = ["# CRR Capability Report", ""]
    lines.append("This report summarizes the local-coordinate symbolic/numeric capabilities currently exposed by CRR.")
    lines.append("")
    for category in categories:
        lines.extend([f"## {category}", ""])
        for cap in [item for item in capabilities if item.category == category]:
            flags = []
            flags.append(f"status: {cap.status}")
            flags.append("tested" if cap.tested else "untested")
            if cap.experimental:
                flags.append("experimental")
            lines.append(f"### {cap.name}")
            lines.append(f"- {'; '.join(flags)}")
            lines.append(f"- {cap.description}")
            if cap.example:
                lines.append(f"- Example: `{cap.example}`")
            if cap.limitations:
                lines.append("- Limitations: " + "; ".join(cap.limitations))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def print_capability_report() -> None:
    """Print the Markdown capability report."""

    print(capability_report_markdown())


def save_capability_report(path) -> Path:
    """Write the Markdown capability report to ``path`` and return it."""

    output_path = Path(path)
    output_path.write_text(capability_report_markdown(), encoding="utf-8")
    return output_path
