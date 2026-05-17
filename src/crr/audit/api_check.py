"""Lightweight public API consistency checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class ApiAuditReport:
    """Structured public API audit result."""

    passed_imports: list[str] = field(default_factory=list)
    failed_imports: list[str] = field(default_factory=list)
    missing_optional_dependencies: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return whether all required imports passed."""

        return not self.failed_imports

    def to_markdown(self) -> str:
        """Return a Markdown summary."""

        lines = ["# CRR Public API Audit", ""]
        lines.append(f"- Passed imports: {len(self.passed_imports)}")
        lines.append(f"- Failed imports: {len(self.failed_imports)}")
        if self.missing_optional_dependencies:
            lines.append(f"- Missing optional dependencies: {', '.join(self.missing_optional_dependencies)}")
        if self.failed_imports:
            lines.extend(["", "## Failed Imports", ""])
            lines.extend(f"- `{item}`" for item in self.failed_imports)
        if self.warnings:
            lines.extend(["", "## Warnings", ""])
            lines.extend(f"- {item}" for item in self.warnings)
        lines.extend(["", "## Passed Imports", ""])
        lines.extend(f"- `{item}`" for item in self.passed_imports)
        return "\n".join(lines) + "\n"


REQUIRED_IMPORTS: tuple[tuple[str, str], ...] = (
    ("crr", "Manifold"),
    ("crr", "Metric"),
    ("crr", "Tensor"),
    ("crr", "ParametrizedMap"),
    ("crr", "DifferentialForm"),
    ("crr.lie", "LieAlgebra"),
    ("crr.lie", "LieAlgebraElement"),
    ("crr.lie", "AlgebraValuedForm"),
    ("crr.gauge", "GaugePotential"),
    ("crr.gauge", "GaugeCurvature"),
    ("crr.presets", "sphere_surface"),
    ("crr.presets", "torus_surface"),
    ("crr.presets", "sphere_metric"),
    ("crr.presets", "schwarzschild_metric"),
    ("crr.presets", "kerr_metric"),
    ("crr.lie", "su2_algebra"),
    ("crr.visualization", "plot_scalar_field_2d"),
    ("crr.visualization", "plot_vector_field_2d"),
    ("crr.visualization", "plot_form_2d"),
)

OPTIONAL_IMPORTS: tuple[tuple[str, str], ...] = (
    ("matplotlib", "matplotlib"),
    ("IPython", "ipython"),
)

EXPECTED_METHOD_GROUPS: tuple[tuple[str, str, tuple[tuple[str, ...], ...]], ...] = (
    ("crr", "Manifold", (("__repr__",),)),
    ("crr", "Metric", (("__repr__",), ("scalar_curvature",), ("to_latex", "display", "display_matrix"))),
    ("crr", "Tensor", (("__repr__",), ("to_latex_components", "display_nonzero"))),
    ("crr", "ParametrizedMap", (("__repr__",), ("to_latex", "display", "display_parametrization"))),
    ("crr", "DifferentialForm", (("__repr__",), ("to_latex", "display"), ("simplify",))),
    ("crr.lie", "LieAlgebra", (("__repr__",), ("to_latex",))),
    ("crr.lie", "LieAlgebraElement", (("__repr__",), ("to_latex",), ("simplify",))),
    ("crr.lie", "AlgebraValuedForm", (("__repr__",), ("to_latex",), ("simplify",))),
    ("crr.gauge", "GaugePotential", (("__repr__",), ("to_latex",), ("simplify",))),
    ("crr.gauge", "GaugeCurvature", (("__repr__",), ("to_latex",), ("simplify",))),
)


def audit_public_api() -> ApiAuditReport:
    """Check that important public objects are importable and have expected methods."""

    passed: list[str] = []
    failed: list[str] = []
    missing_optional: list[str] = []
    warnings: list[str] = []

    for module_name, attr_name in REQUIRED_IMPORTS:
        label = f"{module_name}.{attr_name}"
        try:
            _import_attr(module_name, attr_name)
        except Exception as exc:  # pragma: no cover - exercised only on regressions
            failed.append(f"{label}: {exc}")
        else:
            passed.append(label)

    for module_name, dep_name in OPTIONAL_IMPORTS:
        try:
            import_module(module_name)
        except Exception:
            missing_optional.append(dep_name)

    for module_name, attr_name, method_groups in EXPECTED_METHOD_GROUPS:
        try:
            obj = _import_attr(module_name, attr_name)
        except Exception:
            continue
        for methods in method_groups:
            if not any(hasattr(obj, method) for method in methods):
                joined = " or ".join(methods)
                warnings.append(f"{module_name}.{attr_name} is missing expected method group: {joined}.")

    return ApiAuditReport(
        passed_imports=passed,
        failed_imports=failed,
        missing_optional_dependencies=sorted(set(missing_optional)),
        warnings=warnings,
    )


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = import_module(module_name)
    return getattr(module, attr_name)
