"""Lightweight CRR benchmark cases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from traceback import format_exception_only

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class BenchmarkResult:
    """A single benchmark result."""

    name: str
    elapsed_seconds: float
    status: str
    success: bool
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison between a current benchmark and a saved baseline."""

    name: str
    baseline_seconds: float | None
    current_seconds: float
    ratio: float | None
    regression: bool
    success: bool
    error: str | None = None


def run_benchmarks(quick: bool = False) -> list[BenchmarkResult]:
    """Run lightweight benchmark cases.

    ``quick=True`` runs a subset intended for tests and smoke checks.
    """

    cases = [
        ("sphere_curvature", _bench_sphere_curvature),
        ("forms_hodge_star_r3", _bench_forms_hodge_star_r3),
        ("lie_su2_jacobi", _bench_lie_su2_jacobi),
        ("gauge_su2_curvature", _bench_gauge_su2_curvature),
    ]
    if not quick:
        cases.extend(
            [
                ("torus_curvature", _bench_torus_curvature),
                ("schwarzschild_ricci_representative", _bench_schwarzschild_ricci_representative),
                ("geodesic_sphere_numeric", _bench_geodesic_sphere_numeric),
            ]
        )

    results = []
    for name, func in cases:
        start = perf_counter()
        try:
            func()
        except Exception as exc:  # pragma: no cover - exercised only on benchmark failures
            elapsed = perf_counter() - start
            error = "".join(format_exception_only(type(exc), exc)).strip()
            results.append(BenchmarkResult(name, elapsed, _classify(elapsed), False, error))
        else:
            elapsed = perf_counter() - start
            results.append(BenchmarkResult(name, elapsed, _classify(elapsed), True))
    return results


def benchmark_report_markdown(quick: bool = False) -> str:
    """Run benchmarks and return a Markdown timing table."""

    results = run_benchmarks(quick=quick)
    lines = [
        "# CRR Benchmark Report",
        "",
        "| Benchmark | Seconds | Status | Success | Error |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for result in results:
        error = result.error or ""
        lines.append(
            f"| {result.name} | {result.elapsed_seconds:.4f} | {result.status} | {result.success} | {error} |"
        )
    return "\n".join(lines) + "\n"


def save_benchmark_baseline(path, quick: bool = False) -> Path:
    """Run benchmarks and save a JSON baseline."""

    output_path = Path(path)
    results = run_benchmarks(quick=quick)
    payload = {
        "version": 1,
        "quick": quick,
        "benchmarks": [_benchmark_to_dict(result) for result in results],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def load_benchmark_baseline(path) -> dict[str, BenchmarkResult]:
    """Load a benchmark baseline JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    benchmarks = payload.get("benchmarks", [])
    return {
        item["name"]: BenchmarkResult(
            name=item["name"],
            elapsed_seconds=float(item["elapsed_seconds"]),
            status=item.get("status", _classify(float(item["elapsed_seconds"]))),
            success=bool(item.get("success", True)),
            error=item.get("error"),
        )
        for item in benchmarks
    }


def compare_benchmarks_to_baseline(path, quick: bool = False, tolerance: float = 2.0) -> list[BenchmarkComparison]:
    """Run benchmarks and compare current timings against a saved baseline."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    baseline = load_benchmark_baseline(path)
    current = run_benchmarks(quick=quick)
    comparisons = []
    for result in current:
        base = baseline.get(result.name)
        baseline_seconds = None if base is None else base.elapsed_seconds
        ratio = None
        regression = False
        if baseline_seconds is not None and baseline_seconds > 0:
            ratio = result.elapsed_seconds / baseline_seconds
            regression = ratio > tolerance
        comparisons.append(
            BenchmarkComparison(
                name=result.name,
                baseline_seconds=baseline_seconds,
                current_seconds=result.elapsed_seconds,
                ratio=ratio,
                regression=regression,
                success=result.success,
                error=result.error,
            )
        )
    return comparisons


def benchmark_comparison_markdown(path, quick: bool = False, tolerance: float = 2.0) -> str:
    """Return a Markdown table comparing current timings to a baseline."""

    comparisons = compare_benchmarks_to_baseline(path, quick=quick, tolerance=tolerance)
    lines = [
        "# CRR Benchmark Baseline Comparison",
        "",
        "| Benchmark | Baseline | Current | Ratio | Regression | Success | Error |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in comparisons:
        baseline = "" if item.baseline_seconds is None else f"{item.baseline_seconds:.4f}"
        ratio = "" if item.ratio is None else f"{item.ratio:.2f}"
        error = item.error or ""
        lines.append(
            f"| {item.name} | {baseline} | {item.current_seconds:.4f} | {ratio} | {item.regression} | {item.success} | {error} |"
        )
    return "\n".join(lines) + "\n"


def _benchmark_to_dict(result: BenchmarkResult) -> dict[str, object]:
    return {
        "name": result.name,
        "elapsed_seconds": result.elapsed_seconds,
        "status": result.status,
        "success": result.success,
        "error": result.error,
    }


def _classify(elapsed: float) -> str:
    if elapsed < 0.5:
        return "fast"
    if elapsed < 5:
        return "moderate"
    return "slow"


def _bench_sphere_curvature() -> None:
    from crr.presets import sphere_metric

    metric = sphere_metric(radius=1)
    metric.scalar_curvature(simplify=True)


def _bench_torus_curvature() -> None:
    from crr.presets import torus_surface

    torus = torus_surface(major_radius=2, minor_radius=1)
    torus.pullback_metric(simplify=True).scalar_curvature(simplify=True)
    torus.gaussian_curvature_extrinsic(simplify=True)


def _bench_schwarzschild_ricci_representative() -> None:
    from crr.presets import schwarzschild_metric

    metric = schwarzschild_metric()
    ricci = metric.ricci_tensor(simplify=False)
    _ = ricci.components[0, 0]


def _bench_forms_hodge_star_r3() -> None:
    from crr import DifferentialForm
    from crr.presets import euclidean_metric

    x, y, z = sp.symbols("x y z")
    metric = euclidean_metric(dim=3, coordinates=[x, y, z])
    forms = [DifferentialForm.basis(metric.manifold, i) for i in range(3)]
    for form in forms:
        form.hodge_star(metric, simplify=True)


def _bench_lie_su2_jacobi() -> None:
    from crr.lie import su2_algebra

    su2_algebra().check_jacobi()


def _bench_gauge_su2_curvature() -> None:
    from crr import DifferentialForm, Manifold
    from crr.gauge import su2_connection

    x, y = sp.symbols("x y")
    manifold = Manifold("R2", 2, [x, y])
    dx = DifferentialForm.basis(manifold, 0)
    dy = DifferentialForm.basis(manifold, 1)
    su2_connection(manifold, [dx, dy, DifferentialForm.zero(manifold, 1)]).curvature()


def _bench_geodesic_sphere_numeric() -> None:
    from crr.presets import sphere_metric

    theta, phi = sp.symbols("theta phi")
    metric = sphere_metric(radius=1, coordinates=[theta, phi])
    solution = metric.solve_geodesic([np.pi / 2, 0], [0, 1], (0, 1), num_points=40)
    if not solution.success:
        raise RuntimeError("sphere geodesic solve failed")
