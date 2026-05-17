import subprocess
import sys
from pathlib import Path

from crr.audit import audit_public_api, capability_report_markdown, get_capabilities, run_benchmarks


ROOT = Path(__file__).resolve().parents[1]


def test_get_capabilities_returns_required_categories():
    capabilities = get_capabilities()
    categories = {capability.category for capability in capabilities}

    assert capabilities
    assert "Core geometry" in categories
    assert "Differential forms" in categories
    assert "Lie algebras" in categories
    assert "Gauge fields" in categories
    assert "Visualization" in categories


def test_capability_report_markdown_contains_key_sections():
    report = capability_report_markdown()

    assert isinstance(report, str)
    assert "# CRR Capability Report" in report
    assert "## Core geometry" in report
    assert "## Gauge fields" in report
    assert "## Visualization" in report


def test_audit_public_api_required_imports_pass():
    report = audit_public_api()

    assert report.failed_imports == []
    assert any(item == "crr.Manifold" for item in report.passed_imports)
    assert any(item == "crr.gauge.GaugePotential" for item in report.passed_imports)


def test_run_benchmarks_quick_smoke():
    results = run_benchmarks(quick=True)

    assert results
    assert all(result.success for result in results)
    assert {result.name for result in results} >= {
        "sphere_curvature",
        "forms_hodge_star_r3",
        "lie_su2_jacobi",
        "gauge_su2_curvature",
    }


def test_audit_examples_smoke():
    examples = [
        "examples/audit_capabilities.py",
        "examples/audit_api.py",
        "examples/crr_showcase.py",
    ]

    for example in examples:
        result = subprocess.run(
            [sys.executable, example],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
