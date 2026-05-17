"""Audit, capability reporting, API checks, and lightweight benchmarks."""

from crr.audit.api_check import ApiAuditReport, audit_public_api
from crr.audit.benchmarks import (
    BenchmarkComparison,
    BenchmarkResult,
    benchmark_comparison_markdown,
    benchmark_report_markdown,
    compare_benchmarks_to_baseline,
    load_benchmark_baseline,
    run_benchmarks,
    save_benchmark_baseline,
)
from crr.audit.capabilities import (
    Capability,
    capability_report_markdown,
    get_capabilities,
    print_capability_report,
    save_capability_report,
)
from crr.audit.report import audit_report_markdown, save_audit_report

__all__ = [
    "Capability",
    "ApiAuditReport",
    "BenchmarkResult",
    "BenchmarkComparison",
    "get_capabilities",
    "print_capability_report",
    "capability_report_markdown",
    "save_capability_report",
    "audit_public_api",
    "run_benchmarks",
    "benchmark_report_markdown",
    "save_benchmark_baseline",
    "load_benchmark_baseline",
    "compare_benchmarks_to_baseline",
    "benchmark_comparison_markdown",
    "audit_report_markdown",
    "save_audit_report",
]
