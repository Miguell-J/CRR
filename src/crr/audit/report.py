"""Combined audit report helpers."""

from __future__ import annotations

from pathlib import Path

from crr.audit.api_check import audit_public_api
from crr.audit.benchmarks import benchmark_report_markdown
from crr.audit.capabilities import capability_report_markdown


def audit_report_markdown(include_benchmarks: bool = False) -> str:
    """Return a combined Markdown audit report."""

    sections = [capability_report_markdown(), audit_public_api().to_markdown()]
    if include_benchmarks:
        sections.append(benchmark_report_markdown())
    return "\n".join(section.rstrip() for section in sections) + "\n"


def save_audit_report(path, include_benchmarks: bool = False) -> Path:
    """Write a combined audit report to ``path``."""

    output_path = Path(path)
    output_path.write_text(audit_report_markdown(include_benchmarks=include_benchmarks), encoding="utf-8")
    return output_path
