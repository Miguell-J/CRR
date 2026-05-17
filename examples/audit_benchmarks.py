"""Run CRR's lightweight benchmark report."""

import _bootstrap  # noqa: F401

from crr.audit import benchmark_report_markdown


print(benchmark_report_markdown())
