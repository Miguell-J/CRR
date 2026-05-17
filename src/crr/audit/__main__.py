"""Command line entry point for ``python -m crr.audit``."""

from __future__ import annotations

import argparse

from crr.audit.api_check import audit_public_api
from crr.audit.benchmarks import benchmark_comparison_markdown, benchmark_report_markdown, save_benchmark_baseline
from crr.audit.capabilities import capability_report_markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crr.audit")
    parser.add_argument("command", choices=["capabilities", "benchmarks", "api"])
    parser.add_argument("--quick", action="store_true", help="Run a quick benchmark subset.")
    parser.add_argument("--save-baseline", help="Save benchmark results to this JSON path.")
    parser.add_argument("--compare", help="Compare benchmark results to this JSON baseline path.")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Regression ratio threshold for comparisons.")
    args = parser.parse_args(argv)

    if args.command == "capabilities":
        print(capability_report_markdown())
        return 0
    if args.command == "benchmarks":
        if args.save_baseline:
            save_benchmark_baseline(args.save_baseline, quick=args.quick)
            print(f"Saved benchmark baseline to {args.save_baseline}")
            return 0
        if args.compare:
            print(benchmark_comparison_markdown(args.compare, quick=args.quick, tolerance=args.tolerance))
            return 0
        print(benchmark_report_markdown(quick=args.quick))
        return 0
    if args.command == "api":
        report = audit_public_api()
        print(report.to_markdown())
        return 0 if report.success else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
