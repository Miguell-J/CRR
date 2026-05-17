"""Run CRR's public API audit."""

import _bootstrap  # noqa: F401

from crr.audit import audit_public_api


report = audit_public_api()
print(report.to_markdown())
raise SystemExit(0 if report.success else 1)
