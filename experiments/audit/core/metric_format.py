"""Display precision for audit metrics (matches log / JSON formatting)."""

from __future__ import annotations

AUDIT_METRIC_DECIMALS = 4
AUDIT_LOGIT_DECIMALS = 2
AUDIT_RESIDUAL_NORM_DECIMALS = 3


def round_audit_metric(value: float) -> float:
    return float(round(value, AUDIT_METRIC_DECIMALS))


def round_audit_logit(value: float) -> float:
    return float(round(value, AUDIT_LOGIT_DECIMALS))


def round_audit_residual_norm(value: float) -> float:
    return float(round(value, AUDIT_RESIDUAL_NORM_DECIMALS))
