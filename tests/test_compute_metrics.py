"""Tests for deterministic metric computation guards."""

from src.tools.compute_metrics import compute_metrics


def test_growth_same_period_comparator_does_not_emit_zero_growth():
    out = compute_metrics.invoke(
        {
            "data": [
                {"period": "2026-01-31", "value": 68_127_000_000},
                {"period": "2026-01-31", "value": 68_127_000_000},
            ],
            "computation": "growth",
        }
    )

    assert out["results"][0]["error"] == "same_period_comparator"
    assert "growth" not in out["results"][0]
    assert "growth_pct" not in out["results"][0]


def test_growth_zero_comparator_is_invalid_not_zero_percent():
    out = compute_metrics.invoke(
        {
            "data": [
                {"period": "2025-01-31", "value": 0},
                {"period": "2026-01-31", "value": 68_127_000_000},
            ],
            "computation": "growth",
        }
    )

    assert out["results"][0]["error"] == "zero_comparator"
    assert out["results"][0]["growth"] is None
    assert out["results"][0]["growth_pct"] == "N/A"
