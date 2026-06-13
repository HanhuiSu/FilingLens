"""compute_metrics — pure-computation tool for financial ratios and growth rates."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    period: str = Field(description="Period label, e.g. '2024-09-30' or '2024Q3'")
    value: float = Field(description="Numeric value for this period")


class ComputeMetricsInput(BaseModel):
    """Input schema for the compute_metrics tool."""

    data: list[DataPoint] = Field(
        description=(
            "Time-ordered list of (period, value) pairs. "
            "Typically obtained from query_financial_data results. "
            "Must be sorted chronologically (oldest first)."
        ),
        min_length=1,
    )
    computation: Literal["yoy", "qoq", "margin", "growth", "cagr", "ratio", "difference", "valuation_multiple"] = Field(
        description=(
            "Type of computation: "
            "'yoy' = year-over-year change (needs >=5 quarterly or >=2 annual data points), "
            "'qoq' = quarter-over-quarter change (needs >=2 data points), "
            "'margin' = each value as ratio of the first value (use numerator/denominator data), "
            "'growth' = period-over-period percentage change, "
            "'cagr' = compound annual growth rate (needs >=2 data points with year span), "
            "'ratio' = each numerator divided by aligned denominator, "
            "'difference' = each left value minus aligned right value, "
            "'valuation_multiple' = ratio formatted as a valuation multiple"
        ),
    )
    denominator_data: list[DataPoint] | None = Field(
        default=None,
        description=(
            "Optional denominator/right-hand series for margin, ratio, difference, "
            "or valuation_multiple computations. When provided, result[i] is computed "
            "from data[i] and denominator_data[i]. "
            "Must have same length and period alignment as 'data'."
        ),
    )


def _pct(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return (a - b) / abs(b)


def _compute_yoy(points: list[DataPoint]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i in range(len(points)):
        if i < 4:
            continue
        prev = points[i - 4]
        cur = points[i]
        change = _pct(cur.value, prev.value)
        results.append(
            {
                "period": cur.period,
                "value": cur.value,
                "compare_period": prev.period,
                "compare_value": prev.value,
                "yoy_change": round(change, 6) if change is not None else None,
                "yoy_pct": f"{change * 100:.2f}%" if change is not None else "N/A",
            }
        )
    if not results and len(points) >= 2:
        prev, cur = points[0], points[-1]
        change = _pct(cur.value, prev.value)
        results.append(
            {
                "period": cur.period,
                "value": cur.value,
                "compare_period": prev.period,
                "compare_value": prev.value,
                "yoy_change": round(change, 6) if change is not None else None,
                "yoy_pct": f"{change * 100:.2f}%" if change is not None else "N/A",
                "note": "fallback: first vs last (not 4-quarter gap)",
            }
        )
    return results


def _compute_qoq(points: list[DataPoint]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i in range(1, len(points)):
        prev, cur = points[i - 1], points[i]
        change = _pct(cur.value, prev.value)
        results.append(
            {
                "period": cur.period,
                "value": cur.value,
                "prev_period": prev.period,
                "prev_value": prev.value,
                "qoq_change": round(change, 6) if change is not None else None,
                "qoq_pct": f"{change * 100:.2f}%" if change is not None else "N/A",
            }
        )
    return results


def _compute_margin(
    numerator: list[DataPoint],
    denominator: list[DataPoint] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if denominator and len(denominator) == len(numerator):
        for n, d in zip(numerator, denominator):
            ratio = n.value / d.value if d.value != 0 else None
            results.append(
                {
                    "period": n.period,
                    "numerator": n.value,
                    "denominator": d.value,
                    "margin": round(ratio, 6) if ratio is not None else None,
                    "margin_pct": f"{ratio * 100:.2f}%" if ratio is not None else "N/A",
                }
            )
    else:
        base = numerator[0].value if numerator[0].value != 0 else 1.0
        for p in numerator:
            ratio = p.value / base
            results.append(
                {
                    "period": p.period,
                    "value": p.value,
                    "base_value": base,
                    "ratio": round(ratio, 6),
                    "ratio_pct": f"{ratio * 100:.2f}%",
                }
            )
    return results


def _compute_ratio(
    numerator: list[DataPoint],
    denominator: list[DataPoint] | None,
    *,
    as_multiple: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not denominator or len(denominator) != len(numerator):
        return [{"error": "ratio computation requires aligned denominator_data"}]
    for n, d in zip(numerator, denominator):
        ratio = n.value / d.value if d.value != 0 else None
        item: dict[str, Any] = {
            "period": n.period,
            "numerator": n.value,
            "denominator": d.value,
            "ratio": round(ratio, 6) if ratio is not None else None,
            "ratio_pct": f"{ratio * 100:.2f}%" if ratio is not None else "N/A",
        }
        if as_multiple:
            item["multiple"] = round(ratio, 6) if ratio is not None else None
            item["multiple_label"] = f"{ratio:.2f}x" if ratio is not None else "N/A"
        results.append(item)
    return results


def _compute_difference(
    left: list[DataPoint],
    right: list[DataPoint] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not right or len(right) != len(left):
        return [{"error": "difference computation requires aligned denominator_data"}]
    for left_point, right_point in zip(left, right):
        results.append(
            {
                "period": left_point.period,
                "left_value": left_point.value,
                "right_value": right_point.value,
                "difference": left_point.value - right_point.value,
            }
        )
    return results


def _compute_growth(points: list[DataPoint]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for i in range(1, len(points)):
        prev, cur = points[i - 1], points[i]
        if str(cur.period) == str(prev.period):
            results.append(
                {
                    "period": cur.period,
                    "prev_period": prev.period,
                    "error": "same_period_comparator",
                }
            )
            continue
        change = _pct(cur.value, prev.value)
        if change is None:
            results.append(
                {
                    "period": cur.period,
                    "prev_period": prev.period,
                    "prev_value": prev.value,
                    "growth": None,
                    "growth_pct": "N/A",
                    "error": "zero_comparator",
                }
            )
            continue
        results.append(
            {
                "period": cur.period,
                "value": cur.value,
                "prev_period": prev.period,
                "prev_value": prev.value,
                "growth": round(change, 6),
                "growth_pct": f"{change * 100:.2f}%",
            }
        )
    return results


def _compute_cagr(points: list[DataPoint]) -> list[dict[str, Any]]:
    if len(points) < 2:
        return [{"error": "Need at least 2 data points for CAGR"}]
    first, last = points[0], points[-1]
    try:
        y0 = int(first.period[:4])
        yn = int(last.period[:4])
    except (ValueError, IndexError):
        return [{"error": f"Cannot parse year from periods: {first.period}, {last.period}"}]
    n = yn - y0
    if n <= 0:
        n = max(len(points) - 1, 1) / 4.0
    if first.value <= 0 or last.value <= 0:
        return [{"error": "CAGR requires positive start and end values", "start": first.value, "end": last.value}]
    cagr = (last.value / first.value) ** (1.0 / n) - 1
    return [
        {
            "start_period": first.period,
            "start_value": first.value,
            "end_period": last.period,
            "end_value": last.value,
            "years": round(n, 2),
            "cagr": round(cagr, 6),
            "cagr_pct": f"{cagr * 100:.2f}%",
        }
    ]


@tool("compute_metrics", args_schema=ComputeMetricsInput)
def compute_metrics(
    data: list[dict],
    computation: str,
    denominator_data: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute financial metrics from structured data.

    Accepts a time-ordered list of (period, value) pairs — typically
    the output of query_financial_data — and performs one of:
    yoy (year-over-year), qoq (quarter-over-quarter), margin,
    growth (period-over-period), cagr (compound annual growth rate),
    ratio, difference, or valuation_multiple.

    This is a pure computation tool: it does NOT access any database.
    Call query_financial_data first to obtain the input data.
    """
    points = [DataPoint(**d) if isinstance(d, dict) else d for d in data]
    denom = (
        [DataPoint(**d) if isinstance(d, dict) else d for d in denominator_data]
        if denominator_data
        else None
    )

    dispatch = {
        "yoy": lambda: _compute_yoy(points),
        "qoq": lambda: _compute_qoq(points),
        "margin": lambda: _compute_margin(points, denom),
        "ratio": lambda: _compute_ratio(points, denom),
        "difference": lambda: _compute_difference(points, denom),
        "valuation_multiple": lambda: _compute_ratio(points, denom, as_multiple=True),
        "growth": lambda: _compute_growth(points),
        "cagr": lambda: _compute_cagr(points),
    }
    fn = dispatch.get(computation)
    if fn is None:
        return {"error": f"Unknown computation: {computation}. Choose from {list(dispatch.keys())}"}

    return {
        "computation": computation,
        "input_count": len(points),
        "results": fn(),
    }
