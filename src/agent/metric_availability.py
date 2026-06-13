"""Shared metric availability and derived-metric dependency rules.

This module is intentionally small and deterministic.  It is used by both
offline coverage reports and runtime sufficiency so they do not drift on
metric aliases or derived valuation availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import duckdb


METRIC_ALIASES: dict[str, str] = {
    "adjusted_close": "share_price",
    "close": "share_price",
    "latest_close": "share_price",
    "price": "share_price",
    "share_price": "share_price",
    "cash": "cash",
    "cash_and_equivalents": "cash",
    "cash_equivalents": "cash",
    "cash_and_cash_equivalents": "cash",
    "capex": "capital_expenditure",
    "capital_expenditures": "capital_expenditure",
    "capital_expenditure": "capital_expenditure",
    "shares": "shares_outstanding",
    "shares_outstanding": "shares_outstanding",
    "cfo_to_net_income": "cash_conversion",
}

CANONICAL_TO_RAW_METRICS: dict[str, tuple[str, ...]] = {
    "cash": ("cash_and_equivalents", "cash"),
    "share_price": ("adjusted_close", "close", "price", "share_price", "latest_close"),
    "capital_expenditure": ("capital_expenditure", "capex"),
    "shares_outstanding": ("shares_outstanding", "shares"),
    "cash_conversion": ("cash_conversion", "cfo_to_net_income"),
}

DERIVED_METRIC_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "total_debt": ("short_term_debt", "long_term_debt"),
    "market_cap": ("share_price", "shares_outstanding"),
    "pe_ratio": ("market_cap", "net_income"),
    "ps_ratio": ("market_cap", "revenue"),
    "fcf_yield": ("free_cash_flow", "market_cap"),
}

DIMENSION_CORE_METRICS: dict[str, tuple[str, ...]] = {
    "revenue_quality": ("revenue",),
    "profitability_quality": ("net_income", "net_margin"),
    "cash_flow_quality": ("operating_cash_flow", "free_cash_flow"),
    "balance_sheet_and_capital_intensity": ("cash", "total_debt", "capital_expenditure"),
    "valuation_and_risk_boundary": ("share_price", "shares_outstanding", "market_cap", "pe_ratio", "ps_ratio"),
}

DIMENSION_ENHANCED_METRICS: dict[str, tuple[str, ...]] = {
    "revenue_quality": ("revenue_growth",),
    "profitability_quality": ("gross_profit", "operating_income", "gross_margin", "operating_margin"),
    "cash_flow_quality": ("cash_conversion", "fcf_margin"),
    "balance_sheet_and_capital_intensity": (
        "total_assets",
        "total_liabilities",
        "shareholders_equity",
        "inventory",
        "receivables",
        "net_debt",
        "debt_to_equity",
        "capex_to_revenue",
    ),
    "valuation_and_risk_boundary": ("fcf_yield",),
}


def normalize_metric_name(metric: str | None) -> str:
    """Return the canonical metric name used by methodology contracts."""
    raw = str(metric or "").strip().lower().replace("-", "_").replace(" ", "_")
    return METRIC_ALIASES.get(raw, raw)


def raw_metric_names(metric: str | None) -> tuple[str, ...]:
    canonical = normalize_metric_name(metric)
    return CANONICAL_TO_RAW_METRICS.get(canonical, (canonical,))


def canonical_metric_set(metrics: Iterable[str | None]) -> set[str]:
    return {normalize_metric_name(metric) for metric in metrics if str(metric or "").strip()}


@dataclass
class MetricAvailability:
    metric: str
    available: bool
    row: dict[str, Any] | None = None
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    missing_dependencies: list[str] = field(default_factory=list)
    source: str = "direct"

    def source_row(self) -> dict[str, Any]:
        if self.row:
            out = dict(self.row)
        else:
            out = {
                "metric": self.metric,
                "period_end": "",
                "period_type": "computed",
                "source_provider": "computed",
                "confidence": "derived",
                "extraction_method": "programmatic_dependency_check",
                "row_count": 1,
            }
        out["metric"] = self.metric
        if self.dependencies:
            out["dependencies"] = [dict(dep) for dep in self.dependencies]
        if self.missing_dependencies:
            out["missing_dependencies"] = list(self.missing_dependencies)
        return out


class MetricAvailabilityResolver:
    """Resolve direct and derived metric availability from rows or DuckDB."""

    def __init__(self, rows: Iterable[Mapping[str, Any]] | None = None):
        self._rows_by_metric: dict[str, list[dict[str, Any]]] = {}
        for row in rows or []:
            item = dict(row)
            canonical = normalize_metric_name(str(item.get("metric") or ""))
            if not canonical:
                continue
            item["canonical_metric"] = canonical
            self._rows_by_metric.setdefault(canonical, []).append(item)
        for metric, metric_rows in list(self._rows_by_metric.items()):
            self._rows_by_metric[metric] = sorted(metric_rows, key=_row_sort_key, reverse=True)

    @classmethod
    def from_duckdb(
        cls,
        conn: duckdb.DuckDBPyConnection,
        ticker: str,
        metrics: Iterable[str],
    ) -> "MetricAvailabilityResolver":
        ticker = ticker.upper()
        wanted = _expand_metric_dependencies(canonical_metric_set(metrics))
        rows: list[dict[str, Any]] = []
        fact_metrics = sorted({raw for metric in wanted if metric != "share_price" for raw in raw_metric_names(metric)})
        if fact_metrics:
            placeholders = ", ".join(["?"] * len(fact_metrics))
            fact_rows = conn.execute(
                f"""
                SELECT metric, period_end, period_type, value, unit,
                       COALESCE(source_provider, 'unknown') AS source_provider,
                       COALESCE(confidence, '') AS confidence,
                       COALESCE(extraction_method, '') AS extraction_method,
                       COALESCE(source_tag, '') AS source_tag,
                       COALESCE(reconciliation_warning, '') AS reconciliation_warning,
                       COUNT(*) AS row_count
                FROM financial_facts
                WHERE ticker = ? AND metric IN ({placeholders}) AND value IS NOT NULL
                GROUP BY metric, period_end, period_type, value, unit, source_provider,
                         confidence, extraction_method, source_tag, reconciliation_warning
                ORDER BY period_end DESC
                """,
                [ticker, *fact_metrics],
            ).fetchall()
            rows.extend(
                {
                    "metric": str(row[0]),
                    "period_end": str(row[1]),
                    "period_type": str(row[2]),
                    "value": row[3],
                    "unit": str(row[4] or ""),
                    "source_provider": str(row[5] or ""),
                    "confidence": str(row[6] or ""),
                    "extraction_method": str(row[7] or ""),
                    "source_tag": str(row[8] or ""),
                    "reconciliation_warning": str(row[9] or ""),
                    "row_count": int(row[10] or 0),
                    "ticker": ticker,
                }
                for row in fact_rows
            )
        if "share_price" in wanted:
            price = conn.execute(
                """
                SELECT date, adjusted_close, close
                FROM price_history
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
                """,
                [ticker],
            ).fetchone()
            if price:
                value = price[1] if price[1] is not None else price[2]
                if value is not None:
                    rows.append(
                        {
                            "metric": "share_price",
                            "period_end": str(price[0]),
                            "period_type": "daily",
                            "value": value,
                            "unit": "USD",
                            "source_provider": "yfinance",
                            "confidence": "medium",
                            "extraction_method": "api_price_history",
                            "source_tag": "latest_adjusted_close",
                            "reconciliation_warning": "",
                            "row_count": 1,
                            "ticker": ticker,
                        }
                    )
        return cls(rows)

    def availability(self, metric: str, *, _seen: set[str] | None = None) -> MetricAvailability:
        canonical = normalize_metric_name(metric)
        direct = self._rows_by_metric.get(canonical) or []
        if direct:
            return MetricAvailability(metric=canonical, available=True, row=dict(direct[0]), source="direct")

        seen = set(_seen or set())
        if canonical in seen:
            return MetricAvailability(metric=canonical, available=False, missing_dependencies=[canonical])
        seen.add(canonical)
        dependencies = DERIVED_METRIC_DEPENDENCIES.get(canonical)
        if not dependencies:
            return MetricAvailability(metric=canonical, available=False, missing_dependencies=[canonical])

        dep_records: list[dict[str, Any]] = []
        missing: list[str] = []
        latest_period = ""
        for dep in dependencies:
            dep_availability = self.availability(dep, _seen=seen)
            if not dep_availability.available:
                missing.extend(dep_availability.missing_dependencies or [normalize_metric_name(dep)])
                continue
            source_row = dep_availability.source_row()
            source_row["dependency_metric"] = normalize_metric_name(dep)
            dep_records.append(source_row)
            period = str(source_row.get("period_end") or "")
            if period > latest_period:
                latest_period = period
        if missing:
            return MetricAvailability(
                metric=canonical,
                available=False,
                dependencies=dep_records,
                missing_dependencies=sorted(set(missing)),
                source="derived",
            )
        return MetricAvailability(
            metric=canonical,
            available=True,
            row={
                "metric": canonical,
                "period_end": latest_period,
                "period_type": "computed",
                "source_provider": "computed",
                "confidence": "derived",
                "extraction_method": "programmatic_dependency_check",
                "row_count": 1,
                "dependencies": dep_records,
                "reconciliation_warning": _dependency_warning(dep_records),
            },
            dependencies=dep_records,
            source="derived",
        )

    def available_metrics(self, metrics: Iterable[str]) -> list[str]:
        return [normalize_metric_name(metric) for metric in metrics if self.availability(metric).available]

    def missing_metrics(self, metrics: Iterable[str]) -> list[str]:
        return [normalize_metric_name(metric) for metric in metrics if not self.availability(metric).available]

    def source_rows(self, metrics: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for metric in metrics:
            canonical = normalize_metric_name(metric)
            availability = self.availability(canonical)
            if availability.available:
                out[canonical] = [availability.source_row()]
        return out


def _row_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    provider = str(row.get("source_provider") or "").lower()
    provider_rank = 0 if provider == "sec_companyfacts" else 1 if provider == "yfinance" else 2
    return (
        -provider_rank,
        str(row.get("period_end") or row.get("period") or ""),
        str(row.get("filing_date") or ""),
        str(row.get("source_tag") or ""),
    )


def _expand_metric_dependencies(metrics: set[str]) -> set[str]:
    expanded = set(metrics)
    changed = True
    while changed:
        changed = False
        for metric in list(expanded):
            for dep in DERIVED_METRIC_DEPENDENCIES.get(metric, ()):
                if dep not in expanded:
                    expanded.add(dep)
                    changed = True
    return expanded


def _dependency_warning(rows: list[Mapping[str, Any]]) -> str:
    warnings = [
        str(row.get("reconciliation_warning") or "").strip()
        for row in rows
        if str(row.get("reconciliation_warning") or "").strip()
    ]
    return ";".join(dict.fromkeys(warnings))
