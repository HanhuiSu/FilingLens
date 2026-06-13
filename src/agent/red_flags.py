"""Methodology-v1 red flag and evidence-boundary diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True)
class RedFlag:
    id: str
    severity: Literal["low", "medium", "high"]
    category: str
    message: str
    evidence_refs: list[str]
    user_visible: bool

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _dimension_status(dimension_status_map: Mapping[str, Any], dimension_id: str) -> str:
    item = dimension_status_map.get(dimension_id, {})
    if isinstance(item, Mapping):
        return str(item.get("status") or "")
    return ""


def _numeric_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in packet.get("numeric_table", []) or [] if isinstance(row, Mapping)]


def _refs(rows: list[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for row in rows:
        ref = str(row.get("evidence_id") or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _latest_by_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("metric") or "") != metric:
            continue
        ticker = str(row.get("ticker") or "").upper().strip()
        if not ticker:
            continue
        current = latest.get(ticker)
        period = str(row.get("period_end") or "")
        if current is None or period >= str(current.get("period_end") or ""):
            latest[ticker] = row
    return latest


def _advantage_flag(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    threshold: float,
    flag_id: str,
    category: str,
    message_template: str,
) -> RedFlag | None:
    latest = _latest_by_metric(rows, metric)
    values = [
        (ticker, _to_float(row.get("value")), row)
        for ticker, row in latest.items()
        if _to_float(row.get("value")) is not None
    ]
    if len(values) < 2:
        return None
    values.sort(key=lambda item: float(item[1] or 0), reverse=True)
    winner, top_value, top_row = values[0]
    _, second_value, second_row = values[1]
    if top_value is None or second_value is None:
        return None
    if metric == "net_margin":
        advantage = top_value - second_value
    else:
        if second_value == 0:
            return None
        advantage = (top_value - second_value) / abs(second_value)
    if advantage < threshold:
        return None
    return RedFlag(
        id=flag_id,
        severity="low",
        category=category,
        message=message_template.format(ticker=winner),
        evidence_refs=_refs([top_row, second_row]),
        user_visible=True,
    )


def detect_red_flags(evidence_packet: Mapping[str, Any], dimension_status_map: Mapping[str, Any]) -> list[RedFlag]:
    packet = dict(evidence_packet or {})
    rows = _numeric_rows(packet)
    flags: list[RedFlag] = []

    if _dimension_status(dimension_status_map, "cash_flow_quality") == "missing":
        flags.append(
            RedFlag(
                id="missing_cash_flow_evidence",
                severity="medium",
                category="missing_evidence",
                message="当前缺少经营现金流/自由现金流证据，不能判断利润现金含量。",
                evidence_refs=[],
                user_visible=True,
            )
        )
    if _dimension_status(dimension_status_map, "valuation_and_risk_boundary") == "missing":
        flags.append(
            RedFlag(
                id="missing_valuation_evidence",
                severity="medium",
                category="missing_evidence",
                message="当前缺少估值证据，不能判断价格是否便宜或昂贵。",
                evidence_refs=[],
                user_visible=True,
            )
        )
    if _dimension_status(dimension_status_map, "balance_sheet_and_capital_intensity") == "missing":
        flags.append(
            RedFlag(
                id="missing_balance_sheet_evidence",
                severity="medium",
                category="missing_evidence",
                message="当前缺少现金/债务/资本开支证据，不能判断抗风险能力。",
                evidence_refs=[],
                user_visible=True,
            )
        )
    if _dimension_status(dimension_status_map, "moat_and_competitive_risk") == "missing":
        flags.append(
            RedFlag(
                id="text_risk_evidence_missing",
                severity="medium",
                category="missing_evidence",
                message="缺少可验证风险文本证据，不能做具体风险判断。",
                evidence_refs=[],
                user_visible=True,
            )
        )

    metrics = {str(row.get("metric") or "") for row in rows}
    profitability_status = _dimension_status(dimension_status_map, "profitability_quality")
    if profitability_status in {"satisfied", "partial"} and {"revenue", "net_income", "net_margin"}.issubset(metrics):
        if not ({"gross_margin", "operating_margin"} & metrics):
            profit_rows = [row for row in rows if str(row.get("metric") or "") in {"revenue", "net_income", "net_margin"}]
            flags.append(
                RedFlag(
                    id="numeric_only_profitability",
                    severity="low",
                    category="evidence_boundary",
                    message="限制：盈利判断主要基于净利率，缺少毛利率/营业利润率证据。",
                    evidence_refs=_refs(profit_rows),
                    user_visible=True,
                )
            )

    yfinance_rows = [row for row in rows if str(row.get("source_provider") or "").lower() == "yfinance"]
    if yfinance_rows:
        flags.append(
            RedFlag(
                id="yfinance_fallback_provider",
                severity="medium",
                category="data_provenance",
                message="部分结构化财务数据来自 yfinance fallback，可信度为 medium。",
                evidence_refs=_refs(yfinance_rows),
                user_visible=True,
            )
        )

    scale = _advantage_flag(
        rows,
        metric="revenue",
        threshold=0.20,
        flag_id="revenue_scale_advantage",
        category="advantage_signal",
        message_template="{ticker} 在收入规模上占优。",
    )
    if scale is not None:
        flags.append(scale)
    margin = _advantage_flag(
        rows,
        metric="net_margin",
        threshold=0.05,
        flag_id="profitability_margin_advantage",
        category="advantage_signal",
        message_template="{ticker} 在盈利转化能力上占优。",
    )
    if margin is not None:
        flags.append(margin)

    deduped: list[RedFlag] = []
    seen: set[str] = set()
    for flag in flags:
        if flag.id in seen:
            continue
        seen.add(flag.id)
        deduped.append(flag)
    return deduped


def serialize_red_flags(flags: list[RedFlag]) -> list[dict[str, Any]]:
    return [flag.model_dump() for flag in flags]


def user_visible_red_flags(flags: list[RedFlag] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for raw in flags:
        item = raw.model_dump() if isinstance(raw, RedFlag) else dict(raw)
        if not bool(item.get("user_visible", True)):
            continue
        visible.append(
            {
                "severity": str(item.get("severity", "")),
                "category": str(item.get("category", "")),
                "message": str(item.get("message", "")),
                "evidence_refs": list(item.get("evidence_refs", []) or []),
            }
        )
    return visible
