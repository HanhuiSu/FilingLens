"""Programmatic comparison judgment framing from validated evidence."""

from __future__ import annotations

from typing import Any

from src.agent.metric_availability import normalize_metric_name
from src.agent.metric_display import format_metric_value
from src.agent.types import ComparisonJudgmentFrame


def _latest_metric_rows(packet: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for source in (
        packet.get("numeric_table", []),
        packet.get("comparison_table", []),
        packet.get("numeric_evidence", []),
        packet.get("computed_metrics", []),
    ):
        for row in source or []:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker", "") or row.get("company", "")).strip().upper()
            metric = normalize_metric_name(str(row.get("metric", "")).strip())
            if not ticker or not metric:
                continue
            key = (ticker, metric)
            current = latest.get(key)
            row_period = str(row.get("period_end", "") or row.get("as_of", "") or row.get("period", ""))
            current_period = str((current or {}).get("period_end", "") or (current or {}).get("as_of", "") or (current or {}).get("period", ""))
            if current is None or row_period >= current_period:
                latest[key] = dict(row, metric=metric, ticker=ticker)
    return latest


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _refs(*rows: dict[str, Any] | None) -> list[str]:
    refs: list[str] = []
    for row in rows:
        if not row:
            continue
        ref = str(row.get("evidence_id", "")).strip()
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _display_value(row: dict[str, Any] | None, fallback: float | None = None) -> str:
    row = row or {}
    display = str(row.get("display_value", "")).strip()
    if display:
        return display
    value = _to_float(row.get("value"))
    if value is None:
        value = fallback
    if value is None:
        return ""
    metric = str(row.get("metric", ""))
    unit = str(row.get("unit", ""))
    if metric:
        return format_metric_value(metric, value, unit)
    if unit.lower() == "ratio":
        return f"{value * 100:.2f}%"
    if unit.upper() in {"USD", "$", "USD/SHARE"}:
        return f"{value:,.0f} USD" if float(value).is_integer() else f"{value:,.2f} USD"
    return f"{value:g}"


def _metric_display(
    metric_rows: dict[tuple[str, str], dict[str, Any]],
    company: str,
    metric: str,
    fallback: float | None = None,
) -> str:
    return _display_value(metric_rows.get((company, metric)), fallback)


def _winner(metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str], metric: str) -> str:
    best_company = ""
    best_value: float | None = None
    tied = False
    for company in companies:
        value = _to_float((metric_rows.get((company, metric), {}) or {}).get("value"))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_company = company
            tied = False
        elif best_value is not None and value == best_value:
            tied = True
    if tied:
        return "mixed"
    return best_company or "unavailable"


def _margin_value(
    metric_rows: dict[tuple[str, str], dict[str, Any]],
    company: str,
) -> tuple[float | None, list[str], str]:
    net_margin_row = metric_rows.get((company, "net_margin"), {}) or {}
    net_margin = _to_float(net_margin_row.get("value"))
    if net_margin is not None:
        return net_margin, _refs(net_margin_row), "computed net_margin"

    operating_margin_row = metric_rows.get((company, "operating_margin"), {}) or {}
    operating_margin = _to_float(operating_margin_row.get("value"))
    if operating_margin is not None:
        return operating_margin, _refs(operating_margin_row), "operating margin"

    revenue_row = metric_rows.get((company, "revenue"), {}) or {}
    net_income_row = metric_rows.get((company, "net_income"), {}) or {}
    revenue = _to_float(revenue_row.get("value"))
    net_income = _to_float(net_income_row.get("value"))
    if revenue is None or net_income is None or revenue == 0:
        return None, [], "unavailable"
    return net_income / revenue, _refs(net_income_row, revenue_row), "programmatic net income / revenue"


def _winner_from_values(values: dict[str, float | None]) -> str:
    best_company = ""
    best_value: float | None = None
    tied = False
    for company, value in values.items():
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_company = company
            tied = False
        elif best_value is not None and value == best_value:
            tied = True
    if tied:
        return "mixed"
    return best_company or "unavailable"


def _lowest_from_values(values: dict[str, float | None]) -> str:
    best_company = ""
    best_value: float | None = None
    tied = False
    for company, value in values.items():
        if value is None:
            continue
        if best_value is None or value < best_value:
            best_value = value
            best_company = company
            tied = False
        elif best_value is not None and value == best_value:
            tied = True
    if tied:
        return "mixed"
    return best_company or "unavailable"


def _focus_dimensions(packet: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for source in (
        packet.get("active_dimensions"),
        packet.get("requested_dimensions"),
        dict(packet.get("canonical_intent", {}) or {}).get("requested_dimensions"),
    ):
        for item in source or []:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
    status_map = dict(packet.get("dimension_status_map", {}) or {})
    for item in status_map.keys():
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _requested_dimensions(packet: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for source in (
        packet.get("requested_dimensions"),
        dict(packet.get("canonical_intent", {}) or {}).get("requested_dimensions"),
    ):
        for item in source or []:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
    return out


def _primary_dimension(packet: dict[str, Any], focus_dimensions: list[str]) -> str:
    for source in (
        packet.get("primary_dimension"),
        dict(packet.get("analysis_plan", {}) or {}).get("primary_dimension"),
        dict(packet.get("evidence_policy", {}) or {}).get("primary_dimension"),
        dict(packet.get("canonical_intent", {}) or {}).get("primary_dimension"),
    ):
        text = str(source or "").strip()
        if text:
            return text
    query = str(packet.get("user_query") or packet.get("query") or "").lower()
    if "valuation_and_risk_boundary" in focus_dimensions and any(term in query for term in ("估值", "valuation", "p/e", "p/s", "fcf yield")):
        return "valuation_and_risk_boundary"
    if "cash_flow_quality" in focus_dimensions and any(term in query for term in ("现金流", "自由现金流", "cash flow", "fcf")):
        return "cash_flow_quality"
    if "revenue_quality" in focus_dimensions and any(term in query for term in ("增长质量", "收入质量", "营收质量", "growth quality", "revenue quality")):
        return "revenue_quality"
    if "moat_and_competitive_risk" in focus_dimensions and any(term in query for term in ("哪个更危险", "风险更大", "谁的风险", "riskier", "greater risk")):
        return "moat_and_competitive_risk"
    return focus_dimensions[0] if len(focus_dimensions) == 1 else ""


def _metric_values(metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str], metric: str) -> dict[str, float | None]:
    return {company: _to_float((metric_rows.get((company, metric), {}) or {}).get("value")) for company in companies}


def _refs_for_metric(metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str], *metrics: str) -> list[str]:
    return list(
        dict.fromkeys(
            ref
            for company in companies
            for metric in metrics
            for ref in _refs(metric_rows.get((company, metric)))
        )
    )


def _all_refs_for_metric(packet: dict[str, Any], companies: list[str], *metrics: str) -> list[str]:
    wanted = {normalize_metric_name(metric) for metric in metrics if metric}
    company_set = {company.upper() for company in companies}
    refs: list[str] = []
    for source in (
        packet.get("numeric_table", []),
        packet.get("comparison_table", []),
        packet.get("numeric_evidence", []),
        packet.get("computed_metrics", []),
    ):
        for row in source or []:
            if not isinstance(row, dict):
                continue
            company = str(row.get("ticker") or row.get("company") or "").upper().strip()
            metric = normalize_metric_name(str(row.get("metric") or ""))
            ref = str(row.get("evidence_id") or "").strip()
            if company in company_set and metric in wanted and ref and ref not in refs:
                refs.append(ref)
    return refs


def _metric_period_counts(packet: dict[str, Any], companies: list[str], metric: str) -> dict[str, int]:
    target = normalize_metric_name(metric)
    company_set = {company.upper() for company in companies}
    periods: dict[str, set[str]] = {company: set() for company in company_set}
    for source in (
        packet.get("numeric_table", []),
        packet.get("comparison_table", []),
        packet.get("numeric_evidence", []),
        packet.get("computed_metrics", []),
    ):
        for row in source or []:
            if not isinstance(row, dict):
                continue
            company = str(row.get("ticker") or row.get("company") or "").upper().strip()
            if company not in company_set or normalize_metric_name(str(row.get("metric") or "")) != target:
                continue
            period = str(row.get("period_end") or row.get("as_of") or row.get("period") or row.get("fiscal_period") or "").strip()
            ref = str(row.get("evidence_id") or row.get("display_value") or row.get("value") or "").strip()
            if period or ref:
                periods.setdefault(company, set()).add(period or ref)
    return {company: len(values) for company, values in periods.items()}


def _growth_winner(metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str]) -> str:
    growth_metrics = ("revenue_growth", "growth", "revenue_growth_rate")
    available = False
    best_company = ""
    best_value: float | None = None
    tied = False
    for company in companies:
        for metric in growth_metrics:
            value = _to_float((metric_rows.get((company, metric), {}) or {}).get("value"))
            if value is None:
                continue
            available = True
            if best_value is None or value > best_value:
                best_value = value
                best_company = company
                tied = False
            elif best_value is not None and value == best_value:
                tied = True
            break
    if not available:
        return "unavailable"
    if tied:
        return "mixed"
    return best_company or "unavailable"


def _risk_considerations(packet: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for theme in packet.get("grouped_risk_themes", []) or []:
        if not isinstance(theme, dict):
            continue
        companies = [str(x).upper() for x in theme.get("companies", []) or [] if str(x).strip()]
        if not companies:
            continue
        items.append(
            {
                "companies": companies,
                "theme_code": str(theme.get("theme_code", "")),
                "label": str(theme.get("label", "")),
                "evidence_refs": [str(x) for x in theme.get("evidence_refs", []) or [] if str(x).strip()],
            }
        )
    if items:
        return items[:4]

    text_by_company: dict[str, list[str]] = {}
    def iter_text_rows(source: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(source, dict):
            nested = source.get("items") or source.get("evidence") or source.get("validated_evidence") or []
            if nested:
                return iter_text_rows(nested)
            return [dict(source)]
        if isinstance(source, list):
            for item in source:
                rows.extend(iter_text_rows(item))
        return rows

    for source in (
        packet.get("text_snippets", []),
        packet.get("text_evidence", []),
        packet.get("validated_text_evidence", []),
        packet.get("validated_evidence", []),
        packet.get("evidence_collection_results", []),
    ):
        for snippet in iter_text_rows(source):
            if not (
                snippet.get("claim")
                or snippet.get("supporting_snippet")
                or snippet.get("text_snippet")
                or snippet.get("evidence_summary")
            ):
                continue
            ref = str(snippet.get("evidence_id") or snippet.get("id") or "").strip()
            company = str(snippet.get("ticker") or snippet.get("company") or "").strip().upper()
            dimension = str(snippet.get("dimension_id") or "").strip()
            section = str(snippet.get("section") or "").upper().strip()
            if not ref or not company:
                continue
            if dimension != "moat_and_competitive_risk" and section not in {"ITEM_1A", "ITEM_7", "ITEM_2"}:
                continue
            text_by_company.setdefault(company, [])
            if ref not in text_by_company[company]:
                text_by_company[company].append(ref)
    if text_by_company:
        return [
            {
                "companies": [company],
                "theme_code": "validated_filing_text",
                "label": "Validated risk text",
                "evidence_refs": refs[:2],
            }
            for company, refs in text_by_company.items()
        ]
    return []


def _limitations(packet: dict[str, Any]) -> list[str]:
    requested = set(_requested_dimensions(packet))
    limitations: list[str] = []
    for item in packet.get("limitations", []) or []:
        if not isinstance(item, dict):
            continue
        dimension_id = str(item.get("dimension_id") or "").strip()
        if requested and dimension_id and dimension_id not in requested:
            continue
        message = str(item.get("message", "")).strip()
        if message:
            limitations.append(message)
    if (not requested or "moat_and_competitive_risk" in requested) and not packet.get("text_snippets", []):
        limitations.append("Risk comparison is limited because validated filing text evidence is not available.")
    return list(dict.fromkeys(limitations))


def _risk_tradeoff(packet: dict[str, Any], risk_considerations: list[dict[str, Any]]) -> str:
    if risk_considerations:
        first = risk_considerations[0]
        companies = ", ".join(str(x) for x in first.get("companies", []) or [])
        label = str(first.get("label", "") or first.get("theme_code", "risk evidence"))
        if str(first.get("theme_code", "")) == "validated_filing_text":
            return f"Validated filing text for {companies or 'the compared companies'} provides risk context, but it does not support a strong asymmetric risk conclusion."
        return f"Validated text evidence highlights {label} for {companies}, so the preference should be read alongside those filing risks."
    return "Risk comparison is limited because validated filing text evidence is not available."


def _risk_comparison_frame(packet: dict[str, Any], companies: list[str]) -> ComparisonJudgmentFrame:
    risk_considerations = _risk_considerations(packet)
    refs = list(
        dict.fromkeys(
            str(ref)
            for item in risk_considerations
            for ref in item.get("evidence_refs", []) or []
            if str(ref).strip()
        )
    )
    has_risk_text = bool(refs)
    return ComparisonJudgmentFrame(
        companies=companies,
        preferred_company="no_clear_preference",
        preference_type="risk text comparison",
        profitability_winner="unavailable",
        profitability_reason="Profitability is not the requested risk-comparison dimension.",
        scale_winner="unavailable",
        scale_reason="Revenue scale is not a substitute for risk text in this comparison.",
        margin_winner="unavailable",
        margin_reason="Net margin is not the requested risk-comparison dimension.",
        growth_winner="unavailable",
        growth_reason="Growth is not the requested risk-comparison dimension.",
        risk_tradeoff=_risk_tradeoff(packet, risk_considerations),
        counterpoint="Risk comparison should use comparable filing risk text; numeric evidence is only optional context.",
        risk_considerations=risk_considerations,
        confidence_level="medium" if has_risk_text else "low",
        evidence_basis=[
            {
                "dimension": "moat_and_competitive_risk",
                "winner": "no_clear_preference",
                "metrics": [],
                "evidence_refs": refs,
            }
        ],
        rationale=_risk_tradeoff(packet, risk_considerations),
        limitations=_limitations(packet),
    )


def _counterpoint(preferred: str, companies: list[str], scale_winner: str, margin_winner: str) -> str:
    if preferred and scale_winner not in {"", "unavailable", "mixed", preferred}:
        return f"{scale_winner} remains worth watching because it leads on revenue scale and may offer more expansion optionality."
    if preferred and margin_winner not in {"", "unavailable", "mixed", preferred}:
        return f"{margin_winner} remains worth watching because it leads on margin quality."
    alternatives = [c for c in companies if c != preferred]
    if alternatives:
        return f"{alternatives[0]} remains worth watching because the available evidence does not make the comparison one-dimensional."
    return "The non-preferred company remains worth watching where its validated evidence is stronger."


def _complete_numeric_basis(metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str]) -> bool:
    return all(
        _to_float((metric_rows.get((company, metric), {}) or {}).get("value")) is not None
        for company in companies
        for metric in ("revenue", "net_income")
    )


def _text_coverage_complete(packet: dict[str, Any], companies: list[str]) -> bool:
    covered = {
        str(item.get("ticker", "")).upper()
        for item in packet.get("text_snippets", []) or []
        if isinstance(item, dict) and str(item.get("ticker", "")).strip()
    }
    return bool(companies) and set(companies).issubset(covered)


def _other_companies(companies: list[str], winner: str) -> list[str]:
    return [company for company in companies if company and company != winner]


def _cash_flow_frame(packet: dict[str, Any], metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str]) -> ComparisonJudgmentFrame:
    fcf_margin_values = _metric_values(metric_rows, companies, "fcf_margin")
    fcf_values = _metric_values(metric_rows, companies, "free_cash_flow")
    ocf_values = _metric_values(metric_rows, companies, "operating_cash_flow")
    capex_values = {company: abs(value) if value is not None else None for company, value in _metric_values(metric_rows, companies, "capital_expenditure").items()}
    winner = _winner_from_values(fcf_margin_values)
    preference_type = "cash-flow quality"
    if winner in {"", "unavailable", "mixed"}:
        winner = _winner_from_values(fcf_values)
        preference_type = "free cash flow"
    if winner in {"", "unavailable", "mixed"}:
        winner = _winner_from_values(ocf_values)
        preference_type = "operating cash flow"
    preferred = winner if winner not in {"", "unavailable", "mixed"} else "no_clear_preference"
    reasons = []
    for metric, label in (
        ("operating_cash_flow", "operating cash flow"),
        ("free_cash_flow", "free cash flow"),
        ("capital_expenditure", "capital expenditure"),
        ("fcf_margin", "FCF margin"),
    ):
        line = _two_company_metric_phrase(
            metric_rows=metric_rows,
            companies=companies,
            winner=_winner(metric_rows, companies, metric) if metric != "capital_expenditure" else _lowest_from_values(capex_values),
            metric=metric,
            metric_label=label,
        )
        if line:
            reasons.append(line)
    rationale = " ".join(reasons) or "Cash-flow quality cannot be ranked because operating cash flow, free cash flow, capex, or FCF margin evidence is incomplete."
    return ComparisonJudgmentFrame(
        companies=companies,
        preferred_company=preferred,
        preference_type=preference_type,
        profitability_winner="unavailable",
        profitability_reason="Profitability is not the requested comparison dimension.",
        scale_winner=_winner(metric_rows, companies, "revenue"),
        scale_reason="Revenue scale is background only for this cash-flow-quality comparison.",
        margin_winner="unavailable",
        margin_reason="Net margin is not the requested comparison dimension.",
        growth_winner="unavailable",
        growth_reason="Growth is not the requested comparison dimension.",
        risk_tradeoff="Cash-flow quality should be read through operating cash flow, free cash flow, capital expenditure, and FCF margin rather than net margin.",
        counterpoint="The non-leading company may still look better on other non-requested dimensions, but this frame is limited to cash-flow quality.",
        confidence_level="medium" if preferred != "no_clear_preference" else "low",
        evidence_basis=[
            {
                "dimension": "cash_flow_quality",
                "winner": preferred,
                "metrics": ["operating_cash_flow", "free_cash_flow", "capital_expenditure", "fcf_margin"],
                "evidence_refs": _refs_for_metric(metric_rows, companies, "operating_cash_flow", "free_cash_flow", "capital_expenditure", "fcf_margin"),
            }
        ],
        rationale=rationale,
        limitations=_limitations(packet),
    )


def _available_metric_values(
    metric_rows: dict[tuple[str, str], dict[str, Any]],
    companies: list[str],
    metric: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for company in companies:
        value = _to_float((metric_rows.get((company, metric), {}) or {}).get("value"))
        if value is not None:
            values[company] = value
    return values


def _valuation_metric_judgments(
    metric_rows: dict[tuple[str, str], dict[str, Any]],
    companies: list[str],
) -> list[dict[str, Any]]:
    judgments: list[dict[str, Any]] = []
    for metric, label in (("pe_ratio", "P/E"), ("ps_ratio", "P/S")):
        winner = _winner(metric_rows, companies, metric)
        refs = _refs_for_metric(metric_rows, companies, metric)
        if winner in {"", "unavailable", "mixed"} or not refs:
            continue
        judgments.append(
            {
                "metric": metric,
                "label": label,
                "higher_risk_company": winner,
                "winner": winner,
                "evidence_refs": refs,
                "zh": f"按 {label}：{winner} 更高，估值倍数压力更大。",
                "en": f"By {label}, {winner} is higher, so valuation-multiple pressure is greater on that metric.",
            }
        )

    fcf_values = _available_metric_values(metric_rows, companies, "fcf_yield")
    fcf_refs = _refs_for_metric(metric_rows, companies, "fcf_yield")
    if len(fcf_values) >= 2 and fcf_refs:
        ordered = sorted(fcf_values.items(), key=lambda item: item[1])
        lower_company, lower_value = ordered[0]
        higher_company, higher_value = ordered[-1]
        close = abs(higher_value - lower_value) <= max(abs(higher_value), abs(lower_value), 0.0001) * 0.10
        if close:
            zh = f"按 FCF yield：两者接近，{lower_company} 略低，现金流收益率角度 {lower_company} 略高风险。"
            en = f"By FCF yield, the two are close; {lower_company} is slightly lower, so its cash-flow-yield risk is slightly higher."
        else:
            zh = f"按 FCF yield：{lower_company} 更低，现金流收益率角度 {lower_company} 估值风险更高。"
            en = f"By FCF yield, {lower_company} is lower, so its cash-flow-yield valuation risk is higher."
        judgments.append(
            {
                "metric": "fcf_yield",
                "label": "FCF yield",
                "higher_risk_company": lower_company,
                "lower_yield_company": lower_company,
                "higher_yield_company": higher_company,
                "winner": lower_company,
                "is_close": close,
                "evidence_refs": fcf_refs,
                "zh": zh,
                "en": en,
            }
        )
    return judgments


def _valuation_risk_frame(packet: dict[str, Any], metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str]) -> ComparisonJudgmentFrame:
    pe_winner = _winner(metric_rows, companies, "pe_ratio")
    ps_winner = _winner(metric_rows, companies, "ps_ratio")
    fcf_risk_winner = _lowest_from_values(_metric_values(metric_rows, companies, "fcf_yield"))
    votes = [item for item in (pe_winner, ps_winner, fcf_risk_winner) if item not in {"", "unavailable", "mixed"}]
    preferred = "no_clear_preference"
    if votes:
        counts = {company: votes.count(company) for company in set(votes)}
        top_count = max(counts.values())
        top = [company for company, count in counts.items() if count == top_count]
        preferred = top[0] if len(top) == 1 else "mixed"
    metric_judgments = _valuation_metric_judgments(metric_rows, companies)
    lines = [str(item.get("en") or "") for item in metric_judgments if str(item.get("en") or "").strip()]
    if len(set(votes)) > 1:
        lines.append("The valuation indicators point in different directions, so a single valuation-risk ranking is not supported.")
        preferred = "mixed"
    elif not votes:
        lines.append("Valuation-risk comparison is bounded because P/E, P/S, FCF yield, market cap, or share-price evidence is incomplete.")
    rationale = " ".join(lines)
    return ComparisonJudgmentFrame(
        companies=companies,
        preferred_company=preferred,
        preference_type="valuation risk",
        profitability_winner="unavailable",
        profitability_reason="Profitability is background only; valuation risk is the requested dimension.",
        scale_winner=_winner(metric_rows, companies, "market_cap"),
        scale_reason="Market cap and share price are valuation inputs, not a standalone preference.",
        margin_winner="unavailable",
        margin_reason="Net margin is not the requested comparison dimension.",
        growth_winner="unavailable",
        growth_reason="Growth is not the requested comparison dimension.",
        risk_tradeoff="The valuation-risk boundary depends on P/E, P/S, FCF yield, market cap, and share-price inputs; ordinary risk-factor text is only optional context.",
        counterpoint="If key valuation inputs are missing or mixed-source, the result should be treated as a bounded valuation-risk comparison rather than a normal risk ranking.",
        confidence_level="medium" if preferred not in {"no_clear_preference", "mixed"} else "low",
        evidence_basis=[
            {
                "dimension": "valuation_and_risk_boundary",
                "winner": preferred,
                "metrics": ["price", "shares_outstanding", "market_cap", "pe_ratio", "ps_ratio", "fcf_yield"],
                "evidence_refs": _refs_for_metric(metric_rows, companies, "price", "adjusted_close", "shares_outstanding", "market_cap", "pe_ratio", "ps_ratio", "fcf_yield"),
                "metric_judgments": metric_judgments,
            }
        ],
        rationale=rationale,
        limitations=_limitations(packet),
    )


def _revenue_quality_frame(packet: dict[str, Any], metric_rows: dict[tuple[str, str], dict[str, Any]], companies: list[str]) -> ComparisonJudgmentFrame:
    growth_winner = _growth_winner(metric_rows, companies)
    scale_winner = _winner(metric_rows, companies, "revenue")
    preferred = growth_winner if growth_winner not in {"", "unavailable", "mixed"} else (
        scale_winner if scale_winner not in {"", "unavailable", "mixed"} else "no_clear_preference"
    )
    growth_reason = (
        _two_company_metric_phrase(metric_rows=metric_rows, companies=companies, winner=growth_winner, metric="revenue_growth", metric_label="revenue growth")
        if growth_winner not in {"", "unavailable", "mixed"}
        else "当前只能比较收入规模和有限收入增长证据，不能完整判断增长质量。"
    )
    scale_reason = (
        _two_company_metric_phrase(metric_rows=metric_rows, companies=companies, winner=scale_winner, metric="revenue", metric_label="revenue scale")
        if scale_winner not in {"", "unavailable", "mixed"}
        else "Revenue scale evidence is unavailable."
    )
    latest_revenue_refs = _refs_for_metric(metric_rows, companies, "revenue")
    revenue_history_refs = _all_refs_for_metric(packet, companies, "revenue")
    revenue_growth_refs = _refs_for_metric(metric_rows, companies, "revenue_growth", "growth", "revenue_growth_rate")
    revenue_period_counts = _metric_period_counts(packet, companies, "revenue")
    history_complete = bool(companies) and all(revenue_period_counts.get(company, 0) >= 2 for company in companies)
    return ComparisonJudgmentFrame(
        companies=companies,
        preferred_company=preferred,
        preference_type="revenue growth quality",
        profitability_winner="unavailable",
        profitability_reason="Profitability and net margin are not the requested growth-quality dimension.",
        scale_winner=scale_winner,
        scale_reason=scale_reason,
        margin_winner="unavailable",
        margin_reason="Net margin is not the requested comparison dimension.",
        growth_winner=growth_winner,
        growth_reason=growth_reason,
        risk_tradeoff="Revenue quality should be read through revenue growth, revenue history, and revenue scale; margin is only optional context.",
        counterpoint="A company can lead on current revenue scale without proving better growth quality if revenue-growth history is incomplete.",
        confidence_level="medium" if preferred != "no_clear_preference" else "low",
        evidence_basis=[
            {
                "dimension": "revenue_quality",
                "winner": preferred,
                "metrics": ["revenue", "revenue_growth"],
                "evidence_refs": list(dict.fromkeys(latest_revenue_refs + revenue_growth_refs + revenue_history_refs)),
                "latest_revenue_refs": latest_revenue_refs,
                "revenue_history_refs": revenue_history_refs,
                "revenue_growth_refs": revenue_growth_refs,
                "scale_winner_refs": latest_revenue_refs,
                "growth_winner_refs": revenue_growth_refs,
                "revenue_history_complete": history_complete,
            }
        ],
        rationale=f"{growth_reason} {scale_reason}".strip(),
        limitations=_limitations(packet),
    )


def _two_company_metric_phrase(
    *,
    metric_rows: dict[tuple[str, str], dict[str, Any]],
    companies: list[str],
    winner: str,
    metric: str,
    metric_label: str,
    fallback_values: dict[str, float | None] | None = None,
) -> str:
    if winner in {"", "unavailable", "mixed"}:
        return ""
    others = _other_companies(companies, winner)
    winner_value = _metric_display(metric_rows, winner, metric, (fallback_values or {}).get(winner))
    if not others:
        return f"{winner} leads on {metric_label}."
    other = others[0]
    other_value = _metric_display(metric_rows, other, metric, (fallback_values or {}).get(other))
    if winner_value and other_value:
        return f"{winner} leads on {metric_label}: {winner} {metric_label} is {winner_value}, versus {other} at {other_value}."
    return f"{winner} leads on {metric_label} versus {other}."


def build_comparison_judgment_frame(packet: dict[str, Any]) -> ComparisonJudgmentFrame:
    packet = dict(packet or {})
    companies = list(
        dict.fromkeys(
            [
                str(item.get("ticker", "") or item.get("company", "")).upper()
                for source in (
                    packet.get("numeric_table", []),
                    packet.get("comparison_table", []),
                    packet.get("numeric_evidence", []),
                    packet.get("computed_metrics", []),
                    packet.get("text_snippets", []),
                )
                for item in (source or [])
                if isinstance(item, dict) and str(item.get("ticker", "") or item.get("company", "")).strip()
            ]
        )
    )
    metric_rows = _latest_metric_rows(packet)
    requested_dimensions = _requested_dimensions(packet)
    focus_dimensions = requested_dimensions or _focus_dimensions(packet)
    primary_dimension = _primary_dimension(packet, focus_dimensions)
    dimension_specific = bool(requested_dimensions) or len(focus_dimensions) == 1
    if dimension_specific and primary_dimension == "valuation_and_risk_boundary":
        return _valuation_risk_frame(packet, metric_rows, companies)
    if dimension_specific and primary_dimension == "cash_flow_quality":
        return _cash_flow_frame(packet, metric_rows, companies)
    if dimension_specific and primary_dimension == "revenue_quality" and "profitability_quality" not in focus_dimensions:
        return _revenue_quality_frame(packet, metric_rows, companies)
    if dimension_specific and primary_dimension == "moat_and_competitive_risk":
        return _risk_comparison_frame(packet, companies)
    if dimension_specific and "valuation_and_risk_boundary" in focus_dimensions:
        return _valuation_risk_frame(packet, metric_rows, companies)
    if dimension_specific and "cash_flow_quality" in focus_dimensions:
        return _cash_flow_frame(packet, metric_rows, companies)
    if dimension_specific and "revenue_quality" in focus_dimensions and "profitability_quality" not in focus_dimensions:
        return _revenue_quality_frame(packet, metric_rows, companies)
    if dimension_specific and focus_dimensions == ["moat_and_competitive_risk"]:
        return _risk_comparison_frame(packet, companies)
    margin_values: dict[str, float | None] = {}
    margin_refs: dict[str, list[str]] = {}
    margin_sources: dict[str, str] = {}
    for company in companies:
        margin, refs, source = _margin_value(metric_rows, company)
        margin_values[company] = margin
        margin_refs[company] = refs
        margin_sources[company] = source

    margin_winner = _winner_from_values(margin_values)
    scale_winner = _winner(metric_rows, companies, "revenue")
    growth_winner = _growth_winner(metric_rows, companies)
    net_income_winner = _winner(metric_rows, companies, "net_income")
    if margin_winner not in {"", "unavailable", "mixed"}:
        profitability_winner = margin_winner
        preference_type = "profitability"
    elif net_income_winner not in {"", "unavailable", "mixed"}:
        profitability_winner = net_income_winner
        preference_type = "absolute profitability"
    else:
        profitability_winner = "unavailable"
        preference_type = "unavailable"

    preferred_company = profitability_winner
    if preferred_company in {"unavailable", "mixed", ""}:
        preferred_company = scale_winner if scale_winner not in {"unavailable", "mixed", ""} else "no_clear_preference"
        preference_type = "scale" if preferred_company != "no_clear_preference" else "no clear preference"

    scale_reason = (
        _two_company_metric_phrase(
            metric_rows=metric_rows,
            companies=companies,
            winner=scale_winner,
            metric="revenue",
            metric_label="revenue scale",
        )
        if scale_winner not in {"", "unavailable", "mixed"}
        else "Revenue scale leadership is unavailable from the validated numeric evidence."
    )
    margin_reason = (
        _two_company_metric_phrase(
            metric_rows=metric_rows,
            companies=companies,
            winner=margin_winner,
            metric="net_margin",
            metric_label="net margin",
            fallback_values=margin_values,
        )
        if margin_winner not in {"", "unavailable", "mixed"}
        else "Net margin comparison is unavailable because revenue or net income is missing for at least one company."
    )
    profitability_reason = (
        f"{profitability_winner} leads profitability because the frame compares revenue, net income, and net margin. {margin_reason}"
        if profitability_winner not in {"", "unavailable", "mixed"}
        else "Profitability leadership is unavailable because revenue, net income, or net margin evidence is incomplete."
    )
    growth_reason = (
        f"{growth_winner} leads on available growth evidence."
        if growth_winner not in {"", "unavailable", "mixed"}
        else "No validated growth metric is available for a reliable growth winner."
    )
    risk_considerations = _risk_considerations(packet)
    risk_tradeoff = _risk_tradeoff(packet, risk_considerations)
    counterpoint = _counterpoint(preferred_company, companies, scale_winner, margin_winner)

    numeric_complete = _complete_numeric_basis(metric_rows, companies)
    margin_complete = bool(companies) and all(margin_values.get(company) is not None for company in companies)
    text_complete = _text_coverage_complete(packet, companies)
    has_text = bool(packet.get("text_snippets", []) or [])
    if numeric_complete and margin_complete and text_complete:
        confidence_level = "high"
    elif numeric_complete and has_text:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    evidence_basis = [
        {
            "dimension": "profitability",
            "winner": profitability_winner,
            "metrics": ["revenue", "net_income", "net_margin"],
            "evidence_refs": list(
                dict.fromkeys(
                    ref
                    for company in companies
                    for ref in (
                        _refs(metric_rows.get((company, "revenue")), metric_rows.get((company, "net_income")))
                        + margin_refs.get(company, [])
                    )
                )
            ),
        },
        {
            "dimension": "scale",
            "winner": scale_winner,
            "metrics": ["revenue"],
            "evidence_refs": list(
                dict.fromkeys(ref for company in companies for ref in _refs(metric_rows.get((company, "revenue"))))
            ),
        },
        {
            "dimension": "margin",
            "winner": margin_winner,
            "metrics": ["net_margin"],
            "evidence_refs": list(dict.fromkeys(ref for company in companies for ref in margin_refs.get(company, []))),
            "calculation_basis": {company: margin_sources.get(company, "unavailable") for company in companies},
        },
    ]
    if risk_considerations:
        evidence_basis.append(
            {
                "dimension": "risk",
                "winner": "mixed",
                "metrics": [],
                "evidence_refs": list(
                    dict.fromkeys(ref for item in risk_considerations for ref in item.get("evidence_refs", []) or [])
                ),
            }
        )

    if preferred_company == "no_clear_preference":
        rationale = "Validated evidence does not support a clear preference across profitability, margin, and scale."
    else:
        rationale = f"{profitability_reason} {scale_reason} {counterpoint}"

    return ComparisonJudgmentFrame(
        companies=companies,
        preferred_company=preferred_company,
        preference_type=preference_type,
        profitability_winner=profitability_winner,
        profitability_reason=profitability_reason,
        scale_winner=scale_winner,
        scale_reason=scale_reason,
        margin_winner=margin_winner,
        margin_reason=margin_reason,
        growth_winner=growth_winner,
        growth_reason=growth_reason,
        risk_tradeoff=risk_tradeoff,
        counterpoint=counterpoint,
        risk_considerations=risk_considerations,
        confidence_level=confidence_level,
        evidence_basis=evidence_basis,
        rationale=rationale,
        limitations=_limitations(packet),
    )


def summarize_comparison_judgment_frame(frame: dict[str, Any]) -> dict[str, Any]:
    frame = dict(frame or {})
    return {
        "preferred_company": str(frame.get("preferred_company", "")),
        "preference_type": str(frame.get("preference_type", "")),
        "profitability_winner": str(frame.get("profitability_winner", "")),
        "scale_winner": str(frame.get("scale_winner", "")),
        "margin_winner": str(frame.get("margin_winner", "")),
        "growth_winner": str(frame.get("growth_winner", "")),
        "confidence_level": str(frame.get("confidence_level", "")),
        "profitability_reason": str(frame.get("profitability_reason", "")),
        "scale_reason": str(frame.get("scale_reason", "")),
        "margin_reason": str(frame.get("margin_reason", "")),
        "growth_reason": str(frame.get("growth_reason", "")),
        "risk_tradeoff": str(frame.get("risk_tradeoff", "")),
        "counterpoint": str(frame.get("counterpoint", "")),
        "risk_consideration_count": len(frame.get("risk_considerations", []) or []),
        "evidence_basis_count": len(frame.get("evidence_basis", []) or []),
    }
