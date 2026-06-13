# ruff: noqa: F401,F403,F405
"""Evidence construction, period binding, and claim validation."""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any, Mapping

from src.agent.constants import *
from src.agent.metric_availability import normalize_metric_name
from src.agent.query_plan import _coerce_int, _infer_period_type, _parse_quarter_token
from src.agent.state import AgentState
from src.agent.output_language import detect_output_language
from src.agent.types import EvidenceBundle

_METRIC_LABEL_ZH = {
    "revenue": "营收",
    "net_income": "净利润",
    "eps": "每股收益",
    "gross_margin": "毛利率",
    "operating_margin": "营业利润率",
    "net_margin": "净利率",
    "revenue_growth": "营收增长率",
    "operating_cash_flow": "经营现金流",
    "free_cash_flow": "自由现金流",
    "capital_expenditure": "资本开支",
    "cash_and_equivalents": "现金及等价物",
    "total_debt": "总债务",
    "net_debt": "净债务",
    "inventory": "存货",
    "receivables": "应收款",
    "cfo_to_net_income": "经营现金流/净利润",
    "fcf_margin": "自由现金流率",
    "market_cap": "市值",
    "pe_ratio": "市盈率",
    "ps_ratio": "市销率",
    "fcf_yield": "自由现金流收益率",
    "close": "收盘价",
    "adjusted_close": "复权收盘价",
    "volume": "成交量",
    "post_return_1d": "财报后1日收益率",
    "post_return_3d": "财报后3日收益率",
    "post_return_5d": "财报后5日收益率",
    "post_return_10d": "财报后10日收益率",
}

_METRIC_LABEL_EN = {
    "revenue": "revenue",
    "net_income": "net income",
    "eps": "EPS",
    "gross_margin": "gross margin",
    "operating_margin": "operating margin",
    "net_margin": "net margin",
    "revenue_growth": "revenue growth",
    "operating_cash_flow": "operating cash flow",
    "free_cash_flow": "free cash flow",
    "capital_expenditure": "capital expenditure",
    "cash_and_equivalents": "cash and equivalents",
    "total_debt": "total debt",
    "net_debt": "net debt",
    "inventory": "inventory",
    "receivables": "receivables",
    "cfo_to_net_income": "operating cash flow / net income",
    "fcf_margin": "free cash flow margin",
    "market_cap": "market cap",
    "pe_ratio": "P/E ratio",
    "ps_ratio": "P/S ratio",
    "fcf_yield": "FCF yield",
    "close": "close price",
    "adjusted_close": "adjusted close",
    "volume": "volume",
    "post_return_1d": "post-filing 1-day return",
    "post_return_3d": "post-filing 3-day return",
    "post_return_5d": "post-filing 5-day return",
    "post_return_10d": "post-filing 10-day return",
}

def _extract_claim_period_hint(sentence: str) -> tuple[int | None, int | None]:
    s = sentence or ""
    year: int | None = None
    quarter: int | None = None

    m = re.search(r"(20\d{2})\s*[qQ]\s*([1-4])", s)
    if not m:
        m = re.search(r"[qQ]\s*([1-4])\s*(20\d{2})", s)
        if m:
            quarter = _coerce_int(m.group(1))
            year = _coerce_int(m.group(2))
    else:
        year = _coerce_int(m.group(1))
        quarter = _coerce_int(m.group(2))

    if quarter is None:
        m_cn = re.search(r"(20\d{2})\s*年\s*第?\s*([一二三四1-4])\s*季(?:度)?", s)
        if m_cn:
            year = _coerce_int(m_cn.group(1))
            quarter = _parse_quarter_token(m_cn.group(2))
    if quarter is None:
        zh_map = {"第一季度": 1, "第二季度": 2, "第三季度": 3, "第四季度": 4}
        for token, q in zh_map.items():
            if token in s:
                quarter = q
                break

    if year is None:
        m_year = re.search(r"(20\d{2})(?:\s*年|\s*财年|\s*fiscal year|\s*calendar year)?", s, flags=re.IGNORECASE)
        if m_year:
            year = _coerce_int(m_year.group(1))

    return year, quarter

def _claim_period_matches_evidence(
    sentence: str,
    evidence_items: list[dict[str, Any]],
) -> bool:
    year_hint, quarter_hint = _extract_claim_period_hint(sentence)
    if year_hint is None and quarter_hint is None:
        return True
    if not evidence_items:
        return False

    def _period_end_year(item: dict[str, Any]) -> int | None:
        raw = str(item.get("period_end", ""))
        m = re.match(r"(20\d{2})-", raw)
        return _coerce_int(m.group(1)) if m else None

    for item in evidence_items:
        fy = _coerce_int(item.get("fiscal_year"))
        fq = _coerce_int(item.get("fiscal_quarter"))
        cy = _coerce_int(item.get("calendar_year"))
        cq = _coerce_int(item.get("calendar_quarter"))
        py = _period_end_year(item)

        if quarter_hint is not None:
            if year_hint is not None:
                if (fy == year_hint and fq == quarter_hint) or (cy == year_hint and cq == quarter_hint):
                    return True
                continue
            if fq == quarter_hint or cq == quarter_hint:
                return True
            continue

        if year_hint is not None and (fy == year_hint or cy == year_hint or py == year_hint):
            return True

    return False

def _period_consistency_ok(
    state: AgentState,
    numeric_citations: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    task_type = str(state.get("task_type", "fact_qa"))
    if bool(state.get("market_reaction_requested")):
        if task_type in NUMERIC_REQUIRED_TASK_TYPES and not numeric_citations:
            return False, "no_numeric_citation_for_market_reaction"
        return True, None
    if task_type not in NUMERIC_REQUIRED_TASK_TYPES:
        return True, None

    period_query = dict(state.get("period_query", {}))
    resolved = dict(state.get("resolved_period_context", {}))
    if resolved.get("needs_clarification"):
        return False, str(resolved.get("clarification_reason", "time_period_ambiguous"))

    if task_type == "company_comparison" and period_query.get("comparison_basis") == "same_period":
        if resolved.get("same_period_match") is False:
            return False, "no_common_period_for_same_period_comparison"

    if not numeric_citations and _risk_text_only_comparison_has_citable_text(state):
        return True, None
    if not numeric_citations:
        return False, "no_numeric_citation_for_time_check"

    requested_period_type = str(period_query.get("period_type") or "")
    target_period_type = str(resolved.get("target_period_type") or "")
    if requested_period_type in {"quarterly", "annual"}:
        expected_type = requested_period_type
    elif requested_period_type in {"latest", "trailing"}:
        expected_type = target_period_type
    else:
        expected_type = ""

    if expected_type:
        if not any(str(c.get("period_type", "")) == expected_type for c in numeric_citations):
            return False, f"period_type_mismatch_expected_{expected_type}"

    year = period_query.get("year")
    quarter = period_query.get("quarter")
    year_basis = str(period_query.get("year_basis", "fiscal"))
    if year is not None:
        year_key = "fiscal_year" if year_basis == "fiscal" else "calendar_year"
        if not any(_coerce_int(c.get(year_key)) == _coerce_int(year) for c in numeric_citations):
            return False, f"year_mismatch_expected_{year_basis}_{year}"
    if quarter is not None:
        quarter_key = "fiscal_quarter" if year_basis == "fiscal" else "calendar_quarter"
        if not any(_coerce_int(c.get(quarter_key)) == _coerce_int(quarter) for c in numeric_citations):
            return False, f"quarter_mismatch_expected_{year_basis}_q{quarter}"

    return True, None


def _risk_text_only_comparison_has_citable_text(state: Mapping[str, Any]) -> bool:
    if str(state.get("task_type") or "") != "company_comparison":
        return False
    requested: list[str] = []
    for source in (
        state.get("requested_dimensions"),
        state.get("required_dimensions"),
        [state.get("primary_dimension")],
        (state.get("canonical_intent") or {}).get("requested_dimensions", []) if isinstance(state.get("canonical_intent"), Mapping) else [],
        (state.get("evidence_policy") or {}).get("required_dimensions", []) if isinstance(state.get("evidence_policy"), Mapping) else [],
    ):
        for item in source or []:
            text = str(item or "").strip()
            if text and text not in requested:
                requested.append(text)
    risk_requested = "moat_and_competitive_risk" in requested or any(
        term in str(state.get("user_query") or "").lower()
        for term in ("风险", "危险", "risk", "danger")
    )
    numeric_dimensions = {
        "valuation_and_risk_boundary",
        "cash_flow_quality",
        "revenue_quality",
        "profitability_quality",
        "balance_sheet_and_capital_intensity",
    }
    if not risk_requested or any(dim in numeric_dimensions for dim in requested):
        return False
    packet = state.get("evidence_packet")
    packet = packet if isinstance(packet, Mapping) else {}
    snippets = packet.get("text_snippets") if isinstance(packet, Mapping) else []
    if isinstance(snippets, list):
        for item in snippets:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("evidence_id") or "").strip()
            dimension = str(item.get("dimension_id") or "").strip()
            section = str(item.get("section") or "").upper().strip()
            if ref.startswith("T") and (dimension == "moat_and_competitive_risk" or section in {"ITEM_1A", "ITEM_7", "ITEM_2"}):
                return True
    for item in state.get("text_citations", []) or []:
        if isinstance(item, Mapping) and str(item.get("evidence_id") or item.get("ref") or "").startswith("T"):
            return True
    return False


def _target_lang(user_query: str) -> str:
    return detect_output_language(user_query)

def _metric_label(metric: str, lang: str) -> str:
    if lang == "zh":
        return _METRIC_LABEL_ZH.get(metric, metric)
    return _METRIC_LABEL_EN.get(metric, metric)

def _period_label(period_type: str | None, lang: str) -> str:
    if period_type == "quarterly":
        return "季度" if lang == "zh" else "quarterly"
    if period_type == "annual":
        return "年度" if lang == "zh" else "annual"
    if period_type == "latest":
        return "最新" if lang == "zh" else "latest"
    if period_type == "trailing":
        return "滚动区间" if lang == "zh" else "trailing window"
    return period_type or ("未知周期" if lang == "zh" else "unknown period")

def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def _format_usd_value(value: Any, lang: str) -> str:
    num = _to_float(value)
    if num is None:
        return "N/A"
    if lang == "zh":
        abs_num = abs(num)
        if abs_num >= 1e8:
            return f"{num / 1e8:,.2f} 亿美元"
        if abs_num >= 1e4:
            return f"{num / 1e4:,.2f} 万美元"
        return f"{num:,.2f} 美元"
    if abs(num) >= 1e9:
        return f"${num / 1e9:,.3f}B"
    if abs(num) >= 1e6:
        return f"${num / 1e6:,.3f}M"
    return f"${num:,.2f}"

def _pct_change(current: Any, previous: Any) -> float | None:
    cur = _to_float(current)
    prev = _to_float(previous)
    if cur is None or prev in (None, 0):
        return None
    return (cur - prev) / abs(prev)

def _pct_text(change: float | None, lang: str) -> str:
    if change is None:
        return "N/A"
    sign = "+" if change >= 0 else ""
    pct = f"{sign}{change * 100:.1f}%"
    return pct if lang == "zh" else pct

def _top_ticker(rows: list[dict[str, Any]]) -> str:
    count: dict[str, int] = {}
    for r in rows:
        t = r.get("ticker", "")
        if not t:
            continue
        count[t] = count.get(t, 0) + 1
    if not count:
        return ""
    return sorted(count.items(), key=lambda kv: kv[1], reverse=True)[0][0]

def _rows_for(rows: list[dict[str, Any]], ticker: str, metric: str, period_type: str) -> list[dict[str, Any]]:
    out = [
        r for r in rows
        if r.get("ticker") == ticker
        and r.get("metric") == metric
        and r.get("period_type") == period_type
    ]
    return sorted(out, key=lambda r: r.get("period_end", ""))

def _latest_row(
    rows: list[dict[str, Any]],
    ticker: str,
    metric: str,
    period_type: str | None = None,
) -> dict[str, Any] | None:
    candidates = [
        r for r in rows
        if r.get("ticker") == ticker
        and r.get("metric") == metric
        and (period_type is None or r.get("period_type") == period_type)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda r: r.get("period_end", ""))[-1]

def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("ticker", "")),
        str(row.get("metric", "")),
        str(row.get("period_type", "")),
        str(row.get("period_end", "")),
    )

def _period_year(period_end: str | None) -> int | None:
    m = re.match(r"^\s*(20\d{2})", str(period_end or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None

def _comparison_period_order(state: AgentState) -> list[str]:
    hint = _infer_period_type(state)
    if hint == "quarterly":
        return ["quarterly", "annual"]
    if hint == "annual":
        return ["annual", "quarterly"]
    return ["annual", "quarterly"]

def _latest_comparable_pair(
    rows: list[dict[str, Any]],
    t1: str,
    t2: str,
    metric: str,
    period_type: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    rows1 = _rows_for(rows, t1, metric, period_type)
    rows2 = _rows_for(rows, t2, metric, period_type)
    if not rows1 or not rows2:
        return None

    if period_type == "annual":
        idx1: dict[int, dict[str, Any]] = {}
        idx2: dict[int, dict[str, Any]] = {}
        for r in rows1:
            y = _period_year(r.get("period_end"))
            if y is not None:
                idx1[y] = r
        for r in rows2:
            y = _period_year(r.get("period_end"))
            if y is not None:
                idx2[y] = r
        common_years = sorted(set(idx1.keys()) & set(idx2.keys()))
        if not common_years:
            return None
        y = common_years[-1]
        return idx1[y], idx2[y]

    idx1 = {str(r.get("period_end", "")): r for r in rows1}
    idx2 = {str(r.get("period_end", "")): r for r in rows2}
    common_periods = sorted(set(idx1.keys()) & set(idx2.keys()))
    if not common_periods:
        return None
    p = common_periods[-1]
    return idx1[p], idx2[p]

def _select_comparison_evidence_rows(state: AgentState, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tickers = _ordered_unique_tickers(state, rows)
    if len(tickers) < 2:
        return {"comparable_pairs": [], "supplement_rows": []}
    t1, t2 = tickers[0], tickers[1]

    metric_priority = list(dict.fromkeys((state.get("requested_metrics", []) or []) + ["revenue", "net_income"]))
    comparable_pairs: list[dict[str, Any]] = []
    used_row_keys: set[tuple[str, str, str, str]] = set()
    period_order = _comparison_period_order(state)

    for metric in metric_priority:
        matched = False
        for period_type in period_order:
            pair = _latest_comparable_pair(rows, t1, t2, metric, period_type)
            if not pair:
                continue
            left, right = pair
            comparable_pairs.append(
                {
                    "metric": metric,
                    "period_type": period_type,
                    "left": left,
                    "right": right,
                }
            )
            used_row_keys.add(_row_key(left))
            used_row_keys.add(_row_key(right))
            matched = True
            break
        if matched and len(comparable_pairs) >= 3:
            break

    supplement_rows: list[dict[str, Any]] = []
    for ticker in (t1, t2):
        for metric in metric_priority:
            latest = _latest_row(rows, ticker, metric, period_type=None)
            if not latest:
                continue
            key = _row_key(latest)
            if key in used_row_keys:
                continue
            used_row_keys.add(key)
            supplement_rows.append(latest)
            if len(supplement_rows) >= 4:
                break
        if len(supplement_rows) >= 4:
            break

    return {
        "comparable_pairs": comparable_pairs,
        "supplement_rows": supplement_rows,
    }

def _ordered_unique_tickers(state: AgentState, rows: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    for t in state.get("companies", []):
        if t and t not in ordered:
            ordered.append(t)
    ct = state.get("comparison_target")
    if ct and ct not in ordered:
        ordered.append(ct)
    for r in rows:
        t = r.get("ticker", "")
        if t and t not in ordered:
            ordered.append(t)
    return ordered

def _build_comparison_conclusion(state: AgentState, rows: list[dict[str, Any]], lang: str) -> str | None:
    tickers = _ordered_unique_tickers(state, rows)
    if len(tickers) < 2:
        return None
    t1, t2 = tickers[0], tickers[1]
    picked = _select_comparison_evidence_rows(state, rows)
    pairs = picked.get("comparable_pairs", [])
    if not pairs:
        if lang == "zh":
            return f"{t1} 与 {t2} 当前无共同可比口径（年度同年或季度同截止日），无法形成严格比较结论。"
        return (
            f"{t1} and {t2} currently do not share a strict comparable basis "
            "(same annual year or same quarterly period_end), so no strict comparison conclusion can be made."
        )

    first = pairs[0]
    left = first.get("left", {})
    right = first.get("right", {})
    metric = str(first.get("metric", ""))
    v1 = _to_float(left.get("value"))
    v2 = _to_float(right.get("value"))
    if v1 is None or v2 is None:
        return None

    metric_label = _metric_label(metric, lang)
    period_type = _period_label(str(first.get("period_type", "")), lang)
    period_end = str(left.get("period_end", ""))
    higher = t1 if v1 >= v2 else t2
    lower = t2 if higher == t1 else t1
    high_v = v1 if higher == t1 else v2
    low_v = v2 if higher == t1 else v1
    ratio_text = f"{high_v / low_v:.2f}" if high_v > 0 and low_v > 0 else ""

    if lang == "zh":
        suffix = f"，约为{lower}的{ratio_text}倍" if ratio_text else ""
        return (
            f"在最新可比{period_type}（{period_end}）的{metric_label}上，"
            f"{higher}高于{lower}{suffix}。"
        )
    suffix = f", about {ratio_text}x {lower}" if ratio_text else ""
    return (
        f"In the latest comparable {period_type} ({period_end}), "
        f"{higher} is higher than {lower} on {metric_label}{suffix}."
    )
    return None

def _build_structured_conclusion(state: AgentState, lang: str) -> str | None:
    rows = _collect_financial_rows(state.get("tool_results", []))
    if not rows:
        return None

    comparison_conclusion = _build_comparison_conclusion(state, rows, lang)
    if comparison_conclusion:
        return comparison_conclusion

    ticker = _top_ticker(rows)
    if not ticker:
        return None

    q_rev = _rows_for(rows, ticker, "revenue", "quarterly")
    q_ni = _rows_for(rows, ticker, "net_income", "quarterly")
    a_rev = _rows_for(rows, ticker, "revenue", "annual")
    a_ni = _rows_for(rows, ticker, "net_income", "annual")

    latest_q = q_rev[-1].get("period_end", "") if q_rev else ""
    latest_a = a_rev[-1].get("period_end", "") if a_rev else ""
    q_rev_chg = _pct_change(q_rev[-1]["value"], q_rev[-2]["value"]) if len(q_rev) >= 2 else None
    q_ni_chg = _pct_change(q_ni[-1]["value"], q_ni[-2]["value"]) if len(q_ni) >= 2 else None
    a_rev_chg = _pct_change(a_rev[-1]["value"], a_rev[-2]["value"]) if len(a_rev) >= 2 else None
    a_ni_chg = _pct_change(a_ni[-1]["value"], a_ni[-2]["value"]) if len(a_ni) >= 2 else None

    parts: list[str] = []
    if lang == "zh":
        if a_rev_chg is not None or a_ni_chg is not None:
            if a_rev_chg is not None and a_ni_chg is not None:
                trend = "利润增速高于收入，盈利能力改善" if a_ni_chg > a_rev_chg else "收入与利润整体同步变化"
                parts.append(
                    f"{ticker} 最新年度（{latest_a}）营收同比{_pct_text(a_rev_chg, lang)}，"
                    f"净利润同比{_pct_text(a_ni_chg, lang)}，{trend}。"
                )
            elif a_rev_chg is not None:
                parts.append(f"{ticker} 最新年度（{latest_a}）营收同比{_pct_text(a_rev_chg, lang)}。")
            else:
                parts.append(f"{ticker} 最新年度（{latest_a}）净利润同比{_pct_text(a_ni_chg, lang)}。")
        if q_rev_chg is not None or q_ni_chg is not None:
            if q_rev_chg is not None and q_ni_chg is not None:
                short_trend = "短期经营动能偏强" if q_rev_chg > 0 and q_ni_chg > 0 else "短期波动较大"
                parts.append(
                    f"最新季度（{latest_q}）营收环比{_pct_text(q_rev_chg, lang)}，"
                    f"净利润环比{_pct_text(q_ni_chg, lang)}，{short_trend}。"
                )
            elif q_rev_chg is not None:
                parts.append(f"最新季度（{latest_q}）营收环比{_pct_text(q_rev_chg, lang)}。")
            else:
                parts.append(f"最新季度（{latest_q}）净利润环比{_pct_text(q_ni_chg, lang)}。")
        return " ".join(parts).strip() or None

    if a_rev_chg is not None or a_ni_chg is not None:
        if a_rev_chg is not None and a_ni_chg is not None:
            trend = "profit grew faster than revenue" if a_ni_chg > a_rev_chg else "revenue and profit moved in the same direction"
            parts.append(
                f"In the latest annual period ({latest_a}), {ticker} revenue was {_pct_text(a_rev_chg, lang)} YoY "
                f"and net income was {_pct_text(a_ni_chg, lang)} YoY; {trend}."
            )
        elif a_rev_chg is not None:
            parts.append(f"In the latest annual period ({latest_a}), {ticker} revenue was {_pct_text(a_rev_chg, lang)} YoY.")
        else:
            parts.append(f"In the latest annual period ({latest_a}), {ticker} net income was {_pct_text(a_ni_chg, lang)} YoY.")
    if q_rev_chg is not None or q_ni_chg is not None:
        if q_rev_chg is not None and q_ni_chg is not None:
            short_trend = "near-term momentum is strong" if q_rev_chg > 0 and q_ni_chg > 0 else "near-term volatility remains"
            parts.append(
                f"In the latest quarter ({latest_q}), revenue was {_pct_text(q_rev_chg, lang)} QoQ "
                f"and net income was {_pct_text(q_ni_chg, lang)} QoQ; {short_trend}."
            )
        elif q_rev_chg is not None:
            parts.append(f"In the latest quarter ({latest_q}), revenue was {_pct_text(q_rev_chg, lang)} QoQ.")
        else:
            parts.append(f"In the latest quarter ({latest_q}), net income was {_pct_text(q_ni_chg, lang)} QoQ.")
    return " ".join(parts).strip() or None

def _collect_financial_rows(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in tool_results:
        if tr.get("tool") != "query_financial_data":
            continue
        data = tr.get("data", {})
        ticker = data.get("ticker", tr.get("ticker", ""))
        for row in data.get("financial_facts", []):
            item = dict(row)
            item.setdefault("ticker", ticker)
            rows.append(item)
    return rows

def _collect_price_rows(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in tool_results:
        if tr.get("tool") != "query_financial_data":
            continue
        data = tr.get("data", {})
        ticker = data.get("ticker", tr.get("ticker", ""))
        for row in data.get("price_data", []):
            item = dict(row)
            item.setdefault("ticker", ticker)
            rows.append(item)
    return rows

def _collect_event_rows(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in tool_results:
        if tr.get("tool") != "query_event_price_window":
            continue
        data = tr.get("data", {})
        ticker = data.get("ticker", tr.get("ticker", ""))
        for event in data.get("events", []):
            if not isinstance(event, dict):
                continue
            item = dict(event)
            item.setdefault("ticker", ticker)
            rows.append(item)
    return rows

def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out

def _structured_source_from_row(row: dict[str, Any]) -> dict[str, Any]:
    ticker = str(row.get("ticker", ""))
    metric = str(row.get("metric", ""))
    period_type = str(row.get("period_type", ""))
    period_end = str(row.get("period_end", ""))
    filing_date = str(row.get("filing_date", "") or "")
    source_provider = str(row.get("source_provider") or "unknown")
    structured_type = _structured_source_type(source_provider)
    return {
        "source_kind": "structured",
        "source": ticker,
        "ticker": ticker,
        "filing_type": structured_type,
        "form_type": structured_type,
        "period": period_end,
        "fiscal_period": period_end,
        "section": "STRUCTURED",
        "metric": metric,
        "period_type": period_type,
        "period_end": period_end,
        "filing_date": filing_date,
        "source_provider": source_provider,
        "source_url": str(row.get("source_url", "") or ""),
        "source_filing_id": str(row.get("source_filing_id", "") or ""),
        "confidence": str(row.get("confidence", "") or ""),
        "extraction_method": str(row.get("extraction_method", "") or ""),
        "source_tag": str(row.get("source_tag", "") or ""),
        "reconciliation_warning": str(row.get("reconciliation_warning", "") or ""),
        "part": "",
        "quality": "high",
        "text_snippet": "",
        "section_fallback": False,
    }

def _structured_source_type(source_provider: str) -> str:
    provider = str(source_provider or "").lower()
    if provider == "sec_companyfacts":
        return "STRUCTURED_SEC_XBRL"
    if provider == "yfinance":
        return "STRUCTURED_YFINANCE"
    if provider == "computed":
        return "STRUCTURED_COMPUTED"
    if provider == "event_price_window":
        return "STRUCTURED_EVENT_WINDOW"
    return "STRUCTURED"

def _rows_to_structured_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(_structured_source_from_row(row))
    return out

def _contains_numeric_token(text: str) -> bool:
    return bool(re.search(r"\d", text or ""))

_TEXT_SUPPORT_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "company",
    "discuss",
    "discussed",
    "faces",
    "filing",
    "from",
    "has",
    "highlighted",
    "include",
    "includes",
    "its",
    "management",
    "may",
    "risk",
    "risks",
    "that",
    "the",
    "their",
    "this",
    "with",
}

_TEXT_CLAIM_GENERIC_TERMS = {
    "business",
    "company",
    "context",
    "disclosure",
    "factor",
    "factors",
    "filing",
    "issue",
    "material",
    "risk",
    "risks",
}

_INVESTMENT_ADVICE_TERMS = {
    "买入",
    "卖出",
    "持有",
    "推荐买",
    "推荐卖",
    "建议买",
    "建议卖",
    "应该买",
    "应该卖",
    "值得买",
    "荐股",
}

_INVESTMENT_ADVICE_PATTERNS = (
    re.compile(r"\b(?:should|must|recommend(?:ed|s|ing)?|advise(?:d|s|ing)?)\s+(?:buy|sell|hold)\b", re.I),
    re.compile(r"\b(?:buy|sell|hold)\s+(?:recommendation|rating|signal)\b", re.I),
    re.compile(r"\b(?:outperform|underperform)\s+(?:rating|recommendation)\b", re.I),
)

_DETERMINISTIC_PREDICTION_TERMS = {
    "guaranteed",
    "certainly",
    "will definitely",
    "必然",
    "一定会",
    "肯定会",
    "保证",
}

_COMPANY_ALIASES = {
    "AAPL": {"aapl", "apple"},
    "AMZN": {"amzn", "amazon"},
    "MSFT": {"msft", "microsoft"},
    "NVDA": {"nvda", "nvidia"},
    "GOOGL": {"googl", "google", "alphabet"},
    "TSLA": {"tsla", "tesla"},
    "JPM": {"jpm", "jpmorgan", "chase"},
    "JNJ": {"jnj", "johnson"},
}


def _has_investment_advice_wording(text: str) -> bool:
    raw = str(text or "")
    lowered = raw.lower()
    if any(term in lowered for term in _INVESTMENT_ADVICE_TERMS):
        return True
    return any(pattern.search(raw) for pattern in _INVESTMENT_ADVICE_PATTERNS)


def _semantic_support_terms(text: str) -> set[str]:
    raw = str(text or "")
    lowered = raw.lower()
    aliases: set[str] = set()
    rules = (
        (("gpu", "graphics processing", "图形处理器"), "gpu"),
        (("data center", "datacenter", "数据中心"), "data_center"),
        (("gaming", "game", "游戏"), "gaming"),
        (("professional visualization", "visualization", "专业可视化"), "professional_visualization"),
        (("automotive", "汽车"), "automotive"),
        (("product", "products", "产品"), "products"),
        (("service", "services", "服务"), "services"),
        (("market", "markets", "市场"), "markets"),
        (("customer", "customers", "客户"), "customers"),
        (("segment", "segments", "platform", "platforms", "业务部门", "平台"), "segments"),
        (("net sales", "sales", "revenue", "收入", "销售"), "revenue_sources"),
        (("aws", "amazon web services"), "aws"),
        (("prime",), "prime"),
        (("marketplace", "seller services", "third-party seller", "third party seller", "卖家"), "marketplace"),
        (("advertising", "ads", "广告"), "advertising"),
        (("fulfillment", "logistics", "配送", "履约"), "fulfillment"),
        (("reportable segment", "reportable segments"), "segments"),
        (("competition", "competitive", "competitor", "竞争"), "competition"),
        (("demand", "需求"), "demand"),
        (("supply chain", "supply", "供应链", "供应"), "supply_chain"),
        (("regulation", "regulatory", "监管"), "regulation"),
        (("customer concentration", "客户集中"), "customer_concentration"),
        (("macro", "macroeconomic", "宏观"), "macroeconomic"),
    )
    for needles, canonical in rules:
        if any(needle in lowered or needle in raw for needle in needles):
            aliases.add(canonical)
    return aliases


def _text_support_terms(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9&._-]*", str(text or "").lower())
    base = {t for t in tokens if len(t) > 3 and t not in _TEXT_SUPPORT_STOPWORDS}
    return base | _semantic_support_terms(text)


def _normalized_for_substring(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _numeric_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?%?", str(text or "")))


def _company_entities(text: str) -> set[str]:
    lowered = str(text or "").lower()
    out: set[str] = set()
    for ticker, aliases in _COMPANY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
            out.add(ticker)
    return out


def _text_claim_entities(text: str) -> set[str]:
    terms = _text_support_terms(text) - _TEXT_CLAIM_GENERIC_TERMS
    return set(sorted(terms)) | _company_entities(text)


def _text_validation_error(
    *,
    claim: dict[str, Any],
    code: str,
    message: str,
    evidence_item: dict[str, Any] | None = None,
    citation_ref: str = "",
    supporting_terms: set[str] | None = None,
    missing_support_terms: set[str] | None = None,
    claim_entities: set[str] | None = None,
    snippet_entities: set[str] | None = None,
) -> dict[str, Any]:
    sentence = str(claim.get("sentence") or claim.get("claim") or "").strip()
    return {
        "claim_type": "text",
        "sentence": sentence,
        "evidence_ids": [citation_ref] if citation_ref else [],
        "reason": code,
        "validation_error_code": code,
        "validation_error_message": message,
        "claim_raw_text": sentence,
        "source_requirement_id": str((evidence_item or {}).get("requirement_id", "")),
        "source_snippet_id": citation_ref,
        "citation_ref": citation_ref,
        "supporting_terms": sorted(supporting_terms or set()),
        "missing_support_terms": sorted(missing_support_terms or set()),
        "claim_entities": sorted(claim_entities or set()),
        "snippet_entities": sorted(snippet_entities or set()),
        "company": str(claim.get("company", "")),
        "claim_section": str(claim.get("section", "")),
        "snippet_section": str((evidence_item or {}).get("section", "")),
    }


def _section_support_terms(section: str) -> set[str]:
    normalized = str(section or "").upper().strip()
    if normalized == "ITEM_1A":
        return {"risk", "competition", "regulation", "uncertainty", "supply", "demand"}
    if normalized == "ITEM_7":
        return {"management", "discussion", "operating", "results", "demand", "margin", "execution"}
    if normalized == "ITEM_1":
        return {"business", "segments", "products", "competition", "customers", "sales", "revenue_sources", "aws", "prime", "marketplace", "advertising", "fulfillment"}
    if normalized == "BUSINESS":
        return {"business", "segments", "products", "competition", "customers", "sales", "revenue_sources", "aws", "prime", "marketplace", "advertising", "fulfillment"}
    if normalized in {"MD&A", "MDA"}:
        return {"management", "discussion", "operating", "results", "demand", "margin", "execution"}
    if normalized == "ITEM_2":
        return {"operating", "results", "quarter", "demand", "pressure"}
    return set()

def _text_claim_supported_by_evidence(sentence: str, evidence_items: list[dict[str, Any]]) -> bool:
    claim_terms = _text_support_terms(sentence)
    if not claim_terms:
        return True
    for item in evidence_items:
        snippet = str(item.get("supporting_snippet") or item.get("text_snippet") or "")
        evidence_terms = _text_support_terms(snippet)
        supporting_terms = {str(t).lower() for t in item.get("supporting_terms", []) if str(t).strip()}
        section_terms = _section_support_terms(str(item.get("section", "")))
        ticker = str(item.get("ticker", "")).lower()
        if ticker and ticker in str(sentence).lower() and ticker in snippet.lower():
            return True
        if claim_terms & (evidence_terms | supporting_terms | section_terms):
            return True
    return False

def _has_estimation_phrase(text: str) -> bool:
    t = (text or "").lower()
    return any(term in t for term in ESTIMATION_TERMS)

def _extract_compute_result_value(result: dict[str, Any]) -> tuple[float | None, str, str]:
    numeric_keys = (
        "qoq_change",
        "yoy_change",
        "growth",
        "margin",
        "cagr",
        "ratio",
        "difference",
        "multiple",
        "value",
        "end_value",
    )
    for key in numeric_keys:
        if key not in result:
            continue
        value = _to_float(result.get(key))
        if value is None:
            continue
        unit = "ratio" if key.endswith("change") or key in {"growth", "margin", "cagr", "ratio", "multiple"} else ""
        return value, unit, key
    return None, "", ""

def _build_numeric_evidence(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, str, str, str], int] = {}

    def _append_raw(row: dict[str, Any]) -> None:
        key = (
            str(row.get("source_tool", "")),
            str(row.get("ticker", "")),
            str(row.get("metric", "")),
            str(row.get("period_type", "")),
            str(row.get("period_end", "")),
            str(row.get("provenance", "")),
        )
        if key in seen:
            existing = raw_rows[seen[key]]
            requirement_id = str(row.get("requirement_id", "") or "")
            requirement_ids = list(existing.get("requirement_ids", []) or [])
            if requirement_id and requirement_id not in requirement_ids:
                requirement_ids.append(requirement_id)
            existing["requirement_ids"] = requirement_ids
            if not existing.get("requirement_id") and requirement_id:
                existing["requirement_id"] = requirement_id
            return
        requirement_id = str(row.get("requirement_id", "") or "")
        row["requirement_ids"] = [requirement_id] if requirement_id else []
        seen[key] = len(raw_rows)
        raw_rows.append(row)

    for tr in tool_results:
        tool_name = str(tr.get("tool", ""))
        if tool_name == "query_financial_data" and "data" in tr:
            data = tr.get("data", {})
            ticker = str(data.get("ticker", tr.get("ticker", "")))
            for row in data.get("financial_facts", []):
                if row.get("exclude_from_evidence_matrix"):
                    continue
                value = _to_float(row.get("value"))
                if value is None:
                    continue
                period_end = str(row.get("period_end", ""))
                metric = str(row.get("metric", ""))
                _append_raw(
                    {
                        "source_tool": "query_financial_data",
                        "requirement_id": str(row.get("requirement_id", "") or tr.get("requirement_id", "") or ""),
                        "ticker": str(row.get("ticker", ticker)),
                        "metric": metric,
                        "role": str(row.get("role") or row.get("evidence_role") or ""),
                        "evidence_role": str(row.get("evidence_role") or row.get("role") or ""),
                        "source_requirement_id": str(row.get("source_requirement_id") or row.get("requirement_id") or tr.get("requirement_id", "") or ""),
                        "period_scope": str(row.get("period_scope") or row.get("period_type", "") or "unknown"),
                        "quality_status": str(row.get("quality_status") or "valid"),
                        "period_type": str(row.get("period_type", "")),
                        "period_end": period_end,
                        "value": value,
                        "unit": str(row.get("unit", "")),
                        "fiscal_year": row.get("fiscal_year"),
                        "fiscal_quarter": row.get("fiscal_quarter"),
                        "calendar_year": row.get("calendar_year"),
                        "calendar_quarter": row.get("calendar_quarter"),
                        "source_provider": str(row.get("source_provider") or "yfinance"),
                        "source_url": str(row.get("source_url", "") or ""),
                        "source_filing_id": str(row.get("source_filing_id", "") or ""),
                        "confidence": str(row.get("confidence") or "medium"),
                        "extraction_method": str(row.get("extraction_method") or "api_statement"),
                        "source_tag": str(row.get("source_tag", "") or ""),
                        "reconciliation_warning": str(row.get("reconciliation_warning", "") or ""),
                        "provenance": (
                            f"financial_facts:{row.get('source_provider') or 'yfinance'}:"
                            f"{ticker}:{metric}:{period_end}:{row.get('source_tag') or ''}"
                        ),
                    }
                )

            for row in data.get("price_data", []):
                if row.get("exclude_from_evidence_matrix"):
                    continue
                row_ticker = str(row.get("ticker", ticker))
                row_date = str(row.get("date", ""))
                if str(row.get("metric") or "").strip() and row.get("value") is not None:
                    row_date = str(row.get("period_end") or row.get("period") or row_date)
                    value = _to_float(row.get("value"))
                    if value is None:
                        continue
                    _append_raw(
                        {
                            "source_tool": "query_financial_data",
                            "requirement_id": str(row.get("requirement_id", "") or tr.get("requirement_id", "") or ""),
                            "ticker": row_ticker,
                            "metric": str(row.get("metric")),
                            "role": str(row.get("role") or row.get("evidence_role") or ""),
                            "evidence_role": str(row.get("evidence_role") or row.get("role") or ""),
                            "source_requirement_id": str(row.get("source_requirement_id") or row.get("requirement_id") or tr.get("requirement_id", "") or ""),
                            "period_scope": str(row.get("period_scope") or row.get("period_type") or "daily"),
                            "quality_status": str(row.get("quality_status") or "valid"),
                            "period_type": str(row.get("period_type") or "daily"),
                            "period_end": row_date,
                            "value": value,
                            "unit": str(row.get("unit", "")),
                            "fiscal_year": None,
                            "fiscal_quarter": None,
                            "calendar_year": _period_year(row_date),
                            "calendar_quarter": None,
                            "source_provider": str(row.get("source_provider") or "yfinance"),
                            "source_url": str(row.get("source_url", "") or ""),
                            "source_filing_id": str(row.get("source_filing_id", "") or ""),
                            "confidence": str(row.get("confidence") or "medium"),
                            "extraction_method": str(row.get("extraction_method") or "api_price_history"),
                            "source_tag": str(row.get("source_tag", "") or ""),
                            "reconciliation_warning": str(row.get("reconciliation_warning", "") or ""),
                            "provenance": f"price_data:{row_ticker}:{row.get('metric')}:{row_date}",
                        }
                    )
                    continue
                for metric, metric_value in row.items():
                    if metric in {"ticker", "date"}:
                        continue
                    value = _to_float(metric_value)
                    if value is None:
                        continue
                    _append_raw(
                        {
                            "source_tool": "query_financial_data",
                            "requirement_id": str(row.get("requirement_id", "") or tr.get("requirement_id", "") or ""),
                            "ticker": row_ticker,
                            "metric": str(metric),
                            "role": str(row.get("role") or row.get("evidence_role") or ""),
                            "evidence_role": str(row.get("evidence_role") or row.get("role") or ""),
                            "source_requirement_id": str(row.get("source_requirement_id") or row.get("requirement_id") or tr.get("requirement_id", "") or ""),
                            "period_scope": str(row.get("period_scope") or "daily"),
                            "quality_status": str(row.get("quality_status") or "valid"),
                            "period_type": "daily",
                            "period_end": row_date,
                            "value": value,
                            "unit": "",
                            "fiscal_year": None,
                            "fiscal_quarter": None,
                            "calendar_year": _period_year(row_date),
                            "calendar_quarter": None,
                            "source_provider": str(row.get("source_provider") or "yfinance"),
                            "source_url": str(row.get("source_url", "") or ""),
                            "source_filing_id": str(row.get("source_filing_id", "") or ""),
                            "confidence": str(row.get("confidence") or "medium"),
                            "extraction_method": str(row.get("extraction_method") or "api_price_history"),
                            "source_tag": str(row.get("source_tag", "") or ""),
                            "reconciliation_warning": str(row.get("reconciliation_warning", "") or ""),
                            "provenance": f"price_data:{row_ticker}:{metric}:{row_date}",
                        }
                    )

        if tool_name == "compute_metrics" and "data" in tr:
            data = tr.get("data", {})
            results = data.get("results", [])
            ticker = str(tr.get("ticker", ""))
            metric = str(tr.get("metric", ""))
            computation = str(tr.get("computation", ""))
            for i, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    continue
                value, unit, value_key = _extract_compute_result_value(result)
                if value is None:
                    continue
                period_end = str(
                    result.get("period")
                    or result.get("end_period")
                    or result.get("start_period")
                    or ""
                )
                source_tag = str(result.get("source_tag") or tr.get("source_tag") or value_key)
                row = {
                    "source_tool": "compute_metrics",
                    "requirement_id": str(tr.get("requirement_id", "") or ""),
                    "ticker": ticker,
                    "metric": metric,
                    "source_requirement_id": str(tr.get("requirement_id", "") or ""),
                    "role": str(tr.get("role") or tr.get("evidence_role") or ""),
                    "evidence_role": str(tr.get("evidence_role") or tr.get("role") or ""),
                    "period_type": str(result.get("period_type", "") or ""),
                    "period_scope": str(result.get("period_type", "") or ""),
                    "quality_status": str(result.get("quality_status") or tr.get("quality_status") or "valid"),
                    "period_end": period_end,
                    "value": value,
                    "unit": unit,
                    "fiscal_year": None,
                    "fiscal_quarter": None,
                    "calendar_year": _period_year(period_end),
                    "calendar_quarter": None,
                    "source_provider": "computed",
                    "source_url": "",
                    "source_filing_id": "",
                    "confidence": str(result.get("confidence") or tr.get("confidence") or "medium"),
                    "extraction_method": "compute_metrics",
                    "source_tag": source_tag,
                    "reconciliation_warning": str(
                        result.get("reconciliation_warning") or tr.get("reconciliation_warning") or ""
                    ),
                    "provenance": f"compute_metrics:{metric}:{computation}:{source_tag}:{i}",
                }
                for trace_key in (
                    "share_price",
                    "price_date",
                    "shares_outstanding",
                    "shares_period",
                    "market_cap",
                    "market_cap_period",
                    "statement_period",
                    "revenue_period",
                    "net_income_period",
                    "free_cash_flow_period",
                    "period_basis",
                    "dependencies",
                    "numerator_metric",
                    "denominator_metric",
                    "numerator_requirement_id",
                    "denominator_requirement_id",
                    "left_metric",
                    "right_metric",
                    "left_requirement_id",
                    "right_requirement_id",
                ):
                    if trace_key in result:
                        row[trace_key] = result.get(trace_key)
                _append_raw(row)

        if tool_name == "query_event_price_window" and "data" in tr:
            data = tr.get("data", {})
            ticker = str(data.get("ticker", tr.get("ticker", "")))
            for event in data.get("events", []):
                if not isinstance(event, dict):
                    continue
                event_date = str(event.get("event_date", ""))
                filing_id = str(event.get("filing_id", ""))
                form_type = str(event.get("form_type", ""))
                fiscal_period = str(event.get("fiscal_period", ""))
                returns = event.get("returns", {})
                if not isinstance(returns, dict):
                    continue
                for r_key, r_value in returns.items():
                    m = re.match(r"return_(\d+)d", str(r_key))
                    if not m:
                        continue
                    value = _to_float(r_value)
                    if value is None:
                        continue
                    days = int(m.group(1))
                    metric = f"post_return_{days}d"
                    _append_raw(
                        {
                            "source_tool": "query_event_price_window",
                            "requirement_id": str(event.get("requirement_id", "") or tr.get("requirement_id", "") or ""),
                            "ticker": str(event.get("ticker", ticker)),
                            "metric": metric,
                            "period_type": "event_window",
                            "period_end": event_date,
                            "value": value,
                            "unit": "ratio",
                            "fiscal_year": _period_year(fiscal_period) or _period_year(event_date),
                            "fiscal_quarter": None,
                            "calendar_year": _period_year(event_date),
                            "calendar_quarter": None,
                            "source_provider": "event_price_window",
                            "source_url": "",
                            "source_filing_id": filing_id,
                            "confidence": "high",
                            "extraction_method": "event_price_window",
                            "source_tag": r_key,
                            "reconciliation_warning": "",
                            "provenance": (
                                f"event_price_windows:{ticker}:{filing_id}:{metric}:{event_date}:"
                                f"{form_type}:{fiscal_period}"
                            ),
                        }
                    )

    out: list[dict[str, Any]] = []
    for i, row in enumerate(raw_rows, start=1):
        item = dict(row)
        item["evidence_id"] = f"N{i}"
        out.append(item)
    return out

def _build_text_evidence(retrieved_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: dict[tuple[str, str, str, str, str, int], int] = {}
    for doc in retrieved_docs:
        ticker = str(doc.get("ticker", ""))
        filing_id = str(doc.get("filing_id", ""))
        form_type = str(doc.get("form_type", ""))
        fiscal_period = str(
            doc.get("fiscal_period")
            or doc.get("period")
            or doc.get("period_end")
            or UNKNOWN_PERIOD
        )
        section = str(doc.get("section", ""))
        chunk_order = int(doc.get("chunk_order", 0) or 0)
        key = (ticker, filing_id, form_type, fiscal_period, section, chunk_order)
        requirement_id = str(doc.get("requirement_id", "") or "")
        if key in seen:
            existing = out[seen[key]]
            requirement_ids = list(existing.get("requirement_ids", []) or [])
            if requirement_id and requirement_id not in requirement_ids:
                requirement_ids.append(requirement_id)
            existing["requirement_ids"] = requirement_ids
            if not existing.get("requirement_id") and requirement_id:
                existing["requirement_id"] = requirement_id
            supporting_terms = list(existing.get("supporting_terms", []) or [])
            for term in list(doc.get("supporting_terms", []) or []):
                if term not in supporting_terms:
                    supporting_terms.append(term)
            existing["supporting_terms"] = supporting_terms
            for metadata_key in ("dimension_id", "framework_id", "retrieval_intent", "analysis_purpose"):
                if not existing.get(metadata_key) and doc.get(metadata_key):
                    existing[metadata_key] = str(doc.get(metadata_key) or "")
            continue
        seen[key] = len(out)
        text = re.sub(r"\s+", " ", str(doc.get("text", "")).strip())
        supporting_snippet = re.sub(
            r"\s+",
            " ",
            str(doc.get("supporting_snippet") or doc.get("text_snippet") or text[:220]).strip(),
        )
        if len(supporting_snippet) > 360:
            supporting_snippet = supporting_snippet[:359].rstrip() + "…"
        out.append(
            {
                "evidence_id": f"T{len(out) + 1}",
                "requirement_id": requirement_id,
                "requirement_ids": [requirement_id] if requirement_id else [],
                "dimension_id": str(doc.get("dimension_id", "")),
                "framework_id": str(doc.get("framework_id", "")),
                "retrieval_intent": str(doc.get("retrieval_intent", "")),
                "analysis_purpose": str(doc.get("analysis_purpose", "")),
                "ticker": ticker,
                "filing_id": filing_id,
                "form_type": form_type,
                "fiscal_period": fiscal_period,
                "section": section,
                "chunk_order": chunk_order,
                "quality": str(doc.get("quality", "")),
                "score": _to_float(doc.get("final_score")) or _to_float(doc.get("score")) or 0.0,
                "section_fallback": bool(doc.get("section_fallback", False)),
                "text_snippet": supporting_snippet or text[:220],
                "supporting_snippet": supporting_snippet or text[:220],
                "supporting_terms": list(doc.get("supporting_terms", []) or []),
                "score_breakdown": dict(doc.get("score_breakdown", {}) or {}),
            }
        )
    return out

def _build_evidence_bundle(state: AgentState) -> dict[str, Any]:
    numeric_evidence = _build_numeric_evidence(state.get("tool_results", []))
    text_evidence = _build_text_evidence(state.get("retrieved_docs", []))
    numeric_map = {str(e.get("evidence_id", "")): e for e in numeric_evidence}
    text_map = {str(e.get("evidence_id", "")): e for e in text_evidence}
    return {
        "numeric_evidence": numeric_evidence,
        "text_evidence": text_evidence,
        "numeric_map": numeric_map,
        "text_map": text_map,
        "evidence_map": {**numeric_map, **text_map},
    }

def _evidence_catalog_text(
    evidence: list[dict[str, Any]],
    fields: tuple[str, ...],
    max_items: int = 40,
) -> str:
    if not evidence:
        return "(none)"
    lines: list[str] = []
    for row in evidence[:max_items]:
        payload = {k: row.get(k, "") for k in fields}
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines)

def _normalize_claims(parsed: dict[str, Any], key: str) -> list[dict[str, Any]]:
    claims = parsed.get(key, [])
    if not isinstance(claims, list):
        return []
    out: list[dict[str, Any]] = []
    for item in claims:
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("sentence") or item.get("claim") or "").strip()
        citation_ref = str(item.get("citation_ref", "") or "").strip().upper()
        evidence_ids = item.get("evidence_ids", [])
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        if citation_ref:
            evidence_ids = [citation_ref] + list(evidence_ids)
        norm_ids: list[str] = []
        for eid in evidence_ids:
            eid_text = str(eid).strip().upper()
            if not eid_text or eid_text in norm_ids:
                continue
            norm_ids.append(eid_text)
        normalized = {
            "sentence": sentence,
            "claim": sentence,
            "evidence_ids": norm_ids,
            "citation_ref": norm_ids[0] if norm_ids else citation_ref,
            "company": str(item.get("company", "") or "").upper(),
            "claim_type": str(item.get("claim_type", "") or ""),
            "dimension_id": str(item.get("dimension_id", "") or ""),
            "supporting_quote": str(item.get("supporting_quote", "") or ""),
            "confidence": str(item.get("confidence", "") or ""),
            "claim_source": str(item.get("claim_source", "") or ""),
        }
        out.append(normalized)
    return out

def _raw_numeric_text(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return str(value)
    if float(num).is_integer():
        return str(int(num))
    return f"{num:.6f}".rstrip("0").rstrip(".")


def _ratio_numeric_text(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return str(value)
    return f"{num * 100:.2f}%"

def _effective_expected_period_type(
    period_query: dict[str, Any],
    resolved_period_context: dict[str, Any],
) -> str:
    requested_period_type = str(period_query.get("period_type") or "")
    target_period_type = str(resolved_period_context.get("target_period_type") or "")
    if requested_period_type in {"quarterly", "annual"}:
        return requested_period_type
    if requested_period_type in {"latest", "trailing"}:
        return target_period_type
    return ""

def _filter_numeric_evidence_for_query(
    state: AgentState,
    numeric_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    period_query = dict(state.get("period_query", {}))
    resolved_period_context = dict(state.get("resolved_period_context", {}))
    strict_match = bool(resolved_period_context.get("strict_period_match", True))
    is_single_company_latest_available = (
        str(state.get("analysis_scope", "")) == "single_company"
        and str(state.get("time_policy", "")) == "latest_available"
    )
    if is_single_company_latest_available:
        strict_match = False
    rows = list(numeric_evidence)
    market_reaction_requested = bool(state.get("market_reaction_requested"))
    event_rows = [r for r in rows if str(r.get("source_tool", "")) == "query_event_price_window"]
    if market_reaction_requested:
        rows = [r for r in rows if str(r.get("source_tool", "")) != "query_event_price_window"]
    if not rows:
        return _dedupe_rows(event_rows) if market_reaction_requested else []

    expected_type = _effective_expected_period_type(period_query, resolved_period_context)
    if expected_type and not is_single_company_latest_available:
        typed = [r for r in rows if str(r.get("period_type", "")) == expected_type]
        if typed:
            rows = typed
        elif strict_match:
            return []

    year = period_query.get("year")
    quarter = period_query.get("quarter")
    year_basis = str(period_query.get("year_basis", "fiscal"))
    if year is not None:
        year_key = "fiscal_year" if year_basis == "fiscal" else "calendar_year"
        year_rows = [r for r in rows if _coerce_int(r.get(year_key)) == _coerce_int(year)]
        if year_rows:
            rows = year_rows
        elif strict_match:
            return []
    if quarter is not None:
        quarter_key = "fiscal_quarter" if year_basis == "fiscal" else "calendar_quarter"
        quarter_rows = [r for r in rows if _coerce_int(r.get(quarter_key)) == _coerce_int(quarter)]
        if quarter_rows:
            rows = quarter_rows
        elif strict_match:
            return []

    if (
        str(state.get("task_type", "")) == "company_comparison"
        and str(period_query.get("comparison_basis", "same_period")) == "same_period"
    ):
        common_periods = [str(x) for x in resolved_period_context.get("common_periods", []) if str(x).strip()]
        if common_periods:
            common_rows = [r for r in rows if str(r.get("period_end", "")) in common_periods]
            if common_rows:
                rows = common_rows
            elif strict_match:
                return []
        elif resolved_period_context.get("same_period_match") is False and strict_match:
            return []

    if market_reaction_requested:
        return _dedupe_rows(event_rows + rows)
    return rows

def _metric_priority_from_state(state: AgentState, rows: list[dict[str, Any]]) -> list[str]:
    requested = [str(m) for m in state.get("requested_metrics", []) if str(m).strip()]
    discovered = [str(r.get("metric", "")) for r in rows if str(r.get("metric", "")).strip()]
    event_query = dict(state.get("event_query", {}))
    event_windows = event_query.get("window_days", [])
    event_metrics: list[str] = []
    if bool(state.get("market_reaction_requested")):
        for n in event_windows or [1, 5, 10]:
            try:
                n_int = int(n)
            except (TypeError, ValueError):
                continue
            metric_name = f"post_return_{n_int}d"
            if metric_name not in event_metrics:
                event_metrics.append(metric_name)
    default_order = [
        *event_metrics,
        "revenue",
        "net_income",
        "net_margin",
        "eps",
        "gross_margin",
        "operating_margin",
        "adjusted_close",
        "close",
        "volume",
    ]
    out: list[str] = []
    for m in requested + default_order + discovered:
        if m and m not in out:
            out.append(m)
    return out

def _numeric_claim_sentence(row: dict[str, Any], lang: str) -> str:
    ticker = str(row.get("ticker", ""))
    metric = _metric_label(str(row.get("metric", "")), lang)
    period_type = _period_label(str(row.get("period_type", "")), lang)
    period_end = str(row.get("period_end", ""))
    unit = str(row.get("unit", "")).strip()
    metric_name = str(row.get("metric", ""))
    if unit.lower() == "ratio" or metric_name.endswith("_margin") or metric_name.startswith("post_return_"):
        value_text = _ratio_numeric_text(row.get("value"))
    else:
        value_text = _raw_numeric_text(row.get("value"))
    unit_text = "" if unit.lower() == "ratio" else (f" {unit}" if unit else "")
    if lang == "zh":
        return f"{ticker} {metric}（{period_end}，{period_type}）= {value_text}{unit_text}".strip()
    return f"{ticker} {metric} ({period_end}, {period_type}) = {value_text}{unit_text}".strip()

def _select_numeric_rows_for_fact(
    state: AgentState,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = [r for r in rows if str(r.get("source_tool", "")) == "query_financial_data"] or list(rows)
    base = sorted(base, key=lambda r: (str(r.get("period_end", "")), str(r.get("metric", ""))), reverse=True)
    if not base:
        return []
    if _is_profit_decline_query(str(state.get("user_query") or "")):
        for metric in ("net_income", "operating_income"):
            selected = [r for r in base if str(r.get("metric", "")) == metric]
            if len(selected) >= 2:
                return selected[:4]
    metric_priority = _metric_priority_from_state(state, base)
    for metric in metric_priority:
        for row in base:
            if str(row.get("metric", "")) == metric:
                return [row]
    return [base[0]]


def _is_profit_decline_query(user_query: str) -> bool:
    query = str(user_query or "").lower()
    asks_why = "为什么" in query or "why" in query
    has_decline = any(
        term in query
        for term in (
            "利润下降",
            "利润下滑",
            "净利润下降",
            "净利润下滑",
            "盈利下降",
            "盈利下滑",
            "profit decline",
            "profit declined",
            "earnings decline",
            "earnings declined",
        )
    )
    return asks_why and has_decline

def _select_numeric_rows_for_trend(
    state: AgentState,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = [r for r in rows if str(r.get("source_tool", "")) == "query_financial_data"] or list(rows)
    base = sorted(base, key=lambda r: (str(r.get("period_end", "")), str(r.get("metric", ""))), reverse=True)
    if not base:
        return []

    period_query = dict(state.get("period_query", {}))
    max_rows = int(period_query.get("trailing_n") or 4) if str(period_query.get("period_type", "")) == "trailing" else 4
    max_rows = max(2, min(max_rows, 8))
    metric_priority = _metric_priority_from_state(state, base)
    for metric in metric_priority:
        selected = [r for r in base if str(r.get("metric", "")) == metric]
        if len(selected) >= 2:
            return selected[:max_rows]
    return base[:max_rows]

def _ordered_comparison_tickers(state: AgentState) -> list[str]:
    companies: list[str] = []
    for t in state.get("companies", []):
        s = str(t or "").upper()
        if s and s not in companies:
            companies.append(s)
    ct = str(state.get("comparison_target") or "").upper()
    if ct and ct not in companies:
        companies.append(ct)
    return companies[:2]

def _select_numeric_rows_for_comparison(
    state: AgentState,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = list(rows)
    if not base:
        return []
    tickers = _ordered_comparison_tickers(state)
    if len(tickers) < 2:
        return sorted(base, key=lambda r: str(r.get("period_end", "")), reverse=True)[:4]

    period_query = dict(state.get("period_query", {}))
    basis = str(period_query.get("comparison_basis", "same_period"))
    metric_priority = _metric_priority_from_state(state, base)
    selected: list[dict[str, Any]] = []

    for metric in metric_priority:
        left_rows = sorted(
            [r for r in base if str(r.get("ticker", "")).upper() == tickers[0] and str(r.get("metric", "")) == metric],
            key=lambda r: str(r.get("period_end", "")),
            reverse=True,
        )
        right_rows = sorted(
            [r for r in base if str(r.get("ticker", "")).upper() == tickers[1] and str(r.get("metric", "")) == metric],
            key=lambda r: str(r.get("period_end", "")),
            reverse=True,
        )
        if not left_rows or not right_rows:
            continue

        if basis == "same_period":
            right_by_period = {(str(r.get("period_end", "")), str(r.get("period_type", ""))): r for r in right_rows}
            for left in left_rows:
                key = (str(left.get("period_end", "")), str(left.get("period_type", "")))
                right = right_by_period.get(key)
                if right:
                    selected.extend([left, right])
                    break
        else:
            selected.extend([left_rows[0], right_rows[0]])

        if len(selected) >= 6:
            break

    if not selected:
        for ticker in tickers:
            t_rows = sorted(
                [r for r in base if str(r.get("ticker", "")).upper() == ticker],
                key=lambda r: str(r.get("period_end", "")),
                reverse=True,
            )
            if t_rows:
                selected.append(t_rows[0])

    return _dedupe_rows(selected)[:6]

def _select_numeric_rows_for_single_company_methodology(
    state: AgentState,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select core numeric rows for open-ended single-company methodology answers."""
    base = list(rows)
    if not base:
        return []
    companies = [str(t or "").upper() for t in state.get("companies", []) if str(t or "").strip()]
    company = companies[0] if companies else ""
    if company:
        company_rows = [r for r in base if str(r.get("ticker", "")).upper() == company]
        if company_rows:
            base = company_rows
    selected: list[dict[str, Any]] = []
    trend_metrics = {"revenue", "net_income", "net_margin"}
    latest_metrics = (
        "revenue_growth",
        "gross_margin",
        "operating_margin",
        "eps",
        "operating_cash_flow",
        "free_cash_flow",
        "capital_expenditure",
        "cfo_to_net_income",
        "cash_conversion",
        "fcf_margin",
        "cash_and_equivalents",
        "total_debt",
        "net_debt",
        "total_assets",
        "total_liabilities",
        "shareholders_equity",
        "debt_to_equity",
        "capex_to_revenue",
        "inventory",
        "receivables",
        "receivables_to_revenue",
        "inventory_to_revenue",
        "adjusted_close",
        "close",
        "price",
        "shares_outstanding",
        "market_cap",
        "pe_ratio",
        "ps_ratio",
        "fcf_yield",
    )
    for metric in ("revenue", "net_income", "net_margin"):
        metric_rows = sorted(
            [r for r in base if str(r.get("metric", "")) == metric],
            key=lambda r: str(r.get("period_end", "")),
            reverse=True,
        )
        selected.extend(metric_rows[:4 if metric in trend_metrics else 1])
    for metric in latest_metrics:
        metric_rows = sorted(
            [r for r in base if normalize_metric_name(str(r.get("metric", ""))) == normalize_metric_name(metric)],
            key=lambda r: str(r.get("period_end", "")),
            reverse=True,
        )
        if metric_rows:
            selected.append(metric_rows[0])
    if selected:
        return _dedupe_rows(selected)[:48]
    return _dedupe_rows(sorted(base, key=lambda r: str(r.get("period_end", "")), reverse=True)[:8])

def _select_numeric_rows_for_market_reaction(
    state: AgentState,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    event_rows = [r for r in rows if str(r.get("source_tool", "")) == "query_event_price_window"]
    if not event_rows:
        return []
    event_rows = sorted(event_rows, key=lambda r: str(r.get("period_end", "")), reverse=True)
    metric_priority = [m for m in _metric_priority_from_state(state, event_rows) if m.startswith("post_return_")]
    if not metric_priority:
        metric_priority = ["post_return_1d", "post_return_5d", "post_return_10d"]

    task_type = str(state.get("task_type", "trend_analysis"))
    if task_type == "company_comparison":
        selected: list[dict[str, Any]] = []
        for ticker in _ordered_comparison_tickers(state):
            ticker_rows = [r for r in event_rows if str(r.get("ticker", "")).upper() == ticker]
            if not ticker_rows:
                continue
            latest_date = str(ticker_rows[0].get("period_end", ""))
            latest_rows = [r for r in ticker_rows if str(r.get("period_end", "")) == latest_date]
            for metric in metric_priority[:2]:
                for row in latest_rows:
                    if str(row.get("metric", "")) == metric:
                        selected.append(row)
                        break
            if not any(str(r.get("ticker", "")).upper() == ticker for r in selected):
                selected.append(ticker_rows[0])
        return _dedupe_rows(selected)[:8]

    period_query = dict(state.get("period_query", {}))
    max_events = 1 if str(period_query.get("period_type", "")) == "latest" else int(period_query.get("trailing_n") or 4)
    max_events = max(1, min(max_events, 8))
    selected_dates: list[str] = []
    for row in event_rows:
        event_date = str(row.get("period_end", ""))
        if event_date and event_date not in selected_dates:
            selected_dates.append(event_date)
        if len(selected_dates) >= max_events:
            break

    selected: list[dict[str, Any]] = []
    for event_date in selected_dates:
        date_rows = [r for r in event_rows if str(r.get("period_end", "")) == event_date]
        for metric in metric_priority[:3]:
            for row in date_rows:
                if str(row.get("metric", "")) == metric:
                    selected.append(row)
                    break
    return _dedupe_rows(selected)[:12]

def _build_deterministic_numeric_claims(
    state: AgentState,
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> list[dict[str, Any]]:
    task_type = str(state.get("task_type", "fact_qa"))
    analysis_scope = str(state.get("analysis_scope", ""))
    filtered = _filter_numeric_evidence_for_query(state, numeric_evidence)
    if not filtered:
        return []

    if analysis_scope == "single_company":
        selected_rows = _select_numeric_rows_for_single_company_methodology(state, filtered)
    elif task_type == "fact_qa":
        if bool(state.get("market_reaction_requested")):
            selected_rows = _select_numeric_rows_for_market_reaction(state, filtered)
            if not selected_rows:
                selected_rows = _select_numeric_rows_for_fact(state, filtered)
        else:
            selected_rows = _select_numeric_rows_for_fact(state, filtered)
    elif task_type == "trend_analysis":
        if bool(state.get("market_reaction_requested")):
            selected_rows = _select_numeric_rows_for_market_reaction(state, filtered)
            if not selected_rows:
                selected_rows = _select_numeric_rows_for_trend(state, filtered)
        else:
            selected_rows = _select_numeric_rows_for_trend(state, filtered)
    elif task_type == "company_comparison":
        if bool(state.get("market_reaction_requested")):
            selected_rows = _select_numeric_rows_for_market_reaction(state, filtered)
            if not selected_rows:
                selected_rows = _select_numeric_rows_for_comparison(state, filtered)
        else:
            selected_rows = _select_numeric_rows_for_comparison(state, filtered)
    else:
        selected_rows = []

    claims: list[dict[str, Any]] = []
    for row in selected_rows:
        evidence_id = str(row.get("evidence_id", "")).strip().upper()
        if not evidence_id:
            continue
        claims.append(
            {
                "sentence": _numeric_claim_sentence(row, lang),
                "evidence_ids": [evidence_id],
                "ticker": str(row.get("ticker", "")),
                "metric": str(row.get("metric", "")),
                "period_end": str(row.get("period_end", "")),
                "period_type": str(row.get("period_type", "")),
                "value": _to_float(row.get("value")),
                "unit": str(row.get("unit", "")),
                "source_provider": str(row.get("source_provider", "") or ""),
                "confidence": str(row.get("confidence", "") or ""),
                "extraction_method": str(row.get("extraction_method", "") or ""),
                "source_tag": str(row.get("source_tag", "") or ""),
                "reconciliation_warning": str(row.get("reconciliation_warning", "") or ""),
            }
        )
    return claims

def _validate_numeric_claims_strict(
    claims: list[dict[str, Any]],
    numeric_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for claim in claims:
        sentence = str(claim.get("sentence", "")).strip()
        evidence_ids = [str(x).strip().upper() for x in claim.get("evidence_ids", []) if str(x).strip()]
        reason = ""

        if not sentence:
            reason = "empty_sentence"
        elif not evidence_ids:
            reason = "missing_evidence_ids"
        elif any(not eid.startswith("N") for eid in evidence_ids):
            reason = "numeric_claim_with_non_numeric_evidence"
        elif _has_estimation_phrase(sentence):
            reason = "estimation_word_detected"
        elif not _contains_numeric_token(sentence):
            reason = "numeric_claim_without_number"

        missing = [eid for eid in evidence_ids if eid not in numeric_map]
        if not reason and missing:
            reason = f"missing_evidence:{','.join(missing)}"

        matched_items = [numeric_map[eid] for eid in evidence_ids if eid in numeric_map]
        claim_period_end = str(claim.get("period_end", "")).strip()
        claim_unit = str(claim.get("unit", "")).strip()
        claim_value = _to_float(claim.get("value"))

        if not reason:
            if claim_period_end and any(str(item.get("period_end", "")) != claim_period_end for item in matched_items):
                reason = "numeric_claim_period_mismatch"
            elif claim_value is None:
                reason = "numeric_claim_missing_value"
            else:
                for item in matched_items:
                    item_value = _to_float(item.get("value"))
                    if item_value is None or abs(item_value - claim_value) > 1e-9:
                        reason = "numeric_claim_value_mismatch"
                        break
            if not reason and claim_unit:
                for item in matched_items:
                    if str(item.get("unit", "")).strip() != claim_unit:
                        reason = "numeric_claim_unit_mismatch"
                        break
            if not reason and not _claim_period_matches_evidence(sentence, matched_items):
                reason = "claim_sentence_period_mismatch_with_evidence"

        if reason:
            unsupported.append(
                {
                    "claim_type": "numeric",
                    "sentence": sentence,
                    "evidence_ids": evidence_ids,
                    "reason": reason,
                }
            )
            continue

        dedupe_key = (sentence, tuple(evidence_ids))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        valid.append({"sentence": sentence, "evidence_ids": evidence_ids})

    return valid, unsupported


def _text_claim_section_mismatch(claim: dict[str, Any], evidence_item: dict[str, Any]) -> bool:
    claim_section = str(claim.get("section", "") or "").upper().strip()
    evidence_section = str(evidence_item.get("section", "") or "").upper().strip()
    if claim_section and evidence_section and claim_section != evidence_section:
        return True
    claim_type = str(claim.get("claim_type", "") or "").lower().strip()
    if claim_type in {"mda", "management_discussion"} and evidence_section not in {"ITEM_7", "ITEM_2"}:
        return True
    return False


def _wrong_company_claim(claim: dict[str, Any], evidence_item: dict[str, Any]) -> bool:
    evidence_ticker = str(evidence_item.get("ticker", "") or "").upper().strip()
    claim_company = str(claim.get("company", "") or "").upper().strip()
    if claim_company and evidence_ticker and claim_company != evidence_ticker:
        return True
    claim_entities = _company_entities(str(claim.get("sentence") or claim.get("claim") or ""))
    if evidence_ticker and claim_entities and evidence_ticker not in claim_entities:
        return True
    return False


def _quote_supported_by_snippet(quote: str, snippet: str) -> bool:
    normalized_quote = _normalized_for_substring(quote)
    normalized_snippet = _normalized_for_substring(snippet)
    if not normalized_quote:
        return False
    if normalized_quote in normalized_snippet:
        return True
    quote_terms = _text_support_terms(normalized_quote)
    snippet_terms = _text_support_terms(normalized_snippet)
    if not quote_terms:
        return False
    return len(quote_terms & snippet_terms) >= max(2, min(len(quote_terms), 4))


def _validate_text_claims_enhanced(
    claims: list[dict[str, Any]],
    evidence_map: dict[str, dict[str, Any]],
    validation_context: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    context = dict(validation_context or {})
    analysis_scope = str(context.get("analysis_scope", "") or "")
    requirement_dimension_map = {
        str(k): str(v)
        for k, v in dict(context.get("requirement_dimension_map", {}) or {}).items()
        if str(k).strip() and str(v).strip()
    }

    for claim in claims:
        sentence = str(claim.get("sentence") or claim.get("claim") or "").strip()
        citation_ref = str(claim.get("citation_ref", "") or "").strip().upper()
        evidence_ids = [str(x).strip().upper() for x in claim.get("evidence_ids", []) if str(x).strip()]
        if not citation_ref and evidence_ids:
            citation_ref = evidence_ids[0]
        if citation_ref and citation_ref not in evidence_ids:
            evidence_ids = [citation_ref] + evidence_ids

        if not sentence:
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="claim_not_supported_by_snippet",
                    message="Text claim is empty.",
                    citation_ref=citation_ref,
                )
            )
            continue
        if not citation_ref:
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="missing_citation_ref",
                    message="Text claim must include a T* citation_ref.",
                    citation_ref="",
                    claim_entities=_text_claim_entities(sentence),
                )
            )
            continue
        if not citation_ref.startswith("T") or citation_ref not in evidence_map:
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="invalid_citation_ref",
                    message=f"Text claim citation_ref '{citation_ref}' does not reference available text evidence.",
                    citation_ref=citation_ref,
                    claim_entities=_text_claim_entities(sentence),
                )
            )
            continue

        evidence_item = evidence_map[citation_ref]
        req_ids = [
            str(x).strip()
            for x in list(evidence_item.get("requirement_ids", []) or []) + [evidence_item.get("requirement_id")]
            if str(x).strip()
        ]
        dimension_id = str(claim.get("dimension_id") or evidence_item.get("dimension_id") or "").strip()
        if not dimension_id:
            for req_id in req_ids:
                if req_id in requirement_dimension_map:
                    dimension_id = requirement_dimension_map[req_id]
                    break
        snippet = str(evidence_item.get("supporting_snippet") or evidence_item.get("text_snippet") or "")
        quote = str(claim.get("supporting_quote", "") or "")
        claim_terms = _text_support_terms(sentence)
        snippet_terms = _text_support_terms(snippet)
        quote_terms = _text_support_terms(quote)
        supporting_terms = {str(t).lower() for t in evidence_item.get("supporting_terms", []) or [] if str(t).strip()}
        section_terms = _section_support_terms(str(evidence_item.get("section", "")))
        available_terms = snippet_terms | quote_terms | supporting_terms | section_terms
        missing_terms = claim_terms - available_terms
        claim_entities = _text_claim_entities(sentence)
        snippet_entities = _text_claim_entities(snippet)

        if _has_investment_advice_wording(sentence):
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="claim_safety_violation",
                    message="Text claim contains investment-advice wording.",
                    evidence_item=evidence_item,
                    citation_ref=citation_ref,
                    supporting_terms=available_terms,
                    missing_support_terms=missing_terms,
                    claim_entities=claim_entities,
                    snippet_entities=snippet_entities,
                )
            )
            continue
        if any(term in str(sentence).lower() for term in _DETERMINISTIC_PREDICTION_TERMS):
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="claim_safety_violation",
                    message="Text claim contains deterministic prediction wording.",
                    evidence_item=evidence_item,
                    citation_ref=citation_ref,
                    supporting_terms=available_terms,
                    missing_support_terms=missing_terms,
                    claim_entities=claim_entities,
                    snippet_entities=snippet_entities,
                )
            )
            continue
        if _wrong_company_claim(claim, evidence_item):
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="claim_uses_wrong_company",
                    message="Claim company does not match the cited text evidence company.",
                    evidence_item=evidence_item,
                    citation_ref=citation_ref,
                    supporting_terms=available_terms,
                    missing_support_terms=missing_terms,
                    claim_entities=claim_entities,
                    snippet_entities=snippet_entities,
                )
            )
            continue
        if _text_claim_section_mismatch(claim, evidence_item):
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="claim_section_mismatch",
                    message="Claim section does not match the cited text evidence section.",
                    evidence_item=evidence_item,
                    citation_ref=citation_ref,
                    supporting_terms=available_terms,
                    missing_support_terms=missing_terms,
                    claim_entities=claim_entities,
                    snippet_entities=snippet_entities,
                )
            )
            continue
        claim_numbers = _numeric_tokens(sentence)
        if claim_numbers and not claim_numbers <= (_numeric_tokens(snippet) | _numeric_tokens(quote)):
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="claim_contains_unsupported_number",
                    message="Text claim contains a number not present in the cited snippet or supporting quote.",
                    evidence_item=evidence_item,
                    citation_ref=citation_ref,
                    supporting_terms=available_terms,
                    missing_support_terms=missing_terms,
                    claim_entities=claim_entities,
                    snippet_entities=snippet_entities,
                )
            )
            continue

        quote_supports = _quote_supported_by_snippet(quote, snippet)
        enough_overlap = bool(claim_terms & available_terms) and len(missing_terms) <= max(3, len(claim_terms) // 2)
        single_company_text_context = (
            analysis_scope == "single_company"
            and dimension_id in {"business_model", "moat_and_competitive_risk"}
        )
        if single_company_text_context and not enough_overlap and bool(claim_terms & available_terms):
            # Open-ended single-company analysis often rewrites SEC English into Chinese
            # summaries. Allow a little more semantic distance, while still requiring
            # concrete overlap with the cited snippet and preserving hard failures above.
            enough_overlap = len(missing_terms) <= max(6, len(claim_terms) * 2 // 3)
        if not quote_supports and not enough_overlap:
            unsupported.append(
                _text_validation_error(
                    claim=claim,
                    code="claim_not_supported_by_snippet",
                    message="Claim terms are not sufficiently supported by the cited snippet.",
                    evidence_item=evidence_item,
                    citation_ref=citation_ref,
                    supporting_terms=available_terms,
                    missing_support_terms=missing_terms,
                    claim_entities=claim_entities,
                    snippet_entities=snippet_entities,
                )
            )
            continue

        warning_codes: list[str] = []
        if len(claim_terms - _TEXT_CLAIM_GENERIC_TERMS) <= 2 or len(missing_terms) > 0:
            warning_codes.append("claim_too_generic")
        for code in warning_codes:
            warnings.append(
                {
                    "claim_type": "text",
                    "sentence": sentence,
                    "evidence_ids": [citation_ref],
                    "reason": code,
                    "validation_error_code": code,
                    "validation_error_message": "Claim is broad but allowed because the cited snippet provides support.",
                    "claim_raw_text": sentence,
                    "source_requirement_id": str(evidence_item.get("requirement_id", "")),
                    "source_snippet_id": citation_ref,
                    "citation_ref": citation_ref,
                    "supporting_terms": sorted(available_terms),
                    "missing_support_terms": sorted(missing_terms),
                    "claim_entities": sorted(claim_entities),
                    "snippet_entities": sorted(snippet_entities),
                    "company": str(claim.get("company") or evidence_item.get("ticker") or ""),
                }
            )

        dedupe_key = (sentence, tuple([citation_ref]))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        valid.append(
            {
                "sentence": sentence,
                "claim": sentence,
                "evidence_ids": [citation_ref],
                "citation_ref": citation_ref,
                "company": str(claim.get("company") or evidence_item.get("ticker") or ""),
                "claim_type": str(claim.get("claim_type") or ""),
                "supporting_quote": quote or snippet[:240],
                "confidence": str(claim.get("confidence") or "medium"),
                "dimension_id": dimension_id,
                "source_requirement_id": str(evidence_item.get("requirement_id", "")),
                "claim_source": str(claim.get("claim_source") or claim.get("generated_by") or ""),
                "validation_warnings": warning_codes,
            }
        )

    return valid, unsupported, warnings


def _validate_claims(
    claims: list[dict[str, Any]],
    claim_type: str,
    evidence_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if claim_type == "text":
        valid_text, unsupported_text, _warnings = _validate_text_claims_enhanced(claims, evidence_map)
        return valid_text, unsupported_text

    valid: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for claim in claims:
        sentence = str(claim.get("sentence", "")).strip()
        evidence_ids = [str(x).strip().upper() for x in claim.get("evidence_ids", []) if str(x).strip()]
        reason = ""

        if not sentence:
            reason = "empty_sentence"
        elif not evidence_ids:
            reason = "missing_evidence_ids"
        elif claim_type == "numeric":
            if not _contains_numeric_token(sentence):
                reason = "numeric_claim_without_number"
            elif _has_estimation_phrase(sentence):
                reason = "estimation_word_detected"
            elif any(not eid.startswith("N") for eid in evidence_ids):
                reason = "numeric_claim_with_non_numeric_evidence"
            else:
                matched_items = [evidence_map[eid] for eid in evidence_ids if eid in evidence_map]
                if not _claim_period_matches_evidence(sentence, matched_items):
                    reason = "claim_sentence_period_mismatch_with_evidence"
        elif claim_type == "text":
            if _contains_numeric_token(sentence):
                reason = "text_claim_contains_number"
            elif any(not eid.startswith("T") for eid in evidence_ids):
                reason = "text_claim_with_non_text_evidence"

        missing = [eid for eid in evidence_ids if eid not in evidence_map]
        if not reason and missing:
            reason = f"missing_evidence:{','.join(missing)}"
        if not reason and claim_type == "text":
            matched_items = [evidence_map[eid] for eid in evidence_ids if eid in evidence_map]
            if not _text_claim_supported_by_evidence(sentence, matched_items):
                reason = "snippet_unsupported"

        if reason:
            unsupported.append(
                {
                    "claim_type": claim_type,
                    "sentence": sentence,
                    "evidence_ids": evidence_ids,
                    "reason": reason,
                }
            )
            continue

        dedupe_key = (sentence, tuple(evidence_ids))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        valid.append({"sentence": sentence, "evidence_ids": evidence_ids})

    return valid, unsupported


def build_evidence_bundle(state: AgentState) -> EvidenceBundle:
    return EvidenceBundle(**_build_evidence_bundle(state))


validate_numeric_claims_strict = _validate_numeric_claims_strict
validate_claims = _validate_claims
validate_text_claims_enhanced = _validate_text_claims_enhanced
collect_financial_rows = _collect_financial_rows
collect_event_rows = _collect_event_rows
text_claim_supported_by_evidence = _text_claim_supported_by_evidence
