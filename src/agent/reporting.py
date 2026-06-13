"""Assemble structured company-analysis reports from existing synthesis output."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Mapping

from src.agent.answer_contract import check_answer_contract
from src.agent.report_models import CompanyAnalysisReport, ReportSection


SECTION_ORDER: tuple[tuple[str, str, str | None], ...] = (
    ("executive_summary", "Executive Summary", None),
    ("business_overview", "Business Overview", "business_model"),
    ("revenue_quality", "Revenue Quality", "revenue_quality"),
    ("profitability", "Profitability", "profitability_quality"),
    ("cash_flow_quality", "Cash Flow Quality", "cash_flow_quality"),
    ("balance_sheet_safety", "Balance Sheet Safety", "balance_sheet_and_capital_intensity"),
    ("risk_factors", "Risk Factors", "moat_and_competitive_risk"),
    ("valuation_boundary", "Valuation Boundary", "valuation_and_risk_boundary"),
    ("market_reaction", "Market Reaction / Filing Event Window", None),
    ("limitations", "Limitations and Missing Evidence", None),
)

_OPEN_REPORT_TERMS = (
    "分析",
    "深度分析",
    "基本面",
    "报告",
    "怎么看",
    "analyze",
    "analysis",
    "report",
    "fundamental",
    "overview",
)

_NARROW_TERMS = (
    "现金流",
    "估值",
    "资产负债",
    "风险",
    "收入",
    "营收",
    "盈利",
    "利润",
    "cash flow",
    "valuation",
    "balance sheet",
    "risk",
    "revenue",
    "profit",
    "margin",
)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _lang(state: Mapping[str, Any]) -> str:
    query = str(state.get("user_query") or "")
    return "zh" if re.search(r"[\u4e00-\u9fff]", query) else "en"


def should_build_company_report(state: Mapping[str, Any]) -> bool:
    """Return true for broad single-company analysis requests only."""
    if str(state.get("analysis_scope") or "") != "single_company":
        return False
    if str(state.get("task_type") or "") != "report_summary":
        return False
    if str(state.get("answer_mode") or "") not in {"analytical", "cautious_outlook"}:
        return False
    query = str(state.get("user_query") or "").lower()
    if not any(term in query for term in _OPEN_REPORT_TERMS):
        return False
    methodology_intent = str(state.get("methodology_intent") or "").strip()
    if methodology_intent and methodology_intent not in {"overview", "single_company_overview", "none"}:
        return False
    if any(term in query for term in _NARROW_TERMS) and not any(term in query for term in ("基本面", "报告", "fundamental", "report")):
        return False
    return True


def _company_name(state: Mapping[str, Any]) -> tuple[str, str]:
    ticker = ""
    companies = _as_list(state.get("companies"))
    if companies:
        ticker = str(companies[0]).upper().strip()
    summary = _as_dict(state.get("query_understanding_summary"))
    for item in _as_list(summary.get("companies")):
        if isinstance(item, dict) and str(item.get("ticker") or "").upper().strip() == ticker:
            return str(item.get("company_name") or item.get("name") or ticker), ticker
    return ticker, ticker


def _dimension_status_map(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    for source in (
        state.get("dimension_status_by_id"),
        state.get("dimension_status_map"),
        _as_dict(state.get("evidence_packet")).get("dimension_status_by_id"),
        _as_dict(state.get("evidence_packet")).get("dimension_status_map"),
        _as_dict(state.get("evidence_sufficiency")).get("dimension_status_by_id"),
        _as_dict(state.get("evidence_sufficiency")).get("dimension_status_map"),
    ):
        if isinstance(source, dict) and source:
            return {str(key): _as_dict(value) for key, value in source.items()}
    return {}


def _methodology_answer(state: Mapping[str, Any]) -> dict[str, Any]:
    synthesis = _as_dict(state.get("synthesis"))
    output = _as_dict(state.get("output"))
    view = _as_dict(output.get("view"))
    return (
        _as_dict(synthesis.get("methodology_answer"))
        or _as_dict(view.get("methodology_answer"))
        or _as_dict(output.get("methodology_answer"))
    )


def _methodology_report(state: Mapping[str, Any]) -> dict[str, Any]:
    synthesis = _as_dict(state.get("synthesis"))
    output = _as_dict(state.get("output"))
    view = _as_dict(output.get("view"))
    return (
        _as_dict(synthesis.get("methodology_report"))
        or _as_dict(view.get("methodology_report"))
        or _as_dict(output.get("methodology_report"))
    )


def _dimension_sections_by_id(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    answer = _methodology_answer(state)
    out: dict[str, dict[str, Any]] = {}
    for item in _as_list(answer.get("dimension_sections")):
        if isinstance(item, dict) and str(item.get("dimension_id") or "").strip():
            out[str(item.get("dimension_id"))] = dict(item)
    report = _methodology_report(state)
    for item in _as_list(report.get("dimension_analyses")):
        if isinstance(item, dict) and str(item.get("dimension_id") or "").strip():
            out.setdefault(str(item.get("dimension_id")), dict(item))
    return out


def _section_refs(item: Mapping[str, Any], status: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("evidence_refs", "key_evidence_ids", "supporting_evidence_ids"):
        refs.extend(str(ref) for ref in _as_list(item.get(key)) if str(ref).strip())
        refs.extend(str(ref) for ref in _as_list(status.get(key)) if str(ref).strip())
    return _dedupe(refs)


def _section_limitations(item: Mapping[str, Any], status: Mapping[str, Any], lang: str) -> list[str]:
    limitations: list[str] = []
    for key in ("limitations", "limitation"):
        value = item.get(key)
        if isinstance(value, list):
            limitations.extend(str(x) for x in value if str(x).strip())
        elif str(value or "").strip():
            limitations.append(str(value))
    for key in ("limitations", "caveats"):
        limitations.extend(str(x) for x in _as_list(status.get(key)) if str(x).strip())
    missing = _as_list(status.get("required_missing")) + _as_list(status.get("enhanced_missing"))
    if missing:
        if lang == "zh":
            limitations.append("缺少或仅部分覆盖：" + ", ".join(str(x) for x in missing[:6]))
        else:
            limitations.append("Missing or partially covered: " + ", ".join(str(x) for x in missing[:6]))
    return _dedupe(limitations)


def _confidence(status: str) -> str:
    if status == "satisfied":
        return "high"
    if status == "partial":
        return "medium"
    return "low"


def _missing_markdown(title: str, lang: str) -> str:
    if lang == "zh":
        return f"当前证据不足以支持对 {title} 给出完整结论。该部分保留为缺失证据边界。"
    return f"Current evidence is insufficient to support a complete {title} conclusion. This section is retained as a missing-evidence boundary."


def _dimension_section(state: Mapping[str, Any], section_id: str, title: str, dimension_id: str, lang: str) -> ReportSection:
    status_map = _dimension_status_map(state)
    status = status_map.get(dimension_id, {})
    status_value = str(status.get("status") or "missing")
    if status_value not in {"satisfied", "partial", "missing"}:
        status_value = "missing"
    item = _dimension_sections_by_id(state).get(dimension_id, {})
    summary = str(item.get("summary") or item.get("claim") or "").strip()
    refs = _section_refs(item, status)
    limitations = _section_limitations(item, status, lang)
    if not summary:
        summary = _missing_markdown(title, lang) if status_value == "missing" else (title if lang == "en" else f"{title} 已有部分证据支持。")
    refs_text = " ".join(f"[{ref}]" for ref in refs[:6])
    markdown = f"{summary} {refs_text}".strip()
    if limitations and status_value != "satisfied":
        label = "限制" if lang == "zh" else "Limitations"
        markdown = f"{markdown}\n\n{label}: " + "; ".join(limitations[:4])
    return ReportSection(
        section_id=section_id,
        title=title,
        markdown=markdown,
        section_status=status_value,  # type: ignore[arg-type]
        key_evidence_ids=refs,
        citations=refs,
        limitations=limitations,
        confidence=_confidence(status_value),  # type: ignore[arg-type]
        contract_status="passed",
    )


def _executive_section(state: Mapping[str, Any], lang: str) -> ReportSection:
    synthesis = _as_dict(state.get("synthesis"))
    answer = str(synthesis.get("short_answer") or state.get("final_answer") or "").strip()
    if not answer:
        answer = "基于现有证据生成企业分析摘要。" if lang == "zh" else "Company analysis summary based on available evidence."
    refs = _dedupe(re.findall(r"\[([NT]\d+)\]", answer))
    status_map = _dimension_status_map(state)
    statuses = {str(item.get("status") or "") for item in status_map.values() if isinstance(item, dict)}
    status = "satisfied" if "satisfied" in statuses else "partial" if "partial" in statuses else "missing"
    return ReportSection(
        section_id="executive_summary",
        title="Executive Summary",
        markdown=answer,
        section_status=status,  # type: ignore[arg-type]
        key_evidence_ids=refs,
        citations=refs,
        limitations=[],
        confidence=_confidence(status),  # type: ignore[arg-type]
        contract_status="passed",
    )


def _market_reaction_section(state: Mapping[str, Any], lang: str) -> ReportSection:
    output = _as_dict(state.get("output"))
    market = _as_dict(output.get("market_reaction"))
    evidence = _as_list(state.get("market_reaction_evidence"))
    highlights = [str(x) for x in _as_list(market.get("highlights")) if str(x).strip()]
    refs = _dedupe(
        [
            str(item.get("evidence_id"))
            for item in evidence
            if isinstance(item, dict) and str(item.get("evidence_id") or "").strip()
        ]
    )
    if highlights or evidence:
        markdown = "\n".join(f"- {item}" for item in highlights[:5]) or (
            "已找到财报事件窗口证据。" if lang == "zh" else "Filing event-window evidence is available."
        )
        status = "satisfied"
    else:
        markdown = (
            "本次开放式企业分析没有足够的财报事件窗口证据；该部分不影响基本面报告结论。"
            if lang == "zh"
            else "This open-ended company analysis does not have enough filing event-window evidence; this does not block the fundamental report."
        )
        status = "missing"
    return ReportSection(
        section_id="market_reaction",
        title="Market Reaction / Filing Event Window",
        markdown=markdown,
        section_status=status,  # type: ignore[arg-type]
        key_evidence_ids=refs,
        citations=refs,
        limitations=[] if status == "satisfied" else [markdown],
        confidence=_confidence(status),  # type: ignore[arg-type]
        contract_status="passed",
    )


def _limitations_section(state: Mapping[str, Any], lang: str) -> ReportSection:
    output = _as_dict(state.get("output"))
    limitations: list[str] = []
    for item in _as_list(output.get("limitations")):
        if isinstance(item, dict):
            limitations.append(str(item.get("message") or item.get("code") or ""))
        else:
            limitations.append(str(item))
    for item in _as_list(_methodology_answer(state).get("limitations")):
        limitations.append(str(item))
    for item in _as_list(_methodology_report(state).get("limitations")):
        limitations.append(str(item))
    if not limitations:
        limitations.append(
            "本报告仅基于当前可用 SEC filings、结构化财务事实和程序化指标，不构成投资建议。"
            if lang == "zh"
            else "This report uses currently available SEC filings, structured financial facts, and programmatic metrics only. It is not investment advice."
        )
    markdown = "\n".join(f"- {item}" for item in _dedupe(limitations)[:8])
    return ReportSection(
        section_id="limitations",
        title="Limitations and Missing Evidence",
        markdown=markdown,
        section_status="partial",
        key_evidence_ids=[],
        citations=[],
        limitations=_dedupe(limitations),
        confidence="medium",
        contract_status="passed",
    )


def _assemble_markdown(title: str, sections: list[ReportSection], lang: str) -> str:
    intro = (
        "> 基于可用 SEC filings、结构化财务事实和程序化指标生成；不构成投资建议。"
        if lang == "zh"
        else "> Evidence-grounded analysis based on available SEC filings, structured financial facts, and programmatic metrics. This is not investment advice."
    )
    lines = [f"# {title}", "", intro]
    for section in sections:
        lines.extend(["", f"## {section.title}", section.markdown.strip()])
    citations = _dedupe([ref for section in sections for ref in section.citations])
    if citations:
        lines.extend(["", "## Citations", *[f"- [{ref}]" for ref in citations]])
    return "\n".join(line for line in lines if line is not None).strip()


def build_company_analysis_report(state: Mapping[str, Any]) -> dict[str, Any]:
    """Build and validate output.report for broad single-company analysis."""
    if not should_build_company_report(state):
        return {}
    lang = _lang(state)
    company, ticker = _company_name(state)
    title = f"{ticker} Fundamental Analysis Report" if ticker else "Fundamental Analysis Report"
    sections: list[ReportSection] = []
    for section_id, section_title, dimension_id in SECTION_ORDER:
        if section_id == "executive_summary":
            sections.append(_executive_section(state, lang))
        elif section_id == "market_reaction":
            sections.append(_market_reaction_section(state, lang))
        elif section_id == "limitations":
            sections.append(_limitations_section(state, lang))
        elif dimension_id:
            sections.append(_dimension_section(state, section_id, section_title, dimension_id, lang))
    markdown = _assemble_markdown(title, sections, lang)
    report = CompanyAnalysisReport(
        title=title,
        company=company,
        ticker=ticker,
        period=str(_as_dict(state.get("resolved_period_context")).get("label") or "latest available filings"),
        report_type="company_analysis",
        sections=sections,
        overall_limitations=_dedupe([item for section in sections for item in section.limitations]),
        citations=_dedupe([ref for section in sections for ref in section.citations]),
        contract_status="passed",
        markdown=markdown,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
    contract = check_answer_contract(markdown, state, scope="report")
    if contract.route == "pass":
        report.contract_status = "passed"
        return report.model_dump()
    # Do not expose a failed report draft.
    return {
        "title": title,
        "company": company,
        "ticker": ticker,
        "period": report.period,
        "report_type": "company_analysis",
        "sections": [],
        "overall_limitations": [
            contract.public_summary or "Report did not pass runtime evidence checks.",
        ],
        "citations": [],
        "contract_status": "blocked" if contract.route == "blocked" else "failed",
        "markdown": "",
        "generated_at": report.generated_at,
    }
