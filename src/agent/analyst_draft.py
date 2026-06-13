"""LLM-assisted analyst draft generation and synthesis projection."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.agent.analyst_prompts import GENERATE_ANALYST_DRAFT, REVISE_ANALYST_DRAFT
from src.agent.draft_validation import summarize_draft_validation, validate_analyst_draft
from src.agent.llm import _get_llm, _parse_json_response
from src.agent.types import AnalystDraft


def _packet_citation_refs(packet: dict[str, Any]) -> set[str]:
    refs = {str(item.get("evidence_id", "")).strip() for item in packet.get("numeric_table", []) or [] if isinstance(item, dict)}
    refs |= {str(item.get("evidence_id", "")).strip() for item in packet.get("text_snippets", []) or [] if isinstance(item, dict)}
    refs |= {str(item.get("evidence_id", "")).strip() for item in packet.get("citations", []) or [] if isinstance(item, dict)}
    return {ref for ref in refs if ref}


def build_methodology_context(evidence_packet: dict[str, Any]) -> dict[str, Any]:
    """Extract the methodology contract that the LLM must follow."""
    packet = dict(evidence_packet or {})
    selected = packet.get("selected_framework", {})
    if isinstance(selected, dict):
        selected_framework = str(selected.get("framework_id") or selected.get("id") or "").strip()
    else:
        selected_framework = str(selected or "").strip()
    if not selected_framework:
        selected_framework = str(
            dict(packet.get("selected_framework_summary", {}) or {}).get("framework_id", "")
        ).strip()

    dimension_status_map = dict(packet.get("dimension_status_map", {}) or {})
    if not dimension_status_map:
        dimension_status_map = dict(
            dict(packet.get("dimension_sufficiency", {}) or {}).get("dimension_status_map", {}) or {}
        )
    if not dimension_status_map:
        dimension_status_map = dict(
            dict(packet.get("missing_evidence_summary", {}) or {}).get("dimension_status_map", {}) or {}
        )
    active_dimensions = [
        str(item).strip()
        for item in packet.get("active_dimensions", []) or []
        if str(item).strip()
    ] or [str(key) for key in dimension_status_map.keys()]

    return {
        "selected_framework": selected_framework,
        "active_dimensions": active_dimensions,
        "dimension_status_map": dimension_status_map,
        "red_flags": list(packet.get("red_flags", []) or []),
        "missing_evidence_flags": list(packet.get("missing_evidence_flags", []) or []),
        "allowed_claims": list(packet.get("allowed_claims", []) or []),
        "forbidden_claims": list(packet.get("forbidden_claims", []) or []),
    }


def _trim_text(text: Any, limit: int = 800) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _compact_numeric_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "evidence_id",
        "ticker",
        "metric",
        "display_label",
        "period_type",
        "period_end",
        "value",
        "display_value",
        "unit",
        "source_provider",
        "confidence",
        "source_tag",
        "requirement_id",
        "dependencies",
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
    )
    return {key: row.get(key) for key in keys if row.get(key) not in (None, "", [], {})}


def _compact_text_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "evidence_id": row.get("evidence_id"),
        "ticker": row.get("ticker"),
        "form_type": row.get("form_type"),
        "fiscal_period": row.get("fiscal_period"),
        "section": row.get("section"),
        "claim": _trim_text(row.get("claim"), 320),
        "text_snippet": _trim_text(row.get("text_snippet"), 700),
        "supporting_snippet": _trim_text(row.get("supporting_snippet"), 500),
        "requirement_id": row.get("requirement_id"),
        "dimension_id": row.get("dimension_id"),
    }
    return {key: value for key, value in out.items() if value not in (None, "", [], {})}


def _compact_theme(theme: dict[str, Any]) -> dict[str, Any]:
    keys = ("theme_code", "label", "evidence_refs", "companies", "snippet_count")
    return {key: theme.get(key) for key in keys if theme.get(key) not in (None, "", [], {})}


def _compact_dimension_item(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dimension_id",
        "name",
        "status",
        "limitation",
        "numeric_evidence_refs",
        "text_evidence_refs",
        "evidence_refs",
        "allowed_claims",
        "forbidden_claims",
    )
    return {key: item.get(key) for key in keys if item.get(key) not in (None, "", [], {})}


def _compact_status_item(item: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "dimension_id",
        "dimension_name",
        "status",
        "limitation",
        "allowed_claims",
        "forbidden_claims",
        "satisfied_requirements",
        "missing_requirements",
    )
    return {key: item.get(key) for key in keys if item.get(key) not in (None, "", [], {})}


def _compact_missing_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "overall_status",
        "degradation_reason",
        "missing_requirements",
        "partial_requirements",
        "satisfied_dimensions",
        "covered_dimensions",
        "partial_dimensions",
        "missing_dimensions",
        "dimension_coverage_rate",
        "weighted_dimension_coverage_rate",
        "framework_sufficiency_status",
    )
    return {key: summary.get(key) for key in keys if summary.get(key) not in (None, "", [], {})}


def _compact_evidence_packet_for_prompt(packet: dict[str, Any]) -> dict[str, Any]:
    """Keep validation rich, but avoid sending duplicated trace payload to the LLM."""
    packet = dict(packet or {})
    dimension_status_map = {
        str(key): _compact_status_item(dict(value))
        for key, value in dict(packet.get("dimension_status_map") or packet.get("dimension_status_by_id") or {}).items()
        if isinstance(value, dict)
    }
    return {
        "user_query": packet.get("user_query"),
        "task_type": packet.get("task_type"),
        "answer_mode": packet.get("answer_mode"),
        "safety_intent": packet.get("safety_intent"),
        "analysis_scope": packet.get("analysis_scope"),
        "time_policy": packet.get("time_policy"),
        "period_scope": packet.get("period_scope"),
        "selected_framework_summary": packet.get("selected_framework_summary") or {},
        "active_dimensions": list(packet.get("active_dimensions", []) or []),
        "dimension_summary": [
            _compact_dimension_item(dict(item))
            for item in packet.get("dimension_summary", []) or []
            if isinstance(item, dict)
        ],
        "dimension_status_map": dimension_status_map,
        "numeric_table": [
            _compact_numeric_row(dict(row))
            for row in packet.get("numeric_table", []) or []
            if isinstance(row, dict)
        ],
        "comparison_table": [
            dict(row)
            for row in packet.get("comparison_table", []) or []
            if isinstance(row, dict)
        ],
        "text_snippets": [
            _compact_text_row(dict(row))
            for row in packet.get("text_snippets", []) or []
            if isinstance(row, dict)
        ],
        "available_metrics_by_dimension": dict(packet.get("available_metrics_by_dimension", {}) or {}),
        "missing_metrics_by_dimension": dict(packet.get("missing_metrics_by_dimension", {}) or {}),
        "valuation_period_basis": dict(packet.get("valuation_period_basis", {}) or {}),
        "grouped_risk_themes": [
            _compact_theme(dict(item))
            for item in packet.get("grouped_risk_themes", []) or []
            if isinstance(item, dict)
        ],
        "grouped_business_themes": [
            _compact_theme(dict(item))
            for item in packet.get("grouped_business_themes", []) or []
            if isinstance(item, dict)
        ],
        "provenance_notes": list(packet.get("provenance_notes", []) or []),
        "missing_evidence_summary": _compact_missing_summary(dict(packet.get("missing_evidence_summary", {}) or {})),
        "red_flags": list(packet.get("red_flags", []) or []),
        "missing_evidence_flags": list(packet.get("missing_evidence_flags", []) or []),
        "allowed_claims": list(packet.get("allowed_claims", []) or []),
        "forbidden_claims": list(packet.get("forbidden_claims", []) or []),
        "limitations": list(packet.get("limitations", []) or []),
    }


def _json_for_prompt(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _item_dicts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _dimension_analyses(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        evidence_refs = item.get("evidence_refs", item.get("citation_refs", []))
        if isinstance(evidence_refs, str):
            evidence_refs = [evidence_refs]
        out.append(
            {
                "dimension_id": str(item.get("dimension_id", "")).strip(),
                "status": str(item.get("status", "")).strip(),
                "claim": str(item.get("claim", item.get("statement", ""))).strip(),
                "evidence_refs": [str(ref).strip() for ref in evidence_refs or [] if str(ref).strip()],
            }
        )
    return out


def _normalize_draft(raw: dict[str, Any]) -> AnalystDraft:
    return AnalystDraft(
        framework_summary=str(raw.get("framework_summary", "") or "").strip(),
        dimension_analyses=_dimension_analyses(raw.get("dimension_analyses", [])),
        overall_judgment=str(raw.get("overall_judgment", "") or "").strip(),
        methodology_counterpoints=_string_list(raw.get("methodology_counterpoints", [])),
        methodology_limitations=_string_list(raw.get("methodology_limitations", [])),
        follow_up_metrics=_string_list(raw.get("follow_up_metrics", [])),
        tentative_conclusion=dict(raw.get("tentative_conclusion", {}) or {}),
        decision_basis=_item_dicts(raw.get("decision_basis", [])),
        supporting_points=_item_dicts(raw.get("supporting_points", [])),
        counterpoints=_item_dicts(raw.get("counterpoints", [])),
        risk_tradeoffs=_item_dicts(raw.get("risk_tradeoffs", [])),
        uncertainty_notes=_item_dicts(raw.get("uncertainty_notes", [])),
        citation_refs=[str(x) for x in raw.get("citation_refs", []) or [] if str(x).strip()],
        safety_notes=_item_dicts(raw.get("safety_notes", [])),
    )


def generate_analyst_draft(
    *,
    evidence_packet: dict[str, Any],
    answer_language: str,
    synthesis_mode: str,
    comparison_judgment_frame: dict[str, Any] | None = None,
    methodology_context: dict[str, Any] | None = None,
    prior_draft: dict[str, Any] | None = None,
    repair_instructions: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed_refs = sorted(_packet_citation_refs(evidence_packet))
    if not allowed_refs:
        return {}, [
            {
                "claim_type": "system",
                "sentence": "",
                "evidence_ids": [],
                "reason": "analyst_draft_no_packet_evidence",
            }
        ]

    kwargs = {
        "allowed_citation_refs": ", ".join(allowed_refs),
        "answer_language": answer_language,
        "synthesis_mode": synthesis_mode,
        "methodology_context_json": _json_for_prompt(
            methodology_context or build_methodology_context(evidence_packet),
        ),
        "evidence_packet_json": _json_for_prompt(_compact_evidence_packet_for_prompt(evidence_packet)),
        "comparison_judgment_frame_json": _json_for_prompt(comparison_judgment_frame or {}),
    }
    prompt_text = (
        REVISE_ANALYST_DRAFT.format(
            prior_draft_json=_json_for_prompt(prior_draft or {}),
            repair_instructions_json=_json_for_prompt(list(repair_instructions or [])),
            **kwargs,
        )
        if repair_instructions
        else GENERATE_ANALYST_DRAFT.format(**kwargs)
    )
    system_msg = "You are a strict financial analyst drafter. Return valid JSON only, no markdown."
    issues: list[dict[str, Any]] = []
    max_tokens = max(512, int(settings.analyst_draft_max_tokens or 1800))
    llm = _get_llm(reasoning=True, temperature=0.1, max_tokens=max_tokens)
    try:
        response = llm.invoke([SystemMessage(content=system_msg), HumanMessage(content=prompt_text)])
        raw_response = re.sub(r"<think>.*?</think>", "", response.content or "", flags=re.DOTALL).strip()
        parsed = _parse_json_response(raw_response)
        if not parsed:
            issues.append(
                {
                    "claim_type": "system",
                    "sentence": raw_response[:200],
                    "evidence_ids": [],
                    "reason": "analyst_draft_invalid_json",
                }
            )
            return {}, issues
        return _normalize_draft(parsed).model_dump(exclude_none=True), issues
    except Exception as exc:
        issues.append(
            {
                "claim_type": "system",
                "sentence": "",
                "evidence_ids": [],
                "reason": f"analyst_draft_model_call_failed:{exc}",
            }
        )
        return {}, issues


def project_analyst_draft_to_synthesis(accepted_draft: dict[str, Any]) -> dict[str, Any]:
    draft = dict(accepted_draft or {})
    conclusion = dict(draft.get("tentative_conclusion", {}) or {})
    analysis: list[dict[str, Any]] = []
    for item in draft.get("dimension_analyses", []) or []:
        if not isinstance(item, dict):
            continue
        sentence = str(item.get("claim", "")).strip()
        if not sentence:
            continue
        analysis.append({"sentence": sentence, "claim_ids": list(item.get("evidence_refs", []) or [])})
    for field_name in ("decision_basis", "supporting_points", "counterpoints"):
        for item in draft.get(field_name, []) or []:
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("statement", "")).strip()
            if not sentence:
                continue
            analysis.append({"sentence": sentence, "claim_ids": list(item.get("citation_refs", []) or [])})
    risks: list[dict[str, Any]] = []
    for field_name in ("risk_tradeoffs", "uncertainty_notes", "safety_notes"):
        for item in draft.get(field_name, []) or []:
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("statement", "")).strip()
            if not sentence:
                continue
            risks.append({"sentence": sentence, "claim_ids": list(item.get("citation_refs", []) or [])})
    return {
        "short_answer": str(conclusion.get("statement") or draft.get("overall_judgment") or "").strip(),
        "analysis": analysis,
        "risks_or_uncertainties": risks,
    }


def summarize_analyst_draft(
    draft: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    draft = dict(draft or {})
    validation = dict(validation or {})
    accepted = dict(validation.get("accepted_draft", {}) or {})
    base = {
        "framework_summary": str(draft.get("framework_summary", "")),
        "dimension_analysis_count": len(draft.get("dimension_analyses", []) or []),
        "methodology_counterpoint_count": len(draft.get("methodology_counterpoints", []) or []),
        "methodology_limitation_count": len(draft.get("methodology_limitations", []) or []),
        "follow_up_metric_count": len(draft.get("follow_up_metrics", []) or []),
        "tentative_conclusion": str((draft.get("tentative_conclusion", {}) or {}).get("statement", "")),
        "decision_basis_count": len(draft.get("decision_basis", []) or []),
        "supporting_points_count": len(draft.get("supporting_points", []) or []),
        "counterpoints_count": len(draft.get("counterpoints", []) or []),
        "risk_tradeoffs_count": len(draft.get("risk_tradeoffs", []) or []),
        "uncertainty_notes_count": len(draft.get("uncertainty_notes", []) or []),
        "accepted_item_count": sum(
            len(accepted.get(key, []) or [])
            for key in ("decision_basis", "supporting_points", "counterpoints", "risk_tradeoffs", "uncertainty_notes", "safety_notes")
        ) + (1 if str((accepted.get("tentative_conclusion", {}) or {}).get("statement", "")).strip() else 0),
    }
    base.update(summarize_draft_validation(validation))
    return base


__all__ = [
    "build_methodology_context",
    "generate_analyst_draft",
    "project_analyst_draft_to_synthesis",
    "summarize_analyst_draft",
    "validate_analyst_draft",
]
