#!/usr/bin/env python3
"""Generate the evidence requirement upgrade regression report."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "evidence_requirement_upgrade_report.md"
QUESTIONS = [
    "aapple和amazon你最看好哪个",
    "AAPL 和 AMZN 推荐哪个",
    "你觉得今年苹果财报会怎么样？",
    "苹果现在最大的问题是什么？",
    "帮我分析一下",
    "你是谁？",
    "预测明天苹果股价",
]


@dataclass
class CaseRecord:
    query: str
    task_type: str = ""
    answer_mode: str = ""
    safety_intent: str = ""
    evidence_plan_summary: dict[str, Any] | None = None
    requirement_status_summary: dict[str, Any] | None = None
    sufficiency_status: str = ""
    degradation_reason: str = ""
    final_synthesis_mode: str = ""
    answer_quality_judgment: str = ""
    still_numeric_only_dump: str = ""
    failure_layer: str = ""
    note: str = ""


async def _request_async(method: str, path: str, **kwargs):
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=180.0) as client:
        return await client.request(method=method, url=path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_request_async(method, path, **kwargs))


def _looks_like_numeric_dump(answer: str, synthesis_mode: str) -> bool:
    text = answer or ""
    if synthesis_mode == "limited_judgment" and ("如果更看重" in text or "If current" in text or "If revenue scale" in text):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("- ")]
    return bool(bullet_lines) and len(bullet_lines) >= max(2, len(lines) - 2)


def _answer_quality(trace: dict[str, Any], answer: str) -> tuple[str, str]:
    synthesis_mode = str(trace.get("synthesis_mode", "") or "")
    unsupported = list(trace.get("unsupported_synthesis_items", []) or [])
    limitations = {str(item.get("code", "")) for item in (trace.get("requirement_limitations", []) or []) if isinstance(item, dict)}
    numeric_dump = _looks_like_numeric_dump(answer, synthesis_mode)
    if unsupported:
        return "needs_followup", "yes" if numeric_dump else "no"
    if synthesis_mode == "limited_judgment":
        return "acceptable_limited_judgment", "yes" if numeric_dump else "no"
    if synthesis_mode in {"balanced_comparison", "validated_analysis", "cautious_outlook", "direct_fact", "conversational_short_circuit"}:
        return "acceptable", "yes" if numeric_dump else "no"
    if "requirement_missing" in limitations:
        return "blocked_by_missing_evidence", "yes" if numeric_dump else "no"
    return "needs_review", "yes" if numeric_dump else "no"


def _failure_layer(trace: dict[str, Any], note: str) -> str:
    if note:
        return "runtime"
    unsupported = list(trace.get("unsupported_synthesis_items", []) or [])
    if unsupported:
        return "synthesis"
    requirement_limitations = list(trace.get("requirement_limitations", []) or [])
    if any(str(item.get("failure_reason", "")).startswith("query_") for item in requirement_limitations if isinstance(item, dict)):
        return "executor"
    if any(str(item.get("failure_reason", "")) == "no_matching_evidence" for item in requirement_limitations if isinstance(item, dict)):
        return "retrieval"
    if any(str(item.get("failure_reason", "")) == "dependency_numeric_requirement_missing" for item in requirement_limitations if isinstance(item, dict)):
        return "sufficiency"
    if not (trace.get("evidence_plan", {}) or {}).get("evidence_requirements"):
        return "planner"
    return "none"


def _requirement_status_summary(trace: dict[str, Any]) -> dict[str, Any]:
    summary = dict(trace.get("trace_summary", {}) or {})
    evidence_summary = dict(trace.get("evidence_sufficiency_summary", {}) or {})
    return {
        "satisfied_requirements": evidence_summary.get("satisfied_requirements", []),
        "partial_requirements": evidence_summary.get("partial_requirements", []),
        "missing_requirements": evidence_summary.get("missing_requirements", []),
        "rejected_requirements": evidence_summary.get("rejected_requirements", []),
        "missing_requirements_count": summary.get("missing_requirements_count", evidence_summary.get("missing_count", 0)),
        "required_numeric_satisfied_rate": summary.get(
            "required_numeric_satisfied_rate",
            evidence_summary.get("required_numeric_satisfied_rate"),
        ),
        "required_text_satisfied_rate": summary.get(
            "required_text_satisfied_rate",
            evidence_summary.get("required_text_satisfied_rate"),
        ),
        "company_evidence_balance": summary.get(
            "company_evidence_balance",
            evidence_summary.get("company_evidence_balance"),
        ),
    }


def run_case(query: str) -> CaseRecord:
    try:
        response = _request("POST", "/chat", json={"query": query})
        if response.status_code >= 400:
            detail = ""
            try:
                payload = response.json()
                detail = str(payload.get("detail", ""))
            except Exception:
                detail = response.text
            raise RuntimeError(f"{response.status_code} {response.reason_phrase}: {detail}".strip())
        body = response.json()
        trace_id = str(body.get("trace_id", ""))
        trace_response = _request("GET", f"/trace/{trace_id}")
        if trace_response.status_code >= 400:
            detail = ""
            try:
                payload = trace_response.json()
                detail = str(payload.get("detail", ""))
            except Exception:
                detail = trace_response.text
            raise RuntimeError(f"{trace_response.status_code} {trace_response.reason_phrase}: {detail}".strip())
        trace = trace_response.json()
        quality, numeric_dump = _answer_quality(trace, str(body.get("answer", "")))
        return CaseRecord(
            query=query,
            task_type=str(trace.get("task_type", "")),
            answer_mode=str(trace.get("answer_mode", "")),
            safety_intent=str(trace.get("safety_intent", "")),
            evidence_plan_summary=dict(trace.get("evidence_plan_summary", {}) or {}),
            requirement_status_summary=_requirement_status_summary(trace),
            sufficiency_status=str((trace.get("trace_summary", {}) or {}).get("sufficiency_status", "")),
            degradation_reason=str(trace.get("degradation_reason", "") or ""),
            final_synthesis_mode=str((trace.get("trace_summary", {}) or {}).get("final_synthesis_mode", "")),
            answer_quality_judgment=quality,
            still_numeric_only_dump=numeric_dump,
            failure_layer=_failure_layer(trace, ""),
            note="",
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        return CaseRecord(
            query=query,
            answer_quality_judgment="blocked",
            still_numeric_only_dump="unknown",
            failure_layer="runtime",
            note=str(exc),
        )


def _json_block(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, ensure_ascii=False, indent=2)


def render_report(records: list[CaseRecord]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Evidence Requirement Upgrade Report",
        "",
        f"- Generated at: `{generated_at}`",
        "- Source path: local FastAPI `/chat` + `/trace` route via ASGI transport",
        "- Note: runtime failures usually indicate the local LLM stack or retrieval dependencies were unavailable during report generation.",
        "",
    ]
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {index}. {record.query}",
                "",
                f"- task_type / answer_mode / safety_intent: `{record.task_type}` / `{record.answer_mode}` / `{record.safety_intent}`",
                f"- sufficiency_status: `{record.sufficiency_status}`",
                f"- degradation_reason: `{record.degradation_reason}`",
                f"- final_synthesis_mode: `{record.final_synthesis_mode}`",
                f"- final answer quality judgment: `{record.answer_quality_judgment}`",
                f"- still like numeric-only dump: `{record.still_numeric_only_dump}`",
                f"- failure layer: `{record.failure_layer}`",
            ]
        )
        if record.note:
            lines.append(f"- note: `{record.note}`")
        lines.extend(
            [
                "",
                "### Evidence Plan Summary",
                "```json",
                _json_block(record.evidence_plan_summary),
                "```",
                "",
                "### Requirement Status Summary",
                "```json",
                _json_block(record.requirement_status_summary),
                "```",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    records = [run_case(query) for query in QUESTIONS]
    REPORT_PATH.write_text(render_report(records), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
