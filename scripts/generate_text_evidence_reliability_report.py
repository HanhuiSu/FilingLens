#!/usr/bin/env python3
"""Generate a compact diagnostic report for the text-evidence reliability sprint."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "text_evidence_reliability_report.md"
QUERIES = [
    "苹果现在最大的问题是什么？",
    "苹果面临哪些主要风险？",
    "苹果管理层当前最担心什么？",
    "苹果当前业务最大的压力是什么？",
    "AAPL 和 AMZN 哪家公司风险更高？",
    "AAPL 和 AMZN 最值得关注的差异是什么？",
]


async def _request_async(method: str, path: str, **kwargs):
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=180.0) as client:
        return await client.request(method=method, url=path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_request_async(method, path, **kwargs))


def _json_block(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _failure_layer(trace: dict[str, Any]) -> str:
    diagnostics = dict(trace.get("text_requirement_diagnostics", {}) or {})
    drop_stages = {
        str(item.get("drop_stage", "")).strip()
        for item in diagnostics.values()
        if isinstance(item, dict) and str(item.get("drop_stage", "")).strip()
    }
    if "no_raw_hits" in drop_stages or "no_retrieval_hit" in drop_stages:
        return "retrieval"
    if {"section_filter_dropped", "quality_filter_dropped", "snippet_support_failed", "filtered_before_usable"} & drop_stages:
        return "executor"
    if "claim_validation_failed" in drop_stages:
        return "validation"
    if "citation_policy_dropped" in drop_stages or "citation_policy_filtered" in drop_stages or "comparison_unbalanced" in drop_stages:
        return "validation"
    if "final_bundle_dropped" in drop_stages or "final_bundle_empty" in drop_stages:
        return "synthesis"
    sufficiency = dict(trace.get("evidence_sufficiency", {}) or {})
    if str(sufficiency.get("overall_status", "")) == "insufficient":
        return "sufficiency"
    synthesis_mode = str(trace.get("synthesis_mode", "") or (trace.get("trace_summary", {}) or {}).get("final_synthesis_mode", ""))
    if synthesis_mode.startswith("insufficient_"):
        return "synthesis"
    return "planner"


def _render_blocked(note: str) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    return "\n".join(
        [
            "# Text Evidence Reliability Report",
            "",
            f"- Generated at: `{generated_at}`",
            "- Status: `blocked`",
            f"- Note: `{note}`",
            "",
        ]
    ).strip() + "\n"


def _render_success(records: list[dict[str, Any]]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Text Evidence Reliability Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Queries: `{len(records)}`",
        "",
    ]
    for idx, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## {idx}. {record['query']}",
                "",
                f"- task_type / answer_mode / safety_intent: `{record['task_type']}` / `{record['answer_mode']}` / `{record['safety_intent']}`",
                f"- final_synthesis_mode: `{record['final_synthesis_mode']}`",
                f"- failure_layer: `{record['failure_layer']}`",
                "",
                "### Proposed Requirements",
                "```json",
                _json_block(record["proposed_requirements"]),
                "```",
                "",
                "### Text Requirement Diagnostics",
                "```json",
                _json_block(record["text_requirement_diagnostics"]),
                "```",
                "",
                "### Final Requirement Status",
                "```json",
                _json_block(record["final_requirement_status"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    try:
        records: list[dict[str, Any]] = []
        for query in QUERIES:
            chat_response = _request("POST", "/chat", json={"query": query})
            if chat_response.status_code >= 400:
                detail = ""
                try:
                    payload = chat_response.json()
                    detail = str(payload.get("detail", ""))
                except Exception:
                    detail = chat_response.text
                REPORT_PATH.write_text(_render_blocked(f"{chat_response.status_code} {chat_response.reason_phrase}: {detail}".strip()))
                return 0
            chat_body = chat_response.json()
            trace_id = str(chat_body.get("trace_id", ""))
            trace_response = _request("GET", f"/trace/{trace_id}")
            if trace_response.status_code >= 400:
                detail = ""
                try:
                    payload = trace_response.json()
                    detail = str(payload.get("detail", ""))
                except Exception:
                    detail = trace_response.text
                REPORT_PATH.write_text(_render_blocked(f"{trace_response.status_code} {trace_response.reason_phrase}: {detail}".strip()))
                return 0
            trace = trace_response.json()
            records.append(
                {
                    "query": query,
                    "task_type": trace.get("task_type", ""),
                    "answer_mode": trace.get("answer_mode", ""),
                    "safety_intent": trace.get("safety_intent", ""),
                    "proposed_requirements": (trace.get("evidence_plan", {}) or {}).get("evidence_requirements", []),
                    "text_requirement_diagnostics": trace.get("text_requirement_diagnostics", {}),
                    "final_requirement_status": trace.get("final_requirement_status_map", trace.get("requirement_status_map", {})),
                    "final_synthesis_mode": trace.get("synthesis_mode", "") or (trace.get("trace_summary", {}) or {}).get("final_synthesis_mode", ""),
                    "failure_layer": _failure_layer(trace),
                }
            )
        REPORT_PATH.write_text(_render_success(records))
        return 0
    except Exception as exc:  # pragma: no cover - runtime dependent
        REPORT_PATH.write_text(_render_blocked(str(exc)))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
