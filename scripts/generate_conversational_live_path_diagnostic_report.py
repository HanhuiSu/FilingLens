#!/usr/bin/env python3
"""Generate a minimal diagnostic report for the conversational live path."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "conversational_live_path_diagnostic_report.md"
QUERY = "aapple和amazon你最看好哪个"


async def _request_async(method: str, path: str, **kwargs):
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=180.0) as client:
        return await client.request(method=method, url=path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_request_async(method, path, **kwargs))


def _json_block(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _render_failure(note: str) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    return "\n".join(
        [
            "# Conversational Live Path Diagnostic Report",
            "",
            f"- Generated at: `{generated_at}`",
            f"- Query: `{QUERY}`",
            "- Status: `blocked`",
            f"- Note: `{note}`",
            "",
        ]
    ).strip() + "\n"


def _render_success(chat_body: dict[str, Any], trace: dict[str, Any]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Conversational Live Path Diagnostic Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Query: `{QUERY}`",
        f"- task_type / answer_mode / safety_intent: `{trace.get('task_type', '')}` / `{trace.get('answer_mode', '')}` / `{trace.get('safety_intent', '')}`",
        f"- final_synthesis_mode: `{trace.get('synthesis_mode', '') or (trace.get('trace_summary', {}) or {}).get('final_synthesis_mode', '')}`",
        "",
        "## Proposed Requirements",
        "```json",
        _json_block((trace.get("evidence_plan", {}) or {}).get("evidence_requirements", [])),
        "```",
        "",
        "## Validated Requirements",
        "```json",
        _json_block(trace.get("validated_requirement_ids", [])),
        "```",
        "",
        "## Raw Retrieval Hits",
        "```json",
        _json_block(trace.get("raw_retrieval_hits_by_requirement", {})),
        "```",
        "",
        "## Validated Text Evidence Count",
        "```json",
        _json_block({"validated_text_evidence_count": trace.get("validated_text_evidence_count", 0)}),
        "```",
        "",
        "## Final Requirement Status Map",
        "```json",
        _json_block(trace.get("final_requirement_status_map", trace.get("requirement_status_map", {}))),
        "```",
        "",
        "## Sufficiency Result",
        "```json",
        _json_block(trace.get("evidence_sufficiency", {})),
        "```",
        "",
        "## Final Synthesis Mode",
        "```json",
        _json_block(
            {
                "synthesis_mode": trace.get("synthesis_mode", ""),
                "trace_summary": trace.get("trace_summary", {}),
                "output_synthesis_mode": (chat_body.get("output", {}) or {}).get("synthesis_mode", ""),
            }
        ),
        "```",
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    try:
        chat_response = _request("POST", "/chat", json={"query": QUERY})
        if chat_response.status_code >= 400:
            detail = ""
            try:
                payload = chat_response.json()
                detail = str(payload.get("detail", ""))
            except Exception:
                detail = chat_response.text
            REPORT_PATH.write_text(_render_failure(f"{chat_response.status_code} {chat_response.reason_phrase}: {detail}".strip()))
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
            REPORT_PATH.write_text(_render_failure(f"{trace_response.status_code} {trace_response.reason_phrase}: {detail}".strip()))
            return 0
        REPORT_PATH.write_text(_render_success(chat_body, trace_response.json()))
        return 0
    except Exception as exc:  # pragma: no cover - runtime dependent
        REPORT_PATH.write_text(_render_failure(str(exc)))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
