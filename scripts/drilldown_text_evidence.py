#!/usr/bin/env python3
"""Single-case drilldown for text evidence failures."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from config import settings
from src.api.app import create_app

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "text_evidence_drilldown_report.md"
PROBE_TICKERS = ["AAPL", "AMZN"]
PROBE_SECTIONS = ["ITEM_1A", "ITEM_7", "ITEM_1", "BUSINESS"]
PROBE_QUERIES = [
    "risk factors",
    "management discussion",
    "business risks",
    "competition",
    "operating results",
]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


async def _request_async(method: str, path: str, **kwargs: Any) -> httpx.Response:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=180.0) as client:
        return await client.request(method=method, url=path, **kwargs)


def _request(method: str, path: str, **kwargs: Any) -> httpx.Response:
    return asyncio.run(_request_async(method, path, **kwargs))


def _json_block(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


def _snippet(text: str, limit: int = 320) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _section_alias(section: str) -> str:
    raw = str(section or "").upper().strip()
    if raw == "BUSINESS":
        return "ITEM_1"
    return raw


def _tickers_for_query(query: str) -> list[str]:
    try:
        from src.agent.query_plan import extract_tickers_fallback

        tickers = extract_tickers_fallback(str(query or ""))
    except Exception:
        tickers = []
    clean = [str(t).upper() for t in tickers if str(t).strip()]
    return clean or list(PROBE_TICKERS)


def _duckdb_section_probe(tickers: list[str] | None = None) -> dict[str, Any]:
    try:
        import duckdb
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        return {"status": "skipped", "reason": f"duckdb_import_failed:{exc}", "rows": []}
    if not settings.duckdb_path.exists():
        return {"status": "skipped", "reason": f"duckdb_missing:{settings.duckdb_path}", "rows": []}
    rows: list[dict[str, Any]] = []
    try:
        conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
        try:
            for ticker in list(tickers or PROBE_TICKERS):
                for requested_section in PROBE_SECTIONS:
                    section = _section_alias(requested_section)
                    count = conn.execute(
                        "SELECT COUNT(*) FROM filing_chunks WHERE ticker = ? AND section = ?",
                        [ticker, section],
                    ).fetchone()[0]
                    snippets = conn.execute(
                        """
                        SELECT c.filing_id, m.form_type, COALESCE(m.fiscal_period, '') AS fiscal_period,
                               c.section, c.quality, c.chunk_order, c.chunk_text
                        FROM filing_chunks c
                        LEFT JOIN filings_metadata m ON c.filing_id = m.filing_id
                        WHERE c.ticker = ? AND c.section = ?
                        ORDER BY COALESCE(m.filing_date, DATE '1900-01-01') DESC, c.chunk_order
                        LIMIT 3
                        """,
                        [ticker, section],
                    ).fetchall()
                    rows.append(
                        {
                            "ticker": ticker,
                            "requested_section": requested_section,
                            "mapped_section": section,
                            "raw_hit_count": int(count or 0),
                            "top_snippets": [
                                {
                                    "filing_id": str(r[0]),
                                    "form_type": str(r[1] or ""),
                                    "period": str(r[2] or ""),
                                    "section": str(r[3] or ""),
                                    "quality": str(r[4] or ""),
                                    "chunk_order": int(r[5] or 0),
                                    "text_snippet": _snippet(str(r[6] or "")),
                                }
                                for r in snippets
                            ],
                        }
                    )
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - data/runtime dependent
        return {"status": "skipped", "reason": f"duckdb_probe_failed:{exc}", "rows": rows}
    return {"status": "ok", "rows": rows}


def _chroma_query_probe(tickers: list[str] | None = None) -> dict[str, Any]:
    if os.environ.get("DRILLDOWN_ENABLE_CHROMA_PROBE", "").strip() not in {"1", "true", "TRUE", "yes"}:
        return {
            "status": "skipped",
            "reason": "disabled_by_default_to_avoid_embedding_model_network_download; set DRILLDOWN_ENABLE_CHROMA_PROBE=1 to enable",
            "rows": [],
        }
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        from src.tools.search_filings import _collection_name_for_version, _get_model, _selected_collection_name
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        return {"status": "skipped", "reason": f"chroma_import_failed:{exc}", "rows": []}
    if not settings.chroma_dir.exists():
        return {"status": "skipped", "reason": f"chroma_missing:{settings.chroma_dir}", "rows": []}
    rows: list[dict[str, Any]] = []
    try:
        client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            collection = client.get_collection(_selected_collection_name())
        except Exception:
            collection = client.get_collection(_collection_name_for_version("v1"))
        try:
            model = _get_model()
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": f"embedding_model_unavailable_or_offline:{exc}",
                "rows": rows,
            }
        for ticker in list(tickers or PROBE_TICKERS):
            for query in PROBE_QUERIES:
                embedding = model.encode([query], convert_to_numpy=True)
                result = collection.query(
                    query_embeddings=embedding.tolist(),
                    n_results=12,
                    where={"ticker": ticker},
                    include=["documents", "metadatas", "distances"],
                )
                docs = list(result.get("documents", [[]])[0] or [])
                metas = list(result.get("metadatas", [[]])[0] or [])
                dists = list(result.get("distances", [[]])[0] or [])
                top_snippets: list[dict[str, Any]] = []
                section_counts: Counter[str] = Counter()
                for doc, meta, dist in zip(docs, metas, dists):
                    section = str((meta or {}).get("section", ""))
                    section_counts[section] += 1
                    score = 1.0 / (1.0 + float(dist))
                    top_snippets.append(
                        {
                            "filing_id": str((meta or {}).get("filing_id", "")),
                            "form_type": str((meta or {}).get("form_type", "")),
                            "period": str((meta or {}).get("fiscal_period", "")),
                            "section": section,
                            "quality": str((meta or {}).get("quality", "")),
                            "chunk_order": int((meta or {}).get("chunk_order", 0) or 0),
                            "score": round(score, 4),
                            "text_snippet": _snippet(str(doc)),
                        }
                    )
                rows.append(
                    {
                        "ticker": ticker,
                        "query": query,
                        "raw_hit_count": len(docs),
                        "top_sections": [{"section": k, "count": int(v)} for k, v in section_counts.most_common(8)],
                        "top_snippets": top_snippets[:5],
                    }
                )
    except Exception as exc:  # pragma: no cover - data/runtime dependent
        return {"status": "skipped", "reason": f"chroma_probe_failed:{exc}", "rows": rows}
    return {"status": "ok", "rows": rows}


def run_low_level_rag_probe(query: str = "") -> dict[str, Any]:
    tickers = _tickers_for_query(query)
    return {
        "tickers": tickers,
        "duckdb_section_probe": _duckdb_section_probe(tickers),
        "chroma_query_probe": _chroma_query_probe(tickers),
    }


def _stage_summary(trace: dict[str, Any]) -> dict[str, int]:
    diagnostics = dict(trace.get("text_requirement_diagnostics", {}) or {})
    counts: Counter[str] = Counter()
    for item in diagnostics.values():
        if isinstance(item, dict):
            stage = str(item.get("drop_stage", "") or "").strip()
            if stage:
                counts[stage] += 1
    return dict(counts)


def _low_level_has_content(low_level_probe: dict[str, Any]) -> bool:
    duckdb_rows = (low_level_probe.get("duckdb_section_probe", {}) or {}).get("rows", []) or []
    chroma_rows = (low_level_probe.get("chroma_query_probe", {}) or {}).get("rows", []) or []
    return any(int(row.get("raw_hit_count", 0) or 0) > 0 for row in list(duckdb_rows) + list(chroma_rows) if isinstance(row, dict))


def root_cause_guess(trace: dict[str, Any], low_level_probe: dict[str, Any]) -> str:
    summary = _stage_summary(trace)
    if not summary:
        return "No text requirements were present in the trace, so there is no text evidence failure to classify."
    if set(summary) == {"satisfied"}:
        return "Text evidence was satisfied for the final bundle; no failure is indicated by this trace."
    low_level_has_content = _low_level_has_content(low_level_probe)
    if not low_level_has_content:
        return "Low-level DuckDB/Chroma probes did not find target text content, so the likely root cause is a data/index coverage gap."
    if summary.get("no_raw_hits"):
        return "Low-level probes found content, but the Agent text requirement saw no raw hits; likely retrieval query, ticker, date, or payload constraints are too narrow."
    if summary.get("section_filter_dropped"):
        return "Raw hits existed but section filtering removed them; likely section mapping or allowlist/fallback section policy is the failure layer."
    if summary.get("quality_filter_dropped"):
        return "Hits reached the executor but were rejected as low-quality or MIXED sections; likely corpus section quality or quality filter is the failure layer."
    if summary.get("snippet_support_failed"):
        return "Hits reached the executor but lacked acceptable supporting snippets/terms; likely snippet support validation is too strict for this query."
    if summary.get("claim_validation_failed"):
        claim_summary = _text_claim_validation_summary(trace)
        raw_claims = sum(int(item.get("raw_text_claim_count", 0) or 0) for item in claim_summary.values())
        rejected_claims = sum(len(item.get("rejected_text_claims", []) or []) for item in claim_summary.values())
        generation_errors = {
            str(item.get("claim_generation_error", "") or "")
            for item in claim_summary.values()
            if str(item.get("claim_generation_error", "") or "")
        }
        if raw_claims <= 0:
            return f"Usable snippets existed, but text claim generation produced no candidate claims; likely claim generation is the failure layer. generation_errors={sorted(generation_errors)}"
        if rejected_claims > 0:
            return "Usable snippets existed, but generated text claims were rejected by validation; inspect rejected_text_claims for exact validation_error_code values."
        return "Usable snippets existed, but no text claim survived validation; likely claim generation/validation is the failure layer."
    if summary.get("citation_policy_dropped"):
        return "Text claims existed, but citation policy dropped them; likely balanced comparison citation policy or citation caps are the failure layer."
    if summary.get("final_bundle_dropped"):
        return "Citations existed, but text evidence did not enter the final bundle; likely final evidence assembly is the failure layer."
    return f"Failure is mixed across stages: {summary}."


def _top_raw_snippets(trace: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rid, diag in dict(trace.get("text_requirement_diagnostics", {}) or {}).items():
        if not isinstance(diag, dict):
            continue
        for item in list(diag.get("top_raw_snippets", []) or [])[:5]:
            if isinstance(item, dict):
                row = dict(item)
                row["requirement_id"] = str(rid)
                out.append(row)
    return out[:20]


def _top_rejected_snippets(trace: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rid, diag in dict(trace.get("text_requirement_diagnostics", {}) or {}).items():
        if not isinstance(diag, dict):
            continue
        for item in list(diag.get("top_rejected_snippets", []) or [])[:5]:
            if isinstance(item, dict):
                row = dict(item)
                row["requirement_id"] = str(rid)
                out.append(row)
    return out[:20]


def _text_claim_validation_summary(trace: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for rid, diag in dict(trace.get("text_requirement_diagnostics", {}) or {}).items():
        if not isinstance(diag, dict):
            continue
        rows[str(rid)] = {
            "raw_text_claim_count": int(diag.get("raw_text_claim_count", 0) or 0),
            "candidate_text_claim_count": int(diag.get("candidate_text_claim_count", 0) or 0),
            "validated_text_claim_count": int(
                diag.get("validated_text_claim_count", diag.get("text_claim_validated_count", 0)) or 0
            ),
            "text_citation_kept_count": int(diag.get("text_citation_kept_count", 0) or 0),
            "final_validated_text_count": int(diag.get("final_validated_text_count", 0) or 0),
            "drop_stage": str(diag.get("drop_stage", "") or ""),
            "claim_generation_error": str(diag.get("claim_generation_error", "") or ""),
            "rejected_text_claims": list(diag.get("rejected_text_claims", []) or []),
            "text_claim_validation_warnings": list(diag.get("text_claim_validation_warnings", []) or []),
        }
    return rows


def _text_requirement_metadata(trace: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence_plan = dict(trace.get("evidence_plan", {}) or {})
    out: dict[str, dict[str, Any]] = {}
    for req in evidence_plan.get("evidence_requirements", []) or []:
        if not isinstance(req, dict) or str(req.get("requirement_type", "")) != "text":
            continue
        rid = str(req.get("requirement_id", "") or "")
        if not rid:
            continue
        out[rid] = dict(req)
    return out


def _example_snippet(diag: dict[str, Any]) -> str:
    for key in ("top_raw_snippets", "top_rejected_snippets"):
        for item in list(diag.get(key, []) or []):
            if isinstance(item, dict):
                snippet = str(item.get("supporting_snippet") or item.get("text_snippet") or item.get("snippet") or "")
                if snippet.strip():
                    return _snippet(snippet, 220)
    return ""


def _per_requirement_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    req_meta = _text_requirement_metadata(trace)
    diagnostics = dict(trace.get("text_requirement_diagnostics", {}) or {})
    rows: list[dict[str, Any]] = []
    for rid, req in req_meta.items():
        diag = dict(diagnostics.get(rid, {}) or {})
        rows.append(
            {
                "requirement_id": rid,
                "dimension_id": str(req.get("dimension_id") or diag.get("dimension_id") or ""),
                "ticker": str(req.get("company") or diag.get("company") or ""),
                "retrieval_query": str(req.get("retrieval_query") or diag.get("retrieval_query") or ""),
                "section_preferences": list(req.get("section_preferences") or diag.get("section_preferences") or []),
                "raw_hit_count": int(diag.get("raw_hit_count", 0) or 0),
                "section_filtered_hit_count": int(diag.get("section_filtered_hit_count", 0) or 0),
                "usable_hit_count": int(diag.get("usable_hit_count", 0) or 0),
                "generated_claim_count": int(
                    diag.get("generated_claim_count", diag.get("raw_text_claim_count", diag.get("candidate_text_claim_count", 0))) or 0
                ),
                "validated_claim_count": int(
                    diag.get("validated_claim_count", diag.get("validated_text_claim_count", diag.get("text_claim_validated_count", 0))) or 0
                ),
                "final_validated_text_count": int(diag.get("final_validated_text_count", 0) or 0),
                "drop_stage": str(diag.get("drop_stage", "") or ""),
                "failure_reason": str(diag.get("failure_reason", "") or ""),
                "example_snippet": _example_snippet(diag),
            }
        )
    for rid, diag_raw in diagnostics.items():
        if rid in req_meta or not isinstance(diag_raw, dict):
            continue
        diag = dict(diag_raw)
        rows.append(
            {
                "requirement_id": str(rid),
                "dimension_id": str(diag.get("dimension_id") or ""),
                "ticker": str(diag.get("company") or ""),
                "retrieval_query": str(diag.get("retrieval_query") or ""),
                "section_preferences": list(diag.get("section_preferences") or []),
                "raw_hit_count": int(diag.get("raw_hit_count", 0) or 0),
                "section_filtered_hit_count": int(diag.get("section_filtered_hit_count", 0) or 0),
                "usable_hit_count": int(diag.get("usable_hit_count", 0) or 0),
                "generated_claim_count": int(diag.get("raw_text_claim_count", 0) or 0),
                "validated_claim_count": int(diag.get("validated_text_claim_count", 0) or 0),
                "final_validated_text_count": int(diag.get("final_validated_text_count", 0) or 0),
                "drop_stage": str(diag.get("drop_stage", "") or ""),
                "failure_reason": str(diag.get("failure_reason", "") or ""),
                "example_snippet": _example_snippet(diag),
            }
        )
    return rows


def _offline_planning_trace(query: str) -> dict[str, Any]:
    try:
        from src.agent.query_plan import build_classification_state

        state = build_classification_state(
            user_query=query,
            parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
            trace_id="drilldown-offline-planning",
            today=datetime.now(timezone.utc).date(),
        )
    except Exception as exc:  # pragma: no cover - runtime dependent
        return {
            "trace_id": "drilldown-offline-planning",
            "task_type": "",
            "answer_mode": "",
            "safety_intent": "",
            "evidence_plan": {"evidence_requirements": []},
            "text_requirement_diagnostics": {},
            "planning_error": str(exc),
        }
    diagnostics: dict[str, dict[str, Any]] = {}
    for req in list(dict(state.get("evidence_plan", {}) or {}).get("evidence_requirements", []) or []):
        if not isinstance(req, dict) or str(req.get("requirement_type", "")) != "text":
            continue
        rid = str(req.get("requirement_id", "") or "")
        diagnostics[rid] = {
            "requirement_id": rid,
            "dimension_id": str(req.get("dimension_id", "") or ""),
            "company": str(req.get("company", "") or ""),
            "retrieval_query": str(req.get("retrieval_query", "") or ""),
            "section_preferences": list(req.get("section_preferences", []) or []),
            "fallback_queries": list(req.get("broadened_queries", []) or []),
            "raw_hit_count": 0,
            "section_filtered_hit_count": 0,
            "usable_hit_count": 0,
            "raw_text_claim_count": 0,
            "validated_text_claim_count": 0,
            "final_validated_text_count": 0,
            "drop_stage": "no_raw_hits",
            "failure_reason": "agent_run_skipped",
            "top_raw_snippets": [],
        }
    trace = dict(state)
    trace["trace_id"] = str(trace.get("trace_id") or "drilldown-offline-planning")
    trace["text_requirement_diagnostics"] = diagnostics
    return trace


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No text requirements found."]
    columns = [
        "requirement_id",
        "dimension_id",
        "ticker",
        "retrieval_query",
        "section_preferences",
        "raw_hit_count",
        "section_filtered_hit_count",
        "usable_hit_count",
        "generated_claim_count",
        "validated_claim_count",
        "final_validated_text_count",
        "drop_stage",
        "failure_reason",
        "example_snippet",
    ]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, list):
                text = ", ".join(str(x) for x in value)
            else:
                text = str(value)
            text = text.replace("|", "\\|").replace("\n", " ")
            values.append(text)
        out.append("| " + " | ".join(values) + " |")
    return out


def _final_validated_text_evidence(chat_body: dict[str, Any], trace: dict[str, Any]) -> list[dict[str, Any]]:
    output = dict(chat_body.get("output", {}) or {})
    if output.get("text_evidence"):
        return list(output.get("text_evidence", []) or [])
    if trace.get("text_evidence"):
        return list(trace.get("text_evidence", []) or [])
    packet = dict(trace.get("evidence_packet", {}) or {})
    if packet.get("text_snippets"):
        return list(packet.get("text_snippets", []) or [])
    return []


def render_report(
    *,
    query: str,
    chat_body: dict[str, Any],
    trace: dict[str, Any],
    low_level_probe: dict[str, Any],
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    trace_id = str(chat_body.get("trace_id") or trace.get("trace_id") or "")
    final_synthesis_mode = str(trace.get("synthesis_mode", "") or (trace.get("trace_summary", {}) or {}).get("final_synthesis_mode", ""))
    per_requirement_rows = _per_requirement_rows(trace)
    lines = [
        "# Text Evidence Drilldown Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Query: `{query}`",
        f"- Trace ID: `{trace_id}`",
        f"- answer_mode: `{trace.get('answer_mode', '')}`",
        f"- safety_intent: `{trace.get('safety_intent', '')}`",
        f"- task_type: `{trace.get('task_type', '')}`",
        f"- answer_mode_final / synthesis_mode: `{trace.get('answer_mode', '')}` / `{final_synthesis_mode}`",
        "",
        "## Evidence Requirements",
        "```json",
        _json_block((trace.get("evidence_plan", {}) or {}).get("evidence_requirements", [])),
        "```",
        "",
        "## Per-Requirement Retrieval Diagnostics",
        "",
        "### Drop Stage Table",
        "",
        *_markdown_table(per_requirement_rows),
        "",
        "### Raw Diagnostics",
        "```json",
        _json_block(trace.get("text_requirement_diagnostics", {})),
        "```",
        "",
        "## Top Raw Snippets",
        "```json",
        _json_block(_top_raw_snippets(trace)),
        "```",
        "",
        "## Top Rejected Snippets",
        "```json",
        _json_block(_top_rejected_snippets(trace)),
        "```",
        "",
        "## Text Claim Generation / Validation",
        "```json",
        _json_block(_text_claim_validation_summary(trace)),
        "```",
        "",
        "## Final Validated Text Evidence",
        "```json",
        _json_block(_final_validated_text_evidence(chat_body, trace)),
        "```",
        "",
        "## Low-Level RAG Probe",
        "```json",
        _json_block(low_level_probe),
        "```",
        "",
        "## Failure Stage Summary",
        "```json",
        _json_block(_stage_summary(trace)),
        "```",
        "",
        "## Root Cause Guess",
        "",
        root_cause_guess(trace, low_level_probe),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _render_blocked(query: str, note: str, low_level_probe: dict[str, Any] | None = None) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Text Evidence Drilldown Report",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Query: `{query}`",
        "- Status: `blocked`",
        f"- Note: `{note}`",
    ]
    if low_level_probe is not None:
        lines.extend(["", "## Low-Level RAG Probe", "```json", _json_block(low_level_probe), "```"])
    return "\n".join(lines).strip() + "\n"


def run_drilldown(query: str, *, skip_agent: bool = False) -> str:
    low_level_probe = run_low_level_rag_probe(query)
    if skip_agent:
        trace = _offline_planning_trace(query)
        return render_report(query=query, chat_body={"trace_id": trace.get("trace_id", "")}, trace=trace, low_level_probe=low_level_probe)
    chat_response = _request("POST", "/chat", json={"query": query})
    if chat_response.status_code >= 400:
        return _render_blocked(query, f"{chat_response.status_code} {chat_response.reason_phrase}: {chat_response.text}", low_level_probe)
    chat_body = chat_response.json()
    trace_id = str(chat_body.get("trace_id", ""))
    trace_response = _request("GET", f"/trace/{trace_id}")
    if trace_response.status_code >= 400:
        return _render_blocked(query, f"{trace_response.status_code} {trace_response.reason_phrase}: {trace_response.text}", low_level_probe)
    return render_report(query=query, chat_body=chat_body, trace=trace_response.json(), low_level_probe=low_level_probe)


def main() -> int:
    parser = argparse.ArgumentParser(description="Drill down text evidence failures for a single query.")
    parser.add_argument("--query", required=True, help="User query to run through the Agent.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown report output path.")
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="Only render planning/low-level diagnostics; useful when local embedding/LLM models are unavailable.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_drilldown(str(args.query), skip_agent=bool(args.skip_agent))
    except Exception as exc:  # pragma: no cover - runtime dependent
        report = _render_blocked(str(args.query), str(exc))
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
