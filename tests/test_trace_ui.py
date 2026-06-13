"""Tests for sanitized trace UI model."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import httpx

from config import settings
from src.api.app import create_app
from src.api.trace_view import build_trace_ui_model


ROOT = Path(__file__).resolve().parent.parent


def _trace() -> dict:
    return {
        "trace_id": "trace-ui-test",
        "user_query": "分析 NVIDIA",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "canonical_intent": {
            "intent_family": "cash_flow",
            "analysis_scope": "single_company",
            "requested_dimensions": ["cash_flow_quality"],
        },
        "intent_merge_decision": {"source": "semantic_proposal_validated", "final_intent_family": "cash_flow"},
        "evidence_policy_id": "single_company_cash_flow_v1",
        "evidence_policy": {
            "policy_id": "single_company_cash_flow_v1",
            "core_requirements": ["dimension:cash_flow_quality"],
            "optional_context_requirements": [],
        },
        "research_plan_raw": {"question_type": "causal_explanation", "user_goal": "raw goal"},
        "research_plan_validated": {
            "question_type": "causal_explanation",
            "user_goal": "Explain revenue growth.",
            "required_answer_parts": [
                {"id": "quantify_growth", "description": "Quantify growth", "required": True},
                {"id": "identify_growth_drivers", "description": "Identify drivers", "required": True},
            ],
            "fallback_answer_policy": "If driver text is missing, say causes cannot be explained.",
        },
        "research_plan_used": {
            "question_type": "causal_explanation",
            "user_goal": "Explain revenue growth.",
            "required_answer_parts": [
                {"id": "quantify_growth", "description": "Quantify growth", "required": True},
                {"id": "identify_growth_drivers", "description": "Identify drivers", "required": True},
            ],
            "fallback_answer_policy": "If driver text is missing, say causes cannot be explained.",
        },
        "research_plan_validation": {"valid": True, "used_fallback": False, "warnings": []},
        "research_plan_source": "validated_llm",
        "research_plan_fallback_reason": "",
        "research_plan_duration_ms": 123,
        "required_answer_parts": [
            {"id": "quantify_growth", "description": "Quantify growth", "required": True},
            {"id": "identify_growth_drivers", "description": "Identify drivers", "required": True},
        ],
        "answer_part_status_by_id": {
            "quantify_growth": {"status": "satisfied"},
            "identify_growth_drivers": {"status": "missing_but_analyzable", "reason": "driver_text_evidence_missing_but_analyzable"},
        },
        "evidence_gap_by_answer_part": {
            "identify_growth_drivers": {"status": "missing_but_analyzable", "reason": "driver_text_evidence_missing_but_analyzable"}
        },
        "missing_required_answer_parts": [],
        "partial_required_answer_parts": ["quantify_growth"],
        "missing_but_analyzable_answer_parts": ["identify_growth_drivers"],
        "relevance_status": "analytical_with_gaps",
        "relevance_decision": {
            "decision": "warning",
            "route": "finalize",
            "missing_but_analyzable_answer_parts": ["identify_growth_drivers"],
            "partial_required_answer_parts": ["quantify_growth"],
        },
        "analytical_claims": [
            {"id": "eb1", "tier": "evidence_backed", "text": "Revenue was 100 [N1].", "citation_refs": ["N1"]},
            {"id": "ei1", "tier": "evidence_inferred", "text": "Evidence points to a bounded inference.", "citation_refs": ["N1"]},
            {"id": "hv1", "tier": "hypothesis_to_verify", "text": "待验证假设：AI capex 是否扩张。"},
        ],
        "claim_tiers": {"evidence_backed": 1, "evidence_inferred": 1, "hypothesis_to_verify": 1},
        "analytical_reasoning_status": "used",
        "evidence_health": "degraded",
        "legacy_evidence_plan": {
            "evidence_policy_id": "single_company_cash_flow_v1",
            "expected_synthesis_style": "analytical",
            "evidence_requirements": [{"requirement_id": "REQ-LEGACY", "requirement_type": "numeric"}],
        },
        "plan_coverage_decision": {
            "strategy": "merge",
            "legacy_core_count": 18,
            "research_core_count": 3,
            "retained_legacy_core_count": 18,
            "coverage_ratio": 0.17,
            "warnings": ["research_plan_under_covered_legacy_core"],
            "reason": "overview_research_plan_augments_legacy_coverage",
        },
        "requirement_merge_summary": {
            "strategy": "merge",
            "merged_total_requirements": 21,
            "deduped_requirements": 1,
            "legacy_only_count": 18,
            "research_only_count": 2,
            "legacy_research_count": 1,
            "retained_legacy_core_count": 18,
        },
        "evidence_plan_used": {"source": "merged", "strategy": "merge", "requirement_count": 21},
        "evidence_validation_records": [
            {
                "requirement_id": "REQ-CASH",
                "evidence_type": "numeric",
                "tool": "query_financial_data",
                "tool_returned_count": 1,
                "validated_evidence_count": 0,
                "rejected_evidence_reason": "metric_mapping_failed",
                "status": "missing",
            }
        ],
        "companies": ["NVDA"],
        "selected_tools": ["query_financial_data", "search_filings"],
        "needs_tools": True,
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-REV",
                    "requirement_type": "numeric",
                    "dimension_id": "revenue_quality",
                    "company": "NVDA",
                    "required": True,
                    "requirement_scope": "diagnostic",
                },
                {
                    "requirement_id": "REQ-CASH",
                    "requirement_type": "numeric",
                    "dimension_id": "cash_flow_quality",
                    "company": "NVDA",
                    "required": True,
                    "requirement_scope": "core",
                },
            ]
        },
        "evidence_collection_results": [
            {"requirement_id": "REQ-REV", "status": "satisfied", "items": [{"evidence_id": "N1"}]},
            {"requirement_id": "REQ-CASH", "status": "missing", "items": [], "failure_reason": "no_matching_evidence"},
        ],
        "dimension_status_by_id": {
            "revenue_quality": {"status": "satisfied", "supporting_evidence_ids": ["N1"]},
            "cash_flow_quality": {"status": "missing", "required_missing": ["free_cash_flow"]},
        },
        "evidence_packet": {
            "numeric_table": [
                {
                    "evidence_id": "N1",
                    "ticker": "NVDA",
                    "metric": "revenue",
                    "role": "current_revenue",
                    "evidence_role": "current_revenue",
                    "source_requirement_id": "REQ-REV",
                    "quality_status": "valid",
                    "value": 100.0,
                    "period_end": "2025-01-31",
                    "source_provider": "sec_companyfacts",
                }
            ],
            "text_snippets": [
                {
                    "evidence_id": "T1",
                    "ticker": "NVDA",
                    "section": "ITEM_1A",
                    "driver_level": "segment_level_driver",
                    "driver_levels": ["segment_level_driver"],
                    "claim_scope": "segment",
                    "allowed_claim_strength": "bounded_inference",
                    "scope_reason": "Snippet discusses segment revenue.",
                    "supporting_snippet": "Risk.",
                }
            ],
        },
        "final_answer": "Revenue was 100 [N1].",
        "contract_status": "repaired",
        "final_contract_status": "repaired",
        "semantic_parser_mode": "shadow",
        "semantic_parser": {
            "mode": "shadow",
            "ok": True,
            "source": "test",
            "proposal": {
                "company_mentions": ["NVIDIA"],
                "analysis_scope": "single_company",
                "methodology_intent": "cash_flow",
                "requested_dimensions": ["cash_flow_quality"],
                "requested_metrics": ["free_cash_flow"],
                "safety_intent": "normal",
                "confidence": 0.9,
                "reasons": ["sanitized reason"],
            },
            "warnings": [{"field": "requested_metrics", "value": "bad_metric", "reason": "metric_not_supported"}],
            "diagnostics": {"parse_status": "validated", "confidence": 0.9},
            "disagreement": {
                "parser_ok": True,
                "injected": False,
                "proposed_methodology_intent": "cash_flow",
                "rule_methodology_intent": "revenue",
                "methodology_intent_disagreement": True,
            },
        },
        "semantic_proposal": {"methodology_intent": "cash_flow", "requested_dimensions": ["cash_flow_quality"]},
        "rule_methodology_intent": "revenue",
        "proposed_methodology_intent": "cash_flow",
        "proposal_validation_warnings": [{"field": "requested_metrics", "reason": "metric_not_supported"}],
        "intent_conflict": True,
        "contract_attempts": 1,
        "contract_result": {
            "route": "repair_answer",
            "decision": "repairable",
            "severity": "repairable",
            "public_summary": "Answer was repaired and passed.",
            "violations": [{"code": "caveat_not_visible", "answer_span": "internal span"}],
            "scope_overclaim_check": {
                "status": "passed",
                "checked_claims": 0,
                "evidence_scope_by_ref": {
                    "T1": {
                        "evidence_id": "T1",
                        "driver_level": "segment_level_driver",
                        "claim_scope": "segment",
                        "allowed_claim_strength": "bounded_inference",
                        "scope_reason": "Snippet discusses segment revenue.",
                    }
                },
            },
            "scope_overclaim_violations": [],
        },
        "contract_decision": {
            "route": "repair_answer",
            "decision": "repairable",
            "severity": "repairable",
            "public_summary": "Answer was repaired and passed.",
        },
        "draft_release_decision": {
            "decision": "released_with_warnings",
            "released": True,
            "source": "analyst_draft_initial",
            "warnings": [{"code": "optional_context_missing"}],
        },
        "repair_actions": [{"attempt": 1, "violations": ["caveat_not_visible"]}],
        "tool_call_results": [
            {
                "tool_name": "query_financial_data",
                "tool_version": "1.0.0",
                "requirement_id": "REQ-REV",
                "ok": True,
                "returned_count": 1,
                "latency_ms": 3,
            }
        ],
        "prompt": "raw prompt must not leak",
        "report": {
            "title": "NVIDIA Fundamental Analysis Report",
            "ticker": "NVDA",
            "sections": [
                {
                    "section_id": "executive_summary",
                    "title": "Executive Summary",
                    "section_status": "satisfied",
                    "citations": ["N1"],
                    "key_evidence_ids": ["N1"],
                    "limitations": [],
                    "confidence": "high",
                    "markdown": "Summary [N1].",
                }
            ],
            "markdown": "# Report\n\nSummary [N1].",
            "contract_status": "passed",
        },
    }


def test_trace_ui_model_is_sanitized_and_sorts_missing_dimensions_first():
    ui = build_trace_ui_model(_trace())

    assert ui["trace_id"] == "trace-ui-test"
    assert ui["final_answer"] == "Revenue was 100 [N1]."
    assert ui["local_trace_path"] == "data/traces/trace-ui-test.json"
    assert ui["semantic_parser_mode"] == "shadow"
    assert ui["semantic_parser"]["proposal"]["requested_dimensions"] == ["cash_flow_quality"]
    assert ui["semantic_parser"]["disagreement"]["methodology_intent_disagreement"] is True
    assert ui["canonical_intent"]["intent_family"] == "cash_flow"
    assert ui["intent_merge_decision"]["source"] == "semantic_proposal_validated"
    assert ui["evidence_policy_id"] == "single_company_cash_flow_v1"
    assert ui["evidence_policy"]["core_requirements"] == ["dimension:cash_flow_quality"]
    assert ui["research_plan"]["summary"]["question_type"] == "causal_explanation"
    assert ui["research_plan"]["summary"]["valid"] is True
    assert ui["research_plan"]["legacy_evidence_plan"]["requirement_count"] == 1
    assert ui["answer_part_status_by_id"]["identify_growth_drivers"]["reason"] == "driver_text_evidence_missing_but_analyzable"
    assert ui["evidence_gap_by_answer_part"]["identify_growth_drivers"]["reason"] == "driver_text_evidence_missing_but_analyzable"
    assert ui["research_plan"]["summary"]["source"] == "validated_llm"
    assert ui["research_plan"]["summary"]["duration_ms"] == 123
    assert ui["plan_coverage_decision"]["strategy"] == "merge"
    assert ui["plan_coverage_decision"]["retained_legacy_core_count"] == 18
    assert ui["requirement_merge_summary"]["legacy_only_count"] == 18
    assert ui["evidence_plan_used"]["source"] == "merged"
    assert ui["evidence_validation_records"][0]["rejected_evidence_reason"] == "metric_mapping_failed"
    assert ui["research_plan"]["partial_required_answer_parts"] == ["quantify_growth"]
    assert ui["research_plan"]["missing_but_analyzable_answer_parts"] == ["identify_growth_drivers"]
    assert ui["analytical_reasoning"]["claim_tiers"]["hypothesis_to_verify"] == 1
    assert ui["analytical_reasoning"]["evidence_health"] == "degraded"
    assert ui["evidence_scope_by_ref"]["T1"]["claim_scope"] == "segment"
    assert ui["evidence_scope_by_ref"]["T1"]["allowed_claim_strength"] == "bounded_inference"
    assert ui["contract"]["scope_overclaim_check"]["status"] == "passed"
    assert ui["evidence_packet"]["numeric_evidence"][0]["role"] == "current_revenue"
    assert ui["evidence_packet"]["numeric_evidence"][0]["quality_status"] == "valid"
    assert ui["evidence_packet"]["numeric_evidence"][0]["source_requirement_id"] == "REQ-REV"
    assert ui["relevance_status"] == "analytical_with_gaps"
    assert ui["relevance_decision"]["decision"] == "warning"
    assert ui["intent_conflict"] is True
    assert ui["contract_status"] == "repaired"
    assert ui["contract"]["decision"] == "repairable"
    assert ui["contract_decision"]["decision"] == "repairable"
    assert ui["draft_release_decision"]["decision"] == "released_with_warnings"
    assert ui["repair_attempts"] == 1
    assert ui["evidence_plan"]["summary"]["missing_count"] == 1
    assert ui["evidence_plan"]["summary"]["scope_counts"] == {"core": 1, "optional_context": 0, "diagnostic": 1}
    assert [row["requirement_id"] for row in ui["evidence_plan"]["requirements_by_scope"]["core"]] == ["REQ-CASH"]
    assert ui["dimensions"][0]["dimension_id"] == "cash_flow_quality"
    assert ui["citations"][0]["citation_id"] == "N1"
    assert ui["citations"][0]["valid"] is True
    assert ui["report"]["sections"][0]["section_id"] == "executive_summary"

    dumped = json.dumps(ui)
    assert "raw prompt must not leak" not in dumped
    assert "internal span" not in dumped


def test_trace_ui_model_returns_progress_events():
    trace = _trace()
    trace["progress_events"] = [
        {
            "event": "run_started",
            "status": "started",
            "message": "已接收研究请求。",
            "timestamp": "2026-05-21T12:00:00.000Z",
            "metadata": {"query": "分析 NVIDIA"},
        }
    ]

    ui = build_trace_ui_model(trace)

    assert ui["progress_events"][0]["event"] == "run_started"
    assert ui["progress_events"][0]["message"] == "已接收研究请求。"


def test_trace_ui_progress_event_metadata_is_public_safe():
    trace = _trace()
    trace["progress_events"] = [
        {
            "event": "tool_started",
            "status": "started",
            "message": "正在检索 filing 文本证据。",
            "timestamp": "2026-05-21T12:00:00.000Z",
            "metadata": {
                "tool": "search_filings",
                "requirement_id": "REQ-TEXT-001",
                "returned_count": 3,
                "nested": {"raw_debug": "internal"},
                "items": ["debug-row"],
            },
        }
    ]

    ui = build_trace_ui_model(trace)

    metadata = ui["progress_events"][0]["metadata"]
    assert metadata == {
        "tool": "search_filings",
        "requirement_id": "REQ-TEXT-001",
        "returned_count": 3,
    }


def test_trace_ui_model_defaults_progress_events_to_empty():
    ui = build_trace_ui_model(_trace())

    assert ui["progress_events"] == []


def test_trace_ui_requirement_returned_falls_back_to_collection_count():
    trace = _trace()
    trace["tool_call_results"] = []

    ui = build_trace_ui_model(trace)

    by_id = {row["requirement_id"]: row for row in ui["evidence_plan"]["requirements"]}
    assert by_id["REQ-REV"]["returned"] == 1
    assert by_id["REQ-CASH"]["returned"] == 0


def test_trace_ui_numeric_role_and_quality_fall_back_to_requirement_status():
    trace = _trace()
    row = trace["evidence_packet"]["numeric_table"][0]
    row.pop("role", None)
    row.pop("evidence_role", None)
    row.pop("source_requirement_id", None)
    row.pop("quality_status", None)
    row["requirement_id"] = "REQ-REV"
    trace["requirement_status_map"] = {
        "REQ-REV": {
            "status": "satisfied",
            "evidence_role": "current_revenue",
            "quality_status": "valid",
        }
    }

    ui = build_trace_ui_model(trace)

    numeric = ui["evidence_packet"]["numeric_evidence"][0]
    assert numeric["role"] == "current_revenue"
    assert numeric["evidence_role"] == "current_revenue"
    assert numeric["quality_status"] == "valid"
    assert numeric["source_requirement_id"] == "REQ-REV"


def test_trace_ui_marks_optional_missing_requirement_as_non_blocking():
    trace = _trace()
    trace["evidence_plan"]["evidence_requirements"] = [
        {
            "requirement_id": "REQ-FCF",
            "requirement_type": "numeric",
            "dimension_id": "cash_flow_quality",
            "company": "NVDA",
            "metric": "free_cash_flow",
            "required": True,
            "requirement_scope": "core",
        },
        {
            "requirement_id": "REQ-COMPUTED-FCF",
            "requirement_type": "calculation",
            "dimension_id": "cash_flow_quality",
            "company": "NVDA",
            "metric": "free_cash_flow",
            "required": False,
            "requirement_scope": "diagnostic",
        },
    ]
    trace["evidence_collection_results"] = [
        {"requirement_id": "REQ-FCF", "status": "satisfied", "items": [{"evidence_id": "N1"}]},
        {
            "requirement_id": "REQ-COMPUTED-FCF",
            "status": "missing",
            "items": [],
            "failure_reason": "no_validated_numeric_evidence",
        },
    ]
    trace["requirement_status_map"] = {
        "REQ-FCF": {
            "requirement_id": "REQ-FCF",
            "status": "satisfied",
            "required": True,
        },
        "REQ-COMPUTED-FCF": {
            "requirement_id": "REQ-COMPUTED-FCF",
            "status": "missing",
            "required": False,
            "failure_reason": "no_validated_numeric_evidence",
        },
    }

    ui = build_trace_ui_model(trace)

    plan_summary = ui["evidence_plan"]["summary"]
    assert plan_summary["requirement_count"] == 2
    assert plan_summary["missing_count"] == 0
    assert plan_summary["total_missing_count"] == 1
    assert plan_summary["scope_counts"] == {"core": 1, "optional_context": 0, "diagnostic": 1}
    optional_row = next(row for row in ui["evidence_plan"]["requirements"] if row["requirement_id"] == "REQ-COMPUTED-FCF")
    assert optional_row["status"] == "optional_missing"
    assert optional_row["status_label"] == "optional missing"
    assert optional_row["raw_status"] == "missing"
    assert optional_row["required"] is False
    assert optional_row["scope"] == "diagnostic"
    assert optional_row["blocking"] is False


def test_trace_ui_renders_timeout_with_successful_lexical_fallback_as_success():
    trace = _trace()
    trace["tool_call_results"] = [
        {
            "tool_name": "search_filings",
            "tool_version": "1.0.0",
            "requirement_id": "REQ-TEXT-NVDA-RISK",
            "input_summary": {"ticker": "NVDA", "section_allowlist": ["ITEM_1A"], "strict_sections": True},
            "ok": False,
            "latency_ms": 45000,
            "raw_count": None,
            "returned_count": None,
            "warnings": [],
            "provenance": [],
            "error": {"code": "timeout", "message": "Tool execution timed out", "retryable": True},
        }
    ]
    trace["retrieval_debug"] = {
        "requirement_calls": [
            {
                "requirement_id": "REQ-TEXT-NVDA-RISK",
                "ticker": "NVDA",
                "strategy": "strict_intent_query",
                "fallback_after_timeout": True,
                "backend": "duckdb_lexical",
                "returned": 5,
                "raw_hit_count": 250,
                "usable_hit_count": 5,
                "failure_reason": None,
                "timeout_error": "Tool execution timed out",
            }
        ]
    }

    ui = build_trace_ui_model(trace)

    call = ui["tool_calls"][0]
    assert call["ok"] is True
    assert call["error"] is None
    assert call["returned_count"] == 5
    assert call["raw_count"] == 250
    assert call["input_summary"]["backend"] == "duckdb_lexical"
    assert call["fallback_after_timeout"] is True
    assert "DuckDB lexical fallback succeeded" in call["warnings"][0]


def test_trace_ui_renders_resource_error_with_successful_lexical_fallback_as_success():
    trace = _trace()
    trace["tool_call_results"] = [
        {
            "tool_name": "search_filings",
            "tool_version": "1.0.0",
            "requirement_id": "REQ-TEXT-AMZN-RISK-MDA",
            "input_summary": {"ticker": "AMZN", "section_allowlist": ["ITEM_2"], "strict_sections": True},
            "ok": False,
            "latency_ms": 428,
            "raw_count": None,
            "returned_count": None,
            "warnings": [],
            "provenance": [],
            "error": {"code": "execution_error", "message": "CUDA out of memory", "retryable": False},
        }
    ]
    trace["retrieval_debug"] = {
        "requirement_calls": [
            {
                "requirement_id": "REQ-TEXT-AMZN-RISK-MDA",
                "ticker": "AMZN",
                "strategy": "strict_intent_query",
                "fallback_after_error": True,
                "fallback_error_code": "execution_error",
                "backend": "duckdb_lexical",
                "returned": 5,
                "raw_hit_count": 40,
                "usable_hit_count": 3,
                "failure_reason": None,
                "fallback_error": "CUDA out of memory",
            }
        ]
    }

    ui = build_trace_ui_model(trace)

    call = ui["tool_calls"][0]
    assert call["ok"] is True
    assert call["error"] is None
    assert call["returned_count"] == 3
    assert call["raw_count"] == 40
    assert call["input_summary"]["fallback_after_error"] is True
    assert call["fallback_after_error"] is True
    assert call["fallback_error_code"] == "execution_error"
    assert "resource error" in call["warnings"][0]


def test_trace_ui_route_reads_sanitized_model():
    app = create_app()
    traces_dir = settings.traces_dir
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / "trace-ui-test.json"
    path.write_text(json.dumps(_trace()))

    async def _request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/trace/trace-ui-test/ui")

    try:
        resp = asyncio.run(_request())
        assert resp.status_code == 200
        body = resp.json()
        assert body["trace_id"] == "trace-ui-test"
        assert body["contract"]["violation_codes"] == ["caveat_not_visible"]
        assert body["contract_decision"]["decision"] == "repairable"
        assert body["draft_release_decision"]["decision"] == "released_with_warnings"
        assert "prompt" not in body
    finally:
        path.unlink(missing_ok=True)


def test_trace_ui_route_returns_progress_events_for_partial_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    app = create_app()
    trace_id = "partial-progress-trace"
    traces_dir = settings.traces_dir
    traces_dir.mkdir(parents=True, exist_ok=True)
    path = traces_dir / f"{trace_id}.json"
    path.write_text(
        json.dumps(
            {
                "trace_id": trace_id,
                "progress_events": [
                    {
                        "event": "intent_resolved",
                        "status": "completed",
                        "message": "已识别为单公司分析。",
                        "timestamp": "2026-05-21T12:00:01.000Z",
                    }
                ],
            }
        )
    )

    async def _request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/trace/{trace_id}/ui")

    resp = asyncio.run(_request())
    assert resp.status_code == 200
    body = resp.json()
    assert body["trace_id"] == trace_id
    assert body["progress_events"][0]["event"] == "intent_resolved"


def test_frontend_progress_event_mapping(tmp_path):
    node = shutil.which("node")
    if not node:
        return
    app_copy = tmp_path / "app.mjs"
    app_copy.write_text((ROOT / "frontend" / "app.js").read_text(encoding="utf-8"), encoding="utf-8")
    script = f"""
      import {{ progressStageIndexForEvent, progressBadgeForEvents, safeProgressEvents }} from {str(app_copy)!r};
      const filing = {{ event: 'tool_started', status: 'started', message: 'm', timestamp: 't', metadata: {{ tool: 'search_filings' }} }};
      if (progressStageIndexForEvent(filing) !== 3) throw new Error('filing stage mismatch');
      const badge = progressBadgeForEvents([filing]);
      if (badge.text !== 'Retrieval') throw new Error(`bad badge ${{badge.text}}`);
      const events = safeProgressEvents({{ progress_events: [filing, {{ event: 'bad' }}] }});
      if (events.length !== 1) throw new Error('event filtering mismatch');
    """
    result = subprocess.run([node, "--input-type=module", "-e", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_debug_bundle_includes_progress_events(tmp_path):
    node = shutil.which("node")
    if not node:
        return
    app_copy = tmp_path / "app.mjs"
    app_copy.write_text((ROOT / "frontend" / "app.js").read_text(encoding="utf-8"), encoding="utf-8")
    script = f"""
      import {{ buildDebugBundle }} from {str(app_copy)!r};
      const bundle = buildDebugBundle({{}}, {{
        trace_id: 'progress-debug',
        query: '分析下amazon这家公司',
        task_type: 'report_summary',
        answer_mode: 'analytical',
        contract_status: 'passed',
        final_answer: 'answer',
        plan_coverage_decision: {{ strategy: 'merge', legacy_core_count: 18, research_core_count: 3, retained_legacy_core_count: 18, coverage_ratio: 0.17, warnings: ['research_plan_under_covered_legacy_core'], reason: 'overview_research_plan_augments_legacy_coverage' }},
        requirement_merge_summary: {{ merged_total_requirements: 21, deduped_requirements: 1, legacy_only_count: 18, research_only_count: 2, legacy_research_count: 1, retained_legacy_core_count: 18 }},
        evidence_validation_records: [{{ requirement_id: 'REQ-AMZN-REV', evidence_type: 'numeric', tool_returned_count: 1, validated_evidence_count: 0, rejected_evidence_reason: 'metric_mapping_failed', status: 'missing' }}],
        evidence_plan: {{ summary: {{ requirement_count: 0 }} }},
        evidence_packet: {{
          summary: {{ numeric_count: 0, text_count: 1, computed_count: 0, event_count: 0 }},
          text_evidence: [{{ evidence_id: 'T1', ticker: 'NVDA', claim_scope: 'segment', allowed_claim_strength: 'bounded_inference', driver_level: 'segment_level_driver', supporting_snippet: 'Compute & Networking revenue increased.' }}],
        }},
        evidence_scope: {{ rows: [{{ evidence_id: 'T1', claim_scope: 'segment', allowed_claim_strength: 'bounded_inference', driver_level: 'segment_level_driver', summary_scope_warning: 'evidence_summary_scope_overclaim', scope_reason: 'segment discussion' }}] }},
        scope_overclaim_check: {{ status: 'passed', checked_claims: 1, evidence_summary_warnings: [{{ code: 'evidence_summary_scope_overclaim' }}] }},
        scope_overclaim_violations: [],
        contract: {{ status: 'passed', decision: 'passed', scope_overclaim_check: {{ status: 'passed', checked_claims: 1 }}, scope_overclaim_violations: [] }},
        draft_release_decision: {{ decision: 'released', released: true, warnings: [] }},
        dimensions: [],
        citations: [],
        progress_events: [
          {{ event: 'run_started', status: 'started', message: '已接收研究请求。', timestamp: '2026-05-21T12:00:00Z', elapsed_ms: 0, node: 'api_chat', metadata: {{ query: 'hidden from summary' }} }},
          {{ event: 'tool_started', status: 'started', message: '正在检索 AMZN 的 SEC filing 文本证据。', timestamp: '2026-05-21T12:00:02Z', elapsed_ms: 2000, node: 'execute_tools', metadata: {{ tool: 'search_filings', requirement_id: 'REQ-TEXT-AMZN-RISK', company: 'AMZN', ignored: {{ nested: true }} }} }},
          {{ event: 'answer_released', status: 'completed', message: '分析完成，最终答案已发布。', timestamp: '2026-05-21T12:00:08Z', elapsed_ms: 8000, node: 'finalize', metadata: {{ final_status: 'passed' }} }},
        ],
      }});
      if (!bundle.includes('## Progress Events')) throw new Error(bundle);
      if (!bundle.includes('## 3. Plan Coverage')) throw new Error(bundle);
      if (!bundle.includes('## 4. Requirement Merge')) throw new Error(bundle);
      if (!bundle.includes('### Evidence Validation')) throw new Error(bundle);
      if (!bundle.includes('strategy: merge')) throw new Error(bundle);
      if (!bundle.includes('rejected=metric_mapping_failed')) throw new Error(bundle);
      if (!bundle.includes('### Evidence Scope')) throw new Error(bundle);
      if (!bundle.includes('### Scope Overclaim Check')) throw new Error(bundle);
      if (!bundle.includes('claim_scope=segment')) throw new Error(bundle);
      if (!bundle.includes('evidence_summary_scope_overclaim')) throw new Error(bundle);
      for (const token of ['run_started', 'tool_started', 'answer_released', 'tool=search_filings', 'requirement_id=REQ-TEXT-AMZN-RISK', 'final_status=passed']) {{
        if (!bundle.includes(token)) throw new Error(`missing ${{token}}\\n${{bundle}}`);
      }}
      if (bundle.includes('[object Object]')) throw new Error(bundle);
      if (bundle.includes('nested=true')) throw new Error(bundle);
    """
    result = subprocess.run([node, "--input-type=module", "-e", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
