"""API endpoint tests — Phase 5.

Tests the FastAPI routes:
  - GET /health (always works)
  - POST /chat (requires vLLM + data)
  - GET /trace/{trace_id} (requires a prior /chat call)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import socket
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from src.api import routes
from src.api.app import create_app
from src.api.models import ChatRequest, TaskView

ROOT = Path(__file__).resolve().parent.parent

app = create_app()


async def _request_async(method: str, path: str, **kwargs):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.request(method=method, url=path, **kwargs)


def _request(method: str, path: str, **kwargs):
    return asyncio.run(_request_async(method, path, **kwargs))


# ---------------------------------------------------------------------------
# Health check — always works
# ---------------------------------------------------------------------------


class TestHealth:

    def test_health_ok(self):
        resp = _request("GET", "/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Model / route structure tests (no LLM)
# ---------------------------------------------------------------------------


class TestModels:

    def test_chat_request_validation(self):
        resp = _request("POST", "/chat", json={"query": ""})
        assert resp.status_code == 422  # empty query rejected

    def test_chat_request_too_long(self):
        resp = _request("POST", "/chat", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_trace_not_found(self):
        resp = _request("GET", "/trace/nonexistent-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_task_view_preserves_methodology_fields(self):
        view = TaskView(
            kind="methodology_comparison_brief",
            methodology_answer={"judgment": "methodology", "metric_table": [{"metric_id": "net_margin"}]},
            metric_table=[{"metric_id": "net_margin"}],
            single_company_metric_table=[{"metric_id": "revenue"}],
            dimension_sections=[{"dimension_id": "profitability_quality"}],
            methodology_counterpoints=["counterpoint"],
            methodology_limitations=["limitation"],
            follow_up_metrics=["cash flow"],
        ).model_dump()

        assert view["methodology_answer"]["judgment"] == "methodology"
        assert view["methodology_answer"]["metric_table"][0]["metric_id"] == "net_margin"
        assert view["metric_table"][0]["metric_id"] == "net_margin"
        assert view["single_company_metric_table"][0]["metric_id"] == "revenue"
        assert view["dimension_sections"][0]["dimension_id"] == "profitability_quality"
        assert view["methodology_counterpoints"] == ["counterpoint"]
        assert view["methodology_limitations"] == ["limitation"]
        assert view["follow_up_metrics"] == ["cash flow"]

    def test_chat_route_does_not_run_answer_contract_inline(self):
        chat_source = inspect.getsource(routes.chat)
        assert "check_answer_evidence_contract" not in chat_source
        assert "async def chat" not in chat_source
        assert "def chat" in chat_source

    def test_chat_accepts_client_trace_id(self, monkeypatch, tmp_path):
        from config import settings

        client_trace_id = "client-trace-123"
        monkeypatch.setattr(settings, "data_dir", tmp_path)

        class FakeAgent:
            def invoke(self, payload):
                assert payload["trace_id"] == client_trace_id
                assert payload["run_started_at"]
                return {
                    "trace_id": payload["trace_id"],
                    "final_answer": "Done [N1].",
                    "citations": [],
                    "selected_tools": [],
                    "task_type": "report_summary",
                    "output": {},
                    "contract_status": "passed",
                    "contract_decision": {"decision": "passed"},
                }

        monkeypatch.setattr("src.agent.graph.compile_agent", lambda: FakeAgent())

        resp = routes.chat(ChatRequest(query="分析 AMZN", client_trace_id=client_trace_id))

        assert resp.trace_id == client_trace_id
        trace_path = settings.traces_dir / f"{client_trace_id}.json"
        payload = json.loads(trace_path.read_text())
        assert payload["trace_id"] == client_trace_id
        assert payload["progress_events"][0]["event"] == "run_started"

    def test_chat_progress_events_lifecycle(self, monkeypatch, tmp_path):
        from config import settings
        from src.agent.progress import append_progress_event

        client_trace_id = "client-trace-lifecycle"
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        append_progress_event(client_trace_id, "run_started", "started", "已接收研究请求。")
        append_progress_event(client_trace_id, "intent_resolved", "completed", "已识别为单公司分析。", node="classify")
        append_progress_event(client_trace_id, "evidence_plan_built", "completed", "已生成证据计划。", node="classify")
        append_progress_event(client_trace_id, "contract_checked", "completed", "答案合约检查通过。", node="contract_check")
        append_progress_event(
            client_trace_id,
            "answer_released",
            "completed",
            "分析完成，最终答案已发布。",
            node="finalize",
            metadata={"final_status": "passed"},
        )

        routes._save_trace(
            client_trace_id,
            {
                "trace_id": client_trace_id,
                "final_answer": "Done [N1].",
                "contract_status": "passed",
            },
        )

        payload = json.loads((settings.traces_dir / f"{client_trace_id}.json").read_text())
        events = [item["event"] for item in payload["progress_events"]]
        assert events[:1] == ["run_started"]
        assert "intent_resolved" in events
        assert "evidence_plan_built" in events
        assert "contract_checked" in events
        assert events[-1] == "answer_released"

    def test_chat_rejects_invalid_client_trace_id(self, monkeypatch, tmp_path):
        from config import settings

        monkeypatch.setattr(settings, "data_dir", tmp_path)

        with pytest.raises(ValidationError):
            ChatRequest(query="分析 AMZN", client_trace_id="../bad.trace")
        assert not settings.traces_dir.exists()


class TestTraceRead:

    def test_read_existing_trace(self):
        """Write a trace to the real traces_dir, read it back via API."""
        from config import settings

        traces_dir = settings.traces_dir
        traces_dir.mkdir(parents=True, exist_ok=True)

        trace_data = {
            "trace_id": "test-123",
            "user_query": "test question",
            "task_type": "fact_qa",
            "answer_mode": "direct_fact",
            "safety_intent": "normal",
            "needs_clarification": False,
            "clarification_question": None,
            "needs_tools": True,
            "data_route": "structured_only",
            "analysis_plan_raw": {"proposed_tools": ["query_financial_data"]},
            "analysis_plan": {
                "validated_tools": ["query_financial_data"],
                "metric_requirements": ["revenue"],
                "rejected_plan_items": [],
            },
            "validated_tools": ["query_financial_data"],
            "rejected_plan_items": [],
            "safety_decision": {"answer_mode": "direct_fact", "safety_intent": "normal", "needs_tools": True},
            "safety_policy_reasons": [],
            "safety_limitations": [],
            "evidence_plan": {
                "evidence_requirements": [
                    {
                        "requirement_id": "REQ-NUM-AAPL-FACT",
                        "requirement_type": "numeric",
                        "company": "AAPL",
                        "metrics": ["revenue"],
                        "required": True,
                        "min_results": 1,
                    }
                ],
                "rejected_requirements": [],
            },
            "evidence_plan_summary": {
                "required_count": 1,
                "missing_count": 0,
            },
            "evidence_collection_results": [
                {
                    "requirement_id": "REQ-NUM-AAPL-FACT",
                    "status": "satisfied",
                    "evidence_type": "numeric",
                    "items": [],
                    "retry_count": 0,
                }
            ],
            "evidence_sufficiency": {
                "overall_status": "sufficient",
                "satisfied_requirements": ["REQ-NUM-AAPL-FACT"],
                "partial_requirements": [],
                "missing_requirements": [],
                "can_synthesize": True,
                "requirement_limitations": [],
            },
            "evidence_sufficiency_summary": {
                "overall_status": "sufficient",
                "missing_count": 0,
            },
            "evidence_retry_history": [],
            "retry_history": [],
            "requirement_limitations": [],
            "collected_evidence_by_requirement": {},
            "requirement_status_map": {
                "REQ-NUM-AAPL-FACT": {
                    "requirement_id": "REQ-NUM-AAPL-FACT",
                    "requirement_type": "numeric",
                    "company": "AAPL",
                    "purpose": "",
                    "required": True,
                    "status": "satisfied",
                    "failure_reason": None,
                    "retry_count": 0,
                }
            },
            "trace_summary": {
                "sufficiency_status": "sufficient",
                "missing_requirements_count": 0,
                "required_numeric_satisfied_rate": 1.0,
                "required_text_satisfied_rate": 1.0,
                "company_evidence_balance": 1.0,
                "degradation_reason": None,
                "final_synthesis_mode": "direct_fact",
            },
            "missing_requirements": [],
            "degradation_reason": None,
            "rejected_requirements": [],
            "evidence_packet": {"numeric_table": [], "text_snippets": []},
            "evidence_packet_summary": {"numeric_row_count": 0, "text_snippet_count": 0},
            "analyst_draft": {},
            "analyst_draft_validation": {"status": "invalid", "accepted_draft": {}},
            "draft_status": "invalid",
            "synthesis": {
                "short_answer": "AAPL revenue is 100.",
                "key_facts": [],
                "analysis": [],
                "risks_or_uncertainties": [],
                "limitations": [],
                "citations": [],
                "synthesis_strategy": "numeric_only",
                "synthesis_mode": "direct_fact",
                "unsupported_synthesis_items": [],
            },
            "synthesis_strategy": "numeric_only",
            "synthesis_mode": "direct_fact",
            "unsupported_synthesis_items": [],
            "why_tools_skipped": [],
            "companies": ["AAPL"],
            "comparison_target": None,
            "time_range": None,
            "period_query": {},
            "resolved_period_context": {},
            "comparison_basis_label": "",
            "requested_metrics": ["revenue"],
            "selected_tools": ["query_financial_data"],
            "retrieval_policy": {},
            "retrieval_debug": {},
            "tool_results": [],
            "citations": [],
            "output": {
                "protocol_version": "phase4.v1",
                "task_type": "fact_qa",
                "answer_mode": "direct_fact",
                "safety_intent": "normal",
                "needs_tools": True,
                "needs_clarification": False,
                "clarification_question": None,
                "title": "财报事实问答",
                "summary": "AAPL revenue is 100.",
                "comparison_basis": "",
                "key_points": ["AAPL revenue is 100."],
                "numeric_evidence": [],
                "text_evidence": [],
                "limitations": [],
                "used_tools": ["query_financial_data"],
                "trace_id": "test-123",
                "synthesis_mode": "direct_fact",
                "view": {"kind": "fact_qa"},
            },
            "evidence_loop_count": 1,
            "final_answer": "Test answer",
        }
        trace_path = traces_dir / "test-123.json"
        trace_path.write_text(json.dumps(trace_data))

        try:
            resp = _request("GET", "/trace/test-123")
            assert resp.status_code == 200
            body = resp.json()
            assert body["trace_id"] == "test-123"
            assert body["task_type"] == "fact_qa"
            assert body["answer_mode"] == "direct_fact"
            assert body["safety_intent"] == "normal"
            assert body["needs_tools"] is True
            assert body["needs_clarification"] is False
            assert body["analysis_plan_raw"]["proposed_tools"] == ["query_financial_data"]
            assert body["analysis_plan"]["validated_tools"] == ["query_financial_data"]
            assert body["validated_tools"] == ["query_financial_data"]
            assert body["rejected_plan_items"] == []
            assert body["safety_decision"]["answer_mode"] == "direct_fact"
            assert body["safety_policy_reasons"] == []
            assert body["safety_limitations"] == []
            assert body["evidence_plan"]["evidence_requirements"][0]["requirement_id"] == "REQ-NUM-AAPL-FACT"
            assert body["evidence_plan_summary"]["required_count"] == 1
            assert body["evidence_collection_results"][0]["status"] == "satisfied"
            assert body["evidence_sufficiency"]["overall_status"] == "sufficient"
            assert body["evidence_sufficiency_summary"]["missing_count"] == 0
            assert body["evidence_requirements"][0]["requirement_id"] == "REQ-NUM-AAPL-FACT"
            assert body["evidence_retry_history"] == []
            assert body["retry_history"] == []
            assert body["requirement_limitations"] == []
            assert body["collected_evidence_by_requirement"]["REQ-NUM-AAPL-FACT"]["status"] == "satisfied"
            assert body["requirement_status_map"]["REQ-NUM-AAPL-FACT"]["status"] == "satisfied"
            assert body["trace_summary"]["sufficiency_status"] == "sufficient"
            assert body["trace_summary"]["final_synthesis_mode"] == "direct_fact"
            assert body["missing_requirements"] == []
            assert body["degradation_reason"] is None
            assert body["rejected_requirements"] == []
            assert body["evidence_packet_summary"]["numeric_row_count"] == 0
            assert body["analyst_draft_validation"]["status"] == "invalid"
            assert body["draft_validation"]["status"] == "invalid"
            assert body["draft_revision_attempts"] == []
            assert body["draft_status"] == "invalid"
            assert body["final_answer_source"] == "deterministic_synthesis"
            assert body["synthesis_strategy"] == "numeric_only"
            assert body["synthesis_mode"] == "direct_fact"
            assert body["synthesis"]["short_answer"] == "AAPL revenue is 100."
            assert body["unsupported_synthesis_items"] == []
            assert body["why_tools_skipped"] == []
            assert body["final_answer"] == "Test answer"
            assert isinstance(body["numeric_evidence"], list)
            assert isinstance(body["text_evidence"], list)
            assert isinstance(body["unsupported_claims"], list)
            assert isinstance(body["numeric_citations"], list)
            assert isinstance(body["text_citations"], list)
            assert isinstance(body["period_query"], dict)
            assert isinstance(body["resolved_period_context"], dict)
            assert isinstance(body["comparison_basis_label"], str)
            assert isinstance(body["retrieval_policy"], dict)
            assert isinstance(body["retrieval_debug"], dict)
            assert isinstance(body["event_intent"], str)
            assert isinstance(body["market_reaction_requested"], bool)
            assert isinstance(body["event_query"], dict)
            assert isinstance(body["event_results"], list)
            assert isinstance(body["market_reaction_evidence"], list)
            assert isinstance(body["market_reaction_limitations"], list)
            assert isinstance(body["output"], dict)
        finally:
            trace_path.unlink(missing_ok=True)

    def test_methodology_trace_reports_dimension_gaps(self):
        from config import settings

        traces_dir = settings.traces_dir
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_data = {
            "trace_id": "test-methodology-gaps",
            "user_query": "分析下 nvidia",
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "analysis_scope": "single_company",
            "needs_tools": True,
            "data_route": "hybrid",
            "selected_analysis_framework": {
                "framework_id": "fundamental_quality_analysis",
                "active_dimension_ids": ["revenue_quality", "cash_flow_quality", "moat_and_competitive_risk"],
                "dimensions": [
                    {"id": "revenue_quality", "name": "Revenue Quality"},
                    {"id": "cash_flow_quality", "name": "Cash Flow Quality"},
                    {"id": "moat_and_competitive_risk", "name": "Moat And Competitive Risk"},
                ],
            },
            "evidence_plan": {
                "evidence_requirements": [
                    {
                        "requirement_id": "REQ-METH-NVDA-REVENUE_QUALITY_REVENUE",
                        "requirement_type": "numeric",
                        "dimension_id": "revenue_quality",
                        "company": "NVDA",
                        "metric": "revenue",
                        "metrics": ["revenue"],
                        "required": True,
                    },
                    {
                        "requirement_id": "REQ-METH-NVDA-CASH_FLOW_QUALITY-OPERATING_CASH_FLOW",
                        "requirement_type": "numeric",
                        "dimension_id": "cash_flow_quality",
                        "company": "NVDA",
                        "metric": "operating_cash_flow",
                        "metrics": ["operating_cash_flow"],
                        "required": True,
                    },
                    {
                        "requirement_id": "REQ-TEXT-NVDA-RISK",
                        "requirement_type": "text",
                        "dimension_id": "moat_and_competitive_risk",
                        "company": "NVDA",
                        "required": True,
                    },
                ],
            },
            "evidence_collection_results": [
                {"requirement_id": "REQ-METH-NVDA-REVENUE_QUALITY_REVENUE", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
                {"requirement_id": "REQ-METH-NVDA-CASH_FLOW_QUALITY-OPERATING_CASH_FLOW", "status": "missing", "evidence_type": "numeric", "items": [], "failure_reason": "no_matching_financial_fact"},
                {"requirement_id": "REQ-TEXT-NVDA-RISK", "status": "satisfied", "evidence_type": "text", "items": [{}]},
            ],
            "evidence_sufficiency": {"overall_status": "partial", "can_synthesize": True},
            "evidence_packet": {
                "active_dimensions": ["revenue_quality", "cash_flow_quality", "moat_and_competitive_risk"],
                "numeric_evidence_by_dimension": {
                    "revenue_quality": [{"evidence_id": "N1", "metric": "revenue", "ticker": "NVDA"}]
                },
                "text_evidence_by_dimension": {
                    "moat_and_competitive_risk": [
                        {
                            "evidence_id": "T1",
                            "ticker": "NVDA",
                            "section": "ITEM_1A",
                            "claim": "NVDA faces competition risk.",
                            "citation_ref": "T1",
                        }
                    ]
                },
                "text_snippets": [],
                "numeric_table": [],
            },
            "trace_summary": {"final_synthesis_mode": "methodology_single_company"},
            "companies": ["NVDA"],
            "requested_metrics": [],
            "selected_tools": [],
            "tool_results": [],
            "citations": [],
            "output": {"task_type": "report_summary", "title": "", "summary": "", "key_points": [], "numeric_evidence": [], "text_evidence": [], "limitations": [], "used_tools": [], "trace_id": "test-methodology-gaps", "view": {"kind": "methodology_single_company_brief"}},
            "evidence_loop_count": 1,
            "final_answer": "answer",
        }
        trace_path = traces_dir / "test-methodology-gaps.json"
        trace_path.write_text(json.dumps(trace_data))

        try:
            resp = _request("GET", "/trace/test-methodology-gaps")
            assert resp.status_code == 200
            body = resp.json()
            assert body["available_metrics_by_dimension"]["revenue_quality"] == ["revenue"]
            assert "operating_cash_flow" in body["missing_metrics_by_dimension"]["cash_flow_quality"]
            assert body["text_evidence_by_dimension"]["moat_and_competitive_risk"][0]["evidence_id"] == "T1"
            assert body["final_methodology_coverage_rate"] is not None
        finally:
            trace_path.unlink(missing_ok=True)

    def test_trace_prefers_final_canonical_summary_over_stale_stored_values(self):
        from config import settings

        traces_dir = settings.traces_dir
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_data = {
            "trace_id": "test-final-canonical",
            "user_query": "aapple和amazon你最看好哪个",
            "task_type": "company_comparison",
            "answer_mode": "comparison_brief",
            "safety_intent": "investment_advice_like",
            "needs_tools": True,
            "companies": ["AAPL", "AMZN"],
            "comparison_target": "AMZN",
            "evidence_plan": {
                "task_type": "company_comparison",
                "answer_mode": "comparison_brief",
                "safety_intent": "investment_advice_like",
                "evidence_requirements": [
                    {"requirement_id": "REQ-NUM-AAPL-REVENUE", "requirement_type": "numeric", "company": "AAPL", "required": True},
                    {"requirement_id": "REQ-NUM-AMZN-REVENUE", "requirement_type": "numeric", "company": "AMZN", "required": True},
                    {"requirement_id": "REQ-TEXT-AAPL-BALANCED_CONTEXT", "requirement_type": "text", "company": "AAPL", "required": True},
                    {"requirement_id": "REQ-TEXT-AMZN-BALANCED_CONTEXT", "requirement_type": "text", "company": "AMZN", "required": True},
                ],
            },
            "evidence_collection_results": [
                {"requirement_id": "REQ-NUM-AAPL-REVENUE", "status": "satisfied", "evidence_type": "numeric", "items": [{"requirement_id": "REQ-NUM-AAPL-REVENUE"}]},
                {"requirement_id": "REQ-NUM-AMZN-REVENUE", "status": "satisfied", "evidence_type": "numeric", "items": [{"requirement_id": "REQ-NUM-AMZN-REVENUE"}]},
                {"requirement_id": "REQ-TEXT-AAPL-BALANCED_CONTEXT", "status": "missing", "evidence_type": "text", "items": [], "failure_reason": "comparison_text_unbalanced"},
                {"requirement_id": "REQ-TEXT-AMZN-BALANCED_CONTEXT", "status": "missing", "evidence_type": "text", "items": [], "failure_reason": "comparison_text_unbalanced"},
            ],
            "evidence_sufficiency": {
                "overall_status": "partial",
                "satisfied_requirements": ["REQ-NUM-AAPL-REVENUE", "REQ-NUM-AMZN-REVENUE"],
                "partial_requirements": [],
                "missing_requirements": ["REQ-TEXT-AAPL-BALANCED_CONTEXT", "REQ-TEXT-AMZN-BALANCED_CONTEXT"],
                "degradation_reason": "numeric_only_comparison",
                "can_synthesize": True,
                "requirement_limitations": [
                    {"code": "requirement_missing", "severity": "high", "requirement_id": "REQ-TEXT-AAPL-BALANCED_CONTEXT", "failure_reason": "comparison_text_unbalanced"},
                    {"code": "requirement_missing", "severity": "high", "requirement_id": "REQ-TEXT-AMZN-BALANCED_CONTEXT", "failure_reason": "comparison_text_unbalanced"},
                    {"code": "numeric_only_comparison", "severity": "medium"},
                ],
                "required_numeric_satisfied_rate": 1.0,
                "required_text_satisfied_rate": 0.0,
                "company_evidence_balance": 0.5,
            },
            "collection_evidence_collection_results": [
                {"requirement_id": "REQ-NUM-AAPL-REVENUE", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
                {"requirement_id": "REQ-NUM-AMZN-REVENUE", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
                {"requirement_id": "REQ-TEXT-AAPL-BALANCED_CONTEXT", "status": "satisfied", "evidence_type": "text", "items": [{}]},
                {"requirement_id": "REQ-TEXT-AMZN-BALANCED_CONTEXT", "status": "satisfied", "evidence_type": "text", "items": [{}]},
            ],
            "collection_evidence_sufficiency": {
                "overall_status": "sufficient",
                "satisfied_requirements": [
                    "REQ-NUM-AAPL-REVENUE",
                    "REQ-NUM-AMZN-REVENUE",
                    "REQ-TEXT-AAPL-BALANCED_CONTEXT",
                    "REQ-TEXT-AMZN-BALANCED_CONTEXT",
                ],
                "partial_requirements": [],
                "missing_requirements": [],
                "can_synthesize": True,
                "required_numeric_satisfied_rate": 1.0,
                "required_text_satisfied_rate": 1.0,
                "company_evidence_balance": 1.0,
            },
            "requirement_status_map": {
                "REQ-TEXT-AAPL-BALANCED_CONTEXT": {
                    "requirement_id": "REQ-TEXT-AAPL-BALANCED_CONTEXT",
                    "status": "satisfied",
                    "raw_hit_count": 4,
                    "section_filtered_hit_count": 4,
                    "usable_hit_count": 2,
                    "snippet_support_passed_count": 2,
                    "validated_text_claim_count": 1,
                    "text_claim_validated_count": 1,
                    "text_citation_kept_count": 0,
                    "final_validated_text_count": 0,
                    "drop_stage": "citation_policy_dropped",
                },
            },
            "text_requirement_diagnostics": {
                "REQ-TEXT-AAPL-BALANCED_CONTEXT": {
                    "raw_hit_count": 4,
                    "section_filtered_hit_count": 4,
                    "usable_hit_count": 2,
                    "snippet_support_passed_count": 2,
                    "validated_text_claim_count": 1,
                    "text_claim_validated_count": 1,
                    "text_citation_kept_count": 0,
                    "final_validated_text_count": 0,
                    "drop_stage": "citation_policy_dropped",
                }
            },
            "trace_summary": {
                "sufficiency_status": "sufficient",
                "missing_requirements_count": 0,
                "required_numeric_satisfied_rate": 1.0,
                "required_text_satisfied_rate": 1.0,
                "company_evidence_balance": 1.0,
                "degradation_reason": None,
                "final_synthesis_mode": "balanced_comparison",
            },
            "synthesis_mode": "limited_judgment",
            "numeric_evidence": [{"evidence_id": "N1"}],
            "text_evidence": [],
            "output": {
                "task_type": "company_comparison",
                "answer_mode": "comparison_brief",
                "safety_intent": "investment_advice_like",
                "title": "对比",
                "summary": "limited",
                "key_points": ["limited"],
                "numeric_evidence": [{"evidence_id": "N1"}],
                "text_evidence": [],
                "limitations": [{"code": "numeric_only_comparison", "severity": "medium", "message": "limited"}],
                "used_tools": ["query_financial_data", "search_filings"],
                "trace_id": "test-final-canonical",
                "view": {"kind": "comparison_brief"},
                "protocol_version": "phase4.v1",
                "synthesis_mode": "limited_judgment",
            },
            "requested_metrics": ["revenue", "net_income"],
            "selected_tools": ["query_financial_data", "search_filings"],
            "retrieval_policy": {},
            "retrieval_debug": {},
            "tool_results": [],
            "citations": [],
            "evidence_loop_count": 1,
            "final_answer": "limited",
        }
        trace_path = traces_dir / "test-final-canonical.json"
        trace_path.write_text(json.dumps(trace_data))

        try:
            resp = _request("GET", "/trace/test-final-canonical")
            assert resp.status_code == 200
            body = resp.json()
            assert body["trace_summary"]["sufficiency_status"] == "partial"
            assert body["trace_summary"]["missing_requirements_count"] == 2
            assert body["trace_summary"]["required_text_satisfied_rate"] == 0.0
            assert body["trace_summary"]["final_synthesis_mode"] == "limited_judgment"
            assert body["requirement_status_map"]["REQ-TEXT-AAPL-BALANCED_CONTEXT"]["status"] == "missing"
            assert body["requirement_status_map"]["REQ-TEXT-AAPL-BALANCED_CONTEXT"]["drop_stage"] == "citation_policy_dropped"
            assert body["validated_text_evidence_count"] == 0
            assert body["text_requirement_diagnostics"]["REQ-TEXT-AAPL-BALANCED_CONTEXT"]["validated_text_claim_count"] == 1
            assert body["final_answer_source"] == "deterministic_synthesis"
        finally:
            trace_path.unlink(missing_ok=True)

    def test_trace_exposes_methodology_fields_from_final_accounting_and_packet(self):
        from config import settings

        traces_dir = settings.traces_dir
        traces_dir.mkdir(parents=True, exist_ok=True)
        trace_data = {
            "trace_id": "test-methodology-trace",
            "user_query": "AAPL 估值便宜吗",
            "task_type": "fact_qa",
            "answer_mode": "direct_fact",
            "safety_intent": "normal",
            "needs_tools": True,
            "data_route": "hybrid",
            "companies": ["AAPL"],
            "selected_analysis_framework": {
                "framework_id": "fundamental_quality_analysis",
                "active_dimension_ids": ["revenue_quality", "valuation_and_risk_boundary"],
                "dimensions": [
                    {"id": "revenue_quality", "name": "Revenue Quality"},
                    {"id": "valuation_and_risk_boundary", "name": "Valuation And Risk Boundary"},
                ],
            },
            "evidence_plan": {
                "evidence_requirements": [
                    {
                        "requirement_id": "REQ-NUM-AAPL-REVENUE",
                        "requirement_type": "numeric",
                        "company": "AAPL",
                        "metric": "revenue",
                        "dimension_id": "revenue_quality",
                        "required": True,
                    },
                    {
                        "requirement_id": "REQ-NUM-AAPL-PRICE",
                        "requirement_type": "numeric",
                        "company": "AAPL",
                        "metric": "price",
                        "dimension_id": "valuation_and_risk_boundary",
                        "required": True,
                    },
                ],
            },
            "evidence_collection_results": [
                {
                    "requirement_id": "REQ-NUM-AAPL-REVENUE",
                    "status": "satisfied",
                    "evidence_type": "numeric",
                    "items": [{"requirement_id": "REQ-NUM-AAPL-REVENUE"}],
                    "retry_count": 0,
                },
                {
                    "requirement_id": "REQ-NUM-AAPL-PRICE",
                    "status": "missing",
                    "evidence_type": "numeric",
                    "items": [],
                    "retry_count": 0,
                },
            ],
            "evidence_sufficiency": {
                "overall_status": "partial",
                "satisfied_requirements": ["REQ-NUM-AAPL-REVENUE"],
                "missing_requirements": ["REQ-NUM-AAPL-PRICE"],
                "can_synthesize": True,
            },
            "dimension_status_map": {
                "valuation_and_risk_boundary": {"status": "satisfied"}
            },
            "evidence_packet": {
                "active_dimensions": ["revenue_quality", "valuation_and_risk_boundary"],
                "red_flags": [
                    {
                        "id": "missing_valuation_evidence",
                        "severity": "medium",
                        "category": "missing_evidence",
                        "message": "当前缺少估值证据，不能判断价格是否便宜或昂贵。",
                        "evidence_refs": [],
                        "user_visible": True,
                    }
                ],
                "missing_evidence_flags": [
                    {
                        "id": "missing_valuation_evidence",
                        "severity": "medium",
                        "category": "missing_evidence",
                        "message": "当前缺少估值证据，不能判断价格是否便宜或昂贵。",
                        "evidence_refs": [],
                        "user_visible": True,
                    }
                ],
                "allowed_claims": ["valuation evidence is unavailable"],
                "forbidden_claims": ["cheap or expensive conclusion without valuation data"],
            },
            "output": {
                "task_type": "fact_qa",
                "answer_mode": "direct_fact",
                "safety_intent": "normal",
                "needs_tools": True,
                "title": "估值边界",
                "summary": "limited",
                "key_points": [],
                "numeric_evidence": [],
                "text_evidence": [],
                "limitations": [],
                "used_tools": [],
                "trace_id": "test-methodology-trace",
                "view": {"kind": "fact_qa"},
            },
            "selected_tools": [],
            "requested_metrics": [],
            "tool_results": [],
            "citations": [],
            "evidence_loop_count": 1,
            "final_answer": "limited",
        }
        trace_path = traces_dir / "test-methodology-trace.json"
        trace_path.write_text(json.dumps(trace_data))

        try:
            resp = _request("GET", "/trace/test-methodology-trace")
            assert resp.status_code == 200
            body = resp.json()
            assert body["selected_framework"] == "fundamental_quality_analysis"
            assert body["active_dimensions"] == ["revenue_quality", "valuation_and_risk_boundary"]
            assert body["dimension_status_by_id"] == body["dimension_status_map"]
            assert body["covered_dimensions"] == body["satisfied_dimensions"]
            assert body["satisfied_dimensions"] == ["revenue_quality"]
            assert body["partial_dimensions"] == []
            assert body["missing_dimensions"] == ["valuation_and_risk_boundary"]
            assert body["dimension_status_map"]["valuation_and_risk_boundary"]["status"] == "missing"
            assert body["framework_sufficiency_status"] == "partial"
            assert body["red_flags"][0]["id"] == "missing_valuation_evidence"
            assert body["missing_evidence_flags"][0]["message"].startswith("当前缺少估值证据")
            assert "valuation evidence is unavailable" in body["allowed_claims"]
            assert "cheap or expensive conclusion without valuation data" in body["forbidden_claims"]
        finally:
            trace_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E2E (requires vLLM + data)
# ---------------------------------------------------------------------------

def _vllm_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8000), timeout=0.3):
            return True
    except OSError:
        return False


_VLLM_UP = _vllm_up()

_DB_EXISTS = Path(ROOT / "data" / "db" / "financial.duckdb").exists()


@pytest.mark.skipif(not _VLLM_UP, reason="vLLM server not running")
@pytest.mark.skipif(not _DB_EXISTS, reason="DuckDB not populated")
@pytest.mark.e2e
@pytest.mark.requires_llm
class TestChatE2E:

    def test_chat_structured(self):
        resp = _request("POST", "/chat", json={"query": "Apple 最近一个季度的 EPS 是多少？"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"]
        assert body["trace_id"]
        assert body["task_type"]
        assert isinstance(body["citations"], list)
        assert isinstance(body["used_tools"], list)
        assert isinstance(body.get("output", {}), dict)

    def test_chat_then_trace(self):
        resp = _request("POST", "/chat", json={"query": "Microsoft 最近的营收趋势"})
        assert resp.status_code == 200
        trace_id = resp.json()["trace_id"]

        trace_resp = _request("GET", f"/trace/{trace_id}")
        assert trace_resp.status_code == 200
        trace_body = trace_resp.json()
        assert trace_body["trace_id"] == trace_id
        assert trace_body["user_query"] == "Microsoft 最近的营收趋势"
        assert len(trace_body["tool_results"]) > 0
