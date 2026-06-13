"""Tests for ToolSpec/ToolResult registry integration."""

from __future__ import annotations

from typing import Any

import pytest

from src.agent.requirement_executor import execute_evidence_requirements
from src.tools.adapters.query_financial_data_tool import QueryFinancialDataTool
from src.tools.protocol import ToolError, ToolExecutionContext, ToolResult
from src.tools.registry import ToolRegistry, UnknownToolError, build_default_tool_registry


class FakeTool:
    def __init__(self, response: Any = None, *, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


def _plan(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "evidence_requirements": requirements,
    }


def _context() -> dict[str, Any]:
    return {"period_query": {"period_type": "latest"}, "resolved_period_context": {}, "retrieval_policy": {}}


def test_tool_result_ok_error_contract():
    with pytest.raises(ValueError):
        ToolResult(ok=False, tool_name="x")
    with pytest.raises(ValueError):
        ToolResult(ok=True, tool_name="x", error=ToolError(code="x", message="bad"))


def test_registry_duplicate_and_unknown_tool():
    registry = ToolRegistry()
    registry.register(QueryFinancialDataTool(FakeTool({"financial_facts": [], "price_data": []})))
    with pytest.raises(ValueError):
        registry.register(QueryFinancialDataTool(FakeTool({"financial_facts": [], "price_data": []})))
    with pytest.raises(UnknownToolError):
        registry.get("missing")


def test_default_registry_exposes_four_core_specs():
    registry = build_default_tool_registry(
        {
            "query_financial_data": FakeTool({"financial_facts": [], "price_data": []}),
            "search_filings": FakeTool([]),
            "compute_metrics": FakeTool({"results": []}),
            "query_event_price_window": FakeTool({"events": []}),
        }
    )

    names = {spec.name for spec in registry.list_specs()}
    assert names == {"query_financial_data", "search_filings", "compute_metrics", "query_event_price_window"}
    definitions = registry.to_function_definitions()
    assert {item["name"] for item in definitions} == names


def test_profit_decline_direct_fact_uses_requirement_executor_path():
    from src.agent.tool_executor import _should_use_requirement_executor

    assert _should_use_requirement_executor(
        {
            "needs_tools": True,
            "answer_mode": "direct_fact",
            "user_query": "为什么amazon的利润下降了",
        }
    )
    assert not _should_use_requirement_executor(
        {
            "needs_tools": True,
            "answer_mode": "direct_fact",
            "user_query": "AMZN revenue",
        }
    )


def test_adapter_returns_tool_result_with_provenance():
    adapter = QueryFinancialDataTool(
        FakeTool(
            {
                "financial_facts": [
                    {
                        "evidence_id": "N1",
                        "ticker": "AAPL",
                        "metric": "revenue",
                        "period_end": "2025-12-31",
                        "value": 100,
                        "source_provider": "sec_companyfacts",
                    }
                ],
                "price_data": [],
            }
        )
    )

    result = adapter.execute({"ticker": "AAPL", "metrics": ["revenue"]}, ToolExecutionContext(trace_id="t1"))

    assert result.ok is True
    assert result.tool_name == "query_financial_data"
    assert result.returned_count == 1
    assert result.provenance[0].provider == "sec_companyfacts"


def test_requirement_executor_keeps_legacy_results_and_adds_protocol_trace(monkeypatch):
    financial_tool = FakeTool(
        {
            "financial_facts": [
                {
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "period_end": "2025-12-31",
                    "value": 100,
                    "unit": "USD",
                    "source_provider": "sec_companyfacts",
                }
            ],
            "price_data": [],
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    req = {
        "requirement_id": "REQ-NUM-AAPL-FACT",
        "requirement_type": "numeric",
        "company": "AAPL",
        "metrics": ["revenue"],
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    assert out["tool_results"][0]["tool"] == "query_financial_data"
    assert out["evidence_collection_results"][0]["status"] == "satisfied"
    assert out["tool_call_results"][0]["tool_name"] == "query_financial_data"
    assert out["retrieval_debug"]["tool_call_results"][0]["ok"] is True
