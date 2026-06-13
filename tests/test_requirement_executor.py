"""Tests for requirement-based tool execution."""

from __future__ import annotations

from typing import Any

from src.agent.requirement_executor import execute_evidence_requirements
from src.agent.evidence import _build_numeric_evidence
from src.tools.protocol import ToolError, ToolResult


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


class MetricFinancialTool:
    def __init__(self, rows: dict[str, dict[str, Any]]):
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        metric = str((payload.get("metrics") or [""])[0])
        row = dict(self.rows[metric])
        row["ticker"] = str(payload.get("ticker", ""))
        row["metric"] = metric
        return {"financial_facts": [row], "price_data": []}


class MultiMetricFinancialTool:
    def __init__(self, rows: dict[str, list[dict[str, Any]]]):
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        facts: list[dict[str, Any]] = []
        for metric in payload.get("metrics", []) or []:
            for raw in self.rows.get(str(metric), []):
                row = dict(raw)
                row["ticker"] = str(payload.get("ticker", ""))
                row["metric"] = str(metric)
                facts.append(row)
        return {"financial_facts": facts, "price_data": []}


def _plan(requirements: list[dict[str, Any]], **overrides):
    base = {
        "user_query": "Analyze AAPL",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "safety_intent": "normal",
        "analysis_goal": "Analyze AAPL",
        "evidence_requirements": requirements,
        "sufficiency_criteria": {},
        "expected_synthesis_style": "analytical_brief",
        "rejected_requirements": [],
    }
    base.update(overrides)
    return base


def _context():
    return {
        "period_query": {"period_type": "latest"},
        "resolved_period_context": {},
        "retrieval_policy": {"text_top_k": 1},
        "event_query": {"event_type": "10Q", "latest_n": 1, "window_days": [1]},
        "tickers": ["AAPL"],
        "user_query": "Analyze AAPL risks",
    }


def test_numeric_requirement_binds_provenance_fields(monkeypatch):
    financial_tool = FakeTool(
        {
            "financial_facts": [
                {
                    "ticker": "AAPL",
                    "metric": "revenue",
                    "period_type": "annual",
                    "period_end": "2025-12-31",
                    "value": 100,
                    "unit": "USD",
                    "source_provider": "sec_companyfacts",
                    "source_url": "https://sec.example",
                    "source_filing_id": "0000320193-25-000001",
                    "confidence": "high",
                    "extraction_method": "xbrl_companyfacts",
                    "source_tag": "Revenues",
                    "reconciliation_warning": "",
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
        "period_type": "annual",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    item = out["evidence_collection_results"][0]["items"][0]
    assert item["requirement_id"] == "REQ-NUM-AAPL-FACT"
    assert item["company"] == "AAPL"
    assert item["metric"] == "revenue"
    assert item["period"] == "2025-12-31"
    assert item["value"] == 100
    assert item["unit"] == "USD"
    assert item["source_provider"] == "sec_companyfacts"
    assert item["source_filing_id"] == "0000320193-25-000001"


def test_calculation_requirement_missing_when_numeric_dependency_missing(monkeypatch):
    compute_tool = FakeTool({"results": []})
    monkeypatch.setattr("src.agent.requirement_executor.compute_metrics", compute_tool)
    req = {
        "requirement_id": "REQ-CALC-AAPL-GROWTH",
        "requirement_type": "calculation",
        "company": "AAPL",
        "metrics": ["revenue"],
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    result = out["evidence_collection_results"][0]
    assert compute_tool.calls == []
    assert result["status"] == "missing"
    assert result["failure_reason"] == "dependency_numeric_requirement_missing"


def test_comparison_net_margin_calculation(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "revenue": {
                "period_type": "annual",
                "period_end": "2025-12-31",
                "value": 200.0,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
            },
            "net_income": {
                "period_type": "annual",
                "period_end": "2025-12-31",
                "value": 50.0,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
            },
        }
    )
    compute_tool = FakeTool({"results": [{"period": "2025-12-31", "margin": 0.25, "margin_pct": "25.00%"}]})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.requirement_executor.compute_metrics", compute_tool)
    reqs = [
        {
            "requirement_id": "REQ-NUM-AAPL-REVENUE",
            "requirement_type": "numeric",
            "company": "AAPL",
            "metrics": ["revenue"],
            "required": True,
            "min_results": 1,
        },
        {
            "requirement_id": "REQ-NUM-AAPL-NET_INCOME",
            "requirement_type": "numeric",
            "company": "AAPL",
            "metrics": ["net_income"],
            "required": True,
            "min_results": 1,
        },
        {
            "requirement_id": "REQ-CALC-AAPL-OPERATING_MARGIN",
            "requirement_type": "calculation",
            "company": "AAPL",
            "metrics": ["operating_margin"],
            "required": False,
            "min_results": 1,
        },
    ]

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "comparison_brief", "task_type": "company_comparison"},
        _plan(reqs, task_type="company_comparison", answer_mode="comparison_brief", safety_intent="investment_advice_like"),
        _context(),
    )

    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["status"] == "satisfied"
    assert calc_result["items"][0]["metric"] == "net_margin"
    assert calc_result["items"][0]["source_tag"] == "net_income_over_revenue"
    assert compute_tool.calls[0]["computation"] == "margin"
    assert compute_tool.calls[0]["data"] == [{"period": "2025-12-31", "value": 50.0}]
    assert compute_tool.calls[0]["denominator_data"] == [{"period": "2025-12-31", "value": 200.0}]
    assert out["tool_results"][-1]["metric"] == "net_margin"


def test_methodology_net_margin_calculation_outside_comparison(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "revenue": {"period_type": "annual", "period_end": "2025-12-31", "value": 200.0, "unit": "USD"},
            "net_income": {"period_type": "annual", "period_end": "2025-12-31", "value": 40.0, "unit": "USD"},
        }
    )
    compute_tool = FakeTool({"results": [{"period": "2025-12-31", "margin": 0.2, "margin_pct": "20.00%"}]})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.requirement_executor.compute_metrics", compute_tool)
    reqs = [
        {"requirement_id": "REQ-NUM-AAPL-REV", "requirement_type": "numeric", "company": "AAPL", "metrics": ["revenue"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-NI", "requirement_type": "numeric", "company": "AAPL", "metrics": ["net_income"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-NET_MARGIN", "requirement_type": "calculation", "company": "AAPL", "metrics": ["net_margin"], "required": True, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary"}, _plan(reqs), _context())

    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["status"] == "satisfied"
    assert calc_result["items"][0]["metric"] == "net_margin"


def test_methodology_cfo_to_net_income_calculation(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "operating_cash_flow": {"period_type": "annual", "period_end": "2025-12-31", "value": 60.0, "unit": "USD"},
            "net_income": {"period_type": "annual", "period_end": "2025-12-31", "value": 40.0, "unit": "USD"},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    reqs = [
        {"requirement_id": "REQ-NUM-AAPL-CFO", "requirement_type": "numeric", "company": "AAPL", "metrics": ["operating_cash_flow"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-NI", "requirement_type": "numeric", "company": "AAPL", "metrics": ["net_income"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-CFO_TO_NET_INCOME", "requirement_type": "calculation", "company": "AAPL", "metrics": ["cfo_to_net_income"], "required": False, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary"}, _plan(reqs), _context())

    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["status"] == "satisfied"
    assert calc_result["items"][0]["metric"] == "cfo_to_net_income"
    assert calc_result["items"][0]["source_tag"] == "operating_cash_flow_over_net_income"


def test_methodology_fcf_margin_calculation(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "free_cash_flow": {"period_type": "annual", "period_end": "2025-12-31", "value": 25.0, "unit": "USD"},
            "revenue": {"period_type": "annual", "period_end": "2025-12-31", "value": 100.0, "unit": "USD"},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    reqs = [
        {"requirement_id": "REQ-NUM-AAPL-FCF", "requirement_type": "numeric", "company": "AAPL", "metrics": ["free_cash_flow"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-REV", "requirement_type": "numeric", "company": "AAPL", "metrics": ["revenue"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-FCF_MARGIN", "requirement_type": "calculation", "company": "AAPL", "metrics": ["fcf_margin"], "required": False, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary"}, _plan(reqs), _context())

    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["status"] == "satisfied"
    assert calc_result["items"][0]["metric"] == "fcf_margin"
    assert calc_result["items"][0]["source_tag"] == "free_cash_flow_over_revenue"


def test_methodology_net_debt_calculation(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "total_debt": {"period_type": "annual", "period_end": "2025-12-31", "value": 120.0, "unit": "USD"},
            "cash_and_equivalents": {"period_type": "annual", "period_end": "2025-12-31", "value": 50.0, "unit": "USD"},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    reqs = [
        {"requirement_id": "REQ-NUM-AAPL-DEBT", "requirement_type": "numeric", "company": "AAPL", "metrics": ["total_debt"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-CASH", "requirement_type": "numeric", "company": "AAPL", "metrics": ["cash_and_equivalents"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-NET_DEBT", "requirement_type": "calculation", "company": "AAPL", "metrics": ["net_debt"], "required": True, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary"}, _plan(reqs), _context())

    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["status"] == "satisfied"
    assert calc_result["items"][0]["metric"] == "net_debt"
    assert out["tool_results"][-1]["data"]["results"][0]["value"] == 70.0


def test_methodology_free_cash_flow_normalizes_capex_outflow(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "operating_cash_flow": {"period_type": "annual", "period_end": "2025-12-31", "value": 100.0, "unit": "USD"},
            "capital_expenditure": {"period_type": "annual", "period_end": "2025-12-31", "value": -30.0, "unit": "USD"},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    reqs = [
        {"requirement_id": "REQ-NUM-AAPL-CFO", "requirement_type": "numeric", "company": "AAPL", "metrics": ["operating_cash_flow"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-CAPEX", "requirement_type": "numeric", "company": "AAPL", "metrics": ["capital_expenditure"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-FCF", "requirement_type": "calculation", "company": "AAPL", "metrics": ["free_cash_flow"], "required": False, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary"}, _plan(reqs), _context())

    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["status"] == "satisfied"
    assert calc_result["items"][0]["metric"] == "free_cash_flow"
    assert out["tool_results"][-1]["data"]["results"][0]["value"] == 70.0


def test_methodology_valuation_multiples_use_market_cap_dependencies(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "price": {"period_type": "daily", "period_end": "2026-04-02", "value": 100.0, "unit": "USD"},
            "shares_outstanding": {"period_type": "latest", "period_end": "2026-04-02", "value": 10.0, "unit": "shares"},
            "net_income": {"period_type": "annual", "period_end": "2025-12-31", "value": 50.0, "unit": "USD"},
            "revenue": {"period_type": "annual", "period_end": "2025-12-31", "value": 200.0, "unit": "USD"},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    reqs = [
        {"requirement_id": "REQ-NUM-AAPL-PRICE", "requirement_type": "numeric", "company": "AAPL", "metrics": ["price"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-SHARES", "requirement_type": "numeric", "company": "AAPL", "metrics": ["shares_outstanding"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-NI", "requirement_type": "numeric", "company": "AAPL", "metrics": ["net_income"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NUM-AAPL-REV", "requirement_type": "numeric", "company": "AAPL", "metrics": ["revenue"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-MCAP", "requirement_type": "calculation", "company": "AAPL", "metrics": ["market_cap"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-PE", "requirement_type": "calculation", "company": "AAPL", "metrics": ["pe_ratio"], "required": False, "min_results": 1},
        {"requirement_id": "REQ-CALC-AAPL-PS", "requirement_type": "calculation", "company": "AAPL", "metrics": ["ps_ratio"], "required": False, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary"}, _plan(reqs), _context())

    mcap, pe, ps = out["evidence_collection_results"][-3:]
    price_result, shares_result = out["evidence_collection_results"][:2]
    assert price_result["items"][0]["source_requirement_id"] == "REQ-NUM-AAPL-PRICE"
    assert shares_result["items"][0]["source_requirement_id"] == "REQ-NUM-AAPL-SHARES"
    assert mcap["status"] == pe["status"] == ps["status"] == "satisfied"
    assert mcap["items"][0]["metric"] == "market_cap"
    assert mcap["items"][0]["data"]["results"][0]["value"] == 1000.0
    assert pe["items"][0]["data"]["results"][0]["value"] == 20.0
    assert ps["items"][0]["data"]["results"][0]["value"] == 5.0


def test_methodology_revenue_growth_requires_two_periods(monkeypatch):
    financial_tool = MultiMetricFinancialTool(
        {
            "revenue": [
                {"period_type": "annual", "period_end": "2024-12-31", "value": 100.0, "unit": "USD"},
                {"period_type": "annual", "period_end": "2025-12-31", "value": 125.0, "unit": "USD"},
            ]
        }
    )
    compute_tool = FakeTool({"results": [{"period": "2025-12-31", "growth": 0.25, "growth_pct": "25.00%"}]})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.requirement_executor.compute_metrics", compute_tool)
    reqs = [
        {"requirement_id": "REQ-NUM-AAPL-REV-TREND", "requirement_type": "numeric", "company": "AAPL", "metrics": ["revenue"], "period_type": "ttm", "required": False, "min_results": 2},
        {"requirement_id": "REQ-CALC-AAPL-REVENUE_GROWTH", "requirement_type": "calculation", "company": "AAPL", "metrics": ["revenue_growth"], "required": False, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary"}, _plan(reqs), _context())

    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["status"] == "satisfied"
    assert calc_result["items"][0]["metric"] == "revenue_growth"


def test_causal_revenue_growth_rejects_same_period_comparator(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "revenue": {
                "period_type": "annual",
                "period_end": "2026-01-31",
                "value": 68_127_000_000,
                "unit": "USD",
                "source_provider": "yfinance",
            }
        }
    )
    compute_tool = FakeTool({"results": [{"period": "2026-01-31", "growth": 0.0, "growth_pct": "0.00%"}]})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.requirement_executor.compute_metrics", compute_tool)
    reqs = [
        {
            "requirement_id": "REQ-RP-NVDA-CURRENT_REVENUE_NUM",
            "requirement_type": "numeric",
            "company": "NVDA",
            "metrics": ["revenue"],
            "required": True,
            "min_results": 1,
            "answer_part_ids": ["quantify_growth"],
            "evidence_role": "current_revenue",
        },
        {
            "requirement_id": "REQ-RP-NVDA-COMPARATOR_REVENUE_NUM",
            "requirement_type": "numeric",
            "company": "NVDA",
            "metrics": ["revenue"],
            "required": True,
            "min_results": 1,
            "answer_part_ids": ["quantify_growth"],
            "evidence_role": "comparator_revenue",
        },
        {
            "requirement_id": "REQ-RP-NVDA-REVENUE_GROWTH_CALCULATION_REVENUE_GROWTH",
            "requirement_type": "calculation",
            "company": "NVDA",
            "metrics": ["revenue_growth"],
            "required": True,
            "min_results": 1,
            "answer_part_ids": ["quantify_growth"],
            "evidence_role": "revenue_growth_calculation",
        },
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan(reqs), _context())

    by_req = {item["requirement_id"]: item for item in out["evidence_collection_results"]}
    assert by_req["REQ-RP-NVDA-COMPARATOR_REVENUE_NUM"]["status"] == "missing"
    assert by_req["REQ-RP-NVDA-COMPARATOR_REVENUE_NUM"]["failure_reason"] == "same_period_comparator"
    assert by_req["REQ-RP-NVDA-REVENUE_GROWTH_CALCULATION_REVENUE_GROWTH"]["status"] == "missing"
    assert by_req["REQ-RP-NVDA-REVENUE_GROWTH_CALCULATION_REVENUE_GROWTH"]["failure_reason"] == "same_period_comparator"
    assert compute_tool.calls == []
    numeric = _build_numeric_evidence(out["tool_results"])
    assert [row["metric"] for row in numeric] == ["revenue"]
    assert numeric[0]["source_requirement_id"] == "REQ-RP-NVDA-CURRENT_REVENUE_NUM"


def test_methodology_missing_metric_becomes_missing_not_rejected(monkeypatch):
    financial_tool = FakeTool({"financial_facts": [], "price_data": []})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    req = {
        "requirement_id": "REQ-METH-AAPL-CASH_FLOW_QUALITY-FREE_CASH_FLOW",
        "requirement_type": "numeric",
        "company": "AAPL",
        "metrics": ["free_cash_flow"],
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    result = out["evidence_collection_results"][0]
    assert result["status"] == "missing"
    assert result["failure_reason"] == "no_matching_evidence"


def test_comparison_no_operating_margin_does_not_fail(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "revenue": {
                "period_type": "annual",
                "period_end": "2025-12-31",
                "value": 200.0,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
            },
            "net_income": {
                "period_type": "annual",
                "period_end": "2025-12-31",
                "value": 50.0,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
            },
        }
    )
    compute_tool = FakeTool({"results": [{"period": "2025-12-31", "margin": 0.25, "margin_pct": "25.00%"}]})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.requirement_executor.compute_metrics", compute_tool)
    reqs = [
        {
            "requirement_id": "REQ-NUM-AAPL-REVENUE",
            "requirement_type": "numeric",
            "company": "AAPL",
            "metrics": ["revenue"],
            "required": True,
            "min_results": 1,
        },
        {
            "requirement_id": "REQ-NUM-AAPL-NET_INCOME",
            "requirement_type": "numeric",
            "company": "AAPL",
            "metrics": ["net_income"],
            "required": True,
            "min_results": 1,
        },
        {
            "requirement_id": "REQ-CALC-AAPL-OPERATING_MARGIN",
            "requirement_type": "calculation",
            "company": "AAPL",
            "metrics": ["operating_margin"],
            "required": False,
            "min_results": 1,
        },
    ]

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "comparison_brief", "task_type": "company_comparison"},
        _plan(reqs, task_type="company_comparison", answer_mode="comparison_brief", safety_intent="investment_advice_like"),
        _context(),
    )

    assert "operating_margin" not in financial_tool.rows
    calc_result = out["evidence_collection_results"][-1]
    assert calc_result["requirement_id"] == "REQ-CALC-AAPL-OPERATING_MARGIN"
    assert calc_result["status"] == "satisfied"
    assert calc_result.get("failure_reason") in {None, ""}
    assert calc_result["items"][0]["metric"] == "net_margin"
    assert not any(
        item.get("requirement_id") == "REQ-CALC-AAPL-OPERATING_MARGIN"
        and item.get("code") == "requirement_missing"
        for item in out["requirement_limitations"]
    )


def test_text_requirement_payload_retry_and_snippet_support(monkeypatch):
    search_tool = FakeTool(
        [
            {
                "ticker": "AAPL",
                "filing_id": "f1",
                "section": "ITEM_1A",
                "text": "Risk factors include competition and supply constraints.",
                "supporting_snippet": "Risk factors include competition.",
                "supporting_terms": ["risk", "competition"],
                "score_breakdown": {"final_score": 0.9},
            }
        ]
    )
    monkeypatch.setattr("src.agent.requirement_executor.search_filings", search_tool)
    monkeypatch.setattr(
        "src.agent.requirement_executor.search_filings_lexical_fallback",
        lambda **_kwargs: {"items": [], "diagnostics": {"backend": "duckdb_lexical", "raw_hit_count": 0, "section_filtered_hit_count": 0}},
    )
    req = {
        "requirement_id": "REQ-TEXT-AAPL-RISK",
        "requirement_type": "text",
        "company": "AAPL",
        "section_preferences": ["ITEM_1A"],
        "retrieval_query": "AAPL risk factors competition",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    assert search_tool.calls[0]["ticker"] == "AAPL"
    assert search_tool.calls[0]["query"] == "AAPL risk factors competition"
    assert search_tool.calls[0]["section_allowlist"] == ["ITEM_1A"]
    assert search_tool.calls[0]["strict_sections"] is True
    item = out["evidence_collection_results"][0]["items"][0]
    assert item["requirement_id"] == "REQ-TEXT-AAPL-RISK"
    assert item["filing_id"] == "f1"
    assert item["section"] == "ITEM_1A"
    assert item["snippet"]
    assert item["supporting_terms"] == ["risk", "competition"]
    assert item["score_breakdown"] == {"final_score": 0.9}


def test_text_evidence_drilldown_fields_present(monkeypatch):
    search_tool = FakeTool(
        {
            "items": [
                {
                    "ticker": "AAPL",
                    "filing_id": "f1",
                    "section": "ITEM_1A",
                    "quality": "high",
                    "text": "Risk factors include competition and supply constraints.",
                    "supporting_snippet": "Risk factors include competition and supply constraints.",
                    "supporting_terms": ["competition", "supply"],
                    "score": 0.91,
                }
            ],
            "diagnostics": {
                "raw_hit_count": 3,
                "section_filtered_hit_count": 3,
                "raw_candidates": [
                    {
                        "ticker": "AAPL",
                        "filing_id": "mixed",
                        "section": "ITEM_1A",
                        "quality": "low",
                        "text_snippet": "Mixed filing boilerplate.",
                        "supporting_snippet": "Mixed filing boilerplate.",
                    },
                    {
                        "ticker": "AAPL",
                        "filing_id": "empty",
                        "section": "ITEM_1A",
                        "quality": "high",
                        "text_snippet": "",
                        "supporting_snippet": "",
                    },
                ],
                "top_raw_snippets": [],
            },
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.search_filings", search_tool)
    monkeypatch.setattr(
        "src.agent.requirement_executor.search_filings_lexical_fallback",
        lambda **_kwargs: {"items": [], "diagnostics": {"backend": "duckdb_lexical", "raw_hit_count": 0, "section_filtered_hit_count": 0}},
    )
    req = {
        "requirement_id": "REQ-TEXT-AAPL-RISK",
        "requirement_type": "text",
        "company": "AAPL",
        "section_preferences": ["ITEM_1A"],
        "primary_sections": ["ITEM_1A"],
        "fallback_sections": ["ITEM_7"],
        "broadened_queries": ["AAPL risk factors", "risk factors"],
        "retrieval_query": "AAPL risk factors competition",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    assert search_tool.calls[0]["return_diagnostics"] is True
    result = out["evidence_collection_results"][0]
    assert result["requirement_id"] == "REQ-TEXT-AAPL-RISK"
    assert result["company"] == "AAPL"
    assert result["retrieval_query"] == "AAPL risk factors competition"
    assert result["section_preferences"] == ["ITEM_1A"]
    assert result["fallback_queries"] == ["AAPL risk factors", "risk factors"]
    assert result["fallback_sections"] == ["ITEM_7"]
    assert result["raw_hit_count"] == 3
    assert result["section_filtered_hit_count"] == 3
    assert result["usable_hit_count"] == 1
    assert result["snippet_support_passed_count"] == 1
    assert result["text_claim_validated_count"] == 0
    assert result["text_citation_kept_count"] == 0
    assert result["final_validated_text_count"] == 0
    assert result["drop_stage"] == "satisfied"
    assert result["rejection_reasons"]["quality_filter_dropped"] == 1
    assert result["rejection_reasons"]["snippet_support_failed"] == 1
    assert {item["reason"] for item in result["top_rejected_snippets"]} == {
        "quality_filter_dropped",
        "snippet_support_failed",
    }


def test_comparison_risk_context_text_requirement_uses_methodology_sections(monkeypatch):
    search_tool = FakeTool(
        {
            "items": [
                {
                    "ticker": "AAPL",
                    "filing_id": "aapl-10k",
                    "form_type": "10-K",
                    "fiscal_period": "2025",
                    "section": "ITEM_1A",
                    "quality": "high",
                    "text": "Competition and regulatory risks may affect the business.",
                    "supporting_snippet": "Competition and regulatory risks may affect the business.",
                    "supporting_terms": ["competition", "risks"],
                }
            ],
            "diagnostics": {"raw_hit_count": 1, "section_filtered_hit_count": 1, "raw_candidates": []},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.search_filings", search_tool)
    req = {
        "requirement_id": "REQ-METH-AAPL-MOAT_AND_COMPETITIVE_RISK-TEXT",
        "requirement_type": "text",
        "company": "AAPL",
        "dimension_id": "moat_and_competitive_risk",
        "section_preferences": ["ITEM_1A", "ITEM_1", "BUSINESS"],
        "primary_sections": ["ITEM_1A", "ITEM_1", "BUSINESS"],
        "fallback_sections": ["ITEM_7", "MD&A"],
        "retrieval_query": "AAPL competition risk factors",
        "retrieval_intent": "comparison_risk_context",
        "retrieval_profile": "risk_summary",
        "broadened_queries": [
            "AAPL competition risk factors",
            "AAPL business risks competitive pressure",
            "AAPL risk factors competitive pressure",
        ],
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "comparison_brief", "task_type": "company_comparison"},
        _plan([req], task_type="company_comparison", answer_mode="comparison_brief"),
        _context(),
    )

    assert search_tool.calls[0]["query"] == "AAPL competition risk factors"
    assert search_tool.calls[0]["section_allowlist"] == ["ITEM_1A", "ITEM_1", "BUSINESS"]
    assert search_tool.calls[0]["strict_sections"] is True
    assert search_tool.calls[0]["retrieval_profile"] == "risk_summary"
    result = out["evidence_collection_results"][0]
    assert result["status"] == "satisfied"
    assert result["retrieval_query"] == "AAPL competition risk factors"
    assert result["section_preferences"] == ["ITEM_1A", "ITEM_1", "BUSINESS"]
    assert result["fallback_sections"] == ["ITEM_7", "MD&A"]
    assert result["fallback_queries"] == [
        "AAPL competition risk factors",
        "AAPL business risks competitive pressure",
        "AAPL risk factors competitive pressure",
    ]
    assert result["drop_stage"] == "satisfied"


def test_single_company_risk_context_preserves_dimension_metadata(monkeypatch):
    search_tool = FakeTool(
        {
            "items": [
                {
                    "ticker": "NVDA",
                    "filing_id": "nvda-10k",
                    "form_type": "10-K",
                    "fiscal_period": "2025",
                    "section": "ITEM_1A",
                    "quality": "high",
                    "text": "Competition, demand, and supply chain risks may affect the business.",
                    "supporting_snippet": "Competition, demand, and supply chain risks may affect the business.",
                    "supporting_terms": ["competition", "demand", "supply"],
                }
            ],
            "diagnostics": {"raw_hit_count": 1, "section_filtered_hit_count": 1, "raw_candidates": []},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.search_filings", search_tool)
    monkeypatch.setattr(
        "src.agent.requirement_executor.search_filings_lexical_fallback",
        lambda **_kwargs: {"items": [], "diagnostics": {"backend": "duckdb_lexical", "raw_hit_count": 0, "section_filtered_hit_count": 0}},
    )
    req = {
        "requirement_id": "REQ-TEXT-NVDA-RISK",
        "requirement_type": "text",
        "company": "NVDA",
        "framework_id": "fundamental_quality_analysis",
        "dimension_id": "moat_and_competitive_risk",
        "section_preferences": ["ITEM_1A"],
        "primary_sections": ["ITEM_1A"],
        "fallback_sections": ["ITEM_7", "MD&A", "ITEM_1", "BUSINESS"],
        "retrieval_query": "NVDA risk factors competition demand supply chain regulation customer concentration",
        "retrieval_intent": "single_company_risk_context",
        "retrieval_profile": "risk_summary",
        "broadened_queries": ["NVDA competition risks", "NVDA demand supply chain risks"],
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary", "analysis_scope": "single_company"},
        _plan([req], task_type="report_summary", answer_mode="analytical"),
        _context(),
    )

    assert search_tool.calls[0]["retrieval_profile"] == "risk_summary"
    assert search_tool.calls[0]["section_allowlist"] == ["ITEM_1A"]
    result = out["evidence_collection_results"][0]
    assert result["status"] == "satisfied"
    assert result["items"][0]["dimension_id"] == "moat_and_competitive_risk"
    assert result["items"][0]["retrieval_intent"] == "single_company_risk_context"


def test_single_company_risk_context_uses_lexical_first_when_available(monkeypatch):
    vector_calls: list[dict[str, Any]] = []
    fallback_calls: list[dict[str, Any]] = []

    def vector_tool(tool_name, payload, context, req):
        vector_calls.append(dict(payload))
        raise AssertionError("vector search should not run when lexical-first satisfies the requirement")

    def lexical_fallback(**kwargs):
        fallback_calls.append(dict(kwargs))
        return {
            "items": [
                {
                    "ticker": "NVDA",
                    "filing_id": "nvda-10k",
                    "form_type": "10-K",
                    "fiscal_period": "2026",
                    "section": "ITEM_1A",
                    "quality": "high",
                    "text": "Competition, demand, and supply chain risks may affect the business.",
                    "supporting_snippet": "Competition, demand, and supply chain risks may affect the business.",
                    "supporting_terms": ["competition", "demand", "supply"],
                    "score_breakdown": {"backend": "duckdb_lexical", "final_score": 0.9},
                }
            ],
            "diagnostics": {
                "backend": "duckdb_lexical",
                "raw_hit_count": 1,
                "section_filtered_hit_count": 1,
                "raw_candidates": [],
            },
        }

    monkeypatch.setattr("src.agent.requirement_executor._run_protocol_tool", vector_tool)
    monkeypatch.setattr("src.agent.requirement_executor.search_filings_lexical_fallback", lexical_fallback)
    req = {
        "requirement_id": "REQ-TEXT-NVDA-RISK",
        "requirement_type": "text",
        "company": "NVDA",
        "framework_id": "fundamental_quality_analysis",
        "dimension_id": "moat_and_competitive_risk",
        "section_preferences": ["ITEM_1A"],
        "primary_sections": ["ITEM_1A"],
        "fallback_sections": ["ITEM_7", "MD&A", "ITEM_1", "BUSINESS"],
        "retrieval_query": "NVDA risk factors competition demand supply chain regulation customer concentration",
        "retrieval_intent": "single_company_risk_context",
        "retrieval_profile": "risk_summary",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary", "analysis_scope": "single_company"},
        _plan([req], task_type="report_summary", answer_mode="analytical"),
        _context(),
    )

    result = out["evidence_collection_results"][0]
    assert vector_calls == []
    assert len(fallback_calls) == 1
    assert fallback_calls[0]["section_allowlist"] == ["ITEM_1A"]
    assert result["status"] == "satisfied"
    assert out["tool_results"][0]["lexical_first"] is True
    assert out["tool_call_results"][0]["input_summary"]["backend"] == "duckdb_lexical"


def test_text_requirement_timeout_uses_lexical_fallback_once(monkeypatch):
    vector_calls: list[dict[str, Any]] = []
    fallback_calls: list[dict[str, Any]] = []

    def timeout_once(tool_name, payload, context, req):
        vector_calls.append(dict(payload))
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            error=ToolError(code="timeout", message="Tool execution timed out", retryable=True),
        )

    def lexical_fallback(**kwargs):
        fallback_calls.append(dict(kwargs))
        return {
            "items": [
                {
                    "ticker": "NVDA",
                    "filing_id": "nvda-10k",
                    "form_type": "10-K",
                    "fiscal_period": "2026",
                    "section": "ITEM_1A",
                    "quality": "high",
                    "text": "Risk factors include competition and supply chain constraints.",
                    "supporting_snippet": "Risk factors include competition and supply chain constraints.",
                    "supporting_terms": ["competition", "supply", "chain"],
                    "score_breakdown": {"backend": "duckdb_lexical", "final_score": 0.9},
                }
            ],
            "diagnostics": {
                "backend": "duckdb_lexical",
                "raw_hit_count": 1,
                "section_filtered_hit_count": 1,
                "raw_candidates": [],
            },
        }

    monkeypatch.setattr("src.agent.requirement_executor._run_protocol_tool", timeout_once)
    monkeypatch.setattr("src.agent.requirement_executor.search_filings_lexical_fallback", lexical_fallback)
    req = {
        "requirement_id": "REQ-TEXT-NVDA-RISK",
        "requirement_type": "text",
        "company": "NVDA",
        "dimension_id": "moat_and_competitive_risk",
        "section_preferences": ["ITEM_1A"],
        "primary_sections": ["ITEM_1A"],
        "fallback_sections": ["ITEM_1", "ITEM_7"],
        "retrieval_query": "NVDA risk factors competition supply chain",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "analytical", "task_type": "report_summary", "analysis_scope": "single_company"},
        _plan([req], task_type="report_summary", answer_mode="analytical"),
        _context(),
    )

    result = out["evidence_collection_results"][0]
    assert len(vector_calls) == 1
    assert len(fallback_calls) == 1
    assert fallback_calls[0]["section_allowlist"] == ["ITEM_1A"]
    assert result["status"] == "satisfied"
    assert result["retry_count"] == 0
    assert result["items"][0]["filing_id"] == "nvda-10k"
    assert out["tool_results"][0]["fallback_after_timeout"] is True
    assert out["tool_results"][0]["backend"] == "duckdb_lexical"


def test_text_requirement_timeout_empty_fallback_does_not_retry_vector(monkeypatch):
    vector_calls: list[dict[str, Any]] = []

    def timeout_once(tool_name, payload, context, req):
        vector_calls.append(dict(payload))
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            error=ToolError(code="timeout", message="Tool execution timed out", retryable=True),
        )

    def empty_fallback(**_kwargs):
        return {
            "items": [],
            "diagnostics": {
                "backend": "duckdb_lexical",
                "raw_hit_count": 0,
                "section_filtered_hit_count": 0,
                "raw_candidates": [],
            },
        }

    monkeypatch.setattr("src.agent.requirement_executor._run_protocol_tool", timeout_once)
    monkeypatch.setattr("src.agent.requirement_executor.search_filings_lexical_fallback", empty_fallback)
    req = {
        "requirement_id": "REQ-TEXT-NVDA-RISK",
        "requirement_type": "text",
        "company": "NVDA",
        "section_preferences": ["ITEM_1A"],
        "primary_sections": ["ITEM_1A"],
        "broadened_queries": ["NVDA risks", "risk factors"],
        "retrieval_query": "NVDA risk factors competition supply chain",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    result = out["evidence_collection_results"][0]
    assert len(vector_calls) == 1
    assert result["status"] == "missing"
    assert result["retry_count"] == 0
    assert result["failure_reason"] == "search_filings_timeout_fallback_empty"


def test_text_requirement_oom_uses_lexical_fallback_once(monkeypatch):
    vector_calls: list[dict[str, Any]] = []
    fallback_calls: list[dict[str, Any]] = []

    def oom_once(tool_name, payload, context, req):
        vector_calls.append(dict(payload))
        return ToolResult(
            ok=False,
            tool_name=tool_name,
            error=ToolError(code="execution_error", message="CUDA out of memory. Tried to allocate 96.00 MiB.", retryable=False),
        )

    def lexical_fallback(**kwargs):
        fallback_calls.append(dict(kwargs))
        return {
            "items": [
                {
                    "ticker": "AMZN",
                    "filing_id": "amzn-10q",
                    "form_type": "10-Q",
                    "section": "ITEM_2",
                    "quality": "high",
                    "text": "Management discussion includes operating challenges and demand uncertainty.",
                    "supporting_snippet": "Management discussion includes operating challenges and demand uncertainty.",
                    "supporting_terms": ["management", "discussion", "operating", "demand"],
                    "score_breakdown": {"backend": "duckdb_lexical", "final_score": 0.9},
                }
            ],
            "diagnostics": {
                "backend": "duckdb_lexical",
                "raw_hit_count": 1,
                "section_filtered_hit_count": 1,
                "raw_candidates": [],
            },
        }

    monkeypatch.setattr("src.agent.requirement_executor._run_protocol_tool", oom_once)
    monkeypatch.setattr("src.agent.requirement_executor.search_filings_lexical_fallback", lexical_fallback)
    req = {
        "requirement_id": "REQ-TEXT-AMZN-RISK-MDA",
        "requirement_type": "text",
        "company": "AMZN",
        "dimension_id": "moat_and_competitive_risk",
        "section_preferences": ["ITEM_2"],
        "primary_sections": ["ITEM_2"],
        "fallback_sections": ["ITEM_1A"],
        "retrieval_query": "AMZN management discussion operating challenges demand risk",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "risk_focused_analysis", "task_type": "report_summary", "analysis_scope": "single_company"},
        _plan([req], task_type="report_summary", answer_mode="risk_focused_analysis"),
        _context(),
    )

    result = out["evidence_collection_results"][0]
    assert len(vector_calls) == 1
    assert len(fallback_calls) == 1
    assert result["status"] == "satisfied"
    assert result["retry_count"] == 0
    assert result["items"][0]["filing_id"] == "amzn-10q"
    assert out["tool_results"][0]["fallback_after_error"] is True
    assert out["tool_results"][0]["fallback_error_code"] == "execution_error"
    assert out["tool_results"][0]["backend"] == "duckdb_lexical"


def test_valuation_missing_check_returns_missing_without_tool_calls(monkeypatch):
    financial_tool = FakeTool({"financial_facts": []})
    compute_tool = FakeTool({"results": []})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    monkeypatch.setattr("src.agent.requirement_executor.compute_metrics", compute_tool)
    req = {
        "requirement_id": "REQ-METH-VALUATION_AND_RISK_BOUNDARY_VALUATION_EVIDENCE_MISSING",
        "requirement_type": "calculation",
        "company": None,
        "dimension_id": "valuation_and_risk_boundary",
        "metric": "price",
        "metrics": ["price"],
        "required": True,
        "min_results": 1,
        "fallback_strategy": ["valuation_evidence_missing"],
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "comparison_brief", "task_type": "company_comparison"},
        _plan([req], task_type="company_comparison", answer_mode="comparison_brief"),
        _context(),
    )

    assert financial_tool.calls == []
    assert compute_tool.calls == []
    result = out["evidence_collection_results"][0]
    assert result["status"] == "missing"
    assert result["failure_reason"] == "valuation_evidence_missing"
    assert result["evidence_type"] == "calculation"
    assert result["items"] == []


def test_text_requirement_retries_twice_and_rejects_mixed_chunk(monkeypatch):
    search_tool = FakeTool([{"ticker": "AAPL", "section": "MIXED", "text": "Risk factors."}])
    monkeypatch.setattr("src.agent.requirement_executor.search_filings", search_tool)
    req = {
        "requirement_id": "REQ-TEXT-AAPL-RISK",
        "requirement_type": "text",
        "company": "AAPL",
        "section_preferences": ["ITEM_1A"],
        "retrieval_query": "AAPL risk factors",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    result = out["evidence_collection_results"][0]
    assert len(search_tool.calls) == 3
    assert result["status"] == "missing"
    assert result["retry_count"] == 2
    assert out["retry_history"][1]["strategy"] == "relaxed_sections_intent_query"
    assert out["retry_history"][2]["strategy"] == "generic_query"
    assert result["drop_stage"] == "quality_filter_dropped"


def test_text_requirement_falls_back_to_snippet_terms_when_query_overlap_missing(monkeypatch):
    search_tool = FakeTool(
        [
            {
                "ticker": "AAPL",
                "filing_id": "f1",
                "section": "ITEM_1A",
                "text": "Competition and regulation are material headwinds for the business.",
                "supporting_snippet": "Competition and regulation are material headwinds for the business.",
                "supporting_terms": [],
                "score_breakdown": {"final_score": 0.8},
            }
        ]
    )
    monkeypatch.setattr("src.agent.requirement_executor.search_filings", search_tool)
    req = {
        "requirement_id": "REQ-TEXT-AAPL-PROBLEM",
        "requirement_type": "text",
        "company": "AAPL",
        "section_preferences": ["ITEM_1A"],
        "primary_sections": ["ITEM_1A"],
        "fallback_sections": ["ITEM_7"],
        "retrieval_query": "AAPL completely unrelated prompt",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    item = out["evidence_collection_results"][0]["items"][0]
    assert item["supporting_terms"]
    assert "competition" in item["supporting_terms"]


def test_event_requirement_binds_requirement_id(monkeypatch):
    event_tool = FakeTool(
        {
            "events": [
                {"ticker": "AAPL", "event_type": "10Q", "event_date": "2025-01-31", "returns": {"return_1d": 0.01}}
            ]
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_event_price_window", event_tool)
    req = {
        "requirement_id": "REQ-EVENT-AAPL-FILING",
        "requirement_type": "event",
        "company": "AAPL",
        "required": True,
        "min_results": 1,
    }

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan([req]), _context())

    event = out["evidence_collection_results"][0]["items"][0]
    assert event["requirement_id"] == "REQ-EVENT-AAPL-FILING"
    assert out["tool_results"][0]["data"]["events"][0]["requirement_id"] == "REQ-EVENT-AAPL-FILING"


def test_tool_exception_records_failure_reason(monkeypatch):
    financial_tool = FakeTool(error=RuntimeError("db unavailable"))
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

    result = out["evidence_collection_results"][0]
    assert result["status"] == "missing"
    assert result["failure_reason"] == "query_financial_data_error:db unavailable"
    assert out["tool_results"][0]["error"] == "db unavailable"


def test_meta_clarification_and_refusal_skip_tools(monkeypatch):
    financial_tool = FakeTool({"financial_facts": []})
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    req = {"requirement_id": "REQ-NUM-AAPL-FACT", "requirement_type": "numeric", "company": "AAPL", "required": True}

    for mode in ("meta", "clarification", "refusal_or_redirect"):
        out = execute_evidence_requirements({"needs_tools": True, "answer_mode": mode}, _plan([req], answer_mode=mode), _context())
        assert out["tool_results"] == []
        assert out["why_tools_skipped"]
    assert financial_tool.calls == []
