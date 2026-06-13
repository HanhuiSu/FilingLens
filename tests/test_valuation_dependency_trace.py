from __future__ import annotations

from typing import Any

from src.agent.evidence import _build_numeric_evidence
from src.agent.requirement_executor import execute_evidence_requirements


class MetricFinancialTool:
    def __init__(self, rows: dict[str, dict[str, Any]]):
        self.rows = rows

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        metric = str((payload.get("metrics") or [""])[0])
        row = dict(self.rows.get(metric, {}))
        if not row:
            return {"financial_facts": [], "price_data": []}
        row["ticker"] = str(payload.get("ticker", ""))
        row["metric"] = metric
        if metric == "price":
            return {
                "financial_facts": [],
                "price_data": [
                    {
                        "ticker": row["ticker"],
                        "metric": "adjusted_close",
                        "date": row["period_end"],
                        "value": row["value"],
                        "adjusted_close": row["value"],
                        "source_provider": "yfinance",
                        "confidence": "medium",
                    }
                ],
            }
        return {"financial_facts": [row], "price_data": []}


def _plan(requirements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "user_query": "NVDA valuation",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "evidence_requirements": requirements,
        "rejected_requirements": [],
    }


def _context() -> dict[str, Any]:
    return {"period_query": {"period_type": "latest"}, "retrieval_policy": {"text_top_k": 1}}


def test_valuation_computed_evidence_carries_dependency_trace(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "price": {"period_type": "daily", "period_end": "2026-04-24", "value": 100.0, "unit": "USD"},
            "shares_outstanding": {
                "period_type": "latest",
                "period_end": "2026-01-31",
                "value": 10.0,
                "unit": "shares",
                "source_provider": "sec_companyfacts",
                "confidence": "high",
            },
            "net_income": {
                "period_type": "annual",
                "period_end": "2026-01-31",
                "value": 50.0,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
                "confidence": "high",
            },
            "revenue": {
                "period_type": "annual",
                "period_end": "2026-01-31",
                "value": 200.0,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
                "confidence": "high",
                "reconciliation_warning": "value_mismatch_gt_2pct",
            },
            "free_cash_flow": {
                "period_type": "annual",
                "period_end": "2026-01-31",
                "value": 25.0,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
                "confidence": "high",
            },
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    reqs = [
        {"requirement_id": "REQ-PRICE", "requirement_type": "numeric", "company": "NVDA", "metrics": ["price"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-SHARES", "requirement_type": "numeric", "company": "NVDA", "metrics": ["shares_outstanding"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NI", "requirement_type": "numeric", "company": "NVDA", "metrics": ["net_income"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-REV", "requirement_type": "numeric", "company": "NVDA", "metrics": ["revenue"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-FCF", "requirement_type": "numeric", "company": "NVDA", "metrics": ["free_cash_flow"], "required": False, "min_results": 1},
        {"requirement_id": "REQ-MCAP", "requirement_type": "calculation", "company": "NVDA", "metrics": ["market_cap"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-PE", "requirement_type": "calculation", "company": "NVDA", "metrics": ["pe_ratio"], "required": False, "min_results": 1},
        {"requirement_id": "REQ-PS", "requirement_type": "calculation", "company": "NVDA", "metrics": ["ps_ratio"], "required": False, "min_results": 1},
        {"requirement_id": "REQ-FCFY", "requirement_type": "calculation", "company": "NVDA", "metrics": ["fcf_yield"], "required": False, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan(reqs), _context())

    assert out["evidence_collection_results"][-4]["status"] == "satisfied"
    tool_rows = [row for row in out["tool_results"] if row.get("tool") == "compute_metrics"]
    pe_result = next(row for row in tool_rows if row.get("metric") == "pe_ratio")["data"]["results"][0]
    ps_result = next(row for row in tool_rows if row.get("metric") == "ps_ratio")["data"]["results"][0]
    assert pe_result["price_date"] == "2026-04-24"
    assert pe_result["shares_period"] == "2026-01-31"
    assert pe_result["statement_period"] == "2026-01-31"
    assert pe_result["market_cap"] == 1000.0
    assert len(pe_result["dependencies"]) >= 3
    assert ps_result["reconciliation_warning"] == "value_mismatch_gt_2pct"

    numeric = _build_numeric_evidence(out["tool_results"])
    pe_card = next(row for row in numeric if row["metric"] == "pe_ratio")
    assert pe_card["price_date"] == "2026-04-24"
    assert pe_card["dependencies"]


def test_valuation_multiple_missing_when_denominator_unavailable(monkeypatch):
    financial_tool = MetricFinancialTool(
        {
            "price": {"period_type": "daily", "period_end": "2026-04-24", "value": 100.0, "unit": "USD"},
            "shares_outstanding": {"period_type": "latest", "period_end": "2026-01-31", "value": 10.0, "unit": "shares"},
            "net_income": {"period_type": "annual", "period_end": "2026-01-31", "value": 0.0, "unit": "USD"},
        }
    )
    monkeypatch.setattr("src.agent.requirement_executor.query_financial_data", financial_tool)
    reqs = [
        {"requirement_id": "REQ-PRICE", "requirement_type": "numeric", "company": "NVDA", "metrics": ["price"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-SHARES", "requirement_type": "numeric", "company": "NVDA", "metrics": ["shares_outstanding"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-NI", "requirement_type": "numeric", "company": "NVDA", "metrics": ["net_income"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-MCAP", "requirement_type": "calculation", "company": "NVDA", "metrics": ["market_cap"], "required": True, "min_results": 1},
        {"requirement_id": "REQ-PE", "requirement_type": "calculation", "company": "NVDA", "metrics": ["pe_ratio"], "required": False, "min_results": 1},
    ]

    out = execute_evidence_requirements({"needs_tools": True, "answer_mode": "analytical"}, _plan(reqs), _context())

    assert out["evidence_collection_results"][-1]["status"] == "missing"
    assert out["evidence_collection_results"][-1]["failure_reason"] == "dependency_numeric_requirement_missing"
