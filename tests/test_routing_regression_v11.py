"""Routing boundary regression tests for V1.1 event capability."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_classifier_prompt_stays_compact():
    from datetime import date

    from src.agent.prompts import CLASSIFY_AND_EXTRACT

    prompt = CLASSIFY_AND_EXTRACT.format(
        user_query="分析下amazon这家公司",
        current_date=date.today().isoformat(),
        current_year=date.today().year,
    )

    assert len(prompt) < 3500
    assert prompt.count("\n") + 1 < 60


def _mock_classify(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_type: str,
    companies: list[str],
    comparison_target: str | None = None,
    requested_metrics: list[str] | None = None,
    data_route: str = "hybrid",
):
    from src.agent import nodes

    payload = {
        "task_type": task_type,
        "companies": companies,
        "comparison_target": comparison_target,
        "time_range": None,
        "period_query": None,
        "requested_metrics": requested_metrics or [],
        "data_route": data_route,
    }

    class DummyLLM:
        @staticmethod
        def invoke(_messages):
            class Response:
                content = json.dumps(payload, ensure_ascii=False)

            return Response()

    monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: DummyLLM())
    return nodes


def _sample_event_state() -> dict:
    return {
        "market_reaction_requested": True,
        "event_results": [
            {
                "ticker": "AAPL",
                "data": {
                    "ticker": "AAPL",
                    "events": [
                        {
                            "ticker": "AAPL",
                            "filing_id": "f1",
                            "form_type": "10-Q",
                            "fiscal_period": "2024-12-31",
                            "event_date": "2025-01-31",
                            "trading_anchor_date": "2025-01-31",
                            "coverage_flag": "complete",
                            "returns": {"return_1d": 0.01, "return_5d": 0.03},
                        }
                    ],
                },
            }
        ],
        "tool_results": [],
        "market_reaction_limitations": [],
    }


def test_chinese_event_phrase_quarterly_publish_triggers_event(monkeypatch):
    nodes = _mock_classify(
        monkeypatch,
        task_type="fact_qa",
        companies=["AAPL"],
        data_route="structured_only",
    )
    query = "每个季度apple发布财报的时候股价变化是什么"
    out = nodes.classify_and_extract({"user_query": query})

    assert out["task_type"] == "trend_analysis"
    assert out["event_intent"] == "required"
    assert out["market_reaction_requested"] is True
    assert "query_event_price_window" in out["selected_tools"]

    block = nodes._build_market_reaction_block(state=_sample_event_state(), lang="zh", numeric_cards=[])
    assert isinstance(block, dict)
    assert block.get("events")


def test_classifier_llm_timeout_falls_back_to_deterministic_overview(monkeypatch):
    from src.agent import nodes

    llm_kwargs: dict = {}

    class TimeoutLLM:
        @staticmethod
        def invoke(_messages):
            raise TimeoutError("classifier timed out")

    def fake_get_llm(*_args, **kwargs):
        llm_kwargs.update(kwargs)
        return TimeoutLLM()

    monkeypatch.setattr(nodes, "_get_llm", fake_get_llm)
    monkeypatch.setattr(nodes.settings, "llm_classify_fallback_enabled", True)

    out = nodes.classify_and_extract({"user_query": "分析下amazon这家公司"})

    assert out["companies"] == ["AMZN"]
    assert out["task_type"] == "report_summary"
    assert out["answer_mode"] == "analytical"
    assert out["analysis_scope"] == "single_company"
    assert out["methodology_intent"] == "single_company_overview"
    assert out["classifier_trace"]["source"] == "deterministic_fallback"
    assert out["classifier_trace"]["fallback_used"] is True
    assert out["classifier_trace"]["error_type"] == "TimeoutError"
    assert llm_kwargs["timeout"] <= 90
    assert llm_kwargs["max_retries"] == 0


def test_classifier_llm_timeout_raises_when_fallback_disabled(monkeypatch):
    from src.agent import nodes

    class TimeoutLLM:
        @staticmethod
        def invoke(_messages):
            raise TimeoutError("classifier timed out")

    monkeypatch.setattr(nodes, "_get_llm", lambda *args, **kwargs: TimeoutLLM())
    monkeypatch.setattr(nodes.settings, "llm_classify_fallback_enabled", False)

    with pytest.raises(TimeoutError):
        nodes.classify_and_extract({"user_query": "分析下amazon这家公司"})


def test_chinese_event_phrase_each_season_trigger(monkeypatch):
    nodes = _mock_classify(
        monkeypatch,
        task_type="trend_analysis",
        companies=["AAPL"],
        data_route="hybrid",
    )
    query = "Apple 每次发布季报时股价反应如何"
    out = nodes.classify_and_extract({"user_query": query})

    assert out["task_type"] == "trend_analysis"
    assert out["event_intent"] == "required"
    assert out["market_reaction_requested"] is True
    assert "query_event_price_window" in out["selected_tools"]


def test_non_event_trend_query_keeps_financial_main_path(monkeypatch):
    nodes = _mock_classify(
        monkeypatch,
        task_type="trend_analysis",
        companies=["AAPL"],
        data_route="structured_only",
    )
    query = "Apple 最近四个季度营收变化如何"
    out = nodes.classify_and_extract({"user_query": query})

    assert out["task_type"] == "trend_analysis"
    assert out["event_intent"] == "none"
    assert out["market_reaction_requested"] is False
    assert "query_event_price_window" not in out["selected_tools"]
    assert "query_financial_data" in out["selected_tools"]
    assert "compute_metrics" in out["selected_tools"]
    assert out["requested_metrics"] == ["revenue", "net_income"]


def test_non_event_fact_query_does_not_trigger_event(monkeypatch):
    nodes = _mock_classify(
        monkeypatch,
        task_type="fact_qa",
        companies=["AMZN"],
        requested_metrics=["net_income"],
        data_route="structured_only",
    )
    query = "Amazon 最新一个财年的净利润是多少"
    out = nodes.classify_and_extract({"user_query": query})

    assert out["task_type"] == "fact_qa"
    assert out["event_intent"] == "none"
    assert out["market_reaction_requested"] is False
    assert "query_event_price_window" not in out["selected_tools"]
    assert "query_financial_data" in out["selected_tools"]

    block = nodes._build_market_reaction_block(
        state={"market_reaction_requested": False, "event_results": [], "tool_results": []},
        lang="zh",
        numeric_cards=[],
    )
    assert block is None


def test_non_event_comparison_query_does_not_trigger_event(monkeypatch):
    nodes = _mock_classify(
        monkeypatch,
        task_type="company_comparison",
        companies=["AAPL", "AMZN"],
        comparison_target="AMZN",
        requested_metrics=["revenue"],
        data_route="structured_only",
    )
    query = "比较 Apple 和 Amazon 最新财年营收"
    out = nodes.classify_and_extract({"user_query": query})

    assert out["task_type"] == "company_comparison"
    assert out["event_intent"] == "none"
    assert out["market_reaction_requested"] is False
    assert "query_event_price_window" not in out["selected_tools"]
    assert "query_financial_data" in out["selected_tools"]


def test_why_post_filing_drop_triggers_event_and_text_path(monkeypatch):
    nodes = _mock_classify(
        monkeypatch,
        task_type="report_summary",
        companies=["AAPL"],
        data_route="hybrid",
    )
    query = "为什么 Apple 最近一次财报后股价下跌"
    out = nodes.classify_and_extract({"user_query": query})

    assert out["event_intent"] == "required"
    assert out["market_reaction_requested"] is True
    assert "query_event_price_window" in out["selected_tools"]
    assert "search_filings" in out["selected_tools"]
