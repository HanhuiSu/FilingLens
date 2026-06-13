"""Regression tests for conversational live-path final state consistency."""

from __future__ import annotations

from datetime import date
from typing import Any

from src.agent.answering import _text_drop_stage, _text_requirement_diagnostics, generate_agent_answer
from src.agent.evidence_sufficiency import evaluate_evidence_sufficiency
from src.agent.query_plan import build_classification_state


class FakeLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _messages: list[dict[str, Any]]):
        class Response:
            def __init__(self, content: str):
                self.content = content

        return Response(self.content)


class FakeLLMSequence:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)

    def invoke(self, _messages: list[dict[str, Any]]):
        if not self.contents:
            content = ""
        elif len(self.contents) == 1:
            content = self.contents[0]
        else:
            content = self.contents.pop(0)

        class Response:
            def __init__(self, content: str):
                self.content = content

        return Response(content)


def _patch_llms(monkeypatch, *, text_response: str, draft_response: str | list[str]):
    draft_llm = FakeLLMSequence(draft_response) if isinstance(draft_response, list) else FakeLLM(draft_response)
    monkeypatch.setattr(
        "src.agent.answering._get_llm",
        lambda *args, **kwargs: FakeLLM(text_response),
    )
    monkeypatch.setattr(
        "src.agent.analyst_draft._get_llm",
        lambda *args, **kwargs: draft_llm,
    )


def _assert_final_owner_consistent(out: dict[str, Any]) -> None:
    assert out["final_answer_source"] == out["output"]["final_answer_source"]
    assert out.get("answer_history")
    assert out["answer_history"][-1]["new_owner"] == out["final_answer_source"]


def _base_state(query: str = "aapple和amazon你最看好哪个") -> dict[str, Any]:
    state = build_classification_state(
        user_query=query,
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="trace-live-final-state",
        today=date(2026, 4, 23),
    )
    state["user_query"] = query
    state["resolved_period_context"] = {
        "target_period_type": "annual",
        "same_period_match": True,
        "common_periods": ["2025-12-31"],
        "needs_clarification": False,
    }
    state["period_query"] = {
        "period_type": "latest",
        "year": None,
        "quarter": None,
        "trailing_n": None,
        "year_basis": "fiscal",
        "comparison_basis": "same_period",
    }

    requirements = list(state["evidence_plan"]["evidence_requirements"])
    numeric_values = {
        ("AAPL", "revenue"): 120.0,
        ("AAPL", "net_income"): 32.0,
        ("AAPL", "price"): 200.0,
        ("AMZN", "revenue"): 150.0,
        ("AMZN", "net_income"): 22.0,
        ("AMZN", "price"): 180.0,
    }
    text_docs = {
        "AAPL": {
            "filing_id": "aapl-10k",
            "form_type": "10-K",
            "fiscal_period": "2025-09-28",
            "section": "ITEM_7",
            "chunk_order": 1,
            "quality": "high",
            "text": "Apple MD&A described margin discipline and steady operating execution.",
            "supporting_snippet": "Apple highlighted margin discipline and steady operating execution.",
        },
        "AMZN": {
            "filing_id": "amzn-10k",
            "form_type": "10-K",
            "fiscal_period": "2025-12-31",
            "section": "ITEM_7",
            "chunk_order": 1,
            "quality": "high",
            "text": "Amazon MD&A discussed revenue scale, reinvestment, and operating leverage.",
            "supporting_snippet": "Amazon discussed revenue scale, reinvestment, and operating leverage.",
        },
    }

    tool_results: list[dict[str, Any]] = []
    retrieved_docs: list[dict[str, Any]] = []
    collection_results: list[dict[str, Any]] = []
    for req in requirements:
        rid = str(req.get("requirement_id", ""))
        req_type = str(req.get("requirement_type", ""))
        company = str(req.get("company", ""))
        if req_type == "numeric":
            metric = str(req.get("metric", ""))
            value = numeric_values.get((company, metric))
            if value is None:
                collection_results.append(
                    {
                        "requirement_id": rid,
                        "status": "missing",
                        "evidence_type": "numeric",
                        "items": [],
                        "retry_count": 0,
                        "failure_reason": "no_matching_evidence",
                    }
                )
                continue
            row = {
                "requirement_id": rid,
                "ticker": company,
                "metric": metric,
                "period_type": "annual",
                "period_end": "2025-12-31",
                "value": value,
                "unit": "USD",
                "source_provider": "yfinance" if metric == "price" else "sec_companyfacts",
                "confidence": "high",
                "extraction_method": "companyfacts",
            }
            tool_results.append(
                {
                    "tool": "query_financial_data",
                    "ticker": company,
                    "requirement_id": rid,
                    "data": {"ticker": company, "financial_facts": [row], "price_data": []},
                }
            )
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "satisfied",
                    "evidence_type": "numeric",
                    "items": [row],
                    "retry_count": 0,
                }
            )
        elif req_type == "text":
            doc = {"requirement_id": rid, "ticker": company, **text_docs[company]}
            retrieved_docs.append(doc)
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "satisfied",
                    "evidence_type": "text",
                    "items": [doc],
                    "retry_count": 0,
                }
            )
        elif req_type == "calculation":
            metric = str(req.get("metric") or next(iter(req.get("metrics", []) or []), ""))
            if "valuation_evidence_missing" in {str(item) for item in req.get("fallback_strategy", []) or []}:
                collection_results.append(
                    {
                        "requirement_id": rid,
                        "status": "missing",
                        "evidence_type": "calculation",
                        "items": [],
                        "retry_count": 0,
                        "failure_reason": "valuation_evidence_missing",
                    }
                )
                continue
            if metric != "net_margin" and "OPERATING_MARGIN" not in rid:
                collection_results.append(
                    {
                        "requirement_id": rid,
                        "status": "missing",
                        "evidence_type": "calculation",
                        "items": [],
                        "retry_count": 0,
                        "failure_reason": "dependency_numeric_requirement_missing",
                    }
                )
                continue
            margin = numeric_values[(company, "net_income")] / numeric_values[(company, "revenue")]
            data = {
                "results": [
                    {
                        "period": "2025-12-31",
                        "period_type": "annual",
                        "margin": round(margin, 6),
                        "margin_pct": f"{margin * 100:.2f}%",
                        "numerator_metric": "net_income",
                        "denominator_metric": "revenue",
                        "source_tag": "net_income_over_revenue",
                    }
                ]
            }
            item = {
                "requirement_id": rid,
                "company": company,
                "ticker": company,
                "metric": "net_margin",
                "computation": "margin",
                "source_tag": "net_income_over_revenue",
                "data": data,
            }
            tool_results.append({"tool": "compute_metrics", **item})
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "satisfied",
                    "evidence_type": "calculation",
                    "items": [item],
                    "retry_count": 0,
                }
            )

    state["tool_results"] = tool_results
    state["retrieved_docs"] = retrieved_docs
    state["evidence_collection_results"] = collection_results
    state["evidence_sufficiency"] = evaluate_evidence_sufficiency(
        state["evidence_plan"],
        collection_results,
    ).model_dump(exclude_none=True)
    return state


def _text_requirement_id(state: dict[str, Any], company: str, dimension_id: str = "moat_and_competitive_risk") -> str:
    for req in state["evidence_plan"]["evidence_requirements"]:
        if (
            str(req.get("requirement_type")) == "text"
            and str(req.get("company")) == company
            and str(req.get("dimension_id")) == dimension_id
        ):
            return str(req.get("requirement_id"))
    raise AssertionError(f"Missing text requirement for {company} {dimension_id}")


def _analytical_state(query: str = "苹果现在最大的问题是什么？", *, include_mda: bool = False) -> dict[str, Any]:
    state = build_classification_state(
        user_query=query,
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="trace-live-analytical",
        today=date(2026, 4, 23),
    )
    state["user_query"] = query
    requirements = list(state["evidence_plan"]["evidence_requirements"])

    numeric_values = {
        ("AAPL", "revenue"): 120.0,
        ("AAPL", "net_income"): 32.0,
    }
    tool_results: list[dict[str, Any]] = []
    retrieved_docs: list[dict[str, Any]] = []
    collection_results: list[dict[str, Any]] = []
    for req in requirements:
        rid = str(req.get("requirement_id", ""))
        req_type = str(req.get("requirement_type", ""))
        company = str(req.get("company", ""))
        section = tuple(req.get("section_preferences", []) or [])
        if req_type == "numeric":
            metric = str(req.get("metric", ""))
            value = numeric_values.get((company, metric))
            if value is None:
                collection_results.append(
                    {
                        "requirement_id": rid,
                        "status": "missing",
                        "evidence_type": "numeric",
                        "items": [],
                        "retry_count": 0,
                        "failure_reason": "no_matching_evidence",
                    }
                )
                continue
            row = {
                "requirement_id": rid,
                "ticker": company,
                "metric": metric,
                "period_type": "annual",
                "period_end": "2025-12-31",
                "value": value,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
                "confidence": "high",
                "extraction_method": "companyfacts",
            }
            tool_results.append(
                {
                    "tool": "query_financial_data",
                    "ticker": company,
                    "requirement_id": rid,
                    "data": {"ticker": company, "financial_facts": [row], "price_data": []},
                }
            )
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "satisfied",
                    "evidence_type": "numeric",
                    "items": [row],
                    "retry_count": 0,
                }
            )
            continue
        if req_type == "calculation":
            metric = str(req.get("metric") or next(iter(req.get("metrics", []) or []), ""))
            if "valuation_evidence_missing" in {str(item) for item in req.get("fallback_strategy", []) or []}:
                collection_results.append(
                    {
                        "requirement_id": rid,
                        "status": "missing",
                        "evidence_type": "calculation",
                        "items": [],
                        "retry_count": 0,
                        "failure_reason": "valuation_evidence_missing",
                    }
                )
                continue
            if metric == "net_margin" and (company, "revenue") in numeric_values and (company, "net_income") in numeric_values:
                margin = numeric_values[(company, "net_income")] / numeric_values[(company, "revenue")]
                data = {
                    "results": [
                        {
                            "period": "2025-12-31",
                            "period_type": "annual",
                            "margin": round(margin, 6),
                            "margin_pct": f"{margin * 100:.2f}%",
                            "numerator_metric": "net_income",
                            "denominator_metric": "revenue",
                            "source_tag": "net_income_over_revenue",
                        }
                    ]
                }
                item = {
                    "requirement_id": rid,
                    "company": company,
                    "ticker": company,
                    "metric": "net_margin",
                    "computation": "margin",
                    "source_tag": "net_income_over_revenue",
                    "data": data,
                }
                tool_results.append({"tool": "compute_metrics", **item})
                collection_results.append(
                    {
                        "requirement_id": rid,
                        "status": "satisfied",
                        "evidence_type": "calculation",
                        "items": [item],
                        "retry_count": 0,
                    }
                )
                continue
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "missing",
                    "evidence_type": "calculation",
                    "items": [],
                    "retry_count": 0,
                    "failure_reason": "dependency_numeric_requirement_missing",
                }
            )
            continue
        if req_type != "text":
            continue
        if section == ("ITEM_1A",):
            doc = {
                "requirement_id": rid,
                "ticker": "AAPL",
                "filing_id": "aapl-10k",
                "form_type": "10-K",
                "fiscal_period": "2025-09-28",
                "section": "ITEM_1A",
                "chunk_order": 1,
                "quality": "high",
                "text": "Competition and regulation remain material risks for Apple's business.",
                "supporting_snippet": "Competition and regulation remain material risks for Apple's business.",
            }
            retrieved_docs.append(doc)
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "satisfied",
                    "evidence_type": "text",
                    "items": [doc],
                    "retry_count": 0,
                    "raw_hit_count": 1,
                    "usable_hit_count": 1,
                }
            )
        elif include_mda and section == ("ITEM_7",):
            doc = {
                "requirement_id": rid,
                "ticker": "AAPL",
                "filing_id": "aapl-10k",
                "form_type": "10-K",
                "fiscal_period": "2025-09-28",
                "section": "ITEM_7",
                "chunk_order": 2,
                "quality": "high",
                "text": "Management discussed demand softness and margin pressure in recent periods.",
                "supporting_snippet": "Management discussed demand softness and margin pressure in recent periods.",
            }
            retrieved_docs.append(doc)
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "satisfied",
                    "evidence_type": "text",
                    "items": [doc],
                    "retry_count": 0,
                    "raw_hit_count": 1,
                    "usable_hit_count": 1,
                }
            )
        else:
            collection_results.append(
                {
                    "requirement_id": rid,
                    "status": "missing",
                    "evidence_type": "text",
                    "items": [],
                    "retry_count": 0,
                    "raw_hit_count": 0,
                    "usable_hit_count": 0,
                }
            )

    state["tool_results"] = tool_results
    state["retrieved_docs"] = retrieved_docs
    state["evidence_collection_results"] = collection_results
    state["evidence_sufficiency"] = evaluate_evidence_sufficiency(
        state["evidence_plan"],
        collection_results,
    ).model_dump(exclude_none=True)
    return state


def test_live_path_numeric_only_comparison_becomes_limited_judgment(monkeypatch):
    state = _base_state()
    _patch_llms(
        monkeypatch,
        text_response='{"text_claims":[{"sentence":"Apple discussed margin discipline.","evidence_ids":["T1"]}]}',
        draft_response=(
            '{"framework_summary":"基于基本面质量框架组织证据。",'
            '"dimension_analyses":[{"dimension_id":"revenue_quality","status":"satisfied","claim":"AMZN 在收入规模上占优。","evidence_refs":["N4"]},'
            '{"dimension_id":"profitability_quality","status":"satisfied","claim":"AAPL 在净利润和净利率证据上更强。","evidence_refs":["N2","N3","N5","N6"]}],'
            '"overall_judgment":"如果只基于当前财报证据，我更偏向 AAPL，因为净利润和净利率证据更强；但如果更看重收入规模和增长弹性，AMZN 仍值得关注。",'
            '"methodology_counterpoints":["如果更看重收入规模和增长弹性，AMZN 仍值得继续观察。"],'
            '"methodology_limitations":["当前缺少可验证文本证据，因此不能做完整风险比较。"],'
            '"tentative_conclusion":{"statement":"如果只基于当前财报证据，我更偏向 AAPL，因为净利润和净利率证据更强；但如果更看重收入规模和增长弹性，AMZN 仍值得关注。",'
            '"stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2","N3","N6"]},'
            '"decision_basis":[{"statement":"AAPL 当前净利润高于 AMZN。","citation_refs":["N2","N6"]}],'
            '"supporting_points":[{"statement":"AAPL 当前净利润和净利率证据更强。","citation_refs":["N2","N3","N6"]}],'
            '"counterpoints":[{"statement":"如果更看重收入规模和增长弹性，AMZN 仍值得继续观察。","citation_refs":["N5"]}],'
            '"risk_tradeoffs":[],'
            '"uncertainty_notes":[{"statement":"当前缺少可验证文本证据，因此这只是有限判断。","citation_refs":[]}],'
            '"citation_refs":["N2","N3","N5","N6"],'
            '"safety_notes":[{"statement":"这不是投资建议。","citation_refs":[]}]}'
        ),
    )

    out = generate_agent_answer(state)

    assert out["output"]["answer_mode"] == "comparison_brief"
    assert out["output"]["safety_intent"] == "investment_advice_like"
    assert out["synthesis_mode"] == "limited_judgment"
    assert out["trace_summary"]["sufficiency_status"] == "partial"
    assert out["trace_summary"]["required_text_satisfied_rate"] == 0.0
    assert out["trace_summary"]["final_synthesis_mode"] == "limited_judgment"
    assert out["output"]["text_evidence"] == []
    assert out["validated_text_evidence_count"] == 0
    assert any(item["metric"] == "net_margin" for item in out["output"]["numeric_evidence"])
    codes = {item["code"] for item in out["output"]["limitations"]}
    assert "numeric_only_comparison" in codes
    assert "investment_advice_boundary" in codes
    limitation_text = "\n".join(item.get("message", "") for item in out["output"]["limitations"])
    assert "REQ-" not in limitation_text
    assert "dependency_numeric_requirement_missing" not in limitation_text
    assert "numeric_only_comparison" not in limitation_text
    aapl_text_req = _text_requirement_id(state, "AAPL")
    amzn_text_req = _text_requirement_id(state, "AMZN")
    assert out["requirement_status_map"][aapl_text_req]["status"] == "missing"
    assert out["requirement_status_map"][amzn_text_req]["status"] == "missing"
    assert out["text_requirement_diagnostics"][aapl_text_req]["drop_stage"] == "citation_policy_dropped"
    assert "推荐买" not in out["final_answer"]
    assert "买入" not in out["final_answer"]
    assert "更偏向 AAPL" in out["final_answer"]
    assert "当前缺少足够的可验证文本证据" in out["final_answer"]
    assert "执行 / 再投资" not in out["final_answer"]
    assert out["final_answer_source"] == "comparison_decision_fallback"
    _assert_final_owner_consistent(out)
    assert out["draft_validation"]["passed"] is False
    assert out["draft_status"] == "invalid"
    assert any(item.get("reason") == "missing_citation_ref" for item in out["draft_validation"].get("warnings", []))
    assert out["draft_final_status"] == "fallback"
    assert len(out["draft_attempts"]) >= 1
    assert len(out["output"]["draft_revision_attempts"]) == max(0, len(out["draft_attempts"]) - 1)
    assert out["output"]["comparison_judgment_frame"]["preferred_company"] == "AAPL"
    assert out["output"]["comparison_judgment_frame"]["margin_winner"] == "AAPL"
    assert "反方观点" in out["final_answer"]
    assert not (
        out["synthesis_mode"] == "balanced_comparison"
        and not out["output"]["text_evidence"]
        and out["trace_summary"]["required_text_satisfied_rate"] > 0
    )


def test_live_path_text_grounded_comparison_uses_validated_text(monkeypatch):
    state = _base_state()
    _patch_llms(
        monkeypatch,
        text_response=(
            '{"text_claims":[{"sentence":"Apple discussed margin discipline.","evidence_ids":["T1"]},'
            '{"sentence":"Amazon discussed revenue scale and reinvestment.","evidence_ids":["T2"]}]}'
        ),
        draft_response=(
            '{"framework_summary":"基于基本面质量框架组织证据。",'
            '"dimension_analyses":[{"dimension_id":"revenue_quality","status":"satisfied","claim":"AMZN 在收入规模上占优。","evidence_refs":["N4"]},'
            '{"dimension_id":"profitability_quality","status":"satisfied","claim":"AAPL 在净利润和净利率证据上更强。","evidence_refs":["N2","N3","N5","N6"]},'
            '{"dimension_id":"moat_and_competitive_risk","status":"satisfied","claim":"已验证文本为两家公司提供风险与竞争背景，但不支持强风险排序。","evidence_refs":["T1","T2"]}],'
            '"overall_judgment":"如果更看重净利润、净利率和执行稳定性，我会更偏向 AAPL；但 AMZN 在收入规模和再投资弹性上仍值得关注。",'
            '"methodology_counterpoints":["AMZN 在收入规模和再投资弹性上仍值得关注。"],'
            '"methodology_limitations":["当前缺少估值证据，因此不能判断哪家公司价格更便宜。"],'
            '"tentative_conclusion":{"statement":"如果更看重净利润、净利率和执行稳定性，我会更偏向 AAPL；但 AMZN 在收入规模和再投资弹性上仍值得关注。",'
            '"stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2","T1","T2"]},'
            '"decision_basis":[{"statement":"AAPL 的文本证据强调了利润率纪律和稳定执行。","citation_refs":["T1"]}],'
            '"supporting_points":[{"statement":"AAPL 当前净利润表现也更强。","citation_refs":["N2","N6"]}],'
            '"counterpoints":[{"statement":"AMZN 的文本证据强调了收入规模和再投资弹性。","citation_refs":["T2"]}],'
            '"risk_tradeoffs":[{"statement":"两家公司的优势侧重点并不相同。","citation_refs":["T1","T2"]}],'
            '"uncertainty_notes":[{"statement":"这仍然只是基于当前已验证财报证据的有限比较。","citation_refs":["T1","T2"]}],'
            '"citation_refs":["N2","N6","T1","T2"],'
            '"safety_notes":[{"statement":"这不是投资建议。","citation_refs":[]}]}'
        ),
    )

    out = generate_agent_answer(state)

    planned_dimensions = {
        str(req.get("dimension_id"))
        for req in state["evidence_plan"]["evidence_requirements"]
    }
    assert {
        "revenue_quality",
        "profitability_quality",
        "moat_and_competitive_risk",
        "valuation_and_risk_boundary",
    } <= planned_dimensions
    assert out["synthesis_mode"] == "limited_judgment"
    assert out["trace_summary"]["sufficiency_status"] == "partial"
    assert out["trace_summary"]["required_text_satisfied_rate"] == 1.0
    assert out["trace_summary"]["framework_sufficiency_status"] == "partial"
    assert out["trace_summary"]["final_synthesis_mode"] == "limited_judgment"
    assert out["validated_text_evidence_count"] >= 2
    assert len(out["output"]["text_evidence"]) >= 2
    assert any(item["metric"] == "net_margin" for item in out["output"]["numeric_evidence"])
    assert out["dimension_coverage_rate"] > 0
    assert out["dimension_status_map"]["revenue_quality"]["status"] == "satisfied"
    assert out["dimension_status_map"]["profitability_quality"]["status"] == "satisfied"
    assert out["dimension_status_map"]["moat_and_competitive_risk"]["status"] != "missing"
    assert out["dimension_status_map"]["valuation_and_risk_boundary"]["status"] == "partial"
    codes = {item["code"] for item in out["output"]["limitations"]}
    assert "numeric_only_comparison" not in codes
    aapl_text_req = _text_requirement_id(state, "AAPL")
    amzn_text_req = _text_requirement_id(state, "AMZN")
    assert out["requirement_status_map"][aapl_text_req]["status"] == "satisfied"
    assert out["requirement_status_map"][amzn_text_req]["status"] == "satisfied"
    assert out["text_requirement_diagnostics"][aapl_text_req]["final_validated_text_count"] >= 1
    assert "更偏向 AAPL" in out["final_answer"]
    assert "[T" in out["final_answer"]
    assert "比较判断" in out["final_answer"]
    assert "维度分析" in out["final_answer"]
    assert "反方观点" in out["final_answer"]
    assert "限制" in out["final_answer"]
    assert "收入质量" in out["final_answer"]
    assert "盈利质量" in out["final_answer"]
    assert "竞争与风险" in out["final_answer"]
    assert "估值" in out["final_answer"]
    assert "AAPL" in out["final_answer"] and "AMZN" in out["final_answer"]
    assert "AAPL 净利率" in out["final_answer"]
    assert "AMZN 净利率" in out["final_answer"]
    assert "Apple MD&A described" not in out["final_answer"]
    assert "Amazon MD&A discussed" not in out["final_answer"]
    assert out["output"]["view"]["kind"] == "methodology_comparison_brief"
    assert out["synthesis"]["methodology_answer"]["dimension_sections"]
    assert out["output"]["view"]["methodology_answer"]["dimension_sections"]
    metric_rows = {
        row["metric_id"]: row
        for row in out["output"]["view"]["methodology_answer"]["metric_table"]
    }
    assert {"revenue", "net_income", "net_margin"} <= set(metric_rows)
    assert {"AAPL", "AMZN"} <= set(metric_rows["net_margin"]["company_values"])
    assert out["output"]["view"]["metric_table"]
    assert "估值风险呈分歧" in out["final_answer"]
    assert "No validated text evidence" not in out["final_answer"]
    assert "REQ-" not in out["final_answer"]
    assert "dependency_" not in out["final_answer"]
    assert "numeric_only_" not in out["final_answer"]
    assert "requirement_missing" not in out["final_answer"]
    assert "required_evidence_missing" not in out["final_answer"]
    assert "unsupported_claims_present" not in out["final_answer"]
    _assert_final_owner_consistent(out)
    assert out["output"]["analyst_draft_summary"]["counterpoints_count"] >= 1
    assert out["output"]["comparison_judgment_frame"]["margin_winner"] == "AAPL"
    assert out["draft_final_status"] == "fallback"


def test_validated_text_claim_enters_final_bundle(monkeypatch):
    state = _base_state()
    _patch_llms(
        monkeypatch,
        text_response='{"text_claims":[]}',
        draft_response=(
            '{"tentative_conclusion":{"statement":"如果更看重净利润和净利率质量，我更偏向 AAPL；但 AMZN 在收入规模和再投资弹性上仍值得关注。",'
            '"stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2","T1","T2"]},'
            '"decision_basis":[{"statement":"AAPL 的文本证据提供了利润率纪律和执行背景。","citation_refs":["T1"]}],'
            '"supporting_points":[{"statement":"AAPL 当前净利润表现更强。","citation_refs":["N2","N6"]}],'
            '"counterpoints":[{"statement":"AMZN 的文本证据提供了收入规模和再投资背景。","citation_refs":["T2"]}],'
            '"risk_tradeoffs":[{"statement":"两家公司披露文本支持的风险和业务背景不同。","citation_refs":["T1","T2"]}],'
            '"uncertainty_notes":[{"statement":"这仍然只是基于当前已验证财报证据的有限比较。","citation_refs":["T1","T2"]}],'
            '"citation_refs":["N2","N6","T1","T2"],'
            '"safety_notes":[{"statement":"这不是投资建议。","citation_refs":[]}]}'
        ),
    )

    out = generate_agent_answer(state)

    assert out["validated_text_evidence_count"] >= 2
    assert out["trace_summary"]["required_text_satisfied_rate"] > 0
    assert out["output"]["text_evidence"]
    assert out["text_citations"]
    aapl_text_req = _text_requirement_id(state, "AAPL")
    assert out["text_requirement_diagnostics"][aapl_text_req]["claim_generation_error"] == "model_returned_empty_text_claims"
    assert out["text_requirement_diagnostics"][aapl_text_req]["validated_text_claim_count"] >= 1


def test_comparison_answer_uses_validated_text_claim(monkeypatch):
    state = _base_state()
    _patch_llms(
        monkeypatch,
        text_response='{"text_claims":[]}',
        draft_response=(
            '{"tentative_conclusion":{"statement":"如果更看重净利润和净利率质量，我更偏向 AAPL；但 AMZN 在收入规模和再投资弹性上仍值得关注。",'
            '"stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2","T1","T2"]},'
            '"decision_basis":[{"statement":"AAPL 的文本证据提供了利润率纪律和执行背景。","citation_refs":["T1"]}],'
            '"supporting_points":[{"statement":"AAPL 当前净利润表现更强。","citation_refs":["N2","N6"]}],'
            '"counterpoints":[{"statement":"AMZN 的文本证据提供了收入规模和再投资背景。","citation_refs":["T2"]}],'
            '"risk_tradeoffs":[{"statement":"AAPL 和 AMZN 的文本证据都进入了最终判断。","citation_refs":["T1","T2"]}],'
            '"uncertainty_notes":[{"statement":"这仍然只是基于当前已验证财报证据的有限比较。","citation_refs":["T1","T2"]}],'
            '"citation_refs":["N2","N6","T1","T2"],'
            '"safety_notes":[{"statement":"这不是投资建议。","citation_refs":[]}]}'
        ),
    )

    out = generate_agent_answer(state)

    assert out["synthesis_mode"] == "limited_judgment"
    assert "[T" in out["final_answer"]
    assert "当前缺少足够的可验证文本证据" not in out["final_answer"]
    assert len(out["output"]["text_evidence"]) >= 2


def test_trace_keeps_internal_diagnostics(monkeypatch):
    state = _base_state()
    _patch_llms(
        monkeypatch,
        text_response='{"text_claims":[{"sentence":"Apple discussed margin discipline.","evidence_ids":["T1"]}]}',
        draft_response=(
            '{"tentative_conclusion":{"statement":"如果只基于当前财报证据，我更偏向 AAPL；但如果更看重收入规模，AMZN 仍值得关注。",'
            '"stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2","N5","N6"]},'
            '"decision_basis":[{"statement":"AAPL 当前净利润高于 AMZN。","citation_refs":["N2","N6"]}],'
            '"supporting_points":[],'
            '"counterpoints":[{"statement":"AMZN 的收入规模更大。","citation_refs":["N5"]}],'
            '"risk_tradeoffs":[],'
            '"uncertainty_notes":[{"statement":"当前缺少可验证文本证据，因此这只是有限判断。","citation_refs":[]}],'
            '"citation_refs":["N2","N5","N6"],'
            '"safety_notes":[{"statement":"这不是投资建议。","citation_refs":[]}]}'
        ),
    )

    out = generate_agent_answer(state)

    visible_text = out["final_answer"] + "\n" + "\n".join(
        item.get("message", "") for item in out["output"]["limitations"]
    )
    aapl_text_req = _text_requirement_id(state, "AAPL")
    assert aapl_text_req not in visible_text
    assert "claim_validation_failed" not in visible_text
    assert "comparison_text_unbalanced" not in visible_text
    assert "citation_policy_dropped" not in visible_text
    assert "numeric_only_comparison" not in visible_text

    assert aapl_text_req in out["requirement_status_map"]
    trace_item = out["requirement_status_map"][aapl_text_req]
    assert trace_item["status"] == "missing"
    assert trace_item["failure_reason"] == "comparison_text_unbalanced"
    assert out["text_requirement_diagnostics"][aapl_text_req]["drop_stage"] == "citation_policy_dropped"
    assert any(
        item.get("requirement_id") == aapl_text_req
        and item.get("failure_reason") == "comparison_text_unbalanced"
        for item in out["requirement_limitations"]
    )


def test_text_requirement_drop_stage_classification():
    base = {
        "raw_hit_count": 1,
        "section_filtered_hit_count": 1,
        "usable_hit_count": 1,
        "snippet_support_passed_count": 1,
        "validated_text_claim_count": 1,
        "text_citation_kept_count": 1,
        "final_validated_text_count": 0,
        "failure_reason": "",
        "rejection_reasons": {},
        "collection_drop_stage": "",
    }

    assert _text_drop_stage(**{**base, "raw_hit_count": 0}) == "no_raw_hits"
    assert _text_drop_stage(**{**base, "section_filtered_hit_count": 0}) == "section_filter_dropped"
    assert _text_drop_stage(
        **{
            **base,
            "usable_hit_count": 0,
            "rejection_reasons": {"quality_filter_dropped": 1},
        }
    ) == "quality_filter_dropped"
    assert _text_drop_stage(
        **{
            **base,
            "usable_hit_count": 0,
            "rejection_reasons": {"snippet_support_failed": 1},
        }
    ) == "snippet_support_failed"
    assert _text_drop_stage(**{**base, "validated_text_claim_count": 0}) == "claim_validation_failed"
    assert _text_drop_stage(**{**base, "text_citation_kept_count": 0}) == "citation_policy_dropped"
    assert _text_drop_stage(**base) == "final_bundle_dropped"
    assert _text_drop_stage(**{**base, "final_validated_text_count": 1}) == "satisfied"


def test_final_bundle_drop_is_detected():
    rid = "REQ-TEXT-AAPL-RISK"
    state = {
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": rid,
                    "requirement_type": "text",
                    "company": "AAPL",
                    "section_preferences": ["ITEM_1A"],
                    "retrieval_query": "AAPL risk factors",
                    "fallback_sections": ["ITEM_7"],
                    "broadened_queries": ["risk factors"],
                }
            ]
        },
        "evidence_collection_results": [
            {
                "requirement_id": rid,
                "status": "satisfied",
                "evidence_type": "text",
                "items": [{"requirement_id": rid, "evidence_id": "T1"}],
                "raw_hit_count": 2,
                "section_filtered_hit_count": 2,
                "usable_hit_count": 1,
                "snippet_support_passed_count": 1,
                "drop_stage": "satisfied",
            }
        ],
        "retry_history": [],
    }

    diagnostics = _text_requirement_diagnostics(
        state,
        candidate_text_evidence=[{"evidence_id": "T1", "requirement_id": rid}],
        valid_text_claims=[{"sentence": "Apple has risk factors.", "evidence_ids": ["T1"]}],
        text_citations=[{"evidence_id": "T1", "requirement_id": rid}],
        final_text_evidence=[],
        requirement_status_map={rid: {"failure_reason": ""}},
    )

    assert diagnostics[rid]["text_claim_validated_count"] == 1
    assert diagnostics[rid]["text_citation_kept_count"] == 1
    assert diagnostics[rid]["final_validated_text_count"] == 0
    assert diagnostics[rid]["drop_stage"] == "final_bundle_dropped"


def test_live_path_partial_analytical_text_becomes_limited_analysis(monkeypatch):
    state = _analytical_state(include_mda=False)
    _patch_llms(
        monkeypatch,
        text_response='{"text_claims":[{"sentence":"Competition and regulation remain material risks.","evidence_ids":["T1"]}]}',
        draft_response=(
            '{"tentative_conclusion":{"statement":"基于当前已验证文本，苹果当前更突出的压力在于竞争与监管风险。",'
            '"stance":"main_issue","preferred_company":"AAPL","citation_refs":["T1"]},'
            '"decision_basis":[{"statement":"已验证文本直接提到竞争和监管仍是重要风险。","citation_refs":["T1"]}],'
            '"supporting_points":[{"statement":"当前可验证文本主要集中在风险披露，而不是更完整的管理层讨论。","citation_refs":["T1"]}],'
            '"counterpoints":[],'
            '"risk_tradeoffs":[{"statement":"由于缺少更完整的 MD&A 文本，这个判断仍然有限。","citation_refs":["T1"]}],'
            '"uncertainty_notes":[{"statement":"当前只拿到部分已验证文本证据，因此不能把它当作完整分析。","citation_refs":["T1"]}],'
            '"citation_refs":["T1"],'
            '"safety_notes":[]}'
        ),
    )

    out = generate_agent_answer(state)

    assert out["output"]["answer_mode"] == "risk_focused_analysis"
    assert out["synthesis_mode"] == "risk_focused_analysis"
    assert out["trace_summary"]["sufficiency_status"] in {"partial", "focused_sufficient"}
    assert out["trace_summary"]["final_synthesis_mode"] == "risk_focused_analysis"
    assert out["output"]["view"]["kind"] == "risk_focused_analysis_brief"
    assert out["validated_numeric_evidence_count"] == 3
    assert out["validated_text_evidence_count"] == 1
    assert len(out["output"]["text_evidence"]) == 1
    assert "风险判断" in out["final_answer"]
    assert "为什么这是核心问题" in out["final_answer"]
    assert "财报证据" in out["final_answer"]
    assert "证据边界" in out["final_answer"]
    assert "核心指标" not in out["final_answer"].splitlines()[:4]
    assert "business_model" not in out["final_answer"]
    assert out["final_answer_source"] in {"analyst_draft_initial", "deterministic_synthesis"}


def test_single_company_text_claim_fallback_enters_final_bundle(monkeypatch):
    state = build_classification_state(
        user_query="分析下 nvidia",
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="trace-single-company-text",
        today=date(2026, 4, 23),
    )
    state["user_query"] = "分析下 nvidia"
    state["resolved_period_context"] = {
        "target_period_type": "annual",
        "same_period_match": True,
        "common_periods": ["2025-12-31"],
        "needs_clarification": False,
    }
    state["period_query"] = {
        "period_type": "latest",
        "year": None,
        "quarter": None,
        "trailing_n": None,
        "year_basis": "fiscal",
        "comparison_basis": "latest_available",
    }

    numeric_values = {
        ("NVDA", "revenue"): 60.9,
        ("NVDA", "net_income"): 29.8,
    }
    tool_results: list[dict[str, Any]] = []
    retrieved_docs: list[dict[str, Any]] = []
    collection_results: list[dict[str, Any]] = []
    for req in state["evidence_plan"]["evidence_requirements"]:
        rid = str(req.get("requirement_id", ""))
        req_type = str(req.get("requirement_type", ""))
        company = str(req.get("company") or "")
        if req_type == "numeric":
            metric = str(req.get("metric", ""))
            value = numeric_values.get((company, metric))
            if value is None:
                collection_results.append({"requirement_id": rid, "status": "missing", "evidence_type": "numeric", "items": []})
                continue
            row = {
                "requirement_id": rid,
                "ticker": company,
                "metric": metric,
                "period_type": "annual",
                "period_end": "2025-12-31",
                "value": value,
                "unit": "USD",
                "source_provider": "sec_companyfacts",
                "confidence": "high",
                "extraction_method": "companyfacts",
            }
            tool_results.append({"tool": "query_financial_data", "ticker": company, "requirement_id": rid, "data": {"ticker": company, "financial_facts": [row], "price_data": []}})
            collection_results.append({"requirement_id": rid, "status": "satisfied", "evidence_type": "numeric", "items": [row]})
            continue
        if req_type == "calculation":
            metric = str(req.get("metric") or "")
            if "valuation_evidence_missing" in {str(item) for item in req.get("fallback_strategy", []) or []}:
                collection_results.append({"requirement_id": rid, "status": "missing", "evidence_type": "calculation", "items": [], "failure_reason": "valuation_evidence_missing"})
                continue
            if metric == "net_margin":
                margin = numeric_values[("NVDA", "net_income")] / numeric_values[("NVDA", "revenue")]
                item = {
                    "requirement_id": rid,
                    "company": "NVDA",
                    "ticker": "NVDA",
                    "metric": "net_margin",
                    "computation": "margin",
                    "source_tag": "net_income_over_revenue",
                    "data": {"results": [{"period": "2025-12-31", "period_type": "annual", "margin": margin, "margin_pct": f"{margin * 100:.2f}%"}]},
                }
                tool_results.append({"tool": "compute_metrics", **item})
                collection_results.append({"requirement_id": rid, "status": "satisfied", "evidence_type": "calculation", "items": [item]})
                continue
            collection_results.append({"requirement_id": rid, "status": "missing", "evidence_type": "calculation", "items": []})
            continue
        if req_type != "text":
            continue
        snippets = {
            "REQ-TEXT-NVDA-BUSINESS_MODEL": (
                "ITEM_1",
                "NVIDIA provides accelerated computing platforms, products, services, and solutions for gaming, data center, professional visualization, and automotive markets.",
            ),
            "REQ-TEXT-NVDA-RISK": (
                "ITEM_1A",
                "Risk factors include intense competition, changes in demand, and supply chain constraints across markets.",
            ),
            "REQ-TEXT-NVDA-COMPETITION": (
                "ITEM_1",
                "NVIDIA competes in markets for GPUs, accelerated computing products, customers, and platforms.",
            ),
        }
        if rid not in snippets:
            collection_results.append({"requirement_id": rid, "status": "missing", "evidence_type": "text", "items": [], "raw_hit_count": 0, "usable_hit_count": 0})
            continue
        section, snippet = snippets[rid]
        doc = {
            "requirement_id": rid,
            "dimension_id": str(req.get("dimension_id", "")),
            "retrieval_intent": str(req.get("retrieval_intent", "")),
            "ticker": "NVDA",
            "filing_id": "nvda-10k",
            "form_type": "10-K",
            "fiscal_period": "2025",
            "section": section,
            "chunk_order": 1,
            "quality": "high",
            "text": snippet,
            "supporting_snippet": snippet,
            "supporting_terms": ["gpu", "data", "center", "gaming", "competition", "demand", "supply"],
        }
        retrieved_docs.append(doc)
        collection_results.append({"requirement_id": rid, "status": "satisfied", "evidence_type": "text", "items": [doc], "raw_hit_count": 1, "section_filtered_hit_count": 1, "usable_hit_count": 1, "snippet_support_passed_count": 1})

    state["tool_results"] = tool_results
    state["retrieved_docs"] = retrieved_docs
    state["evidence_collection_results"] = collection_results
    state["evidence_sufficiency"] = evaluate_evidence_sufficiency(state["evidence_plan"], collection_results).model_dump(exclude_none=True)

    _patch_llms(monkeypatch, text_response='{"text_claims":[]}', draft_response="{}")
    out = generate_agent_answer(state)

    assert out["synthesis_mode"] == "methodology_single_company"
    assert out["validated_text_evidence_count"] >= 2
    assert out["dimension_status_map"]["business_model"]["status"] == "satisfied"
    assert out["dimension_status_map"]["moat_and_competitive_risk"]["status"] in {"satisfied", "partial"}
    assert any(item.get("claim_source") == "deterministic_fallback" for item in out["text_evidence"])
    assert all(item.get("dimension_id") for item in out["text_evidence"])
    assert "业务模式" in out["final_answer"]
    assert "数据中心" in out["final_answer"] or "GPU" in out["final_answer"]
    assert "风险" in out["final_answer"]
    assert "当前缺少风险文本证据" not in out["final_answer"]
    assert "accelerated computing platforms, products, services" not in out["final_answer"]


def test_live_path_comparison_falls_back_to_decision_frame_when_draft_stays_invalid(monkeypatch):
    state = _base_state()
    _patch_llms(
        monkeypatch,
        text_response=(
            '{"text_claims":[{"sentence":"Apple discussed margin discipline.","evidence_ids":["T1"]},'
            '{"sentence":"Amazon discussed revenue scale and reinvestment.","evidence_ids":["T2"]}]}'
        ),
        draft_response=[
            '{"tentative_conclusion":{"statement":"你应该买入 AAPL。","stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2"]},"decision_basis":[{"statement":"AAPL 当前净利润更高。","citation_refs":["N2","N6"]}],"supporting_points":[],"counterpoints":[],"risk_tradeoffs":[],"uncertainty_notes":[],"citation_refs":["N2","N6"],"safety_notes":[]}',
            '{"tentative_conclusion":{"statement":"AAPL 一定更好。","stance":"leans_toward_company","preferred_company":"AAPL","citation_refs":["N2"]},"decision_basis":[{"statement":"AAPL 当前净利润更高。","citation_refs":["N2","N6"]}],"supporting_points":[],"counterpoints":[],"risk_tradeoffs":[],"uncertainty_notes":[],"citation_refs":["N2","N6"],"safety_notes":[]}',
        ],
    )

    out = generate_agent_answer(state)

    assert out["draft_final_status"] == "fallback"
    assert len(out["draft_attempts"]) == 2
    assert len(out["draft_revision_attempts"]) == 1
    assert out["final_answer_source"] == "comparison_decision_fallback"
    assert out["comparison_judgment_frame"]["preferred_company"] == "AAPL"
    assert out["comparison_judgment_frame"]["margin_winner"] == "AAPL"
    assert "买入" not in out["final_answer"]
    assert "更偏向 AAPL" in out["final_answer"] or "偏向 AAPL" in out["final_answer"]
    assert "反方观点" in out["final_answer"]


def test_live_path_prediction_refusal_skips_analyst_draft():
    state = build_classification_state(
        user_query="预测明天苹果股价",
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="trace-live-refusal",
        today=date(2026, 4, 23),
    )
    state["user_query"] = "预测明天苹果股价"

    out = generate_agent_answer(state)

    assert out["output"]["answer_mode"] == "refusal_or_redirect"
    assert out["final_answer_source"] == "unsupported_or_refusal"
    assert out.get("analyst_draft", {}) == {}
    assert out.get("draft_attempts", []) == []
