"""Runtime AnswerContract guard tests."""

from __future__ import annotations

import re

from src.agent.answer_contract import check_answer_contract, check_answer_evidence_contract
from src.agent.answering import (
    _rewrite_fcf_causal_answer_if_needed,
    _rewrite_profit_decline_premise_if_needed,
    _rewrite_risk_comparison_answer_if_needed,
    _rewrite_valuation_boundary_contradiction,
)
from src.agent.driver_evidence import apply_profit_decline_summary_neutralization
from src.agent.nodes import (
    build_contract_debt_business_fallback,
    contract_check_node,
    finalize_node,
    generate_answer,
    prepare_contract_evidence_retry_node,
    repair_generate_node,
    relevance_repair_node,
    route_after_contract,
    route_after_relevance,
    safe_blocked_answer_node,
)


def _state() -> dict:
    return {
        "user_query": "分析 NVDA",
        "final_answer": "NVDA 的现金流质量很好。",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "missing_requirements": ["REQ-CASH"],
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 100.0, "unit": "USD"}],
            "text_snippets": [{"evidence_id": "T1", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk"}],
            "dimension_status_map": {"cash_flow_quality": {"status": "partial", "supporting_evidence_ids": ["N1"]}},
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }


def test_runtime_contract_blocks_invalid_citation_without_breaking_post_hoc_contract():
    state = _state()
    answer = "市值为 $100.00。[N999]"

    result = check_answer_contract(answer, state)
    post_hoc = check_answer_evidence_contract({**state, "final_answer": answer})

    assert result.route == "blocked"
    assert result.severity == "hard_fail"
    assert any(item.code == "invalid_citation" for item in result.violations)
    assert {"passed", "violations", "metrics"} <= set(post_hoc)


def test_runtime_contract_blocks_invented_number():
    state = _state()
    answer = "NVDA 的收入为 $999.00B。[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert result.decision == "blocked"
    assert any(item.code == "unsupported_numeric" for item in result.violations)


def test_runtime_contract_accepts_negative_currency_expressed_with_text_cue():
    answer = "AMZN 的自由现金流为负 $2.47B，FCF margin 为 -1.36%。[N1][N2]"
    state = {
        "user_query": "分析 AMZN",
        "final_answer": answer,
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "missing_requirements": [],
        "requirement_status_map": {},
        "evidence_packet": {
            "numeric_table": [
                {
                    "evidence_id": "N1",
                    "ticker": "AMZN",
                    "metric": "free_cash_flow",
                    "value": -2472000000,
                    "unit": "USD",
                    "display_value": "$-2.47B",
                },
                {
                    "evidence_id": "N2",
                    "ticker": "AMZN",
                    "metric": "fcf_margin",
                    "value": -0.013618,
                    "unit": "ratio",
                    "display_value": "-1.36%",
                },
            ],
            "text_snippets": [],
            "dimension_status_map": {
                "cash_flow_quality": {
                    "status": "satisfied",
                    "supporting_evidence_ids": ["N1", "N2"],
                }
            },
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }

    result = check_answer_contract(answer, state)

    assert result.route == "pass"
    assert result.decision == "passed"
    assert not any(item.code == "unsupported_numeric" for item in result.violations)


def test_runtime_contract_accepts_numbers_grounded_in_text_evidence():
    answer = "数据中心网络营收增长 142%；网络营收达 82 亿美元，同比增长 162%。[T1][T2]"
    state = {
        "user_query": "为什么 NVIDIA 的营收增长这么多",
        "final_answer": answer,
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "missing_requirements": [],
        "evidence_packet": {
            "numeric_table": [],
            "text_snippets": [
                {
                    "evidence_id": "T1",
                    "ticker": "NVDA",
                    "section": "ITEM_7",
                    "supporting_snippet": "Revenue from Data Center networking grew 142% driven by the introduction of NVLink compute fabric.",
                },
                {
                    "evidence_id": "T2",
                    "ticker": "NVDA",
                    "section": "ITEM_2",
                    "supporting_snippet": "Networking revenue was $8.2 billion, up 162% from a year ago from the continued growth of NVLink compute fabric.",
                },
            ],
            "dimension_status_map": {},
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }

    result = check_answer_contract(answer, state)

    assert result.route == "pass"
    assert not any(item.code == "unsupported_numeric" for item in result.violations)


def _scope_state(scope: str = "segment") -> dict:
    return {
        "user_query": "为什么 NVIDIA 的营收增长这么多",
        "final_answer": "",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "missing_requirements": [],
        "evidence_packet": {
            "numeric_table": [],
            "text_snippets": [
                {
                    "evidence_id": "T1",
                    "ticker": "NVDA",
                    "section": "ITEM_7",
                    "claim_scope": scope,
                    "driver_level": "segment_level_driver" if scope == "segment" else "product_level_driver",
                    "allowed_claim_strength": "bounded_inference",
                    "scope_reason": "Snippet discusses segment/product revenue, not consolidated company revenue.",
                    "supporting_snippet": "Compute & Networking revenue increase was driven by accelerated computing and AI.",
                }
            ],
            "dimension_status_map": {},
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }


def test_runtime_contract_repairs_segment_evidence_overstated_as_company_driver():
    answer = "NVIDIA 总营收增长主要由 AI 加速计算平台驱动。[T1]"
    result = check_answer_contract(answer, _scope_state("segment"))

    assert result.route == "repair_answer"
    assert any(item.code == "segment_evidence_overstated_as_company_driver" for item in result.violations)
    assert result.scope_overclaim_check["status"] == "repairable"
    assert result.scope_overclaim_violations[0]["affected_citations"] == ["T1"]


def test_runtime_contract_allows_bounded_segment_driver_wording():
    answer = "分部层面显示，Compute & Networking 增长与 AI 加速计算相关，可能参与总营收增长线索，但不能完整代表总公司原因。[T1]"
    result = check_answer_contract(answer, _scope_state("segment"))

    assert result.route == "pass"
    assert not any(item.code == "segment_evidence_overstated_as_company_driver" for item in result.violations)


def test_runtime_contract_repairs_product_evidence_overstated_as_company_driver():
    answer = "NVDA 的营收增长主要由 XDR InfiniBand 和 NVLink 驱动。[T1]"
    result = check_answer_contract(answer, _scope_state("product"))

    assert result.route == "repair_answer"
    assert any(item.code == "segment_evidence_overstated_as_company_driver" for item in result.violations)


def test_runtime_contract_accepts_negative_currency_with_minus_before_dollar():
    answer = "AMZN 的自由现金流为 -$2.47B，说明资本开支后自由现金流为负。[N1]"
    state = {
        "user_query": "分析 AMZN",
        "final_answer": answer,
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "missing_requirements": [],
        "evidence_packet": {
            "numeric_table": [
                {
                    "evidence_id": "N1",
                    "ticker": "AMZN",
                    "metric": "free_cash_flow",
                    "value": -2472000000,
                    "unit": "USD",
                    "display_value": "-$2.47B",
                },
            ],
            "text_snippets": [],
            "dimension_status_map": {
                "cash_flow_quality": {
                    "status": "satisfied",
                    "supporting_evidence_ids": ["N1"],
                }
            },
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }

    result = check_answer_contract(answer, state)

    assert result.route == "pass"
    assert not any(item.code == "unsupported_numeric" for item in result.violations)


def test_post_hoc_contract_allows_bounded_comparison_without_bilateral_risk_text():
    answer = (
        "不能在当前证据边界内强行判断 AMZN 和 NVDA 谁的风险更大。"
        "证据限制包括风险因素文本证据不足，因此只能说明比较口径和限制。"
    )
    trace = {
        "final_answer": answer,
        "task_type": "company_comparison",
        "analysis_scope": "comparison",
        "companies": ["AMZN", "NVDA"],
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AMZN", "metric": "revenue", "value": 1.0},
                {"evidence_id": "N2", "ticker": "NVDA", "metric": "revenue", "value": 2.0},
            ],
            "text_snippets": [],
            "dimension_status_map": {},
        },
        "output": {},
    }

    result = check_answer_evidence_contract(trace)

    assert result["passed"]
    assert not any(item.get("type") == "comparison_balance" for item in result["violations"])


def test_runtime_contract_blocks_buy_sell_advice():
    state = _state()
    answer = "NVDA 现在应该买。[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert result.decision == "blocked"
    assert any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_blocks_advice_after_disclaimer():
    state = _state()
    answer = "这不是投资建议；但我建议买入 NVDA。[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_blocks_english_advice_after_disclaimer():
    state = _state()
    answer = "This is not investment advice, but NVDA should buy now.[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_blocks_target_price_after_negated_phrase():
    state = _state()
    answer = "这不是目标价，但目标价格为 100 美元。[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_routes_partial_dimension_to_repair():
    state = _state()
    answer = "NVDA 的现金流质量很好。[N1]"

    result = check_answer_contract(answer, {**state, "final_answer": answer})

    assert result.route == "repair_answer"
    assert any(item.code == "dimension_status_violation" for item in result.violations)


def test_optional_context_gap_passes_with_warning_not_repair():
    answer = "NVDA 的主要风险来自竞争压力和需求波动。[T1]"
    state = {
        **_state(),
        "draft_answer": answer,
        "final_answer": answer,
        "missing_requirements": ["REQ-TEXT-NVDA-RISK_MDA"],
        "missing_required_requirements": [],
        "missing_required_requirements_count": 0,
        "missing_optional_requirements_count": 1,
        "requirement_status_map": {
            "REQ-TEXT-NVDA-RISK_FACTORS": {
                "status": "satisfied",
                "required": True,
                "requirement_scope": "core",
            },
            "REQ-TEXT-NVDA-RISK_MDA": {
                "status": "missing",
                "required": False,
                "requirement_scope": "optional_context",
            },
        },
        "trace_summary": {
            "missing_required_requirements_count": 0,
            "missing_optional_requirements_count": 1,
        },
        "evidence_packet": {
            "numeric_table": [],
            "text_snippets": [
                {
                    "evidence_id": "T1",
                    "ticker": "NVDA",
                    "dimension_id": "moat_and_competitive_risk",
                    "supporting_snippet": "Competition and demand risks are discussed in risk factors.",
                }
            ],
            "dimension_status_map": {
                "moat_and_competitive_risk": {
                    "status": "satisfied",
                    "supporting_evidence_ids": ["T1"],
                    "required_missing": [],
                    "enhanced_missing": [],
                }
            },
        },
        "output": {"limitations": [], "view": {"kind": "risk_focused_analysis_brief"}},
    }

    result = check_answer_contract(answer, state)
    checked = contract_check_node(state)

    assert result.route == "pass"
    assert result.decision == "warning"
    assert result.blocking_missing_requirements == []
    assert not result.violations
    assert any(item.code == "missing_growth_quantification_caveat" for item in result.warnings)
    assert checked["contract_status"] == "passed_with_warnings"
    assert checked["contract_result"]["decision"] == "warning"


def test_runtime_contract_recognizes_visible_growth_and_segment_caveats():
    answer = "证据边界：当前无法可靠计算总营收增长率；这些证据属于分部/产品层面，不能完整代表总公司营收增长原因。"
    state = {
        "final_answer": answer,
        "trace_summary": {"missing_optional_requirements_count": 1},
        "evidence_packet": {
            "numeric_table": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "confidence": "medium"}],
            "text_snippets": [],
            "dimension_status_map": {},
        },
        "output": {"limitations": []},
    }

    result = check_answer_contract(answer, state)

    assert result.decision == "passed"
    assert not result.warnings


def test_repair_generate_routes_contract_debt_without_becoming_author():
    state = _state()
    checked = contract_check_node(state)
    repaired = repair_generate_node({**state, **checked})

    assert repaired["contract_attempts"] == 1
    assert "[N2]" not in repaired["draft_answer"]
    assert "123" not in repaired["draft_answer"]
    assert "证据边界" in repaired["draft_answer"]
    assert repaired["primary_generation_contract_debt"] is True
    assert repaired["repair_owner"] == "contract_repair"
    assert repaired["final_answer_source"] != "contract_repair"
    assert "contract repair" not in repaired["draft_answer"]
    assert "candidate layer" not in repaired["draft_answer"]


def test_repair_generate_does_not_rewrite_scope_overclaim_as_analysis():
    state = {
        **_scope_state("segment"),
        "draft_answer": "NVIDIA 总营收增长主要由 AI 加速计算平台驱动。[T1]",
    }
    checked = contract_check_node(state)
    repaired = repair_generate_node({**state, **checked})

    assert repaired["primary_generation_contract_debt"] is True
    assert repaired["repair_owner"] == "contract_repair"
    assert repaired["final_answer_source"] != "contract_repair"
    assert "contract repair" not in repaired["draft_answer"]
    assert "candidate layer" not in repaired["draft_answer"]
    assert "证据边界" in repaired["draft_answer"]


def test_contract_debt_segment_fallback_is_business_answer(monkeypatch):
    class PassResult:
        route = "pass"
        decision = "passed"
        public_summary = "passed"
        violations = []

        @staticmethod
        def model_dump():
            return {"route": "pass", "decision": "passed", "violations": []}

    monkeypatch.setattr("src.agent.nodes.check_answer_contract", lambda *_args, **_kwargs: PassResult())
    state = {
        "user_query": "为什么 NVDA 的网络业务增长这么快？",
        "final_answer": "当前主答案存在较多未引用的重要陈述。",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "companies": ["NVDA"],
        "canonical_intent": {"segment_or_product_scope": "Networking"},
        "analysis_plan": {"segment_or_product_scope": "Networking"},
        "evidence_packet": {
            "canonical_source": True,
            "companies": ["NVDA"],
            "text_snippets": [
                {
                    "evidence_id": "T1",
                    "ticker": "NVDA",
                    "dimension_id": "business_model",
                    "claim": "Networking revenue growth was supported by InfiniBand, Ethernet and NVLink demand.",
                }
            ],
            "numeric_table": [],
            "dimension_status_map": {},
        },
        "contract_result": {
            "route": "repair_answer",
            "violations": [
                {"code": "citation_free_material_claim", "answer_span": "a"},
                {"code": "citation_free_material_claim", "answer_span": "b"},
                {"code": "citation_free_material_claim", "answer_span": "c"},
            ],
        },
        "output": {"limitations": []},
    }

    repaired = repair_generate_node(state)

    assert repaired["primary_generation_contract_debt"] is True
    assert repaired["final_answer_source"] == "bounded_segment_product_driver_candidate"
    assert "合理推断" in repaired["draft_answer"]
    assert "网络业务增长大概率与 AI 集群建设" in repaired["draft_answer"]
    assert "[T1]" in repaired["draft_answer"]
    assert "总公司级营收增长的完整因果" in repaired["draft_answer"]
    assert "重新生成候选答案" not in repaired["draft_answer"]


def test_contract_debt_segment_fallback_reads_trace_ui_evidence_shape():
    state = {
        "user_query": "NVDA 的营收增长主要是数据中心驱动的吗？",
        "answer_mode": "analytical",
        "task_type": "report_summary",
        "companies": [{"TICKER": "NVDA"}],
        "canonical_intent": {"segment_or_product_scope": "data center"},
        "research_plan_used": {"question_type": "causal_explanation"},
        "answer_part_status_by_id": {
            "quantify_growth": {"status": "partial"},
            "identify_growth_drivers": {"status": "partial"},
        },
        "evidence_packet": {
            "text_evidence": [
                {
                    "evidence_id": "T1",
                    "ticker": "NVDA",
                    "section": "ITEM_7",
                    "claim": "NVDA 的披露文本提供了可用于比较的业务和风险背景。",
                    "supporting_snippet": "Compute & Networking revenue increased 57%, driven by demand for accelerated computing and networking.",
                }
            ],
            "numeric_evidence": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 68127000000.0}],
        },
    }

    answer, owner = build_contract_debt_business_fallback(
        state,
        {"route": "repair_answer", "violations": [{"code": "citation_free_material_claim"}] * 3},
        "zh",
    )

    assert owner == "bounded_segment_product_driver_candidate"
    assert "合理推断" in answer
    assert "NVDA" in answer
    assert "Compute & Networking" in answer
    assert "bounded_scope_limit" not in owner


def test_contract_debt_overview_fallback_precedes_risk(monkeypatch):
    class PassResult:
        route = "pass"
        decision = "passed"
        public_summary = "passed"
        violations = []

        @staticmethod
        def model_dump():
            return {"route": "pass", "decision": "passed", "violations": []}

    monkeypatch.setattr("src.agent.nodes.check_answer_contract", lambda *_args, **_kwargs: PassResult())
    state = {
        "user_query": "AAPL overview",
        "final_answer": "bad",
        "task_type": "report_summary",
        "answer_mode": "risk_focused_analysis",
        "analysis_scope": "single_company",
        "companies": ["AAPL"],
        "canonical_intent": {"intent_family": "overview"},
        "evidence_policy_id": "single_company_overview_v1",
        "evidence_packet": {
            "canonical_source": True,
            "companies": ["AAPL"],
            "intent_family": "overview",
            "evidence_policy_id": "single_company_overview_v1",
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "AAPL", "dimension_id": "business_model", "claim": "AAPL sells products and services."},
                {"evidence_id": "T2", "ticker": "AAPL", "dimension_id": "moat_and_competitive_risk", "claim": "AAPL faces competitive and regulatory risks."},
            ],
            "numeric_table": [{"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "display_value": "$100B"}],
            "dimension_status_map": {},
        },
        "contract_result": {
            "route": "repair_answer",
            "violations": [
                {"code": "citation_free_material_claim", "answer_span": "a"},
                {"code": "citation_free_material_claim", "answer_span": "b"},
                {"code": "citation_free_material_claim", "answer_span": "c"},
            ],
        },
        "output": {"limitations": []},
    }

    repaired = repair_generate_node(state)

    assert repaired["final_answer_source"] == "bounded_overview_candidate"
    assert "结论" in repaired["draft_answer"]
    assert "单一风险" in repaired["draft_answer"]


def test_bounded_risk_analysis_ranks_amzn_business_risks():
    state = {
        "user_query": "Amazon 的主要风险是什么，按重要性排序",
        "answer_mode": "risk_focused_analysis",
        "task_type": "report_summary",
        "companies": ["AMZN"],
        "evidence_packet": {
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "AMZN", "section": "ITEM_1A", "dimension_id": "moat_and_competitive_risk", "claim": "Amazon faces fulfillment, logistics, inventory, capital expenditure and supply chain risks."},
                {"evidence_id": "T2", "ticker": "AMZN", "section": "ITEM_1A", "dimension_id": "moat_and_competitive_risk", "claim": "Amazon faces regulatory, legal and compliance risks across jurisdictions."},
                {"evidence_id": "T3", "ticker": "AMZN", "section": "ITEM_1A", "dimension_id": "moat_and_competitive_risk", "claim": "Amazon faces AWS cloud competition and competitive pricing pressure."},
            ],
            "numeric_table": [],
        },
    }

    answer, owner = build_contract_debt_business_fallback(state, {"route": "repair_answer", "violations": []}, "zh")

    assert owner == "bounded_risk_analysis"
    assert "结论" in answer
    assert "已验证风险文本" in answer
    assert "基于业务模型的风险排序" in answer
    assert "履约/库存/资本开支压力" in answer
    assert "监管/合规" in answer
    assert "AWS/云竞争" in answer
    assert "无法可靠排序" not in answer
    assert "只能列出边界" not in answer


def test_bounded_scenario_risk_uses_economic_slowdown_transmission_path():
    state = {
        "user_query": "如果下季度经济放缓，MSFT 最大的财务风险是什么？",
        "answer_mode": "risk_focused_analysis",
        "task_type": "report_summary",
        "companies": ["MSFT"],
        "evidence_packet": {
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "MSFT", "section": "ITEM_1A", "dimension_id": "moat_and_competitive_risk", "claim": "Microsoft discloses customer spending, cloud demand slowdown and supply chain risks."},
            ],
            "numeric_table": [],
        },
    }

    answer, owner = build_contract_debt_business_fallback(state, {"route": "repair_answer", "violations": []}, "zh")

    assert owner == "bounded_scenario_risk_candidate"
    assert "客户 IT/云支出放缓 -> 收入增速压力 -> 毛利率/营业利润率压力 -> 经营现金流和 FCF 受影响" in answer
    assert "供应链只作为已披露风险线索之一" in answer
    assert "无法可靠排序" not in answer
    assert "只能列出边界" not in answer


def test_one_sentence_valuation_boundary_is_directional_without_duplicate_advice_boundary():
    state = {
        **_state(),
        "user_query": "请用一句话回答：NVDA 估值贵不贵？",
        "companies": ["NVDA"],
        "safety_intent": "investment_advice_like",
        "canonical_intent": {"requested_dimensions": ["valuation_and_risk_boundary"]},
        "contract_result": {"route": "blocked", "decision": "blocked", "violations": [{"code": "forbidden_claim"}]},
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "NVDA", "metric": "pe_ratio", "value": 60.0},
                {"evidence_id": "N2", "ticker": "NVDA", "metric": "ps_ratio", "value": 25.0},
                {"evidence_id": "N3", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.01},
            ],
            "text_snippets": [],
        },
    }

    out = safe_blocked_answer_node(state)

    sentence_count = len([part for part in re.split(r"[。！？!?]+|\n+", out["final_answer"]) if part.strip()])
    assert sentence_count == 1
    assert "估值风险偏高" in out["final_answer"]
    assert out["final_answer"].count("买卖建议") == 1
    assert not any(term in out["final_answer"] for term in ("建议买入", "建议卖出", "建议持有", "目标价"))


def test_unsupported_benchmark_repair_uses_public_replacement(monkeypatch):
    class PassResult:
        route = "pass"
        decision = "passed"
        public_summary = "passed"
        violations = []

        @staticmethod
        def model_dump():
            return {"route": "pass", "decision": "passed", "violations": []}

    monkeypatch.setattr("src.agent.nodes.check_answer_contract", lambda *_args, **_kwargs: PassResult())
    answer = "NVDA 估值显著高于行业。[N1]"
    state = {
        **_state(),
        "final_answer": answer,
        "draft_answer": answer,
        "contract_result": check_answer_contract(answer, {**_state(), "final_answer": answer}).model_dump(),
    }

    repaired = repair_generate_node(state)

    assert "估值倍数较高，但缺少历史/行业基准，不能严格判断是否处于高位" in repaired["draft_answer"]
    assert "Rewrite only" not in repaired["draft_answer"]
    assert "repair instruction" not in repaired["draft_answer"].lower()


def test_finalize_scrubs_internal_public_answer_terms():
    answer = "Rewrite only the affected wording and keep all facts inside existing evidence."
    out = finalize_node(
        {
            "user_query": "分析 NVDA",
            "draft_answer": answer,
            "final_answer": answer,
            "final_answer_source": "test_source",
            "contract_result": {"route": "pass", "decision": "passed", "violations": []},
            "output": {"limitations": []},
        }
    )

    assert "Rewrite only" not in out["final_answer"]


def test_manual_probe_summary_marks_targeted_runs(tmp_path):
    from eval.run_manual_api_probe_20 import _write_summary

    summary = _write_summary(
        [{"case_id": "Q09", "query": "NVDA 的估值贵不贵？"}],
        tmp_path / "summary.json",
        "2026-05-27T00:00:00Z",
        "http://127.0.0.1:8080",
        1.0,
        "test-run",
        is_full_probe=False,
    )

    assert summary["selected_case_ids"] == ["Q09"]
    assert summary["selected_case_count"] == 1
    assert summary["full_case_count"] == 20
    assert summary["is_full_probe"] is False


def test_route_after_contract_paths():
    pass_state = {"contract_result": {"route": "pass"}, "contract_attempts": 0}
    repair_state = {"contract_result": {"route": "repair_answer"}, "contract_attempts": 0, "max_contract_repairs": 2}
    action_repair_state = {"contract_result": {"route": "blocked", "action": "strip_sentence"}, "contract_attempts": 0, "max_contract_repairs": 2}
    action_retry_state = {
        "contract_result": {"route": "blocked", "action": "retry_evidence", "missing_requirements": ["REQ-CASH"]},
        "contract_evidence_retry_count": 0,
        "max_contract_evidence_retries": 2,
    }
    missing_state = {
        "contract_result": {"route": "need_more_evidence", "missing_requirements": ["REQ-CASH"]},
        "contract_evidence_retry_count": 0,
        "max_contract_evidence_retries": 2,
    }
    business_fallback_state = {
        "primary_generation_contract_debt": True,
        "final_answer_source": "bounded_segment_product_driver_candidate",
        "contract_result": {"route": "repair_answer", "violations": [{"code": "citation_free_material_claim"}]},
        "contract_attempts": 2,
        "max_contract_repairs": 2,
    }
    blocked_state = {"contract_result": {"route": "blocked"}}

    assert route_after_contract(pass_state) == "relevance_check"
    assert route_after_contract(repair_state) == "repair_generate"
    assert route_after_contract(action_repair_state) == "repair_generate"
    assert route_after_contract(missing_state) == "prepare_contract_evidence_retry"
    assert route_after_contract(action_retry_state) == "prepare_contract_evidence_retry"
    assert route_after_contract(business_fallback_state) == "relevance_check"
    assert route_after_contract(blocked_state) == "blocked"


def test_route_after_relevance_uses_action():
    repair_state = {"relevance_decision": {"route": "finalize", "action": "downgrade_to_bounded"}, "relevance_attempts": 0}
    exhausted_state = {"relevance_decision": {"route": "finalize", "action": "downgrade_to_bounded"}, "relevance_attempts": 1}
    pass_state = {"relevance_decision": {"route": "finalize", "action": "pass"}, "relevance_attempts": 0}
    block_state = {"relevance_decision": {"route": "finalize", "action": "block"}, "relevance_attempts": 0}

    assert route_after_relevance(repair_state) == "relevance_repair"
    assert route_after_relevance(exhausted_state) == "blocked"
    assert route_after_relevance(pass_state) == "finalize"
    assert route_after_relevance(block_state) == "blocked"


def test_relevance_repair_uses_bounded_analysis_with_citable_evidence():
    state = {
        **_state(),
        "user_query": "为什么 NVIDIA 的营收增长这么多？",
        "draft_answer": "NVIDIA revenue increased significantly [N1].",
        "final_answer": "NVIDIA revenue increased significantly [N1].",
        "canonical_intent": {"requested_dimensions": ["revenue_quality"]},
        "relevance_decision": {
            "route": "repair_answer",
            "action": "downgrade_to_bounded",
            "public_summary": "driver missing",
            "deterministic_relevance_failures": [{"code": "driver_explanation_missing"}],
        },
        "answer_candidates": [],
    }

    out = relevance_repair_node(state)

    assert out["final_answer_source"] == "bounded_analysis"
    assert out["answer_quality_tier"] == "bounded_analysis"
    assert out["main_question_covered"] is True
    assert "待验证假设" in out["final_answer"]
    assert "Blackwell" not in out["final_answer"]
    assert "合理推断" in out["final_answer"]
    assert out["answer_history"][-1]["new_owner"] == "bounded_analysis"


def test_relevance_repair_risk_comparison_uses_bounded_answer_not_scope_limit():
    state = {
        **_state(),
        "user_query": "AMZN 和 NVDA 哪个更危险？",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AMZN", "NVDA"],
        "draft_answer": "证据边界：当前答案未可靠覆盖竞争与风险。",
        "final_answer": "证据边界：当前答案未可靠覆盖竞争与风险。",
        "canonical_intent": {
            "intent_family": "comparison",
            "requested_dimensions": ["moat_and_competitive_risk"],
        },
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [],
            "text_snippets": [],
            "dimension_status_map": {},
        },
        "relevance_decision": {
            "route": "repair_answer",
            "action": "scope_limit",
            "deterministic_relevance_failures": [{"code": "comparison_risk_text_dimension_missing"}],
        },
        "answer_candidates": [],
    }

    out = relevance_repair_node(state)

    assert out["final_answer_source"] == "bounded_risk_comparison_answer"
    assert out["answer_quality_tier"] == "bounded_analysis"
    assert out["main_question_covered"] is True
    assert out["answered_dimensions"] == ["moat_and_competitive_risk"]
    assert "AMZN" in out["final_answer"]
    assert "NVDA" in out["final_answer"]
    assert "不能" in out["final_answer"] and "比较" in out["final_answer"]
    assert "相关性修复" not in out["final_answer"]
    assert "scope limit" not in out["final_answer"].lower()


def test_relevance_repair_selects_existing_bounded_candidate():
    state = {
        **_state(),
        "draft_answer": "旧答案。",
        "final_answer": "旧答案。",
        "canonical_intent": {"requested_dimensions": ["valuation_and_risk_boundary"]},
        "relevance_decision": {
            "route": "repair_answer",
            "action": "downgrade_to_bounded",
            "deterministic_relevance_failures": [{"code": "valuation_dimension_missing"}],
        },
        "answer_candidates": [
            {
                "body": "已有有边界估值答案。[N1]",
                "owner": "bounded_valuation_postprocess",
                "requested_dimensions": ["valuation_and_risk_boundary"],
                "evidence_refs": ["N1"],
                "allowed_repairs": ["downgrade_to_bounded"],
            }
        ],
    }

    out = relevance_repair_node(state)

    assert out["final_answer"] == "已有有边界估值答案。[N1]"
    assert out["final_answer_source"] == "bounded_valuation_postprocess"
    assert out["answer_history"][-1]["transform"] == "relevance_select_existing_bounded_candidate"


def test_safe_blocked_answer_replaces_draft_and_finalize_exposes_contract_summary():
    state = {
        **_state(),
        "draft_answer": "NVDA 估值便宜，应该买。",
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-CASH",
                    "requirement_type": "numeric",
                    "dimension_id": "cash_flow_quality",
                    "dimension_name": "Cash Flow Quality",
                }
            ]
        },
        "contract_result": {
            "route": "blocked",
            "public_summary": "blocked summary",
            "missing_requirements": ["REQ-CASH"],
            "violations": [{"code": "forbidden_claim"}],
        },
    }

    blocked = safe_blocked_answer_node(state)
    final = finalize_node({**state, **blocked})

    assert "应该买" not in final["final_answer"]
    assert "现金流质量数值证据" in final["final_answer"]
    assert "REQ-" not in final["final_answer"]
    assert final["contract_status"] == "blocked"
    assert final["output"]["contract"]["status"] == "blocked"
    assert final["output"]["contract"]["public_summary"] == "blocked summary"


def test_safe_blocked_answer_does_not_label_optional_missing_as_missing_core():
    state = {
        **_state(),
        "draft_answer": "AMZN 的自由现金流为 $2.47B。",
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-OPTIONAL-MDA",
                    "requirement_type": "text",
                    "dimension_id": "profitability_quality",
                    "dimension_name": "Profitability Quality",
                    "requirement_scope": "optional_context",
                    "required": False,
                }
            ]
        },
        "contract_result": {
            "route": "blocked",
            "public_summary": "blocked summary",
            "missing_requirements": ["REQ-OPTIONAL-MDA"],
            "blocking_missing_requirements": [],
            "violations": [{"code": "unsupported_numeric"}],
        },
    }

    blocked = safe_blocked_answer_node(state)

    assert "缺失的信息包括" not in blocked["final_answer"]
    assert "盈利质量文本证据" not in blocked["final_answer"]
    assert "未通过契约校验" in blocked["final_answer"]


def test_clarification_short_circuit_has_no_unsupported_claims_and_passes_contract():
    state = {
        **_state(),
        "user_query": "这个股票能买吗？",
        "answer_mode": "clarification",
        "safety_intent": "investment_advice_like",
        "needs_tools": False,
        "needs_clarification": True,
        "clarification_question": "请补充 ticker/公司名。",
    }

    out = generate_answer(state)
    result = check_answer_contract(out["final_answer"], {**state, **out}, scope="answer")

    assert out["unsupported_claims"] == []
    assert result.route == "pass"
    assert "不提供买入、卖出或持有建议" in out["final_answer"]
    assert "不构成投资建议" in out["final_answer"]
    assert "风险、估值和证据边界" in out["final_answer"]


def test_safe_blocked_investment_boundary_releases_warning_answer():
    state = {
        **_state(),
        "user_query": "AMZN 现在值得买吗？",
        "companies": ["AMZN"],
        "answer_mode": "analytical",
        "safety_intent": "investment_advice_like",
        "draft_answer": "AMZN 估值便宜，应该买。",
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-VALUATION",
                    "requirement_type": "numeric",
                    "dimension_id": "valuation_and_risk_boundary",
                    "dimension_name": "Valuation Boundary",
                }
            ]
        },
        "contract_result": {
            "route": "blocked",
            "decision": "blocked",
            "public_summary": "blocked summary",
            "missing_requirements": ["REQ-VALUATION"],
            "violations": [{"code": "forbidden_claim"}],
        },
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AMZN", "metric": "pe_ratio"},
                {"evidence_id": "N2", "ticker": "AMZN", "metric": "ps_ratio"},
                {"evidence_id": "N3", "ticker": "AMZN", "metric": "fcf_yield"},
            ],
            "text_snippets": [],
            "dimension_status_map": {"valuation_and_risk_boundary": {"status": "partial", "supporting_evidence_ids": ["N1", "N2", "N3"]}},
        },
    }

    boundary = safe_blocked_answer_node(state)
    final = finalize_node({**state, **boundary})

    assert final["contract_status"] == "passed_with_warnings"
    assert "AMZN" in final["final_answer"]
    assert "不能给买卖建议" in final["final_answer"]
    assert final["final_answer"].count("买卖建议") == 1
    assert final["answer_quality_tier"] == "bounded_analysis"
    assert final["main_question_covered"] is True
    assert "估值风险偏高" in final["final_answer"]
    assert "同业/历史基准" in final["final_answer"]
    assert "应该买" not in final["final_answer"]


def test_safe_blocked_revenue_boundary_does_not_use_worth_buying_template():
    state = {
        **_state(),
        "user_query": "NVDA 的营收增长主要是数据中心驱动的吗？",
        "companies": ["NVDA"],
        "task_type": "fact_qa",
        "methodology_intent": "revenue_quality_analysis",
        "primary_dimension": "revenue_quality",
        "requested_dimensions": ["revenue_quality"],
        "safety_intent": "normal",
        "contract_result": {
            "route": "blocked",
            "decision": "blocked",
            "missing_requirements": ["REQ-REV"],
            "violations": [{"code": "citation_free_material_claim"}],
        },
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue"}],
            "text_snippets": [],
        },
    }

    boundary = safe_blocked_answer_node(state)
    final = finalize_node({**state, **boundary})

    assert final["answer_quality_tier"] == "bounded_analysis"
    assert "营收" in final["final_answer"]
    assert "值得买" not in final["final_answer"]
    assert final["fallback_intent_match"] is True


def test_safe_blocked_valuation_intent_releases_bounded_valuation_answer():
    state = {
        **_state(),
        "user_query": "NVDA 的估值贵不贵？",
        "companies": ["NVDA"],
        "canonical_intent": {
            "intent_family": "valuation",
            "requested_dimensions": ["valuation_and_risk_boundary"],
            "legacy_methodology_intent": "valuation_boundary_analysis",
        },
        "contract_result": {
            "route": "blocked",
            "decision": "blocked",
            "violations": [{"code": "unsupported_numeric"}],
        },
        "evidence_packet": {
            "numeric_evidence": [
                {"evidence_id": "N26", "ticker": "NVDA", "metric": "pe_ratio", "value": 121.75},
                {"evidence_id": "N27", "ticker": "NVDA", "metric": "ps_ratio", "value": 76.77},
                {"evidence_id": "N28", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.0067},
                {"evidence_id": "N22", "ticker": "NVDA", "metric": "revenue", "value": 68127000000.0},
                {"evidence_id": "N23", "ticker": "NVDA", "metric": "net_income", "value": 42960000000.0},
            ],
            "text_evidence": [],
        },
        "output": {"limitations": []},
    }

    out = safe_blocked_answer_node(state)

    assert out["final_answer_source"] == "valuation_bounded_answer"
    assert out["answer_quality_tier"] == "bounded_analysis"
    assert out["main_question_covered"] is True
    assert out["answered_dimensions"] == ["valuation_and_risk_boundary"]
    assert "估值风险偏高" in out["final_answer"]
    assert "同业/历史基准" in out["final_answer"]
    assert "[N26]" in out["final_answer"] and "[N27]" in out["final_answer"] and "[N28]" in out["final_answer"]
    assert "safe_blocked_answer" != out["final_answer_source"]


def test_safe_blocked_overview_intent_precedes_risk_boundary():
    state = {
        **_state(),
        "user_query": "AAPL overview",
        "companies": ["AAPL"],
        "answer_mode": "analytical",
        "canonical_intent": {
            "intent_family": "overview",
            "legacy_methodology_intent": "single_company_overview",
        },
        "contract_result": {
            "route": "blocked",
            "decision": "blocked",
            "violations": [{"code": "unsupported_numeric"}],
        },
        "evidence_packet": {
            "numeric_evidence": [
                {"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "value": 111184000000.0},
                {"evidence_id": "N5", "ticker": "AAPL", "metric": "net_income", "value": 29578000000.0},
                {"evidence_id": "N44", "ticker": "AAPL", "metric": "net_margin", "value": 0.2928},
                {"evidence_id": "N57", "ticker": "AAPL", "metric": "pe_ratio", "value": 145.65},
            ],
            "computed_metrics": [
                {"evidence_id": "N50", "ticker": "AAPL", "metric": "fcf_margin", "value": 0.2404},
            ],
            "text_evidence": [
                {"evidence_id": "T1", "ticker": "AAPL", "dimension_id": "business_model", "claim": "AAPL operates in competitive product and service markets."},
                {"evidence_id": "T6", "ticker": "AAPL", "dimension_id": "moat_and_competitive_risk", "claim": "AAPL faces supply chain risks."},
            ],
        },
        "output": {"limitations": []},
    }

    out = safe_blocked_answer_node(state)

    assert out["final_answer_source"] == "overview_bounded_answer"
    assert out["answer_quality_tier"] == "bounded_analysis"
    assert out["main_question_covered"] is True
    assert "结论" in out["final_answer"]
    assert "业务定位" in out["final_answer"]
    assert "收入和盈利" in out["final_answer"]
    assert "现金流与估值" in out["final_answer"]
    assert "主要风险" in out["final_answer"]
    assert "收入：" not in out["final_answer"]
    assert "net_income" not in out["final_answer"]
    assert "market_cap" not in out["final_answer"]
    assert "A bounded risk answer is releasable" not in out["final_answer"]
    assert out["answered_dimensions"] != ["moat_and_competitive_risk"]


def test_scope_limit_quality_tier_is_not_pass():
    state = {
        **_state(),
        "user_query": "AMZN 和 NVDA 哪个更危险？",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AMZN"],
        "comparison_target": "NVDA",
        "requested_dimensions": ["moat_and_competitive_risk"],
        "primary_dimension": "moat_and_competitive_risk",
        "draft_answer": "AMZN 和 NVDA 哪个更危险目前无法判断。",
        "contract_status": "passed",
        "contract_result": {"route": "pass", "decision": "passed", "violations": []},
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [],
            "text_snippets": [],
        },
    }

    final = finalize_node(state)

    assert final["contract_status"] == "scope_limited"
    assert final["final_route"] == "scope_limited"
    assert final["answer_quality_tier"] == "scope_limit"


def test_quality_tier_not_derived_from_source_name_only():
    state = {
        **_state(),
        "draft_answer": "NVDA 的收入为 100 USD。[N1]",
        "final_answer_source": "bounded_legacy_source_name",
        "contract_status": "passed",
        "contract_result": {"route": "pass", "decision": "passed", "violations": []},
        "missing_required_answer_parts": [],
        "partial_required_answer_parts": [],
    }

    final = finalize_node(state)

    assert final["answer_quality_tier"] == "true_answer"
    assert final["quality_tier_reason"] == "core_question_answered_without_substantive_failures"


def test_one_sentence_relevance_repair_compresses_without_existing_multisentence_candidate():
    state = {
        **_state(),
        "user_query": "请用一句话回答：NVDA 是不是可以买？",
        "safety_intent": "investment_advice_like",
        "format_constraints": {"one_sentence": True, "max_sentences": 1},
        "final_answer": "第一句。[N1]\n第二句。",
        "answer_candidates": [
            {
                "body": "第一段。[N1]\n第二段。",
                "owner": "bounded_fallback",
                "requested_dimensions": ["valuation_and_risk_boundary"],
                "allowed_repairs": ["downgrade_to_bounded"],
            }
        ],
        "relevance_decision": {
            "route": "repair_answer",
            "action": "scope_limit",
            "deterministic_relevance_failures": [{"code": "one_sentence_constraint_violated"}],
        },
    }

    repaired = relevance_repair_node(state)

    assert repaired["final_answer_source"] == "format_one_sentence_repair"
    assert repaired["format_constraints_satisfied"] is True
    assert "\n" not in repaired["final_answer"]


def test_unsupported_benchmark_claim_requires_benchmark_specific_evidence():
    state = {
        **_state(),
        "final_answer": "NVDA 的 PE 显著高于行业均值。[N1]",
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "pe_ratio", "value": 40.0}],
            "text_snippets": [],
        },
    }

    result = check_answer_contract(state["final_answer"], state)

    assert result.route == "repair_answer"
    assert any(item.code == "unsupported_benchmark_claim" for item in result.violations)


def test_risk_ranking_template_only_item_violates_contract_metrics():
    state = {
        **_state(),
        "final_answer": "核心风险排序\n- 最高优先级：供应链或库存压力。[T1]",
        "synthesis": {
            "risk_focused_answer": {
                "risk_ranking": [
                    {
                        "theme_name": "供应链或库存压力",
                        "evidence_refs": ["T1"],
                        "mechanism_support_level": "template_only",
                    }
                ],
                "template_only_risk_items": [
                    {"theme_name": "供应链或库存压力", "mechanism_support_level": "template_only"}
                ],
            }
        },
    }

    result = check_answer_contract(state["final_answer"], state)

    assert any(item.code == "template_only_core_risk" for item in result.violations)
    assert result.metrics["risk_items_template_only_count"] >= 1
    assert result.metrics["risk_items_directly_supported_count"] == 0


def test_rewrite_valuation_boundary_keeps_partial_multiples_without_missing_evidence_contradiction():
    answer = (
        "PE比率32.31x，PS比率16.16x，FCF收益率-0.08%。"
        "当前估值证据不足，无法判断AMZN是否值得购买。"
        "缺少估值证据，无法判断价格是否便宜或昂贵。"
        "缺少估值证据，不能判断价格是否便宜或昂贵。"
        "\n已有部分估值倍数（P/E、P/S、FCF yield）可作为有限估值边界观察。"
        "\n已有部分估值倍数（P/E、P/S、FCF yield）可作为有限估值边界观察。"
    )
    rewritten = _rewrite_valuation_boundary_contradiction(
        answer,
        [
            {"evidence_id": "N1", "metric": "pe_ratio"},
            {"evidence_id": "N2", "metric": "ps_ratio"},
            {"evidence_id": "N3", "metric": "fcf_yield"},
        ],
        "zh",
    )

    assert "估值风险偏高" in rewritten
    assert rewritten.count("不能给买卖建议") == 1
    assert "估值证据不足" not in rewritten
    assert "缺少估值证据" not in rewritten
    assert "同业/历史基准" in rewritten


def test_profit_decline_false_premise_direct_answer():
    state = {
        **_state(),
        "user_query": "为什么amazon的利润下降了",
        "task_type": "fact_qa",
        "answer_mode": "direct_fact",
        "analysis_scope": "",
        "companies": ["AMZN"],
        "period_query": {"period_type": "latest"},
        "resolved_period_context": {},
        "tool_results": [
            {
                "tool": "query_financial_data",
                "ticker": "AMZN",
                "data": {
                    "ticker": "AMZN",
                    "financial_facts": [
                        {
                            "requirement_id": "REQ-NI-HISTORY",
                            "ticker": "AMZN",
                            "metric": "net_income",
                            "period_type": "quarterly",
                            "period_end": "2025-12-31",
                            "value": 10.0,
                            "unit": "USD",
                        },
                        {
                            "requirement_id": "REQ-NI-HISTORY",
                            "ticker": "AMZN",
                            "metric": "net_income",
                            "period_type": "quarterly",
                            "period_end": "2026-03-31",
                            "value": 20.0,
                            "unit": "USD",
                        },
                    ],
                },
            }
        ],
        "retrieved_docs": [
            {
                "ticker": "AMZN",
                "filing_id": "f1",
                "form_type": "10-Q",
                "fiscal_period": "2026Q1",
                "section": "ITEM_7",
                "chunk_order": 1,
                "supporting_snippet": "Revenue recognition and income variability can change across periods.",
                "claim": "Amazon 的利润下降可能源于收入确认口径变化。",
                "retrieval_query": "为什么amazon的利润下降了",
            }
        ],
    }

    out = generate_answer(state)

    assert out["final_answer_source"] == "deterministic_false_premise"
    assert "不支持“利润下降”这个前提" in out["final_answer"]
    assert "证据边界" in out["final_answer"]
    assert "利润下降原因" not in out["final_answer"]
    assert "可能源于" not in out["final_answer"]


def test_profit_decline_text_summary_neutralized():
    row = apply_profit_decline_summary_neutralization(
        {
            "claim": "Amazon 的利润下降可能源于收入确认口径变化。",
            "supporting_snippet": "Revenue recognition and income variability can change across periods.",
            "retrieval_query": "为什么amazon的利润下降了",
        },
        user_query="为什么amazon的利润下降了",
    )

    assert row["claim"] == "披露文本涉及收入/收益波动或收入确认口径，可作为利润波动相关背景，但不能证明本期利润下降原因。"
    assert "利润下降可能" not in row["claim"]


def test_rewrite_risk_comparison_answers_who_has_greater_risk_with_boundary():
    rewritten = _rewrite_risk_comparison_answer_if_needed(
        "如果只基于当前财报证据，我更偏向 AMZN。",
        state={
            "user_query": "比较一下 AMZN 和 NVDA 谁的风险更大",
            "task_type": "company_comparison",
            "answer_mode": "comparison_brief",
            "companies": ["AMZN"],
            "comparison_target": "NVDA",
        },
        synthesis_payload={
            "comparison_judgment_frame": {
                "companies": ["AMZN", "NVDA"],
                "risk_considerations": [
                    {"companies": ["AMZN"], "label": "Competition", "evidence_refs": ["T1"]},
                    {"companies": ["NVDA"], "label": "Demand / Macro Pressure", "evidence_refs": ["T2"]},
                ],
            }
        },
        user_query="比较一下 AMZN 和 NVDA 谁的风险更大",
        lang="zh",
    )

    assert "证据不足以强行判断 AMZN 和 NVDA 哪个更危险或谁的风险更大" in rewritten
    assert "AMZN:" in rewritten
    assert "NVDA:" in rewritten
    assert "比较边界" in rewritten


def test_safe_blocked_riskier_synonym_releases_bounded_comparison():
    state = {
        **_state(),
        "user_query": "AMZN 和 NVDA 哪个更危险？",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AMZN", "NVDA"],
        "safety_intent": "normal",
        "contract_result": {
            "route": "blocked",
            "decision": "blocked",
            "missing_requirements": ["REQ-RISK-AMZN", "REQ-RISK-NVDA"],
            "violations": [{"code": "comparison_balance"}],
        },
        "text_evidence": [
            {
                "evidence_id": "T1",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "claim": "AMZN faces demand and macro pressure risk.",
            },
            {
                "evidence_id": "T2",
                "ticker": "NVDA",
                "dimension_id": "moat_and_competitive_risk",
                "claim": "NVDA faces supply and customer concentration risk.",
            },
        ],
        "numeric_evidence": [
            {"evidence_id": "N1", "ticker": "AMZN", "metric": "revenue", "value": 100.0},
            {"evidence_id": "N2", "ticker": "NVDA", "metric": "revenue", "value": 80.0},
        ],
    }

    boundary = safe_blocked_answer_node(state)
    final = finalize_node({**state, **boundary})

    assert final["final_route"] == "released_with_warnings"
    assert final["final_contract_status"] == "passed_with_warnings"
    assert "风险比较" in final["final_answer"]
    assert "AMZN" in final["final_answer"] and "NVDA" in final["final_answer"]
    assert "哪个更危险" in final["final_answer"]
    assert "证据不足以强行判断" in final["final_answer"]
    assert "不构成投资建议" in final["final_answer"]


def test_profit_decline_premise_rewrite_blocks_false_decline_explanation():
    rewritten = _rewrite_profit_decline_premise_if_needed(
        "利润下降原因是宏观压力和成本上升。",
        user_query="为什么amazon的利润下降了",
        numeric_evidence=[
            {
                "evidence_id": "N1",
                "metric": "net_income",
                "period_end": "2025-12-31",
                "value": 10.0,
                "display_value": "$10.00B",
            },
            {
                "evidence_id": "N2",
                "metric": "net_income",
                "period_end": "2026-03-31",
                "value": 20.0,
                "display_value": "$20.00B",
            },
        ],
        lang="zh",
    )

    assert "不支持“利润下降”这个前提" in rewritten
    assert "最新净利润" in rewritten
    assert "高于上一期" in rewritten
    assert "利润下降原因是" not in rewritten


def test_rewrite_fcf_causal_adds_worsening_path_and_hypotheses():
    rewritten = _rewrite_fcf_causal_answer_if_needed(
        "经营现金流为正，自由现金流为负，资本开支较高。",
        user_query="为什么 Amazon 的自由现金流变差？",
        numeric_evidence=[
            {"evidence_id": "N1", "metric": "operating_cash_flow"},
            {"evidence_id": "N2", "metric": "free_cash_flow"},
            {"evidence_id": "N3", "metric": "capital_expenditure"},
        ],
        lang="zh",
    )

    assert "自由现金流变差/承压" in rewritten
    assert "资本开支" in rewritten
    assert "待验证假设" in rewritten
    assert "[N1][N2][N3]" in rewritten


def test_contract_summary_labels_pure_insufficiency_as_safe_response():
    answer = "财报综合摘要\n证据不足：当前证据无法支持可追溯结论。"
    state = {
        **_state(),
        "draft_answer": answer,
        "final_answer": answer,
        "numeric_evidence": [],
        "text_evidence": [],
        "citations": [],
        "evidence_packet": {"numeric_table": [], "text_snippets": [], "dimension_status_map": {}},
        "output": {"numeric_evidence": [], "text_evidence": [], "limitations": []},
    }

    checked = contract_check_node(state)
    final = finalize_node({**state, **checked})

    assert checked["contract_status"] == "passed"
    assert checked["contract_public_summary"] == "Safe insufficient-evidence response returned."
    assert checked["contract_result"]["public_summary"] == "Safe insufficient-evidence response returned."
    assert final["output"]["contract"]["public_summary"] == "Safe insufficient-evidence response returned."


def test_contract_summary_labels_partial_grounded_response(monkeypatch):
    class PassResult:
        route = "pass"
        public_summary = "Answer passed runtime grounding checks."
        violations = []

        @staticmethod
        def model_dump():
            return {"route": "pass", "public_summary": "Answer passed runtime grounding checks.", "violations": []}

    monkeypatch.setattr("src.agent.nodes.check_answer_contract", lambda *_args, **_kwargs: PassResult())
    state = {
        **_state(),
        "draft_answer": "NVDA 经营现金流为 $36.19B。[N1]",
        "final_answer": "NVDA 经营现金流为 $36.19B。[N1]",
        "numeric_evidence": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "operating_cash_flow"}],
        "text_evidence": [],
        "citations": [{"evidence_id": "N1"}],
        "missing_requirements": ["REQ-TEXT-NVDA-RISK"],
        "dimension_status_map": {
            "cash_flow_quality": {"status": "partial"},
            "moat_and_competitive_risk": {"status": "missing"},
        },
        "output": {"numeric_evidence": [{"evidence_id": "N1"}], "text_evidence": [], "limitations": []},
    }

    checked = contract_check_node(state)

    assert checked["contract_status"] == "passed"
    assert checked["contract_public_summary"] == "Partial grounded answer returned; some requested evidence remains unavailable."
    assert checked["contract_result"]["public_summary"] == "Partial grounded answer returned; some requested evidence remains unavailable."


def test_contract_summary_ignores_optional_only_missing_when_dimensions_satisfied(monkeypatch):
    class PassResult:
        route = "pass"
        public_summary = "Answer passed runtime grounding checks."
        violations = []

        @staticmethod
        def model_dump():
            return {"route": "pass", "public_summary": "Answer passed runtime grounding checks.", "violations": []}

    monkeypatch.setattr("src.agent.nodes.check_answer_contract", lambda *_args, **_kwargs: PassResult())
    state = {
        **_state(),
        "draft_answer": "NVDA 经营现金流为 $36.19B。[N1]",
        "final_answer": "NVDA 经营现金流为 $36.19B。[N1]",
        "numeric_evidence": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "operating_cash_flow"}],
        "text_evidence": [],
        "citations": [{"evidence_id": "N1"}],
        "missing_requirements": ["REQ-COMPUTED-FCF"],
        "missing_required_requirements_count": 0,
        "missing_optional_requirements_count": 1,
        "requirement_status_map": {
            "REQ-COMPUTED-FCF": {"status": "missing", "required": False},
        },
        "dimension_status_map": {
            "cash_flow_quality": {"status": "satisfied"},
        },
        "evidence_packet": {
            "numeric_table": [{"evidence_id": "N1", "ticker": "NVDA", "metric": "operating_cash_flow"}],
            "text_snippets": [],
            "dimension_status_map": {"cash_flow_quality": {"status": "satisfied"}},
        },
        "output": {"numeric_evidence": [{"evidence_id": "N1"}], "text_evidence": [], "limitations": []},
    }

    checked = contract_check_node(state)

    assert checked["contract_status"] == "passed"
    assert checked["contract_public_summary"] == "Answer passed runtime grounding checks."
    assert checked["contract_result"]["public_summary"] == "Answer passed runtime grounding checks."


def test_prepare_contract_evidence_retry_records_bounded_retry():
    state = {
        **_state(),
        "contract_result": {
            "route": "need_more_evidence",
            "missing_requirements": ["REQ-CASH"],
            "violations": [{"code": "dimension_status_violation"}],
        },
    }

    out = prepare_contract_evidence_retry_node(state)

    assert out["contract_evidence_retry_count"] == 1
    assert out["evidence_sufficient"] is False
    assert out["evidence_retry_history"][0]["source"] == "runtime_answer_contract"


def test_runtime_contract_allows_cited_analytical_judgment_language():
    answer = "NVDA 现金流质量较强，估值压力较高，风险优先级较高；这些都是基于当前证据的分析性判断。[C1][V1][T1]"
    state = {
        "user_query": "分析 NVIDIA 的现金流质量、估值边界和主要风险。",
        "draft_answer": answer,
        "final_answer": answer,
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "C1", "ticker": "NVDA", "metric": "operating_cash_flow", "value": 36_188_000_000, "unit": "USD"},
                {"evidence_id": "V1", "ticker": "NVDA", "metric": "pe_ratio", "value": 68.4, "unit": "ratio"},
            ],
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk"},
            ],
            "dimension_status_map": {
                "cash_flow_quality": {"status": "satisfied", "supporting_evidence_ids": ["C1"]},
                "valuation_and_risk_boundary": {"status": "satisfied", "supporting_evidence_ids": ["V1"]},
                "moat_and_competitive_risk": {"status": "satisfied", "supporting_evidence_ids": ["T1"]},
            },
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }

    result = check_answer_contract(answer, state)

    assert result.route == "pass"
    assert result.decision in {"passed", "warning"}
    assert not any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_blocks_hold_advice_wording():
    state = _state()
    answer = "NVDA 建议持有。[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_blocks_target_price_wording():
    state = _state()
    answer = "NVDA 目标价格为 $100.00。[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_blocks_deterministic_next_quarter_prediction():
    state = _state()
    answer = "NVDA 下季度利润一定增长。[N1]"

    result = check_answer_contract(answer, state)

    assert result.route == "blocked"
    assert any(item.code == "forbidden_claim" for item in result.violations)


def test_runtime_contract_allows_extreme_margin_when_caveat_is_visible():
    answer = "AMZN 净利率为 50.02%，盈利质量需要结合口径核验。[N1]"
    state = {
        **_state(),
        "draft_answer": answer,
        "final_answer": answer,
        "missing_requirements": [],
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AMZN", "metric": "net_margin", "value": 0.5002, "unit": "ratio", "display_value": "50.02%"}
            ],
            "text_snippets": [],
            "dimension_status_map": {"profitability_quality": {"status": "satisfied", "supporting_evidence_ids": ["N1"]}},
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }

    result = check_answer_contract(answer, state)

    assert result.route == "pass"
    assert result.decision in {"passed", "warning"}
    assert not any(item.code == "unsupported_numeric" for item in result.violations)


def test_finalize_promotes_draft_with_answer_history_owner():
    state = {
        **_state(),
        "draft_answer": "有边界的最终答案。[N1]",
        "final_answer": "旧的最终答案。",
        "final_answer_source": "contract_repair",
        "contract_result": {"route": "pass", "decision": "passed", "violations": []},
    }

    out = finalize_node(state)

    assert out["final_answer"] == "有边界的最终答案。[N1]"
    assert out["final_answer_source"] == "contract_repair"
    assert out["answer_history"][-1]["transform"] == "finalize_promote_draft"
    assert out["answer_history"][-1]["new_owner"] == "contract_repair"
    assert out["answer_candidate"]["body"] == "有边界的最终答案。[N1]"


def test_safe_boundary_risk_comparison_uses_canonical_packet_not_raw_text():
    state = {
        **_state(),
        "user_query": "AMZN 和 NVDA 哪个更危险？",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AMZN", "NVDA"],
        "text_evidence": [
            {
                "evidence_id": "T9",
                "ticker": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "claim": "RAW TEXT SHOULD NOT APPEAR",
            }
        ],
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [],
            "text_snippets": [
                {
                    "evidence_id": "T1",
                    "ticker": "AMZN",
                    "dimension_id": "moat_and_competitive_risk",
                    "claim": "AMZN packet risk.",
                },
                {
                    "evidence_id": "T2",
                    "ticker": "NVDA",
                    "dimension_id": "moat_and_competitive_risk",
                    "claim": "NVDA packet risk.",
                },
            ],
            "dimension_status_map": {"moat_and_competitive_risk": {"status": "partial"}},
        },
        "contract_result": {
            "route": "blocked",
            "decision": "blocked",
            "missing_requirements": [],
            "violations": [{"code": "comparison_balance"}],
        },
    }

    boundary = safe_blocked_answer_node(state)
    final = finalize_node({**state, **boundary})

    assert final["final_route"] == "released_with_warnings"
    assert "AMZN packet risk" in final["final_answer"]
    assert "NVDA packet risk" in final["final_answer"]
    assert "RAW TEXT SHOULD NOT APPEAR" not in final["final_answer"]
    assert final["answer_history"][-1]["new_owner"] == "safe_boundary_bounded_answer"


def test_finalize_preserves_bounded_risk_comparison_without_citable_text():
    answer = (
        "风险比较结论\n"
        "证据不足以强行判断 AMZN 和 NVDA 哪个更危险；当前只能做有边界的风险比较。\n"
        "AMZN 已验证风险线索\n"
        "- 当前缺少足够的单独风险文本证据。\n"
        "NVDA 已验证风险线索\n"
        "- 当前缺少足够的单独风险文本证据。"
    )
    state = {
        **_state(),
        "user_query": "AMZN 和 NVDA 哪个更危险？",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "companies": ["AMZN", "NVDA"],
        "draft_answer": answer,
        "final_answer": answer,
        "final_answer_source": "bounded_risk_comparison_answer",
        "final_route": "bounded_fallback",
        "answer_quality_tier": "bounded_analysis",
        "main_question_covered": True,
        "fallback_intent_match": True,
        "answered_dimensions": ["moat_and_competitive_risk"],
        "evidence_packet": {
            "canonical_source": True,
            "packet_kind": "canonical_validated_evidence_packet",
            "numeric_table": [],
            "text_snippets": [],
            "dimension_status_map": {},
        },
        "contract_result": {"route": "pass", "decision": "passed", "violations": []},
    }

    final = finalize_node(state)

    assert final["final_answer_source"] == "bounded_risk_comparison_answer"
    assert final["answer_quality_tier"] == "bounded_analysis"
    assert final["main_question_covered"] is True
    assert final["final_route"] == "bounded_fallback"
    assert "相关性修复" not in final["final_answer"]
