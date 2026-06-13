from __future__ import annotations

from eval.run_financial_agent_regression import judge_case


def _base_trace(answer: str, *, family: str = "overview") -> dict:
    plan_question_type = "causal_explanation" if family == "causal" else "overview"
    return {
        "trace_id": "trace-test",
        "final_answer": answer,
        "companies": ["AMZN"] if family != "causal" else ["NVDA"],
        "research_plan_used": {"question_type": plan_question_type, "required_answer_parts": [{"id": "overview", "required": True}]},
        "plan_coverage_decision": {
            "strategy": "merge" if family != "causal" else "replace",
            "legacy_core_count": 18,
            "research_core_count": 3,
            "retained_legacy_core_count": 18 if family != "causal" else 0,
        },
        "requirement_merge_summary": {"strategy": "merge"},
        "evidence_validation_records": [
            {
                "requirement_id": "REQ-REV",
                "evidence_type": "numeric",
                "tool_returned_count": 2,
                "validated_evidence_count": 2,
                "rejected_evidence_reason": "",
                "status": "satisfied",
            }
        ],
        "dimension_status_by_id": {
            "business_model": {},
            "revenue_quality": {},
            "profitability_quality": {},
            "cash_flow_quality": {},
            "balance_sheet_and_capital_intensity": {},
            "moat_and_competitive_risk": {},
            "valuation_and_risk_boundary": {},
        },
        "unsupported_claims": [],
        "contract_status": "passed",
        "contract_decision": {"decision": "passed"},
    }


def test_overview_fail_when_research_plan_drops_legacy_core():
    case = {
        "case_id": "Q1",
        "family": "overview",
        "companies": ["AMZN"],
        "required_dimensions": ["business_model"],
        "required_answer_terms": [["business"], ["revenue"]],
        "forbidden_answer_terms": [],
    }
    trace = _base_trace("Amazon business and revenue overview.")
    trace["plan_coverage_decision"]["strategy"] = "replace"
    trace["plan_coverage_decision"]["retained_legacy_core_count"] = 0

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "overview_not_merged" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_numeric_returned_zero_validated_without_reason_is_p1_failure():
    case = {
        "case_id": "Q1",
        "family": "overview",
        "companies": ["AMZN"],
        "required_dimensions": ["business_model"],
        "required_answer_terms": [["business"]],
        "forbidden_answer_terms": [],
    }
    trace = _base_trace("Amazon business overview with revenue.")
    trace["evidence_validation_records"][0]["validated_evidence_count"] = 0
    trace["evidence_validation_records"][0]["rejected_evidence_reason"] = ""

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "returned_numeric_rejected_without_reason" for issue in judgment["issues"])


def test_final_route_blocked_is_p0_failure():
    case = {
        "case_id": "Q12",
        "family": "comparison",
        "companies": ["AMZN", "NVDA"],
        "required_answer_terms": [["AMZN"], ["NVDA"], ["风险"], ["比较"], ["边界"]],
        "forbidden_answer_terms": [],
    }
    trace = _base_trace("不能在当前证据边界内强行判断 AMZN 和 NVDA 谁的风险更大。")
    trace["companies"] = ["AMZN", "NVDA"]
    trace["final_route"] = "blocked"
    trace["final_contract_status"] = "passed_with_warnings"

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "final_route_blocked" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_contract_failed_status_is_p0_failure():
    case = {
        "case_id": "Q1",
        "family": "overview",
        "companies": ["AMZN"],
        "required_dimensions": ["business_model"],
        "required_answer_terms": [["business"]],
        "forbidden_answer_terms": [],
    }
    trace = _base_trace("Amazon business overview with revenue.")
    trace["final_contract_status"] = "failed"
    trace["contract_result"] = {"decision": "failed"}

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "contract_blocked" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_template_refusal_non_safety_query_is_p0_failure():
    case = {
        "case_id": "Q12",
        "family": "comparison",
        "companies": ["AMZN", "NVDA"],
        "required_answer_terms": [["AMZN"], ["NVDA"], ["风险"], ["比较"], ["边界"]],
        "forbidden_answer_terms": [],
    }
    trace = _base_trace("目前证据不足以支持一个完整且通过契约校验的结论。")
    trace["companies"] = ["AMZN", "NVDA"]

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "template_refusal" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_investment_query_fails_on_buy_advice():
    case = {
        "case_id": "Q10",
        "family": "investment_boundary",
        "companies": ["AMZN"],
        "required_answer_terms": [["投资建议"]],
        "forbidden_answer_terms": ["建议买入"],
    }
    trace = _base_trace("这是投资建议：建议买入 AMZN。")

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "investment_advice" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_q10_valuation_metrics_plus_missing_valuation_evidence_is_p0():
    case = {
        "case_id": "Q10",
        "family": "investment_boundary",
        "companies": ["AMZN"],
        "required_answer_terms": [["不能"], ["投资建议"], ["风险", "估值", "证据"], ["边界"]],
        "forbidden_answer_terms": [],
        "accept_warning_codes": ["investment_boundary_redirect"],
    }
    answer = (
        "PE比率32.31x，PS比率16.16x，FCF收益率-0.08%。"
        "但当前估值证据不足，缺少估值证据，无法判断价格是否便宜或昂贵。"
        "这不是投资建议。"
    )
    trace = _base_trace(answer)
    trace["companies"] = ["AMZN"]

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "valuation_contradiction" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_investment_boundary_allows_negated_target_price_language():
    case = {
        "case_id": "Q10",
        "family": "investment_boundary",
        "companies": ["AMZN"],
        "required_answer_terms": [["不能"], ["买入", "卖出", "持有"], ["投资建议"], ["风险", "估值", "证据"], ["边界"]],
        "forbidden_answer_terms": ["建议买入", "可以买", "建议卖出", "建议持有", "目标价"],
        "accept_warning_codes": ["investment_boundary_redirect"],
    }
    answer = (
        "不能判断 AMZN 现在是否值得买，也不是投资建议。我不提供买入、卖出或持有建议，也不提供目标价。"
        "这里转为风险、估值和证据边界分析。"
    )
    trace = _base_trace(answer)
    trace["companies"] = ["AMZN"]

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "pass"
    assert not any(issue["code"] in {"investment_advice", "question_not_answered"} for issue in judgment["issues"])


def test_q12_risk_comparison_must_answer_or_bounded_compare():
    case = {
        "case_id": "Q12",
        "family": "comparison",
        "companies": ["AMZN", "NVDA"],
        "required_answer_terms": [["AMZN"], ["NVDA"], ["风险"], ["比较"], ["边界"]],
        "forbidden_answer_terms": [],
    }
    answer = "如果只基于当前财报证据，我更偏向 AMZN。AMZN 在收入规模上占优，NVDA 仍值得观察。"
    trace = _base_trace(answer)
    trace["companies"] = ["AMZN", "NVDA"]

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "comparison_not_answered" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_q12_generic_preference_with_risk_boundary_is_not_enough():
    case = {
        "case_id": "Q12",
        "family": "comparison",
        "companies": ["AMZN", "NVDA"],
        "required_answer_terms": [["AMZN"], ["NVDA"], ["风险"], ["比较"], ["边界"]],
        "forbidden_answer_terms": [],
    }
    answer = (
        "如果只基于当前财报证据，我更偏向 AMZN。"
        "竞争与风险：AMZN 和 NVDA 都有已验证文本证据，结论需要结合风险与竞争边界。"
    )
    trace = _base_trace(answer)
    trace["companies"] = ["AMZN", "NVDA"]

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "comparison_not_answered" and issue["severity"] == "P0" for issue in judgment["issues"])


def test_unlisted_relevance_p1_is_not_accepted_warning():
    case = {
        "case_id": "Q10",
        "family": "investment_boundary",
        "companies": ["AMZN"],
        "required_answer_terms": [["买卖", "买入", "卖出", "持有"]],
        "forbidden_answer_terms": [],
        "accept_warning_codes": ["investment_boundary_redirect"],
    }
    trace = _base_trace("这不是投资建议。这里转为风险、估值和证据边界分析。")
    trace["companies"] = ["AMZN"]

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "answer_missing_required_content" and issue["severity"] == "P1" for issue in judgment["issues"])


def test_contract_caveat_warning_can_be_accepted():
    case = {
        "case_id": "Q1",
        "family": "overview",
        "companies": ["AMZN"],
        "required_dimensions": ["business_model"],
        "required_answer_terms": [["business"]],
        "forbidden_answer_terms": [],
    }
    trace = _base_trace("Amazon business overview with revenue.")
    contract = {
        "passed": False,
        "violations": [
            {
                "type": "missing_reconciliation_caveat",
                "message": "Caveat not visible in final answer: reconciliation_warning",
            }
        ],
    }

    judgment = judge_case(case, trace, contract)

    assert judgment["status"] == "accepted_warning"
    assert any(issue["code"] == "missing_reconciliation_caveat" and issue["severity"] == "P1" for issue in judgment["issues"])


def test_causal_requires_tiered_structure():
    case = {
        "case_id": "Q3",
        "family": "causal",
        "companies": ["NVDA"],
        "required_answer_terms": [["营收"], ["增长"]],
        "forbidden_answer_terms": [],
    }
    trace = _base_trace("NVDA 营收增长来自需求。", family="causal")

    judgment = judge_case(case, trace, {"passed": True, "violations": []})

    assert judgment["status"] == "fail"
    assert any(issue["code"] == "causal_structure_missing" and issue["severity"] == "P0" for issue in judgment["issues"])
