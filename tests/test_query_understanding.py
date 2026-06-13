import inspect

from src.agent import entity_resolution
from src.agent.methodology_intent import classify_methodology_intent
from src.agent.query_understanding import build_query_understanding, query_understanding_summary


def _tickers(query: str) -> list[str]:
    return [company.ticker for company in build_query_understanding(query).companies]


def test_query_understanding_acceptance_cases():
    overview = build_query_understanding("分析下 nvidai")
    assert [company.ticker for company in overview.companies] == ["NVDA"]
    assert overview.analysis_scope == "single_company"
    assert overview.methodology_intent == "overview"
    assert not overview.needs_clarification

    risk = build_query_understanding("nvidai 最大的问题是什么")
    assert [company.ticker for company in risk.companies] == ["NVDA"]
    assert risk.analysis_scope == "single_company"
    assert risk.methodology_intent == "risk"

    cash_flow = build_query_understanding("苹果现金流怎么样")
    assert [company.ticker for company in cash_flow.companies] == ["AAPL"]
    assert cash_flow.methodology_intent == "cash_flow"

    valuation = build_query_understanding("NVIDIA 估值贵不贵")
    assert [company.ticker for company in valuation.companies] == ["NVDA"]
    assert valuation.methodology_intent == "valuation"

    comparison = build_query_understanding("apple 和 amazon 更看好哪个")
    assert [company.ticker for company in comparison.companies] == ["AAPL", "AMZN"]
    assert comparison.analysis_scope == "comparison"
    assert comparison.methodology_intent == "comparison"
    assert comparison.safety_intent == "investment_advice_like"


def test_query_understanding_extracts_explicit_composite_dimensions():
    understanding = build_query_understanding("分析英伟达的现金流质量、估值边界和主要风险")

    assert [company.ticker for company in understanding.companies] == ["NVDA"]
    assert understanding.analysis_scope == "single_company"
    assert understanding.methodology_intent == "cash_flow"
    assert understanding.requested_dimensions == [
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    ]


def test_semantic_equivalent_queries_preserve_dimensions():
    expected = [
        "cash_flow_quality",
        "valuation_and_risk_boundary",
        "moat_and_competitive_risk",
    ]

    english = build_query_understanding("Analyze NVIDIA's cash flow quality, valuation boundary, and key risks")
    paraphrase = build_query_understanding("英伟达利润能不能变成现金？估值有没有风险？主要不确定性是什么？")

    assert [company.ticker for company in english.companies] == ["NVDA"]
    assert [company.ticker for company in paraphrase.companies] == ["NVDA"]
    assert english.requested_dimensions == expected
    assert paraphrase.requested_dimensions == expected


def test_single_dimension_paraphrases_map_to_canonical_dimensions():
    cases = [
        ("苹果现金流质量怎么样", "cash_flow_quality"),
        ("Is Microsoft valuation stretched?", "valuation_and_risk_boundary"),
        ("Amazon 的资产负债表和资本开支压力怎么样", "balance_sheet_and_capital_intensity"),
        ("How good is Apple's profitability quality?", "profitability_quality"),
    ]

    for query, dimension in cases:
        understanding = build_query_understanding(query)
        assert understanding.analysis_scope == "single_company"
        assert understanding.requested_dimensions == [dimension]


def test_comparison_growth_quality_maps_to_revenue_quality_dimension():
    understanding = build_query_understanding("比较 AMZN 和 AAPL 谁的增长质量更好")

    assert understanding.analysis_scope == "comparison"
    assert understanding.requested_dimensions == ["revenue_quality"]


def test_query_understanding_clarifies_companyless_analysis():
    out = build_query_understanding("这家公司怎么样")
    assert out.needs_clarification
    assert out.user_expectation == "clarification"
    assert out.clarification_reason == "company_mention_ambiguous"


def test_entity_resolution_is_separate_from_methodology_intent():
    source = inspect.getsource(entity_resolution).lower()
    for forbidden in ("risk", "cash", "valuation", "recommend", "风险", "现金", "估值", "推荐"):
        assert forbidden not in source


def test_methodology_intent_invalid_llm_proposal_falls_back_to_rules():
    def bad_classifier(_payload):
        return {"methodology_intent": "not_a_label", "confidence": 0.99}

    understanding = build_query_understanding("苹果现金流怎么样", optional_llm_client=bad_classifier)
    assert understanding.methodology_intent == "cash_flow"
    assert understanding.intent_source == "fallback_rules"


def test_semantic_proposal_drives_complex_methodology_intent():
    parsed = {
        "query_understanding_proposal": {
            "company_mentions": ["Apple"],
            "analysis_scope": "single_company",
            "methodology_intent": "profitability",
            "requested_dimensions": ["profitability_quality", "cash_flow_quality"],
            "requested_metrics": ["gross_margin", "fcf"],
            "user_expectation": "diagnostic",
            "safety_intent": "normal",
            "time_scope": {"policy": "latest_available", "period_scope": "latest annual + latest quarterly", "is_explicit": False},
            "ambiguity": False,
            "needs_clarification": False,
            "confidence": 0.86,
            "reasons": ["Question asks whether profit growth quality is durable."],
        }
    }

    understanding = build_query_understanding("苹果利润增长是不是有质量", parsed=parsed)
    summary = query_understanding_summary(understanding)

    assert understanding.methodology_intent == "profitability"
    assert understanding.intent_source == "semantic_proposal_validated"
    assert summary["rule_methodology_intent"] == "none"
    assert summary["proposed_methodology_intent"] == "profitability"
    assert summary["intent_conflict"] is True
    assert summary["requested_dimensions"] == ["profitability_quality", "cash_flow_quality"]
    assert summary["requested_metrics"] == ["gross_margin", "free_cash_flow"]


def test_semantic_proposal_invalid_or_low_confidence_falls_back_to_rules():
    invalid = build_query_understanding(
        "苹果现金流怎么样",
        parsed={
            "query_understanding_proposal": {
                "company_mentions": ["Apple"],
                "methodology_intent": "stock_pick",
                "safety_intent": "normal",
                "confidence": 0.9,
            }
        },
    )
    low_confidence = build_query_understanding(
        "苹果现金流怎么样",
        parsed={
            "query_understanding_proposal": {
                "company_mentions": ["Apple"],
                "methodology_intent": "valuation",
                "safety_intent": "normal",
                "confidence": 0.2,
            }
        },
    )

    assert invalid.methodology_intent == "cash_flow"
    assert invalid.intent_source == "fallback_rules"
    assert any(item["reason"] == "methodology_intent_not_allowed" for item in invalid.proposal_validation_warnings)
    assert low_confidence.methodology_intent == "cash_flow"
    assert low_confidence.intent_source == "fallback_rules"
    assert any(item["reason"] == "proposal_confidence_too_low" for item in low_confidence.proposal_validation_warnings)


def test_semantic_proposal_cannot_override_direct_fact_or_safety():
    direct_fact = build_query_understanding(
        "Apple revenue 是多少",
        parsed={
            "query_understanding_proposal": {
                "company_mentions": ["Apple"],
                "methodology_intent": "revenue",
                "safety_intent": "normal",
                "confidence": 0.9,
            }
        },
    )
    prediction = build_query_understanding(
        "预测明天苹果股价",
        parsed={
            "query_understanding_proposal": {
                "company_mentions": ["Apple"],
                "methodology_intent": "valuation",
                "safety_intent": "normal",
                "confidence": 0.9,
            }
        },
    )

    assert direct_fact.methodology_intent == "none"
    assert direct_fact.intent_source == "fallback_rules"
    assert any(item["reason"] == "direct_fact_forces_none" for item in direct_fact.proposal_validation_warnings)
    assert prediction.methodology_intent == "none"
    assert prediction.safety_intent == "prediction"
    assert prediction.intent_source == "program_safety_override"
    assert prediction.intent_conflict is True


def test_company_resolution_handles_aliases_and_typos():
    assert _tickers("帮我看看苹果怎么样") == ["AAPL"]
    assert _tickers("analyze nvidai fundamentals") == ["NVDA"]
    assert _tickers("apple vs amazon") == ["AAPL", "AMZN"]


def test_methodology_classifier_valid_llm_proposal_is_validated():
    def classifier(_payload):
        return {"methodology_intent": "valuation", "confidence": 0.8, "reasons": ["test"]}

    companies = build_query_understanding("NVIDIA").companies
    result = classify_methodology_intent("nvidia", companies, optional_llm_client=classifier)
    assert result.methodology_intent == "valuation"
    assert result.source == "llm_validated"
