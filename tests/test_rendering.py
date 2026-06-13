"""Unit tests for rendering and phase4 output boundaries."""

from __future__ import annotations

from src.agent.analysis_framework import select_analysis_framework
from src.agent.constants import OUTPUT_PROTOCOL_VERSION
from src.agent.rendering import (
    build_market_reaction_block,
    build_phase4_output,
    render_answer_from_output,
    render_methodology_comparison_brief,
    render_methodology_single_company_brief,
    render_risk_focused_analysis_brief,
    sanitize_user_facing_answer_text,
)


def test_build_phase4_output_keeps_protocol_shape():
    numeric_evidence = [
        {
            "evidence_id": "N1",
            "ticker": "AAPL",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-09-27",
            "value": 100.0,
            "unit": "USD",
            "source_tool": "query_financial_data",
            "provenance": "financial_facts:AAPL:revenue:2025-09-27",
        }
    ]
    numeric_citations = [
        {
            "evidence_id": "N1",
            "ticker": "AAPL",
            "source": "AAPL",
            "metric": "revenue",
            "period_type": "quarterly",
            "period_end": "2025-09-27",
            "source_kind": "structured",
        }
    ]

    output = build_phase4_output(
        {"user_query": "AAPL revenue", "companies": ["AAPL"], "requested_metrics": ["revenue"]},
        "en",
        "fact_qa",
        "same_period",
        {},
        [{"sentence": "AAPL revenue was 100.0.", "evidence_ids": ["N1"]}],
        [],
        numeric_evidence,
        [],
        numeric_citations,
        [],
        [],
    )

    assert output["protocol_version"] == OUTPUT_PROTOCOL_VERSION
    assert output["task_type"] == "fact_qa"
    assert output["synthesis_mode"] == ""
    assert output["numeric_evidence"][0]["evidence_id"] == "N1"
    assert "AAPL revenue was 100.0" in render_answer_from_output(output, "en")


def test_user_facing_cleanup_removes_raw_metric_names_and_duplicate_boundaries():
    raw = (
        "Company Overview\n"
        "net_income 为 $10B。\n"
        "market_cap 为 $100B。\n"
        "- 缺少可验证数据。\n"
        "- 缺少可验证数据。\n"
        "不能给买卖建议；但从 P/E、P/S 较高看，估值风险偏高。\n"
        "当前分析不构成投资建议。\n"
        "有限判断 / 基于业务结构推断：Amazon 的履约压力会影响利润率。"
    )

    cleaned = sanitize_user_facing_answer_text(raw, "zh")

    assert "Company Overview" not in cleaned
    assert "net_income" not in cleaned
    assert "market_cap" not in cleaned
    assert "净利润" in cleaned
    assert "市值" in cleaned
    assert cleaned.count("缺少可验证数据") == 1
    assert cleaned.count("投资建议") == 0
    assert cleaned.count("不能给买卖建议") == 1
    assert "有限判断 / 基于业务结构推断" not in cleaned


def test_build_phase4_output_includes_analysis_framework_summary():
    selected = select_analysis_framework({"user_query": "apple 和 amazon 更推荐哪个"}).model_dump()

    output = build_phase4_output(
        {
            "user_query": "apple 和 amazon 更推荐哪个",
            "companies": ["AAPL", "AMZN"],
            "requested_metrics": ["revenue", "net_income"],
            "selected_analysis_framework": selected,
        },
        "zh",
        "company_comparison",
        "same_period",
        {},
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    framework = output["analysis_framework"]
    assert framework["id"] == "fundamental_quality_analysis"
    assert {
        "revenue_quality",
        "profitability_quality",
        "moat_and_competitive_risk",
        "valuation_and_risk_boundary",
    } <= set(framework["active_dimension_ids"])
    assert "Business Model" not in framework["active_dimension_names"]


def test_build_phase4_output_exposes_user_visible_red_flags_only():
    output = build_phase4_output(
        {
            "user_query": "AAPL analysis",
            "companies": ["AAPL"],
            "requested_metrics": ["revenue"],
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
        },
        "zh",
        "report_summary",
        "",
        {},
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    assert output["red_flags"] == [
        {
            "severity": "medium",
            "category": "missing_evidence",
            "message": "当前缺少估值证据，不能判断价格是否便宜或昂贵。",
            "evidence_refs": [],
        }
    ]


def test_build_phase4_output_preserves_text_supporting_snippet():
    output = build_phase4_output(
        {"user_query": "AAPL risks", "companies": ["AAPL"], "requested_metrics": []},
        "en",
        "report_summary",
        "",
        {},
        [],
        [{"sentence": "Competition is a risk factor.", "evidence_ids": ["T1"]}],
        [],
        [
            {
                "evidence_id": "T1",
                "ticker": "AAPL",
                "filing_id": "f1",
                "form_type": "10-K",
                "fiscal_period": "2025",
                "section": "ITEM_1A",
                "chunk_order": 1,
                "text_snippet": "Competition is a risk factor.",
                "supporting_snippet": "Competition is a risk factor.",
                "supporting_terms": ["competition", "risk"],
                "score_breakdown": {"lexical_bonus": 0.08},
            }
        ],
        [],
        [{"evidence_id": "T1"}],
        [],
    )

    card = output["text_evidence"][0]
    assert card["supporting_snippet"] == "Competition is a risk factor."
    assert card["supporting_terms"] == ["competition", "risk"]
    assert card["score_breakdown"]["lexical_bonus"] == 0.08


def test_user_facing_limitations_are_sanitized():
    output = build_phase4_output(
        {
            "user_query": "AAPL vs AMZN",
            "companies": ["AAPL", "AMZN"],
            "requested_metrics": ["revenue", "net_income"],
            "answer_mode": "comparison_brief",
            "safety_intent": "investment_advice_like",
            "evidence_plan": {"evidence_requirements": []},
            "requirement_limitations": [
                {
                    "code": "requirement_missing",
                    "severity": "high",
                    "message": "Evidence requirement REQ-CALC-AAPL-OPERATING_MARGIN was missing: dependency_numeric_requirement_missing.",
                    "requirement_id": "REQ-CALC-AAPL-OPERATING_MARGIN",
                    "failure_reason": "dependency_numeric_requirement_missing",
                },
                {
                    "code": "requirement_partial",
                    "severity": "medium",
                    "message": "profitability_quality numeric_only_comparison",
                    "requirement_id": "REQ-CALC-AMZN-OPERATING_MARGIN",
                    "failure_reason": "dependency_numeric_requirement_missing",
                },
                {
                    "code": "numeric_only_comparison",
                    "severity": "medium",
                    "message": "numeric_only_comparison",
                },
                {
                    "code": "provider_notice",
                    "severity": "low",
                    "message": "Some structured data came from yfinance fallback.",
                },
                {
                    "code": "required_evidence_missing",
                    "severity": "medium",
                    "message": "Required evidence is missing.",
                },
            ],
        },
        "zh",
        "company_comparison",
        "same_period",
        {},
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    messages = [item["message"] for item in output["limitations"]]
    joined = "\n".join(messages)
    assert "REQ-" not in joined
    assert "dependency_numeric_requirement_missing" not in joined
    assert "numeric_only_comparison" not in joined
    assert "Required evidence is missing" not in joined
    assert "fallback" not in joined
    assert "profitability_quality" not in joined
    assert any("yfinance" in message and "fallback" not in message for message in messages)
    assert any("部分派生指标不可用" in message for message in messages)
    assert any("有限财务视角判断" in message for message in messages)
    assert {item["code"] for item in output["limitations"]} >= {"requirement_missing", "numeric_only_comparison"}


def test_methodology_comparison_uses_methodology_sections():
    methodology_answer = {
        "judgment": "基于当前已验证证据，我更偏向 AAPL 的盈利质量。",
        "metric_table": [
            {
                "metric_id": "revenue",
                "label": "收入",
                "company_values": {"AAPL": "$120.00", "AMZN": "$150.00"},
                "judgment": "AMZN 规模更大",
                "evidence_refs": ["N1", "N4"],
            },
            {
                "metric_id": "net_income",
                "label": "净利润",
                "company_values": {"AAPL": "$32.00", "AMZN": "$22.00"},
                "judgment": "AAPL 更强",
                "evidence_refs": ["N2", "N5"],
            },
            {
                "metric_id": "net_margin",
                "label": "净利率",
                "company_values": {"AAPL": "26.67%", "AMZN": "14.67%"},
                "judgment": "AAPL 盈利转化更强",
                "evidence_refs": ["N3", "N6"],
            },
        ],
        "dimension_sections": [
            {
                "dimension_id": "revenue_quality",
                "title": "收入质量",
                "status": "satisfied",
                "summary": "AMZN 收入更高；AAPL 收入 $120，AMZN 收入 $150。",
                "evidence_refs": ["N1", "N4"],
            },
            {
                "dimension_id": "valuation_and_risk_boundary",
                "title": "估值边界",
                "status": "missing",
                "summary": "",
                "limitation": "当前缺少估值证据，因此不能判断谁更便宜或更值得买。",
                "evidence_refs": [],
            },
        ],
        "counterpoint": "AMZN 的收入规模仍是反方观点。",
        "limitations": [
            "当前缺少估值证据，因此不能判断谁更便宜或更值得买。",
            "当前缺少估值证据，因此不能判断哪家公司价格更便宜。",
            "以下内容仅是基于已验证证据的基本面比较，不构成投资建议。",
        ],
    }

    rendered = render_methodology_comparison_brief(methodology_answer, lang="zh")

    assert "基本面方法论比较" in rendered
    assert "比较判断" in rendered
    assert "核心指标对比" in rendered
    assert "| 指标 | AAPL | AMZN | 当前判断 |" in rendered
    assert "| 净利率 | 26.67% | 14.67% | AAPL 盈利转化更强 |" in rendered
    assert "维度分析" in rendered
    assert "反方观点" in rendered
    assert "限制" in rendered
    assert "1. 收入质量" in rendered
    assert "2. 估值边界：当前缺少估值证据" in rendered
    assert rendered.count("当前缺少估值证据") == 1
    assert "Key Points" not in rendered


def test_methodology_comparison_does_not_use_key_points_as_primary():
    output = {
        "task_type": "company_comparison",
        "title": "公司横向对比",
        "summary": "Legacy summary",
        "key_points": ["Legacy key point"],
        "limitations": [],
        "view": {
            "kind": "methodology_comparison_brief",
            "methodology_answer": {
                "judgment": "基于当前已验证证据，我更偏向 AAPL 的盈利质量。",
                "metric_table": [
                    {
                        "metric_id": "net_margin",
                        "label": "净利率",
                        "company_values": {"AAPL": "26.67%", "AMZN": "14.67%"},
                        "judgment": "AAPL 盈利转化更强",
                    }
                ],
                "dimension_sections": [
                    {
                        "dimension_id": "profitability_quality",
                        "title": "盈利质量",
                        "status": "satisfied",
                        "summary": "AAPL 净利率 26.67%，AMZN 净利率 14.67%。",
                        "evidence_refs": ["N3", "N6"],
                    }
                ],
                "counterpoint": "AMZN 收入规模更大。",
                "limitations": ["当前缺少估值证据，因此不能判断谁更便宜或更值得买。"],
            },
        },
    }

    rendered = render_answer_from_output(output, "zh")

    assert "比较判断" in rendered
    assert "基本面方法论比较" in rendered
    assert "核心指标对比" in rendered
    assert "盈利质量" in rendered
    assert "Legacy summary" not in rendered
    assert "Legacy key point" not in rendered
    assert "Key Points" not in rendered

    rendered = render_answer_from_output(output, "zh")
    assert "REQ-" not in rendered
    assert "dependency_numeric_requirement_missing" not in rendered
    assert "numeric_only_comparison" not in rendered
    assert "profitability_quality" not in rendered


def test_single_company_renderer_outputs_fundamental_quick_analysis():
    methodology_answer = {
        "analysis_scope": "single_company",
        "judgment": "基于当前已验证证据，NVDA 可以做有限基本面分析。",
        "single_company_metric_table": [
            {"metric_id": "revenue", "label": "收入", "value": "$60.90B", "interpretation": "反映当前收入规模", "evidence_refs": ["N1"]},
            {"metric_id": "net_income", "label": "净利润", "value": "$29.80B", "interpretation": "反映当前盈利规模", "evidence_refs": ["N2"]},
            {"metric_id": "net_margin", "label": "净利率", "value": "48.93%", "interpretation": "反映盈利转化能力", "evidence_refs": ["N3"]},
        ],
        "dimension_sections": [
            {
                "dimension_id": "business_model",
                "title": "业务模式",
                "status": "satisfied",
                "summary": "NVDA 的业务模式分析可基于产品和服务组合展开。",
                "evidence_refs": ["T1"],
            },
            {
                "dimension_id": "revenue_quality",
                "title": "收入质量",
                "status": "satisfied",
                "summary": "NVDA 当前收入为 $60.90B，可作为收入规模和收入质量的核心证据。",
                "evidence_refs": ["N1"],
            },
            {
                "dimension_id": "profitability_quality",
                "title": "盈利质量",
                "status": "satisfied",
                "summary": "NVDA 当前净利润为 $29.80B，净利率为 48.93%。",
                "evidence_refs": ["N2", "N3"],
            },
            {
                "dimension_id": "moat_and_competitive_risk",
                "title": "竞争与风险",
                "status": "missing",
                "limitation": "当前缺少风险文本证据，不能做具体风险判断。",
                "evidence_refs": [],
            },
            {
                "dimension_id": "valuation_and_risk_boundary",
                "title": "估值边界",
                "status": "missing",
                "limitation": "当前缺少估值证据，因此不能判断估值吸引力或是否便宜/昂贵。",
                "evidence_refs": [],
            },
        ],
        "limitations": [
            "当前缺少风险文本证据，不能做具体风险判断。",
            "当前缺少估值证据，因此不能判断估值吸引力或是否便宜/昂贵。",
            "当前分析基于已验证财报和文本证据，不构成投资建议。",
        ],
        "follow_up_metrics": ["市值、市盈率、市销率、自由现金流收益率"],
    }

    rendered = render_methodology_single_company_brief(methodology_answer, lang="zh")

    assert "基本面快速分析" in rendered
    assert "初步判断" in rendered
    assert "核心指标" in rendered
    assert "| 指标 | 数值 | 当前解读 |" in rendered
    assert "业务模式" in rendered
    assert "收入质量" in rendered
    assert "盈利质量" in rendered
    assert "竞争与风险" in rendered
    assert "估值边界" in rendered
    assert "限制" in rendered
    assert "后续应关注指标" in rendered
    assert rendered.count("当前缺少估值证据") == 1
    assert "Key Points" not in rendered
    assert "财务趋势分析" not in rendered
    assert "REQ-" not in rendered


def test_single_company_methodology_view_is_primary_output():
    output = {
        "task_type": "report_summary",
        "title": "财务趋势分析",
        "summary": "Legacy trend summary",
        "key_points": ["Legacy key point"],
        "limitations": [],
        "view": {
            "kind": "methodology_single_company_brief",
            "methodology_answer": {
                "analysis_scope": "single_company",
                "judgment": "基于当前已验证证据，NVDA 可以做有限基本面分析。",
                "single_company_metric_table": [
                    {"metric_id": "revenue", "label": "收入", "value": "$60.90B", "interpretation": "反映当前收入规模"},
                    {"metric_id": "net_income", "label": "净利润", "value": "$29.80B", "interpretation": "反映当前盈利规模"},
                    {"metric_id": "net_margin", "label": "净利率", "value": "48.93%", "interpretation": "反映盈利转化能力"},
                ],
                "dimension_sections": [
                    {
                        "dimension_id": "valuation_and_risk_boundary",
                        "title": "估值边界",
                        "status": "missing",
                        "limitation": "当前缺少估值证据，因此不能判断估值吸引力或是否便宜/昂贵。",
                    }
                ],
                "limitations": ["当前分析基于已验证财报和文本证据，不构成投资建议。"],
            },
        },
    }

    rendered = render_answer_from_output(output, "zh")

    assert "基本面快速分析" in rendered
    assert "初步判断" in rendered
    assert "核心指标" in rendered
    assert "估值边界" in rendered
    assert "Legacy trend summary" not in rendered
    assert "Legacy key point" not in rendered
    assert "Key Points" not in rendered
    assert "财务趋势分析" not in rendered


def test_risk_renderer_does_not_show_metric_table_first():
    risk_answer = {
        "company": "NVDA",
        "direct_judgment": "我认为当前最需要关注的是：NVDA 的新产品和需求不确定性。",
        "why_core_issue": [
            "这会直接影响收入增长持续性、库存安排和利润率弹性。[T2]",
            "该判断来自已验证风险文本，而不是估值或股价预测。[T2]",
        ],
        "filing_evidence": [
            {
                "theme_name": "新产品和需求不确定性",
                "why_it_matters": "这会直接影响收入增长持续性、库存安排和利润率弹性。",
                "evidence_refs": ["T2"],
                "supporting_quote": "The success of new product and service introductions depends on demand.",
            }
        ],
        "financial_context": ["收入：$60.90B [N1]", "净利率：48.93% [N3]"],
        "risk_ranking": [
            {
                "theme_name": "新产品和需求不确定性",
                "rationale": "披露文本直接提及或支撑“新产品和需求不确定性”。",
                "evidence_refs": ["T2"],
                "mechanism_support_level": "direct",
            }
        ],
        "secondary_risks": [
            {
                "theme_name": "竞争加剧",
                "why_it_matters": "竞争压力可能影响定价能力、市场份额和利润率。",
                "evidence_refs": ["T3"],
                "mechanism_support_level": "inferred",
            }
        ],
        "evidence_boundaries": [
            "当前缺少估值证据，因此不能判断价格是否便宜或昂贵。",
            "这不是投资建议；更合适的做法是按盈利稳定性、增长弹性和风险暴露来比较。",
        ],
    }

    rendered = render_risk_focused_analysis_brief(risk_answer, lang="zh")

    assert rendered.startswith("结论")
    assert "基于业务模型的风险排序" in rendered
    assert "财务传导路径" in rendered
    assert "待验证数据" in rendered
    assert "证据边界" in rendered
    assert "核心指标" not in rendered.splitlines()[:4]
    assert "The success of new product" not in rendered
    assert "[T2]" in rendered
    assert "买入" not in rendered and "卖出" not in rendered


def test_risk_focused_view_is_primary_output():
    output = {
        "task_type": "report_summary",
        "title": "基本面快速分析",
        "summary": "Legacy summary",
        "key_points": ["Legacy key point"],
        "view": {
            "kind": "risk_focused_analysis_brief",
            "risk_focused_answer": {
                "company": "NVDA",
                "direct_judgment": "我认为当前最需要关注的是：NVDA 的竞争加剧。",
                "why_core_issue": ["竞争压力可能影响定价能力、市场份额和利润率。[T1]"],
                "filing_evidence": [
                    {"theme_name": "竞争加剧", "why_it_matters": "竞争压力可能影响定价能力、市场份额和利润率。", "evidence_refs": ["T1"]}
                ],
                "financial_context": ["收入：$60.90B [N1]"],
                "evidence_boundaries": ["当前缺少估值证据，因此不能判断价格是否便宜或昂贵。"],
            },
        },
    }

    rendered = render_answer_from_output(output, "zh")

    assert "结论" in rendered
    assert "财务传导路径" in rendered
    assert "待验证数据" in rendered
    assert "Legacy summary" not in rendered


def test_risk_focused_renderer_drops_non_text_ranking_fallback_refs():
    rendered = render_risk_focused_analysis_brief(
        {
            "company": "NVDA",
            "direct_judgment": "基于披露信息，我认为当前最需要关注的是竞争风险。",
            "top_risk": {
                "theme_name": "竞争风险",
                "why_it_matters": "竞争可能影响利润率。",
                "evidence_refs": ["N1"],
            },
            "secondary_risks": [
                {
                    "theme_name": "需求风险",
                    "why_it_matters": "需求可能影响收入。",
                    "evidence_refs": ["N2"],
                }
            ],
            "evidence_boundaries": ["仅基于已验证文本证据。"],
        },
        lang="zh",
    )

    assert "核心风险排序" not in rendered
    assert "[N1]" not in rendered
    assert "[N2]" not in rendered


def test_overview_risk_renderer_drops_non_t_refs():
    rendered = render_methodology_single_company_brief(
        {
            "analysis_scope": "single_company",
            "judgment": "当前只能做有边界公司概览。",
            "overview_profile": {"kind": "single_company_overview", "company": "AMZN"},
            "dimension_sections": [
                {
                    "dimension_id": "moat_and_competitive_risk",
                    "title": "主要风险",
                    "status": "satisfied",
                    "summary": "风险判断需要可验证文本证据。",
                    "evidence_refs": ["T1"],
                }
            ],
            "overview_risk_ranking": [
                {
                    "rank_label": "最高优先级",
                    "theme": "竞争风险",
                    "mechanism": "可能影响利润率",
                    "evidence_refs": ["D1"],
                }
            ],
        },
        lang="zh",
    )

    assert "[D1]" not in rendered
    assert "竞争风险，传导机制" not in rendered
    assert "Legacy key point" not in rendered
    assert "基本面快速分析" not in rendered
    assert "Key Points" not in rendered


def test_build_market_reaction_block_renders_event_windows():
    block = build_market_reaction_block(
        {
            "market_reaction_requested": True,
            "tool_results": [
                {
                    "tool": "query_event_price_window",
                    "ticker": "AAPL",
                    "data": {
                        "ticker": "AAPL",
                        "events": [
                            {
                                "ticker": "AAPL",
                                "event_type": "10Q",
                                "event_date": "2025-01-31",
                                "fiscal_period": "2024-12-31",
                                "returns": {"return_1d": 0.01, "return_5d": -0.02},
                            }
                        ],
                    },
                }
            ],
        },
        "en",
        [],
    )

    assert block is not None
    assert block["title"] == "Market Reaction"
    assert block["events"][0]["ticker"] == "AAPL"
