"""Unit tests for evidence-constrained analytical synthesis."""

from __future__ import annotations

from copy import deepcopy

from src.agent.synthesis import (
    build_analytical_synthesis,
    build_bounded_valuation_risk_comparison_candidate,
    build_methodology_answer,
    build_risk_focused_answer,
    build_synthesis_view,
    render_synthesis_text,
    validate_synthesis_output,
)


def _numeric_claim():
    return {"sentence": "AAPL revenue (2025-12-31, quarterly) = 100 USD.", "evidence_ids": ["N1"]}


def _text_claim():
    return {"sentence": "Management discussed demand risk.", "evidence_ids": ["T1"]}


def test_networking_driver_scope_renders_driver_analysis_before_metric_template():
    synthesis = {
        "segment_or_product_scope": "networking",
        "analysis_plan": {
            "companies": ["NVDA"],
            "canonical_intent": {"segment_or_product_scope": "networking"},
        },
        "text_evidence_cards": [
            {
                "evidence_id": "T13",
                "ticker": "NVDA",
                "claim": "Networking revenue was $8.2 billion, up 162% from a year ago from the introduction and continued growth of NVLink compute fabric for GB200 and GB300 systems.",
            },
            {
                "evidence_id": "T1",
                "ticker": "NVDA",
                "claim": "Compute & Networking includes Data Center accelerated computing, networking platforms, AI solutions and software.",
            },
        ],
        "methodology_answer": {
            "analysis_scope": "single_company",
            "judgment": "Generic methodology answer.",
            "single_company_metric_table": [
                {"label": "P/E", "value": "100x", "interpretation": "valuation"},
                {"label": "现金及等价物", "value": "$1", "interpretation": "cash"},
            ],
        },
    }

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")

    assert "结论" in rendered
    assert "已验证线索" in rendered
    assert "合理推断" in rendered
    assert "待验证数据" in rendered
    assert "证据边界" in rendered
    assert "AI 集群建设" in rendered
    assert "GPU 集群互连" in rendered
    assert "NVLink" in rendered
    assert "InfiniBand/Ethernet" in rendered
    assert "供给/出货节奏" in rendered
    assert "[T13]" in rendered
    assert "基本面快速分析" not in rendered
    assert "核心指标表" not in rendered
    assert "P/E" not in rendered
    assert "现金及等价物" not in rendered


def test_data_center_driver_scope_states_candidate_not_confirmed_share():
    synthesis = {
        "segment_or_product_scope": "data center",
        "analysis_plan": {
            "companies": ["NVDA"],
            "canonical_intent": {"segment_or_product_scope": "data center"},
        },
        "text_evidence_cards": [
            {
                "evidence_id": "T8",
                "ticker": "NVDA",
                "claim": "Data Center revenue was $41.1 billion, up 56% from a year ago and up 5% sequentially.",
            },
            {
                "evidence_id": "T9",
                "ticker": "NVDA",
                "claim": "Revenue from Data Center networking grew 142% driven by NVLink compute fabric, GB200 and GB300, Ethernet and InfiniBand platforms.",
            },
        ],
        "methodology_answer": {
            "analysis_scope": "single_company",
            "judgment": "Generic methodology answer.",
            "single_company_metric_table": [{"label": "P/E", "value": "100x", "interpretation": "valuation"}],
        },
    }

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")

    assert "数据中心很可能是 NVDA 营收增长的重要候选驱动" in rendered
    assert "不能确认它对总营收增长的主要贡献比例" in rendered
    assert "[T8]" in rendered and "[T9]" in rendered
    assert "核心指标表" not in rendered


def test_bounded_valuation_risk_candidate_preserves_per_metric_judgments():
    frame = {
        "focus_dimensions": ["valuation_and_risk_boundary"],
        "evidence_basis": [
            {
                "dimension": "valuation_and_risk_boundary",
                "evidence_refs": ["N1", "N2", "N3"],
                "metric_judgments": [
                    {"metric": "pe_ratio", "zh": "按 P/E：AAPL 更高，AAPL 倍数压力更大", "evidence_refs": ["N1"]},
                    {"metric": "ps_ratio", "zh": "按 P/S：NVDA 更高，NVDA 倍数压力更大", "evidence_refs": ["N2"]},
                    {"metric": "fcf_yield", "zh": "按 FCF yield：两者接近，AAPL 略低，AAPL 现金流收益率角度风险略高", "evidence_refs": ["N3"]},
                ],
            }
        ],
    }

    answer = build_bounded_valuation_risk_comparison_candidate(frame, {}, lang="zh")

    assert "不能给单一绝对排序" in answer
    assert "按 P/E" in answer
    assert "按 P/S" in answer
    assert "按 FCF yield" in answer
    assert "[N1]" in answer and "[N2]" in answer and "[N3]" in answer


def _methodology_packet():
    return {
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "safety_intent": "investment_advice_like",
        "selected_framework": {"id": "fundamental_quality_analysis"},
        "active_dimensions": [
            "revenue_quality",
            "profitability_quality",
            "moat_and_competitive_risk",
            "valuation_and_risk_boundary",
        ],
        "dimension_status_map": {
            "revenue_quality": {"status": "satisfied"},
            "profitability_quality": {"status": "satisfied"},
            "moat_and_competitive_risk": {"status": "satisfied"},
            "valuation_and_risk_boundary": {"status": "missing"},
        },
        "dimension_summary": [
            {
                "dimension_id": "revenue_quality",
                "status": "satisfied",
                "numeric_evidence_refs": ["N1", "N4"],
                "text_evidence_refs": [],
                "evidence_refs": ["N1", "N4"],
            },
            {
                "dimension_id": "profitability_quality",
                "status": "satisfied",
                "numeric_evidence_refs": ["N2", "N3", "N5", "N6"],
                "text_evidence_refs": [],
                "evidence_refs": ["N2", "N3", "N5", "N6"],
            },
            {
                "dimension_id": "moat_and_competitive_risk",
                "status": "satisfied",
                "numeric_evidence_refs": [],
                "text_evidence_refs": ["T1", "T2"],
                "evidence_refs": ["T1", "T2"],
            },
            {
                "dimension_id": "valuation_and_risk_boundary",
                "status": "missing",
                "numeric_evidence_refs": [],
                "text_evidence_refs": [],
                "evidence_refs": [],
                "limitation": "当前缺少估值证据，因此不能判断哪家公司价格更便宜。",
            },
        ],
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "value": 120.0, "display_value": "$120.00", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N2", "ticker": "AAPL", "metric": "net_income", "value": 32.0, "display_value": "$32.00", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N3", "ticker": "AAPL", "metric": "net_margin", "value": 0.2667, "display_value": "26.67%", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N4", "ticker": "AMZN", "metric": "revenue", "value": 150.0, "display_value": "$150.00", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N5", "ticker": "AMZN", "metric": "net_income", "value": 22.0, "display_value": "$22.00", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N6", "ticker": "AMZN", "metric": "net_margin", "value": 0.1467, "display_value": "14.67%", "source_provider": "sec_companyfacts"},
        ],
        "text_snippets": [
            {
                "evidence_id": "T1",
                "ticker": "AAPL",
                "supporting_snippet": "Apple highlighted margin discipline and steady operating execution.",
            },
            {
                "evidence_id": "T2",
                "ticker": "AMZN",
                "supporting_snippet": "Amazon discussed revenue scale, reinvestment, and operating leverage.",
            },
        ],
    }


def _single_company_methodology_packet():
    return {
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "safety_intent": "normal",
        "analysis_scope": "single_company",
        "selected_framework": {"id": "fundamental_quality_analysis"},
        "active_dimensions": [
            "business_model",
            "revenue_quality",
            "profitability_quality",
            "cash_flow_quality",
            "balance_sheet_and_capital_intensity",
            "moat_and_competitive_risk",
            "valuation_and_risk_boundary",
        ],
        "dimension_status_map": {
            "business_model": {"status": "satisfied"},
            "revenue_quality": {"status": "satisfied"},
            "profitability_quality": {"status": "satisfied"},
            "cash_flow_quality": {"status": "missing"},
            "balance_sheet_and_capital_intensity": {"status": "missing"},
            "moat_and_competitive_risk": {"status": "satisfied"},
            "valuation_and_risk_boundary": {"status": "missing"},
        },
        "dimension_summary": [
            {
                "dimension_id": "business_model",
                "status": "satisfied",
                "text_evidence_refs": ["T1"],
                "evidence_refs": ["T1"],
            },
            {
                "dimension_id": "revenue_quality",
                "status": "satisfied",
                "numeric_evidence_refs": ["N1", "N1A"],
                "evidence_refs": ["N1", "N1A"],
            },
            {
                "dimension_id": "profitability_quality",
                "status": "satisfied",
                "numeric_evidence_refs": ["N2", "N2A", "N3", "N3A"],
                "evidence_refs": ["N2", "N2A", "N3", "N3A"],
            },
            {
                "dimension_id": "cash_flow_quality",
                "status": "missing",
                "evidence_refs": [],
                "limitation": "当前缺少经营现金流、自由现金流或资本开支证据，无法验证利润能否转化为现金。",
            },
            {
                "dimension_id": "balance_sheet_and_capital_intensity",
                "status": "missing",
                "evidence_refs": [],
                "limitation": "当前缺少现金、债务、资本开支、应收款或存货证据，不能判断抗风险能力和资本投入强度。",
            },
            {
                "dimension_id": "moat_and_competitive_risk",
                "status": "satisfied",
                "text_evidence_refs": ["T2"],
                "evidence_refs": ["T2"],
            },
            {
                "dimension_id": "valuation_and_risk_boundary",
                "status": "missing",
                "evidence_refs": [],
                "limitation": "当前缺少估值证据，因此不能判断估值吸引力或是否便宜/昂贵。",
            },
        ],
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "period_end": "2025-01-31", "value": 60.9, "display_value": "$60.90B", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N1A", "ticker": "NVDA", "metric": "revenue", "period_end": "2024-01-31", "value": 26.9, "display_value": "$26.90B", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N2", "ticker": "NVDA", "metric": "net_income", "period_end": "2025-01-31", "value": 29.8, "display_value": "$29.80B", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N2A", "ticker": "NVDA", "metric": "net_income", "period_end": "2024-01-31", "value": 4.4, "display_value": "$4.40B", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N3", "ticker": "NVDA", "metric": "net_margin", "period_end": "2025-01-31", "value": 0.4893, "display_value": "48.93%", "source_provider": "sec_companyfacts"},
            {"evidence_id": "N3A", "ticker": "NVDA", "metric": "net_margin", "period_end": "2024-01-31", "value": 0.1636, "display_value": "16.36%", "source_provider": "sec_companyfacts"},
        ],
        "text_snippets": [
            {
                "evidence_id": "T1",
                "ticker": "NVDA",
                "section": "ITEM_1",
                "supporting_snippet": "NVIDIA provides GPUs and accelerated computing platforms, products, services, and solutions for gaming, data center, professional visualization, and automotive markets.",
            },
            {
                "evidence_id": "T2",
                "ticker": "NVDA",
                "section": "ITEM_1A",
                "supporting_snippet": "NVIDIA faces intense competition, demand uncertainty, supply constraints, and macroeconomic risks across markets.",
            },
        ],
        "red_flags": [],
    }


def _risk_focused_packet():
    return {
        "task_type": "report_summary",
        "answer_mode": "risk_focused_analysis",
        "analysis_scope": "single_company",
        "selected_framework": {"id": "fundamental_quality_analysis"},
        "active_dimensions": ["business_model", "moat_and_competitive_risk", "revenue_quality", "profitability_quality"],
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 60.9, "display_value": "$60.90B"},
            {"evidence_id": "N2", "ticker": "NVDA", "metric": "net_income", "value": 29.8, "display_value": "$29.80B"},
            {"evidence_id": "N3", "ticker": "NVDA", "metric": "net_margin", "value": 0.4893, "display_value": "48.93%"},
        ],
        "text_snippets": [
            {
                "evidence_id": "T1",
                "ticker": "NVDA",
                "dimension_id": "business_model",
                "section": "ITEM_1",
                "claim": "NVDA provides GPUs and accelerated computing platforms for data center, gaming, professional visualization, and automotive markets.",
                "supporting_snippet": "NVDA provides GPUs and accelerated computing platforms for data center, gaming, professional visualization, and automotive markets.",
            },
            {
                "evidence_id": "T2",
                "ticker": "NVDA",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "claim": "NVDA faces demand uncertainty and new product introduction risks that can affect revenue growth and margin.",
                "supporting_snippet": "The success of new product introductions depends on demand uncertainty and can affect revenue growth and margin.",
            },
            {
                "evidence_id": "T3",
                "ticker": "NVDA",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "claim": "NVDA faces intense competition and supply chain constraints across markets.",
                "supporting_snippet": "Intense competition and supply chain constraints may affect our markets.",
            },
            {
                "ticker": "NVDA",
                "dimension_id": "moat_and_competitive_risk",
                "section": "ITEM_1A",
                "claim": "This row has no citation and must not generate a theme.",
                "supporting_snippet": "competition demand margin",
            },
        ],
    }


def test_causal_research_plan_render_layers_company_and_segment_drivers():
    synthesis = {
        "short_answer": "NVIDIA 的营收增长原因需要同时看总量和披露驱动。",
        "research_plan": {"question_type": "causal_explanation", "companies": ["NVDA"]},
        "requirement_summary": {"partial_required_answer_parts": ["quantify_growth"]},
        "numeric_evidence_cards": [
            {
                "evidence_id": "N1",
                "citation_ref": "N1",
                "ticker": "NVDA",
                "metric": "revenue",
                "period_end": "2026-01-31",
                "display_value": "$68.13B",
            }
        ],
        "text_evidence_cards": [
            {
                "evidence_id": "T1",
                "citation_ref": "T1",
                "driver_levels": ["company_level_driver"],
                "claim_scope": "company",
                "allowed_claim_strength": "definitive",
                "claim": "NVIDIA's revenue growth is driven by demand for accelerated computing platforms.",
            },
            {
                "evidence_id": "T2",
                "citation_ref": "T2",
                "driver_levels": ["segment_level_driver", "product_level_driver"],
                "claim_scope": "product",
                "allowed_claim_strength": "bounded_inference",
                "claim": "Data Center networking revenue grew 142% driven by NVLink and InfiniBand platforms.",
            },
        ],
    }

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")

    assert "公司级证据" in rendered
    assert "分部 / 产品层面证据" in rendered
    assert "证据边界" in rendered
    assert "[T1]" in rendered
    assert "[T2]" in rendered
    assert "单期收入或分部增长文本不能单独满足" in rendered


def test_causal_research_plan_render_caveats_segment_only_driver():
    synthesis = {
        "short_answer": "当前只有分部层面的增长驱动证据。",
        "research_plan": {"question_type": "causal_explanation", "companies": ["NVDA"]},
        "requirement_summary": {"partial_required_answer_parts": ["identify_growth_drivers"]},
        "text_evidence_cards": [
            {
                "evidence_id": "T2",
                "citation_ref": "T2",
                "driver_levels": ["segment_level_driver"],
                "claim_scope": "segment",
                "allowed_claim_strength": "bounded_inference",
                "claim": "Networking revenue grew from NVLink and InfiniBand platforms.",
            }
        ],
    }

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")

    assert "分部/产品层面证据" in rendered
    assert "不能完整代表公司总收入增长原因" in rendered


def test_causal_research_plan_segment_claim_scope_does_not_render_company_definitive_driver():
    synthesis = {
        "short_answer": "从当前证据看，NVDA 的增长线索主要来自分部层面。",
        "research_plan": {"question_type": "causal_explanation", "companies": ["NVDA"]},
        "requirement_summary": {"partial_required_answer_parts": ["identify_growth_drivers"]},
        "text_evidence_cards": [
            {
                "evidence_id": "T1",
                "citation_ref": "T1",
                "driver_levels": ["segment_level_driver"],
                "claim_scope": "segment",
                "allowed_claim_strength": "bounded_inference",
                "claim": "NVIDIA 的营收增长主要由加速计算和人工智能两大平台转型驱动。",
            }
        ],
    }

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")

    assert "NVIDIA 的营收增长主要由加速计算和人工智能两大平台转型驱动" not in rendered
    assert "分部层面证据提供了增长线索" in rendered
    assert "不能完整代表总公司营收增长原因" in rendered


def test_segment_product_scope_renders_dedicated_driver_layers():
    synthesis = {
        "short_answer": "从当前证据看，NVDA 的 networking 增长只能在分部/产品层面解释。",
        "segment_or_product_scope": "networking",
        "analysis_plan": {"segment_or_product_scope": "networking"},
        "research_plan": {"question_type": "causal_explanation", "companies": ["NVDA"]},
        "requirement_summary": {},
        "analytical_claims": [
            {
                "tier": "evidence_backed",
                "text": "Networking revenue grew from InfiniBand and Ethernet products. [T1]",
                "citation_refs": ["T1"],
                "claim_scope": "segment",
            },
            {
                "tier": "evidence_inferred",
                "text": "分部/产品证据说明网络业务增长有产品驱动线索，但不能证明总公司归因。",
                "citation_refs": ["T1"],
            },
            {
                "tier": "hypothesis_to_verify",
                "text": "待验证假设：Spectrum-X 与 Blackwell 系统放量是否继续推动网络收入。",
                "citation_refs": [],
            },
        ],
    }

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")

    assert "已验证事实" in rendered
    assert "分部/产品层面驱动" in rendered
    assert "可引用推断" in rendered
    assert "待验证假设" in rendered
    assert "不能推出的公司级结论" in rendered
    assert "不能单独证明总公司营收增长" in rendered


def test_build_methodology_answer_uses_dimension_packet_symmetrically():
    answer = build_methodology_answer(
        _methodology_packet(),
        comparison_judgment_frame={"preferred_company": "AAPL", "scale_winner": "AMZN", "companies": ["AAPL", "AMZN"]},
        lang="zh",
    )

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    rendered = render_synthesis_text(
        {"methodology_answer": payload},
        lang="zh",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )

    assert payload["framework_id"] == "fundamental_quality_analysis"
    metric_rows = {row["metric_id"]: row for row in payload["metric_table"]}
    assert set(metric_rows) == {"revenue", "net_income", "net_margin"}
    assert set(metric_rows["net_margin"]["company_values"]) == {"AAPL", "AMZN"}
    assert metric_rows["net_margin"]["company_values"]["AAPL"] == "26.67%"
    assert metric_rows["net_margin"]["company_values"]["AMZN"] == "14.67%"
    assert "核心指标对比" in rendered
    assert "| 指标 | AAPL | AMZN | 当前判断 |" in rendered
    assert "收入质量" in rendered
    assert "盈利质量" in rendered
    assert "竞争" in rendered
    assert "估值边界" in rendered
    assert "AAPL" in rendered and "AMZN" in rendered
    assert "净利润" in rendered and "净利率" in rendered
    assert "AAPL 净利率" in rendered
    assert "AMZN 净利率" in rendered
    assert "AAPL 收入" in rendered
    assert "AMZN 收入" in rendered
    risk_section = next(item for item in payload["dimension_sections"] if item["dimension_id"] == "moat_and_competitive_risk")
    assert "AAPL" in risk_section["summary"]
    assert "AMZN" in risk_section["summary"]
    assert "Apple highlighted" not in rendered
    assert "Amazon discussed" not in rendered
    assert "主要风险来自" in rendered
    assert "当前缺少估值证据" in rendered
    assert "REQ-" not in rendered
    assert "required_evidence_missing" not in rendered


def test_risk_theme_extraction_requires_citation():
    answer = build_risk_focused_answer(_risk_focused_packet(), lang="zh")

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    refs = {ref for theme in [payload["top_risk"], *payload["secondary_risks"]] for ref in theme["evidence_refs"]}
    assert refs <= {"T1", "T2", "T3"}
    assert "" not in refs
    assert payload["top_risk"]["evidence_refs"]


def test_risk_theme_extraction_ignores_business_model_without_risk_text():
    packet = _risk_focused_packet()
    packet["text_snippets"] = [
        {
            "evidence_id": "T1",
            "ticker": "NVDA",
            "dimension_id": "business_model",
            "section": "ITEM_1",
            "claim": "NVDA describes competitive products and customer markets.",
            "supporting_snippet": "NVDA describes competitive products and customer markets.",
        }
    ]

    answer = build_risk_focused_answer(packet, lang="zh")

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    assert "top_risk" not in payload
    assert payload["risk_ranking"] == []
    assert "缺少足够的已验证风险文本证据" in payload["direct_judgment"]


def test_generic_risk_text_does_not_create_empty_core_ranking():
    packet = _risk_focused_packet()
    packet["text_snippets"] = [
        {
            "evidence_id": "T1",
            "ticker": "NVDA",
            "dimension_id": "moat_and_competitive_risk",
            "section": "ITEM_1A",
            "claim": "Risk factors should be read together with validated company disclosures.",
            "supporting_snippet": "Risk factors should be read together with validated company disclosures.",
        }
    ]

    answer = build_risk_focused_answer(packet, lang="zh")

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    assert payload["risk_ranking"] == []
    assert "公司披露的主要经营风险" not in str(payload)
    assert "当前风险文本过于泛化，无法可靠排序" in payload["direct_judgment"]


def test_risk_theme_ranking_selects_top_risk():
    answer = build_risk_focused_answer(_risk_focused_packet(), lang="zh")

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    assert payload["top_risk"]["theme_id"] == "product_demand_uncertainty"
    assert "需求" in payload["top_risk"]["theme_name"]
    assert payload["secondary_risks"]


def test_scenario_risk_query_is_scenario_first_and_cited():
    packet = _risk_focused_packet()
    packet["user_query"] = "如果下季度经济放缓，NVDA 最大的财务风险是什么？"
    packet["text_snippets"].append(
        {
            "evidence_id": "T4",
            "ticker": "NVDA",
            "dimension_id": "moat_and_competitive_risk",
            "section": "ITEM_1A",
            "claim": "Macroeconomic conditions and customer spending changes may affect demand and revenue growth.",
            "supporting_snippet": "Macroeconomic conditions and customer spending changes may affect demand and revenue growth.",
        }
    )

    answer = build_risk_focused_answer(packet, lang="zh")

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    rendered = render_synthesis_text(
        {"risk_focused_answer": payload},
        lang="zh",
        answer_mode="risk_focused_analysis",
        safety_intent="normal",
    )
    assert payload["scenario_anchor"]["id"] == "economic_slowdown"
    assert "经济放缓" in payload["direct_judgment"]
    assert payload["risk_ranking"]
    assert all(any(str(ref).startswith("T") for ref in item["evidence_refs"]) for item in payload["risk_ranking"])
    assert "经济放缓" in rendered
    assert "[T" in rendered


def test_economic_slowdown_without_macro_text_is_caveated():
    packet = _risk_focused_packet()
    packet["user_query"] = "如果下季度经济放缓，NVDA 最大的财务风险是什么？"

    answer = build_risk_focused_answer(packet, lang="zh")

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    rendered = render_synthesis_text(
        {"risk_focused_answer": payload},
        lang="zh",
        answer_mode="risk_focused_analysis",
        safety_intent="normal",
    )
    assert payload["scenario_anchor"]["id"] == "economic_slowdown"
    assert payload["scenario_caveated"] is True
    assert "没有直接量化" in rendered or "没有完整建模" in rendered
    assert "该风险与经济放缓情景相关" not in rendered


def test_risk_answer_starts_with_direct_risk_judgment():
    packet = _risk_focused_packet()
    synthesis = build_analytical_synthesis(
        user_query="nvidia现在最大的问题是什么",
        analysis_plan={"answer_mode": "risk_focused_analysis", "analysis_scope": "single_company", "answer_policy": {}},
        evidence_plan={"task_type": "report_summary", "answer_mode": "risk_focused_analysis", "analysis_scope": "single_company", "evidence_requirements": []},
        evidence_collection_results=[],
        evidence_sufficiency={"overall_status": "focused_sufficient", "can_synthesize": True},
        valid_numeric_claims=[
            {"sentence": "NVDA revenue = 60.9B.", "evidence_ids": ["N1"]},
            {"sentence": "NVDA net income = 29.8B.", "evidence_ids": ["N2"]},
            {"sentence": "NVDA net margin = 48.93%.", "evidence_ids": ["N3"]},
        ],
        valid_text_claims=[
            {"sentence": "NVDA faces demand uncertainty and margin pressure.", "evidence_ids": ["T2"]},
            {"sentence": "NVDA faces competition and supply chain constraints.", "evidence_ids": ["T3"]},
        ],
        numeric_citations=[{"evidence_id": "N1"}, {"evidence_id": "N2"}, {"evidence_id": "N3"}],
        text_citations=[{"evidence_id": "T2"}, {"evidence_id": "T3"}],
        numeric_evidence_cards=[{"evidence_id": "N1"}, {"evidence_id": "N2"}, {"evidence_id": "N3"}],
        text_evidence_cards=[{"evidence_id": "T2"}, {"evidence_id": "T3"}],
        limitations=[],
        answer_policy={},
        answer_mode="risk_focused_analysis",
        safety_intent="normal",
        task_type="report_summary",
        lang="zh",
        evidence_packet=packet,
    ).model_dump()
    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="risk_focused_analysis", safety_intent="normal")
    view = build_synthesis_view(synthesis, answer_mode="risk_focused_analysis", task_type="report_summary", safety_intent="normal", lang="zh")

    assert synthesis["synthesis_mode"] == "risk_focused_analysis"
    assert view["kind"] == "risk_focused_analysis_brief"
    assert rendered.startswith("结论")
    assert "我认为当前最需要关注的是" in rendered
    assert "基于业务模型的风险排序" in rendered
    assert "财务传导路径" in rendered
    assert "待验证数据" in rendered
    assert "证据边界" in rendered
    assert "核心指标" not in rendered.splitlines()[:4]
    assert "买入" not in rendered and "卖出" not in rendered
    assert "The success of new product introductions" not in rendered


def test_single_company_methodology_answer_renders_fundamental_quick_analysis():
    packet = _single_company_methodology_packet()
    answer = build_methodology_answer(packet, lang="zh")

    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    rendered = render_synthesis_text(
        {"methodology_answer": payload},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert payload["analysis_scope"] == "single_company"
    metric_ids = [row["metric_id"] for row in payload["single_company_metric_table"]]
    assert metric_ids[:3] == ["revenue", "net_income", "net_margin"]
    assert {"operating_cash_flow", "free_cash_flow", "market_cap", "pe_ratio", "ps_ratio"} <= set(metric_ids)
    assert "结论" in rendered
    assert "业务定位" in rendered
    assert "收入和盈利" in rendered
    assert "现金流与估值" in rendered
    assert "主要风险" in rendered
    assert "证据边界" in rendered
    assert "[T2]" in rendered
    assert "intense competition" not in rendered
    assert "Key Points" not in rendered
    assert "财务趋势分析" not in rendered
    assert "REQ-" not in rendered
    assert "dependency_" not in rendered
    assert "required_evidence_missing" not in rendered


def test_single_company_uses_concrete_business_themes():
    answer = build_methodology_answer(_single_company_methodology_packet(), lang="zh")
    assert answer is not None
    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert "GPU" in rendered
    assert "数据中心" in rendered
    assert "业务模式背景" not in rendered
    assert "已验证披露文本风险背景" not in rendered


def test_single_company_revenue_trend_when_history_available():
    answer = build_methodology_answer(_single_company_methodology_packet(), lang="zh")
    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    revenue_row = next(row for row in payload["single_company_metric_table"] if row["metric_id"] == "revenue")

    assert "趋势为上升" in revenue_row["interpretation"]
    assert "历史趋势证据不足" not in revenue_row["interpretation"]


def test_focused_balance_sheet_metric_table_only_shows_balance_metrics_and_semantics():
    packet = deepcopy(_single_company_methodology_packet())
    packet["active_dimensions"] = ["balance_sheet_and_capital_intensity"]
    packet["dimension_status_map"] = {
        "balance_sheet_and_capital_intensity": {"status": "satisfied"},
    }
    packet["dimension_summary"] = [
        {
            "dimension_id": "balance_sheet_and_capital_intensity",
            "status": "satisfied",
            "numeric_evidence_refs": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
            "evidence_refs": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"],
        }
    ]
    packet["numeric_table"] = [
        {"evidence_id": "B1", "ticker": "NVDA", "metric": "cash", "value": 10_610_000_000, "unit": "USD"},
        {"evidence_id": "B2", "ticker": "NVDA", "metric": "total_debt", "value": 11_045_000_000, "unit": "USD"},
        {"evidence_id": "B3", "ticker": "NVDA", "metric": "net_debt", "value": 435_000_000, "unit": "USD"},
        {"evidence_id": "B4", "ticker": "NVDA", "metric": "total_assets", "value": 111_600_000_000, "unit": "USD"},
        {"evidence_id": "B5", "ticker": "NVDA", "metric": "total_liabilities", "value": 32_300_000_000, "unit": "USD"},
        {"evidence_id": "B6", "ticker": "NVDA", "metric": "shareholders_equity", "value": 79_300_000_000, "unit": "USD"},
        {"evidence_id": "B7", "ticker": "NVDA", "metric": "debt_to_equity", "value": 0.0702, "unit": "ratio"},
        {"evidence_id": "B8", "ticker": "NVDA", "metric": "capex_to_revenue", "value": 0.0321, "unit": "ratio"},
        {"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 147_810_000_000, "unit": "USD"},
        {"evidence_id": "V1", "ticker": "NVDA", "metric": "pe_ratio", "value": 100.36, "unit": "ratio"},
    ]

    answer = build_methodology_answer(packet, lang="zh")
    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    metric_rows = {row["metric_id"]: row for row in payload["single_company_metric_table"]}

    assert "revenue" not in metric_rows
    assert "pe_ratio" not in metric_rows
    assert metric_rows["net_debt"]["value"] == "$0.44B"
    assert metric_rows["debt_to_equity"]["value"] == "7.02%"
    assert all("历史趋势证据不足" not in row["interpretation"] for row in metric_rows.values())
    assert "当前资产负债判断主要基于最近一期指标，缺少多期趋势验证。" in payload["limitations"]
    assert "短期杠杆压力不高" in payload["judgment"]
    assert "债务/权益约 7.02%" in payload["judgment"]

    rendered = render_synthesis_text(
        {"methodology_answer": payload},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )
    assert "净债务约 $0.44B" in rendered
    assert "债务/权益为 7.02%" in rendered
    assert "短期杠杆压力" in rendered
    assert "不是完整偿债能力评估" in rendered
    assert "债务期限、利息支出和现金流覆盖" in rendered


def test_focused_valuation_metric_table_formats_multiples_not_percentages():
    packet = deepcopy(_single_company_methodology_packet())
    packet["active_dimensions"] = ["valuation_and_risk_boundary"]
    packet["dimension_status_map"] = {"valuation_and_risk_boundary": {"status": "satisfied"}}
    packet["dimension_summary"] = [
        {
            "dimension_id": "valuation_and_risk_boundary",
            "status": "satisfied",
            "numeric_evidence_refs": ["V1", "V2", "V3", "V4", "V5"],
            "evidence_refs": ["V1", "V2", "V3", "V4", "V5"],
        }
    ]
    packet["numeric_table"] = [
        {"evidence_id": "V1", "ticker": "NVDA", "metric": "share_price", "value": 176.24, "unit": "USD"},
        {"evidence_id": "V2", "ticker": "NVDA", "metric": "market_cap", "value": 4_311_463_935_165.4053, "unit": "USD"},
        {"evidence_id": "V3", "ticker": "NVDA", "metric": "pe_ratio", "value": 100.36, "unit": "ratio"},
        {"evidence_id": "V4", "ticker": "NVDA", "metric": "ps_ratio", "value": 63.29, "unit": "ratio"},
        {"evidence_id": "V5", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.0081, "unit": "ratio"},
        {"evidence_id": "B1", "ticker": "NVDA", "metric": "cash", "value": 10_610_000_000, "unit": "USD"},
    ]

    answer = build_methodology_answer(packet, lang="zh")
    assert answer is not None
    payload = answer.model_dump(exclude_none=True)
    metric_rows = {row["metric_id"]: row for row in payload["single_company_metric_table"]}
    rendered = render_synthesis_text(
        {"methodology_answer": payload},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert set(metric_rows) == {"share_price", "market_cap", "pe_ratio", "ps_ratio", "fcf_yield"}
    assert metric_rows["market_cap"]["value"] == "$4.31T"
    assert metric_rows["pe_ratio"]["value"] == "100.36x"
    assert metric_rows["ps_ratio"]["value"] == "63.29x"
    assert "10036.00%" not in rendered
    assert "6328.57%" not in rendered
    assert "4311463935165" not in rendered


def test_explicit_multi_dimension_metric_table_omits_unrequested_missing_rows():
    packet = deepcopy(_single_company_methodology_packet())
    packet["active_dimensions"] = ["cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"]
    packet["dimension_status_map"] = {
        "cash_flow_quality": {"status": "partial"},
        "valuation_and_risk_boundary": {"status": "partial"},
        "moat_and_competitive_risk": {"status": "missing"},
    }
    packet["dimension_summary"] = [
        {
            "dimension_id": "cash_flow_quality",
            "status": "partial",
            "numeric_evidence_refs": ["C1", "C2", "C3", "C4", "C5"],
            "evidence_refs": ["C1", "C2", "C3", "C4", "C5"],
        },
        {
            "dimension_id": "valuation_and_risk_boundary",
            "status": "partial",
            "numeric_evidence_refs": ["V1", "V2", "V3", "V4"],
            "evidence_refs": ["V1", "V2", "V3", "V4"],
        },
        {
            "dimension_id": "moat_and_competitive_risk",
            "status": "missing",
            "numeric_evidence_refs": [],
            "text_evidence_refs": [],
            "evidence_refs": [],
        },
    ]
    packet["numeric_table"] = [
        {"evidence_id": "C1", "ticker": "NVDA", "metric": "operating_cash_flow", "value": 36_188_000_000, "unit": "USD"},
        {"evidence_id": "C2", "ticker": "NVDA", "metric": "free_cash_flow", "value": 34_904_000_000, "unit": "USD"},
        {"evidence_id": "C3", "ticker": "NVDA", "metric": "capital_expenditure", "value": 1_284_000_000, "unit": "USD"},
        {"evidence_id": "C4", "ticker": "NVDA", "metric": "cfo_to_net_income", "value": 1.12, "unit": "ratio"},
        {"evidence_id": "C5", "ticker": "NVDA", "metric": "fcf_margin", "value": 0.51, "unit": "ratio"},
        {"evidence_id": "V1", "ticker": "NVDA", "metric": "adjusted_close", "value": 215.2, "unit": "USD"},
        {"evidence_id": "V2", "ticker": "NVDA", "metric": "market_cap", "value": 5_200_000_000_000, "unit": "USD"},
        {"evidence_id": "V3", "ticker": "NVDA", "metric": "pe_ratio", "value": 68.4, "unit": "ratio"},
        {"evidence_id": "V4", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.0067, "unit": "ratio"},
        {"evidence_id": "N1", "ticker": "NVDA", "metric": "net_margin", "value": 0.48, "unit": "ratio"},
        {"evidence_id": "B1", "ticker": "NVDA", "metric": "cash", "value": 10_000_000_000, "unit": "USD"},
        {"evidence_id": "B2", "ticker": "NVDA", "metric": "total_debt", "value": 9_000_000_000, "unit": "USD"},
    ]

    answer = build_methodology_answer(packet, lang="zh")

    assert answer is not None
    metric_ids = [row["metric_id"] for row in answer.model_dump(exclude_none=True)["single_company_metric_table"]]
    assert "operating_cash_flow" in metric_ids
    assert "free_cash_flow" in metric_ids
    assert "market_cap" in metric_ids
    assert "pe_ratio" in metric_ids
    assert "fcf_yield" in metric_ids
    assert "net_margin" not in metric_ids
    assert "cash" not in metric_ids
    assert "total_debt" not in metric_ids


def test_full_evidence_chinese_composite_methodology_wording_is_not_partial():
    packet = deepcopy(_single_company_methodology_packet())
    packet["active_dimensions"] = ["cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"]
    packet["dimension_status_map"] = {
        "cash_flow_quality": {"status": "satisfied"},
        "valuation_and_risk_boundary": {"status": "satisfied"},
        "moat_and_competitive_risk": {"status": "satisfied"},
    }
    packet["dimension_summary"] = [
        {
            "dimension_id": "cash_flow_quality",
            "status": "satisfied",
            "numeric_evidence_refs": ["C1", "C2", "C3", "C4", "C5"],
            "evidence_refs": ["C1", "C2", "C3", "C4", "C5"],
        },
        {
            "dimension_id": "valuation_and_risk_boundary",
            "status": "satisfied",
            "numeric_evidence_refs": ["V1", "V2", "V3", "V4"],
            "evidence_refs": ["V1", "V2", "V3", "V4"],
        },
        {
            "dimension_id": "moat_and_competitive_risk",
            "status": "satisfied",
            "text_evidence_refs": ["T2"],
            "evidence_refs": ["T2"],
        },
    ]
    packet["numeric_table"] = [
        {"evidence_id": "C1", "ticker": "NVDA", "metric": "operating_cash_flow", "value": 36_188_000_000, "unit": "USD"},
        {"evidence_id": "C2", "ticker": "NVDA", "metric": "free_cash_flow", "value": 34_904_000_000, "unit": "USD"},
        {"evidence_id": "C3", "ticker": "NVDA", "metric": "capital_expenditure", "value": 1_284_000_000, "unit": "USD"},
        {"evidence_id": "C4", "ticker": "NVDA", "metric": "cfo_to_net_income", "value": 1.12, "unit": "ratio"},
        {"evidence_id": "C5", "ticker": "NVDA", "metric": "fcf_margin", "value": 0.51, "unit": "ratio"},
        {"evidence_id": "V1", "ticker": "NVDA", "metric": "adjusted_close", "value": 215.2, "unit": "USD"},
        {"evidence_id": "V2", "ticker": "NVDA", "metric": "market_cap", "value": 5_200_000_000_000, "unit": "USD"},
        {"evidence_id": "V3", "ticker": "NVDA", "metric": "pe_ratio", "value": 68.4, "unit": "ratio"},
        {"evidence_id": "V4", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.0067, "unit": "ratio"},
    ]

    answer = build_methodology_answer(packet, lang="zh")
    assert answer is not None
    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert "现金流质量" in rendered
    assert "估值边界" in rendered
    assert "风险" in rendered
    assert "缺失维度" not in rendered
    assert "只能做有限" not in rendered
    assert "有限基本面分析" not in rendered
    assert "cash、debt" not in rendered
    assert "receivables" not in rendered
    assert "inventory" not in rendered
    assert "经营现金流覆盖净利润" in rendered
    assert "收入转化为自由现金流的能力较强" in rendered
    assert "P/E、P/S、FCF yield" in rendered
    assert "[C1]" in rendered
    assert "[V2]" in rendered
    assert "[T2]" in rendered


def test_single_company_cash_flow_missing_is_reported():
    answer = build_methodology_answer(_single_company_methodology_packet(), lang="zh")
    assert answer is not None
    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert "现金流与估值" in rendered
    assert "无法验证利润能否转化为现金" in rendered


def test_single_company_balance_sheet_missing_is_reported():
    answer = build_methodology_answer(_single_company_methodology_packet(), lang="zh")
    assert answer is not None
    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert "证据边界" in rendered
    assert "不能判断抗风险能力和资本投入强度" in rendered
    assert "现金、总债务、资本开支" in rendered
    assert "cash、debt、capex、receivables、inventory" not in rendered


def test_single_company_does_not_claim_valuation_without_data():
    answer = build_methodology_answer(_single_company_methodology_packet(), lang="zh")
    assert answer is not None
    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert "现金流与估值" in rendered
    assert "不能判断估值吸引力" in rendered
    assert "买入" not in rendered
    assert "卖出" not in rendered
    assert "推荐" not in rendered


def test_single_company_methodology_sections_render():
    answer = build_methodology_answer(_single_company_methodology_packet(), lang="zh")
    assert answer is not None
    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    for heading in ("结论", "业务定位", "收入和盈利", "现金流与估值", "主要风险", "证据边界"):
        assert heading in rendered


def test_single_company_missing_valuation_does_not_fail_whole_answer():
    packet = _single_company_methodology_packet()
    synthesis = build_analytical_synthesis(
        user_query="分析下 nvidia",
        analysis_plan={"analysis_scope": "single_company", "answer_policy": {}},
        evidence_plan={
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "analysis_scope": "single_company",
            "evidence_requirements": [],
        },
        evidence_collection_results=[],
        evidence_sufficiency={
            "overall_status": "partial",
            "can_synthesize": True,
            "dimension_status_map": packet["dimension_status_map"],
            "degradation_reason": "valuation_evidence_missing",
        },
        valid_numeric_claims=[
            {"sentence": "NVDA revenue = 60.9B.", "evidence_ids": ["N1"]},
            {"sentence": "NVDA net income = 29.8B.", "evidence_ids": ["N2"]},
            {"sentence": "NVDA net margin = 48.93%.", "evidence_ids": ["N3"]},
        ],
        valid_text_claims=[
            {"sentence": "NVIDIA faces competition and demand uncertainty.", "evidence_ids": ["T2"]},
        ],
        numeric_citations=[{"evidence_id": "N1"}, {"evidence_id": "N2"}, {"evidence_id": "N3"}],
        text_citations=[{"evidence_id": "T2"}],
        numeric_evidence_cards=[{"evidence_id": "N1"}, {"evidence_id": "N2"}, {"evidence_id": "N3"}],
        text_evidence_cards=[{"evidence_id": "T2"}],
        limitations=[],
        answer_policy={},
        answer_mode="analytical",
        safety_intent="normal",
        task_type="report_summary",
        lang="zh",
        evidence_packet=packet,
    ).model_dump()
    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")
    view = build_synthesis_view(synthesis, answer_mode="analytical", task_type="report_summary", safety_intent="normal", lang="zh")

    assert synthesis["synthesis_mode"] == "methodology_single_company"
    assert synthesis["degradation_reason"] == "valuation_evidence_missing"
    assert view["kind"] == "methodology_single_company_brief"
    assert view["single_company_metric_table"]
    assert "结论" in rendered
    assert "现金流与估值" in rendered
    assert "不能判断估值吸引力" in rendered
    assert "limited_judgment" not in rendered


def test_composite_methodology_render_takes_priority_over_shallow_analyst_draft():
    packet = deepcopy(_single_company_methodology_packet())
    packet["active_dimensions"] = ["cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"]
    packet["dimension_status_map"] = {
        "cash_flow_quality": {"status": "satisfied"},
        "valuation_and_risk_boundary": {"status": "satisfied"},
        "moat_and_competitive_risk": {"status": "satisfied"},
    }
    packet["dimension_summary"] = [
        {"dimension_id": "cash_flow_quality", "status": "satisfied", "numeric_evidence_refs": ["C1", "C2", "C3", "C4", "C5"], "evidence_refs": ["C1", "C2", "C3", "C4", "C5"]},
        {"dimension_id": "valuation_and_risk_boundary", "status": "satisfied", "numeric_evidence_refs": ["V1", "V2", "V3", "V4"], "evidence_refs": ["V1", "V2", "V3", "V4"]},
        {"dimension_id": "moat_and_competitive_risk", "status": "satisfied", "text_evidence_refs": ["T2"], "evidence_refs": ["T2"]},
    ]
    packet["numeric_table"] = [
        {"evidence_id": "C1", "ticker": "NVDA", "metric": "operating_cash_flow", "value": 36_188_000_000, "unit": "USD"},
        {"evidence_id": "C2", "ticker": "NVDA", "metric": "free_cash_flow", "value": 34_904_000_000, "unit": "USD"},
        {"evidence_id": "C3", "ticker": "NVDA", "metric": "capital_expenditure", "value": 1_284_000_000, "unit": "USD"},
        {"evidence_id": "C4", "ticker": "NVDA", "metric": "cfo_to_net_income", "value": 1.12, "unit": "ratio"},
        {"evidence_id": "C5", "ticker": "NVDA", "metric": "fcf_margin", "value": 0.51, "unit": "ratio"},
        {"evidence_id": "V1", "ticker": "NVDA", "metric": "adjusted_close", "value": 215.2, "unit": "USD"},
        {"evidence_id": "V2", "ticker": "NVDA", "metric": "market_cap", "value": 5_200_000_000_000, "unit": "USD"},
        {"evidence_id": "V3", "ticker": "NVDA", "metric": "pe_ratio", "value": 68.4, "unit": "ratio"},
        {"evidence_id": "V4", "ticker": "NVDA", "metric": "fcf_yield", "value": 0.0067, "unit": "ratio"},
    ]
    methodology_answer = build_methodology_answer(packet, lang="zh").model_dump(exclude_none=True)
    synthesis = {
        "short_answer": "模型判断更深入。",
        "accepted_draft": {
            "framework_summary": "基本面分析框架。",
            "dimension_analyses": [
                {
                    "dimension_id": "cash_flow_quality",
                    "status": "satisfied",
                    "claim": "现金流质量较强，但仍需要跟踪资本开支变化。",
                    "evidence_refs": ["N1", "N2"],
                }
            ],
            "tentative_conclusion": {
                "statement": "模型判断更深入。",
                "citation_refs": ["N1", "N2"],
            },
            "decision_basis": [
                {"statement": "现金流数据支持该判断。", "citation_refs": ["N1", "N2"]}
            ],
            "uncertainty_notes": [
                {"statement": "结构化数据可信度仍需结合披露文本。", "citation_refs": []}
            ],
            "citation_refs": ["N1", "N2"],
        },
        "methodology_answer": methodology_answer,
        "analysis": [],
        "risks_or_uncertainties": [],
        "limitations": [],
        "final_answer_source": "analyst_draft_initial",
        "draft_status": "accepted",
        "draft_final_status": "passed",
        "synthesis_mode": "methodology_single_company",
    }

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")
    view = build_synthesis_view(synthesis, answer_mode="analytical", task_type="report_summary", safety_intent="normal", lang="zh")

    assert "单公司组合分析" in rendered
    assert "总体判断" in rendered
    assert "现金流质量" in rendered
    assert "估值边界" in rendered
    assert "风险" in rendered
    assert "简短结论" not in rendered
    assert view["kind"] == "methodology_single_company_brief"
    assert view["dimension_sections"]


def test_single_company_risk_requires_text_evidence():
    packet = _single_company_methodology_packet()
    packet["dimension_status_map"]["moat_and_competitive_risk"] = {"status": "missing"}
    for item in packet["dimension_summary"]:
        if item["dimension_id"] == "moat_and_competitive_risk":
            item["status"] = "missing"
            item["text_evidence_refs"] = []
            item["evidence_refs"] = []
            item["limitation"] = "当前缺少风险文本证据，不能做具体风险判断。"
    packet["text_snippets"] = [row for row in packet["text_snippets"] if row["evidence_id"] != "T2"]

    answer = build_methodology_answer(packet, lang="zh")
    assert answer is not None
    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert "风险" in rendered
    assert "当前缺少风险文本证据，不能做具体风险判断。" in rendered
    assert "demand uncertainty" not in rendered


def test_methodology_comparison_mode_when_only_valuation_missing():
    packet = _methodology_packet()
    synthesis = build_analytical_synthesis(
        user_query="AAPL vs AMZN",
        analysis_plan={"answer_policy": {}},
        evidence_plan={"task_type": "company_comparison", "answer_mode": "comparison_brief", "evidence_requirements": []},
        evidence_collection_results=[],
        evidence_sufficiency={
            "overall_status": "partial",
            "can_synthesize": True,
            "dimension_status_map": packet["dimension_status_map"],
            "degradation_reason": "valuation_evidence_missing",
        },
        valid_numeric_claims=[
            {"sentence": "AAPL net margin = 26.67%.", "evidence_ids": ["N3"]},
            {"sentence": "AMZN net margin = 14.67%.", "evidence_ids": ["N6"]},
        ],
        valid_text_claims=[
            {"sentence": "Apple highlighted margin discipline.", "evidence_ids": ["T1"]},
            {"sentence": "Amazon discussed revenue scale.", "evidence_ids": ["T2"]},
        ],
        numeric_citations=[{"evidence_id": "N3"}, {"evidence_id": "N6"}],
        text_citations=[{"evidence_id": "T1"}, {"evidence_id": "T2"}],
        numeric_evidence_cards=[{"evidence_id": "N3"}, {"evidence_id": "N6"}],
        text_evidence_cards=[{"evidence_id": "T1"}, {"evidence_id": "T2"}],
        limitations=[],
        answer_policy={},
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        task_type="company_comparison",
        lang="zh",
        comparison_judgment_frame={"preferred_company": "AAPL", "scale_winner": "AMZN", "companies": ["AAPL", "AMZN"]},
        evidence_packet=packet,
    ).model_dump()

    rendered = render_synthesis_text(
        synthesis,
        lang="zh",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )
    view = build_synthesis_view(
        synthesis,
        answer_mode="comparison_brief",
        task_type="company_comparison",
        safety_intent="investment_advice_like",
        lang="zh",
    )

    assert synthesis["synthesis_mode"] == "methodology_comparison"
    assert synthesis["degradation_reason"] == "valuation_evidence_missing"
    assert synthesis["methodology_answer"]["dimension_sections"]
    assert synthesis["methodology_answer"]["metric_table"]
    assert view["kind"] == "methodology_comparison_brief"
    assert view["metric_table"]
    assert "比较判断" in rendered
    assert "维度分析" in rendered
    assert "反方观点" in rendered
    assert "限制" in rendered
    assert "Key Points" not in rendered
    assert "required_evidence_missing" not in rendered


def test_methodology_comparison_shows_missing_symmetric_metric():
    packet = _methodology_packet()
    packet["numeric_table"] = [row for row in packet["numeric_table"] if row["evidence_id"] != "N6"]
    answer = build_methodology_answer(
        packet,
        comparison_judgment_frame={"preferred_company": "AAPL", "scale_winner": "AMZN", "companies": ["AAPL", "AMZN"]},
        lang="zh",
    )

    rendered = render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )

    assert "AAPL 净利率" in rendered
    assert "AMZN 净利率 缺少可验证数据" in rendered
    metric_rows = {row["metric_id"]: row for row in answer.model_dump(exclude_none=True)["metric_table"]}
    assert metric_rows["net_margin"]["company_values"]["AMZN"] == "缺少可验证数据"


def test_synthesis_uses_validated_numeric_claims_as_key_facts():
    synthesis = build_analytical_synthesis(
        user_query="AAPL outlook",
        analysis_plan={"answer_policy": {}},
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[_text_claim()],
        numeric_citations=[{"evidence_id": "N1"}],
        text_citations=[{"evidence_id": "T1"}],
        numeric_evidence_cards=[{"evidence_id": "N1"}],
        text_evidence_cards=[{"evidence_id": "T1"}],
        limitations=[],
        answer_policy={},
        answer_mode="analytical",
        safety_intent="normal",
        task_type="report_summary",
        lang="en",
    ).model_dump()

    assert synthesis["key_facts"][0]["sentence"] == _numeric_claim()["sentence"]
    assert synthesis["analysis"][0]["sentence"] == _text_claim()["sentence"]
    assert synthesis["synthesis_strategy"] == "deterministic_numeric_plus_llm_text"


def test_synthesis_rejects_unvalidated_number_in_llm_proposal():
    synthesis = validate_synthesis_output(
        {
            "short_answer": "AAPL revenue is 999 USD.",
            "key_facts": [{"sentence": _numeric_claim()["sentence"], "claim_ids": ["N1"]}],
            "analysis": [{"sentence": "Revenue will be 999 USD next year.", "claim_ids": ["T1"]}],
            "risks_or_uncertainties": [],
            "limitations": [],
            "citations": [],
            "synthesis_strategy": "deterministic_numeric_plus_llm_text",
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[_text_claim()],
        answer_mode="analytical",
        safety_intent="normal",
        lang="en",
    ).model_dump()

    assert synthesis["analysis"] == []
    assert any(item["reason"] == "synthesis_unvalidated_number" for item in synthesis["unsupported_synthesis_items"])
    assert synthesis["short_answer"] == _numeric_claim()["sentence"]


def test_synthesis_accepts_scaled_chinese_financial_numbers():
    synthesis = validate_synthesis_output(
        {
            "short_answer": "经营现金流为361.9亿美元。",
            "key_facts": [{"sentence": "NVDA operating cash flow = $36.19B.", "claim_ids": ["N1"]}],
            "analysis": [{"sentence": "最近季度经营现金流361.9亿美元，显示现金生成能力较强。", "claim_ids": ["N1"]}],
            "risks_or_uncertainties": [],
            "limitations": [],
            "citations": [],
            "synthesis_mode": "methodology_single_company",
        },
        valid_numeric_claims=[{"sentence": "NVDA operating cash flow = $36.19B.", "evidence_ids": ["N1"]}],
        valid_text_claims=[_text_claim()],
        answer_mode="analytical",
        safety_intent="normal",
        lang="zh",
    ).model_dump()

    assert not synthesis["unsupported_synthesis_items"]
    assert synthesis["analysis"][0]["sentence"].startswith("最近季度经营现金流361.9亿美元")


def test_synthesis_accepts_numbers_from_valid_text_claims():
    synthesis = validate_synthesis_output(
        {
            "short_answer": "数据中心网络营收增长 142%，网络营收达 82 亿美元，同比增长 162%。",
            "key_facts": [],
            "analysis": [
                {"sentence": "数据中心网络营收增长 142%，受 NVLink 推动。", "claim_ids": ["T1"]},
                {"sentence": "网络营收达 82 亿美元，同比增长 162%。", "claim_ids": ["T2"]},
            ],
            "risks_or_uncertainties": [],
            "limitations": [],
            "citations": [],
            "synthesis_mode": "methodology_single_company",
        },
        valid_numeric_claims=[],
        valid_text_claims=[
            {
                "sentence": "Revenue from Data Center networking grew 142% driven by NVLink compute fabric.",
                "evidence_ids": ["T1"],
            },
            {
                "sentence": "Networking revenue was $8.2 billion, up 162% from a year ago from NVLink compute fabric.",
                "evidence_ids": ["T2"],
            },
        ],
        answer_mode="analytical",
        safety_intent="normal",
        lang="zh",
    ).model_dump()

    assert not synthesis["unsupported_synthesis_items"]
    assert len(synthesis["analysis"]) == 2


def test_synthesis_allows_numeric_dimension_analysis_without_text_ref():
    synthesis = validate_synthesis_output(
        {
            "short_answer": "估值倍数较高。",
            "key_facts": [{"sentence": "NVDA P/E = 121.75x.", "claim_ids": ["N31"]}],
            "analysis": [{"sentence": "估值边界维度满足要求，提供了非建议性比较边界。", "claim_ids": ["N31", "N32"]}],
            "risks_or_uncertainties": [],
            "limitations": [],
            "citations": [],
            "synthesis_mode": "methodology_single_company",
        },
        valid_numeric_claims=[
            {"sentence": "NVDA P/E = 121.75x.", "evidence_ids": ["N31"]},
            {"sentence": "NVDA P/S = 76.77x.", "evidence_ids": ["N32"]},
        ],
        valid_text_claims=[_text_claim()],
        answer_mode="analytical",
        safety_intent="normal",
        lang="zh",
    ).model_dump()

    assert not synthesis["unsupported_synthesis_items"]
    assert synthesis["analysis"][0]["claim_ids"] == ["N31", "N32"]


def test_text_analysis_requires_validated_text_claim_reference():
    synthesis = validate_synthesis_output(
        {
            "short_answer": _numeric_claim()["sentence"],
            "key_facts": [{"sentence": _numeric_claim()["sentence"], "claim_ids": ["N1"]}],
            "analysis": [{"sentence": "Management discussed demand risk.", "claim_ids": ["T999"]}],
            "risks_or_uncertainties": [],
            "limitations": [],
            "citations": [],
            "synthesis_strategy": "deterministic_numeric_plus_llm_text",
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[_text_claim()],
        answer_mode="analytical",
        safety_intent="normal",
        lang="en",
    ).model_dump()

    assert synthesis["analysis"] == []
    assert any(item["reason"] == "text_analysis_without_validated_text_claim" for item in synthesis["unsupported_synthesis_items"])


def test_cautious_outlook_includes_caution_and_no_prediction_language():
    synthesis = build_analytical_synthesis(
        user_query="AAPL outlook",
        analysis_plan={"answer_policy": {"forward_looking_caution": True}},
        evidence_sufficiency={
            "overall_status": "partial",
            "satisfied_requirements": ["REQ-NUM-AAPL-OUTLOOK"],
            "missing_requirements": ["REQ-TEXT-AAPL-MDA"],
            "degradation_reason": "limited_outlook",
            "can_synthesize": True,
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[],
        numeric_citations=[{"evidence_id": "N1"}],
        text_citations=[],
        numeric_evidence_cards=[{"evidence_id": "N1"}],
        text_evidence_cards=[],
        limitations=[],
        answer_policy={"forward_looking_caution": True},
        answer_mode="cautious_outlook",
        safety_intent="normal",
        task_type="fact_qa",
        lang="en",
    ).model_dump()

    rendered = render_synthesis_text(synthesis, lang="en", answer_mode="cautious_outlook", safety_intent="normal")
    assert "not a prediction" in rendered.lower()
    assert synthesis["synthesis_mode"] == "limited_outlook"


def test_investment_advice_like_rejects_buy_sell_wording():
    synthesis = validate_synthesis_output(
        {
            "short_answer": "You should buy AAPL.",
            "key_facts": [{"sentence": _numeric_claim()["sentence"], "claim_ids": ["N1"]}],
            "analysis": [{"sentence": "You should buy AAPL.", "claim_ids": ["T1"]}],
            "risks_or_uncertainties": [],
            "limitations": [],
            "citations": [],
            "synthesis_strategy": "deterministic_numeric_plus_llm_text",
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[_text_claim()],
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        lang="en",
    ).model_dump()

    rendered = render_synthesis_text(synthesis, lang="en", answer_mode="comparison_brief", safety_intent="investment_advice_like")
    assert "should buy" not in rendered.lower()
    assert any(item["reason"] == "investment_advice_wording" for item in synthesis["unsupported_synthesis_items"])


def test_synthesis_rejects_deterministic_short_term_forecast_wording():
    synthesis = validate_synthesis_output(
        {
            "short_answer": "AAPL will rise tomorrow.",
            "key_facts": [{"sentence": _numeric_claim()["sentence"], "claim_ids": ["N1"]}],
            "analysis": [{"sentence": "AAPL will rise tomorrow.", "claim_ids": ["T1"]}],
            "risks_or_uncertainties": [],
            "limitations": [],
            "citations": [],
            "synthesis_strategy": "deterministic_numeric_plus_llm_text",
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[_text_claim()],
        answer_mode="cautious_outlook",
        safety_intent="normal",
        lang="en",
    ).model_dump()

    rendered = render_synthesis_text(synthesis, lang="en", answer_mode="cautious_outlook", safety_intent="normal")
    assert "will rise tomorrow" not in rendered.lower()
    assert any(item["reason"] == "unsupported_forecast_wording" for item in synthesis["unsupported_synthesis_items"])


def test_numeric_only_degradation_declares_limited_financial_comparison():
    synthesis = build_analytical_synthesis(
        user_query="AAPL vs AMZN",
        analysis_plan={"answer_policy": {}},
        evidence_plan={
            "task_type": "company_comparison",
            "answer_mode": "comparison_brief",
            "safety_intent": "investment_advice_like",
            "evidence_requirements": [
                {"requirement_id": "REQ-NUM-AAPL", "requirement_type": "numeric", "company": "AAPL", "required": True},
                {"requirement_id": "REQ-NUM-AMZN", "requirement_type": "numeric", "company": "AMZN", "required": True},
                {"requirement_id": "REQ-TEXT-AAPL", "requirement_type": "text", "company": "AAPL", "required": True},
            ],
        },
        evidence_collection_results=[
            {"requirement_id": "REQ-NUM-AAPL", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
            {"requirement_id": "REQ-NUM-AMZN", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
        ],
        evidence_sufficiency={
            "overall_status": "partial",
            "satisfied_requirements": ["REQ-NUM-AAPL", "REQ-NUM-AMZN"],
            "missing_requirements": ["REQ-TEXT-AAPL"],
            "degradation_reason": "numeric_only_comparison",
            "can_synthesize": True,
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[],
        numeric_citations=[{"evidence_id": "N1"}],
        text_citations=[],
        numeric_evidence_cards=[{"evidence_id": "N1"}],
        text_evidence_cards=[],
        limitations=[],
        answer_policy={},
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        task_type="company_comparison",
        lang="en",
    ).model_dump()

    rendered = render_synthesis_text(synthesis, lang="en", answer_mode="comparison_brief", safety_intent="investment_advice_like")
    assert "limited financial comparison" in rendered.lower()
    assert synthesis["degradation_reason"] == "numeric_only_comparison"
    assert synthesis["missing_requirements"] == ["REQ-TEXT-AAPL"]
    assert synthesis["synthesis_mode"] == "limited_judgment"


def test_missing_required_text_suppresses_risk_judgment():
    synthesis = build_analytical_synthesis(
        user_query="AAPL risks",
        analysis_plan={"answer_policy": {}},
        evidence_plan={
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "evidence_requirements": [
                {"requirement_id": "REQ-TEXT-AAPL", "requirement_type": "text", "company": "AAPL", "required": True}
            ],
        },
        evidence_collection_results=[],
        evidence_sufficiency={
            "overall_status": "insufficient",
            "missing_requirements": ["REQ-TEXT-AAPL"],
            "degradation_reason": "text_evidence_missing",
            "can_synthesize": False,
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[],
        numeric_citations=[{"evidence_id": "N1"}],
        text_citations=[],
        numeric_evidence_cards=[{"evidence_id": "N1"}],
        text_evidence_cards=[],
        limitations=[],
        answer_policy={},
        answer_mode="analytical",
        safety_intent="normal",
        task_type="report_summary",
        lang="en",
        proposed_synthesis={"risks_or_uncertainties": [{"sentence": "Demand risk is high.", "claim_ids": []}]},
    ).model_dump()

    rendered = render_synthesis_text(synthesis, lang="en", answer_mode="analytical", safety_intent="normal")
    assert "Demand risk is high" not in rendered
    assert any(item["reason"] == "risk_analysis_without_required_text_evidence" for item in synthesis["unsupported_synthesis_items"])


def test_partial_validated_text_enables_limited_analysis():
    synthesis = build_analytical_synthesis(
        user_query="苹果现在最大的问题是什么？",
        analysis_plan={"answer_policy": {}},
        evidence_plan={
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "evidence_requirements": [
                {"requirement_id": "REQ-TEXT-AAPL-RISK", "requirement_type": "text", "company": "AAPL", "required": True},
                {"requirement_id": "REQ-TEXT-AAPL-MDA", "requirement_type": "text", "company": "AAPL", "required": True},
            ],
        },
        evidence_collection_results=[
            {"requirement_id": "REQ-TEXT-AAPL-RISK", "status": "satisfied", "evidence_type": "text", "items": [{}]},
        ],
        evidence_sufficiency={
            "overall_status": "partial",
            "satisfied_requirements": ["REQ-TEXT-AAPL-RISK"],
            "missing_requirements": ["REQ-TEXT-AAPL-MDA"],
            "degradation_reason": "text_evidence_partial",
            "can_synthesize": True,
        },
        valid_numeric_claims=[],
        valid_text_claims=[_text_claim()],
        numeric_citations=[],
        text_citations=[{"evidence_id": "T1"}],
        numeric_evidence_cards=[],
        text_evidence_cards=[{"evidence_id": "T1"}],
        limitations=[],
        answer_policy={},
        answer_mode="analytical",
        safety_intent="normal",
        task_type="report_summary",
        lang="en",
    ).model_dump()

    rendered = render_synthesis_text(synthesis, lang="en", answer_mode="analytical", safety_intent="normal")
    assert synthesis["synthesis_mode"] == "limited_analysis"
    assert synthesis["analysis"]
    assert "limited analysis" in rendered.lower()


def test_sufficient_evidence_gets_more_natural_short_answer():
    synthesis = build_analytical_synthesis(
        user_query="AAPL outlook",
        analysis_plan={"answer_policy": {}},
        evidence_plan={
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "evidence_requirements": [
                {"requirement_id": "REQ-NUM-AAPL", "requirement_type": "numeric", "company": "AAPL", "required": True},
                {"requirement_id": "REQ-TEXT-AAPL", "requirement_type": "text", "company": "AAPL", "required": True},
            ],
        },
        evidence_collection_results=[
            {"requirement_id": "REQ-NUM-AAPL", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
            {"requirement_id": "REQ-TEXT-AAPL", "status": "satisfied", "evidence_type": "text", "items": [{}]},
        ],
        evidence_sufficiency={
            "overall_status": "sufficient",
            "satisfied_requirements": ["REQ-NUM-AAPL", "REQ-TEXT-AAPL"],
            "missing_requirements": [],
            "can_synthesize": True,
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[_text_claim()],
        numeric_citations=[{"evidence_id": "N1"}],
        text_citations=[{"evidence_id": "T1"}],
        numeric_evidence_cards=[{"evidence_id": "N1"}],
        text_evidence_cards=[{"evidence_id": "T1"}],
        limitations=[],
        answer_policy={},
        answer_mode="analytical",
        safety_intent="normal",
        task_type="report_summary",
        lang="en",
    ).model_dump()

    assert synthesis["short_answer"].startswith("Based on validated financial data and filing evidence")
    assert synthesis["synthesis_mode"] == "validated_analysis"


def test_limited_judgment_rejects_text_based_risk_wording():
    synthesis = validate_synthesis_output(
        {
            "short_answer": "Current verifiable text evidence is missing, so this is only a limited financial comparison, not a complete analysis.",
            "key_facts": [{"sentence": _numeric_claim()["sentence"], "claim_ids": ["N1"]}],
            "analysis": [{"sentence": "Management appears better positioned on risk.", "claim_ids": ["T1"]}],
            "risks_or_uncertainties": [],
            "limitations": [{"code": "numeric_only_comparison", "message": "limited"}],
            "citations": [],
            "synthesis_strategy": "numeric_only",
            "synthesis_mode": "limited_judgment",
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[_text_claim()],
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        lang="en",
        suppress_risk_judgment=True,
    ).model_dump()

    assert synthesis["analysis"] == []
    assert any(item["reason"] == "limited_judgment_requires_numeric_only" for item in synthesis["unsupported_synthesis_items"])


def test_accepted_draft_projects_judgment_into_synthesis():
    synthesis = build_analytical_synthesis(
        user_query="aapple和amazon你最看好哪个",
        analysis_plan={"answer_policy": {}},
        evidence_plan={
            "task_type": "company_comparison",
            "answer_mode": "comparison_brief",
            "evidence_requirements": [
                {"requirement_id": "REQ-NUM-AAPL", "requirement_type": "numeric", "company": "AAPL", "required": True},
                {"requirement_id": "REQ-NUM-AMZN", "requirement_type": "numeric", "company": "AMZN", "required": True},
            ],
        },
        evidence_collection_results=[
            {"requirement_id": "REQ-NUM-AAPL", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
            {"requirement_id": "REQ-NUM-AMZN", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
        ],
        evidence_sufficiency={
            "overall_status": "partial",
            "satisfied_requirements": ["REQ-NUM-AAPL", "REQ-NUM-AMZN"],
            "missing_requirements": ["REQ-TEXT-AAPL", "REQ-TEXT-AMZN"],
            "degradation_reason": "numeric_only_comparison",
            "can_synthesize": True,
        },
        valid_numeric_claims=[
            {"sentence": "AAPL net income (2025-12-31, annual) = 32 USD.", "evidence_ids": ["N2"]},
            {"sentence": "AMZN revenue (2025-12-31, annual) = 150 USD.", "evidence_ids": ["N3"]},
        ],
        valid_text_claims=[],
        numeric_citations=[{"evidence_id": "N2"}, {"evidence_id": "N3"}],
        text_citations=[],
        numeric_evidence_cards=[{"evidence_id": "N2"}, {"evidence_id": "N3"}],
        text_evidence_cards=[],
        limitations=[{"code": "numeric_only_comparison", "message": "limited"}],
        answer_policy={},
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        task_type="company_comparison",
        lang="zh",
        final_answer_source="analyst_draft_initial",
        draft_status="accepted",
        draft_final_status="passed",
        accepted_draft={
            "tentative_conclusion": {
                "statement": "如果只基于当前财报证据，我更偏向 AAPL；但如果更看重收入规模，AMZN 仍值得关注。",
                "citation_refs": ["N2", "N3"],
            },
            "decision_basis": [{"statement": "AAPL 当前盈利表现更强。", "citation_refs": ["N2"]}],
            "supporting_points": [],
            "counterpoints": [{"statement": "AMZN 的收入规模更大。", "citation_refs": ["N3"]}],
            "risk_tradeoffs": [],
            "uncertainty_notes": [{"statement": "当前缺少可验证文本证据，因此这只是有限判断。", "citation_refs": []}],
            "safety_notes": [{"statement": "这不是投资建议。", "citation_refs": []}],
        },
    ).model_dump()

    assert synthesis["short_answer"].startswith("如果只基于当前财报证据")
    assert any("AAPL 当前盈利表现更强" in item["sentence"] for item in synthesis["analysis"])
    assert synthesis["final_answer_source"] == "analyst_draft_initial"

    rendered = render_synthesis_text(synthesis, lang="zh", answer_mode="comparison_brief", safety_intent="investment_advice_like")
    assert "比较判断" in rendered
    assert "关键依据" in rendered
    assert "反方观点" in rendered
    assert "非投资建议" in rendered
    assert "[N2]" in rendered


def test_comparison_answer_starts_with_judgment():
    rendered = render_synthesis_text(
        {
            "short_answer": "如果只基于当前财报证据，我更偏向 AAPL；但如果更看重收入规模，AMZN 仍值得关注。",
            "analysis": [{"sentence": "AAPL looks stronger on profitability quality.", "claim_ids": ["N2"]}],
            "risks_or_uncertainties": [{"sentence": "Current verifiable text evidence is incomplete.", "claim_ids": []}],
            "limitations": [{"code": "numeric_only_comparison", "message": "limited comparison"}],
            "comparison_judgment_frame": {
                "preferred_company": "AAPL",
                "scale_winner": "AMZN",
                "profitability_winner": "AAPL",
                "rationale": "AAPL looks stronger on profitability quality, while AMZN leads on revenue scale.",
            },
            "final_answer_source": "comparison_decision_fallback",
            "synthesis_mode": "limited_judgment",
        },
        lang="en",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )

    assert "Judgment" in rendered
    assert "Counterpoint" in rendered
    assert "Not Investment Advice" in rendered
    assert "revenue" not in rendered.splitlines()[1].lower()
    assert rendered.splitlines()[1].startswith("如果只基于当前财报证据")


def test_comparison_contains_counterpoint():
    synthesis = {
        "short_answer": "AMZN revenue is higher.",
        "accepted_draft": {
            "tentative_conclusion": {
                "statement": "AMZN revenue is higher.",
                "citation_refs": ["N3"],
            },
            "decision_basis": [{"statement": "AAPL net income is higher.", "citation_refs": ["N2"]}],
            "supporting_points": [],
            "counterpoints": [],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "safety_notes": [{"statement": "This is not investment advice.", "citation_refs": []}],
        },
        "analysis": [],
        "risks_or_uncertainties": [],
        "limitations": [],
        "comparison_judgment_frame": {
            "preferred_company": "AAPL",
            "profitability_winner": "AAPL",
            "scale_winner": "AMZN",
            "margin_winner": "AAPL",
            "risk_considerations": [
                {"theme_code": "competition", "label": "Competition", "companies": ["AMZN"], "evidence_refs": ["T2"]}
            ],
            "evidence_basis": [
                {"dimension": "profitability", "evidence_refs": ["N1", "N2", "N3", "N4"]},
                {"dimension": "margin", "evidence_refs": ["N5", "N6"]},
                {"dimension": "scale", "evidence_refs": ["N1", "N3"]},
                {"dimension": "risk", "evidence_refs": ["T2"]},
            ],
        },
        "final_answer_source": "analyst_draft_initial",
        "synthesis_mode": "balanced_comparison",
    }

    rendered = render_synthesis_text(synthesis, lang="en", answer_mode="comparison_brief", safety_intent="investment_advice_like")

    assert rendered.splitlines()[1].startswith("If I stay within")
    assert "Counterpoint" in rendered
    assert "AMZN still looks stronger if revenue scale matters more" in rendered
    assert "Competition" in rendered
    assert "[T2]" in rendered


def test_text_evidence_controls_risk_claims():
    no_text_synthesis = {
        "accepted_draft": {
            "tentative_conclusion": {
                "statement": "If profitability matters more, AAPL looks stronger.",
                "citation_refs": ["N2"],
            },
            "decision_basis": [{"statement": "AAPL net income is higher.", "citation_refs": ["N2", "N4"]}],
            "supporting_points": [],
            "counterpoints": [{"statement": "AMZN has larger revenue scale.", "citation_refs": ["N3"]}],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "safety_notes": [{"statement": "This is not investment advice.", "citation_refs": []}],
        },
        "analysis": [],
        "risks_or_uncertainties": [],
        "limitations": [],
        "comparison_judgment_frame": {
            "preferred_company": "AAPL",
            "profitability_winner": "AAPL",
            "scale_winner": "AMZN",
            "margin_winner": "AAPL",
            "risk_considerations": [],
            "evidence_basis": [
                {"dimension": "profitability", "evidence_refs": ["N1", "N2", "N3", "N4"]},
                {"dimension": "margin", "evidence_refs": ["N5", "N6"]},
                {"dimension": "scale", "evidence_refs": ["N1", "N3"]},
            ],
        },
        "final_answer_source": "analyst_draft_initial",
        "synthesis_mode": "limited_judgment",
    }

    rendered = render_synthesis_text(no_text_synthesis, lang="en", answer_mode="comparison_brief", safety_intent="investment_advice_like")

    assert "Risk comparison is limited" in rendered
    assert "Competition" not in rendered
    assert "regulatory" not in rendered.lower()

    with_text_synthesis = {
        "accepted_draft": {
            "tentative_conclusion": {
                "statement": "If profitability matters more, AAPL looks stronger.",
                "citation_refs": ["N2"],
            },
            "decision_basis": [{"statement": "AAPL net income is higher.", "citation_refs": ["N2", "N4"]}],
            "supporting_points": [],
            "counterpoints": [{"statement": "AMZN has larger revenue scale.", "citation_refs": ["N3"]}],
            "risk_tradeoffs": [],
            "uncertainty_notes": [],
            "safety_notes": [{"statement": "This is not investment advice.", "citation_refs": []}],
        },
        "analysis": [],
        "risks_or_uncertainties": [],
        "limitations": [],
        "comparison_judgment_frame": {
            "preferred_company": "AAPL",
            "profitability_winner": "AAPL",
            "scale_winner": "AMZN",
            "margin_winner": "AAPL",
            "risk_considerations": [
                {"theme_code": "competition", "label": "Competition", "companies": ["AMZN"], "evidence_refs": ["T2"]}
            ],
            "evidence_basis": [
                {"dimension": "profitability", "evidence_refs": ["N1", "N2", "N3", "N4"]},
                {"dimension": "margin", "evidence_refs": ["N5", "N6"]},
                {"dimension": "scale", "evidence_refs": ["N1", "N3"]},
                {"dimension": "risk", "evidence_refs": ["T2"]},
            ],
        },
        "final_answer_source": "analyst_draft_initial",
        "synthesis_mode": "balanced_comparison",
    }

    rendered_with_text = render_synthesis_text(
        with_text_synthesis,
        lang="en",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )

    assert "Competition" in rendered_with_text
    assert "[T2]" in rendered_with_text


def test_methodology_report_renders_before_legacy_output():
    synthesis = build_analytical_synthesis(
        user_query="AAPL vs AMZN",
        analysis_plan={"answer_policy": {}},
        evidence_plan={
            "task_type": "company_comparison",
            "answer_mode": "comparison_brief",
            "safety_intent": "investment_advice_like",
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-REV-AAPL",
                    "requirement_type": "numeric",
                    "company": "AAPL",
                    "required": True,
                    "dimension_id": "revenue_quality",
                    "dimension_name": "Revenue Quality",
                },
                {
                    "requirement_id": "REQ-REV-AMZN",
                    "requirement_type": "numeric",
                    "company": "AMZN",
                    "required": True,
                    "dimension_id": "revenue_quality",
                    "dimension_name": "Revenue Quality",
                },
            ],
        },
        evidence_collection_results=[
            {"requirement_id": "REQ-REV-AAPL", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
            {"requirement_id": "REQ-REV-AMZN", "status": "satisfied", "evidence_type": "numeric", "items": [{}]},
        ],
        evidence_sufficiency={
            "overall_status": "sufficient",
            "satisfied_requirements": ["REQ-REV-AAPL", "REQ-REV-AMZN"],
            "can_synthesize": True,
            "dimension_status_map": {
                "revenue_quality": {
                    "status": "satisfied",
                    "dimension_name": "Revenue Quality",
                    "satisfied_requirements": ["REQ-REV-AAPL", "REQ-REV-AMZN"],
                    "missing_requirements": [],
                }
            },
        },
        valid_numeric_claims=[_numeric_claim()],
        valid_text_claims=[],
        numeric_citations=[{"evidence_id": "N1"}],
        text_citations=[],
        numeric_evidence_cards=[{"evidence_id": "N1"}],
        text_evidence_cards=[],
        limitations=[],
        answer_policy={},
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        task_type="company_comparison",
        lang="zh",
        accepted_draft={
            "framework_summary": "基于基本面质量框架组织证据。",
            "overall_judgment": "基于当前证据，比较应保持条件化。",
            "dimension_analyses": [
                {
                    "dimension_id": "revenue_quality",
                    "status": "satisfied",
                    "claim": "AMZN 在收入规模上占优。",
                    "evidence_refs": ["N1"],
                }
            ],
            "methodology_counterpoints": ["AAPL 仍可能在其他维度更强。"],
            "methodology_limitations": ["这不是投资建议。"],
            "follow_up_metrics": ["净利率"],
        },
    ).model_dump()

    rendered = render_synthesis_text(
        synthesis,
        lang="zh",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
    )

    assert "比较判断" in rendered
    assert "维度分析" in rendered
    assert "收入质量" in rendered
    assert "REQ-" not in rendered
    assert "numeric_only" not in rendered
    assert synthesis["methodology_report"]["kind"] == "methodology_comparison_brief"
