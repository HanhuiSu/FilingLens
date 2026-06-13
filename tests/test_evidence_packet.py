from __future__ import annotations

from src.agent.evidence_packet import build_evidence_packet, summarize_evidence_packet


def _numeric_item(evidence_id: str, ticker: str, metric: str, value: float) -> dict:
    return {
        "evidence_id": evidence_id,
        "ticker": ticker,
        "metric": metric,
        "period_type": "annual",
        "period_end": "2025-12-31",
        "value": value,
        "unit": "USD",
        "provenance": "sec_companyfacts",
        "source_provider": "sec_companyfacts",
    }


def _text_item(evidence_id: str, ticker: str, section: str, snippet: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "ticker": ticker,
        "form_type": "10-K",
        "fiscal_period": "2025-12-31",
        "section": section,
        "text_snippet": snippet,
        "supporting_snippet": snippet,
    }


def test_comparison_packet_contains_validated_numeric_and_text_evidence():
    packet = build_evidence_packet(
        user_query="aapple和amazon你最看好哪个",
        task_type="company_comparison",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        companies=["AAPL"],
        comparison_target="AMZN",
        requested_metrics=["revenue", "net_income"],
        period_query={"period_type": "latest", "comparison_basis": "same_period"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[
            _numeric_item("N1", "AAPL", "revenue", 120.0),
            _numeric_item("N2", "AAPL", "net_income", 32.0),
            _numeric_item("N3", "AMZN", "revenue", 150.0),
            _numeric_item("N4", "AMZN", "net_income", 22.0),
        ],
        text_evidence=[
            _text_item("T1", "AAPL", "ITEM_7", "Apple described margin discipline and steady execution."),
            _text_item("T2", "AMZN", "ITEM_1A", "Amazon described reinvestment and competition risk."),
        ],
        citations=[{"evidence_id": "N1"}, {"evidence_id": "T1"}, {"evidence_id": "T2"}],
        evidence_sufficiency={
            "overall_status": "sufficient",
            "missing_requirements": [],
            "partial_requirements": [],
            "degradation_reason": None,
            "required_numeric_satisfied_rate": 1.0,
            "required_text_satisfied_rate": 1.0,
            "company_evidence_balance": 1.0,
        },
        requirement_limitations=[],
        safety_limitations=[{"code": "investment_advice_boundary", "message": "not advice"}],
    ).model_dump(exclude_none=True)

    assert len(packet["numeric_table"]) == 4
    assert len(packet["text_snippets"]) == 2
    assert packet["comparison_table"]
    assert {item["ticker"] for item in packet["text_snippets"]} == {"AAPL", "AMZN"}
    assert any(theme["evidence_refs"] for theme in packet["grouped_risk_themes"])
    assert any(theme["evidence_refs"] for theme in packet["grouped_business_themes"])
    assert "raw_hit_count" not in packet["text_snippets"][0]

    summary = summarize_evidence_packet(packet)
    assert summary["numeric_row_count"] == 4
    assert summary["text_snippet_count"] == 2


def test_evidence_packet_summary_exposes_text_flow_counts():
    packet = {
        "numeric_table": [],
        "comparison_table": [],
        "text_snippets": [{"evidence_id": "T1"}],
        "citations": [{"evidence_id": "T1"}],
        "text_evidence_flow_summary": {
            "text_candidate_count": 4,
            "text_pre_citation_validated_count": 3,
            "text_citable_count": 1,
            "text_final_packet_count": 1,
            "text_drop_stage_counts": {"citation_policy_dropped": 2},
        },
    }

    summary = summarize_evidence_packet(packet)

    assert summary["text_candidate_count"] == 4
    assert summary["text_pre_citation_validated_count"] == 3
    assert summary["text_citable_count"] == 1
    assert summary["text_final_packet_count"] == 1
    assert summary["text_drop_stage_counts"] == {"citation_policy_dropped": 2}


def test_evidence_packet_preserves_numeric_role_and_quality_metadata():
    row = _numeric_item("N1", "NVDA", "revenue", 68_127_000_000)
    row.update(
        {
            "source_requirement_id": "REQ-RP-NVDA-CURRENT_REVENUE_NUM",
            "period_scope": "annual",
        }
    )

    packet = build_evidence_packet(
        user_query="为什么 NVIDIA 营收增长这么多",
        task_type="report_summary",
        answer_mode="analytical",
        safety_intent="normal",
        companies=["NVDA"],
        comparison_target=None,
        requested_metrics=["revenue"],
        period_query={"period_type": "latest"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[row],
        text_evidence=[],
        citations=[{"evidence_id": "N1"}],
        evidence_sufficiency={},
        requirement_limitations=[],
        safety_limitations=[],
        requirement_status_map={
            "REQ-RP-NVDA-CURRENT_REVENUE_NUM": {
                "status": "satisfied",
                "evidence_role": "current_revenue",
                "quality_status": "valid",
            }
        },
    ).model_dump(exclude_none=True)

    numeric = packet["numeric_table"][0]
    assert numeric["role"] == "current_revenue"
    assert numeric["evidence_role"] == "current_revenue"
    assert numeric["quality_status"] == "valid"
    assert numeric["source_requirement_id"] == "REQ-RP-NVDA-CURRENT_REVENUE_NUM"
    assert numeric["requirement_ids"] == ["REQ-RP-NVDA-CURRENT_REVENUE_NUM"]


def test_evidence_packet_preserves_text_claim_metadata_for_final_bundle():
    text = _text_item("T1", "NVDA", "ITEM_1", "NVIDIA describes GPUs, data center, and gaming markets.")
    text.update(
        {
            "requirement_id": "REQ-TEXT-NVDA-BUSINESS_MODEL",
            "dimension_id": "business_model",
            "citation_ref": "T1",
            "claim": "NVDA 的业务主要围绕 GPU、数据中心和游戏等市场展开。",
            "claim_source": "deterministic_fallback",
        }
    )
    packet = build_evidence_packet(
        user_query="分析下 nvidia",
        task_type="report_summary",
        answer_mode="analytical",
        safety_intent="normal",
        analysis_scope="single_company",
        companies=["NVDA"],
        comparison_target=None,
        requested_metrics=[],
        period_query={"period_type": "latest"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[],
        text_evidence=[text],
        citations=[{"evidence_id": "T1"}],
        evidence_sufficiency={"dimension_status_map": {"business_model": {"status": "satisfied"}}},
        requirement_limitations=[],
        safety_limitations=[],
        selected_framework={"framework_id": "fundamental_quality_analysis", "active_dimension_ids": ["business_model"]},
        requirement_status_map={"REQ-TEXT-NVDA-BUSINESS_MODEL": {"dimension_id": "business_model", "status": "satisfied", "required": True}},
    ).model_dump(exclude_none=True)

    row = packet["text_snippets"][0]
    assert row["dimension_id"] == "business_model"
    assert row["citation_ref"] == "T1"
    assert row["claim_source"] == "deterministic_fallback"
    assert "GPU" in row["claim"]
    assert packet["dimension_summary"][0]["text_evidence_refs"] == ["T1"]


def test_evidence_packet_downgrades_segment_summary_scope_overclaim():
    text = _text_item(
        "T1",
        "NVDA",
        "ITEM_7",
        "Compute & Networking revenue - The year over year increase was driven by platform shifts to accelerated computing and AI.",
    )
    text["claim"] = "NVIDIA的营收增长主要由加速计算和人工智能的平台转型驱动"
    text["claim_source"] = "llm_text_claim"

    packet = build_evidence_packet(
        user_query="为什么nvidia营收增长这么多",
        task_type="report_summary",
        answer_mode="analytical",
        safety_intent="normal",
        analysis_scope="single_company",
        companies=["NVDA"],
        comparison_target=None,
        requested_metrics=[],
        period_query={"period_type": "latest"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[],
        text_evidence=[text],
        citations=[{"evidence_id": "T1"}],
        evidence_sufficiency={"overall_status": "partial"},
        requirement_limitations=[],
        safety_limitations=[],
    ).model_dump(exclude_none=True)

    row = packet["text_snippets"][0]
    assert row["claim_scope"] == "segment"
    assert row["allowed_claim_strength"] == "bounded_inference"
    assert row["summary_scope_warning"] == "evidence_summary_scope_overclaim"
    assert "Compute & Networking" in row["claim"]
    assert "分部" in row["claim"]
    assert "NVIDIA的营收增长主要由" not in row["claim"]


def test_evidence_packet_formats_computed_net_margin_as_percentage():
    row = _numeric_item("N1", "AAPL", "net_margin", 0.25)
    row["unit"] = "ratio"
    row["source_provider"] = "computed"
    row["provenance"] = "compute_metrics:net_margin:margin:net_income_over_revenue:1"

    packet = build_evidence_packet(
        user_query="AAPL margin",
        task_type="company_comparison",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        companies=["AAPL"],
        comparison_target="AMZN",
        requested_metrics=["revenue", "net_income"],
        period_query={"period_type": "latest"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[row],
        text_evidence=[],
        citations=[{"evidence_id": "N1"}],
        evidence_sufficiency={"overall_status": "partial"},
        requirement_limitations=[],
        safety_limitations=[],
    ).model_dump(exclude_none=True)

    assert packet["numeric_table"][0]["display_value"] == "25.00%"


def test_evidence_packet_formats_valuation_multiples_and_large_currency_by_metric_semantics():
    pe_row = _numeric_item("N1", "NVDA", "pe_ratio", 100.36)
    pe_row["unit"] = "ratio"
    pe_row["source_provider"] = "computed"
    market_cap_row = _numeric_item("N2", "NVDA", "market_cap", 4_311_463_935_165.4053)
    market_cap_row["source_provider"] = "computed"

    packet = build_evidence_packet(
        user_query="NVIDIA 估值贵不贵",
        task_type="report_summary",
        answer_mode="analytical",
        safety_intent="normal",
        analysis_scope="single_company",
        companies=["NVDA"],
        comparison_target=None,
        requested_metrics=[],
        period_query={"period_type": "latest"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[pe_row, market_cap_row],
        text_evidence=[],
        citations=[{"evidence_id": "N1"}, {"evidence_id": "N2"}],
        evidence_sufficiency={"overall_status": "partial"},
        requirement_limitations=[],
        safety_limitations=[],
    ).model_dump(exclude_none=True)

    rows = {row["metric"]: row for row in packet["numeric_table"]}
    assert rows["pe_ratio"]["display_value"] == "100.36x"
    assert "%" not in rows["pe_ratio"]["display_value"]
    assert rows["market_cap"]["display_value"] == "$4.31T"
    assert rows["market_cap"]["period_category"] == "annual"


def test_analytical_packet_carries_risk_and_mda_text_and_missing_summary():
    packet = build_evidence_packet(
        user_query="苹果现在最大的问题是什么？",
        task_type="report_summary",
        answer_mode="analytical",
        safety_intent="normal",
        companies=["AAPL"],
        comparison_target=None,
        requested_metrics=[],
        period_query={"period_type": "latest"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[],
        text_evidence=[
            _text_item("T1", "AAPL", "ITEM_1A", "Competition and regulation remain material risks."),
            _text_item("T2", "AAPL", "ITEM_7", "Management discussed demand softness and margin pressure."),
        ],
        citations=[{"evidence_id": "T1"}, {"evidence_id": "T2"}],
        evidence_sufficiency={
            "overall_status": "partial",
            "missing_requirements": ["REQ-TEXT-AAPL-MDA"],
            "partial_requirements": ["REQ-TEXT-AAPL-RISK"],
            "degradation_reason": "text_evidence_partial",
            "required_numeric_satisfied_rate": None,
            "required_text_satisfied_rate": 0.5,
            "company_evidence_balance": 1.0,
        },
        requirement_limitations=[{"code": "text_evidence_partial", "message": "limited"}],
        safety_limitations=[],
    ).model_dump(exclude_none=True)

    assert [item["section"] for item in packet["text_snippets"]] == ["ITEM_1A", "ITEM_7"]
    assert packet["missing_evidence_summary"]["degradation_reason"] == "text_evidence_partial"
    assert packet["grouped_risk_themes"]
    assert packet["grouped_business_themes"]


def test_evidence_packet_builds_dimension_summary_and_flags():
    packet = build_evidence_packet(
        user_query="AAPL vs AMZN",
        task_type="company_comparison",
        answer_mode="comparison_brief",
        safety_intent="investment_advice_like",
        companies=["AAPL", "AMZN"],
        comparison_target="AMZN",
        requested_metrics=["revenue", "net_income"],
        period_query={"period_type": "latest"},
        resolved_period_context={"target_period_type": "annual"},
        numeric_evidence=[
            {**_numeric_item("N1", "AAPL", "revenue", 120.0), "requirement_id": "REQ-REV-AAPL"},
            {**_numeric_item("N2", "AMZN", "revenue", 80.0), "requirement_id": "REQ-REV-AMZN"},
            {**_numeric_item("N3", "AAPL", "net_income", 30.0), "requirement_id": "REQ-PROFIT-AAPL"},
            {**_numeric_item("N4", "AMZN", "net_income", 10.0), "requirement_id": "REQ-PROFIT-AMZN"},
        ],
        text_evidence=[],
        citations=[{"evidence_id": "N1"}, {"evidence_id": "N2"}, {"evidence_id": "N3"}, {"evidence_id": "N4"}],
        evidence_sufficiency={
            "overall_status": "partial",
            "dimension_status_map": {
                "revenue_quality": {
                    "dimension_name": "Revenue Quality",
                    "status": "satisfied",
                    "satisfied_requirements": ["REQ-REV-AAPL", "REQ-REV-AMZN"],
                    "missing_requirements": [],
                    "allowed_claims": ["revenue scale comparison"],
                    "forbidden_claims": [],
                },
                "moat_and_competitive_risk": {
                    "dimension_name": "Moat And Competitive Risk",
                    "status": "missing",
                    "satisfied_requirements": [],
                    "missing_requirements": ["REQ-TEXT-AAPL"],
                    "allowed_claims": [],
                    "forbidden_claims": ["specific risk judgment"],
                    "limitation": "缺少可验证风险文本证据，不能做具体风险判断。",
                },
            },
            "covered_dimensions": ["revenue_quality"],
            "missing_dimensions": ["moat_and_competitive_risk"],
            "dimension_coverage_rate": 0.5,
            "framework_sufficiency_status": "partial",
        },
        requirement_limitations=[],
        safety_limitations=[],
        selected_framework={
            "framework_id": "fundamental_quality_analysis",
            "active_dimension_ids": ["revenue_quality", "moat_and_competitive_risk"],
        },
        requirement_status_map={
            "REQ-REV-AAPL": {"dimension_id": "revenue_quality", "status": "satisfied", "required": True},
            "REQ-REV-AMZN": {"dimension_id": "revenue_quality", "status": "satisfied", "required": True},
            "REQ-PROFIT-AAPL": {"dimension_id": "profitability_quality", "status": "satisfied", "required": True},
            "REQ-PROFIT-AMZN": {"dimension_id": "profitability_quality", "status": "satisfied", "required": True},
            "REQ-TEXT-AAPL": {"dimension_id": "moat_and_competitive_risk", "status": "missing", "required": True},
        },
    ).model_dump(exclude_none=True)

    assert packet["active_dimensions"] == ["revenue_quality", "moat_and_competitive_risk"]
    assert packet["dimension_status_by_id"] == packet["dimension_status_map"]
    for summary_key in ("missing_evidence_summary", "dimension_sufficiency"):
        summary = packet[summary_key]
        assert summary["dimension_status_by_id"] == summary["dimension_status_map"]
        assert summary["covered_dimensions"] == summary["satisfied_dimensions"]
    revenue_summary = next(item for item in packet["dimension_summary"] if item["dimension_id"] == "revenue_quality")
    assert revenue_summary["numeric_evidence_refs"] == ["N1", "N2"]
    assert revenue_summary["evidence_refs"] == ["N1", "N2"]
    assert "revenue scale comparison" in packet["allowed_claims"]
    assert "specific risk judgment" in packet["forbidden_claims"]
    assert any(flag["id"] == "text_risk_evidence_missing" for flag in packet["missing_evidence_flags"])
    assert any("风险文本证据" in item["message"] for item in packet["limitations"])
