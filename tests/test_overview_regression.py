from __future__ import annotations

from src.agent import nodes


def _legacy_overview_plan(core_count: int = 18) -> dict:
    requirements = []
    dimensions = [
        "business_model",
        "revenue_quality",
        "profitability_quality",
        "cash_flow_quality",
        "balance_sheet_and_capital_intensity",
        "moat_and_competitive_risk",
        "valuation_and_risk_boundary",
    ]
    for idx in range(core_count):
        dim = dimensions[idx % len(dimensions)]
        req_type = "text" if dim in {"business_model", "moat_and_competitive_risk"} else "numeric"
        dimension_id = f"{dim}_{idx}"
        requirements.append(
            {
                "requirement_id": f"REQ-LEGACY-{idx}",
                "requirement_type": req_type,
                "company": "AMZN",
                "metric": ["revenue", "net_income", "operating_cash_flow", "free_cash_flow", "total_debt", "market_cap"][idx % 6] if req_type == "numeric" else "",
                "metrics": [[
                    "revenue",
                    "net_income",
                    "operating_cash_flow",
                    "free_cash_flow",
                    "total_debt",
                    "market_cap",
                ][idx % 6]] if req_type == "numeric" else [],
                "section_preferences": ["ITEM_1"] if dim == "business_model" else (["ITEM_1A"] if req_type == "text" else []),
                "retrieval_query": "business segments AWS North America International" if req_type == "text" else "",
                "dimension_id": dimension_id,
                "required": True,
                "requirement_scope": "core",
            }
        )
    return {
        "user_query": "AMZN overview",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "evidence_requirements": requirements,
        "core_requirement_ids": [req["requirement_id"] for req in requirements],
    }


def test_overview_research_plan_node_merges_instead_of_replacing_legacy_core(monkeypatch):
    raw_plan = {
        "question_type": "overview",
        "user_goal": "Analyze Amazon overview.",
        "companies": ["AMZN"],
        "required_answer_parts": [
            {"id": "overview", "description": "Complete company overview.", "required": True},
        ],
        "evidence_requests": [
            {
                "id": "planner_revenue",
                "type": "numeric",
                "scope": "core",
                "company": "AMZN",
                "metrics": ["latest_revenue"],
                "answer_part_ids": ["overview"],
            },
            {
                "id": "planner_net_income",
                "type": "numeric",
                "scope": "core",
                "company": "AMZN",
                "metrics": ["profit"],
                "answer_part_ids": ["overview"],
            },
            {
                "id": "planner_business",
                "type": "text",
                "scope": "core",
                "company": "AMZN",
                "sections": ["ITEM_1"],
                "queries": ["business segments AWS North America International"],
                "answer_part_ids": ["overview"],
            },
        ],
    }
    monkeypatch.setattr(nodes, "planner_mode", lambda: "expanded")
    monkeypatch.setattr(
        nodes,
        "build_research_plan_raw",
        lambda **_kwargs: (raw_plan, {"source": "llm", "status": "ok", "duration_ms": 1}),
    )

    out = nodes.research_plan_node(
        {
            "user_query": "AMZN overview",
            "companies": ["AMZN"],
            "needs_tools": True,
            "answer_mode": "analytical",
            "task_type": "report_summary",
            "methodology_intent": "overview",
            "analysis_scope": "single_company",
            "required_dimensions": ["business_model", "revenue_quality", "cash_flow_quality"],
            "evidence_plan": _legacy_overview_plan(18),
        }
    )

    assert out["research_plan_used"]["question_type"] == "overview"
    assert out["plan_coverage_decision"]["strategy"] == "merge"
    assert out["plan_coverage_decision"]["retained_legacy_core_count"] == 18
    assert out["requirement_merge_summary"]["retained_legacy_core_count"] == 18
    assert out["evidence_plan_used"]["source"] == "merged"
    assert len(out["evidence_plan"]["evidence_requirements"]) >= 18
    assert set(_legacy_overview_plan(18)["core_requirement_ids"]) <= set(out["evidence_plan"]["core_requirement_ids"])
