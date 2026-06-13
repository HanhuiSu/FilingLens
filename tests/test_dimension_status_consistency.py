from src.agent.evidence_sufficiency import build_dimension_sufficiency
from src.agent.synthesis import render_synthesis_text


def test_dimension_status_map_exposes_canonical_aliases():
    plan = {
        "analysis_scope": "single_company",
        "evidence_requirements": [
            {
                "requirement_id": "REQ-TEXT-NVDA-BUSINESS_MODEL",
                "requirement_type": "text",
                "required": True,
                "dimension_id": "business_model",
                "dimension_name": "Business Model",
            },
            {
                "requirement_id": "REQ-TEXT-NVDA-RISK",
                "requirement_type": "text",
                "required": True,
                "dimension_id": "moat_and_competitive_risk",
                "dimension_name": "Risk",
            },
        ],
    }
    status = {
        "REQ-TEXT-NVDA-BUSINESS_MODEL": {"status": "satisfied"},
        "REQ-TEXT-NVDA-RISK": {"status": "missing"},
    }
    out = build_dimension_sufficiency(plan, status)
    assert out["dimension_status_by_id"] == out["dimension_status_map"]
    assert out["covered_dimensions"] == out["satisfied_dimensions"]
    assert out["satisfied_dimensions"] == ["business_model"]
    assert out["missing_dimensions"] == ["moat_and_competitive_risk"]
    assert out["partial_dimensions"] == []
    assert out["dimension_status_by_id"]["business_model"]["status"] == "satisfied"
    item = out["dimension_status_map"]["business_model"]
    assert item["required_available"] == ["REQ-TEXT-NVDA-BUSINESS_MODEL"]
    assert item["required_missing"] == []
    assert item["supporting_evidence_ids"] == ["REQ-TEXT-NVDA-BUSINESS_MODEL"]
    assert out["weighted_dimension_coverage_rate"] == 0.5


def test_answer_uses_business_evidence_when_dimension_satisfied():
    synthesis = {
        "methodology_answer": {
            "framework_id": "fundamental_quality_analysis",
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "judgment": "基于已验证证据，NVDA 可以做有限基本面分析。",
            "dimension_sections": [
                {
                    "dimension_id": "business_model",
                    "title": "业务模式",
                    "status": "satisfied",
                    "summary": "业务模式：NVDA 的业务主要围绕 GPU、数据中心和游戏等产品与市场展开。[T1]",
                    "evidence_refs": ["T1"],
                    "limitation": None,
                }
            ],
            "limitations": [],
            "follow_up_metrics": [],
            "evidence_refs": ["T1"],
        },
        "synthesis_mode": "methodology_single_company",
    }
    answer = render_synthesis_text(synthesis, lang="zh", answer_mode="analytical", safety_intent="normal")
    assert "GPU" in answer
    assert "业务文本证据不足" not in answer
