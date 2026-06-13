from __future__ import annotations

from copy import deepcopy

from src.agent.synthesis import build_methodology_answer, render_synthesis_text


def _balance_packet() -> dict:
    return {
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "selected_framework": {"id": "fundamental_quality_analysis"},
        "active_dimensions": ["balance_sheet_and_capital_intensity"],
        "dimension_status_map": {
            "balance_sheet_and_capital_intensity": {
                "status": "satisfied",
                "required_available": ["cash", "total_debt", "capital_expenditure"],
                "required_missing": [],
                "enhanced_available": [
                    "total_assets",
                    "total_liabilities",
                    "shareholders_equity",
                    "net_debt",
                    "debt_to_equity",
                    "capex_to_revenue",
                ],
                "enhanced_missing": [],
                "supporting_evidence_ids": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
            }
        },
        "dimension_summary": [
            {
                "dimension_id": "balance_sheet_and_capital_intensity",
                "status": "satisfied",
                "numeric_evidence_refs": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
                "evidence_refs": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"],
            }
        ],
        "numeric_table": [
            {"evidence_id": "B1", "ticker": "NVDA", "metric": "cash", "value": 10_610_000_000, "unit": "USD"},
            {"evidence_id": "B2", "ticker": "NVDA", "metric": "total_debt", "value": 11_045_000_000, "unit": "USD"},
            {"evidence_id": "B3", "ticker": "NVDA", "metric": "net_debt", "value": 435_000_000, "unit": "USD"},
            {"evidence_id": "B4", "ticker": "NVDA", "metric": "total_assets", "value": 206_800_000_000, "unit": "USD"},
            {"evidence_id": "B5", "ticker": "NVDA", "metric": "total_liabilities", "value": 49_510_000_000, "unit": "USD"},
            {"evidence_id": "B6", "ticker": "NVDA", "metric": "shareholders_equity", "value": 157_290_000_000, "unit": "USD"},
            {"evidence_id": "B7", "ticker": "NVDA", "metric": "debt_to_equity", "value": 0.0702, "unit": "ratio"},
            {"evidence_id": "B8", "ticker": "NVDA", "metric": "capital_expenditure", "value": 2_190_000_000, "unit": "USD"},
            {"evidence_id": "B9", "ticker": "NVDA", "metric": "capex_to_revenue", "value": 0.0321, "unit": "ratio"},
            {"evidence_id": "R1", "ticker": "NVDA", "metric": "revenue", "value": 68_130_000_000, "unit": "USD"},
        ],
    }


def _render(packet: dict) -> str:
    answer = build_methodology_answer(packet, lang="zh")
    assert answer is not None
    return render_synthesis_text(
        {"methodology_answer": answer.model_dump(exclude_none=True)},
        lang="zh",
        answer_mode="analytical",
        safety_intent="normal",
    )


def test_capex_to_revenue_available_does_not_render_missing():
    rendered = _render(_balance_packet())

    assert "capex/revenue" in rendered
    assert "capex/revenue 缺少可验证数据" not in rendered
    assert "无法计算 capex/revenue" not in rendered
    assert "capex_to_revenue" not in rendered
    assert "3.21%" in rendered


def test_capex_to_revenue_missing_explains_missing_capex_dependency():
    packet = deepcopy(_balance_packet())
    packet["numeric_table"] = [
        row for row in packet["numeric_table"] if row["metric"] not in {"capital_expenditure", "capex_to_revenue"}
    ]
    status = packet["dimension_status_map"]["balance_sheet_and_capital_intensity"]
    status["enhanced_available"] = [metric for metric in status["enhanced_available"] if metric != "capex_to_revenue"]
    status["enhanced_missing"] = ["capex_to_revenue"]

    rendered = _render(packet)

    assert "缺少资本开支，无法计算资本开支/收入" in rendered
    assert "capex/revenue 缺少可验证数据" not in rendered


def test_capex_to_revenue_missing_explains_missing_revenue_dependency():
    packet = deepcopy(_balance_packet())
    packet["numeric_table"] = [
        row for row in packet["numeric_table"] if row["metric"] not in {"revenue", "capex_to_revenue"}
    ]
    status = packet["dimension_status_map"]["balance_sheet_and_capital_intensity"]
    status["enhanced_available"] = [metric for metric in status["enhanced_available"] if metric != "capex_to_revenue"]
    status["enhanced_missing"] = ["capex_to_revenue"]

    rendered = _render(packet)

    assert "缺少可匹配收入期间，无法计算资本开支/收入" in rendered
    assert "capex/revenue 缺少可验证数据" not in rendered


def test_capex_to_revenue_missing_explains_period_mismatch_when_dependencies_exist():
    packet = deepcopy(_balance_packet())
    packet["numeric_table"] = [row for row in packet["numeric_table"] if row["metric"] != "capex_to_revenue"]
    status = packet["dimension_status_map"]["balance_sheet_and_capital_intensity"]
    status["enhanced_available"] = [metric for metric in status["enhanced_available"] if metric != "capex_to_revenue"]
    status["enhanced_missing"] = ["capex_to_revenue"]

    rendered = _render(packet)

    assert "资本开支和收入期间不匹配，暂不计算资本开支/收入" in rendered
    assert "capex/revenue 缺少可验证数据" not in rendered


def test_user_visible_renderer_does_not_leak_internal_fields():
    packet = deepcopy(_balance_packet())
    packet["red_flags"] = [
        {
            "message": "REQ-METH-NVDA-CAPEX dependency_metric_id dependency_numeric_requirement_missing numeric_only_balance",
            "user_visible": True,
        }
    ]

    rendered = _render(packet)

    assert "REQ-" not in rendered
    assert "dependency_" not in rendered
    assert "numeric_only_" not in rendered
    assert "dependency_metric_id" not in rendered
