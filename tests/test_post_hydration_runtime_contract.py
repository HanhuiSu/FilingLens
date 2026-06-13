from __future__ import annotations

from datetime import date

from scripts.report_methodology_data_coverage import build_coverage_report
from src.agent.evidence_packet import build_evidence_packet
from src.agent.evidence_sufficiency import evaluate_evidence_sufficiency
from src.agent.rendering import render_methodology_single_company_brief
from src.agent.synthesis import _trend_interpretation, build_methodology_answer, derive_synthesis_mode
from src.db.queries import FinancialFactRow, PriceHistoryRow, insert_financial_facts_batch, insert_price_history_batch
from src.db.schema import init_db


def _req(
    rid: str,
    metric: str,
    dimension_id: str,
    *,
    req_type: str = "numeric",
    required: bool = True,
) -> dict[str, object]:
    return {
        "requirement_id": rid,
        "requirement_type": req_type,
        "company": "NVDA",
        "metric": metric,
        "metrics": [metric],
        "required": required,
        "min_results": 1,
        "framework_id": "fundamental_quality_analysis",
        "dimension_id": dimension_id,
        "dimension_name": dimension_id,
    }


def _result(rid: str, status: str = "satisfied", *, evidence_type: str = "numeric") -> dict[str, object]:
    return {
        "requirement_id": rid,
        "status": status,
        "evidence_type": evidence_type,
        "items": [{"requirement_id": rid}] if status == "satisfied" else [],
    }


def _single_company_plan(requirements: list[dict[str, object]], **overrides: object) -> dict[str, object]:
    plan: dict[str, object] = {
        "user_query": "NVIDIA 估值贵不贵",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "methodology_intent": "valuation_boundary_analysis",
        "safety_intent": "normal",
        "evidence_requirements": requirements,
        "rejected_requirements": [],
    }
    plan.update(overrides)
    return plan


def _insert_fact(conn, metric: str, value: float, *, period_type: str = "annual") -> None:
    insert_financial_facts_batch(
        conn,
        [
            FinancialFactRow(
                "NVDA",
                date(2026, 1, 31),
                period_type,
                metric,
                value,
                "shares" if metric == "shares_outstanding" else "USD",
                source_provider="sec_companyfacts",
                confidence="high",
                extraction_method="xbrl_companyfacts",
            )
        ],
    )


def test_coverage_satisfied_valuation_matches_runtime_dimension_status(tmp_path):
    db_path = tmp_path / "coverage_runtime_contract.duckdb"
    conn = init_db(db_path=db_path)
    try:
        for metric, value in (
            ("revenue", 200.0),
            ("net_income", 50.0),
            ("free_cash_flow", 25.0),
            ("shares_outstanding", 10.0),
        ):
            _insert_fact(conn, metric, value)
        insert_price_history_batch(
            conn,
            [PriceHistoryRow("NVDA", date(2026, 4, 24), 100.0, 101.0, 99.0, 100.0, 100.0, 1000)],
        )
    finally:
        conn.close()

    coverage = build_coverage_report(db_path=db_path, tickers=["NVDA"])
    valuation = coverage["companies"]["NVDA"]["dimensions"]["valuation_and_risk_boundary"]
    assert valuation["status"] == "satisfied"
    assert {"share_price", "shares_outstanding", "market_cap", "pe_ratio", "ps_ratio"}.issubset(
        set(valuation["available_metrics"])
    )

    reqs = [
        _req("REQ-PRICE", "share_price", "valuation_and_risk_boundary"),
        _req("REQ-SHARES", "shares_outstanding", "valuation_and_risk_boundary"),
        _req("REQ-REV", "revenue", "valuation_and_risk_boundary"),
        _req("REQ-NI", "net_income", "valuation_and_risk_boundary"),
        _req("REQ-FCF", "free_cash_flow", "valuation_and_risk_boundary"),
        _req("REQ-MCAP", "market_cap", "valuation_and_risk_boundary", req_type="calculation"),
        _req("REQ-PE", "pe_ratio", "valuation_and_risk_boundary", req_type="calculation"),
        _req("REQ-PS", "ps_ratio", "valuation_and_risk_boundary", req_type="calculation"),
        _req("REQ-FCFY", "fcf_yield", "valuation_and_risk_boundary", req_type="calculation"),
    ]
    sufficiency = evaluate_evidence_sufficiency(
        _single_company_plan(reqs),
        [_result(str(req["requirement_id"]), evidence_type=str(req["requirement_type"])) for req in reqs],
    ).model_dump()

    runtime_valuation = sufficiency["dimension_status_map"]["valuation_and_risk_boundary"]
    assert runtime_valuation["status"] == "satisfied"
    assert runtime_valuation["required_missing"] == []
    assert "share_price" in runtime_valuation["required_available"]
    assert "market_cap" in runtime_valuation["required_available"]


def test_balance_sheet_enhanced_gaps_stay_satisfied_with_caveat():
    reqs = [
        _req("REQ-CASH", "cash_and_equivalents", "balance_sheet_and_capital_intensity"),
        _req("REQ-DEBT", "total_debt", "balance_sheet_and_capital_intensity"),
        _req("REQ-CAPEX", "capital_expenditure", "balance_sheet_and_capital_intensity"),
        _req("REQ-ASSETS", "total_assets", "balance_sheet_and_capital_intensity"),
        _req("REQ-LIAB", "total_liabilities", "balance_sheet_and_capital_intensity"),
        _req("REQ-EQUITY", "shareholders_equity", "balance_sheet_and_capital_intensity"),
    ]
    sufficiency = evaluate_evidence_sufficiency(
        _single_company_plan(
            reqs,
            methodology_intent="balance_sheet_analysis",
            user_query="NVIDIA 资产负债表安全吗",
        ),
        [_result("REQ-CASH"), _result("REQ-DEBT"), _result("REQ-CAPEX")],
    ).model_dump()

    balance = sufficiency["dimension_status_map"]["balance_sheet_and_capital_intensity"]
    assert balance["status"] == "satisfied"
    assert balance["required_missing"] == []
    assert set(balance["required_available"]) == {"capital_expenditure", "cash", "total_debt"}
    assert {"total_assets", "total_liabilities", "shareholders_equity"}.issubset(set(balance["enhanced_missing"]))
    limitation = balance["limitation"]
    assert "核心资产负债证据可用" in limitation
    assert "缺少现金、债务或资本开支" not in limitation


def test_focused_balance_sheet_query_uses_available_dimension_instead_of_generic_insufficient():
    packet = {
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "selected_framework": {"id": "fundamental_quality_analysis", "active_dimension_ids": ["balance_sheet_and_capital_intensity"]},
        "active_dimensions": ["balance_sheet_and_capital_intensity"],
        "dimension_status_map": {
            "balance_sheet_and_capital_intensity": {
                "dimension_id": "balance_sheet_and_capital_intensity",
                "status": "partial",
                "required_available": ["cash", "total_debt", "capital_expenditure"],
                "required_missing": [],
                "enhanced_missing": ["total_assets", "total_liabilities", "shareholders_equity"],
                "limitation": "已有现金、债务和资本开支证据，但缺少总资产、总负债和股东权益，因此只能做有限资产负债判断。",
            }
        },
        "dimension_summary": [
            {
                "dimension_id": "balance_sheet_and_capital_intensity",
                "status": "partial",
                "numeric_evidence_refs": ["N1", "N2", "N3", "N4"],
                "evidence_refs": ["N1", "N2", "N3", "N4"],
            }
        ],
        "numeric_table": [
            {"evidence_id": "N1", "ticker": "NVDA", "metric": "cash", "value": 30.0, "display_value": "$30.00B"},
            {"evidence_id": "N2", "ticker": "NVDA", "metric": "total_debt", "value": 10.0, "display_value": "$10.00B"},
            {"evidence_id": "N3", "ticker": "NVDA", "metric": "capital_expenditure", "value": 5.0, "display_value": "$5.00B"},
            {"evidence_id": "N4", "ticker": "NVDA", "metric": "inventory", "value": 8.0, "display_value": "$8.00B"},
        ],
    }

    answer = build_methodology_answer(packet, lang="zh")
    assert answer is not None
    rendered = render_methodology_single_company_brief(answer.model_dump(exclude_none=True), lang="zh")

    assert "资产负债与资本强度" in rendered
    assert "现金及等价物为 $30.00B" in rendered
    assert "总债务为 $10.00B" in rendered
    assert "资本开支为 $5.00B" in rendered
    assert "当前验证证据不足" not in rendered
    assert "无法形成可靠分析结论" not in rendered

    mode = derive_synthesis_mode(
        answer_mode="analytical",
        task_type="report_summary",
        safety_intent="normal",
        evidence_sufficiency={
            "overall_status": "partial",
            "can_synthesize": True,
            "dimension_status_map": packet["dimension_status_map"],
        },
        has_validated_numeric=True,
        has_validated_text=False,
    )
    assert mode == "methodology_single_company"


def test_business_model_satisfied_answer_does_not_claim_business_text_is_missing():
    packet = {
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "selected_framework": {"id": "fundamental_quality_analysis", "active_dimension_ids": ["business_model"]},
        "active_dimensions": ["business_model"],
        "dimension_status_map": {
            "business_model": {
                "dimension_id": "business_model",
                "status": "satisfied",
                "required_available": ["business_text"],
                "required_missing": [],
                "limitation": None,
            }
        },
        "dimension_summary": [
            {
                "dimension_id": "business_model",
                "status": "satisfied",
                "text_evidence_refs": ["T1"],
                "evidence_refs": ["T1"],
            }
        ],
        "text_snippets": [
            {
                "evidence_id": "T1",
                "ticker": "NVDA",
                "section": "ITEM_1",
                "claim": "NVDA provides GPUs, accelerated computing platforms, data center systems, gaming products, professional visualization, and automotive products.",
                "supporting_snippet": "NVIDIA provides GPUs and accelerated computing platforms, products, services, and solutions for gaming, data center, professional visualization, and automotive markets.",
            }
        ],
    }

    answer = build_methodology_answer(packet, lang="zh")
    assert answer is not None
    rendered = render_methodology_single_company_brief(answer.model_dump(exclude_none=True), lang="zh")

    assert "业务模式" in rendered
    assert "GPU" in rendered or "数据中心" in rendered
    assert "业务文本证据不足" not in rendered
    assert "不能具体拆解公司靠什么赚钱" not in rendered


def test_mixed_annual_and_quarterly_periods_do_not_support_trend_claim():
    trend = _trend_interpretation(
        "revenue",
        [
            {"metric": "revenue", "value": 147.81, "display_value": "$147.81B", "period_type": "annual", "period_end": "2026-01-31"},
            {"metric": "revenue", "value": 68.13, "display_value": "$68.13B", "period_type": "quarterly", "period_end": "2025-10-31"},
        ],
        "zh",
    )

    assert trend == "当前期间口径不一致，因此不做趋势结论"


def test_reconciliation_warning_and_valuation_dependencies_enter_evidence_packet():
    sufficiency = {
        "overall_status": "sufficient",
        "can_synthesize": True,
        "dimension_status_map": {
            "valuation_and_risk_boundary": {
                "dimension_id": "valuation_and_risk_boundary",
                "status": "satisfied",
                "required_available": ["share_price", "market_cap", "pe_ratio", "ps_ratio"],
                "required_missing": [],
            }
        },
        "dimension_status_by_id": {},
        "covered_dimensions": ["valuation_and_risk_boundary"],
        "satisfied_dimensions": ["valuation_and_risk_boundary"],
        "partial_dimensions": [],
        "missing_dimensions": [],
        "dimension_coverage_rate": 1.0,
        "weighted_dimension_coverage_rate": 1.0,
        "framework_sufficiency_status": "sufficient",
    }
    requirement_status_map = {
        "REQ-PS": {
            "dimension_id": "valuation_and_risk_boundary",
            "status": "satisfied",
            "required": True,
        }
    }

    packet = build_evidence_packet(
        user_query="NVIDIA 估值贵不贵",
        task_type="report_summary",
        answer_mode="analytical",
        safety_intent="normal",
        analysis_scope="single_company",
        companies=["NVDA"],
        comparison_target=None,
        requested_metrics=[],
        period_query=None,
        resolved_period_context=None,
        numeric_evidence=[
            {
                "evidence_id": "N1",
                "ticker": "NVDA",
                "metric": "ps_ratio",
                "value": 5.0,
                "unit": "ratio",
                "period_type": "annual",
                "period_end": "2026-01-31",
                "source_provider": "computed",
                "confidence": "derived",
                "requirement_id": "REQ-PS",
                "share_price": 100.0,
                "price_date": "2026-04-24",
                "shares_outstanding": 10.0,
                "shares_period": "2026-01-31",
                "market_cap": 1000.0,
                "statement_period": "2026-01-31",
                "dependencies": [
                    {"metric": "market_cap", "requirement_id": "REQ-MCAP"},
                    {"metric": "revenue", "requirement_id": "REQ-REV"},
                ],
                "reconciliation_warning": "value_mismatch_gt_2pct",
            }
        ],
        text_evidence=[],
        citations=[],
        evidence_sufficiency=sufficiency,
        requirement_limitations=[],
        safety_limitations=[],
        selected_framework={
            "id": "fundamental_quality_analysis",
            "active_dimension_ids": ["valuation_and_risk_boundary"],
            "dimensions": [{"id": "valuation_and_risk_boundary", "name": "估值边界"}],
        },
        requirement_status_map=requirement_status_map,
    ).model_dump()

    row = packet["numeric_table"][0]
    assert row["reconciliation_warning"] == "value_mismatch_gt_2pct"
    assert row["price_date"] == "2026-04-24"
    assert packet["computed_metric_dependencies"]["ps_ratio"][0]["dependencies"]
    assert packet["computed_metric_dependencies"]["ps_ratio"][0]["reconciliation_warning"] == "value_mismatch_gt_2pct"
