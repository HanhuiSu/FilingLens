"""Shared ontology labels for semantic query understanding."""

from __future__ import annotations

from typing import Any

from src.agent.constants import ALLOWED_ANALYSIS_METRICS


ALLOWED_METHODOLOGY_INTENTS: set[str] = {
    "overview",
    "risk",
    "cash_flow",
    "profitability",
    "revenue",
    "balance_sheet",
    "valuation",
    "comparison",
    "none",
}

SUPPORTED_DIMENSIONS: set[str] = {
    "business_model",
    "revenue_quality",
    "profitability_quality",
    "cash_flow_quality",
    "balance_sheet_and_capital_intensity",
    "moat_and_competitive_risk",
    "valuation_and_risk_boundary",
}

DIMENSION_ALIASES: dict[str, str] = {
    "growth_quality": "revenue_quality",
    "growth": "revenue_quality",
    "revenue_growth_quality": "revenue_quality",
    "sales_growth_quality": "revenue_quality",
    "增长质量": "revenue_quality",
    "营收增长质量": "revenue_quality",
    "收入增长质量": "revenue_quality",
}

ALLOWED_ANALYSIS_SCOPES: set[str] = {"single_company", "comparison", "meta", "unsupported", "unknown"}
ALLOWED_USER_EXPECTATIONS: set[str] = {"quick_answer", "deep_analysis", "recommendation_like", "diagnostic", "clarification"}
ALLOWED_SAFETY_INTENTS: set[str] = {"normal", "investment_advice_like", "prediction", "unsupported"}

DETERMINISTIC_DIMENSION_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "cash_flow_quality",
        (
            "现金流",
            "自由现金流",
            "利润能不能变成现金",
            "利润转化",
            "利润含金量",
            "cash flow",
            "free cash flow",
            "cash conversion",
        ),
    ),
    (
        "valuation_and_risk_boundary",
        (
            "估值边界",
            "估值",
            "贵不贵",
            "便宜",
            "昂贵",
            "valuation boundary",
            "valuation",
            "stretched",
            "multiple",
            "expensive",
            "cheap",
            "fcf yield",
        ),
    ),
    (
        "moat_and_competitive_risk",
        (
            "主要风险",
            "风险",
            "不确定性",
            "竞争",
            "护城河",
            "risk",
            "key risk",
            "major risk",
            "uncertainty",
            "competition",
            "competitive",
            "moat",
        ),
    ),
    (
        "profitability_quality",
        (
            "盈利质量",
            "盈利能力",
            "利润率",
            "毛利率",
            "净利率",
            "profitability quality",
            "profitability",
            "margin",
        ),
    ),
    (
        "balance_sheet_and_capital_intensity",
        (
            "资产负债表",
            "资产负债",
            "资本开支",
            "资本投入",
            "债务",
            "现金储备",
            "balance sheet",
            "capital intensity",
            "capex",
            "capital expenditure",
            "debt",
        ),
    ),
    (
        "revenue_quality",
        (
            "增长质量",
            "营收增长质量",
            "收入增长质量",
            "收入质量",
            "营收质量",
            "营收",
            "收入",
            "growth quality",
            "revenue growth quality",
            "revenue quality",
            "sales quality",
            "revenue",
        ),
    ),
    (
        "business_model",
        (
            "业务模式",
            "商业模式",
            "产品和服务",
            "business model",
            "products and services",
        ),
    ),
)

METRIC_ALIASES: dict[str, str] = {
    "capex": "capital_expenditure",
    "capital_expenditures": "capital_expenditure",
    "operating_cashflow": "operating_cash_flow",
    "cash_from_operations": "operating_cash_flow",
    "cfo": "operating_cash_flow",
    "fcf": "free_cash_flow",
    "p_e": "pe_ratio",
    "pe": "pe_ratio",
    "p_s": "ps_ratio",
    "ps": "ps_ratio",
    "price_to_earnings": "pe_ratio",
    "price_to_sales": "ps_ratio",
}

SAFETY_INTENT_ALIASES: dict[str, str] = {
    "unsupported_or_out_of_scope": "unsupported",
}


def normalize_metric_label(value: Any) -> str:
    metric = str(value or "").strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    return METRIC_ALIASES.get(metric, metric)


def normalize_dimension_label(value: Any) -> str:
    dimension = str(value or "").strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    if dimension in SUPPORTED_DIMENSIONS:
        return dimension
    return DIMENSION_ALIASES.get(dimension, str(value or "").strip())


def normalize_safety_intent_label(value: Any) -> str:
    raw = str(value or "").strip()
    return SAFETY_INTENT_ALIASES.get(raw, raw)


def allowed_parser_labels() -> dict[str, list[str]]:
    return {
        "analysis_scope": sorted(ALLOWED_ANALYSIS_SCOPES),
        "methodology_intent": sorted(ALLOWED_METHODOLOGY_INTENTS),
        "requested_dimensions": sorted(SUPPORTED_DIMENSIONS),
        "requested_metrics": sorted(ALLOWED_ANALYSIS_METRICS),
        "user_expectation": sorted(ALLOWED_USER_EXPECTATIONS),
        "safety_intent": sorted(ALLOWED_SAFETY_INTENTS),
    }
