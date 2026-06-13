"""Fundamental-analysis framework selection for methodology v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class AnalysisDimension:
    id: str
    name: str
    description: str
    required_numeric_metrics: list[str]
    optional_numeric_metrics: list[str]
    required_text_sections: list[str]
    optional_text_sections: list[str]
    evidence_purpose: str
    missing_behavior: str
    allowed_claims: list[str]
    forbidden_claims: list[str]


@dataclass(frozen=True)
class SelectedAnalysisFramework:
    framework_id: str
    name: str
    active_dimension_ids: list[str]
    dimensions: list[AnalysisDimension]
    selection_reasons: list[dict[str, str]]
    inactive_dimension_ids: list[str]

    def model_dump(self, **_: Any) -> dict[str, Any]:
        """Mirror Pydantic's model_dump for easier state serialization."""
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        names = {dimension.id: dimension.name for dimension in self.dimensions}
        return {
            "id": self.framework_id,
            "name": self.name,
            "active_dimension_ids": list(self.active_dimension_ids),
            "active_dimension_names": [
                names[dimension_id]
                for dimension_id in self.active_dimension_ids
                if dimension_id in names
            ],
            "selection_reasons": list(self.selection_reasons),
        }


FRAMEWORK_ID = "fundamental_quality_analysis"
FRAMEWORK_NAME = "Fundamental Quality Analysis"

BUSINESS_MODEL = "business_model"
REVENUE_QUALITY = "revenue_quality"
PROFITABILITY_QUALITY = "profitability_quality"
CASH_FLOW_QUALITY = "cash_flow_quality"
BALANCE_SHEET_AND_CAPITAL_INTENSITY = "balance_sheet_and_capital_intensity"
MOAT_AND_COMPETITIVE_RISK = "moat_and_competitive_risk"
VALUATION_AND_RISK_BOUNDARY = "valuation_and_risk_boundary"


def get_fundamental_quality_analysis() -> list[AnalysisDimension]:
    """Return the fixed methodology-v1 fundamental-analysis dimensions."""
    return [
        AnalysisDimension(
            id=BUSINESS_MODEL,
            name="Business Model",
            description="Describe what the company sells, who it serves, and how the business makes money.",
            required_numeric_metrics=[],
            optional_numeric_metrics=["revenue"],
            required_text_sections=["ITEM_1"],
            optional_text_sections=["BUSINESS", "ITEM_7"],
            evidence_purpose="Ground business-model claims in the Business section before adding financial interpretation.",
            missing_behavior="If ITEM_1 or equivalent business text is missing, only make a limited company-description claim.",
            allowed_claims=[
                "business model description",
                "segment or product mix description",
                "customer or market exposure description",
            ],
            forbidden_claims=[
                "durable moat conclusion without text evidence",
                "growth sustainability conclusion from business description alone",
                "investment attractiveness conclusion",
            ],
        ),
        AnalysisDimension(
            id=REVENUE_QUALITY,
            name="Revenue Quality",
            description="Assess revenue scale, trend context, and whether revenue evidence is broad enough for quality claims.",
            required_numeric_metrics=["revenue"],
            optional_numeric_metrics=["revenue_growth", "segment_revenue", "deferred_revenue"],
            required_text_sections=[],
            optional_text_sections=["ITEM_7", "ITEM_1"],
            evidence_purpose="Use revenue facts and MD&A context to avoid overclaiming sustainability from a single period.",
            missing_behavior="If only one revenue point is available, report scale only and do not claim sustainable growth.",
            allowed_claims=[
                "revenue scale comparison",
                "period revenue direction when multiple periods exist",
                "limited revenue-quality observation with explicit caveat",
            ],
            forbidden_claims=[
                "growth is sustainable from a single revenue period",
                "demand quality conclusion without trend or text support",
                "forward-looking revenue prediction",
            ],
        ),
        AnalysisDimension(
            id=PROFITABILITY_QUALITY,
            name="Profitability Quality",
            description="Assess profit conversion using revenue, net income, and derived net margin when available.",
            required_numeric_metrics=["revenue", "net_income"],
            optional_numeric_metrics=["net_margin", "operating_margin", "gross_margin"],
            required_text_sections=[],
            optional_text_sections=["ITEM_7"],
            evidence_purpose="Ground profitability-quality claims in numeric profit and margin evidence.",
            missing_behavior="If revenue and net income are available, allow a limited profitability judgment using net margin.",
            allowed_claims=[
                "absolute net income comparison",
                "net margin comparison",
                "limited profitability-quality judgment",
            ],
            forbidden_claims=[
                "operating leverage conclusion without margin or MD&A evidence",
                "cash earnings quality conclusion without cash-flow evidence",
                "investment recommendation",
            ],
        ),
        AnalysisDimension(
            id=CASH_FLOW_QUALITY,
            name="Cash Flow Quality",
            description="Assess whether reported earnings are supported by operating cash flow and free cash flow.",
            required_numeric_metrics=["operating_cash_flow", "free_cash_flow"],
            optional_numeric_metrics=["capital_expenditure", "cash_flow_margin", "ocf_to_net_income"],
            required_text_sections=[],
            optional_text_sections=["ITEM_7", "ITEM_8"],
            evidence_purpose="Use CFO/FCF evidence before making cash-flow-quality conclusions.",
            missing_behavior="If CFO or FCF is missing, do not make a cash-flow-quality conclusion.",
            allowed_claims=[
                "cash-flow evidence is unavailable",
                "operating cash flow comparison when CFO exists",
                "free cash flow comparison when FCF exists",
            ],
            forbidden_claims=[
                "cash flow quality is strong without CFO/FCF",
                "earnings quality conclusion from net income alone",
                "liquidity comfort conclusion without cash-flow evidence",
            ],
        ),
        AnalysisDimension(
            id=BALANCE_SHEET_AND_CAPITAL_INTENSITY,
            name="Balance Sheet And Capital Intensity",
            description="Assess leverage, liquidity, and reinvestment intensity when balance-sheet and capex data exist.",
            required_numeric_metrics=["total_debt", "cash_and_equivalents", "capital_expenditure"],
            optional_numeric_metrics=["total_assets", "shareholders_equity", "current_assets", "current_liabilities"],
            required_text_sections=[],
            optional_text_sections=["ITEM_7", "ITEM_8"],
            evidence_purpose="Support financial-health and capital-intensity claims with balance-sheet and capex evidence.",
            missing_behavior="If debt, cash, or capex evidence is missing, present this dimension as unavailable or partial.",
            allowed_claims=[
                "debt and cash evidence summary",
                "capital intensity observation when capex exists",
                "limited balance-sheet-health caveat",
            ],
            forbidden_claims=[
                "balance sheet is healthy without debt and liquidity evidence",
                "capital-light conclusion without capex evidence",
                "solvency conclusion without balance-sheet evidence",
            ],
        ),
        AnalysisDimension(
            id=MOAT_AND_COMPETITIVE_RISK,
            name="Moat And Competitive Risk",
            description="Assess competitive position and business risk only from validated filing text evidence.",
            required_numeric_metrics=[],
            optional_numeric_metrics=["revenue", "net_income", "net_margin"],
            required_text_sections=["ITEM_1A"],
            optional_text_sections=["ITEM_1", "ITEM_7"],
            evidence_purpose="Use validated risk and business text before discussing competitive risk or moat limits.",
            missing_behavior="If validated text evidence is missing, do not make specific competitive-risk claims.",
            allowed_claims=[
                "validated disclosed risk theme",
                "competitive pressure described in filings",
                "limited moat-related caveat tied to text evidence",
            ],
            forbidden_claims=[
                "specific risk theme without text evidence",
                "durable moat conclusion without filing support",
                "risk ranking without balanced company text evidence",
            ],
        ),
        AnalysisDimension(
            id=VALUATION_AND_RISK_BOUNDARY,
            name="Valuation And Risk Boundary",
            description="State valuation and investment-advice boundaries when valuation evidence is absent or incomplete.",
            required_numeric_metrics=[],
            optional_numeric_metrics=["market_cap", "enterprise_value", "pe_ratio", "price_to_sales"],
            required_text_sections=[],
            optional_text_sections=["ITEM_1A", "ITEM_7"],
            evidence_purpose="Prevent valuation or recommendation claims when valuation evidence is not in the packet.",
            missing_behavior="If valuation data is missing, do not claim the stock is cheap, expensive, or worth buying.",
            allowed_claims=[
                "valuation evidence is unavailable",
                "non-advisory comparison boundary",
                "risk and valuation limitation statement",
            ],
            forbidden_claims=[
                "cheap or expensive conclusion without valuation data",
                "worth buying conclusion",
                "buy sell hold recommendation",
            ],
        ),
    ]


def _dump_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _query_text(query_plan: Any) -> str:
    payload = _dump_mapping(query_plan)
    query = str(payload.get("user_query") or payload.get("query") or "").strip()
    if query:
        return query
    analysis_plan = _dump_mapping(payload.get("analysis_plan"))
    return str(analysis_plan.get("user_intent") or "").strip()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _activate(
    active: list[str],
    reasons: list[dict[str, str]],
    dimension_ids: tuple[str, ...],
    *,
    rule: str,
    reason: str,
) -> None:
    for dimension_id in dimension_ids:
        if dimension_id not in active:
            active.append(dimension_id)
    reasons.append({"rule": rule, "reason": reason})


def _requested_dimensions(payload: Mapping[str, Any], analysis_plan: Mapping[str, Any], dimension_ids: list[str]) -> list[str]:
    requested: list[str] = []
    query_understanding = _dump_mapping(payload.get("query_understanding"))
    for source in (
        analysis_plan.get("requested_dimensions", []),
        query_understanding.get("requested_dimensions", []),
    ):
        for item in source or []:
            dimension_id = str(item or "").strip()
            if dimension_id in dimension_ids and dimension_id not in requested:
                requested.append(dimension_id)
    return requested


def select_analysis_framework(query_plan: Any) -> SelectedAnalysisFramework:
    """Select the methodology-v1 analysis framework and active dimensions."""
    payload = _dump_mapping(query_plan)
    analysis_plan = _dump_mapping(payload.get("analysis_plan"))
    analysis_scope = str(payload.get("analysis_scope") or analysis_plan.get("analysis_scope") or "").strip()
    answer_mode = str(payload.get("answer_mode") or analysis_plan.get("answer_mode") or "").strip()
    methodology_intent = str(
        payload.get("methodology_intent") or analysis_plan.get("methodology_intent") or ""
    ).strip()
    query = _query_text(query_plan)
    normalized = query.lower()
    dimensions = get_fundamental_quality_analysis()
    dimension_ids = [dimension.id for dimension in dimensions]
    active: list[str] = []
    reasons: list[dict[str, str]] = []
    requested_dimensions = _requested_dimensions(payload, analysis_plan, dimension_ids)

    if analysis_scope == "comparison" and requested_dimensions:
        _activate(
            active,
            reasons,
            tuple(requested_dimensions),
            rule="explicit_requested_dimensions",
            reason="Comparison query explicitly requests analysis dimensions, so those dimensions drive the framework selection.",
        )
        return SelectedAnalysisFramework(
            framework_id=FRAMEWORK_ID,
            name=FRAMEWORK_NAME,
            active_dimension_ids=active,
            dimensions=dimensions,
            selection_reasons=reasons,
            inactive_dimension_ids=[dimension_id for dimension_id in dimension_ids if dimension_id not in active],
        )

    if analysis_scope == "single_company" and len(requested_dimensions) > 1:
        _activate(
            active,
            reasons,
            tuple(requested_dimensions),
            rule="explicit_requested_dimensions",
            reason="Single-company query explicitly requests multiple analysis dimensions, so those dimensions drive the framework selection.",
        )
        return SelectedAnalysisFramework(
            framework_id=FRAMEWORK_ID,
            name=FRAMEWORK_NAME,
            active_dimension_ids=active,
            dimensions=dimensions,
            selection_reasons=reasons,
            inactive_dimension_ids=[dimension_id for dimension_id in dimension_ids if dimension_id not in active],
        )

    intent_dimension_map: dict[str, tuple[str, ...]] = {
        "risk_focused_analysis": (
            BUSINESS_MODEL,
            MOAT_AND_COMPETITIVE_RISK,
        ),
        "single_company_overview": (
            BUSINESS_MODEL,
            REVENUE_QUALITY,
            PROFITABILITY_QUALITY,
            CASH_FLOW_QUALITY,
            BALANCE_SHEET_AND_CAPITAL_INTENSITY,
            MOAT_AND_COMPETITIVE_RISK,
            VALUATION_AND_RISK_BOUNDARY,
        ),
        "company_comparison": (
            REVENUE_QUALITY,
            PROFITABILITY_QUALITY,
            MOAT_AND_COMPETITIVE_RISK,
            VALUATION_AND_RISK_BOUNDARY,
        ),
        "investment_advice_like": (
            REVENUE_QUALITY,
            PROFITABILITY_QUALITY,
            MOAT_AND_COMPETITIVE_RISK,
            VALUATION_AND_RISK_BOUNDARY,
        ),
        "revenue_quality_analysis": (REVENUE_QUALITY,),
        "profitability_quality_analysis": (PROFITABILITY_QUALITY,),
        "cash_flow_quality_analysis": (CASH_FLOW_QUALITY,),
        "balance_sheet_analysis": (BALANCE_SHEET_AND_CAPITAL_INTENSITY,),
        "valuation_boundary_analysis": (VALUATION_AND_RISK_BOUNDARY,),
    }
    if methodology_intent in intent_dimension_map:
        _activate(
            active,
            reasons,
            intent_dimension_map[methodology_intent],
            rule=f"methodology_intent:{methodology_intent}",
            reason="Intent-family routing selects the dimensions needed for this methodology analysis.",
        )
        return SelectedAnalysisFramework(
            framework_id=FRAMEWORK_ID,
            name=FRAMEWORK_NAME,
            active_dimension_ids=active,
            dimensions=dimensions,
            selection_reasons=reasons,
            inactive_dimension_ids=[dimension_id for dimension_id in dimension_ids if dimension_id not in active],
        )

    if answer_mode == "risk_focused_analysis":
        _activate(
            active,
            reasons,
            (
                BUSINESS_MODEL,
                MOAT_AND_COMPETITIVE_RISK,
            ),
            rule="risk_focused_analysis",
            reason="Risk-focused single-company questions use business context and risk text as active sufficiency dimensions.",
        )
        return SelectedAnalysisFramework(
            framework_id=FRAMEWORK_ID,
            name=FRAMEWORK_NAME,
            active_dimension_ids=active,
            dimensions=dimensions,
            selection_reasons=reasons,
            inactive_dimension_ids=[dimension_id for dimension_id in dimension_ids if dimension_id not in active],
        )

    if analysis_scope == "single_company":
        _activate(
            active,
            reasons,
            (
                BUSINESS_MODEL,
                REVENUE_QUALITY,
                PROFITABILITY_QUALITY,
                CASH_FLOW_QUALITY,
                BALANCE_SHEET_AND_CAPITAL_INTENSITY,
                MOAT_AND_COMPETITIVE_RISK,
                VALUATION_AND_RISK_BOUNDARY,
            ),
            rule="single_company_methodology_analysis",
            reason="Open-ended single-company analysis uses the core fundamental-quality dimensions, including cash-flow and balance-sheet checks.",
        )

    if _contains_any(
        normalized,
        (
            "更推荐",
            "推荐哪个",
            "更看好",
            "最看好",
            "哪个更好",
            "长期",
            "值得关注",
            "应该降级",
            "是否应该降级",
            "recommend",
            "better",
            "which is best",
            "should i buy",
            "which stock should i buy",
            "downgrade",
            "downgraded",
            "long term",
            "long-term",
            "worth watching",
        ),
    ):
        _activate(
            active,
            reasons,
            (
                REVENUE_QUALITY,
                PROFITABILITY_QUALITY,
                MOAT_AND_COMPETITIVE_RISK,
                VALUATION_AND_RISK_BOUNDARY,
            ),
            rule="advice_like_or_preference",
            reason="Preference-style queries need revenue, profitability, competitive risk, and valuation-boundary dimensions.",
        )

    if _contains_any(normalized, ("长期", "long term", "long-term")):
        _activate(
            active,
            reasons,
            (CASH_FLOW_QUALITY, BALANCE_SHEET_AND_CAPITAL_INTENSITY),
            rule="long_term_extra_health_checks",
            reason="Long-term questions need cash-flow and balance-sheet health checks.",
        )

    if _contains_any(
        normalized,
        (
            "风险",
            "隐患",
            "压力",
            "risk",
            "risks",
            "pressure",
        ),
    ):
        _activate(
            active,
            reasons,
            (
                BUSINESS_MODEL,
                MOAT_AND_COMPETITIVE_RISK,
                CASH_FLOW_QUALITY,
                BALANCE_SHEET_AND_CAPITAL_INTENSITY,
            ),
            rule="risk_or_pressure",
            reason="Risk and pressure questions need business context, disclosed risks, cash-flow quality, and balance-sheet health.",
        )

    if _contains_any(
        normalized,
        (
            "最大问题",
            "problem",
            "concern",
            "headwind",
        ),
    ):
        _activate(
            active,
            reasons,
            (
                BUSINESS_MODEL,
                MOAT_AND_COMPETITIVE_RISK,
            ),
            rule="problem_or_concern",
            reason="Problem and concern questions need business context and validated disclosed-risk evidence.",
        )

    if _contains_any(
        normalized,
        (
            "最近财报",
            "表现",
            "趋势",
            "营收",
            "收入",
            "利润",
            "盈利",
            "earnings",
            "performance",
            "trend",
            "revenue",
            "sales",
            "profit",
            "profitability",
            "income",
        ),
    ):
        _activate(
            active,
            reasons,
            (REVENUE_QUALITY, PROFITABILITY_QUALITY),
            rule="recent_performance_or_metrics",
            reason="Performance and filing-result questions need revenue and profitability dimensions.",
        )

    if _contains_any(normalized, ("现金流", "财务健康", "cash flow", "financial health")):
        _activate(
            active,
            reasons,
            (CASH_FLOW_QUALITY,),
            rule="cash_flow_quality_or_health",
            reason="Cash-flow, quality, or health wording requires the cash-flow-quality guardrail.",
        )

    if _contains_any(normalized, ("估值", "便宜", "昂贵", "valuation", "cheap", "expensive")):
        _activate(
            active,
            reasons,
            (VALUATION_AND_RISK_BOUNDARY,),
            rule="valuation_or_price_boundary",
            reason="Valuation wording requires valuation evidence and recommendation-boundary guardrails.",
        )

    if _contains_any(
        normalized,
        (
            "这家公司怎么样",
            "公司怎么样",
            "分析一下",
            "综合分析",
            "analyze",
            "analysis",
            "how is this company",
            "what do you think of",
        ),
    ):
        _activate(
            active,
            reasons,
            (
                BUSINESS_MODEL,
                REVENUE_QUALITY,
                PROFITABILITY_QUALITY,
                CASH_FLOW_QUALITY,
                BALANCE_SHEET_AND_CAPITAL_INTENSITY,
                MOAT_AND_COMPETITIVE_RISK,
                VALUATION_AND_RISK_BOUNDARY,
            ),
            rule="generic_company_analysis",
            reason="General company-analysis queries need business, revenue, profitability, cash-flow, balance-sheet, risk, and valuation-boundary dimensions.",
        )

    return SelectedAnalysisFramework(
        framework_id=FRAMEWORK_ID,
        name=FRAMEWORK_NAME,
        active_dimension_ids=active,
        dimensions=dimensions,
        selection_reasons=reasons,
        inactive_dimension_ids=[dimension_id for dimension_id in dimension_ids if dimension_id not in active],
    )


def serialize_selected_analysis_framework(selected: Any) -> dict[str, Any]:
    if isinstance(selected, SelectedAnalysisFramework):
        return selected.model_dump()
    return _dump_mapping(selected)


def summarize_selected_analysis_framework(selected: Any) -> dict[str, Any]:
    if isinstance(selected, SelectedAnalysisFramework):
        return selected.summary()
    payload = _dump_mapping(selected)
    dimensions = payload.get("dimensions", [])
    names: dict[str, str] = {}
    if isinstance(dimensions, list):
        for item in dimensions:
            dimension = _dump_mapping(item)
            dimension_id = str(dimension.get("id") or "").strip()
            if dimension_id:
                names[dimension_id] = str(dimension.get("name") or dimension_id)
    active_ids = [str(item) for item in payload.get("active_dimension_ids", []) if str(item)]
    return {
        "id": str(payload.get("framework_id") or FRAMEWORK_ID),
        "name": str(payload.get("name") or FRAMEWORK_NAME),
        "active_dimension_ids": active_ids,
        "active_dimension_names": [names.get(dimension_id, dimension_id) for dimension_id in active_ids],
        "selection_reasons": list(payload.get("selection_reasons", []) or []),
    }


def analysis_framework_trace_fields(selected: Any) -> dict[str, Any]:
    summary = summarize_selected_analysis_framework(selected)
    if not summary.get("id"):
        return {}
    return {
        "analysis_framework_id": summary["id"],
        "active_analysis_dimensions": list(summary.get("active_dimension_ids", []) or []),
    }
