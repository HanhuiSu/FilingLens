"""Central methodology intent classification.

The public entry point supports a future LLM proposal step, validates that
proposal, and then falls back to deterministic family rules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Literal, Mapping

from src.agent.entity_resolution import ResolvedCompany, companies_to_tickers
from src.agent.types import AgentDomainModel

MethodologyIntent = Literal[
    "overview",
    "risk",
    "cash_flow",
    "profitability",
    "revenue",
    "balance_sheet",
    "valuation",
    "comparison",
    "none",
]
UserExpectation = Literal["quick_answer", "deep_analysis", "recommendation_like", "diagnostic", "clarification"]
SafetyIntent = Literal["normal", "investment_advice_like", "prediction", "unsupported"]

VALID_METHODOLOGY_INTENTS: set[str] = {
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


class MethodologyIntentResult(AgentDomainModel):
    methodology_intent: MethodologyIntent = "none"
    confidence: float = 0.0
    source: str = "fallback_rules"
    reasons: list[str] = []


_RISK_TERMS = (
    "最大问题",
    "最大的问题",
    "最大风险",
    "主要风险",
    "有什么风险",
    "有哪些风险",
    "风险有什么",
    "风险有哪些",
    "风险是什么",
    "经营风险",
    "什么风险",
    "最担心",
    "最值得担心",
    "最需要担心",
    "隐患",
    "哪里有问题",
    "出什么问题",
    "可能出问题",
    "风险点",
    "压力",
    "警惕",
    "值得警惕",
    "challenge",
    "key risk",
    "key risks",
    "major risk",
    "uncertainty",
    "biggest risk",
    "main issue",
    "main risk",
)
_CASH_FLOW_TERMS = (
    "现金流",
    "自由现金流",
    "利润能不能变成现金",
    "利润含金量",
    "cash flow",
    "free cash flow",
    "cash conversion",
)
_PROFITABILITY_TERMS = (
    "盈利质量",
    "盈利能力",
    "利润率",
    "净利率",
    "毛利率",
    "营业利润率",
    "profitability",
    "margin",
    "net margin",
)
_REVENUE_TERMS = ("收入质量", "营收质量", "收入怎么样", "营收怎么样", "revenue quality", "sales quality")
_BALANCE_SHEET_TERMS = (
    "资产负债",
    "抗风险能力",
    "债务",
    "现金储备",
    "资本投入",
    "资本开支",
    "balance sheet",
    "debt",
    "capex",
    "capital expenditure",
    "capital intensity",
)
_VALUATION_TERMS = (
    "贵不贵",
    "估值",
    "便宜",
    "昂贵",
    "cheap",
    "expensive",
    "valuation",
    "valuation boundary",
    "stretched",
    "multiple",
    "fcf yield",
)
_AWS_PROFIT_TERMS = (
    "aws",
    "amazon web services",
)
_SEGMENT_PROFIT_TERMS = (
    "利润",
    "盈利",
    "operating income",
    "整体利润",
    "贡献",
    "重要",
    "profit",
)
_COMPARISON_TERMS = (
    "更推荐",
    "更看好",
    "哪个更好",
    "哪一个更好",
    "选哪个",
    "值得关注哪个",
    "哪个更值得关注",
    "vs",
    "versus",
    "compare",
    "better",
)
_COMPARISON_RISK_TERMS = (
    "风险更高",
    "riskier",
    "higher risk",
    "更多风险",
    "主要风险",
    "风险",
)
_COMPARISON_DIFFERENCE_TERMS = ("差异", "difference", "different", "区别", "不同")
_OVERVIEW_TERMS = (
    "分析下",
    "分析一下",
    "帮我看看",
    "看看",
    "怎么样",
    "概览",
    "公司概览",
    "基本面",
    "研究一下",
    "总结一下",
    "综合分析",
    "analyze",
    "analysis",
    "overview",
    "company overview",
    "fundamentals",
    "fundamental",
    "fundamental read",
    "take a look",
)
_INVESTMENT_TERMS = (
    "推荐",
    "更推荐",
    "最推荐",
    "更看好",
    "最看好",
    "更值得关注",
    "值得关注哪个",
    "该买",
    "能买吗",
    "可以买",
    "值得买",
    "买入",
    "卖出",
    "持有",
    "should i buy",
    "should i sell",
    "which should i buy",
    "recommend",
)
_PREDICTION_TERMS = (
    "预测明天股价",
    "明天股价",
    "明天会涨",
    "明天会跌",
    "tomorrow stock price",
    "stock price tomorrow",
    "will rise tomorrow",
    "will fall tomorrow",
)
_UNSUPPORTED_TERMS = ("天气", "菜谱", "做饭", "体育比分", "weather", "recipe", "sports score")
_DIRECT_FACT_TERMS = ("是多少", "多少", "what is", "what was", "how much", "how many")
_OUTLOOK_TERMS = ("会怎么样", "你觉得", "未来", "明年", "outlook", "forecast", "guidance")

_LEGACY_INTENT_MAP = {
    "overview": "single_company_overview",
    "risk": "risk_focused_analysis",
    "cash_flow": "cash_flow_quality_analysis",
    "profitability": "profitability_quality_analysis",
    "revenue": "revenue_quality_analysis",
    "balance_sheet": "balance_sheet_analysis",
    "valuation": "valuation_boundary_analysis",
    "comparison": "company_comparison",
    "none": "",
}


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)


def normalize_query_text(raw_query: Any) -> str:
    return re.sub(r"\s+", " ", str(raw_query or "").strip().lower())


def legacy_methodology_intent(intent: str) -> str:
    return _LEGACY_INTENT_MAP.get(str(intent or ""), "")


def canonical_methodology_intent(legacy_intent: str) -> MethodologyIntent:
    reverse = {legacy: canonical for canonical, legacy in _LEGACY_INTENT_MAP.items()}
    return reverse.get(str(legacy_intent or ""), "none")  # type: ignore[return-value]


def infer_safety_intent(normalized_query: str) -> SafetyIntent:
    q = normalize_query_text(normalized_query)
    if _contains_any(q, _UNSUPPORTED_TERMS):
        return "unsupported"
    if _contains_any(q, _PREDICTION_TERMS):
        return "prediction"
    if (
        _contains_any(q, ("股价", "股票价格", "stock price", "share price"))
        and _contains_any(q, ("预测", "明天", "后天", "下周", "tomorrow", "predict", "forecast", "会涨", "会跌"))
    ):
        return "prediction"
    if re.search(r"(明天|后天|下周).*(会涨|会跌|涨吗|跌吗)", q):
        return "prediction"
    if re.search(r"\bwill\b.*\b(rise|fall|go up|go down)\b.*\b(tomorrow|next week)\b", q):
        return "prediction"
    if _contains_any(q, _INVESTMENT_TERMS):
        return "investment_advice_like"
    return "normal"


def legacy_safety_intent(safety_intent: str) -> str:
    if safety_intent in {"prediction", "unsupported"}:
        return "unsupported_or_out_of_scope"
    if safety_intent == "investment_advice_like":
        return "investment_advice_like"
    return "normal"


def _is_direct_fact_metric_question(q: str) -> bool:
    if not _contains_any(q, _DIRECT_FACT_TERMS):
        return False
    return _contains_any(
        q,
        (
            "营收",
            "收入",
            "利润",
            "净利润",
            "eps",
            "revenue",
            "sales",
            "net income",
        ),
    )


def _fallback_rules(normalized_query: str, resolved_companies: list[ResolvedCompany]) -> MethodologyIntentResult:
    q = normalize_query_text(normalized_query)
    company_count = len(set(companies_to_tickers(resolved_companies)))
    safety = infer_safety_intent(q)
    if safety in {"prediction", "unsupported"}:
        return MethodologyIntentResult(methodology_intent="none", confidence=0.9, reasons=["safety_redirect"])
    if _is_direct_fact_metric_question(q):
        return MethodologyIntentResult(methodology_intent="none", confidence=0.86, reasons=["direct_fact_metric_question"])
    if company_count >= 2 or _contains_any(q, _COMPARISON_TERMS):
        reasons = ["comparison_family"]
        if _contains_any(q, _COMPARISON_RISK_TERMS):
            reasons.append("comparison_risk_family")
        if _contains_any(q, _COMPARISON_DIFFERENCE_TERMS):
            reasons.append("comparison_difference_family")
        return MethodologyIntentResult(methodology_intent="comparison", confidence=0.91, reasons=reasons)
    if company_count <= 0:
        return MethodologyIntentResult(methodology_intent="none", confidence=0.55, reasons=["no_resolved_company"])
    if _contains_any(q, _AWS_PROFIT_TERMS) and _contains_any(q, _SEGMENT_PROFIT_TERMS):
        return MethodologyIntentResult(methodology_intent="profitability", confidence=0.9, reasons=["aws_segment_profitability_family"])
    if _contains_any(q, _CASH_FLOW_TERMS):
        return MethodologyIntentResult(methodology_intent="cash_flow", confidence=0.89, reasons=["cash_flow_family"])
    if _contains_any(q, _VALUATION_TERMS):
        return MethodologyIntentResult(methodology_intent="valuation", confidence=0.89, reasons=["valuation_family"])
    if _contains_any(q, _BALANCE_SHEET_TERMS):
        return MethodologyIntentResult(methodology_intent="balance_sheet", confidence=0.88, reasons=["balance_sheet_family"])
    if _contains_any(q, _PROFITABILITY_TERMS):
        return MethodologyIntentResult(methodology_intent="profitability", confidence=0.87, reasons=["profitability_family"])
    if _contains_any(q, _REVENUE_TERMS):
        return MethodologyIntentResult(methodology_intent="revenue", confidence=0.87, reasons=["revenue_family"])
    if _contains_any(q, _RISK_TERMS):
        return MethodologyIntentResult(methodology_intent="risk", confidence=0.9, reasons=["risk_family"])
    if _contains_any(q, _OUTLOOK_TERMS):
        return MethodologyIntentResult(methodology_intent="none", confidence=0.68, reasons=["outlook_not_methodology"])
    if _contains_any(q, _OVERVIEW_TERMS):
        return MethodologyIntentResult(methodology_intent="overview", confidence=0.86, reasons=["overview_family"])
    return MethodologyIntentResult(methodology_intent="none", confidence=0.5, reasons=["no_methodology_family"])


def _coerce_llm_payload(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, Mapping) else None
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return _coerce_llm_payload(content)
    return None


def _llm_proposal(
    normalized_query: str,
    resolved_companies: list[ResolvedCompany],
    optional_llm_client: Any | None,
) -> MethodologyIntentResult | None:
    if optional_llm_client is None:
        return None
    prompt_payload = {
        "query": normalized_query,
        "companies": [item.model_dump(exclude_none=True) for item in resolved_companies],
        "allowed_labels": sorted(VALID_METHODOLOGY_INTENTS),
    }
    try:
        if hasattr(optional_llm_client, "invoke"):
            raw = optional_llm_client.invoke(prompt_payload)
        elif callable(optional_llm_client):
            raw = optional_llm_client(prompt_payload)
        else:
            return None
    except Exception:
        return None
    payload = _coerce_llm_payload(raw)
    if not payload:
        return None
    label = str(payload.get("methodology_intent") or payload.get("intent") or "").strip()
    confidence = float(payload.get("confidence", 0) or 0)
    if label not in VALID_METHODOLOGY_INTENTS or confidence < 0.55:
        return None
    reasons = payload.get("reasons", [])
    return MethodologyIntentResult(
        methodology_intent=label,  # type: ignore[arg-type]
        confidence=round(confidence, 3),
        source="llm_validated",
        reasons=[str(item) for item in reasons if str(item).strip()],
    )


def classify_methodology_intent(
    normalized_query: str,
    resolved_companies: list[ResolvedCompany] | None = None,
    optional_llm_client: Any | None = None,
) -> MethodologyIntentResult:
    resolved = list(resolved_companies or [])
    proposal = _llm_proposal(normalized_query, resolved, optional_llm_client)
    if proposal is not None:
        return proposal
    return _fallback_rules(normalized_query, resolved)


def infer_user_expectation(
    normalized_query: str,
    *,
    methodology_intent: str,
    safety_intent: str,
    needs_clarification: bool,
) -> UserExpectation:
    if needs_clarification:
        return "clarification"
    if safety_intent == "investment_advice_like":
        return "recommendation_like"
    if methodology_intent in {"risk", "cash_flow", "balance_sheet", "valuation"}:
        return "diagnostic"
    if methodology_intent == "overview":
        return "deep_analysis"
    return "quick_answer"
