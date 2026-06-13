# ruff: noqa: F401,F403,F405
"""Query planning, period normalization, routing, and retrieval policy."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Mapping

from config import settings
from src.agent.analysis_framework import (
    analysis_framework_trace_fields,
    select_analysis_framework,
    serialize_selected_analysis_framework,
)
from src.agent.canonical_intent import build_canonical_intent
from src.agent.output_language import detect_output_language
from src.agent.constants import *
from src.agent.evidence_planner import build_evidence_plan
from src.agent.entity_resolution import ResolvedCompany, resolve_companies
from src.agent.intent_policy import resolve_evidence_policy
from src.agent.methodology_intent import (
    classify_methodology_intent,
    infer_safety_intent,
    legacy_methodology_intent,
    legacy_safety_intent,
)
from src.agent.query_understanding import build_query_understanding, query_understanding_summary
from src.agent.safety import apply_safety_policy
from src.agent.semantic_query_parser import build_semantic_query_proposal, normalize_semantic_parser_mode
from src.agent.state import AgentState
from src.agent.types import AnalysisPlan, QueryPlan

logger = logging.getLogger(__name__)

_TICKER_ALIASES: dict[str, str] = {
    "apple": "AAPL", "苹果": "AAPL",
    "microsoft": "MSFT", "微软": "MSFT",
    "nvidia": "NVDA", "英伟达": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "谷歌": "GOOGL",
    "amazon": "AMZN", "亚马逊": "AMZN",
    "tesla": "TSLA", "特斯拉": "TSLA",
    "jpmorgan": "JPM", "jp morgan": "JPM", "摩根大通": "JPM",
    "johnson": "JNJ", "j&j": "JNJ", "强生": "JNJ",
}

_SECTION_ALIASES: dict[str, str] = {
    "business": "ITEM_1",
    "item 1": "ITEM_1",
    "item_1": "ITEM_1",
    "risk": "ITEM_1A",
    "risk factors": "ITEM_1A",
    "item 1a": "ITEM_1A",
    "item_1a": "ITEM_1A",
    "md&a": "ITEM_7",
    "mda": "ITEM_7",
    "management discussion": "ITEM_7",
    "management discussion and analysis": "ITEM_7",
    "item 7": "ITEM_7",
    "item_7": "ITEM_7",
    "market risk": "ITEM_7A",
    "item 7a": "ITEM_7A",
    "item_7a": "ITEM_7A",
    "financial statements": "ITEM_8",
    "item 8": "ITEM_8",
    "item_8": "ITEM_8",
    "legal": "ITEM_3",
    "legal proceedings": "ITEM_3",
    "item 3": "ITEM_3",
    "item_3": "ITEM_3",
}

def _default_period_query() -> dict[str, Any]:
    return {
        "period_type": None,
        "year": None,
        "quarter": None,
        "trailing_n": None,
        "year_basis": "fiscal",
        "comparison_basis": "same_period",
        "is_explicit": False,
        "needs_clarification": False,
        "clarification_reason": None,
    }

def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(k in lowered for k in keywords)


def _format_constraints_from_query(user_query: str) -> dict[str, Any]:
    query = str(user_query or "").lower()
    one_sentence = bool(
        re.search(
            r"(一句话|一段话|用一句|只用一句|one sentence|single sentence|in one sentence)",
            query,
        )
    )
    if one_sentence:
        return {"one_sentence": True, "max_sentences": 1}
    return {"one_sentence": False}


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in out:
            out.append(clean)
    return out

def _string_list(value: Any, *, max_items: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= max_items:
            break
    return out

def _normalize_ticker_value(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    result = resolve_companies(raw)
    if result.resolved_companies:
        return result.resolved_companies[0].ticker
    return None

def _normalize_company_values(values: Any) -> list[str]:
    raw_values = values if isinstance(values, list) else []
    result = resolve_companies(" ".join(str(item or "") for item in raw_values), parsed_companies=raw_values)
    return [item.ticker for item in result.resolved_companies]

def _query_understanding_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}

def _query_understanding_tickers(query_understanding: Mapping[str, Any]) -> list[str]:
    tickers: list[str] = []
    for item in list(query_understanding.get("companies", []) or []):
        if isinstance(item, Mapping):
            ticker = str(item.get("ticker", "")).upper().strip()
        else:
            ticker = str(getattr(item, "ticker", "")).upper().strip()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers

def _query_understanding_scope(query_understanding: Mapping[str, Any]) -> str:
    scope = str(query_understanding.get("analysis_scope") or "").strip()
    return scope if scope in {"single_company", "comparison"} else ""

def _query_understanding_safety(query_understanding: Mapping[str, Any]) -> str:
    safety = str(query_understanding.get("legacy_safety_intent") or "").strip()
    return safety if safety in SAFETY_INTENTS else ""

def _query_understanding_methodology(
    query_understanding: Mapping[str, Any],
    *,
    companies: list[str],
    comparison_target: str | None,
    safety_intent: str,
) -> str:
    intent = str(query_understanding.get("legacy_methodology_intent") or "").strip()
    canonical_safety = str(query_understanding.get("safety_intent") or "").strip()
    if not intent and canonical_safety == "prediction":
        return "unsupported_prediction"
    if safety_intent == "investment_advice_like":
        count = _company_count(companies, comparison_target)
        if intent == "company_comparison" or count >= 2:
            return "investment_advice_like"
        if count == 1:
            return "valuation_boundary_analysis"
    return intent

def _query_understanding_time_scope(query_understanding: Mapping[str, Any]) -> dict[str, Any]:
    time_scope = query_understanding.get("time_scope")
    return dict(time_scope) if isinstance(time_scope, Mapping) else {}

def _query_understanding_requested_dimensions(query_understanding: Mapping[str, Any]) -> list[str]:
    dimensions: list[str] = []
    for item in query_understanding.get("requested_dimensions", []) or []:
        dimension = str(item or "").strip()
        if dimension and dimension not in dimensions:
            dimensions.append(dimension)
    return dimensions


def _query_understanding_requested_metrics(query_understanding: Mapping[str, Any]) -> list[str]:
    metrics: list[str] = []
    for item in query_understanding.get("requested_metrics", []) or []:
        metric = str(item or "").strip()
        if metric and metric not in metrics:
            metrics.append(metric)
    return metrics


def _semantic_parser_trace_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return dict(result) if isinstance(result, Mapping) else {}


def _semantic_parser_disagreement(
    semantic_parser_trace: Mapping[str, Any],
    query_understanding: Mapping[str, Any],
    *,
    injected: bool,
) -> dict[str, Any]:
    proposal = semantic_parser_trace.get("proposal")
    if not isinstance(proposal, Mapping) or semantic_parser_trace.get("ok") is not True:
        return {"parser_ok": False, "injected": False}
    proposed_intent = str(proposal.get("methodology_intent") or "")
    proposed_safety = str(proposal.get("safety_intent") or "")
    proposed_dimensions = [str(item) for item in proposal.get("requested_dimensions", []) or [] if str(item)]
    proposed_metrics = [str(item) for item in proposal.get("requested_metrics", []) or [] if str(item)]
    rule_intent = str(query_understanding.get("rule_methodology_intent") or query_understanding.get("methodology_intent") or "")
    final_intent = str(query_understanding.get("methodology_intent") or "")
    final_safety = str(query_understanding.get("safety_intent") or "")
    final_dimensions = [str(item) for item in query_understanding.get("requested_dimensions", []) or [] if str(item)]
    final_metrics = [str(item) for item in query_understanding.get("requested_metrics", []) or [] if str(item)]
    return {
        "parser_ok": True,
        "injected": bool(injected),
        "proposed_methodology_intent": proposed_intent,
        "rule_methodology_intent": rule_intent,
        "final_methodology_intent": final_intent,
        "methodology_intent_disagreement": bool(proposed_intent and proposed_intent != rule_intent),
        "proposed_safety_intent": proposed_safety,
        "final_safety_intent": final_safety,
        "safety_intent_disagreement": bool(proposed_safety and proposed_safety != final_safety),
        "proposed_dimensions": proposed_dimensions,
        "final_dimensions": final_dimensions,
        "dimensions_only_in_proposal": [item for item in proposed_dimensions if item not in final_dimensions],
        "dimensions_only_in_final": [item for item in final_dimensions if item not in proposed_dimensions],
        "proposed_metrics": proposed_metrics,
        "final_metrics": final_metrics,
        "metrics_only_in_proposal": [item for item in proposed_metrics if item not in final_metrics],
        "metrics_only_in_final": [item for item in final_metrics if item not in proposed_metrics],
    }


def _is_risk_collapsed_composite_request(
    *,
    analysis_scope: str,
    methodology_intent: str,
    requested_dimensions: list[str],
) -> bool:
    return (
        analysis_scope == "single_company"
        and methodology_intent == "risk_focused_analysis"
        and len(requested_dimensions) > 1
    )


def _normalize_metric_value(value: Any) -> str | None:
    metric = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    metric_aliases = {
        "total_revenue": "revenue",
        "sales": "revenue",
        "net_sales": "revenue",
        "net_income_loss": "net_income",
        "earnings": "net_income",
        "diluted_eps": "eps",
        "earnings_per_share": "eps",
        "operating_income_margin": "operating_margin",
        "operating_cashflow": "operating_cash_flow",
        "cash_from_operations": "operating_cash_flow",
        "cfo": "operating_cash_flow",
        "fcf": "free_cash_flow",
        "capex": "capital_expenditure",
        "capital_expenditures": "capital_expenditure",
        "pe": "pe_ratio",
        "p_e": "pe_ratio",
        "price_to_earnings": "pe_ratio",
        "ps": "ps_ratio",
        "p_s": "ps_ratio",
        "price_to_sales": "ps_ratio",
        "adj_close": "adjusted_close",
    }
    metric = metric_aliases.get(metric, metric)
    if metric in ALLOWED_ANALYSIS_METRICS:
        return metric
    return None

def _normalize_section_value(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.upper().replace(" ", "_").replace("-", "_")
    normalized = normalized.replace("ITEM__", "ITEM_")
    if normalized in KNOWN_SEC_SECTIONS:
        return normalized
    lowered = raw.lower().strip()
    if lowered in _SECTION_ALIASES:
        return _SECTION_ALIASES[lowered]
    m = re.search(r"item\s*([0-9]+[a-z]?)", lowered)
    if m:
        candidate = f"ITEM_{m.group(1).upper()}"
        if candidate in KNOWN_SEC_SECTIONS:
            return candidate
    return None

def _has_market_reaction_terms(user_query: str) -> bool:
    q = str(user_query or "").lower()
    if any(k in q for k in MARKET_REACTION_TERMS):
        return True
    zh_stock_terms = ("股价", "市场")
    zh_reaction_terms = ("反应", "涨", "跌", "变化", "收益率", "回报", "回撤")
    if any(t in q for t in zh_stock_terms) and any(t in q for t in zh_reaction_terms):
        return True
    en_stock_terms = ("stock", "price", "market")
    en_reaction_terms = ("reaction", "move", "return", "change", "up", "down")
    return any(t in q for t in en_stock_terms) and any(t in q for t in en_reaction_terms)

def _has_event_anchor_terms(user_query: str) -> bool:
    q = str(user_query or "").lower()
    if any(k in q for k in EVENT_ANCHOR_TERMS):
        return True

    # Chinese natural expressions around filing timing.
    if re.search(r"(财报|季报|年报).*(发布|公布|披露|发布时|披露时)", q):
        return True
    if re.search(r"(发布|公布|披露).*(财报|季报|年报)", q):
        return True
    if re.search(r"(每次|每个季度|每季度).*(财报|季报|年报)", q):
        return True

    # “过去 N 次财报后 / last N earnings” style anchors.
    if re.search(r"过去\s*\d+\s*(次|个季度|季).*财报后", q):
        return True
    if re.search(r"last\s*\d+\s*(earnings|filings)", q):
        return True
    return False

def _normalize_task_type(task_type: str) -> str:
    if task_type in {"fact_qa", "trend_analysis", "company_comparison", "report_summary"}:
        return task_type
    return "fact_qa"

def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

def _is_meta_query(user_query: str) -> bool:
    q = str(user_query or "").lower().strip()
    compact = re.sub(r"[\s?？!！。,.，]+", "", q)
    meta_terms = (
        "你是谁",
        "你是什么",
        "你能做什么",
        "你可以做什么",
        "你的能力",
        "介绍一下你自己",
        "who are you",
        "what are you",
        "what can you do",
        "what do you do",
        "your capabilities",
    )
    return any(term.replace(" ", "") in compact for term in meta_terms if _has_chinese(term)) or any(
        term in q for term in meta_terms if not _has_chinese(term)
    )

def _is_investment_advice_like_query(user_query: str) -> bool:
    q = str(user_query or "").lower()
    return any(
        term in q
        for term in (
            "最看好哪个",
            "更看好哪个",
            "最看好哪一个",
            "更看好哪一个",
            "更值得关注哪个",
            "更值得关注哪一个",
            "值得重点关注哪个",
            "更值得跟踪哪个",
            "推荐哪个",
            "推荐哪一个",
            "买哪个",
            "该买",
            "能买吗",
            "可以买",
            "值得买",
            "买入",
            "卖出",
            "持有",
            "投资哪个",
            "recommend",
            "should i buy",
            "should i sell",
            "which should i buy",
            "which stock should i buy",
            "best investment",
            "buy rating",
            "sell rating",
        )
    )

def _is_cautious_outlook_query(user_query: str) -> bool:
    q = str(user_query or "").lower()
    return any(
        term in q
        for term in (
            "会怎么样",
            "怎么看",
            "你觉得",
            "前景",
            "展望",
            "未来",
            "接下来",
            "outlook",
            "prospect",
            "forecast",
            "future",
            "what do you think",
            "how will",
            "what will",
        )
    )

def _is_underspecified_analysis_query(user_query: str, companies: list[str]) -> bool:
    q = str(user_query or "").lower().strip()
    if companies:
        return False
    if re.fullmatch(r"(帮我)?分析一下[。！？\s]*", q):
        return True
    if re.fullmatch(r"(分析|analyse|analyze|analysis)[\s。！？?.!]*", q):
        return True
    return False

def _is_single_company_open_analysis_query(
    user_query: str,
    companies: list[str],
    comparison_target: str | None = None,
) -> bool:
    q = str(user_query or "").lower().strip()
    company_count = len(set([c for c in companies if c] + ([comparison_target] if comparison_target else [])))
    if company_count != 1:
        return False
    if _contains_any(q, COMPARISON_TERMS):
        return False
    if _is_cautious_outlook_query(q):
        return False
    if re.search(r"(是多少|多少|what\s+is|how\s+much)", q) and _contains_any(
        q,
        ("营收", "收入", "利润", "revenue", "sales", "net income", "eps"),
    ):
        return False
    return _contains_any(
        q,
        (
            "分析下",
            "分析一下",
            "分析",
            "帮我看看",
            "看看",
            "怎么样",
            "这家公司",
            "基本面",
            "最大的问题",
            "最大问题",
            "主要问题",
            "盈利质量",
            "收入质量",
            "风险",
            "analyze",
            "analysis",
            "take a look",
            "look at",
            "fundamental",
            "fundamentals",
            "how is",
            "main issue",
            "biggest problem",
        ),
    )


RISK_FOCUSED_TERMS = (
    "最大问题",
    "最大的问题",
    "最大风险",
    "主要风险",
    "最担心",
    "值得担心",
    "担心",
    "隐患",
    "哪里有问题",
    "出什么问题",
    "可能出问题",
    "风险点",
    "经营风险",
    "最大压力",
    "最大的压力",
    "主要压力",
    "压力是什么",
    "警惕",
    "值得警惕",
    "challenge",
    "challenges",
    "biggest risk",
    "key risk",
    "key risks",
    "main issue",
    "biggest problem",
)

METHODOLOGY_INTENTS = {
    "single_company_overview",
    "risk_focused_analysis",
    "company_comparison",
    "revenue_quality_analysis",
    "profitability_quality_analysis",
    "cash_flow_quality_analysis",
    "balance_sheet_analysis",
    "valuation_boundary_analysis",
    "investment_advice_like",
    "unsupported_prediction",
}

REVENUE_QUALITY_TERMS = (
    "收入质量",
    "营收质量",
    "收入怎么样",
    "营收怎么样",
    "收入增长",
    "营收增长",
    "revenue quality",
    "sales quality",
    "revenue growth",
)

PROFITABILITY_QUALITY_TERMS = (
    "盈利质量",
    "盈利能力",
    "利润率",
    "净利率",
    "毛利率",
    "营业利润率",
    "profitability",
    "margin",
    "net margin",
    "gross margin",
    "operating margin",
)

CASH_FLOW_QUALITY_TERMS = (
    "现金流",
    "利润能不能变成现金",
    "利润变成现金",
    "利润含金量",
    "自由现金流",
    "经营现金流",
    "cash flow",
    "free cash flow",
    "operating cash flow",
    "cash conversion",
)

BALANCE_SHEET_TERMS = (
    "资产负债",
    "抗风险能力",
    "资本投入",
    "资本强度",
    "现金债务",
    "债务",
    "库存",
    "应收",
    "balance sheet",
    "capital intensity",
    "debt",
    "inventory",
    "receivables",
)

VALUATION_BOUNDARY_TERMS = (
    "估值",
    "贵不贵",
    "便宜",
    "昂贵",
    "市盈率",
    "市销率",
    "market cap",
    "valuation",
    "cheap",
    "expensive",
    "p/e",
    "pe ratio",
    "p/s",
    "ps ratio",
)

SINGLE_COMPANY_OVERVIEW_TERMS = (
    "分析下",
    "分析一下",
    "分析",
    "帮我看看",
    "看看",
    "怎么样",
    "基本面",
    "研究一下",
    "总结一下",
    "综合分析",
    "analyze",
    "analysis",
    "take a look",
    "look at",
    "fundamental",
    "fundamentals",
    "how is",
)

UNSUPPORTED_PREDICTION_TERMS = (
    "预测明天股价",
    "明天股价",
    "明天会涨",
    "明天会跌",
    "tomorrow stock price",
    "stock price tomorrow",
    "will rise tomorrow",
    "will fall tomorrow",
)


def _is_unsupported_prediction_family(text: str) -> bool:
    return infer_safety_intent(str(text or "")) == "prediction"


def _company_count(companies: list[str], comparison_target: str | None = None) -> int:
    return len(set([c for c in companies if c] + ([comparison_target] if comparison_target else [])))


def detect_methodology_intent(
    user_query: str,
    *,
    companies: list[str] | None = None,
    comparison_target: str | None = None,
    safety_intent: str = "normal",
) -> str:
    """Compatibility wrapper around the centralized methodology classifier."""
    normalized = str(user_query or "").lower().strip()
    companies = companies or []
    if infer_safety_intent(normalized) == "prediction":
        return "unsupported_prediction"
    resolved = [
        ResolvedCompany(ticker=str(company).upper().strip(), canonical_name=str(company).upper().strip())
        for company in companies
        if str(company).strip()
    ]
    if comparison_target:
        target = str(comparison_target).upper().strip()
        if target and target not in {item.ticker for item in resolved}:
            resolved.append(ResolvedCompany(ticker=target, canonical_name=target))
    result = classify_methodology_intent(normalized, resolved)
    legacy = legacy_methodology_intent(result.methodology_intent)
    if safety_intent == "investment_advice_like":
        count = _company_count(companies, comparison_target)
        if legacy == "company_comparison" or count >= 2:
            return "investment_advice_like"
        if count == 1:
            return "valuation_boundary_analysis"
    return legacy


def _is_risk_focused_analysis_query(
    user_query: str,
    companies: list[str],
    comparison_target: str | None = None,
) -> bool:
    if _is_cautious_outlook_query(user_query):
        return False
    if re.search(r"(是多少|多少|what\s+is|how\s+much)", str(user_query or "").lower()) and _contains_any(
        user_query,
        ("营收", "收入", "利润", "revenue", "sales", "net income", "eps"),
    ):
        return False
    return detect_methodology_intent(
        user_query,
        companies=companies,
        comparison_target=comparison_target,
    ) == "risk_focused_analysis"


def _is_single_company_methodology_intent(intent: str) -> bool:
    return intent in {
        "single_company_overview",
        "risk_focused_analysis",
        "revenue_quality_analysis",
        "profitability_quality_analysis",
        "cash_flow_quality_analysis",
        "balance_sheet_analysis",
        "valuation_boundary_analysis",
    }


def _methodology_intent_dimensions(intent: str) -> dict[str, Any]:
    if intent == "risk_focused_analysis":
        return {
            "primary_dimension": "moat_and_competitive_risk",
            "required_dimensions": ["moat_and_competitive_risk"],
            "optional_dimensions": ["business_model", "revenue_quality", "profitability_quality"],
        }
    if intent == "single_company_overview":
        return {
            "primary_dimension": "",
            "required_dimensions": ["business_model", "revenue_quality", "profitability_quality", "moat_and_competitive_risk"],
            "optional_dimensions": ["cash_flow_quality", "balance_sheet_and_capital_intensity", "valuation_and_risk_boundary"],
        }
    if intent == "company_comparison":
        return {
            "primary_dimension": "",
            "required_dimensions": ["revenue_quality", "profitability_quality", "moat_and_competitive_risk"],
            "optional_dimensions": ["valuation_and_risk_boundary"],
        }
    if intent == "investment_advice_like":
        return {
            "primary_dimension": "",
            "required_dimensions": ["revenue_quality", "profitability_quality", "moat_and_competitive_risk"],
            "optional_dimensions": ["valuation_and_risk_boundary"],
        }
    if intent == "revenue_quality_analysis":
        return {"primary_dimension": "revenue_quality", "required_dimensions": ["revenue_quality"], "optional_dimensions": []}
    if intent == "profitability_quality_analysis":
        return {"primary_dimension": "profitability_quality", "required_dimensions": ["profitability_quality"], "optional_dimensions": []}
    if intent == "cash_flow_quality_analysis":
        return {"primary_dimension": "cash_flow_quality", "required_dimensions": ["cash_flow_quality"], "optional_dimensions": []}
    if intent == "balance_sheet_analysis":
        return {
            "primary_dimension": "balance_sheet_and_capital_intensity",
            "required_dimensions": ["balance_sheet_and_capital_intensity"],
            "optional_dimensions": [],
        }
    if intent == "valuation_boundary_analysis":
        return {
            "primary_dimension": "valuation_and_risk_boundary",
            "required_dimensions": ["valuation_and_risk_boundary"],
            "optional_dimensions": [],
        }
    return {"primary_dimension": "", "required_dimensions": [], "optional_dimensions": []}


def _analysis_scope_for_query(
    user_query: str,
    task_type: str,
    companies: list[str],
    comparison_target: str | None,
    methodology_intent: str = "",
) -> str:
    intent = methodology_intent or detect_methodology_intent(
        user_query,
        companies=companies,
        comparison_target=comparison_target,
    )
    if _is_single_company_methodology_intent(intent):
        return "single_company"
    if _is_single_company_open_analysis_query(user_query, companies, comparison_target):
        return "single_company"
    if intent in {"company_comparison", "investment_advice_like"}:
        return "comparison"
    if task_type == "company_comparison" or comparison_target or len(set(companies)) >= 2:
        return "comparison"
    return ""

def _is_unsupported_or_out_of_scope_query(user_query: str) -> bool:
    q = str(user_query or "").lower()
    unsupported_terms = (
        "天气",
        "菜谱",
        "做饭",
        "电影推荐",
        "体育比分",
        "weather",
        "recipe",
        "cook",
        "sports score",
        "movie recommendation",
    )
    return any(term in q for term in unsupported_terms)

def _normalize_safety_intent(raw: Any, user_query: str) -> str:
    safety = str(raw or "").strip()
    if safety not in SAFETY_INTENTS:
        safety = "normal"
    understood_safety = legacy_safety_intent(infer_safety_intent(user_query))
    if understood_safety != "normal":
        return understood_safety
    if _is_unsupported_or_out_of_scope_query(user_query):
        return "unsupported_or_out_of_scope"
    return safety

def _default_answer_mode_for_task(task_type: str) -> str:
    return DEFAULT_ANSWER_MODE_BY_TASK.get(_normalize_task_type(task_type), "direct_fact")

def _normalize_answer_mode(
    raw: Any,
    *,
    user_query: str,
    task_type: str,
    companies: list[str],
    comparison_target: str | None,
    safety_intent: str,
    methodology_intent: str = "",
) -> str:
    raw_mode = str(raw or "").strip()
    if raw_mode not in ANSWER_MODES:
        raw_mode = ""

    if _is_meta_query(user_query):
        return "meta"
    if _is_underspecified_analysis_query(user_query, companies):
        return "clarification"
    if safety_intent == "unsupported_or_out_of_scope":
        return "refusal_or_redirect"
    intent = str(methodology_intent or "").strip() or detect_methodology_intent(
        user_query,
        companies=companies,
        comparison_target=comparison_target,
        safety_intent=safety_intent,
    )
    if intent == "unsupported_prediction":
        return "refusal_or_redirect"
    if intent == "risk_focused_analysis":
        return "risk_focused_analysis"
    if intent in {
        "single_company_overview",
        "revenue_quality_analysis",
        "profitability_quality_analysis",
        "cash_flow_quality_analysis",
        "balance_sheet_analysis",
        "valuation_boundary_analysis",
    }:
        return "analytical"
    if intent in {"company_comparison", "investment_advice_like"} and (
        task_type == "company_comparison" or comparison_target or len(set(companies)) >= 2
    ):
        return "comparison_brief"
    if _is_cautious_outlook_query(user_query):
        return "cautious_outlook"
    if raw_mode in {"meta", "clarification", "refusal_or_redirect"}:
        raw_mode = ""
    if raw_mode:
        return raw_mode
    return _default_answer_mode_for_task(task_type)

def _needs_tools_for_answer_mode(answer_mode: str, safety_intent: str) -> bool:
    if answer_mode in {"meta", "clarification", "refusal_or_redirect"}:
        return False
    if safety_intent == "unsupported_or_out_of_scope":
        return False
    return True

def _build_clarification_question(user_query: str, answer_mode: str) -> str | None:
    if answer_mode != "clarification":
        return None
    if _has_chinese(user_query):
        return "你想分析哪家公司或哪类问题？请补充 ticker/公司名，以及关注财务指标、风险因素、经营趋势或公司对比。"
    return (
        "Which company or filing topic should I analyze? Please include a ticker/company and whether you care "
        "about metrics, risks, operating trends, or a company comparison."
    )

def _normalize_needs_clarification(answer_mode: str) -> bool:
    return answer_mode == "clarification"

def detect_answer_mode(
    user_query: str,
    task_type: str = "fact_qa",
    companies: list[str] | None = None,
    comparison_target: str | None = None,
    parsed_answer_mode: str | None = None,
    parsed_safety_intent: str | None = None,
) -> str:
    safety_intent = _normalize_safety_intent(parsed_safety_intent, user_query)
    return _normalize_answer_mode(
        parsed_answer_mode,
        user_query=user_query,
        task_type=task_type,
        companies=companies or [],
        comparison_target=comparison_target,
        safety_intent=safety_intent,
    )

def _is_trend_intent_query(user_query: str) -> bool:
    return _contains_any(user_query, TREND_TERMS)

def _is_causal_revenue_growth_query(user_query: str) -> bool:
    q = str(user_query or "").lower()
    causal = ("为什么", "原因", "由什么驱动", "驱动", "why", "driven by", "what drove")
    revenue = ("营收", "收入", "revenue", "sales")
    growth = ("增长", "增速", "growth", "grew", "increase", "increased")
    return _contains_any(q, causal) and _contains_any(q, revenue) and _contains_any(q, growth)


def _explicit_metric_terms_from_query(user_query: str) -> list[str]:
    q = str(user_query or "").lower()
    metrics: list[str] = []
    if any(term in q for term in ("毛利率", "gross margin")):
        metrics.append("gross_margin")
    if any(term in q for term in ("增长质量", "收入增长", "营收增长", "revenue growth", "growth quality")):
        metrics.append("revenue_growth")
    return metrics


def _is_summary_intent_query(user_query: str) -> bool:
    return _contains_any(user_query, SUMMARY_TERMS)

def _apply_task_type_guardrails(
    user_query: str,
    task_type: str,
    companies: list[str],
    comparison_target: str | None,
    methodology_intent: str = "",
    analysis_scope: str = "",
) -> str:
    """Program-level task routing guardrails to reduce LLM drift."""
    q = str(user_query or "").lower()
    task = _normalize_task_type(task_type)

    intent = str(methodology_intent or "").strip() or detect_methodology_intent(
        q,
        companies=companies,
        comparison_target=comparison_target,
        safety_intent=_normalize_safety_intent(None, q),
    )
    if analysis_scope == "comparison" or intent in {"company_comparison", "investment_advice_like"}:
        return "company_comparison"

    summary_intent = _is_summary_intent_query(q)
    trend_intent = _is_trend_intent_query(q)

    if _is_causal_revenue_growth_query(q) and len(companies) == 1:
        return "report_summary"
    if analysis_scope == "single_company" and _is_single_company_methodology_intent(intent):
        return "report_summary"
    if _is_single_company_methodology_intent(intent):
        return "report_summary"
    if _is_single_company_open_analysis_query(q, companies, comparison_target):
        return "report_summary"
    if summary_intent and not trend_intent:
        return "report_summary"
    if "综合分析" in q or "总结" in q or "概括" in q:
        return "report_summary"
    if trend_intent:
        return "trend_analysis"
    return task

def _detect_event_intent(
    user_query: str,
    task_type: str = "fact_qa",
) -> str:
    """Return event intent level: none | optional | required."""
    q = str(user_query or "").lower()
    task = _normalize_task_type(task_type)
    has_market_terms = _has_market_reaction_terms(q)
    has_event_anchor = _has_event_anchor_terms(q)
    required = has_market_terms and has_event_anchor
    partial = has_market_terms or has_event_anchor

    # Negative constraints for non-event default paths.
    if not required:
        if task in {"fact_qa", "company_comparison"}:
            return "none"
        if task == "report_summary" and not has_market_terms:
            return "none"
        return "optional" if partial else "none"
    return "required"

def _is_market_reaction_query(user_query: str, task_type: str = "fact_qa") -> bool:
    return _detect_event_intent(user_query, task_type=task_type) == "required"

def _extract_event_window_days(user_query: str) -> list[int]:
    q = str(user_query or "").lower()
    found: list[int] = []
    for n in EVENT_WINDOW_DAYS:
        if re.search(rf"\b{n}\s*(?:day|days|d)\b", q) or re.search(rf"{n}\s*天", q):
            found.append(n)
    return found or [1, 5, 10]

def _infer_event_type(user_query: str) -> str:
    q = str(user_query or "").lower().replace(" ", "")
    if "10-q" in q or "10q" in q:
        return "10Q"
    if "10-k" in q or "10k" in q:
        return "10K"
    return "any"

def _build_event_query(
    user_query: str,
    task_type: str,
    period_query: dict[str, Any],
) -> dict[str, Any]:
    windows = _extract_event_window_days(user_query)
    mode = str(period_query.get("period_type") or "")
    latest_n = 4
    if mode == "latest":
        latest_n = 1
    elif mode == "trailing":
        latest_n = int(period_query.get("trailing_n") or 4)
    elif task_type == "company_comparison":
        latest_n = 4
    sort_by = "return_abs" if _contains_any(user_query, ("最强", "最大反应", "strongest", "largest reaction")) else "event_date"
    return {
        "event_type": _infer_event_type(user_query),
        "latest_n": max(1, min(latest_n, 12)),
        "window_days": windows,
        "sort_by": sort_by,
        "sort_order": "desc",
    }

def _parse_quarter_token(token: str) -> int | None:
    token = (token or "").strip().lower()
    mapping = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "q1": 1,
        "q2": 2,
        "q3": 3,
        "q4": 4,
    }
    return mapping.get(token)

def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _infer_comparison_basis_from_query(user_query: str) -> str:
    if _contains_any(user_query, ("各自最新财年", "各自最新年度", "respective latest fiscal year", "latest fiscal year each")):
        return "latest_fiscal_year"
    if _contains_any(user_query, ("自然年", "calendar year")):
        return "calendar_year"
    return "same_period"

def _extract_period_query_from_text(user_query: str, today: date) -> dict[str, Any]:
    pq = _default_period_query()
    q = user_query or ""
    q_lower = q.lower()

    # "最近四个季度" / "last 4 quarters"
    trailing_match = re.search(r"最近\s*([0-9一二三四五六七八九十]+)\s*个?\s*季度", q)
    if trailing_match:
        raw_n = trailing_match.group(1)
        cn_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        n = _coerce_int(raw_n)
        if n is None:
            n = cn_map.get(raw_n)
        if n is not None:
            pq.update({"period_type": "trailing", "trailing_n": max(1, min(n, 16)), "is_explicit": True})
            return pq

    trailing_en = re.search(r"(?:last|recent)\s*(\d+)\s*quarters?", q_lower)
    if trailing_en:
        n = _coerce_int(trailing_en.group(1))
        if n is not None:
            pq.update({"period_type": "trailing", "trailing_n": max(1, min(n, 16)), "is_explicit": True})
            return pq

    if _contains_any(q, ("最近几个季度", "recent quarters")):
        pq.update({"period_type": "trailing", "trailing_n": 4, "is_explicit": False})
        return pq

    if _contains_any(q, ("最近一个季度", "最新一个季度", "latest quarter", "current quarter")):
        pq.update({"period_type": "latest", "is_explicit": False})
        return pq

    # 2025Q1 / Q1 2025
    m = re.search(r"(20\d{2})\s*[qQ]\s*([1-4])", q)
    if not m:
        m = re.search(r"[qQ]\s*([1-4])\s*(20\d{2})", q)
        if m:
            year = _coerce_int(m.group(2))
            quarter = _coerce_int(m.group(1))
            pq.update(
                {
                    "period_type": "quarterly",
                    "year": year,
                    "quarter": quarter,
                    "year_basis": "fiscal",
                    "is_explicit": True,
                }
            )
            return pq
    if m:
        year = _coerce_int(m.group(1))
        quarter = _coerce_int(m.group(2))
        pq.update(
            {
                "period_type": "quarterly",
                "year": year,
                "quarter": quarter,
                "year_basis": "fiscal",
                "is_explicit": True,
            }
        )
        return pq

    # 2025年第一季度
    m = re.search(r"(20\d{2})\s*年\s*第?\s*([一二三四1-4])\s*季(?:度)?", q)
    if m:
        year = _coerce_int(m.group(1))
        quarter = _parse_quarter_token(m.group(2))
        pq.update(
            {
                "period_type": "quarterly",
                "year": year,
                "quarter": quarter,
                "year_basis": "fiscal",
                "is_explicit": True,
            }
        )
        return pq

    # FY2025 / 2025财年
    m = re.search(r"(?:fy\s*(20\d{2})|(20\d{2})\s*财年)", q_lower)
    if m:
        year = _coerce_int(m.group(1) or m.group(2))
        pq.update(
            {
                "period_type": "annual",
                "year": year,
                "year_basis": "fiscal",
                "is_explicit": True,
            }
        )
        return pq

    # 2025 calendar year / 2025自然年
    m = re.search(r"(20\d{2}).*(?:自然年|calendar year)", q_lower)
    if m:
        year = _coerce_int(m.group(1))
        pq.update(
            {
                "period_type": "annual",
                "year": year,
                "year_basis": "calendar",
                "is_explicit": True,
            }
        )
        return pq

    # 2025年全年 / 2025年度
    m = re.search(r"(20\d{2})\s*年.*(?:全年|年度|annual|year)", q_lower)
    if m:
        year = _coerce_int(m.group(1))
        pq.update(
            {
                "period_type": "annual",
                "year": year,
                "year_basis": "fiscal",
                "is_explicit": True,
            }
        )
        return pq

    # 去年
    if _contains_any(q, ("去年", "last year")):
        year = today.year - 1
        if _contains_any(q, ("q1", "q2", "q3", "q4", "季度", "quarter")):
            mq = re.search(r"[qQ]\s*([1-4])", q)
            quarter = _coerce_int(mq.group(1)) if mq else None
            if quarter is None and _contains_any(q, ("q几", "第几季度")):
                pq.update(
                    {
                        "period_type": "quarterly",
                        "year": year,
                        "year_basis": "fiscal",
                        "is_explicit": True,
                        "needs_clarification": True,
                        "clarification_reason": "quarter_unspecified_for_last_year",
                    }
                )
                return pq
            pq.update(
                {
                    "period_type": "quarterly",
                    "year": year,
                    "quarter": quarter,
                    "year_basis": "fiscal",
                    "is_explicit": True,
                    "needs_clarification": quarter is None,
                    "clarification_reason": "quarter_missing" if quarter is None else None,
                }
            )
            return pq
        pq.update(
            {
                "period_type": "annual",
                "year": year,
                "year_basis": "fiscal",
                "is_explicit": True,
            }
        )
        return pq

    # "全年/年度" but no year -> clarify
    if _contains_any(q, ("全年", "年度", "annual", "yearly")) and not re.search(r"20\d{2}", q):
        pq.update(
            {
                "period_type": "annual",
                "year_basis": "fiscal",
                "needs_clarification": True,
                "clarification_reason": "year_missing_for_annual_query",
            }
        )
        return pq

    # "Q1/第一季度" no year -> clarify
    if re.search(r"[qQ]\s*[1-4]", q) or _contains_any(q, ("第一季度", "第二季度", "第三季度", "第四季度")):
        quarter = None
        mq = re.search(r"[qQ]\s*([1-4])", q)
        if mq:
            quarter = _coerce_int(mq.group(1))
        if quarter is None:
            zh_map = {"第一季度": 1, "第二季度": 2, "第三季度": 3, "第四季度": 4}
            for k, v in zh_map.items():
                if k in q:
                    quarter = v
                    break
        pq.update(
            {
                "period_type": "quarterly",
                "quarter": quarter,
                "year_basis": "fiscal",
                "needs_clarification": True,
                "clarification_reason": "year_missing_for_quarter_query",
            }
        )
        return pq

    return pq

def _normalize_period_query(user_query: str, raw: Any, today: date, task_type: str) -> dict[str, Any]:
    pq = _default_period_query()

    if isinstance(raw, dict):
        if raw.get("period_type") in PERIOD_TYPES:
            pq["period_type"] = raw.get("period_type")
        y = _coerce_int(raw.get("year"))
        if y and 1900 <= y <= 2100:
            pq["year"] = y
        q = _coerce_int(raw.get("quarter"))
        if q in (1, 2, 3, 4):
            pq["quarter"] = q
        n = _coerce_int(raw.get("trailing_n"))
        if n and n > 0:
            pq["trailing_n"] = min(n, 16)
        if raw.get("year_basis") in YEAR_BASIS:
            pq["year_basis"] = raw.get("year_basis")
        if raw.get("comparison_basis") in COMPARISON_BASIS:
            pq["comparison_basis"] = raw.get("comparison_basis")
        if isinstance(raw.get("is_explicit"), bool):
            pq["is_explicit"] = raw.get("is_explicit")
        if isinstance(raw.get("needs_clarification"), bool):
            pq["needs_clarification"] = raw.get("needs_clarification")
        reason = raw.get("clarification_reason")
        if reason is not None:
            pq["clarification_reason"] = str(reason)

    parsed = _extract_period_query_from_text(user_query, today)
    for key in ("period_type", "year", "quarter", "trailing_n", "year_basis"):
        if parsed.get(key) is not None:
            pq[key] = parsed.get(key)
    if parsed.get("is_explicit"):
        pq["is_explicit"] = True
    if parsed.get("needs_clarification"):
        pq["needs_clarification"] = True
        pq["clarification_reason"] = parsed.get("clarification_reason")

    if task_type == "trend_analysis" and pq.get("period_type") is None:
        pq["period_type"] = "trailing"
        pq["trailing_n"] = 4
        pq["is_explicit"] = False
    if task_type in ("fact_qa", "company_comparison") and pq.get("period_type") is None:
        pq["period_type"] = "latest"
        pq["is_explicit"] = False

    pq["comparison_basis"] = _infer_comparison_basis_from_query(user_query)
    if task_type != "company_comparison":
        pq["comparison_basis"] = "same_period"
    elif pq["comparison_basis"] not in COMPARISON_BASIS:
        pq["comparison_basis"] = "same_period"

    if pq.get("period_type") == "trailing" and not pq.get("trailing_n"):
        pq["trailing_n"] = 4
    if pq.get("period_type") == "annual" and pq.get("year_basis") not in YEAR_BASIS:
        pq["year_basis"] = "fiscal"
    if pq.get("period_type") == "quarterly" and pq.get("year_basis") not in YEAR_BASIS:
        pq["year_basis"] = "fiscal"

    return pq

def _resolve_query_plan(state: AgentState) -> dict[str, Any]:
    user_query = state.get("user_query", "")
    format_constraints = _format_constraints_from_query(str(user_query or ""))
    task_type = state.get("task_type", "fact_qa")
    raw_period_query = state.get("period_query")
    if isinstance(raw_period_query, dict) and raw_period_query:
        period_query = dict(raw_period_query)
    else:
        period_query = _normalize_period_query(
            user_query=user_query,
            raw=None,
            today=date.today(),
            task_type=task_type,
        )
    comparison_basis = period_query.get("comparison_basis") or "same_period"
    if task_type == "company_comparison" and comparison_basis not in COMPARISON_BASIS:
        comparison_basis = "same_period"
    if task_type != "company_comparison":
        comparison_basis = "same_period"
    period_query["comparison_basis"] = comparison_basis

    target_period_type = "quarterly"
    if period_query.get("period_type") == "annual":
        target_period_type = "annual"
    elif period_query.get("period_type") == "latest":
        if _contains_any(user_query, ("财年", "全年", "年度", "annual", "fiscal year", "yearly")):
            target_period_type = "annual"
    elif period_query.get("period_type") == "trailing":
        if _contains_any(user_query, ("年", "年度", "year", "annual")) and not _contains_any(user_query, ("季度", "quarter", "q1", "q2", "q3", "q4")):
            target_period_type = "annual"

    needs_clarification = bool(period_query.get("needs_clarification"))
    reason = period_query.get("clarification_reason")
    if task_type in ("fact_qa", "company_comparison", "trend_analysis"):
        p_type = period_query.get("period_type")
        if p_type == "quarterly" and (period_query.get("year") is None or period_query.get("quarter") is None):
            needs_clarification = True
            reason = reason or "missing_year_or_quarter_for_quarterly_query"
        if p_type == "annual" and period_query.get("year") is None:
            # Annual queries without year should be clarified.
            needs_clarification = True
            reason = reason or "missing_year_for_annual_query"

    if comparison_basis == "same_period":
        basis_label = "同季度" if target_period_type == "quarterly" else "同财年（同年）"
    elif comparison_basis == "latest_fiscal_year":
        basis_label = "各自最新财年（日期不同）"
    else:
        basis_label = "自然年（calendar year）"

    resolved_context = {
        "target_period_type": target_period_type,
        "comparison_basis": comparison_basis,
        "needs_clarification": needs_clarification,
        "clarification_reason": reason,
        "strict_period_match": True,
        "same_period_match": None,
        "common_periods": [],
    }
    return {
        "period_query": period_query,
        "resolved_period_context": resolved_context,
        "comparison_basis_label": basis_label,
    }

def _extract_tickers_fallback(query: str) -> list[str]:
    """Last-resort ticker extraction delegated to QueryUnderstanding entity resolution."""
    return [item.ticker for item in resolve_companies(query).resolved_companies]

def _reject_plan_item(
    rejected: list[dict[str, Any]],
    *,
    item_type: str,
    value: Any,
    reason: str,
) -> None:
    rejected.append(
        {
            "type": item_type,
            "value": value,
            "reason": reason,
        }
    )

def _default_policy_for_safety(answer_mode: str, safety_intent: str) -> dict[str, Any]:
    policy = {
        "use_validated_evidence_only": True,
        "require_citations_for_external_claims": True,
        "allow_investment_recommendation": False,
        "allow_price_prediction": False,
    }
    if answer_mode == "cautious_outlook":
        policy["forward_looking_caution"] = True
    if safety_intent == "investment_advice_like":
        policy["non_advisory_comparison_only"] = True
    return policy

def _plan_default_sections(answer_mode: str, task_type: str, safety_intent: str) -> list[str]:
    if answer_mode == "risk_focused_analysis":
        return ["ITEM_1A", "ITEM_7", "ITEM_1", "ITEM_2"]
    if answer_mode in {"cautious_outlook", "analytical"}:
        return list(ANALYTICAL_SECTION_PREFERENCES)
    if task_type == "company_comparison" and safety_intent == "investment_advice_like":
        return list(ANALYTICAL_SECTION_PREFERENCES)
    return []

def _plan_required_tools(answer_mode: str, task_type: str, safety_intent: str, event_intent: str) -> list[str]:
    required: list[str] = []
    if answer_mode == "cautious_outlook":
        required.extend(["query_financial_data", "compute_metrics", "search_filings"])
    elif answer_mode == "risk_focused_analysis":
        required.extend(["search_filings", "query_financial_data", "compute_metrics"])
    elif answer_mode == "analytical":
        required.append("search_filings")
    if task_type == "company_comparison" and safety_intent == "investment_advice_like":
        required.extend(["query_financial_data", "compute_metrics", "search_filings"])
    if event_intent == "required":
        required.append("query_event_price_window")
    return _ordered_unique(required)

def _plan_default_metrics(answer_mode: str, task_type: str, safety_intent: str) -> list[str]:
    if answer_mode == "cautious_outlook":
        return ["revenue", "net_income", "operating_margin"]
    if answer_mode == "risk_focused_analysis":
        return ["revenue", "net_income"]
    if task_type == "company_comparison" and safety_intent == "investment_advice_like":
        return ["revenue", "net_income", "operating_margin"]
    return []

def _merge_validated_tools(
    *,
    deterministic_tools: list[str],
    proposed_tools: list[str],
    answer_mode: str,
    task_type: str,
    safety_intent: str,
    event_intent: str,
    needs_tools: bool,
    rejected: list[dict[str, Any]],
) -> list[str]:
    if not needs_tools:
        return []
    if answer_mode == "direct_fact":
        for tool in proposed_tools:
            if tool not in deterministic_tools:
                _reject_plan_item(
                    rejected,
                    item_type="tool",
                    value=tool,
                    reason="ignored_for_direct_fact_deterministic_route",
                )
        return _ordered_unique([t for t in deterministic_tools if t in ALLOWED_ANALYSIS_TOOLS])

    merged = [t for t in deterministic_tools if t in ALLOWED_ANALYSIS_TOOLS]
    for tool in proposed_tools:
        if tool == "query_event_price_window" and event_intent != "required":
            _reject_plan_item(
                rejected,
                item_type="tool",
                value=tool,
                reason="event_window_tool_requires_event_intent",
            )
            continue
        if tool not in merged:
            merged.append(tool)
    for tool in _plan_required_tools(answer_mode, task_type, safety_intent, event_intent):
        if tool not in merged:
            merged.append(tool)
    return _ordered_unique(merged)

def build_validated_analysis_plan(
    *,
    user_query: str,
    raw_plan: Any,
    task_type: str,
    answer_mode: str,
    safety_intent: str,
    methodology_intent: str = "",
    analysis_scope: str = "",
    time_policy: str = "",
    period_scope: str = "",
    companies: list[str],
    comparison_target: str | None,
    time_range: dict[str, str] | None,
    period_query: dict[str, Any],
    requested_metrics: list[str],
    deterministic_tools: list[str],
    needs_tools: bool,
    event_intent: str,
) -> AnalysisPlan:
    """Validate an LLM-proposed plan into a program-approved AnalysisPlan."""
    raw = raw_plan if isinstance(raw_plan, Mapping) else {}
    rejected: list[dict[str, Any]] = []
    user_intent = str(raw.get("user_intent") or user_query).strip()[:500]
    raw_task_type = _normalize_task_type(str(raw.get("task_type", task_type)))
    raw_safety = _normalize_safety_intent(raw.get("safety_intent", safety_intent), user_query)
    raw_answer_mode = _normalize_answer_mode(
        raw.get("answer_mode", answer_mode),
        user_query=user_query,
        task_type=raw_task_type,
        companies=companies,
        comparison_target=comparison_target,
        safety_intent=raw_safety,
        methodology_intent=methodology_intent,
    )
    if raw_task_type != task_type:
        _reject_plan_item(rejected, item_type="task_type", value=raw.get("task_type"), reason="overridden_by_router")
    if raw_answer_mode != answer_mode:
        _reject_plan_item(
            rejected,
            item_type="answer_mode",
            value=raw.get("answer_mode"),
            reason="overridden_by_answer_mode_guardrails",
        )
    if raw_safety != safety_intent:
        _reject_plan_item(
            rejected,
            item_type="safety_intent",
            value=raw.get("safety_intent"),
            reason="overridden_by_safety_guardrails",
        )

    plan_companies = list(companies)
    for item in _string_list(raw.get("companies")):
        ticker = _normalize_ticker_value(item)
        if ticker and ticker not in plan_companies:
            plan_companies.append(ticker)
        elif not ticker:
            _reject_plan_item(rejected, item_type="ticker", value=item, reason="unknown_or_unsupported_ticker")
    if comparison_target and comparison_target not in plan_companies:
        plan_companies.append(comparison_target)

    metrics: list[str] = []
    for metric in _string_list(raw.get("metric_requirements")) + _string_list(raw.get("metrics")):
        normalized = _normalize_metric_value(metric)
        if normalized and normalized not in metrics:
            metrics.append(normalized)
        elif not normalized:
            _reject_plan_item(rejected, item_type="metric", value=metric, reason="metric_not_allowed")
    for metric in requested_metrics:
        normalized = _normalize_metric_value(metric)
        if normalized and normalized not in metrics:
            metrics.append(normalized)
        elif not normalized:
            _reject_plan_item(rejected, item_type="metric", value=metric, reason="requested_metric_not_allowed")
    if analysis_scope == "single_company":
        for metric in ("revenue", "net_income"):
            if metric not in metrics:
                metrics.append(metric)
    for metric in _plan_default_metrics(answer_mode, task_type, safety_intent):
        if metric not in metrics:
            metrics.append(metric)

    sections: list[str] = []
    for section in _string_list(raw.get("section_preferences")) + _string_list(raw.get("sections")):
        normalized = _normalize_section_value(section)
        if normalized and normalized not in sections:
            sections.append(normalized)
        elif not normalized:
            _reject_plan_item(rejected, item_type="section", value=section, reason="section_not_allowed")
    for section in _plan_default_sections(answer_mode, task_type, safety_intent):
        if section not in sections:
            sections.append(section)

    proposed_tools: list[str] = []
    for tool in _string_list(raw.get("proposed_tools")):
        if tool in ALLOWED_ANALYSIS_TOOLS and tool not in proposed_tools:
            proposed_tools.append(tool)
        elif tool not in ALLOWED_ANALYSIS_TOOLS:
            _reject_plan_item(rejected, item_type="tool", value=tool, reason="tool_not_allowed")
    validated_tools = _merge_validated_tools(
        deterministic_tools=deterministic_tools,
        proposed_tools=proposed_tools,
        answer_mode=answer_mode,
        task_type=task_type,
        safety_intent=safety_intent,
        event_intent=event_intent,
        needs_tools=needs_tools,
        rejected=rejected,
    )

    answer_policy = raw.get("answer_policy") if isinstance(raw.get("answer_policy"), dict) else {}
    answer_policy = {**_default_policy_for_safety(answer_mode, safety_intent), **answer_policy}
    answer_policy["allow_investment_recommendation"] = False
    answer_policy["allow_price_prediction"] = False
    intent = methodology_intent or detect_methodology_intent(
        user_query,
        companies=companies,
        comparison_target=comparison_target,
        safety_intent=safety_intent,
    )
    dimension_map = _methodology_intent_dimensions(intent)
    primary_dimension = str(dimension_map.get("primary_dimension", ""))
    required_dimensions = list(dimension_map.get("required_dimensions", []) or [])
    optional_dimensions = list(dimension_map.get("optional_dimensions", []) or [])

    return AnalysisPlan(
        user_intent=user_intent,
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        methodology_intent=intent,
        analysis_scope=analysis_scope,
        primary_dimension=primary_dimension,
        required_dimensions=required_dimensions,
        optional_dimensions=optional_dimensions,
        time_policy=time_policy,
        period_scope=period_scope,
        companies=plan_companies,
        time_range=time_range,
        analysis_dimensions=_string_list(raw.get("analysis_dimensions"), max_items=8),
        needed_evidence=_string_list(raw.get("needed_evidence"), max_items=8),
        proposed_tools=proposed_tools,
        validated_tools=validated_tools,
        section_preferences=sections,
        metric_requirements=metrics,
        answer_policy=answer_policy,
        rejected_plan_items=rejected,
        period_query=period_query,
    )

validate_analysis_plan = build_validated_analysis_plan

def _has_explicit_year(query: str) -> bool:
    """Whether user explicitly mentioned a calendar year like 2023 / 2023年."""
    return bool(re.search(r"(?:^|[^0-9])(20\d{2})(?:年|[^0-9]|$)", query))

def _is_recency_query(query: str) -> bool:
    """Whether user asks for latest/current/recent information."""
    q = query.lower()
    recency_markers = (
        "最近", "最新", "当前", "目前", "现在", "近期",
        "latest", "recent", "current", "now", "as of now",
    )
    return any(m in q for m in recency_markers)

def _sanitize_time_range_for_recency(
    user_query: str,
    time_range: Any,
    today: date | None = None,
) -> dict[str, str] | None:
    """Clear inferred ranges for recency queries without explicit year.

    For "latest/recent/current/最近/最新/当前" style questions we prefer
    ranking by the newest available data in our local store rather than
    trusting an LLM-inferred calendar range. This is intentionally strict:
    even a *recent-looking* inferred range can still anchor the query to
    the wrong quarter/year and silently return a stale snapshot.
    """
    if not isinstance(time_range, dict):
        return None
    start = time_range.get("start")
    end = time_range.get("end")
    if not start or not end:
        return time_range

    if _has_explicit_year(user_query):
        return time_range
    if not _is_recency_query(user_query):
        return time_range

    today = today or date.today()
    logger.info(
        "Clearing inferred time_range for recency query without explicit year: %s (today=%s)",
        time_range,
        today.isoformat(),
    )
    return None

def _select_tools(
    task_type: str,
    data_route: str,
    user_query: str = "",
    event_intent: str | None = None,
) -> list[str]:
    intent = str(event_intent or _detect_event_intent(user_query, task_type=task_type)).lower()
    if intent not in EVENT_INTENT_TYPES:
        intent = "none"
    market_reaction = intent == "required"

    if data_route == "documents_only":
        tools = ["search_filings"]
        if market_reaction:
            tools.append("query_event_price_window")
        return tools
    if data_route == "structured_only":
        tools = ["query_financial_data"]
        if task_type in ("trend_analysis", "company_comparison") and not market_reaction:
            tools.append("compute_metrics")
        if market_reaction:
            tools.append("query_event_price_window")
        return tools
    # hybrid
    tools = ["search_filings", "query_financial_data"]
    if task_type in ("trend_analysis", "company_comparison") and not market_reaction:
        tools.append("compute_metrics")
    if market_reaction:
        tools.append("query_event_price_window")
    return tools

def _infer_period_type(state: AgentState) -> str | None:
    """Infer quarterly vs annual for structured querying."""
    period_query = state.get("period_query", {})
    pq_type = str(period_query.get("period_type", "") or "")
    if pq_type in ("quarterly", "annual"):
        return pq_type
    if pq_type in ("latest", "trailing"):
        target = str(state.get("resolved_period_context", {}).get("target_period_type", "") or "")
        if target in ("quarterly", "annual"):
            return target

    query_lower = state.get("user_query", "").lower()
    if any(kw in query_lower for kw in ("季度", "quarter", "q1", "q2", "q3", "q4", "qoq")):
        return "quarterly"
    if any(kw in query_lower for kw in ("年度", "annual", "全年", "yearly", "yoy")):
        return "annual"
    tr = state.get("time_range")
    if tr and tr.get("start") and tr.get("end"):
        try:
            from datetime import date as _d
            s = _d.fromisoformat(tr["start"])
            e = _d.fromisoformat(tr["end"])
            span = (e - s).days
            if span <= 100:
                return "quarterly"
        except ValueError:
            pass
    return None

def _section_constraints_for_query(task_type: str, user_query: str) -> tuple[list[str] | None, bool]:
    """Infer section allowlist for narrative queries (report_summary first)."""
    if task_type != "report_summary":
        return None, False

    q = (user_query or "").lower()
    risk_terms = ("风险", "risk", "risk factor", "uncertaint")
    mda_terms = ("经营分析", "管理层讨论", "展望", "mda", "md&a", "outlook", "management discussion")
    legal_terms = ("诉讼", "法律", "监管", "法规", "legal", "litigation", "regulator", "compliance")

    if any(t in q for t in risk_terms):
        return ["ITEM_1A", "ITEM_7", "ITEM_2"], False
    if any(t in q for t in mda_terms):
        return ["ITEM_2", "ITEM_7"], False
    if any(t in q for t in legal_terms):
        return ["ITEM_3", "ITEM_1"], False
    return None, False

def _is_risk_intent_query(user_query: str) -> bool:
    q = (user_query or "").lower()
    return any(
        k in q
        for k in (
            "风险",
            "风险因素",
            "risk",
            "risk factor",
            "litigation",
            "legal",
            "监管",
            "合规",
            "compliance",
            "regulator",
        )
    )

def _build_retrieval_policy(state: AgentState, selected_tools: list[str]) -> dict[str, Any]:
    task_type = str(state.get("task_type", "fact_qa"))
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    safety_intent = str(state.get("safety_intent", "normal"))
    user_query = str(state.get("user_query", ""))
    data_route = str(state.get("data_route", "hybrid"))
    analysis_plan = dict(state.get("analysis_plan", {}) or {})
    is_risk = _is_risk_intent_query(user_query)
    event_intent = str(state.get("event_intent", _detect_event_intent(user_query, task_type=task_type)))
    market_reaction = event_intent == "required"

    profile = "summary"
    if task_type == "fact_qa":
        profile = "fact_support"
    elif task_type == "trend_analysis":
        profile = "trend_support"
    elif task_type == "company_comparison":
        profile = "comparison_support"
    elif task_type == "report_summary":
        profile = "risk_summary" if is_risk else "summary"

    text_top_k = {
        "fact_qa": 2,
        "trend_analysis": 3,
        "company_comparison": 4,
        "report_summary": 5,
    }.get(task_type, 3)
    if "search_filings" not in selected_tools:
        text_top_k = 0
    if data_route == "structured_only":
        text_top_k = 0

    section_allowlist, strict_sections = _section_constraints_for_query(task_type, user_query)
    if profile == "risk_summary" and not section_allowlist:
        section_allowlist = ["ITEM_1A", "ITEM_7", "ITEM_2"]
    plan_sections = [
        section
        for section in analysis_plan.get("section_preferences", [])
        if isinstance(section, str) and section in KNOWN_SEC_SECTIONS
    ]
    if answer_mode in {"cautious_outlook", "analytical"} or (
        task_type == "company_comparison" and safety_intent == "investment_advice_like"
    ):
        for section in ANALYTICAL_SECTION_PREFERENCES:
            if section not in plan_sections:
                plan_sections.append(section)
    if plan_sections and "search_filings" in selected_tools:
        section_allowlist = _ordered_unique(list(section_allowlist or []) + plan_sections)

    max_per_filing = {
        "fact_qa": 1,
        "trend_analysis": 2,
        "company_comparison": 1,
        "report_summary": 2,
    }.get(task_type, 2)
    max_per_section = {
        "fact_qa": 1,
        "trend_analysis": 1,
        "company_comparison": 1,
        "report_summary": 2,
    }.get(task_type, 1)
    if plan_sections and "search_filings" in selected_tools:
        text_top_k = max(text_top_k, 4 if task_type == "company_comparison" else 5)
        max_per_section = max(max_per_section, 2)

    return {
        "retrieval_profile": profile,
        "text_top_k": int(max(0, text_top_k)),
        "max_per_filing": int(max(0, max_per_filing)),
        "max_per_section": int(max(0, max_per_section)),
        "comparison_text_cap_per_company": 2,
        "require_balanced_comparison_text": task_type == "company_comparison",
        "skip_fact_text_when_structured_sufficient": task_type == "fact_qa",
        "section_allowlist": section_allowlist,
        "strict_sections": bool(strict_sections),
        "event_intent": event_intent,
        "market_reaction_requested": market_reaction,
    }

def build_query_plan(state_or_raw: Mapping[str, Any]) -> QueryPlan:
    """Build a typed query plan while preserving legacy dict semantics."""
    raw = _resolve_query_plan(dict(state_or_raw))
    user_query = str(state_or_raw.get("user_query", "")) if isinstance(state_or_raw, Mapping) else ""
    raw_query_understanding = (
        state_or_raw.get("query_understanding_summary") or state_or_raw.get("query_understanding")
        if isinstance(state_or_raw, Mapping)
        else {}
    )
    if not raw_query_understanding and isinstance(state_or_raw, Mapping):
        raw_query_understanding = query_understanding_summary(build_query_understanding(user_query))
    query_understanding = _query_understanding_payload(raw_query_understanding)
    selected_tools = list(state_or_raw.get("selected_tools", [])) if isinstance(state_or_raw, Mapping) else []
    retrieval_policy = state_or_raw.get("retrieval_policy", {}) if isinstance(state_or_raw, Mapping) else {}
    event_query = state_or_raw.get("event_query", {}) if isinstance(state_or_raw, Mapping) else {}
    analysis_plan = state_or_raw.get("analysis_plan", {}) if isinstance(state_or_raw, Mapping) else {}
    evidence_plan = state_or_raw.get("evidence_plan", {}) if isinstance(state_or_raw, Mapping) else {}
    task_type = str(state_or_raw.get("task_type", "fact_qa"))
    understood_companies = _query_understanding_tickers(query_understanding)
    companies = understood_companies or list(state_or_raw.get("companies", []) or [])
    comparison_target = state_or_raw.get("comparison_target") if isinstance(state_or_raw, Mapping) else None
    normalized_comparison_target = str(comparison_target).upper().strip() if comparison_target else None
    analysis_scope = str(
        _query_understanding_scope(query_understanding)
        or state_or_raw.get("analysis_scope")
        or dict(analysis_plan or {}).get("analysis_scope", "")
        or ""
    )
    time_policy = str(state_or_raw.get("time_policy") or dict(analysis_plan or {}).get("time_policy", "") or "")
    period_scope = str(state_or_raw.get("period_scope") or dict(analysis_plan or {}).get("period_scope", "") or "")
    raw_safety = str(state_or_raw.get("safety_intent", "") or "")
    safety_intent = _query_understanding_safety(query_understanding) or _normalize_safety_intent(raw_safety, user_query)
    understood_methodology = _query_understanding_methodology(
        query_understanding,
        companies=companies,
        comparison_target=normalized_comparison_target,
        safety_intent=safety_intent,
    )
    methodology_intent = str(
        understood_methodology
        or state_or_raw.get("methodology_intent")
        or dict(analysis_plan or {}).get("methodology_intent", "")
        or detect_methodology_intent(
            user_query,
            companies=companies,
            comparison_target=normalized_comparison_target,
            safety_intent=safety_intent,
        )
        or ""
    )
    answer_mode = _normalize_answer_mode(
        state_or_raw.get("answer_mode"),
        user_query=user_query,
        task_type=task_type,
        companies=companies,
        comparison_target=normalized_comparison_target,
        safety_intent=safety_intent,
        methodology_intent=methodology_intent,
    )
    needs_tools = bool(state_or_raw.get("needs_tools", _needs_tools_for_answer_mode(answer_mode, safety_intent)))
    supporting_context_dimensions = [
        str(item)
        for item in (
            state_or_raw.get("supporting_context_dimensions")
            or dict(analysis_plan or {}).get("supporting_context_dimensions", [])
            or (
                dict(analysis_plan or {}).get("optional_dimensions", [])
                if answer_mode == "risk_focused_analysis"
                else []
            )
        )
        if str(item)
    ]
    return QueryPlan(
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        methodology_intent=methodology_intent,
        analysis_scope=analysis_scope,
        supporting_context_dimensions=supporting_context_dimensions,
        time_policy=time_policy,
        period_scope=period_scope,
        needs_clarification=bool(state_or_raw.get("needs_clarification", _normalize_needs_clarification(answer_mode))),
        clarification_question=state_or_raw.get("clarification_question")
        or _build_clarification_question(str(state_or_raw.get("user_query", "")), answer_mode),
        needs_tools=needs_tools,
        period_query=raw.get("period_query", {}),
        resolved_period_context=raw.get("resolved_period_context", {}),
        comparison_basis_label=str(raw.get("comparison_basis_label", "same_period")),
        selected_tools=selected_tools,
        retrieval_policy=retrieval_policy or {},
        event_intent=str(state_or_raw.get("event_intent", "none")),
        market_reaction_requested=bool(state_or_raw.get("market_reaction_requested", False)),
        event_query=event_query or {},
        analysis_plan=analysis_plan or {},
        evidence_plan=evidence_plan or {},
    )


def build_classification_state(
    *,
    user_query: str,
    parsed: Mapping[str, Any],
    trace_id: str,
    today: date,
) -> dict[str, Any]:
    """Normalize classifier JSON into the AgentState update emitted by classify_and_extract."""
    output_language = detect_output_language(user_query)
    format_constraints = _format_constraints_from_query(str(user_query or ""))
    semantic_parser_mode = normalize_semantic_parser_mode()
    semantic_parser_result = build_semantic_query_proposal(user_query, mode=semantic_parser_mode)
    semantic_parser_trace = _semantic_parser_trace_payload(semantic_parser_result)
    semantic_parser_injected = False
    parsed_for_understanding = dict(parsed or {})
    if (
        semantic_parser_mode == "validated"
        and semantic_parser_trace.get("ok") is True
        and isinstance(semantic_parser_trace.get("proposal"), Mapping)
    ):
        existing_proposal = parsed_for_understanding.get("query_understanding_proposal")
        if isinstance(existing_proposal, Mapping):
            parsed_for_understanding["classifier_query_understanding_proposal"] = dict(existing_proposal)
        parsed_for_understanding["query_understanding_proposal"] = dict(semantic_parser_trace["proposal"])
        semantic_parser_injected = True
    parsed = parsed_for_understanding

    task_type = _normalize_task_type(str(parsed.get("task_type", "fact_qa")))
    companies = _normalize_company_values(parsed.get("companies", []))
    comparison_target = _normalize_ticker_value(parsed.get("comparison_target"))
    time_range = _sanitize_time_range_for_recency(
        user_query=user_query,
        time_range=parsed.get("time_range"),
        today=today,
    )
    period_query = _normalize_period_query(
        user_query=user_query,
        raw=parsed.get("period_query"),
        today=today,
        task_type=task_type,
    )
    requested_metrics = parsed.get("requested_metrics", [])
    requested_metrics = requested_metrics if isinstance(requested_metrics, list) else []
    data_route = parsed.get("data_route", "hybrid")
    data_route = data_route if data_route in {"documents_only", "structured_only", "hybrid"} else "hybrid"
    raw_analysis_plan = parsed.get("analysis_plan") if isinstance(parsed.get("analysis_plan"), dict) else {}
    query_understanding_model = build_query_understanding(user_query, parsed=parsed)
    query_understanding = query_understanding_summary(query_understanding_model)
    semantic_parser_trace["disagreement"] = _semantic_parser_disagreement(
        semantic_parser_trace,
        query_understanding,
        injected=semantic_parser_injected,
    )
    understood_companies = _query_understanding_tickers(query_understanding)
    understood_scope = _query_understanding_scope(query_understanding)
    understood_safety = _query_understanding_safety(query_understanding)
    understood_time_scope = _query_understanding_time_scope(query_understanding)
    understood_requested_dimensions = _query_understanding_requested_dimensions(query_understanding)
    understood_requested_metrics = _query_understanding_requested_metrics(query_understanding)
    understood_canonical_intent = str(query_understanding.get("methodology_intent") or "").strip()

    if understood_companies:
        companies = understood_companies
    if not companies:
        companies = _extract_tickers_fallback(user_query)
    if not companies and raw_analysis_plan:
        companies = _normalize_company_values(raw_analysis_plan.get("companies", []))

    if understood_scope == "comparison" and len(companies) >= 2:
        task_type = "company_comparison"
    elif understood_scope == "single_company" and understood_canonical_intent not in {"", "none"}:
        task_type = "report_summary"
    if len(companies) >= 2 and task_type == "fact_qa":
        logger.info("Upgrading task_type to company_comparison for multi-company query: %s", companies)
        task_type = "company_comparison"
    if len(companies) >= 2 and not comparison_target:
        comparison_target = companies[1]
    if comparison_target and comparison_target not in companies:
        companies.append(comparison_target)

    safety_intent = understood_safety or _normalize_safety_intent(parsed.get("safety_intent"), user_query)
    methodology_intent = _query_understanding_methodology(
        query_understanding,
        companies=companies,
        comparison_target=comparison_target,
        safety_intent=safety_intent,
    ) or detect_methodology_intent(
        user_query,
        companies=companies,
        comparison_target=comparison_target,
        safety_intent=safety_intent,
    )
    if _is_causal_revenue_growth_query(user_query) and len(companies) == 1 and not comparison_target:
        understood_scope = "single_company"
        methodology_intent = "revenue_quality_analysis"
        if "revenue_quality" not in understood_requested_dimensions:
            understood_requested_dimensions.append("revenue_quality")
    task_type = _apply_task_type_guardrails(
        user_query=user_query,
        task_type=task_type,
        companies=companies,
        comparison_target=comparison_target,
        methodology_intent=methodology_intent,
        analysis_scope=understood_scope,
    )
    analysis_scope = understood_scope or _analysis_scope_for_query(
        user_query=user_query,
        task_type=task_type,
        companies=companies,
        comparison_target=comparison_target,
        methodology_intent=methodology_intent,
    )
    time_policy = (
        str(understood_time_scope.get("policy") or "")
        if analysis_scope == "single_company" and not bool(period_query.get("is_explicit"))
        else ""
    )
    period_scope = (
        str(understood_time_scope.get("period_scope") or "")
        if analysis_scope == "single_company"
        else ""
    )
    preliminary_canonical_intent = build_canonical_intent(
        user_query=user_query,
        query_understanding=query_understanding,
        companies=companies,
        comparison_target=comparison_target,
        methodology_intent=methodology_intent,
        analysis_scope=analysis_scope,
        safety_intent=safety_intent,
        period_query=period_query,
        output_language=output_language,
    ).model_dump(exclude_none=True)
    if str(preliminary_canonical_intent.get("legacy_methodology_intent") or ""):
        methodology_intent = str(preliminary_canonical_intent.get("legacy_methodology_intent") or methodology_intent)
    if str(preliminary_canonical_intent.get("analysis_scope") or "") in {"single_company", "comparison", "unsupported"}:
        analysis_scope = str(preliminary_canonical_intent.get("analysis_scope") or analysis_scope)
    if analysis_scope == "single_company" and not bool(period_query.get("is_explicit")):
        period_query["period_type"] = "latest"
        period_query["trailing_n"] = None
        period_query["needs_clarification"] = False
        period_query["clarification_reason"] = None
        data_route = "hybrid"
    if understood_requested_metrics:
        requested_metrics = _ordered_unique(list(requested_metrics) + list(understood_requested_metrics))
    if _is_meta_query(user_query):
        answer_mode = "meta"
    elif _is_underspecified_analysis_query(user_query, companies):
        answer_mode = "clarification"
    else:
        answer_mode = str(preliminary_canonical_intent.get("answer_mode") or "").strip() or _normalize_answer_mode(
            parsed.get("answer_mode"),
            user_query=user_query,
            task_type=task_type,
            companies=companies,
            comparison_target=comparison_target,
            safety_intent=safety_intent,
            methodology_intent=methodology_intent,
        )
    risk_collapsed_composite = _is_risk_collapsed_composite_request(
        analysis_scope=analysis_scope,
        methodology_intent=methodology_intent,
        requested_dimensions=understood_requested_dimensions,
    )
    composite_dimension_request = analysis_scope == "single_company" and len(understood_requested_dimensions) > 1
    if (risk_collapsed_composite or composite_dimension_request) and answer_mode == "risk_focused_analysis":
        answer_mode = "analytical"
    if _is_causal_revenue_growth_query(user_query) and analysis_scope == "single_company":
        answer_mode = "analytical"
    if query_understanding_model.needs_clarification:
        answer_mode = "clarification"
    needs_clarification = _normalize_needs_clarification(answer_mode)
    clarification_question = _build_clarification_question(user_query, answer_mode)
    needs_tools = _needs_tools_for_answer_mode(answer_mode, safety_intent)
    if not companies and needs_tools:
        logger.info("classify_and_extract: no companies extracted; asking for clarification")
        answer_mode = "clarification"
        needs_clarification = True
        clarification_question = _build_clarification_question(user_query, answer_mode)
        needs_tools = False
        analysis_scope = ""
        time_policy = ""
        period_scope = ""
        methodology_intent = ""

    safety_model = apply_safety_policy(
        user_query=user_query,
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        needs_tools=needs_tools,
        needs_clarification=needs_clarification,
        companies=companies,
        comparison_target=comparison_target,
    )
    safety_decision = safety_model.model_dump(exclude_none=True)
    answer_mode = safety_model.answer_mode
    safety_intent = safety_model.safety_intent
    needs_tools = safety_model.needs_tools
    needs_clarification = _normalize_needs_clarification(answer_mode)
    clarification_question = _build_clarification_question(user_query, answer_mode)
    if answer_mode in {"clarification", "meta", "refusal_or_redirect"}:
        analysis_scope = ""
        time_policy = ""
        period_scope = ""
        if answer_mode == "refusal_or_redirect" and methodology_intent == "":
            methodology_intent = "unsupported_prediction" if detect_methodology_intent(user_query, companies=companies, comparison_target=comparison_target) == "unsupported_prediction" else ""

    event_intent = "none" if not needs_tools else _detect_event_intent(user_query=user_query, task_type=task_type)
    market_reaction_requested = bool(needs_tools and event_intent == "required")

    if task_type in ("trend_analysis", "company_comparison") and not requested_metrics and event_intent != "required":
        requested_metrics = ["revenue", "net_income"]
    if analysis_scope == "single_company" and not requested_metrics and event_intent != "required":
        requested_metrics = ["revenue", "net_income"]
    requested_metrics = _ordered_unique(list(requested_metrics) + _explicit_metric_terms_from_query(user_query))

    if market_reaction_requested:
        if data_route != "hybrid":
            data_route = "hybrid"
        if task_type == "fact_qa":
            task_type = "trend_analysis"
        metric_set = {str(m).strip() for m in requested_metrics if str(m).strip()}
        if "adjusted_close" not in metric_set and "close" not in metric_set:
            requested_metrics = list(requested_metrics) + ["adjusted_close"]

    canonical_intent_model = build_canonical_intent(
        user_query=user_query,
        query_understanding=query_understanding,
        companies=companies,
        comparison_target=comparison_target,
        methodology_intent=methodology_intent,
        analysis_scope=analysis_scope,
        safety_intent=safety_intent,
        period_query=period_query,
        answer_mode_override=answer_mode,
        output_language=output_language,
    )
    canonical_intent = canonical_intent_model.model_dump(exclude_none=True)
    intent_merge_decision = dict(canonical_intent.get("intent_merge_decision", {}) or {})
    evidence_policy_model = resolve_evidence_policy(canonical_intent)
    evidence_policy = evidence_policy_model.model_dump(exclude_none=True)
    evidence_policy_id = str(evidence_policy.get("policy_id") or "")

    selected_tools = (
        _select_tools(task_type, data_route, user_query=user_query, event_intent=event_intent)
        if needs_tools
        else []
    )
    analysis_plan_model = build_validated_analysis_plan(
        user_query=user_query,
        raw_plan=raw_analysis_plan,
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        methodology_intent=methodology_intent,
        analysis_scope=analysis_scope,
        time_policy=time_policy,
        period_scope=period_scope,
        companies=companies,
        comparison_target=comparison_target,
        time_range=time_range,
        period_query=period_query,
        requested_metrics=requested_metrics,
        deterministic_tools=selected_tools,
        needs_tools=needs_tools,
        event_intent=event_intent,
    )
    analysis_plan = analysis_plan_model.model_dump(exclude_none=True)
    analysis_plan["canonical_intent"] = canonical_intent
    analysis_plan["output_language"] = output_language
    analysis_plan["evidence_policy"] = evidence_policy
    analysis_plan["evidence_policy_id"] = evidence_policy_id
    if str(canonical_intent.get("segment_focus") or "").strip():
        analysis_plan["segment_focus"] = str(canonical_intent.get("segment_focus") or "").strip()
    if str(canonical_intent.get("segment_or_product_scope") or "").strip():
        analysis_plan["segment_or_product_scope"] = str(canonical_intent.get("segment_or_product_scope") or "").strip()
    policy_driven_mode = answer_mode not in {"direct_fact", "meta", "clarification", "refusal_or_redirect"}
    if policy_driven_mode:
        analysis_plan["primary_dimension"] = str(evidence_policy.get("primary_dimension") or "")
        analysis_plan["required_dimensions"] = list(evidence_policy.get("required_dimensions", []) or [])
        analysis_plan["optional_dimensions"] = list(evidence_policy.get("optional_dimensions", []) or [])
        analysis_plan["requested_dimensions"] = list(canonical_intent.get("requested_dimensions", []) or [])
        analysis_plan["supporting_context_dimensions"] = [
            dimension
            for dimension in list(evidence_policy.get("optional_dimensions", []) or [])
            if str(dimension).strip()
        ]
    if composite_dimension_request and answer_mode == "analytical":
        analysis_plan["primary_dimension"] = ""
        analysis_plan["required_dimensions"] = list(understood_requested_dimensions)
        analysis_plan["optional_dimensions"] = []
        analysis_plan["requested_dimensions"] = list(understood_requested_dimensions)
        analysis_plan["supporting_context_dimensions"] = []
    rejected_plan_items = list(analysis_plan.get("rejected_plan_items", []))
    validated_tools = list(analysis_plan.get("validated_tools", []))
    primary_dimension = str(analysis_plan.get("primary_dimension", "") or "")
    methodology_intent = str(analysis_plan.get("methodology_intent", methodology_intent) or "")
    required_dimensions = [str(item) for item in analysis_plan.get("required_dimensions", []) or [] if str(item)]
    optional_dimensions = [str(item) for item in analysis_plan.get("optional_dimensions", []) or [] if str(item)]
    supporting_context_dimensions = _ordered_unique(
        [
            str(item)
            for item in (
                list(analysis_plan.get("supporting_context_dimensions", []) or [])
                + (optional_dimensions if answer_mode == "risk_focused_analysis" else [])
                + [dimension for dimension in understood_requested_dimensions if dimension not in required_dimensions]
            )
            if str(item)
        ]
    )
    if understood_requested_dimensions:
        analysis_plan["requested_dimensions"] = understood_requested_dimensions
        analysis_plan["supporting_context_dimensions"] = supporting_context_dimensions
    if analysis_scope == "single_company" and needs_tools and "compute_metrics" not in validated_tools:
        validated_tools.append("compute_metrics")
        analysis_plan["validated_tools"] = validated_tools
    if needs_tools:
        selected_tools = validated_tools
        metric_requirements = list(analysis_plan.get("metric_requirements", []))
        if metric_requirements:
            requested_metrics = metric_requirements
        if any(tool in selected_tools for tool in ("search_filings", "query_financial_data")):
            data_route = "hybrid" if len(set(selected_tools) & {"search_filings", "query_financial_data"}) == 2 else data_route
    selected_analysis_framework = serialize_selected_analysis_framework(
        select_analysis_framework(
            {
                "user_query": user_query,
                "query_understanding": query_understanding,
                "analysis_plan": analysis_plan,
                "task_type": task_type,
                "answer_mode": answer_mode,
                "safety_intent": safety_intent,
                "methodology_intent": methodology_intent,
                "analysis_scope": analysis_scope,
            }
        )
    )
    evidence_plan_model = build_evidence_plan(
        {
            "user_query": user_query,
            "query_understanding": query_understanding,
            "task_type": task_type,
            "answer_mode": answer_mode,
            "safety_intent": safety_intent,
            "methodology_intent": methodology_intent,
            "analysis_scope": analysis_scope,
            "time_policy": time_policy,
            "period_scope": period_scope,
            "needs_tools": needs_tools,
            "companies": companies,
            "comparison_target": comparison_target,
            "requested_metrics": requested_metrics,
            "period_query": period_query,
            "resolved_period_context": {},
            "analysis_plan": analysis_plan,
            "selected_analysis_framework": selected_analysis_framework,
            "event_intent": event_intent,
            "canonical_intent": canonical_intent,
            "output_language": output_language,
            "evidence_policy": evidence_policy,
            "evidence_policy_id": evidence_policy_id,
        }
    )
    evidence_plan = evidence_plan_model.model_dump(exclude_none=True)
    rejected_requirements = list(evidence_plan.get("rejected_requirements", []))
    event_query = _build_event_query(user_query=user_query, task_type=task_type, period_query=period_query)
    retrieval_policy = _build_retrieval_policy(
        {
            "task_type": task_type,
            "answer_mode": answer_mode,
            "safety_intent": safety_intent,
            "user_query": user_query,
            "data_route": data_route,
            "event_intent": event_intent,
            "needs_tools": needs_tools,
            "analysis_plan": analysis_plan,
        },
        selected_tools=selected_tools,
    )
    plan = _resolve_query_plan(
        {
            "user_query": user_query,
            "task_type": task_type,
            "period_query": period_query,
        }
    )
    period_query = dict(plan.get("period_query", period_query))
    resolved_period_context = dict(plan.get("resolved_period_context", {}))
    comparison_basis_label = str(plan.get("comparison_basis_label", ""))

    return {
        "user_query": user_query,
        "output_language": output_language,
        "task_type": task_type,
        "answer_mode": answer_mode,
        "safety_intent": safety_intent,
        "query_understanding": query_understanding,
        "query_understanding_summary": {**query_understanding, "output_language": output_language},
        "canonical_intent": canonical_intent,
        "segment_focus": str(canonical_intent.get("segment_focus") or ""),
        "segment_or_product_scope": str(canonical_intent.get("segment_or_product_scope") or ""),
        "intent_merge_decision": intent_merge_decision,
        "semantic_parser_mode": semantic_parser_mode,
        "semantic_parser": semantic_parser_trace,
        "semantic_proposal": dict(query_understanding.get("semantic_proposal", {}) or {}),
        "rule_methodology_intent": str(query_understanding.get("rule_methodology_intent", "") or ""),
        "proposed_methodology_intent": str(query_understanding.get("proposed_methodology_intent", "") or ""),
        "proposal_validation_warnings": list(query_understanding.get("proposal_validation_warnings", []) or []),
        "intent_conflict": bool(query_understanding.get("intent_conflict", False)),
        "methodology_intent": methodology_intent,
        "analysis_scope": analysis_scope,
        "primary_dimension": primary_dimension,
        "required_dimensions": required_dimensions,
        "optional_dimensions": optional_dimensions,
        "supporting_context_dimensions": supporting_context_dimensions,
        "format_constraints": format_constraints,
        "time_policy": time_policy,
        "period_scope": period_scope,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "needs_tools": needs_tools,
        "companies": companies,
        "comparison_target": comparison_target,
        "time_range": time_range,
        "period_query": period_query,
        "resolved_period_context": resolved_period_context,
        "comparison_basis_label": comparison_basis_label,
        "requested_metrics": requested_metrics,
        "data_route": data_route,
        "analysis_plan_raw": raw_analysis_plan,
        "analysis_plan": analysis_plan,
        "selected_analysis_framework": selected_analysis_framework,
        "research_plan_raw": {},
        "research_plan_validated": {},
        "research_plan_used": {},
        "research_plan_validation": {},
        "required_answer_parts": [],
        "legacy_evidence_plan": {},
        "evidence_policy": evidence_policy,
        "evidence_policy_id": evidence_policy_id,
        "rejected_plan_items": rejected_plan_items,
        "validated_tools": validated_tools,
        "safety_decision": safety_decision,
        "safety_policy_reasons": list(safety_decision.get("policy_reasons", [])),
        "safety_limitations": list(safety_decision.get("limitations", [])),
        "evidence_plan": evidence_plan,
        "rejected_requirements": rejected_requirements,
        "selected_tools": selected_tools,
        "retrieval_policy": retrieval_policy,
        "retrieval_debug": {},
        "trace_summary": {
            **analysis_framework_trace_fields(selected_analysis_framework),
            "semantic_parser_mode": semantic_parser_mode,
            "semantic_parser": semantic_parser_trace,
            "query_understanding_summary": {**query_understanding, "output_language": output_language},
            "canonical_intent": canonical_intent,
            "output_language": output_language,
            "intent_merge_decision": intent_merge_decision,
            "evidence_policy_id": evidence_policy_id,
            "analysis_scope": analysis_scope,
            "methodology_intent": methodology_intent,
            "primary_dimension": primary_dimension,
            "required_dimensions": required_dimensions,
            "optional_dimensions": optional_dimensions,
            "supporting_context_dimensions": supporting_context_dimensions,
            "format_constraints": format_constraints,
            "time_policy": time_policy,
            "period_scope": period_scope,
        },
        "event_intent": event_intent,
        "market_reaction_requested": market_reaction_requested,
        "event_query": event_query,
        "event_results": [],
        "market_reaction_evidence": [],
        "market_reaction_limitations": [],
        "trace_id": trace_id,
        "tool_results": [],
        "retrieved_docs": [],
        "evidence_collection_results": [],
        "evidence_sufficiency": {},
        "answer_part_status_by_id": {},
        "evidence_gap_by_answer_part": {},
        "missing_required_answer_parts": [],
        "numeric_evidence": [],
        "text_evidence": [],
        "unsupported_claims": [],
        "numeric_citations": [],
        "text_citations": [],
        "citations": [],
        "output": {},
        "structured_sources": [],
        "document_citations": [],
        "synthesis": {},
        "synthesis_strategy": "",
        "unsupported_synthesis_items": [],
        "answer_history": [],
        "answer_candidate": {},
        "answer_candidates": [],
        "why_tools_skipped": (
            [
                {"reason": f"answer_mode:{answer_mode}", "message": "tools_skipped_for_conversational_response"},
                *[
                    {
                        "reason": str(item.get("code", "safety_policy")),
                        "message": str(item.get("message", "")),
                    }
                    for item in safety_decision.get("policy_reasons", [])
                ],
            ]
            if not needs_tools
            else []
        ),
        "evidence_sufficient": False,
        "evidence_loop_count": 0,
        "relevance_decision": {},
        "relevance_status": "not_run",
        "relevance_attempts": 0,
        "relevance_repair_attempts": 0,
        "final_route": "",
        "answer_quality_tier": "",
        "main_question_covered": True,
        "fallback_intent_match": True,
        "answered_dimensions": [],
        "unresolved_relevance_failures": [],
        "format_constraints_satisfied": True,
        "partial_required_answer_parts": [],
        "research_plan_source": "",
        "research_plan_fallback_reason": "",
        "research_plan_duration_ms": 0,
    }


extract_tickers_fallback = _extract_tickers_fallback
detect_event_intent = _detect_event_intent
select_tools = _select_tools
infer_period_type = _infer_period_type
normalize_safety_intent = _normalize_safety_intent
normalize_answer_mode = _normalize_answer_mode
needs_tools_for_answer_mode = _needs_tools_for_answer_mode
build_clarification_question = _build_clarification_question
normalize_ticker_value = _normalize_ticker_value
