"""Shared constants for the financial-analysis agent."""

from __future__ import annotations

MAX_EVIDENCE_LOOPS = 2

NUMERIC_REQUIRED_TASK_TYPES = {"fact_qa", "trend_analysis", "company_comparison"}

ESTIMATION_TERMS = (
    "estimated",
    "estimate",
    "infer",
    "inferred",
    "approximate",
    "approximately",
    "推测",
    "估算",
    "大概",
    "约等于",
)

PERIOD_TYPES = {"quarterly", "annual", "latest", "trailing"}

YEAR_BASIS = {"fiscal", "calendar"}

COMPARISON_BASIS = {"same_period", "latest_fiscal_year", "calendar_year"}

TEXT_CITATION_CAPS = {
    "fact_qa": 2,
    "trend_analysis": 3,
    "company_comparison": 4,  # actual per-company cap handled separately
    "report_summary": 5,
}

OUTPUT_PROTOCOL_VERSION = "phase4.v1"

ANSWER_MODES = {
    "direct_fact",
    "analytical",
    "risk_focused_analysis",
    "cautious_outlook",
    "comparison_brief",
    "clarification",
    "meta",
    "refusal_or_redirect",
}

SAFETY_INTENTS = {
    "normal",
    "investment_advice_like",
    "unsupported_or_out_of_scope",
}

DEFAULT_ANSWER_MODE_BY_TASK = {
    "fact_qa": "direct_fact",
    "trend_analysis": "analytical",
    "company_comparison": "comparison_brief",
    "report_summary": "analytical",
}

ALLOWED_ANALYSIS_TOOLS = {
    "search_filings",
    "query_financial_data",
    "compute_metrics",
    "query_event_price_window",
}

ALLOWED_ANALYSIS_METRICS = {
    "revenue",
    "revenue_growth",
    "net_income",
    "eps",
    "gross_profit",
    "operating_income",
    "consolidated_operating_income",
    "aws_operating_income",
    "aws_revenue",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "operating_cash_flow",
    "free_cash_flow",
    "capital_expenditure",
    "cash_and_equivalents",
    "short_term_debt",
    "long_term_debt",
    "total_debt",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "net_debt",
    "inventory",
    "receivables",
    "shares_outstanding",
    "cfo_to_net_income",
    "cash_conversion",
    "fcf_margin",
    "debt_to_equity",
    "capex_to_revenue",
    "receivables_to_revenue",
    "inventory_to_revenue",
    "market_cap",
    "pe_ratio",
    "ps_ratio",
    "fcf_yield",
    "segment_profit_contribution",
    "price",
    "open",
    "high",
    "low",
    "adjusted_close",
    "close",
    "volume",
}

KNOWN_SEC_SECTIONS = {
    "ITEM_1",
    "ITEM_1A",
    "ITEM_2",
    "ITEM_3",
    "ITEM_7",
    "ITEM_7A",
    "ITEM_8",
    "BUSINESS",
    "MD&A",
}

ANALYTICAL_SECTION_PREFERENCES = ["ITEM_7", "ITEM_1A", "ITEM_1", "ITEM_2"]

OUTPUT_EVIDENCE_CAPS = {
    "fact_qa": {"numeric": 2, "text": 1},
    "trend_analysis": {"numeric": 6, "text": 3},
    "company_comparison": {"numeric_per_company": 3, "text_per_company": 2},
    "report_summary": {"numeric": 4, "text": 5},
}

UNKNOWN_PERIOD = "UNKNOWN_PERIOD"

EVENT_WINDOW_DAYS = (1, 3, 5, 10)

EVENT_INTENT_TYPES = {"none", "optional", "required"}

MARKET_REACTION_TERMS = (
    "股价",
    "股价变化",
    "股价表现",
    "股价波动",
    "涨跌",
    "上涨",
    "下跌",
    "反应",
    "收益率",
    "回报",
    "市场表现",
    "盘后反应",
    "市场怎么反应",
    "跑赢",
    "跑输",
    "after-hours reaction",
    "after hours reaction",
    "market reaction",
    "market move",
    "market response",
    "market performance",
    "stock reaction",
    "price reaction",
    "price movement",
    "price change",
    "price volatility",
    "stock move",
    "stock return",
    "outperform",
    "underperform",
)

EVENT_ANCHOR_TERMS = (
    "财报后",
    "发布后",
    "发布财报的时候",
    "公布财报的时候",
    "发财报的时候",
    "披露财报的时候",
    "财报发布时",
    "季度财报发布时",
    "每次财报",
    "每个季度财报",
    "季报发布时",
    "年报发布时",
    "季度业绩发布时",
    "年报披露时",
    "10-q",
    "10-k",
    "10q",
    "10k",
    "季度财报",
    "季报",
    "年报",
    "财报发布",
    "财报披露",
    "季度业绩后",
    "earnings",
    "post-earnings",
    "post earnings",
    "after earnings",
    "after filing",
)

COMPARISON_TERMS = ("比较", "对比", "vs", "versus", "compare")

TREND_TERMS = (
    "趋势",
    "变化",
    "走势",
    "相比",
    "同比",
    "环比",
    "增长",
    "过去",
    "最近几个季度",
    "how much change",
    "trend",
)

SUMMARY_TERMS = (
    "总结",
    "概括",
    "综合",
    "业务概况",
    "风险因素",
    "主要风险",
    "主要问题",
    "最大的问题",
    "最大风险",
    "核心业务",
    "竞争优势",
    "法律风险",
    "经营表现",
    "biggest problem",
    "main issue",
    "biggest risk",
    "summary",
)
