"""FilingLens tool registry."""

from src.tools.compute_metrics import compute_metrics
from src.tools.query_event_price_window import query_event_price_window
from src.tools.query_financial_data import query_financial_data
from src.tools.registry import ToolRegistry, build_default_tool_registry
from src.tools.search_filings import search_filings

ALL_TOOLS = [search_filings, query_financial_data, query_event_price_window, compute_metrics]

__all__ = [
    "search_filings",
    "query_financial_data",
    "query_event_price_window",
    "compute_metrics",
    "ToolRegistry",
    "build_default_tool_registry",
    "ALL_TOOLS",
]
