"""Static registry for internal evidence tools."""

from __future__ import annotations

from typing import Any, Mapping

from src.tools.adapters import ComputeMetricsTool, QueryEventPriceWindowTool, QueryFinancialDataTool, SearchFilingsTool
from src.tools.protocol import BaseToolAdapter, ToolExecutionContext, ToolResult, ToolSpec, execute_tool_with_timeout


class UnknownToolError(KeyError):
    """Raised when a tool name is not present in the registry."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseToolAdapter] = {}

    def register(self, tool: BaseToolAdapter) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> BaseToolAdapter:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(name) from exc

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def to_function_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            }
            for spec in self.list_specs()
        ]

    def execute(self, name: str, input_data: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        return execute_tool_with_timeout(self.get(name), input_data, context)


def build_default_tool_registry(overrides: Mapping[str, Any] | None = None) -> ToolRegistry:
    """Build the static internal registry.

    overrides lets requirement_executor pass its module-level callables so
    existing monkeypatch-based tests continue to patch the same symbols.
    """
    overrides = dict(overrides or {})
    registry = ToolRegistry()
    registry.register(SearchFilingsTool(overrides.get("search_filings")))
    registry.register(QueryFinancialDataTool(overrides.get("query_financial_data")))
    registry.register(ComputeMetricsTool(overrides.get("compute_metrics")))
    registry.register(QueryEventPriceWindowTool(overrides.get("query_event_price_window")))
    return registry
