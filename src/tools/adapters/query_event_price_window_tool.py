"""ToolSpec adapter for query_event_price_window."""

from __future__ import annotations

from typing import Any, Mapping

from src.tools.protocol import BaseToolAdapter, ProvenancePolicy, ToolExecutionContext, ToolProvenance, ToolResult, ToolSpec
from src.tools.query_event_price_window import query_event_price_window


class QueryEventPriceWindowTool(BaseToolAdapter):
    spec = ToolSpec(
        name="query_event_price_window",
        description="Query precomputed filing-event price reaction windows from local DuckDB.",
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "event_type": {"type": "string"},
                "fiscal_period": {"type": "string"},
                "event_date": {"type": "string"},
                "latest_n": {"type": "integer"},
                "window_days": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["ticker"],
        },
        output_schema={"type": "object"},
        timeout_seconds=20.0,
        deterministic=True,
        requires_live_services=True,
        tags=["finance", "event", "market_reaction"],
        provenance_policy=ProvenancePolicy(required_fields=["filing_id", "window"], require_period=True),
    )

    def __init__(self, tool: Any | None = None):
        super().__init__(tool or query_event_price_window)

    def execute(self, input_data: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        data = self.invoke_legacy(input_data)
        data = dict(data or {}) if isinstance(data, Mapping) else {}
        events = [event for event in data.get("events", []) or [] if isinstance(event, Mapping)]
        evidence_ids = [
            str(event.get("evidence_id") or "")
            for event in events
            if str(event.get("evidence_id") or "")
        ]
        filing_ids = [
            str(event.get("filing_id") or "")
            for event in events
            if str(event.get("filing_id") or "")
        ]
        provenance = [
            ToolProvenance(
                provider="event_price_window",
                source_table="filing_events/event_price_windows",
                query={
                    "ticker": input_data.get("ticker"),
                    "event_type": input_data.get("event_type"),
                    "window_days": input_data.get("window_days"),
                },
                source_filing_id=",".join(list(dict.fromkeys(filing_ids))[:5]) or None,
                confidence="medium",
                evidence_ids=list(dict.fromkeys(evidence_ids)),
            )
        ]
        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            data=data,
            provenance=provenance,
            raw_count=len(events),
            returned_count=len(events),
        )
