"""ToolSpec adapter for search_filings."""

from __future__ import annotations

from typing import Any, Mapping

from src.tools.protocol import BaseToolAdapter, ProvenancePolicy, ToolExecutionContext, ToolProvenance, ToolResult, ToolSpec
from src.tools.search_filings import search_filings


class SearchFilingsTool(BaseToolAdapter):
    spec = ToolSpec(
        name="search_filings",
        description="Retrieve SEC filing text chunks from the local Chroma collection.",
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "section_allowlist": {"type": "array", "items": {"type": "string"}},
                "strict_sections": {"type": "boolean"},
            },
            "required": ["ticker", "query"],
        },
        output_schema={"type": "object"},
        timeout_seconds=45.0,
        deterministic=True,
        requires_live_services=True,
        tags=["filings", "text", "retrieval"],
        provenance_policy=ProvenancePolicy(required_fields=["filing_id", "ticker", "section"]),
    )

    def __init__(self, tool: Any | None = None):
        super().__init__(tool or search_filings)

    def execute(self, input_data: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        raw = self.invoke_legacy(input_data)
        if isinstance(raw, Mapping):
            data = dict(raw)
            items = [item for item in data.get("items", []) or [] if isinstance(item, Mapping)]
            diagnostics = dict(data.get("diagnostics", {}) or {})
        else:
            items = [item for item in raw or [] if isinstance(item, Mapping)] if isinstance(raw, list) else []
            diagnostics = {}
            data = {"items": items}
        evidence_ids = [
            str(item.get("evidence_id") or item.get("citation_ref") or "")
            for item in items
            if str(item.get("evidence_id") or item.get("citation_ref") or "")
        ]
        filing_ids = [
            str(item.get("filing_id") or "")
            for item in items
            if str(item.get("filing_id") or "")
        ]
        provenance = [
            ToolProvenance(
                provider="chroma",
                source_table="filing_chunks",
                query={
                    "ticker": input_data.get("ticker"),
                    "query": input_data.get("query"),
                    "section_allowlist": input_data.get("section_allowlist"),
                    "strict_sections": input_data.get("strict_sections"),
                },
                source_filing_id=",".join(list(dict.fromkeys(filing_ids))[:5]) or None,
                evidence_ids=list(dict.fromkeys(evidence_ids)),
            )
        ]
        raw_count = int(diagnostics.get("raw_hit_count", len(items)) or 0)
        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            data=data,
            provenance=provenance,
            raw_count=raw_count,
            returned_count=len(items),
        )
