"""ToolSpec adapter for query_financial_data."""

from __future__ import annotations

from typing import Any, Mapping

from src.tools.protocol import BaseToolAdapter, ProvenancePolicy, ToolExecutionContext, ToolProvenance, ToolResult, ToolSpec
from src.tools.query_financial_data import query_financial_data


class QueryFinancialDataTool(BaseToolAdapter):
    spec = ToolSpec(
        name="query_financial_data",
        description="Look up structured financial facts and historical price rows from local DuckDB.",
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "period_type": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["ticker", "metrics"],
        },
        output_schema={"type": "object"},
        timeout_seconds=30.0,
        deterministic=True,
        requires_live_services=True,
        tags=["finance", "structured", "retrieval"],
        provenance_policy=ProvenancePolicy(require_period=True, require_provider=True),
    )

    def __init__(self, tool: Any | None = None):
        super().__init__(tool or query_financial_data)

    def execute(self, input_data: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        data = self.invoke_legacy(input_data)
        data = dict(data or {}) if isinstance(data, Mapping) else {}
        facts = [row for row in data.get("financial_facts", []) or [] if isinstance(row, Mapping)]
        prices = [row for row in data.get("price_data", []) or [] if isinstance(row, Mapping)]
        warnings: list[str] = []
        evidence_ids: list[str] = []
        providers: set[str] = set()
        for row in facts + prices:
            if row.get("reconciliation_warning"):
                warnings.append(str(row.get("reconciliation_warning")))
            if row.get("evidence_id"):
                evidence_ids.append(str(row.get("evidence_id")))
            provider = str(row.get("source_provider") or "")
            if provider:
                providers.add(provider)
        provenance = [
            ToolProvenance(
                provider=",".join(sorted(providers)) or "structured",
                source_table="financial_facts/price_data",
                query={
                    "ticker": input_data.get("ticker"),
                    "metrics": input_data.get("metrics", []),
                    "period_type": input_data.get("period_type"),
                },
                confidence="mixed" if len(providers) > 1 else None,
                reconciliation_warnings=list(dict.fromkeys(warnings)),
                evidence_ids=list(dict.fromkeys(evidence_ids)),
            )
        ]
        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            data=data,
            warnings=list(dict.fromkeys(warnings)),
            provenance=provenance,
            raw_count=len(facts) + len(prices),
            returned_count=len(facts) + len(prices),
        )
