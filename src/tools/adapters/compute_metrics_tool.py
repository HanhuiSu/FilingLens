"""ToolSpec adapter for compute_metrics."""

from __future__ import annotations

from typing import Any, Mapping

from src.tools.compute_metrics import compute_metrics
from src.tools.protocol import BaseToolAdapter, ProvenancePolicy, ToolExecutionContext, ToolProvenance, ToolResult, ToolSpec


class ComputeMetricsTool(BaseToolAdapter):
    spec = ToolSpec(
        name="compute_metrics",
        description="Compute deterministic ratios, growth rates, margins, and valuation multiples from provided inputs.",
        input_schema={
            "type": "object",
            "properties": {
                "data": {"type": "array"},
                "denominator_data": {"type": "array"},
                "computation": {"type": "string"},
            },
            "required": ["data", "computation"],
        },
        output_schema={"type": "object"},
        timeout_seconds=10.0,
        deterministic=True,
        requires_live_services=False,
        tags=["finance", "computation"],
        provenance_policy=ProvenancePolicy(require_evidence_ids=False),
    )

    def __init__(self, tool: Any | None = None):
        super().__init__(tool or compute_metrics)

    def execute(self, input_data: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        data = self.invoke_legacy(input_data)
        data = dict(data or {}) if isinstance(data, Mapping) else {}
        rows = [row for row in input_data.get("data", []) or [] if isinstance(row, Mapping)]
        denom = [row for row in input_data.get("denominator_data", []) or [] if isinstance(row, Mapping)]
        evidence_ids = [
            str(row.get("evidence_id") or "")
            for row in rows + denom
            if str(row.get("evidence_id") or "")
        ]
        provenance = [
            ToolProvenance(
                provider="computed",
                source_table="deterministic_compute_metrics",
                query={"computation": input_data.get("computation")},
                evidence_ids=list(dict.fromkeys(evidence_ids)),
            )
        ]
        return ToolResult(
            ok=True,
            tool_name=self.spec.name,
            data=data,
            provenance=provenance,
            raw_count=len(rows) + len(denom),
            returned_count=len(data.get("results", []) or []),
        )
