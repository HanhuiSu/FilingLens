"""Internal tool protocol for evidence-producing financial tools."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Literal, Mapping, Protocol

from pydantic import BaseModel, Field, model_validator


class InvokableTool(Protocol):
    def invoke(self, input_data: dict[str, Any]) -> Any:
        ...


class ProvenancePolicy(BaseModel):
    required_fields: list[str] = Field(default_factory=list)
    allow_missing_source_url: bool = True
    allow_medium_confidence: bool = True
    require_period: bool = False
    require_provider: bool = False
    require_evidence_ids: bool = False


class ToolSpec(BaseModel):
    name: str
    description: str
    version: str = "1.0.0"
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30.0
    deterministic: bool = True
    requires_live_services: bool = False
    tags: list[str] = Field(default_factory=list)
    provenance_policy: ProvenancePolicy = Field(default_factory=ProvenancePolicy)


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolProvenance(BaseModel):
    provider: str | None = None
    source_url: str | None = None
    source_filing_id: str | None = None
    source_table: str | None = None
    query: dict[str, Any] = Field(default_factory=dict)
    as_of: str | None = None
    confidence: str | None = None
    reconciliation_warnings: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    data: Any = None
    error: ToolError | None = None
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ToolProvenance] = Field(default_factory=list)
    latency_ms: int | None = None
    raw_count: int | None = None
    returned_count: int | None = None

    @model_validator(mode="after")
    def _validate_error_shape(self) -> "ToolResult":
        if self.ok and self.error is not None:
            raise ValueError("ok=True ToolResult cannot include error")
        if not self.ok and self.error is None:
            raise ValueError("ok=False ToolResult must include error")
        return self

    def trace_summary(self, *, tool_version: str = "", input_summary: dict[str, Any] | None = None, requirement_id: str = "") -> dict[str, Any]:
        error = self.error.model_dump(exclude_none=True) if self.error else None
        return {
            "tool_name": self.tool_name,
            "tool_version": tool_version,
            "requirement_id": requirement_id,
            "input_summary": dict(input_summary or {}),
            "ok": self.ok,
            "latency_ms": self.latency_ms,
            "raw_count": self.raw_count,
            "returned_count": self.returned_count,
            "warnings": list(self.warnings),
            "provenance": [item.model_dump(exclude_none=True) for item in self.provenance],
            "error": error,
        }


class ToolExecutionContext(BaseModel):
    trace_id: str = ""
    query_id: str | None = None
    requirement_id: str | None = None
    company: str | None = None
    dimension: str | None = None
    deadline_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseToolAdapter(ABC):
    spec: ToolSpec

    def __init__(self, tool: InvokableTool | None = None):
        self.tool = tool

    def validate_input(self, input_data: dict[str, Any]) -> None:
        schema = self.spec.input_schema
        required = schema.get("required", []) if isinstance(schema, Mapping) else []
        for key in required:
            if key not in input_data or input_data.get(key) is None:
                raise ValueError(f"Missing required tool input: {key}")

    def validate_output(self, result: ToolResult) -> None:
        if result.ok:
            policy = self.spec.provenance_policy
            if policy.require_evidence_ids and not any(item.evidence_ids for item in result.provenance):
                raise ValueError(f"{self.spec.name} result is missing provenance evidence ids")

    def invoke_legacy(self, input_data: dict[str, Any]) -> Any:
        if self.tool is None:
            raise RuntimeError(f"{self.spec.name} has no legacy tool callable")
        return self.tool.invoke(input_data)

    @abstractmethod
    def execute(self, input_data: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        ...


def execute_tool_with_timeout(
    tool: BaseToolAdapter,
    input_data: dict[str, Any],
    context: ToolExecutionContext,
) -> ToolResult:
    started = time.monotonic()
    try:
        tool.validate_input(input_data)
    except Exception as exc:
        return ToolResult(
            ok=False,
            tool_name=tool.spec.name,
            error=ToolError(code="invalid_input", message=str(exc), retryable=False),
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(tool.execute, input_data, context)
    try:
        result = future.result(timeout=tool.spec.timeout_seconds)
        result.latency_ms = int((time.monotonic() - started) * 1000)
        tool.validate_output(result)
        return result
    except FutureTimeoutError:
        future.cancel()
        return ToolResult(
            ok=False,
            tool_name=tool.spec.name,
            error=ToolError(code="timeout", message="Tool execution timed out", retryable=True),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    except Exception as exc:
        return ToolResult(
            ok=False,
            tool_name=tool.spec.name,
            error=ToolError(code="execution_error", message=str(exc), retryable=False),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def tool_result_route(result: ToolResult) -> Literal["ok", "retryable_error", "error"]:
    if result.ok:
        return "ok"
    if result.error and result.error.retryable:
        return "retryable_error"
    return "error"
