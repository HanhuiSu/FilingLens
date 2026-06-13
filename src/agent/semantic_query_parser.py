"""LLM-assisted semantic query parsing behind deterministic validation."""

from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ConfigDict, ValidationError, field_validator

from config import settings
from src.agent.llm import _get_llm, _parse_json_response
from src.agent.query_ontology import (
    ALLOWED_ANALYSIS_METRICS,
    ALLOWED_ANALYSIS_SCOPES,
    ALLOWED_METHODOLOGY_INTENTS,
    ALLOWED_SAFETY_INTENTS,
    ALLOWED_USER_EXPECTATIONS,
    SUPPORTED_DIMENSIONS,
    normalize_dimension_label,
    allowed_parser_labels,
    normalize_metric_label,
    normalize_safety_intent_label,
)
from src.agent.types import AgentDomainModel


class SemanticQueryProposal(AgentDomainModel):
    """Strict LLM proposal contract; deterministic code still decides final behavior."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    company_mentions: list[str]
    analysis_scope: str
    methodology_intent: str
    requested_dimensions: list[str]
    requested_metrics: list[str]
    user_expectation: str
    safety_intent: str
    time_scope: dict[str, Any]
    confidence: float
    ambiguity: bool
    needs_clarification: bool
    reasons: list[str]

    @field_validator("analysis_scope")
    @classmethod
    def _allowed_scope(cls, value: str) -> str:
        if value not in ALLOWED_ANALYSIS_SCOPES:
            raise ValueError("analysis_scope_not_allowed")
        return value

    @field_validator("methodology_intent")
    @classmethod
    def _allowed_methodology_intent(cls, value: str) -> str:
        if value not in ALLOWED_METHODOLOGY_INTENTS:
            raise ValueError("methodology_intent_not_allowed")
        return value

    @field_validator("requested_dimensions")
    @classmethod
    def _allowed_dimensions(cls, value: list[str]) -> list[str]:
        dimensions: list[str] = []
        for item in value or []:
            dimension = normalize_dimension_label(item)
            if dimension not in SUPPORTED_DIMENSIONS:
                raise ValueError("dimension_not_supported")
            if dimension not in dimensions:
                dimensions.append(dimension)
        return dimensions

    @field_validator("requested_metrics", mode="before")
    @classmethod
    def _normalize_metrics(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [normalize_metric_label(item) for item in value]

    @field_validator("requested_metrics")
    @classmethod
    def _allowed_metrics(cls, value: list[str]) -> list[str]:
        metrics: list[str] = []
        for item in value or []:
            metric = normalize_metric_label(item)
            if metric not in ALLOWED_ANALYSIS_METRICS:
                raise ValueError("metric_not_supported")
            if metric not in metrics:
                metrics.append(metric)
        return metrics

    @field_validator("user_expectation")
    @classmethod
    def _allowed_user_expectation(cls, value: str) -> str:
        if value not in ALLOWED_USER_EXPECTATIONS:
            raise ValueError("user_expectation_not_allowed")
        return value

    @field_validator("safety_intent", mode="before")
    @classmethod
    def _normalize_safety(cls, value: Any) -> str:
        return normalize_safety_intent_label(value)

    @field_validator("safety_intent")
    @classmethod
    def _allowed_safety_intent(cls, value: str) -> str:
        if value not in ALLOWED_SAFETY_INTENTS:
            raise ValueError("safety_intent_not_allowed")
        return value

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        try:
            raw = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(max(0.0, min(raw, 1.0)), 3)


class SemanticQueryParserResult(AgentDomainModel):
    mode: str = "off"
    ok: bool = False
    source: str = "not_run"
    proposal: dict[str, Any] = {}
    error: str = ""
    warnings: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}


def normalize_semantic_parser_mode(value: Any = None) -> str:
    raw = str(settings.semantic_query_parser_mode if value is None else value).strip().lower()
    return raw if raw in {"off", "shadow", "validated"} else "off"


def _coerce_payload(raw: Any) -> Mapping[str, Any] | None:
    if isinstance(raw, Mapping):
        payload = raw
    elif isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            try:
                payload = _parse_json_response(raw)
            except Exception:
                return None
            if not payload:
                return None
    else:
        content = getattr(raw, "content", None)
        if not isinstance(content, str):
            return None
        return _coerce_payload(content)
    nested = payload.get("query_understanding_proposal") if isinstance(payload, Mapping) else None
    return nested if isinstance(nested, Mapping) else payload


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item or "").strip() for item in value if str(item or "").strip()]


def _contract_warnings(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    allowed_fields = set(SemanticQueryProposal.model_fields)
    for field in sorted(set(payload) - allowed_fields):
        warnings.append({"field": field, "reason": "schema_extra_forbidden"})
    for field, allowed in (
        ("analysis_scope", ALLOWED_ANALYSIS_SCOPES),
        ("methodology_intent", ALLOWED_METHODOLOGY_INTENTS),
        ("user_expectation", ALLOWED_USER_EXPECTATIONS),
    ):
        value = str(payload.get(field) or "").strip()
        if value and value not in allowed:
            warnings.append({"field": field, "value": value, "reason": f"{field}_not_allowed"})
    safety = normalize_safety_intent_label(payload.get("safety_intent"))
    if safety and safety not in ALLOWED_SAFETY_INTENTS:
        warnings.append({"field": "safety_intent", "value": payload.get("safety_intent"), "reason": "safety_intent_not_allowed"})
    for item in _string_list(payload.get("requested_dimensions")):
        if item not in SUPPORTED_DIMENSIONS:
            warnings.append({"field": "requested_dimensions", "value": item, "reason": "dimension_not_supported"})
    for item in _string_list(payload.get("requested_metrics")):
        metric = normalize_metric_label(item)
        if metric not in ALLOWED_ANALYSIS_METRICS:
            warnings.append({"field": "requested_metrics", "value": item, "reason": "metric_not_supported"})
    return warnings


def _schema_error_warnings(exc: ValidationError) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for err in exc.errors(include_url=False, include_context=False, include_input=False):
        loc = ".".join(str(item) for item in err.get("loc", []) if str(item))
        warnings.append(
            {
                "field": loc or "query_understanding_proposal",
                "reason": "proposal_schema_invalid",
                "detail": str(err.get("msg") or ""),
            }
        )
    return warnings


def _messages_for_query(raw_query: str) -> list[Any]:
    allowed_payload = allowed_parser_labels()
    system = (
        "You are a semantic parser for FilingLens, a financial filings analysis agent. "
        "Return JSON only. Choose only from the provided labels. "
        "Do not invent dimensions, metrics, task types, or safety categories. "
        "If the request is ambiguous or unsupported, mark ambiguity/needs_clarification instead of guessing. "
        "Direct fact questions must use methodology_intent='none'. "
        "Investment-advice-like or prediction-like wording must be reflected in safety_intent."
    )
    human = {
        "query": str(raw_query or ""),
        "allowed_labels": allowed_payload,
        "required_json_shape": {
            "company_mentions": ["NVIDIA"],
            "analysis_scope": "single_company",
            "methodology_intent": "risk",
            "requested_dimensions": ["cash_flow_quality"],
            "requested_metrics": ["operating_cash_flow"],
            "user_expectation": "diagnostic",
            "safety_intent": "normal",
            "time_scope": {
                "policy": "latest_available",
                "period_scope": "latest annual + latest quarterly",
                "is_explicit": False,
            },
            "confidence": 0.85,
            "ambiguity": False,
            "needs_clarification": False,
            "reasons": ["Short reason for the chosen labels."],
        },
    }
    return [SystemMessage(content=system), HumanMessage(content=json.dumps(human, ensure_ascii=False))]


def build_semantic_query_proposal(
    raw_query: str,
    *,
    llm_client: Any | None = None,
    mode: str | None = None,
) -> SemanticQueryParserResult:
    rollout_mode = normalize_semantic_parser_mode(mode)
    if rollout_mode == "off":
        return SemanticQueryParserResult(
            mode=rollout_mode,
            ok=False,
            source="disabled",
            diagnostics={"parse_status": "disabled"},
        )

    client = llm_client or _get_llm(reasoning=False, temperature=0.0, max_tokens=900)
    try:
        if hasattr(client, "invoke"):
            raw = client.invoke(_messages_for_query(raw_query))
        elif callable(client):
            raw = client(_messages_for_query(raw_query))
        else:
            return SemanticQueryParserResult(mode=rollout_mode, ok=False, source="llm", error="invalid_llm_client")
    except Exception as exc:  # pragma: no cover - runtime availability depends on local LLM service
        return SemanticQueryParserResult(mode=rollout_mode, ok=False, source="llm", error=str(exc))

    payload = _coerce_payload(raw)
    if not isinstance(payload, Mapping):
        return SemanticQueryParserResult(
            mode=rollout_mode,
            ok=False,
            source="llm",
            error="invalid_json_response",
            diagnostics={"parse_status": "invalid_json_response"},
        )
    warnings = _contract_warnings(payload)
    try:
        proposal = SemanticQueryProposal(**dict(payload))
    except ValidationError as exc:
        return SemanticQueryParserResult(
            mode=rollout_mode,
            ok=False,
            source="llm",
            error="proposal_schema_invalid",
            warnings=[*warnings, *_schema_error_warnings(exc)],
            diagnostics={"parse_status": "proposal_schema_invalid"},
        )
    sanitized = proposal.model_dump(exclude_none=True)
    return SemanticQueryParserResult(
        mode=rollout_mode,
        ok=True,
        source="llm",
        proposal=sanitized,
        warnings=warnings,
        diagnostics={
            "parse_status": "validated",
            "confidence": sanitized.get("confidence", 0.0),
            "proposal_fields": sorted(sanitized.keys()),
        },
    )
