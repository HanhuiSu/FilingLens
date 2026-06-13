"""Post-hoc answer-level evidence contract checks.

The checker validates an already-rendered user-visible final answer against the
evidence packet and DimensionStatusMap. It is intentionally deterministic, does
not call an LLM, and does not mutate the trace, final answer, output, synthesis,
or EvidencePacket.

This module is used by eval/reporting/CLI gates after an answer exists. It is
not a live `/chat` generation policy, synthesis step, repair loop, or fail-fast
middleware.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field

from src.agent.driver_evidence import evidence_summary_scope_warning
from src.agent.metric_availability import normalize_metric_name
from src.agent.metric_display import format_metric_value, metric_semantics
from src.agent.output_language import detect_output_language, language_leakage_terms


_INTERNAL_PATTERNS = (
    r"\bREQ-[A-Za-z0-9_-]+",
    r"\bdependency_[A-Za-z0-9_]+",
    r"\bnumeric_only_[A-Za-z0-9_]+",
    r"\brequired_evidence_missing\b",
    r"\bunsupported_claims_present\b",
)

_LIMITING_TERMS = (
    "有限",
    "合理推断",
    "有限判断",
    "基于业务结构推断",
    "边界",
    "限制",
    "不足",
    "缺少",
    "不能",
    "无法",
    "仅",
    "待验证",
    "假设",
    "需要验证",
    "仍需",
    "caveat",
    "limited",
    "boundary",
    "missing",
    "cannot",
)

_BENCHMARK_CLAIM_TERMS = (
    "历史高位",
    "历史低位",
    "行业均值",
    "同业均值",
    "显著高于行业",
    "显著低于行业",
    "高于同业",
    "低于同业",
    "peer average",
    "industry average",
    "historical high",
    "historical low",
    "above peers",
    "below peers",
)

_BENCHMARK_EVIDENCE_TERMS = _BENCHMARK_CLAIM_TERMS + (
    "benchmark",
    "peer",
    "peers",
    "industry",
    "historical",
    "history",
    "行业",
    "同业",
    "历史",
)

_COMPANY_SPECIFIC_TOKEN_RULES: dict[str, tuple[str, ...]] = {
    "AMZN": ("aws", "amazon web services", "prime", "marketplace", "fulfillment", "retail margins"),
    "AAPL": ("iphone", "ipad", "mac", "app store", "services"),
    "MSFT": ("azure", "office 365", "linkedin", "windows", "xbox"),
    "NVDA": ("cuda", "blackwell", "hopper", "infiniband", "xdr infiniband", "nvlink", "ethernet", "spectrum-x", "gb200", "gb300"),
}

_TOKEN_OWNER_BY_TERM: dict[str, str] = {
    term: company
    for company, terms in _COMPANY_SPECIFIC_TOKEN_RULES.items()
    for term in terms
}

_NEGATION_TERMS = (
    "不",
    "不能",
    "无法",
    "不构成",
    "不做",
    "不应",
    "not",
    "cannot",
    "can't",
    "do not",
    "does not",
    "without",
    "no ",
)

_NEGATION_CONTRAST_TERMS = (
    "但",
    "但是",
    "不过",
    "然而",
    "可是",
    "but",
    "however",
    "although",
    "though",
    ";",
    "；",
)

_NEGATIVE_NUMBER_CONTEXT_TERMS = (
    "为负",
    "亏损",
    "下降",
    "下滑",
    "减少",
    "降低",
    "negative",
    "loss",
    "decline",
    "declined",
    "decrease",
    "decreased",
    "down",
)

_VALUATION_FORBIDDEN = (
    "买入",
    "卖出",
    "推荐买",
    "推荐卖",
    "值得买",
    "应该买",
    "应该卖",
    "目标价",
    "目标价格",
    "目标股价",
    "target price",
    "price target",
    "DCF",
    "dcf",
    "should buy",
    "should sell",
    "should hold",
    "recommend buying",
    "recommend selling",
    "recommend holding",
    "buy rating",
    "sell rating",
    "hold rating",
    "建议持有",
    "应该持有",
    "应持有",
    "继续持有",
    "持有评级",
)

_DETERMINISTIC_FORECAST_FORBIDDEN = (
    "一定会上涨",
    "一定会下跌",
    "一定上涨",
    "一定下跌",
    "必然上涨",
    "必然下跌",
    "肯定上涨",
    "肯定下跌",
    "will definitely rise",
    "will definitely fall",
    "guaranteed to rise",
    "guaranteed to fall",
    "is certain to rise",
    "is certain to fall",
)

_DETERMINISTIC_FORECAST_PATTERNS = (
    r"下(?:一)?季度.{0,30}(?:一定|必然|肯定).{0,24}(?:增长|下降|上升|下滑|改善|恶化)",
    r"next quarter.{0,40}(?:will definitely|is certain to|guaranteed to).{0,30}(?:increase|decrease|rise|fall|improve|worsen)",
)

_MATERIAL_CLAIM_TERMS = (
    "收入",
    "营收",
    "利润",
    "净利润",
    "现金流",
    "自由现金流",
    "估值",
    "市值",
    "风险",
    "竞争",
    "毛利率",
    "净利率",
    "revenue",
    "profit",
    "income",
    "cash flow",
    "free cash flow",
    "valuation",
    "market cap",
    "risk",
    "competition",
    "margin",
    "P/E",
    "P/S",
)

_STRONG_CAUSAL_TERMS = (
    "确定由",
    "必然导致",
    "证明了",
    "主要原因就是",
    "一定是因为",
    "definitively driven by",
    "proved that",
    "proves that",
    "the main reason is",
    "must be because",
)

_DEFINITIVE_CAUSAL_PATTERNS = (
    r"营收增长.{0,24}主要由",
    r"收入增长.{0,24}主要由",
    r"总营收.{0,24}(?:由|得益于|归因于|推动)",
    r"总收入.{0,24}(?:由|得益于|归因于|推动)",
    r"核心原因是",
    r"主要原因就是",
    r"\b(?:revenue growth|total revenue|company revenue|consolidated revenue).{0,40}\b(?:driven by|due to|because of|attributable to)\b",
)

_COMPANY_CAUSAL_TARGET_PATTERNS = (
    r"(?:nvidia|nvda|公司|总|整体).{0,18}(?:营收|收入).{0,20}(?:增长|增加)",
    r"(?:营收|收入).{0,20}(?:增长|增加).{0,18}(?:nvidia|nvda|公司|总|整体)",
    r"\b(?:nvidia|nvda|company|total|overall|consolidated).{0,25}\b(?:revenue|sales).{0,25}\b(?:growth|increase|grew|increased)\b",
    r"\b(?:revenue|sales).{0,25}\b(?:growth|increase|grew|increased).{0,25}\b(?:company|total|overall|consolidated)\b",
)

_BOUNDED_SCOPE_TERMS = (
    "分部层面",
    "产品层面",
    "业务线索",
    "线索",
    "可能",
    "补充证据",
    "不能完整代表",
    "不能直接证明",
    "bounded",
    "suggests",
    "points to",
    "may",
    "possible",
    "segment-level",
    "product-level",
    "cannot fully represent",
)

_HYPOTHESIS_MARKERS = (
    "待验证",
    "假设",
    "可能因素",
    "需要验证",
    "hypothesis",
    "to verify",
    "needs verification",
    "possible factor",
)

_MISSING_DIMENSION_FORBIDDEN = {
    "cash_flow_quality": ("现金流强", "现金流弱", "strong cash flow", "weak cash flow"),
    "valuation_and_risk_boundary": ("便宜", "昂贵", "cheap", "expensive", "worth buying"),
    "moat_and_competitive_risk": ("主要竞争风险", "具体风险", "major competitive risk"),
    "business_model": ("靠", "业务主要", "business model"),
}

_DIMENSION_LABELS = {
    "business_model": ("业务模式", "靠什么赚钱", "business model"),
    "revenue_quality": ("收入质量", "收入", "revenue"),
    "profitability_quality": ("盈利质量", "净利润", "净利率", "profitability"),
    "cash_flow_quality": ("现金流质量", "经营现金流", "自由现金流", "cash flow"),
    "balance_sheet_and_capital_intensity": ("资产负债", "资本强度", "债务", "balance sheet"),
    "moat_and_competitive_risk": ("竞争与风险", "风险", "竞争", "risk"),
    "valuation_and_risk_boundary": ("估值", "市值", "P/E", "P/S", "valuation"),
}

_CHINESE_NUMERIC_SUFFIXES: tuple[tuple[str, float, str], ...] = (
    ("万亿美元", 1_000_000_000_000.0, "currency"),
    ("万亿", 1_000_000_000_000.0, "scaled"),
    ("十亿美元", 1_000_000_000.0, "currency"),
    ("亿美元", 100_000_000.0, "currency"),
    ("亿", 100_000_000.0, "scaled"),
    ("万美元", 10_000.0, "currency"),
    ("万", 10_000.0, "scaled"),
)

_CHINESE_NUMERIC_SUFFIX_PATTERN = "万亿美元|十亿美元|亿美元|万美元|万亿|亿|万"


@dataclass(frozen=True)
class _NumericObservation:
    text: str
    value: float
    kind: str


class ContractViolation(BaseModel):
    """Runtime-safe violation object for live guard routing."""

    code: str
    message: str
    severity: Literal["hard_fail", "repairable", "evidence_missing", "warning"] = "repairable"
    evidence_id: str | None = None
    affected_citations: list[str] = Field(default_factory=list)
    answer_span: str | None = None
    suggested_fix: str | None = None
    public_replacement: str | None = None
    repair_instruction: str | None = None
    dimension_id: str | None = None


class ContractResult(BaseModel):
    """Runtime AnswerContract result used by LangGraph guard nodes."""

    passed: bool
    severity: Literal["pass", "hard_fail", "repairable", "evidence_missing", "warning"] = "pass"
    decision: Literal["passed", "warning", "repairable", "blocked"] = "passed"
    route: Literal["pass", "repair_answer", "need_more_evidence", "blocked"] = "pass"
    action: Literal["pass", "add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded", "retry_evidence", "block"] = "pass"
    violations: list[ContractViolation] = Field(default_factory=list)
    suggested_repair: str | None = None
    missing_requirements: list[str] = Field(default_factory=list)
    blocking_missing_requirements: list[str] = Field(default_factory=list)
    warnings: list[ContractViolation] = Field(default_factory=list)
    public_summary: str | None = None
    scope_overclaim_check: dict[str, Any] = Field(default_factory=dict)
    scope_overclaim_violations: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _answer_text(trace: Mapping[str, Any]) -> str:
    final = str(trace.get("final_answer") or "").strip()
    if final:
        return final
    output = _as_dict(trace.get("output"))
    for key in ("answer", "summary", "short_answer"):
        value = str(output.get(key) or "").strip()
        if value:
            return value
    synthesis = _as_dict(trace.get("synthesis"))
    return str(synthesis.get("short_answer") or "").strip()


def _packet(trace: Mapping[str, Any]) -> dict[str, Any]:
    packet = _as_dict(trace.get("evidence_packet"))
    if packet:
        return packet
    output = _as_dict(trace.get("output"))
    return _as_dict(output.get("evidence_packet"))


def _status_map(trace: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    packet = _packet(trace)
    for source in (
        trace.get("dimension_status_by_id"),
        trace.get("dimension_status_map"),
        packet.get("dimension_status_by_id"),
        packet.get("dimension_status_map"),
        _as_dict(trace.get("evidence_sufficiency")).get("dimension_status_map"),
        _as_dict(trace.get("trace_summary")).get("dimension_status_by_id"),
    ):
        if isinstance(source, dict) and source:
            return {str(key): _as_dict(value) for key, value in source.items()}
    return {}


def _numeric_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    packet = _packet(trace)
    rows: list[dict[str, Any]] = []
    output = _as_dict(trace.get("output"))

    def extend_rows(source: Any) -> None:
        if isinstance(source, dict):
            if source.get("metric") or source.get("value") is not None or source.get("display_value"):
                rows.append(dict(source))
            for key in ("items", "validated_evidence", "numeric_evidence", "computed_metrics", "evidence"):
                extend_rows(source.get(key))
            return
        for item in _as_list(source):
            if isinstance(item, dict):
                extend_rows(item)

    for source in (
        packet.get("numeric_table"),
        packet.get("numeric_evidence"),
        packet.get("comparison_table"),
        packet.get("computed_metrics"),
        output.get("numeric_evidence"),
        output.get("computed_metrics"),
        trace.get("numeric_evidence"),
        trace.get("computed_metrics"),
        trace.get("validated_evidence"),
        trace.get("evidence_collection_results"),
    ):
        extend_rows(source)
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("evidence_id") or ""),
            normalize_metric_name(str(row.get("metric") or "")),
            str(row.get("value") or row.get("display_value") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _text_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    packet = _packet(trace)
    rows: list[dict[str, Any]] = []
    for source in (
        packet.get("text_snippets"),
        packet.get("text_evidence"),
        _as_dict(trace.get("output")).get("text_evidence"),
        trace.get("text_evidence"),
    ):
        rows.extend([dict(item) for item in _as_list(source) if isinstance(item, dict)])
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        evidence_id = str(row.get("evidence_id") or "").strip()
        key = evidence_id or json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_numeric_text(text: str) -> _NumericObservation | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    normalized = raw.replace(",", "").replace(" ", "")
    suffix_multiplier = 1.0
    kind = "raw"
    normalized_lower_initial = normalized.lower()
    if normalized.endswith("股"):
        kind = "shares"
        normalized = normalized[:-1]
    elif normalized_lower_initial.endswith("shares"):
        kind = "shares"
        normalized = normalized[: -len("shares")]
    if normalized.startswith("-$"):
        kind = "currency"
        normalized = "-" + normalized[2:]
    elif normalized.startswith("$"):
        kind = "currency"
        normalized = normalized[1:]
    if normalized.endswith("%"):
        kind = "percentage"
        normalized = normalized[:-1]
        value = _to_float(normalized)
        return _NumericObservation(raw, value / 100.0, kind) if value is not None else None
    if normalized.lower().endswith("x"):
        kind = "multiple"
        normalized = normalized[:-1]
        value = _to_float(normalized)
        return _NumericObservation(raw, value, kind) if value is not None else None
    for suffix_text, multiplier, suffix_kind in _CHINESE_NUMERIC_SUFFIXES:
        if normalized.endswith(suffix_text):
            suffix_multiplier = multiplier
            normalized = normalized[: -len(suffix_text)]
            if kind == "shares":
                kind = "shares"
            elif kind == "currency" or suffix_kind == "currency":
                kind = "currency"
            else:
                kind = "scaled"
            break
    normalized_lower = normalized.lower()
    if suffix_multiplier == 1.0:
        for suffix_text, multiplier in (
            ("trillion", 1_000_000_000_000.0),
            ("billion", 1_000_000_000.0),
            ("million", 1_000_000.0),
        ):
            if normalized_lower.endswith(suffix_text):
                suffix_multiplier = multiplier
                normalized = normalized[: -len(suffix_text)]
                kind = "currency" if kind == "currency" else ("shares" if kind == "shares" else "scaled")
                break
    suffix = normalized[-1:].upper()
    if suffix_multiplier == 1.0 and suffix in {"T", "B", "M"}:
        suffix_multiplier = {"T": 1_000_000_000_000.0, "B": 1_000_000_000.0, "M": 1_000_000.0}[suffix]
        normalized = normalized[:-1]
        kind = "currency" if kind == "currency" else ("shares" if kind == "shares" else "scaled")
    value = _to_float(normalized)
    if value is None:
        return None
    return _NumericObservation(raw, value * suffix_multiplier, kind)


def _number_context_is_negative(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 24) : min(len(text), end + 8)].lower()
    return any(term.lower() in window for term in _NEGATIVE_NUMBER_CONTEXT_TERMS)


def _extract_answer_numbers(answer: str) -> list[_NumericObservation]:
    patterns = (
        r"-?\d[\d,]*(?:\.\d+)?\s*(?:T|B|M)\s*shares\b",
        r"-?\d[\d,]*(?:\.\d+)?\s*(?:trillion|billion|million)\s*shares\b",
        r"-?\d[\d,]*(?:\.\d+)?\s*(?:万亿|亿|万)股",
        r"-?\$-?\d[\d,]*(?:\.\d+)?\s*(?:trillion|billion|million)\b",
        rf"-?\$-?\d[\d,]*(?:\.\d+)?\s*(?:{_CHINESE_NUMERIC_SUFFIX_PATTERN})",
        r"-?\$-?\d[\d,]*(?:\.\d+)?\s*[TBM]?",
        r"-?\d[\d,]*(?:\.\d+)?%",
        r"-?\d[\d,]*(?:\.\d+)?x",
        r"-?\d[\d,]*(?:\.\d+)?\s*(?:trillion|billion|million)\b",
        r"-?\d[\d,]*(?:\.\d+)?\s*(?:T|B|M)\b",
        rf"-?\d[\d,]*(?:\.\d+)?\s*(?:{_CHINESE_NUMERIC_SUFFIX_PATTERN})",
    )
    observations: list[_NumericObservation] = []
    spans: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, answer, flags=re.IGNORECASE):
            span = match.span()
            if any(max(span[0], start) < min(span[1], end) for start, end in spans):
                continue
            parsed = _parse_numeric_text(match.group(0))
            if parsed:
                if parsed.value > 0 and _number_context_is_negative(answer, match.start(), match.end()):
                    parsed = _NumericObservation(parsed.text, -abs(parsed.value), parsed.kind)
                observations.append(parsed)
                spans.append(span)
    deduped: list[_NumericObservation] = []
    seen: set[tuple[str, float, str]] = set()
    for item in observations:
        key = (item.text, round(item.value, 6), item.kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _numeric_aliases_for_row(row: Mapping[str, Any]) -> list[_NumericObservation]:
    metric = normalize_metric_name(str(row.get("metric") or ""))
    value = row.get("value")
    numeric = _to_float(value)
    aliases: list[_NumericObservation] = []

    def add(text: Any, *, fallback_kind: str | None = None) -> None:
        parsed = _parse_numeric_text(str(text or ""))
        if parsed:
            aliases.append(parsed)
        elif fallback_kind and numeric is not None and str(text or "").strip():
            aliases.append(_NumericObservation(str(text), numeric, fallback_kind))

    if numeric is not None:
        semantics = metric_semantics(metric)
        kind = semantics.unit_type
        if kind == "currency_per_share":
            kind = "currency"
        aliases.append(_NumericObservation(str(value), numeric, kind))
        if float(numeric).is_integer():
            aliases.append(_NumericObservation(str(int(numeric)), numeric, kind))
        display = format_metric_value(metric, numeric, str(row.get("unit") or ""))
        add(display)
        if display.startswith("$"):
            add(display[1:])
        if kind == "currency":
            add(f"${numeric / 1_000_000_000_000:.2f}T")
            add(f"{numeric / 1_000_000_000_000:.2f}T")
            add(f"${numeric / 1_000_000_000:.2f}B")
            add(f"{numeric / 1_000_000_000:.2f}B")
            add(f"{numeric / 1_000_000_000:.2f}十亿美元")
            add(f"{numeric / 100_000_000:.2f}亿美元")
            add(f"{numeric / 100_000_000:.2f}亿")
            add(f"{numeric / 10_000:.2f}万美元")
            add(f"{numeric / 10_000:.2f}万")
        elif kind == "shares":
            add(f"{numeric / 1_000_000_000:.2f}B shares")
            add(f"{numeric / 1_000_000:.2f}M shares")
            add(f"{numeric / 100_000_000:.2f}亿股")
            add(f"{numeric / 10_000:.2f}万股")
        elif kind == "percentage":
            add(f"{numeric * 100:.2f}%")
        elif kind == "multiple":
            add(f"{numeric:.2f}x")
    for key in ("display_value", "formatted_value"):
        display = str(row.get(key) or "").strip()
        if not display:
            continue
        add(display)
        if display.startswith("$"):
            add(display[1:])
    deduped: list[_NumericObservation] = []
    seen: set[tuple[str, float, str]] = set()
    for item in aliases:
        key = (item.text, round(item.value, 6), item.kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _allowed_numeric_observations(trace: Mapping[str, Any]) -> list[_NumericObservation]:
    allowed: list[_NumericObservation] = []
    for row in _numeric_rows(trace):
        allowed.extend(_numeric_aliases_for_row(row))
    for row in _text_rows(trace):
        for key in ("supporting_snippet", "text_snippet", "claim", "evidence_summary"):
            text = str(row.get(key) or "").strip()
            if not text:
                continue
            allowed.extend(_extract_answer_numbers(text))
    return allowed


def _numeric_close(candidate: _NumericObservation, allowed: _NumericObservation) -> bool:
    kind_match = candidate.kind == allowed.kind or {candidate.kind, allowed.kind} <= {"currency", "scaled"}
    if not kind_match and candidate.kind != "raw" and allowed.kind != "raw":
        return False
    tolerance = max(abs(allowed.value) * 0.015, 0.01)
    if candidate.kind in {"percentage", "multiple"}:
        tolerance = max(abs(allowed.value) * 0.02, 0.0005)
    return abs(candidate.value - allowed.value) <= tolerance


def _numeric_value_close(candidate: _NumericObservation, allowed: _NumericObservation) -> bool:
    tolerance = max(abs(allowed.value) * 0.015, 0.01)
    if candidate.kind in {"percentage", "multiple"} or allowed.kind in {"percentage", "multiple"}:
        tolerance = max(abs(allowed.value) * 0.02, 0.0005)
    return abs(candidate.value - allowed.value) <= tolerance


def _numeric_repair_alias(candidate: _NumericObservation, allowed: list[_NumericObservation]) -> str:
    matches = [item for item in allowed if _numeric_value_close(candidate, item)]
    if not matches:
        return ""

    def score(item: _NumericObservation) -> tuple[int, int]:
        text = str(item.text or "")
        has_display_unit = bool(re.search(r"[$%]|[0-9][TBM]\b|shares|股|亿|万|x\b", text, flags=re.IGNORECASE))
        kind_match = item.kind == candidate.kind or {item.kind, candidate.kind} <= {"currency", "scaled"}
        return (0 if has_display_unit else 1, 0 if kind_match else 1)

    return str(sorted(matches, key=score)[0].text or "")


def _check_numeric_grounding(trace: Mapping[str, Any], answer: str) -> tuple[list[dict[str, Any]], float]:
    observations = _extract_answer_numbers(answer)
    if not observations:
        return [], 1.0
    allowed = _allowed_numeric_observations(trace)
    grounded = 0
    violations: list[dict[str, Any]] = []
    for item in observations:
        if any(_numeric_close(item, allowed_item) for allowed_item in allowed):
            grounded += 1
        else:
            suggested_value = _numeric_repair_alias(item, allowed)
            violations.append(
                {
                    "type": "unsupported_numeric",
                    "message": f"Numeric value {item.text} is not grounded in validated numeric evidence.",
                    "value": item.text,
                    "suggested_value": suggested_value,
                }
            )
    return violations, grounded / len(observations)


def _check_citations(trace: Mapping[str, Any], answer: str) -> tuple[list[dict[str, Any]], float]:
    refs = re.findall(r"\[([NT]\d+)\]", answer)
    if not refs:
        return [], 1.0
    valid = {
        str(row.get("evidence_id") or "").strip()
        for row in [*_numeric_rows(trace), *_text_rows(trace)]
        if str(row.get("evidence_id") or "").strip()
    }
    ok = 0
    violations: list[dict[str, Any]] = []
    for ref in refs:
        if ref in valid:
            ok += 1
        else:
            violations.append({"type": "invalid_citation", "message": f"Citation [{ref}] is not present in evidence."})
    return violations, ok / len(refs)


def _contains_any(text: str, terms: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(str(term).lower() in lowered for term in terms if str(term))


def _context_is_negated(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 48) : start].lower()
    for term in sorted(_NEGATION_TERMS, key=len, reverse=True):
        needle = term.lower()
        idx = before.rfind(needle)
        if idx < 0:
            continue
        between = before[idx + len(needle) :]
        if any(contrast in between for contrast in _NEGATION_CONTRAST_TERMS):
            return False
        if len(between) <= 28:
            return True
    return False


def _term_violations(text: str, terms: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    lowered = text.lower()
    for term in terms:
        needle = term.lower()
        start = 0
        while needle:
            idx = lowered.find(needle, start)
            if idx < 0:
                break
            if not _context_is_negated(text, idx, idx + len(term)):
                out.append(term)
            start = idx + len(term)
    return out


def _dimension_refs(status: Mapping[str, Any]) -> list[str]:
    refs = []
    for key in ("supporting_evidence_ids", "evidence_refs", "numeric_evidence_refs", "text_evidence_refs"):
        refs.extend(str(item) for item in _as_list(status.get(key)) if str(item).strip())
    return refs


def _check_dimension_status(trace: Mapping[str, Any], answer: str) -> list[dict[str, Any]]:
    status_map = _status_map(trace)
    violations: list[dict[str, Any]] = []
    lowered = answer.lower()
    for dimension_id, status in status_map.items():
        status_value = str(status.get("status") or "")
        labels = _DIMENSION_LABELS.get(dimension_id, (dimension_id,))
        dimension_mentioned = _contains_any(answer, list(labels))
        if status_value == "missing":
            forbidden = _term_violations(answer, _MISSING_DIMENSION_FORBIDDEN.get(dimension_id, ()))
            if forbidden:
                violations.append(
                    {
                        "type": "dimension_status_violation",
                        "dimension_id": dimension_id,
                        "message": f"Missing dimension has unsupported positive claim: {', '.join(sorted(set(forbidden)))}",
                    }
                )
        elif status_value == "partial" and dimension_mentioned and not _contains_any(lowered, _LIMITING_TERMS):
            violations.append(
                {
                    "type": "dimension_status_violation",
                    "dimension_id": dimension_id,
                    "message": "Partial dimension is discussed without limited/bounded wording.",
                }
            )
        elif status_value == "satisfied" and dimension_mentioned and not _dimension_refs(status):
            violations.append(
                {
                    "type": "dimension_status_violation",
                    "dimension_id": dimension_id,
                    "message": "Satisfied dimension lacks supporting evidence ids.",
                }
            )
    return violations


def _pattern_violations(text: str, patterns: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            if not _context_is_negated(text, match.start(), match.end()):
                out.append(match.group(0))
    return out


def _check_forbidden_claims(answer: str) -> list[dict[str, Any]]:
    violations = []
    for term in _term_violations(answer, _VALUATION_FORBIDDEN):
        violations.append({"type": "forbidden_claim", "message": f"Forbidden advice/valuation term used: {term}"})
    for term in _term_violations(answer, _DETERMINISTIC_FORECAST_FORBIDDEN):
        violations.append({"type": "forbidden_claim", "message": f"Unsupported deterministic forecast used: {term}"})
    for span in _pattern_violations(answer, _DETERMINISTIC_FORECAST_PATTERNS):
        violations.append({"type": "forbidden_claim", "message": f"Unsupported deterministic forecast used: {span}"})
    return violations


def _format_constraints(trace: Mapping[str, Any]) -> dict[str, Any]:
    raw = trace.get("format_constraints")
    if isinstance(raw, Mapping):
        return dict(raw)
    output = trace.get("output")
    if isinstance(output, Mapping) and isinstance(output.get("format_constraints"), Mapping):
        return dict(output.get("format_constraints") or {})
    query = str(trace.get("user_query") or "").lower()
    if re.search(r"(一句话|一段话|用一句|只用一句|one sentence|single sentence|in one sentence)", query):
        return {"one_sentence": True, "max_sentences": 1}
    return {}


def _format_sentence_count(answer: str) -> int:
    text = re.sub(r"\[[A-Z]\d+\]", "", str(answer or ""))
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    return len([part for part in re.split(r"[。！？!?]+|\n+", text) if part.strip()])


def _check_format_constraints(trace: Mapping[str, Any], answer: str) -> list[dict[str, Any]]:
    constraints = _format_constraints(trace)
    max_sentences = int(constraints.get("max_sentences") or (1 if constraints.get("one_sentence") else 0) or 0)
    if max_sentences > 0 and _format_sentence_count(answer) > max_sentences:
        return [
            {
                "type": "format_constraint_violation",
                "message": f"Answer must contain at most {max_sentences} sentence(s).",
                "answer_span": str(answer or "")[:500],
                "suggested_fix": "Compress the answer to one evidence-cited sentence.",
                "expected_max_sentences": max_sentences,
            }
        ]
    return []


def _benchmark_evidence_refs(trace: Mapping[str, Any]) -> set[str]:
    refs: set[str] = set()
    for row in _numeric_rows(trace):
        haystack = " ".join(
            str(row.get(key) or "")
            for key in (
                "metric",
                "metric_label",
                "dimension_id",
                "requirement_id",
                "source_tag",
                "provenance",
                "evidence_role",
                "role",
                "description",
                "label",
            )
        ).lower()
        if _contains_any(haystack, _BENCHMARK_EVIDENCE_TERMS):
            ref = str(row.get("evidence_id") or "").strip()
            if ref:
                refs.add(ref)
    for row in _text_rows(trace):
        haystack = " ".join(
            str(row.get(key) or "")
            for key in (
                "claim",
                "summary",
                "supporting_snippet",
                "text_snippet",
                "dimension_id",
                "requirement_id",
                "section",
                "source_tag",
            )
        ).lower()
        if _contains_any(haystack, _BENCHMARK_EVIDENCE_TERMS):
            ref = str(row.get("evidence_id") or "").strip()
            if ref:
                refs.add(ref)
    return refs


def _check_unsupported_benchmark_claims(trace: Mapping[str, Any], answer: str) -> list[dict[str, Any]]:
    benchmark_refs = _benchmark_evidence_refs(trace)
    violations: list[dict[str, Any]] = []
    for sentence in _answer_sentences(answer):
        if not _contains_any(sentence.lower(), _BENCHMARK_CLAIM_TERMS):
            continue
        if _contains_any(sentence.lower(), ("缺少", "不足", "不能", "无法", "missing", "without", "cannot", "not enough")):
            continue
        cited_refs = set(_citation_refs_in_sentence(sentence))
        if cited_refs and cited_refs & benchmark_refs:
            continue
        violations.append(
            {
                "type": "unsupported_benchmark_claim",
                "message": "Benchmark claim requires benchmark-specific evidence; ordinary current valuation rows are not enough.",
                "answer_span": sentence,
                "suggested_fix": "估值倍数较高，但缺少历史/行业基准，不能严格判断是否处于高位。",
            }
        )
    return violations


def _evidence_text_by_ref(trace: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in _text_rows(trace):
        ref = str(row.get("evidence_id") or "").strip()
        if not ref:
            continue
        out[ref] = " ".join(
            str(row.get(key) or "")
            for key in (
                "claim",
                "summary",
                "supporting_snippet",
                "text_snippet",
                "source_text",
                "section",
                "ticker",
                "company",
            )
        )
    for row in _numeric_rows(trace):
        ref = str(row.get("evidence_id") or "").strip()
        if not ref:
            continue
        out.setdefault(ref, " ".join(str(row.get(key) or "") for key in ("metric", "metric_label", "ticker", "company", "source_tag")))
    return out


def _answer_companies(trace: Mapping[str, Any]) -> list[str]:
    companies = _companies(trace)
    if companies:
        return companies
    packet = _as_dict(trace.get("evidence_packet"))
    packet_companies: list[str] = []
    for item in _as_list(packet.get("companies")):
        if isinstance(item, Mapping):
            text = str(item.get("ticker") or item.get("TICKER") or item.get("symbol") or item.get("company") or "").upper().strip()
        else:
            text = str(item or "").upper().strip()
        if text:
            packet_companies.append(text)
    comparison_target = str(packet.get("comparison_target") or "").upper().strip()
    if comparison_target:
        packet_companies.append(comparison_target)
    return list(dict.fromkeys(packet_companies))


def _check_company_specific_token_leakage(trace: Mapping[str, Any], answer: str) -> list[dict[str, Any]]:
    companies = set(_answer_companies(trace))
    if not companies:
        return []
    evidence_text = _evidence_text_by_ref(trace)
    violations: list[dict[str, Any]] = []
    for sentence in _answer_sentences(answer):
        lowered_sentence = sentence.lower()
        refs = _citation_refs_in_sentence(sentence)
        cited_text = " ".join(evidence_text.get(ref, "") for ref in refs).lower()
        for term, owner in _TOKEN_OWNER_BY_TERM.items():
            if term not in lowered_sentence:
                continue
            if owner in companies:
                continue
            else:
                if refs and term in cited_text:
                    continue
            violations.append(
                {
                    "type": "company_specific_token_leakage",
                    "message": f"Company-specific token {term!r} is not supported for the requested company context.",
                    "answer_span": sentence,
                    "token": term,
                    "token_owner": owner,
                    "suggested_fix": "Remove the unsupported company-specific template sentence.",
                }
            )
            break
    return violations


def _risk_focused_payload(trace: Mapping[str, Any]) -> dict[str, Any]:
    for source in (
        _as_dict(trace.get("risk_focused_answer")),
        _as_dict(_as_dict(trace.get("synthesis")).get("risk_focused_answer")),
        _as_dict(_as_dict(trace.get("output")).get("risk_focused_answer")),
        _as_dict(_as_dict(_as_dict(trace.get("output")).get("synthesis")).get("risk_focused_answer")),
    ):
        if source:
            return source
    return {}


def _risk_ranking_support_counts(trace: Mapping[str, Any]) -> dict[str, int]:
    risk_answer = _risk_focused_payload(trace)
    ranking = [dict(item) for item in _as_list(risk_answer.get("risk_ranking")) if isinstance(item, dict)]
    supplemental = [dict(item) for item in _as_list(risk_answer.get("supplemental_risk_signals")) if isinstance(item, dict)]
    template_only = [dict(item) for item in _as_list(risk_answer.get("template_only_risk_items")) if isinstance(item, dict)]
    direct_count = sum(
        1
        for item in ranking
        if str(item.get("mechanism_support_level") or "") == "direct"
        and any(str(ref).startswith("T") for ref in _as_list(item.get("evidence_refs")))
    )
    template_core_count = sum(1 for item in ranking if str(item.get("mechanism_support_level") or "") == "template_only")
    missing_text_refs = sum(
        1
        for item in ranking
        if not any(str(ref).startswith("T") for ref in _as_list(item.get("evidence_refs")))
    )
    return {
        "risk_items_directly_supported_count": direct_count,
        "risk_items_inferred_count": len(supplemental),
        "risk_items_template_only_count": len(template_only) + template_core_count,
        "risk_items_missing_text_ref_count": missing_text_refs,
    }


def _check_risk_ranking_support(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    risk_answer = _risk_focused_payload(trace)
    if not risk_answer:
        return []
    violations: list[dict[str, Any]] = []
    for item in _as_list(risk_answer.get("risk_ranking")):
        if not isinstance(item, dict):
            continue
        support_level = str(item.get("mechanism_support_level") or "").strip()
        refs = [str(ref) for ref in _as_list(item.get("evidence_refs")) if str(ref).strip()]
        if support_level == "template_only":
            violations.append(
                {
                    "type": "template_only_core_risk",
                    "message": "Template-only risk items cannot enter the core risk ranking.",
                    "answer_span": str(item.get("theme_name") or ""),
                    "suggested_fix": "Move template-only risk items to observation metrics or remove them from the ranking.",
                }
            )
        if not any(ref.startswith("T") for ref in refs):
            violations.append(
                {
                    "type": "risk_item_missing_text_ref",
                    "message": "Core risk ranking items require at least one validated text-evidence reference.",
                    "answer_span": str(item.get("theme_name") or ""),
                    "suggested_fix": "Remove the unsupported ranking item or cite validated risk text.",
                }
            )
    return violations


def _material_sentences_without_citation(answer: str) -> list[str]:
    out: list[str] = []
    sentences: list[str] = []
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.search(r"\[[A-Z]\d+\]", line):
            continue
        if _is_material_heading_line(line):
            continue
        sentences.extend(re.split(r"(?<=[。！？.!?；;])\s+", line))
    for sentence in sentences:
        clean = sentence.strip(" -•\t")
        if _is_material_heading_line(clean):
            continue
        if len(clean) < 8:
            continue
        if clean.startswith("|") or clean.count("|") >= 2:
            continue
        if re.search(r"\[[NT]\d+\]", clean):
            continue
        if not _contains_any(clean, _MATERIAL_CLAIM_TERMS):
            continue
        if _contains_any(clean, _LIMITING_TERMS) or _contains_any(
            clean,
            (
                "证据不足",
                "非投资建议",
                "不构成投资建议",
                "不构成买卖建议",
                "不能形成买卖建议",
                "更合适的做法是按",
                "仅基于当前已验证证据",
                "只基于当前已验证证据",
                "not investment advice",
                "not a recommendation",
                "not a buy or sell recommendation",
                "insufficient evidence",
                "validated evidence only",
            ),
        ):
            continue
        out.append(clean[:160])
    return out


_KNOWN_PUBLIC_HEADINGS = {
    "business model and revenue sources",
    "business model",
    "revenue quality",
    "profitability quality",
    "cash flow and capex",
    "balance sheet and capital intensity",
    "valuation boundary",
    "primary risks",
    "overall judgment",
    "key follow-up indicators",
    "evidence boundary",
    "risk judgment",
    "monitorable disclosed risk signals",
    "core risk ranking",
    "supplemental risk signals",
    "risk transmission path",
    "potential revenue / profit / cash-flow impact",
    "key observation indicators",
    "结论",
    "已验证事实",
    "合理推断",
    "待验证假设",
    "已验证风险文本",
    "基于业务模型的风险排序",
    "财务传导路径",
    "待验证数据",
    "公司概览",
    "业务与收入来源",
    "财务画像",
    "风险边界",
    "证据限制",
    "单公司深度概览",
    "总体判断",
    "业务模式与收入来源",
    "收入质量",
    "盈利质量",
    "现金流与资本开支",
    "资产负债与资本强度",
    "估值边界",
    "主要风险",
    "主要风险（竞争与风险）",
    "关键观察指标",
    "证据边界",
    "风险判断",
    "可监控的已披露风险线索",
    "核心风险排序",
    "补充风险线索",
    "风险传导路径",
    "对收入 / 利润 / 现金流的可能影响",
}


def _is_material_heading_line(line: str) -> bool:
    text = str(line or "").strip().strip("#:：")
    text = text.strip("*_` ")
    if not text:
        return True
    normalized = re.sub(r"\s+", " ", text).strip()
    lowered = normalized.lower()
    if lowered in _KNOWN_PUBLIC_HEADINGS or normalized in _KNOWN_PUBLIC_HEADINGS:
        return True
    if len(normalized) > 80:
        return False
    if re.search(r"[。！？.!?；;]\s*$", normalized):
        return False
    if re.search(r"\d", normalized):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z/&-]*", normalized)
    if words and len(words) <= 8:
        title_words = sum(1 for word in words if word[:1].isupper() or word.lower() in {"and", "or", "of", "the", "to", "a", "an"})
        if title_words == len(words):
            return True
    return False


def _metrics_for_material_sentence(sentence: str) -> tuple[str, ...]:
    lowered = sentence.lower()
    metrics: list[str] = []
    if _contains_any(lowered, ("收入", "营收", "revenue", "sales", "增长", "growth")):
        metrics.extend(["revenue", "revenue_growth"])
    if _contains_any(lowered, ("利润", "净利润", "profit", "income")):
        metrics.extend(["net_income", "operating_income"])
    if _contains_any(lowered, ("毛利率", "gross margin")):
        metrics.append("gross_margin")
    if _contains_any(lowered, ("净利率", "net margin")):
        metrics.append("net_margin")
    if _contains_any(lowered, ("现金流", "自由现金流", "fcf", "cash flow")):
        metrics.extend(["operating_cash_flow", "free_cash_flow", "fcf_yield", "fcf_margin"])
    if _contains_any(lowered, ("估值", "p/e", "p / e", "p/s", "p / s", "市值", "股价", "valuation", "market cap", "share price")):
        metrics.extend(["pe_ratio", "ps_ratio", "fcf_yield", "market_cap", "share_price", "price", "adjusted_close"])
    return tuple(dict.fromkeys(normalize_metric_name(metric) for metric in metrics if metric))


def _suggested_refs_for_material_sentence(trace: Mapping[str, Any], sentence: str) -> list[str]:
    sentence_upper = sentence.upper()
    trace_companies = _companies(trace)
    mentioned = [company for company in trace_companies if company and company in sentence_upper]
    comparison_claim = _contains_any(sentence, ("占优", "更高", "更低", "领先", "leads", "higher", "lower", "versus", "对比"))
    companies = trace_companies if comparison_claim and trace_companies else (mentioned or trace_companies)
    company_set = {company.upper() for company in companies}
    metrics = set(_metrics_for_material_sentence(sentence))
    refs: list[str] = []
    if metrics:
        for row in _numeric_rows(trace):
            if not isinstance(row, dict):
                continue
            company = str(row.get("ticker") or row.get("company") or "").upper().strip()
            metric = normalize_metric_name(str(row.get("metric") or ""))
            ref = str(row.get("evidence_id") or "").strip()
            if company and company_set and company not in company_set:
                continue
            if metric in metrics and ref and ref not in refs:
                refs.append(ref)
            if len(refs) >= 4:
                return refs
    if _contains_any(sentence, ("风险", "竞争", "risk", "competition")):
        for row in _text_rows(trace):
            company = str(row.get("ticker") or row.get("company") or "").upper().strip()
            ref = str(row.get("evidence_id") or "").strip()
            if company and company_set and company not in company_set:
                continue
            if ref and ref not in refs:
                refs.append(ref)
            if len(refs) >= 4:
                break
    return refs


def _check_material_claim_citations(trace: Mapping[str, Any], answer: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for sentence in _material_sentences_without_citation(answer):
        suggested_refs = _suggested_refs_for_material_sentence(trace, sentence)
        item = {
            "type": "citation_free_material_claim",
            "message": "Material financial claim lacks citation.",
            "answer_span": sentence,
        }
        if suggested_refs:
            item["suggested_refs"] = suggested_refs
            item["suggested_fix"] = "".join(f"[{ref}]" for ref in suggested_refs)
        violations.append(item)
    return violations


def _skip_material_claim_citation_check(trace: Mapping[str, Any], answer: str) -> bool:
    answer_mode = str(trace.get("answer_mode") or _as_dict(trace.get("output")).get("answer_mode") or "")
    if answer_mode in {"meta", "clarification", "refusal_or_redirect"} or bool(trace.get("needs_clarification")):
        return True
    output = _as_dict(trace.get("output"))
    view = _as_dict(output.get("view"))
    if str(view.get("kind") or "") in {"meta_response", "clarification", "refusal_or_redirect"}:
        return True
    if not _numeric_rows(trace) and not _text_rows(trace):
        lowered = str(answer or "").lower()
        boundary_terms = (
            "证据不足",
            "无法",
            "不能",
            "不提供",
            "边界",
            "限制",
            "insufficient evidence",
            "cannot",
            "not provide",
            "boundary",
            "limited",
        )
        if _contains_any(lowered, boundary_terms):
            return True
    return False


def _required_caveats(trace: Mapping[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    rows = _numeric_rows(trace)
    caveats: list[tuple[str, tuple[str, ...]]] = []
    if any(str(row.get("source_provider") or "").lower() == "yfinance" for row in rows) or any(
        str(row.get("confidence") or "").lower() == "medium" for row in rows
    ):
        caveats.append(("medium_confidence_provider", ("yfinance", "medium", "可信度", "中等", "证据边界")))
    if any(str(row.get("reconciliation_warning") or "").strip() for row in rows):
        caveats.append(("reconciliation_warning", ("reconciliation", "warning", "差异", "口径", "不一致")))
    latest_by_metric: dict[str, dict[str, Any]] = {}
    for row in rows:
        metric = normalize_metric_name(str(row.get("metric") or ""))
        if not metric:
            continue
        current = latest_by_metric.get(metric)
        period = str(row.get("period_end") or row.get("period") or "")
        if current is None or period >= str(current.get("period_end") or current.get("period") or ""):
            latest_by_metric[metric] = row
    net_margin = _to_float(latest_by_metric.get("net_margin", {}).get("value"))
    if net_margin is not None and net_margin > 0.30:
        caveats.append(("unusually_high_net_margin", ("异常", "口径", "核验", "持续性", "unusual", "basis")))
    elif net_margin is not None and net_margin < 0:
        caveats.append(("negative_net_margin", ("为负", "亏损", "口径", "negative", "loss")))
    fcf_margin = _to_float(latest_by_metric.get("fcf_margin", {}).get("value"))
    if fcf_margin is not None and fcf_margin < 0:
        caveats.append(("negative_fcf_margin", ("自由现金流为负", "FCF margin", "资本开支", "压制", "negative", "capex")))
    capex_to_revenue = _to_float(latest_by_metric.get("capex_to_revenue", {}).get("value"))
    if capex_to_revenue is not None and capex_to_revenue > 0.30:
        caveats.append(("high_capex_intensity", ("资本开支", "资本强度", "capex", "capital intensity")))
    cfo_to_net_income = _to_float(latest_by_metric.get("cfo_to_net_income", {}).get("value"))
    if cfo_to_net_income is not None and cfo_to_net_income > 1.5:
        caveats.append(("high_cash_conversion_or_income_quality_mismatch", ("CFO/净利润", "口径", "错配", "cash conversion", "basis")))
    cfo = _to_float(latest_by_metric.get("operating_cash_flow", {}).get("value"))
    fcf = _to_float(latest_by_metric.get("free_cash_flow", {}).get("value"))
    capex = _to_float(latest_by_metric.get("capital_expenditure", {}).get("value"))
    if cfo is not None and fcf is not None and capex is not None and cfo > 0 and fcf <= 0 and abs(capex) >= abs(cfo) * 0.8:
        caveats.append(("capex_drag_on_fcf", ("自由现金流为负", "资本开支", "吞噬", "压制", "capex", "free cash flow")))
    for dimension_id, status in _status_map(trace).items():
        if _as_list(status.get("enhanced_missing")):
            caveats.append((f"enhanced_missing:{dimension_id}", ("缺少", "限制", "边界", "不足", "caveat", "limited")))
    trace_summary = _as_dict(trace.get("trace_summary"))
    if int(trace_summary.get("missing_optional_requirements_count") or 0) > 0:
        caveats.append(("optional_missing", ("缺少", "限制", "边界", "不足", "optional", "limited")))
    return caveats


def _specific_caveat_code(caveat_id: str) -> tuple[str, str, tuple[str, ...]]:
    normalized = str(caveat_id or "")
    if "medium_confidence" in normalized:
        return (
            "missing_medium_confidence_source_caveat",
            "evidence_specific_caveat",
            ("yfinance", "medium", "可信度", "中等", "证据边界"),
        )
    if "reconciliation" in normalized:
        return (
            "missing_reconciliation_caveat",
            "evidence_specific_caveat",
            ("差异", "口径", "不一致", "证据边界"),
        )
    if "enhanced_missing" in normalized or "optional_missing" in normalized:
        return (
            "missing_growth_quantification_caveat",
            "generic_boundary_caveat",
            ("不能量化", "当前无法可靠计算", "证据边界", "不足", "缺少", "限制"),
        )
    if any(token in normalized for token in ("margin", "fcf", "capex", "cash_conversion")):
        return (
            "missing_sustainability_caveat",
            "evidence_specific_caveat",
            ("不能判断是否可持续", "口径", "核验", "持续性", "证据边界"),
        )
    return (
        "missing_segment_scope_caveat",
        "generic_boundary_caveat",
        ("分部/产品层面", "不能完整代表总公司", "证据边界", "待验证假设"),
    )


def _check_caveat_visibility(trace: Mapping[str, Any], answer: str) -> tuple[list[dict[str, Any]], float]:
    required = _required_caveats(trace)
    if not required:
        return [], 1.0
    visible = 0
    violations: list[dict[str, Any]] = []
    for caveat_id, terms in required:
        code, caveat_type, fallback_terms = _specific_caveat_code(caveat_id)
        visibility_terms = tuple(dict.fromkeys([*terms, *fallback_terms]))
        if _contains_any(answer, list(visibility_terms)):
            visible += 1
        else:
            violations.append(
                {
                    "type": code,
                    "caveat_type": caveat_type,
                    "message": f"Caveat not visible in final answer: {caveat_id}",
                }
            )
    return violations, visible / len(required)


def _check_raw_leakage(answer: str) -> list[dict[str, Any]]:
    violations = []
    for pattern in _INTERNAL_PATTERNS:
        if re.search(pattern, answer):
            violations.append({"type": "raw_internal_leakage", "message": f"Internal diagnostic leaked: {pattern}"})
    return violations


def _claim_tier_rows(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(item) for item in _as_list(trace.get("analytical_claims")) if isinstance(item, dict)]
    if rows:
        return rows
    synthesis = _as_dict(trace.get("synthesis"))
    return [dict(item) for item in _as_list(synthesis.get("analytical_claims")) if isinstance(item, dict)]


def _has_hypothesis_marker(text: str) -> bool:
    return _contains_any(text, _HYPOTHESIS_MARKERS)


_INFERENCE_MARKERS = (
    "合理推断",
    "有限判断",
    "基于业务结构推断",
    "基于业务模型",
    "基于已验证事实",
    "citable inference",
    "reasonable inference",
    "limited judgment",
    "business-structure inference",
)


def _has_inference_marker(text: str) -> bool:
    return _contains_any(text, _INFERENCE_MARKERS)


def _has_strong_causal_wording(text: str) -> bool:
    return bool(_term_violations(text, _STRONG_CAUSAL_TERMS))


def _sentence_has_citation(sentence: str) -> bool:
    return bool(re.search(r"\[[NT]\d+\]", sentence or ""))


def _check_claim_tier_contract(trace: Mapping[str, Any], answer: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for claim in _claim_tier_rows(trace):
        raw_tier = claim.get("tier")
        tier = str(getattr(raw_tier, "value", raw_tier) or "")
        text = str(claim.get("text") or "").strip()
        citations = [str(item).strip() for item in _as_list(claim.get("citation_refs")) if str(item).strip()]
        support = [str(item).strip() for item in _as_list(claim.get("supporting_claim_ids")) if str(item).strip()]
        if tier == "evidence_backed" and not citations:
            violations.append(
                {
                    "type": "evidence_backed_claim_without_citation",
                    "message": "Evidence-backed claim must include citation_refs.",
                    "answer_span": text[:180],
                }
            )
        elif tier == "evidence_inferred":
            if not _has_inference_marker(text):
                violations.append(
                    {
                        "type": "inferred_claim_missing_marker",
                        "message": "Evidence-inferred claim must be explicitly marked as a reasonable or limited inference.",
                        "answer_span": text[:180],
                    }
                )
            if not citations and not support:
                violations.append(
                    {
                        "type": "inferred_claim_missing_support",
                        "message": "Evidence-inferred claim must cite or reference supporting evidence-backed claims.",
                        "answer_span": text[:180],
                    }
                )
            if _has_strong_causal_wording(text):
                violations.append(
                    {
                        "type": "strong_causal_in_inferred_claim",
                        "message": "Evidence-inferred claim uses definitive causal wording.",
                        "answer_span": text[:180],
                    }
                )
        elif tier == "hypothesis_to_verify" and not _has_hypothesis_marker(text):
            violations.append(
                {
                    "type": "hypothesis_missing_marker",
                    "message": "Hypothesis claim must be explicitly marked as a hypothesis or to-verify item.",
                    "answer_span": text[:180],
                }
            )
    for sentence in re.split(r"(?<=[。！？.!?；;])\s+|\n+", answer or ""):
        clean = sentence.strip(" -•\t")
        if not clean:
            continue
        if not _has_strong_causal_wording(clean):
            continue
        if _sentence_has_citation(clean) or _has_hypothesis_marker(clean):
            continue
        violations.append(
            {
                "type": "uncited_strong_causal_claim",
                "message": "Definitive causal wording lacks citation or hypothesis marker.",
                "answer_span": clean[:180],
            }
        )
    return violations


def _evidence_scope_by_ref(trace: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    scope_by_ref: dict[str, dict[str, Any]] = {}
    for row in _text_rows(trace):
        ref = str(row.get("citation_ref") or row.get("evidence_id") or "").strip()
        if not ref:
            continue
        driver_level = str(row.get("driver_level") or "").strip()
        claim_scope = str(row.get("claim_scope") or "").strip()
        if not claim_scope:
            claim_scope = {
                "company_level_driver": "company",
                "segment_level_driver": "segment",
                "product_level_driver": "product",
                "market_context": "market_context",
                "risk_context": "market_context",
            }.get(driver_level, "unknown")
        allowed_strength = str(row.get("allowed_claim_strength") or "").strip()
        if not allowed_strength:
            allowed_strength = "definitive" if claim_scope == "company" else ("hypothesis_only" if claim_scope == "market_context" else "bounded_inference")
        summary_warning = evidence_summary_scope_warning(row)
        scope_by_ref[ref] = {
            "evidence_id": ref,
            "driver_level": driver_level or "unknown",
            "driver_levels": list(row.get("driver_levels", []) or []),
            "claim_scope": claim_scope or "unknown",
            "allowed_claim_strength": allowed_strength,
            "scope_reason": str(row.get("scope_reason") or "").strip(),
            "summary_scope_warning": str(row.get("summary_scope_warning") or (summary_warning or {}).get("code") or ""),
            "evidence_summary_scope_overclaim": bool(row.get("evidence_summary_scope_overclaim") or summary_warning),
        }
    return scope_by_ref


def _check_evidence_summary_scope(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in _text_rows(trace):
        warning = evidence_summary_scope_warning(row)
        if not warning:
            continue
        key = (str(warning.get("citation_ref") or warning.get("evidence_id") or ""), str(warning.get("summary") or ""))
        if key in seen:
            continue
        seen.add(key)
        warnings.append(warning)
    return warnings


def _answer_sentences(answer: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?；;])\s+|\n+", str(answer or "").strip())
    return [part.strip(" -•\t") for part in parts if part.strip(" -•\t")]


def _citation_refs_in_sentence(sentence: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([NT]\d+)\]", sentence or "")))


def _matches_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text or "", flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _is_company_level_definitive_causal_claim(sentence: str) -> bool:
    clean = str(sentence or "").strip()
    if not clean:
        return False
    if _contains_any(clean, _BOUNDED_SCOPE_TERMS):
        return False
    return _matches_pattern(clean, _DEFINITIVE_CAUSAL_PATTERNS) and _matches_pattern(clean, _COMPANY_CAUSAL_TARGET_PATTERNS)


def _check_scope_overclaim(trace: Mapping[str, Any], answer: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scope_by_ref = _evidence_scope_by_ref(trace)
    evidence_summary_warnings = _check_evidence_summary_scope(trace)
    violations: list[dict[str, Any]] = []
    checked_claims = 0
    for sentence in _answer_sentences(answer):
        refs = _citation_refs_in_sentence(sentence)
        if not refs or not _is_company_level_definitive_causal_claim(sentence):
            continue
        checked_claims += 1
        scoped_refs = [ref for ref in refs if ref in scope_by_ref]
        if not scoped_refs:
            continue
        has_company_support = any(scope_by_ref[ref].get("claim_scope") == "company" for ref in scoped_refs)
        if has_company_support:
            continue
        affected = [
            ref
            for ref in scoped_refs
            if str(scope_by_ref[ref].get("claim_scope") or "") in {"segment", "product", "market_context", "unknown"}
        ]
        if not affected:
            continue
        violations.append(
            {
                "type": "segment_evidence_overstated_as_company_driver",
                "message": "Segment/product/context evidence was used as definitive company-level revenue-growth causality.",
                "answer_span": sentence[:220],
                "affected_citations": affected,
                "citation_scopes": {ref: scope_by_ref[ref] for ref in affected},
            }
        )
    status = "repairable" if violations else "passed"
    return violations, {
        "status": status,
        "checked_claims": checked_claims,
        "violations": violations,
        "evidence_summary_warnings": evidence_summary_warnings,
        "evidence_scope_by_ref": scope_by_ref,
    }


def _companies(trace: Mapping[str, Any]) -> list[str]:
    qu = _as_dict(trace.get("query_understanding_summary"))
    companies = [
        str(item.get("ticker") or "").upper().strip()
        for item in _as_list(qu.get("companies"))
        if isinstance(item, dict) and str(item.get("ticker") or "").strip()
    ]
    if companies:
        return companies
    out: list[str] = []
    for item in _as_list(trace.get("companies")):
        if isinstance(item, Mapping):
            text = str(item.get("ticker") or item.get("TICKER") or item.get("symbol") or item.get("company") or "").upper().strip()
        else:
            text = str(item or "").upper().strip()
        if text:
            out.append(text)
    return list(dict.fromkeys(out))


def _evidence_companies_for_metric(trace: Mapping[str, Any], metric: str) -> set[str]:
    target = normalize_metric_name(metric)
    return {
        str(row.get("ticker") or row.get("company") or "").upper().strip()
        for row in _numeric_rows(trace)
        if normalize_metric_name(str(row.get("metric") or "")) == target
        and str(row.get("ticker") or row.get("company") or "").strip()
    }


def _evidence_companies_for_any_metric(trace: Mapping[str, Any], metrics: tuple[str, ...]) -> set[str]:
    companies: set[str] = set()
    for metric in metrics:
        companies.update(_evidence_companies_for_metric(trace, metric))
    return companies


def _requested_comparison_dimensions(trace: Mapping[str, Any]) -> set[str]:
    dimensions: set[str] = set()
    sources = [
        trace.get("requested_dimensions"),
        dict(trace.get("canonical_intent", {}) or {}).get("requested_dimensions"),
        dict(trace.get("analysis_plan", {}) or {}).get("requested_dimensions"),
        dict(trace.get("evidence_packet", {}) or {}).get("requested_dimensions"),
    ]
    policy = dict(trace.get("evidence_policy", {}) or {})
    policy_id = str(policy.get("policy_id") or "")
    if policy_id == "comparison_dimension_specific_v1":
        sources.append(policy.get("required_dimensions"))
    for source in sources:
        for item in source or []:
            text = str(item or "").strip()
            if text:
                dimensions.add(text)
    return dimensions


def _risk_text_companies(trace: Mapping[str, Any]) -> set[str]:
    return {
        str(row.get("ticker") or row.get("company") or "").upper().strip()
        for row in _text_rows(trace)
        if (
            str(row.get("dimension_id") or "") == "moat_and_competitive_risk"
            or str(row.get("section") or "").upper().strip() in {"ITEM_1A", "ITEM_7", "ITEM_2"}
        )
        and str(row.get("ticker") or row.get("company") or "").strip()
    }


def _check_comparison_balance(trace: Mapping[str, Any], answer: str) -> tuple[list[dict[str, Any]], float]:
    companies = _companies(trace)
    if len(companies) < 2:
        return [], 1.0
    comparison_like = str(trace.get("analysis_scope") or "") == "comparison" or str(trace.get("task_type") or "") == "company_comparison"
    if not comparison_like:
        return [], 1.0
    if _contains_any(
        answer,
        (
            "无法明确比较",
            "不能强行判断",
            "证据不足",
            "证据边界",
            "证据限制",
            "cannot force a ranking",
            "cannot rank",
            "insufficient evidence",
            "evidence boundary",
            "evidence limits",
        ),
    ):
        return [], 1.0
    expected = set(companies[:2])
    requested_dimensions = _requested_comparison_dimensions(trace)

    def should_check(dimension_id: str) -> bool:
        return not requested_dimensions or dimension_id in requested_dimensions

    checks: list[tuple[str, bool]] = []
    if should_check("revenue_quality") and _contains_any(answer, ("收入", "revenue")):
        companies_with_revenue = _evidence_companies_for_metric(trace, "revenue") | _evidence_companies_for_metric(trace, "revenue_growth")
        checks.append(("revenue", expected.issubset(companies_with_revenue)))
    if should_check("profitability_quality") and _contains_any(answer, ("盈利", "净利润", "净利率", "profit", "margin")):
        companies_with_profit = _evidence_companies_for_metric(trace, "net_income") | _evidence_companies_for_metric(trace, "net_margin")
        checks.append(("profitability", expected.issubset(companies_with_profit)))
    if should_check("cash_flow_quality") and _contains_any(answer, ("现金流", "经营现金流", "自由现金流", "FCF", "cash flow", "free cash flow")):
        companies_with_cash_flow = _evidence_companies_for_any_metric(
            trace,
            ("operating_cash_flow", "free_cash_flow", "capital_expenditure", "fcf_margin"),
        )
        checks.append(("cash_flow", expected.issubset(companies_with_cash_flow)))
    if should_check("valuation_and_risk_boundary") and _contains_any(answer, ("估值", "市值", "股价", "P/E", "P/S", "FCF yield", "valuation", "market cap")):
        companies_with_valuation = _evidence_companies_for_any_metric(
            trace,
            ("pe_ratio", "ps_ratio", "fcf_yield", "market_cap", "share_price"),
        )
        checks.append(("valuation", expected.issubset(companies_with_valuation)))
    if should_check("moat_and_competitive_risk") and _contains_any(answer, ("风险", "竞争", "risk", "competitive")):
        checks.append(("risk", expected.issubset(_risk_text_companies(trace))))
    if not checks:
        return [], 1.0
    passed = sum(1 for _name, ok in checks if ok)
    violations = [
        {"type": "comparison_balance", "message": f"Comparison claim lacks bilateral evidence for {name}."}
        for name, ok in checks
        if not ok
    ]
    return violations, passed / len(checks)


def check_answer_evidence_contract(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return post-hoc validation result for a trace's user-visible answer.

    The returned dict is release/eval metadata only. The function does not
    repair, rewrite, or otherwise mutate the answer or trace.
    """
    answer = _answer_text(trace)
    violations: list[dict[str, Any]] = []
    numeric_violations, numeric_rate = _check_numeric_grounding(trace, answer)
    citation_violations, citation_rate = _check_citations(trace, answer)
    dimension_violations = _check_dimension_status(trace, answer)
    forbidden_violations = _check_forbidden_claims(answer)
    material_citation_violations = [] if _skip_material_claim_citation_check(trace, answer) else _check_material_claim_citations(trace, answer)
    caveat_violations, caveat_rate = _check_caveat_visibility(trace, answer)
    leakage_violations = _check_raw_leakage(answer)
    comparison_violations, comparison_rate = _check_comparison_balance(trace, answer)
    claim_tier_violations = _check_claim_tier_contract(trace, answer)
    scope_overclaim_violations, scope_overclaim_check = _check_scope_overclaim(trace, answer)
    format_violations = _check_format_constraints(trace, answer)
    benchmark_violations = _check_unsupported_benchmark_claims(trace, answer)
    company_token_violations = _check_company_specific_token_leakage(trace, answer)
    risk_ranking_violations = _check_risk_ranking_support(trace)
    risk_support_counts = _risk_ranking_support_counts(trace)
    evidence_summary_warnings = list(scope_overclaim_check.get("evidence_summary_warnings", []) or [])
    violations.extend(numeric_violations)
    violations.extend(citation_violations)
    violations.extend(dimension_violations)
    violations.extend(forbidden_violations)
    violations.extend(material_citation_violations)
    violations.extend(caveat_violations)
    violations.extend(leakage_violations)
    violations.extend(comparison_violations)
    violations.extend(claim_tier_violations)
    violations.extend(scope_overclaim_violations)
    violations.extend(format_violations)
    violations.extend(benchmark_violations)
    violations.extend(company_token_violations)
    violations.extend(risk_ranking_violations)
    metrics = {
        "numeric_grounding_rate": round(float(numeric_rate), 4),
        "citation_validity_rate": round(float(citation_rate), 4),
        "dimension_status_violations": len(dimension_violations),
        "forbidden_claim_violations": len(forbidden_violations),
        "citation_free_material_claims": len(material_citation_violations),
        "caveat_visibility_rate": round(float(caveat_rate), 4),
        "comparison_balance_rate": round(float(comparison_rate), 4),
        "raw_internal_leakage_count": len(leakage_violations),
        "claim_tier_violations": len(claim_tier_violations),
        "scope_overclaim_violations": len(scope_overclaim_violations),
        "format_constraint_violations": len(format_violations),
        "unsupported_benchmark_claims": len(benchmark_violations),
        "company_specific_token_leakage": len(company_token_violations),
        "risk_ranking_support_violations": len(risk_ranking_violations),
        **risk_support_counts,
        "evidence_summary_scope_warnings": len(evidence_summary_warnings),
    }
    return {
        "passed": not violations,
        "violations": violations,
        "warnings": evidence_summary_warnings,
        "metrics": metrics,
        "scope_overclaim_check": scope_overclaim_check,
        "scope_overclaim_violations": scope_overclaim_violations,
        "evidence_scope_by_ref": scope_overclaim_check.get("evidence_scope_by_ref", {}),
    }


def _runtime_trace(answer: str, state: Mapping[str, Any]) -> dict[str, Any]:
    """Build the trace-shaped payload expected by the existing check helpers."""
    trace = dict(state)
    trace["final_answer"] = str(answer or "")
    output = _as_dict(trace.get("output"))
    if output:
        trace["output"] = dict(output)
        trace["output"]["answer"] = str(answer or "")
        trace["output"]["summary"] = str(answer or output.get("summary") or "")
    if not trace.get("evidence_packet") and isinstance(output.get("evidence_packet"), dict):
        trace["evidence_packet"] = output.get("evidence_packet")
    return trace


def _runtime_status_map(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return _status_map(_runtime_trace(str(state.get("final_answer") or ""), state))


def _runtime_missing_requirements(state: Mapping[str, Any], *, blocking_only: bool = False) -> list[str]:
    missing: list[str] = []
    for source in (
        state.get("missing_requirements"),
        _as_dict(state.get("evidence_sufficiency")).get("missing_requirements"),
    ):
        missing.extend(str(item) for item in _as_list(source) if str(item).strip())
    if blocking_only:
        explicit_missing_required: list[str] = []
        for source in (
            state.get("missing_required_requirements"),
            _as_dict(state.get("evidence_sufficiency_summary")).get("missing_required_requirements"),
            _as_dict(state.get("trace_summary")).get("missing_required_requirements"),
        ):
            explicit_missing_required.extend(str(item) for item in _as_list(source) if str(item).strip())
        if explicit_missing_required:
            return list(dict.fromkeys(explicit_missing_required))
    requirement_status = state.get("requirement_status_map")
    if isinstance(requirement_status, dict):
        if blocking_only:
            out: list[str] = []
            for rid, item in requirement_status.items():
                if not isinstance(item, dict):
                    continue
                if str(item.get("status") or "") != "missing":
                    continue
                scope = str(item.get("requirement_scope") or "core")
                if bool(item.get("required", True)) and scope == "core":
                    out.append(str(rid))
            return list(dict.fromkeys(out))
        for rid, item in requirement_status.items():
            if isinstance(item, dict) and str(item.get("status") or "") == "missing":
                missing.append(str(rid))
    return list(dict.fromkeys(missing))


def _violation_severity(violation: Mapping[str, Any], state: Mapping[str, Any]) -> tuple[str, str, str | None]:
    code = str(violation.get("type") or violation.get("code") or "contract_violation")
    if code == "format_constraint_violation":
        return "repairable", "repair_answer", None
    if code == "unsupported_benchmark_claim":
        return "repairable", "repair_answer", None
    if code == "unsupported_numeric" and str(violation.get("suggested_value") or "").strip():
        return "repairable", "repair_answer", None
    if code == "citation_free_material_claim":
        return "repairable", "repair_answer", None
    if code in {"invalid_citation", "unsupported_numeric", "forbidden_claim", "raw_internal_leakage", "evidence_backed_claim_without_citation"}:
        return "hard_fail", "blocked", None
    if code in {"inferred_claim_missing_marker", "inferred_claim_missing_support", "strong_causal_in_inferred_claim", "hypothesis_missing_marker", "uncited_strong_causal_claim"}:
        return "repairable", "repair_answer", None
    if code == "dimension_status_violation":
        dimension_id = str(violation.get("dimension_id") or "").strip() or None
        if dimension_id:
            status = str(_runtime_status_map(state).get(dimension_id, {}).get("status") or "")
            if status == "missing" and _runtime_missing_requirements(state, blocking_only=True):
                return "evidence_missing", "need_more_evidence", dimension_id
            if status == "partial":
                return "repairable", "repair_answer", dimension_id
        return "warning", "pass", dimension_id
    if code in {
        "caveat_not_visible",
        "missing_medium_confidence_source_caveat",
        "missing_growth_quantification_caveat",
        "missing_segment_scope_caveat",
        "missing_sustainability_caveat",
        "missing_reconciliation_caveat",
        "evidence_summary_scope_overclaim",
    }:
        return "warning", "pass", str(violation.get("dimension_id") or "").strip() or None
    if code == "comparison_balance":
        return "repairable", "repair_answer", str(violation.get("dimension_id") or "").strip() or None
    return "repairable", "repair_answer", str(violation.get("dimension_id") or "").strip() or None


def _runtime_route(violations: list[ContractViolation]) -> tuple[str, str, bool]:
    if not violations:
        return "pass", "pass", True
    severities = [item.severity for item in violations]
    if "hard_fail" in severities:
        return "hard_fail", "blocked", False
    if "evidence_missing" in severities:
        return "evidence_missing", "need_more_evidence", False
    if "repairable" in severities:
        return "repairable", "repair_answer", False
    return "warning", "pass", True


def _runtime_decision(severity: str) -> str:
    if severity == "hard_fail" or severity == "evidence_missing":
        return "blocked"
    if severity == "repairable":
        return "repairable"
    if severity == "warning":
        return "warning"
    return "passed"


def _runtime_public_summary(result: ContractResult) -> str:
    if result.decision == "warning":
        return "Answer passed runtime grounding checks with warnings or optional evidence gaps."
    if result.route == "pass":
        return "Answer passed runtime grounding checks."
    if result.route == "repair_answer":
        return "Answer required wording, citation, balance, or caveat repair before release."
    if result.route == "need_more_evidence":
        return "Answer required more evidence before release."
    return "Answer was blocked because it did not satisfy evidence or safety requirements."


def _runtime_action(violations: list[ContractViolation], route: str) -> str:
    if route == "pass":
        return "pass"
    if route == "need_more_evidence":
        return "retry_evidence"
    if route == "blocked":
        return "block"
    codes = {item.code for item in violations}
    if "format_constraint_violation" in codes:
        return "scope_limit"
    if "unsupported_benchmark_claim" in codes:
        return "scope_limit"
    if "citation_free_material_claim" in codes:
        return "add_citation" if any(item.suggested_fix for item in violations if item.code == "citation_free_material_claim") else "strip_sentence"
    if "dimension_status_violation" in codes or "comparison_balance" in codes:
        return "downgrade_to_bounded"
    if "segment_evidence_overstated_as_company_driver" in codes:
        return "scope_limit"
    return "scope_limit"


def _runtime_suggested_repair(violations: list[ContractViolation]) -> str | None:
    if not violations:
        return None
    fixes = [str(item.suggested_fix or "").strip() for item in violations if str(item.suggested_fix or "").strip()]
    if fixes:
        return " ".join(dict.fromkeys(fixes))
    codes = {item.code for item in violations}
    caveat_codes = {
        "caveat_not_visible",
        "missing_medium_confidence_source_caveat",
        "missing_growth_quantification_caveat",
        "missing_segment_scope_caveat",
        "missing_sustainability_caveat",
        "missing_reconciliation_caveat",
    }
    if codes & caveat_codes:
        return "Add the specific missing caveat: source confidence, growth quantification, segment scope, sustainability, or reconciliation boundary."
    if "comparison_balance" in codes:
        return "Bound comparison language and disclose asymmetric evidence coverage."
    if "format_constraint_violation" in codes:
        return "Compress the answer to the requested sentence limit without adding new analysis."
    if "unsupported_benchmark_claim" in codes:
        return "Rewrite benchmark language to say valuation multiples are high, but historical or peer benchmarks are missing."
    if "segment_evidence_overstated_as_company_driver" in codes:
        return (
            "The answer overstates segment/product-level evidence as company-level causality. "
            "Keep the citations but downgrade the affected claims with wording such as 分部层面显示, 业务线索指向, 可能参与, "
            "and 不能完整代表总公司营收增长原因."
        )
    if "dimension_status_violation" in codes:
        return "Use bounded wording for partial dimensions and avoid conclusions for missing dimensions."
    return "Revise the answer to stay inside validated evidence and visible limitations."


def check_answer_contract(
    answer: str,
    state: Mapping[str, Any],
    *,
    scope: str = "answer",
) -> ContractResult:
    """Runtime-safe AnswerContract guard.

    This deterministic adapter reuses the post-hoc check helpers against the
    current LangGraph state while returning a route-oriented result for live
    graph control. It does not mutate state and does not call an LLM.
    """
    trace = _runtime_trace(answer, state)
    raw = check_answer_evidence_contract(trace)
    violations: list[ContractViolation] = []
    output_language = str(state.get("output_language") or _as_dict(state.get("canonical_intent")).get("output_language") or detect_output_language(str(state.get("user_query") or "")))
    leaked_terms = language_leakage_terms(answer, output_language)
    for term in leaked_terms:
        violations.append(
            ContractViolation(
                code="language_leakage",
                message=f"English output contains Chinese user-visible text: {term}",
                severity="repairable",
                answer_span=term,
                repair_instruction="Translate or localize the leaked Chinese label without adding new facts.",
            )
        )
    for item in _as_list(raw.get("violations")):
        if not isinstance(item, dict):
            continue
        severity, route, dimension_id = _violation_severity(item, state)
        code = str(item.get("type") or item.get("code") or "contract_violation")
        message = str(item.get("message") or code)
        suggested_fix = None
        public_replacement = None
        repair_instruction = None
        if code == "segment_evidence_overstated_as_company_driver":
            repair_instruction = (
                "Keep the same citations and rewrite segment/product evidence as bounded business-line support, "
                "not definitive total-company revenue-growth causality."
            )
        elif code == "unsupported_benchmark_claim":
            public_replacement = "估值倍数较高，但缺少历史/行业基准，不能严格判断是否处于高位。"
            repair_instruction = "Replace the unsupported benchmark judgment with the public bounded valuation sentence."
        elif code == "unsupported_numeric" and str(item.get("suggested_value") or "").strip():
            public_replacement = str(item.get("suggested_value") or "").strip()
        elif code == "citation_free_material_claim" and str(item.get("suggested_fix") or "").strip():
            suggested_fix = str(item.get("suggested_fix") or "").strip()
        elif route == "repair_answer":
            repair_instruction = "Rewrite only the affected wording and keep all facts inside existing evidence."
        elif route == "need_more_evidence":
            repair_instruction = "Collect additional evidence for the missing dimension before making this claim."
        elif route == "blocked":
            repair_instruction = "Do not return the unsupported draft answer."
        violations.append(
            ContractViolation(
                code=code,
                message=message,
                severity=severity,  # type: ignore[arg-type]
                evidence_id=str(item.get("evidence_id") or item.get("citation_ref") or "").strip() or None,
                affected_citations=[str(ref) for ref in _as_list(item.get("affected_citations")) if str(ref).strip()],
                answer_span=str(item.get("answer_span") or item.get("value") or item.get("term") or "").strip() or None,
                suggested_fix=suggested_fix,
                public_replacement=public_replacement,
                repair_instruction=repair_instruction,
                dimension_id=dimension_id,
            )
        )
    extra_warnings: list[ContractViolation] = []
    for item in _as_list(raw.get("warnings")):
        if not isinstance(item, Mapping):
            continue
        code = str(item.get("type") or item.get("code") or "contract_warning")
        extra_warnings.append(
            ContractViolation(
                code=code,
                message=str(item.get("message") or code),
                severity="warning",
                evidence_id=str(item.get("evidence_id") or item.get("citation_ref") or "").strip() or None,
                affected_citations=[str(ref) for ref in _as_list(item.get("affected_citations")) if str(ref).strip()],
                answer_span=str(item.get("summary") or item.get("answer_span") or "").strip() or None,
                suggested_fix=str(item.get("suggested_summary") or item.get("suggested_fix") or "").strip() or None,
                dimension_id=str(item.get("dimension_id") or "").strip() or None,
            )
        )
    severity, route, passed = _runtime_route(violations)
    decision = _runtime_decision(severity)
    warnings = [item for item in violations if item.severity == "warning"] + extra_warnings
    blocking_violations = [item for item in violations if item.severity != "warning"]
    result = ContractResult(
        passed=passed,
        severity=severity,  # type: ignore[arg-type]
        decision=decision,  # type: ignore[arg-type]
        route=route,  # type: ignore[arg-type]
        action=_runtime_action(violations, route),  # type: ignore[arg-type]
        violations=blocking_violations,
        suggested_repair=_runtime_suggested_repair(violations),
        missing_requirements=_runtime_missing_requirements(state),
        blocking_missing_requirements=_runtime_missing_requirements(state, blocking_only=True),
        warnings=warnings,
        public_summary=None,
        scope_overclaim_check=dict(raw.get("scope_overclaim_check", {}) or {}),
        scope_overclaim_violations=list(raw.get("scope_overclaim_violations", []) or []),
        metrics=dict(raw.get("metrics", {}) or {}),
    )
    result.public_summary = _runtime_public_summary(result)
    return result
