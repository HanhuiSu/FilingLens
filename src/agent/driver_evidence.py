"""Deterministic driver-evidence tagging for causal revenue questions."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.agent.types import AllowedClaimStrength, ClaimScope, DriverLevel


COMPANY_LEVEL_TERMS = (
    "overall revenue",
    "total revenue",
    "company revenue",
    "consolidated revenue",
    "total net revenue",
    "nvidia's revenue growth",
    "revenue growth was driven",
    "strong year-on-year and sequential growth",
    "year-on-year and sequential growth",
    "total net sales",
    "net sales increased",
    "revenue increased",
)

SEGMENT_LEVEL_TERMS = (
    "data center",
    "networking",
    "professional visualization",
    "gaming",
    "automotive",
    "oem",
    "segment",
)

PRODUCT_LEVEL_TERMS = (
    "blackwell",
    "nvlink",
    "infiniband",
    "dgx spark",
    "gb200",
    "gb300",
    "ethernet",
    "xdr",
)

RISK_CONTEXT_TERMS = ("risk", "uncertainty", "竞争", "风险", "不确定")
MARKET_CONTEXT_TERMS = ("demand", "ai", "market", "customer", "需求", "客户", "市场")

COMPANY_DRIVER_PATTERNS = (
    r"\b(?:total|overall|company|consolidated)\s+(?:net\s+)?(?:revenue|sales).{0,90}\b(?:driven|due to|because|from)\b",
    r"\b(?:total|overall|company|consolidated)\s+(?:net\s+)?(?:revenue|sales).{0,90}\b(?:increased|grew).{0,90}\b(?:driven|due to|because|from)\b",
    r"\bnvidia'?s\s+(?:revenue|revenue growth|net sales).{0,90}\b(?:driven|due to|because|from)\b",
    r"\bstrong\s+year[-\s]on[-\s]year\s+and\s+sequential\s+growth\s+was\s+driven\b",
)

SEGMENT_DRIVER_PATTERNS = (
    r"\b(?:compute\s*&\s*networking|data\s+center|networking|graphics|gaming|professional visualization|automotive|oem).{0,90}\b(?:revenue|growth|increase|increased|grew).{0,90}\b(?:driven|due to|because|from)\b",
    r"\b(?:compute\s*&\s*networking|data\s+center|networking|graphics|gaming|professional visualization|automotive|oem)\s+revenue\b",
    r"\b(?:segment|reportable segment)\b",
)

PRODUCT_DRIVER_PATTERNS = (
    r"\b(?:blackwell|hopper|nvlink|infiniband|dgx|gb200|gb300|ethernet for ai|xdr)\b",
)

MARKET_CONTEXT_PATTERNS = (
    r"\b(?:ai demand|cloud capex|accelerated computing trend|market demand|customer demand)\b",
)

SUMMARY_COMPANY_OVERCLAIM_PATTERNS = (
    r"(?:NVIDIA|NVDA|公司|总|整体).{0,18}(?:营收|收入).{0,20}(?:增长|增加).{0,24}(?:主要由|由|得益于|归因于|驱动|推动)",
    r"(?:营收|收入).{0,20}(?:增长|增加).{0,24}(?:主要由|由|得益于|归因于).{0,80}(?:驱动|推动)",
    r"\b(?:nvidia|nvda|company|total|overall|consolidated).{0,25}\b(?:revenue|sales).{0,25}\b(?:growth|increase).{0,35}\b(?:driven by|due to|because of|attributable to)\b",
)

SUMMARY_BOUNDED_TERMS = (
    "分部层面",
    "产品层面",
    "业务线索",
    "线索",
    "可能",
    "补充证据",
    "不能完整代表",
    "不能直接证明",
    "segment-level",
    "product-level",
    "suggests",
    "points to",
    "may",
    "possible",
    "cannot fully represent",
)

PROFIT_DECLINE_FORBIDDEN_SUMMARY_PATTERNS = (
    r"利润下降可能(?:源于|与)",
    r"profit decline may be (?:influenced by|due to)",
)

PROFIT_DECLINE_SOURCE_PATTERNS = (
    r"(?:net income|profit|profits|operating income|earnings).{0,80}(?:decreased|declined|fell|lower|down)",
    r"(?:decrease|decline|fell|lower|down).{0,80}(?:net income|profit|profits|operating income|earnings)",
    r"(?:净利润|利润|营业利润|盈利).{0,20}(?:下降|下滑|减少|降低)",
    r"(?:下降|下滑|减少|降低).{0,20}(?:净利润|利润|营业利润|盈利)",
)

PROFIT_DECLINE_CONTEXT_TERMS = (
    "利润下降",
    "净利润下降",
    "盈利下降",
    "profit decline",
    "profit declined",
    "earnings decline",
)

PROFIT_DECLINE_NEUTRAL_SUMMARY = "披露文本涉及收入/收益波动或收入确认口径，可作为利润波动相关背景，但不能证明本期利润下降原因。"

_LEVEL_TO_SCOPE: dict[str, ClaimScope] = {
    "company_level_driver": "company",
    "segment_level_driver": "segment",
    "product_level_driver": "product",
    "market_context": "market_context",
    "risk_context": "market_context",
    "unknown": "unknown",
}

_SCOPE_TO_STRENGTH: dict[str, AllowedClaimStrength] = {
    "company": "definitive",
    "segment": "bounded_inference",
    "product": "bounded_inference",
    "market_context": "hypothesis_only",
    "unknown": "bounded_inference",
}


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def _clean_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).strip(" -;；。.")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(term.lower() in lowered for term in terms)


def _is_profit_decline_context(text: str) -> bool:
    return _contains_any(text, PROFIT_DECLINE_CONTEXT_TERMS)


def _explicit_profit_decline_source(text: str) -> bool:
    return _matches_any(str(text or ""), PROFIT_DECLINE_SOURCE_PATTERNS)


def _profit_decline_summary_overclaims(item: Mapping[str, Any], user_query: str = "") -> bool:
    context = " ".join(
        str(value or "")
        for value in (
            user_query,
            item.get("retrieval_query"),
            item.get("query"),
            item.get("purpose"),
            item.get("claim"),
            item.get("evidence_summary"),
            item.get("theme_name"),
        )
        if str(value or "").strip()
    )
    if not _is_profit_decline_context(context):
        return False
    summary = _claim_summary_text(item)
    source = source_text_for_scope_classification(item)
    forbidden_summary = _matches_any(summary, PROFIT_DECLINE_FORBIDDEN_SUMMARY_PATTERNS)
    causal_summary = _is_profit_decline_context(summary) and _contains_any(summary, ("源于", "与", "due to", "influenced by"))
    return (forbidden_summary or causal_summary) and not _explicit_profit_decline_source(source)


def source_text_for_scope_classification(item: Mapping[str, Any]) -> str:
    """Use source snippets for scope; generated claims can overstate scope."""
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "supporting_snippet",
            "text_snippet",
            "snippet",
            "evidence_summary",
            "theme_name",
            "section",
        )
        if str(item.get(key) or "").strip()
    )


def classify_driver_levels(text: str) -> list[str]:
    lowered = str(text or "").lower()
    levels: list[str] = []
    if _matches_any(lowered, COMPANY_DRIVER_PATTERNS) or any(term in lowered for term in COMPANY_LEVEL_TERMS):
        levels.append("company_level_driver")
    if _matches_any(lowered, SEGMENT_DRIVER_PATTERNS) or any(term in lowered for term in SEGMENT_LEVEL_TERMS):
        levels.append("segment_level_driver")
    if _matches_any(lowered, PRODUCT_DRIVER_PATTERNS) or any(term in lowered for term in PRODUCT_LEVEL_TERMS):
        levels.append("product_level_driver")
    if not levels and any(term in lowered for term in RISK_CONTEXT_TERMS):
        levels.append("market_context")
    if not levels and (_matches_any(lowered, MARKET_CONTEXT_PATTERNS) or any(term in lowered for term in MARKET_CONTEXT_TERMS)):
        levels.append("market_context")
    return levels or ["unknown"]


def primary_driver_level(levels: list[str]) -> str:
    for level in ("company_level_driver", "segment_level_driver", "product_level_driver", "market_context", "unknown"):
        if level in levels:
            return level
    return levels[0] if levels else "unknown"


def classify_evidence_scope(text: str) -> dict[str, Any]:
    lowered = str(text or "").lower()
    levels = classify_driver_levels(lowered)
    has_company_driver = _matches_any(lowered, COMPANY_DRIVER_PATTERNS)
    has_segment_driver = _matches_any(lowered, SEGMENT_DRIVER_PATTERNS) or any(term in lowered for term in SEGMENT_LEVEL_TERMS)
    has_product_driver = _matches_any(lowered, PRODUCT_DRIVER_PATTERNS) or any(term in lowered for term in PRODUCT_LEVEL_TERMS)
    has_market_context = _matches_any(lowered, MARKET_CONTEXT_PATTERNS) or any(term in lowered for term in MARKET_CONTEXT_TERMS)

    if has_product_driver:
        driver_level: DriverLevel = "product_level_driver"
        reason = "Snippet references product-level driver terms."
    elif has_segment_driver:
        driver_level = "segment_level_driver"
        reason = "Snippet discusses segment or business-line revenue, not consolidated company revenue."
    elif has_company_driver:
        driver_level = "company_level_driver"
        reason = "Snippet explicitly ties total/company revenue growth to a driver."
    elif has_market_context:
        driver_level = "market_context"
        reason = "Snippet provides market or demand context without direct company-level attribution."
    else:
        driver_level = "unknown"
        reason = "Scope could not be determined from source text."

    # Keep backward-compatible levels while forcing scope to the conservative primary level.
    if driver_level not in levels:
        levels.append(driver_level)
    claim_scope: ClaimScope = _LEVEL_TO_SCOPE.get(driver_level, "unknown")
    allowed_strength: AllowedClaimStrength = _SCOPE_TO_STRENGTH.get(claim_scope, "bounded_inference")
    return {
        "driver_level": driver_level,
        "driver_levels": levels,
        "claim_scope": claim_scope,
        "allowed_claim_strength": allowed_strength,
        "scope_reason": reason,
    }


def text_for_driver_classification(item: Mapping[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "claim",
            "supporting_snippet",
            "snippet",
            "text_snippet",
            "evidence_summary",
            "theme_name",
            "section",
        )
        if str(item.get(key) or "").strip()
    )


def annotate_driver_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(item)
    scope = classify_evidence_scope(source_text_for_scope_classification(row))
    row["driver_levels"] = list(scope.get("driver_levels", []) or [])
    row["driver_level"] = str(scope.get("driver_level") or primary_driver_level(row["driver_levels"]))
    row["claim_scope"] = str(scope.get("claim_scope") or "unknown")
    row["allowed_claim_strength"] = str(scope.get("allowed_claim_strength") or "bounded_inference")
    row["scope_reason"] = str(scope.get("scope_reason") or "")
    return row


def _claim_summary_text(item: Mapping[str, Any]) -> str:
    for key in ("claim", "evidence_summary", "theme_name", "source_title"):
        value = _clean_summary_text(str(item.get(key) or ""))
        if value:
            return value
    return ""


def _summary_overstates_company_driver(item: Mapping[str, Any], summary: str | None = None) -> bool:
    claim_scope = str(item.get("claim_scope") or "").strip()
    if claim_scope not in {"segment", "product", "market_context", "unknown"}:
        return False
    text = str(summary if summary is not None else _claim_summary_text(item)).strip()
    if not text or _contains_any(text, SUMMARY_BOUNDED_TERMS):
        return False
    return _matches_any(text, SUMMARY_COMPANY_OVERCLAIM_PATTERNS)


def _known_driver_phrase(source_text: str) -> str:
    lowered = str(source_text or "").lower()
    product_terms: list[str] = []
    if "xdr" in lowered or "infiniband" in lowered:
        product_terms.append("XDR InfiniBand")
    if "nvlink" in lowered:
        product_terms.append("NVLink")
    if "ethernet for ai" in lowered or "ethernet" in lowered:
        product_terms.append("Ethernet for AI solutions")
    if "blackwell" in lowered:
        product_terms.append("Blackwell")
    if "hopper" in lowered:
        product_terms.append("Hopper")
    if "gb200" in lowered:
        product_terms.append("GB200")
    if "gb300" in lowered:
        product_terms.append("GB300")
    if product_terms:
        if len(product_terms) == 1:
            return product_terms[0]
        return "、".join(product_terms[:-1]) + " 和 " + product_terms[-1]
    if "accelerated computing" in lowered and (" ai" in f" {lowered}" or "artificial intelligence" in lowered):
        return "accelerated computing 和 AI 平台转型"
    if "accelerated computing" in lowered:
        return "accelerated computing"
    return ""


def _extract_driver_phrase(source_text: str) -> str:
    known = _known_driver_phrase(source_text)
    if known:
        return known
    source = _clean_summary_text(source_text)
    patterns = (
        r"(?:driven by|due to|because of|attributable to|from)\s+(.{4,160}?)(?:\.|;|, while|, and|$)",
        r"(?:由|得益于|归因于)(.{2,120}?)(?:驱动|推动|导致|。|；|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return _clean_summary_text(match.group(1))
    return ""


def _scope_subject(source_text: str, claim_scope: str) -> str:
    lowered = str(source_text or "").lower()
    if claim_scope == "segment":
        if "compute & networking" in lowered or "compute and networking" in lowered:
            return "Compute & Networking 分部收入增长"
        if "data center" in lowered:
            return "Data Center 分部收入增长"
        if "professional visualization" in lowered:
            return "Professional Visualization 分部收入增长"
        if "networking" in lowered:
            return "Networking 业务收入增长"
        if "graphics" in lowered:
            return "Graphics 分部收入增长"
        if "gaming" in lowered:
            return "Gaming 分部收入增长"
        if "automotive" in lowered:
            return "Automotive 分部收入增长"
        return "相关分部收入增长"
    if claim_scope == "product":
        if "networking" in lowered:
            return "网络产品/业务增长"
        return "产品层面增长"
    if claim_scope == "market_context":
        return "市场/需求背景"
    return "该文本"


def scope_aware_evidence_summary(item: Mapping[str, Any]) -> str:
    """Render evidence summary at the maximum scope supported by the snippet."""
    row = annotate_driver_evidence(item)
    claim_scope = str(row.get("claim_scope") or "unknown")
    source_text = source_text_for_scope_classification(row)
    driver_phrase = _extract_driver_phrase(source_text)
    subject = _scope_subject(source_text, claim_scope)
    if claim_scope == "segment":
        if driver_phrase:
            return f"{subject}与 {driver_phrase} 相关。"
        return f"{subject}提供分部层面的增长线索，不能直接代表总公司营收增长原因。"
    if claim_scope == "product":
        if driver_phrase:
            return f"{subject}与 {driver_phrase} 相关，可作为产品层面的增长线索。"
        return f"{subject}提供产品层面的增长线索，不能直接代表总公司营收增长原因。"
    if claim_scope == "market_context":
        return "该文本提供市场或需求背景，只能作为待验证假设或分析背景。"
    if driver_phrase:
        return f"该文本显示与 {driver_phrase} 相关，但证据范围不足以支持公司级强因果。"
    return "该文本的证据范围不明确，不能作为公司级强因果结论。"


def evidence_summary_scope_warning(item: Mapping[str, Any]) -> dict[str, Any] | None:
    row = annotate_driver_evidence(item)
    summary = _claim_summary_text(row)
    if not _summary_overstates_company_driver(row, summary):
        return None
    ref = str(row.get("citation_ref") or row.get("evidence_id") or "").strip()
    return {
        "type": "evidence_summary_scope_overclaim",
        "code": "evidence_summary_scope_overclaim",
        "message": "Text evidence summary overstates segment/product/context evidence as company-level causality.",
        "evidence_id": ref or None,
        "citation_ref": ref or None,
        "claim_scope": str(row.get("claim_scope") or "unknown"),
        "allowed_claim_strength": str(row.get("allowed_claim_strength") or ""),
        "summary": summary[:220],
        "suggested_summary": scope_aware_evidence_summary(row),
    }


def apply_profit_decline_summary_neutralization(item: Mapping[str, Any], user_query: str = "") -> dict[str, Any]:
    row = dict(item)
    if not _profit_decline_summary_overclaims(row, user_query=user_query):
        return row
    original = _claim_summary_text(row)
    row["original_claim"] = str(row.get("original_claim") or original)
    row["claim"] = PROFIT_DECLINE_NEUTRAL_SUMMARY
    row["claim_source"] = "profit_decline_neutral_summary"
    row["profit_decline_summary_neutralized"] = True
    row["summary_scope_warning"] = "profit_decline_summary_neutralized"
    return row


def apply_scope_aware_summary(item: Mapping[str, Any], user_query: str = "") -> dict[str, Any]:
    """Downgrade generated evidence summaries that overstate the source scope."""
    row = annotate_driver_evidence(item)
    row = apply_profit_decline_summary_neutralization(row, user_query=user_query)
    if row.get("profit_decline_summary_neutralized"):
        return row
    warning = evidence_summary_scope_warning(row)
    if not warning:
        return row
    original = _claim_summary_text(row)
    row["original_claim"] = original
    row["claim"] = str(warning.get("suggested_summary") or scope_aware_evidence_summary(row))
    row["claim_source"] = "scope_aware_summary"
    row["evidence_summary_scope_overclaim"] = True
    row["summary_scope_warning"] = "evidence_summary_scope_overclaim"
    return row
