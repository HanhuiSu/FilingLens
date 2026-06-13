"""Claim-tier analytical reasoning helpers for evidence-grounded answers."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.agent.metric_availability import normalize_metric_name
from src.agent.metric_display import format_metric_value, metric_display_name
from src.agent.types import AnalyticalClaim, ClaimTier


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(text or "")))


def citation_for_item(item: Mapping[str, Any]) -> str:
    ref = str(item.get("citation_ref") or item.get("evidence_id") or "").strip()
    if not ref:
        return ""
    return ref[1:-1] if ref.startswith("[") and ref.endswith("]") else ref


def bracket_ref(ref: str) -> str:
    clean = str(ref or "").strip()
    if not clean:
        return ""
    return clean if clean.startswith("[") else f"[{clean}]"


def claim_tier_counts(claims: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {tier.value: 0 for tier in ClaimTier}
    for claim in claims:
        raw_tier = claim.get("tier")
        tier = str(getattr(raw_tier, "value", raw_tier) or "")
        if tier in counts:
            counts[tier] += 1
    return counts


def _format_numeric_fact(card: Mapping[str, Any], lang: str) -> str:
    ticker = str(card.get("ticker") or card.get("company") or "").upper().strip()
    metric = str(card.get("metric") or "").strip()
    period = str(card.get("period_end") or card.get("period") or "").strip()
    value = str(card.get("display_value") or card.get("formatted_value") or card.get("value") or "").strip()
    ref = bracket_ref(citation_for_item(card))
    if lang == "zh":
        metric_label = metric_display_name(metric, "zh")
        raw_value = card.get("value")
        if normalize_metric_name(metric) == "revenue" and raw_value is not None:
            try:
                value = f"{float(raw_value) / 100_000_000:.1f} 亿美元"
            except (TypeError, ValueError):
                value = value or format_metric_value(metric, raw_value, unit=card.get("unit"))
        elif raw_value is not None and not value:
            value = format_metric_value(metric, raw_value, unit=card.get("unit"))
        if period:
            return f"{ticker} 在 {period} 期间的{metric_label}为 {value}。{ref}".strip()
        return f"{ticker} 的{metric_label}为 {value}。{ref}".strip()
    parts = [part for part in (ticker, metric, period, value) if part]
    return f"{' '.join(parts)}. {ref}".strip()


def _text_for_card(card: Mapping[str, Any]) -> str:
    text = str(card.get("claim") or card.get("supporting_snippet") or card.get("text_snippet") or card.get("snippet") or "").strip()
    return text[:279].rstrip() + "..." if len(text) > 280 else text


def _zh_driver_summary(card: Mapping[str, Any]) -> str:
    text = _text_for_card(card)
    lowered = text.lower()
    claim_scope = str(card.get("claim_scope") or "").strip()
    levels = {str(item) for item in card.get("driver_levels", []) or [] if str(item)}
    if not claim_scope:
        if "company_level_driver" in levels:
            claim_scope = "company"
        elif "product_level_driver" in levels:
            claim_scope = "product"
        elif "segment_level_driver" in levels:
            claim_scope = "segment"
    scope = "公司层面" if claim_scope == "company" else ("产品层面" if claim_scope == "product" else ("分部层面" if claim_scope == "segment" else "披露文本"))
    if "compute & networking" in lowered or "compute and networking" in lowered:
        return f"{scope}披露显示，Compute & Networking 收入增长与 accelerated computing 和 AI 平台转移相关。"
    if "networking" in lowered and any(term in lowered for term in ("nvlink", "infiniband", "ethernet", "gb200", "gb300")):
        return f"{scope}披露显示，Networking 增长与 NVLink、InfiniBand、Ethernet 或 GB200/GB300 等 AI 网络互连产品相关。"
    if "accelerated computing platform" in lowered or "large language models" in lowered or "generative" in lowered:
        return f"{scope}披露显示，增长与用于大语言模型、推荐引擎、生成式或 agentic AI 应用的 accelerated computing platform 需求相关。"
    if "data center" in lowered:
        return f"{scope}披露显示，Data Center 相关业务增长是重要线索，但需要区分分部表现和总公司营收原因。"
    if "revenue" in lowered and "driven" in lowered:
        return f"{scope}披露提供了收入增长驱动线索，但仍需核对其适用范围。"
    if scope != "公司层面" and any(term in text for term in ("营收增长主要由", "总营收增长由", "收入增长主要由")):
        return f"{scope}证据提供了增长线索，但不能完整代表总公司营收增长原因。"
    return text


def _driver_buckets(text_cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    company: list[dict[str, Any]] = []
    supplemental: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []
    for card in text_cards:
        claim_scope = str(card.get("claim_scope") or "").strip()
        if claim_scope == "company":
            company.append(card)
            continue
        if claim_scope in {"segment", "product"}:
            supplemental.append(card)
            continue
        levels = {str(item) for item in card.get("driver_levels", []) or [] if str(item)}
        if not levels and card.get("driver_level"):
            levels.add(str(card.get("driver_level")))
        if "company_level_driver" in levels:
            company.append(card)
        elif levels & {"segment_level_driver", "product_level_driver"}:
            supplemental.append(card)
        else:
            other.append(card)
    return company, supplemental, other


def build_causal_analytical_claims(
    *,
    company: str,
    numeric_cards: list[dict[str, Any]],
    text_cards: list[dict[str, Any]],
    requirement_summary: Mapping[str, Any],
    evidence_health: str,
    tool_error_context: list[dict[str, Any]] | None,
    lang: str,
) -> list[dict[str, Any]]:
    """Build deterministic claim tiers for causal revenue-growth answers."""
    claims: list[AnalyticalClaim] = []
    company = company or ("该公司" if lang == "zh" else "the company")
    company_level, supplemental, other = _driver_buckets(text_cards)
    text_driver_cards = company_level + supplemental + other
    missing_but = {
        str(item)
        for item in list(requirement_summary.get("missing_but_analyzable_answer_parts", []) or [])
        if str(item)
    }
    missing = {str(item) for item in list(requirement_summary.get("missing_required_answer_parts", []) or []) if str(item)}
    partial = {str(item) for item in list(requirement_summary.get("partial_required_answer_parts", []) or []) if str(item)}

    for card in numeric_cards[:3]:
        ref = citation_for_item(card)
        if not ref:
            continue
        claims.append(
            AnalyticalClaim(
                id=f"eb{len(claims) + 1}",
                text=_format_numeric_fact(card, lang),
                tier=ClaimTier.EVIDENCE_BACKED,
                citation_refs=[ref],
                confidence="medium" if str(card.get("confidence") or "").lower() == "medium" else "high",
            )
        )

    for card in text_driver_cards[:5]:
        ref = citation_for_item(card)
        text = _zh_driver_summary(card) if lang == "zh" else _text_for_card(card)
        if not ref or not text:
            continue
        claims.append(
            AnalyticalClaim(
                id=f"eb{len(claims) + 1}",
                text=f"{text} {bracket_ref(ref)}".strip(),
                tier=ClaimTier.EVIDENCE_BACKED,
                citation_refs=[ref],
                confidence="high",
                claim_scope=str(card.get("claim_scope") or "unknown"),
                allowed_claim_strength=str(card.get("allowed_claim_strength") or "bounded_inference"),
                scope_reason=str(card.get("scope_reason") or ""),
            )
        )

    supporting_ids = [claim.id for claim in claims if claim.tier == ClaimTier.EVIDENCE_BACKED]
    supporting_refs = [ref for claim in claims for ref in claim.citation_refs]
    if lang == "zh":
        if company_level:
            inferred = f"这些公司层面披露证据表明，{company} 的营收增长可能与公司披露的需求、平台或业务增长因素有关。"
        elif supplemental:
            inferred = f"现有分部/产品层面证据指向相关业务线可能参与了增长，但不能完整代表 {company} 总营收增长原因。"
        elif numeric_cards:
            inferred = f"已验证收入证据只能说明 {company} 的营收规模或变化，不能单独证明增长原因；合理分析需要继续验证需求、产品周期、供给和收入确认路径。"
        else:
            inferred = f"本轮直接证据不足，不能对 {company} 营收增长作确定归因；只能给出需要验证的分析路径。"
    else:
        if company_level:
            inferred = f"The company-level filing evidence suggests {company}'s revenue growth may be linked to disclosed demand, platform, or operating-growth factors."
        elif supplemental:
            inferred = f"The segment/product evidence points to possible contributors, but it cannot fully represent {company}'s total-revenue growth causes."
        elif numeric_cards:
            inferred = f"The verified revenue evidence can show {company}'s scale or change, but it cannot prove the cause by itself; demand, product cycle, supply, and revenue-recognition paths need verification."
        else:
            inferred = f"Direct evidence is insufficient for a definitive causal attribution for {company}; only a verification framework is supportable."
    claims.append(
        AnalyticalClaim(
            id="ei1",
            text=inferred,
            tier=ClaimTier.EVIDENCE_INFERRED,
            citation_refs=supporting_refs[:5],
            supporting_claim_ids=supporting_ids[:5],
            confidence="medium" if company_level else "low",
            caveat="Driver evidence incomplete" if ("identify_growth_drivers" in missing_but or "identify_growth_drivers" in missing or partial) else None,
        )
    )

    zh_hypotheses = [
        "待验证假设：云厂商和企业 AI 资本开支是否继续扩张。",
        "待验证假设：Blackwell / Hopper 等产品周期是否推动出货、ASP 或收入确认。",
        "待验证假设：网络互连产品是否随 GPU 集群建设放量。",
        "待验证假设：供应链产能释放是否让收入确认加速。",
        "待验证假设：产品组合变化是否推高数据中心或总收入贡献。",
    ]
    en_hypotheses = [
        "Hypothesis to verify: whether cloud and enterprise AI capex continued to expand.",
        "Hypothesis to verify: whether Blackwell / Hopper product cycles lifted shipments, ASP, or revenue recognition.",
        "Hypothesis to verify: whether networking interconnect products scaled with GPU cluster buildouts.",
        "Hypothesis to verify: whether supply availability accelerated revenue recognition.",
        "Hypothesis to verify: whether product mix increased data-center or total-revenue contribution.",
    ]
    for text in (zh_hypotheses if lang == "zh" else en_hypotheses)[:5]:
        claims.append(
            AnalyticalClaim(
                id=f"hv{len([c for c in claims if c.tier == ClaimTier.HYPOTHESIS_TO_VERIFY]) + 1}",
                text=text,
                tier=ClaimTier.HYPOTHESIS_TO_VERIFY,
                confidence="low",
                caveat="Must be validated with segment revenue, MD&A, customer demand, pricing, shipment, or supply evidence.",
            )
        )

    if tool_error_context:
        text = (
            "待验证假设：本轮部分检索或工具执行退化，因此不能把缺失披露当作不存在；需要补跑 filing 检索后再确认。"
            if lang == "zh"
            else "Hypothesis to verify: some retrieval or tool execution degraded in this run, so missing disclosures should not be treated as non-existence; filing retrieval should be rerun before confirmation."
        )
        claims.append(
            AnalyticalClaim(
                id="hv_tool",
                text=text,
                tier=ClaimTier.HYPOTHESIS_TO_VERIFY,
                confidence="low",
                caveat=str(evidence_health or "degraded"),
            )
        )
    return [claim.model_dump(mode="json", exclude_none=True) for claim in claims]


def analytical_reasoning_payload(
    *,
    research_plan: Mapping[str, Any],
    numeric_cards: list[dict[str, Any]],
    text_cards: list[dict[str, Any]],
    requirement_summary: Mapping[str, Any],
    evidence_sufficiency: Mapping[str, Any] | None,
    lang: str,
) -> dict[str, Any]:
    evidence_sufficiency = dict(evidence_sufficiency or {})
    evidence_health = str(
        evidence_sufficiency.get("evidence_health")
        or requirement_summary.get("evidence_health")
        or "complete"
    )
    tool_context = list(
        evidence_sufficiency.get("tool_error_context")
        or requirement_summary.get("tool_error_context")
        or []
    )
    if str(research_plan.get("question_type") or "") != "causal_explanation":
        return {
            "analytical_claims": [],
            "claim_tiers": claim_tier_counts([]),
            "analytical_reasoning_status": "not_applicable",
            "evidence_health": evidence_health,
            "tool_error_context": tool_context,
        }
    company = next((str(item).upper() for item in research_plan.get("companies", []) or [] if str(item)), "")
    claims = build_causal_analytical_claims(
        company=company,
        numeric_cards=numeric_cards,
        text_cards=text_cards,
        requirement_summary=requirement_summary,
        evidence_health=evidence_health,
        tool_error_context=tool_context,
        lang=lang,
    )
    return {
        "analytical_claims": claims,
        "claim_tiers": claim_tier_counts(claims),
        "analytical_reasoning_status": "used",
        "evidence_health": evidence_health,
        "tool_error_context": tool_context,
    }


def analytical_gap_structure_present(answer: str) -> bool:
    text = str(answer or "").lower()
    section_terms = (
        "已验证证据",
        "合理推断",
        "待验证假设",
        "反方因素",
        "其他解释",
        "关键观察指标",
        "证据边界",
        "verified evidence",
        "evidence-based inference",
        "hypothesis to verify",
        "counterpoints",
        "observation metrics",
        "evidence boundary",
    )
    return sum(1 for term in section_terms if term.lower() in text) >= 3


def hypothesis_marker_present(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        term in lowered
        for term in (
            "待验证",
            "假设",
            "可能因素",
            "需要验证",
            "hypothesis",
            "to verify",
            "needs verification",
            "possible factor",
        )
    )
