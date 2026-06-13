# ruff: noqa: F401,F403,F405
"""Citation generation, sorting, dedupe, and citation policy."""

from __future__ import annotations

from typing import Any

from src.agent.constants import *
from src.agent.evidence import _structured_source_type, _to_float

def _build_citation_refs(citations: list[dict[str, Any]], lang: str, limit: int = 5) -> list[str]:
    seen: set[tuple[str, str, str, str]] = set()
    refs: list[str] = []
    for c in citations:
        key = (
            c.get("ticker", ""),
            c.get("form_type", ""),
            c.get("fiscal_period", ""),
            c.get("section", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        if lang == "zh":
            refs.append(f"{key[0]} {key[1]} {key[2]}（{key[3]}）")
        else:
            refs.append(f"{key[0]} {key[1]} {key[2]} ({key[3]})")
        if len(refs) >= limit:
            break
    return refs

def _doc_key(doc: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        str(doc.get("ticker", "")),
        str(doc.get("filing_id", "")),
        str(doc.get("fiscal_period", "")),
        str(doc.get("section", "")),
        int(doc.get("chunk_order", 0) or 0),
    )

def _dedupe_document_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for c in citations:
        key = (
            str(c.get("ticker", c.get("source", ""))),
            str(c.get("form_type", c.get("filing_type", ""))),
            str(c.get("fiscal_period", c.get("period", ""))),
            str(c.get("section", "")),
            int(c.get("chunk_order", 0) or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        item = dict(c)
        item.setdefault("source_kind", "document")
        item.setdefault("source", item.get("ticker", ""))
        item.setdefault("filing_type", item.get("form_type", ""))
        item.setdefault("period", item.get("fiscal_period", ""))
        out.append(item)
    return out

def _numeric_citation_from_evidence(item: dict[str, Any]) -> dict[str, Any]:
    period_end = str(item.get("period_end", ""))
    source_provider = str(item.get("source_provider") or "unknown")
    structured_type = _structured_source_type(source_provider)
    return {
        "source_kind": "structured",
        "source": str(item.get("ticker", "")),
        "ticker": str(item.get("ticker", "")),
        "filing_type": structured_type,
        "form_type": structured_type,
        "period": period_end,
        "fiscal_period": period_end,
        "section": "STRUCTURED",
        "metric": str(item.get("metric", "")),
        "period_type": str(item.get("period_type", "")),
        "period_end": period_end,
        "fiscal_year": item.get("fiscal_year"),
        "fiscal_quarter": item.get("fiscal_quarter"),
        "calendar_year": item.get("calendar_year"),
        "calendar_quarter": item.get("calendar_quarter"),
        "filing_date": str(item.get("filing_date", "") or ""),
        "source_provider": source_provider,
        "source_url": str(item.get("source_url", "") or ""),
        "source_filing_id": str(item.get("source_filing_id", "") or ""),
        "confidence": str(item.get("confidence", "") or ""),
        "extraction_method": str(item.get("extraction_method", "") or ""),
        "source_tag": str(item.get("source_tag", "") or ""),
        "reconciliation_warning": str(item.get("reconciliation_warning", "") or ""),
        "part": "",
        "quality": "high",
        "text_snippet": "",
        "section_fallback": False,
        "evidence_id": str(item.get("evidence_id", "")),
        "requirement_id": str(item.get("requirement_id", "")),
        "requirement_ids": list(item.get("requirement_ids", []) or []),
    }

def _text_citation_from_evidence(item: dict[str, Any]) -> dict[str, Any]:
    fiscal_period = str(item.get("fiscal_period") or UNKNOWN_PERIOD)
    return {
        "source_kind": "document",
        "source": str(item.get("ticker", "")),
        "ticker": str(item.get("ticker", "")),
        "filing_id": str(item.get("filing_id", "")),
        "filing_type": str(item.get("form_type", "")),
        "form_type": str(item.get("form_type", "")),
        "period": fiscal_period,
        "fiscal_period": fiscal_period,
        "section": str(item.get("section", "")),
        "metric": "",
        "period_type": "",
        "period_end": fiscal_period,
        "filing_date": "",
        "part": "",
        "quality": str(item.get("quality", "")),
        "score": _to_float(item.get("score")) or 0.0,
        "text_snippet": str(item.get("text_snippet", "")),
        "supporting_snippet": str(item.get("supporting_snippet") or item.get("text_snippet", "")),
        "supporting_terms": list(item.get("supporting_terms", []) or []),
        "score_breakdown": dict(item.get("score_breakdown", {}) or {}),
        "section_fallback": bool(item.get("section_fallback", False)),
        "evidence_id": str(item.get("evidence_id", "")),
        "requirement_id": str(item.get("requirement_id", "")),
        "requirement_ids": list(item.get("requirement_ids", []) or []),
        "chunk_order": int(item.get("chunk_order", 0) or 0),
    }

def _collect_citations_from_claims(
    numeric_claims: list[dict[str, Any]],
    text_claims: list[dict[str, Any]],
    numeric_evidence: list[dict[str, Any]],
    text_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    used_numeric_ids = {
        eid
        for claim in numeric_claims
        for eid in claim.get("evidence_ids", [])
        if eid.startswith("N")
    }
    used_text_ids = {
        eid
        for claim in text_claims
        for eid in claim.get("evidence_ids", [])
        if eid.startswith("T")
    }

    numeric_citations = [
        _numeric_citation_from_evidence(item)
        for item in numeric_evidence
        if str(item.get("evidence_id", "")) in used_numeric_ids
    ]
    text_citations = [
        _text_citation_from_evidence(item)
        for item in text_evidence
        if str(item.get("evidence_id", "")) in used_text_ids
    ]
    return numeric_citations, text_citations

def _is_mixed_section(section: str) -> bool:
    sec = str(section or "").upper().strip()
    return sec in {"", "MIXED", "UNKNOWN"}

def _sort_text_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(c: dict[str, Any]) -> tuple[Any, ...]:
        mixed = 1 if _is_mixed_section(str(c.get("section", ""))) else 0
        fallback = 1 if bool(c.get("section_fallback", False)) else 0
        quality = str(c.get("quality", "")).lower().strip()
        q_rank = {"high": 0, "medium": 1, "low": 2}.get(quality, 3)
        score = _to_float(c.get("score")) or 0.0
        period = str(c.get("fiscal_period", ""))
        return (mixed, fallback, q_rank, -score, period)

    return sorted(citations, key=_key)

def _dedupe_text_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, int]] = set()
    for c in citations:
        key = (
            str(c.get("ticker", "")),
            str(c.get("filing_id", c.get("form_type", ""))),
            str(c.get("fiscal_period", "")),
            str(c.get("section", "")),
            int(c.get("chunk_order", 0) or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out

def _apply_text_citation_policy(
    state: AgentState,
    text_citations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    task_type = str(state.get("task_type", "fact_qa"))
    retrieval_policy = dict(state.get("retrieval_policy", {}))
    reasons: list[dict[str, Any]] = []
    capped = _sort_text_citations(_dedupe_text_citations(text_citations))
    if not capped:
        return [], reasons, False

    # Keep MIXED-like citations as low-priority fallback only.
    non_mixed = [c for c in capped if not _is_mixed_section(str(c.get("section", "")))]
    mixed = [c for c in capped if _is_mixed_section(str(c.get("section", "")))]
    prioritized = non_mixed + mixed
    if len(non_mixed) >= 2:
        head = prioritized[:2]
        if any(_is_mixed_section(str(c.get("section", ""))) for c in head):
            prioritized = non_mixed + mixed
            reasons.append(
                {
                    "claim_type": "system",
                    "sentence": "",
                    "evidence_ids": [],
                    "reason": "mixed_section_demoted_in_primary_positions",
                }
            )

    if task_type == "company_comparison":
        per_company_cap = int(retrieval_policy.get("comparison_text_cap_per_company", 2) or 2)
        by_ticker: dict[str, list[dict[str, Any]]] = {}
        for c in prioritized:
            ticker = str(c.get("ticker") or c.get("company") or c.get("entity") or "").upper().strip()
            by_ticker.setdefault(ticker, []).append(c)
        companies = []
        for t in state.get("companies", []):
            s = str(t or "").upper()
            if s and s not in companies:
                companies.append(s)
        ct = str(state.get("comparison_target") or "").upper()
        if ct and ct not in companies:
            companies.append(ct)
        if not companies:
            companies = sorted(by_ticker.keys())
        selected: list[dict[str, Any]] = []
        coverage = 0
        for ticker in companies[:2]:
            rows = by_ticker.get(ticker, [])
            picked = rows[: max(0, per_company_cap)]
            if picked:
                coverage += 1
            selected.extend(picked)
        if retrieval_policy.get("require_balanced_comparison_text", True) and coverage < 2 and selected:
            reasons.append(
                {
                    "claim_type": "system",
                    "sentence": "",
                    "evidence_ids": [],
                    "reason": "comparison_text_evidence_unbalanced",
                }
            )
            return [], reasons, True
        return selected[:TEXT_CITATION_CAPS.get(task_type, 4)], reasons, False

    cap = int(TEXT_CITATION_CAPS.get(task_type, 3))
    return prioritized[: max(0, cap)], reasons, False


collect_citations_from_claims = _collect_citations_from_claims
apply_text_citation_policy = _apply_text_citation_policy
numeric_citation_from_evidence = _numeric_citation_from_evidence
text_citation_from_evidence = _text_citation_from_evidence
