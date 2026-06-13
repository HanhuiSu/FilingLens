"""search_filings — retrieve relevant SEC filing chunks via vector search + light rerank."""

from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any, Optional

import chromadb
import duckdb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import settings

_model: SentenceTransformer | None = None
_collections: dict[str, chromadb.Collection] = {}
RETRIEVAL_PROFILES = {
    "default",
    "fact_support",
    "trend_support",
    "comparison_support",
    "summary",
    "risk_summary",
}

_LEXICAL_BONUS_WEIGHT = 0.08
_RETRIEVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "which",
    "with",
    "about",
    "discuss",
    "describe",
    "explain",
    "show",
    "summarize",
    "summary",
}
_SUPPORTING_TERM_STOPWORDS = _RETRIEVAL_STOPWORDS | {
    "business",
    "company",
    "discussion",
    "filing",
    "item",
    "management",
    "operating",
    "results",
    "risk",
    "risks",
}


def _collection_name_for_version(version: str) -> str:
    v = str(version or "").lower()
    if v == "v2":
        return settings.rag_collection_v2
    return settings.rag_collection_v1


def _selected_collection_name() -> str:
    return _collection_name_for_version(settings.rag_index_version)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = settings.embedding_device
        try:
            _model = SentenceTransformer(
                settings.embedding_model_name,
                device=device,
                trust_remote_code=True,
            )
            _model.encode(["warmup"], convert_to_numpy=True)
        except RuntimeError:
            _model = SentenceTransformer(
                settings.embedding_model_name,
                device="cpu",
                trust_remote_code=True,
            )
        except (OSError, Exception) as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{settings.embedding_model_name}'. "
                f"Ensure the model is downloaded locally or network is available: {exc}"
            ) from exc
    return _model


def _get_collection() -> chromadb.Collection:
    name = _selected_collection_name()
    if name in _collections:
        return _collections[name]

    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_collection(name)
    except Exception:
        # Fallback to v1 collection for rollout safety.
        fallback = _collection_name_for_version("v1")
        coll = client.get_collection(fallback)
    _collections[name] = coll
    return coll


def _filing_ids_in_date_range(
    ticker: str,
    date_start: str | None,
    date_end: str | None,
    form_type: str | None,
) -> set[str] | None:
    """Return filing_ids matching date/form filters, or None if no date filter."""
    if not date_start and not date_end:
        return None
    params: list[Any] = [ticker.upper()]
    sql = "SELECT filing_id FROM filings_metadata WHERE ticker = ?"
    if form_type:
        sql += " AND form_type = ?"
        params.append(form_type.upper().replace(" ", ""))
    if date_start:
        sql += " AND filing_date >= CAST(? AS DATE)"
        params.append(date_start)
    if date_end:
        sql += " AND filing_date <= CAST(? AS DATE)"
        params.append(date_end)
    conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _is_mixed_like(section: str, quality: str) -> bool:
    sec = (section or "").upper().strip()
    q = (quality or "").lower().strip()
    return sec in {"", "MIXED", "UNKNOWN"} or q == "low"


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for c in candidates:
        text = (c.get("text") or "").strip()
        # Deduplicate same filing/section with highly similar textual payload.
        text_key = hashlib.md5(text[:500].encode("utf-8", errors="ignore")).hexdigest()
        key = (str(c.get("filing_id", "")), str(c.get("section", "")), text_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _normalize_section_allowlist(section_allowlist: list[str] | None) -> set[str]:
    allow: set[str] = set()
    for sec in section_allowlist or []:
        s = str(sec or "").upper().strip()
        if s:
            allow.add(s)
    return allow


def _apply_section_allowlist(
    candidates: list[dict[str, Any]],
    section_allowlist: list[str] | None,
    strict_sections: bool,
    min_section_hits: int,
) -> tuple[list[dict[str, Any]], set[str], bool]:
    allow = _normalize_section_allowlist(section_allowlist)
    if not allow:
        return candidates, allow, False

    preferred = [c for c in candidates if str(c.get("section", "")).upper() in allow]
    if strict_sections:
        return preferred, allow, False
    if len(preferred) >= min_section_hits:
        return preferred, allow, False
    return candidates, allow, True


def _diagnostic_snippet(item: dict[str, Any], *, max_chars: int = 360) -> dict[str, Any]:
    text = str(item.get("supporting_snippet") or item.get("text_snippet") or item.get("text") or "")
    snippet = re.sub(r"\s+", " ", text).strip()[:max_chars]
    return {
        "ticker": str(item.get("ticker", "")),
        "filing_id": str(item.get("filing_id", "")),
        "form_type": str(item.get("form_type", "")),
        "fiscal_period": str(item.get("fiscal_period", "")),
        "section": str(item.get("section", "")),
        "quality": str(item.get("quality", "")),
        "chunk_order": int(item.get("chunk_order", 0) or 0),
        "score": round(float(item.get("score", 0.0) or 0.0), 4),
        "final_score": round(float(item.get("final_score", item.get("score", 0.0)) or 0.0), 4),
        "text_snippet": snippet,
        "supporting_snippet": snippet,
        "supporting_terms": list(item.get("supporting_terms", []) or []),
        "score_breakdown": dict(item.get("score_breakdown", {}) or {}),
    }


def _diagnostic_envelope(
    *,
    items: list[dict[str, Any]],
    ticker: str,
    query: str,
    section_allowlist: list[str] | None,
    strict_sections: bool,
    fetch_k: int,
    raw_candidates: list[dict[str, Any]],
    section_filtered_candidates: list[dict[str, Any]],
    used_section_fallback: bool,
    backend: str = "chroma_vector",
) -> dict[str, Any]:
    raw_hit_count = len(raw_candidates)
    section_filtered_hit_count = len(section_filtered_candidates)
    top_raw = [_diagnostic_snippet(c) for c in raw_candidates[:10]]
    top_section_filtered = [_diagnostic_snippet(c) for c in section_filtered_candidates[:10]]
    return {
        "items": items,
        "diagnostics": {
            "ticker": ticker.upper(),
            "query": query,
            "section_allowlist": list(section_allowlist or []),
            "strict_sections": bool(strict_sections),
            "fetch_k": int(fetch_k),
            "backend": str(backend),
            "raw_hit_count": raw_hit_count,
            "section_filtered_hit_count": section_filtered_hit_count,
            "used_section_fallback": bool(used_section_fallback),
            "top_raw_snippets": top_raw,
            "top_section_filtered_snippets": top_section_filtered,
            "raw_candidates": top_raw,
            "section_filtered_candidates": top_section_filtered,
        },
    }


def _normalize_profile(profile: str | None) -> str:
    p = str(profile or "default").strip().lower()
    if p in RETRIEVAL_PROFILES:
        return p
    return "default"


def _period_year(text: str) -> int | None:
    raw = str(text or "").strip()
    if len(raw) >= 4 and raw[:4].isdigit():
        try:
            return int(raw[:4])
        except ValueError:
            return None
    return None


def _section_group(section: str) -> str:
    sec = str(section or "").upper().strip()
    if sec == "ITEM_1A":
        return "risk"
    if sec in {"ITEM_2", "ITEM_7", "ITEM_7A"}:
        return "mda"
    if sec in {"ITEM_6", "ITEM_8", "ITEM_15"}:
        return "financial"
    return "other"


def _section_bonus(section: str, profile: str) -> float:
    sec = str(section or "").upper().strip()
    if profile == "risk_summary":
        if sec == "ITEM_1A":
            return 0.18
        if sec == "ITEM_7":
            return 0.10
        if sec in {"ITEM_2", "ITEM_3"}:
            return 0.03
        return -0.02
    if profile == "summary":
        group = _section_group(sec)
        if group == "mda":
            return 0.09
        if group == "risk":
            return 0.08
        if group == "financial":
            return 0.07
        return 0.01
    if profile == "comparison_support":
        if sec in {"ITEM_2", "ITEM_7", "ITEM_1A", "ITEM_8"}:
            return 0.05
        return 0.01
    if profile == "trend_support":
        if sec in {"ITEM_2", "ITEM_7"}:
            return 0.05
        if sec == "ITEM_1A":
            return 0.02
        return 0.0
    if profile == "fact_support":
        if sec in {"ITEM_2", "ITEM_8", "ITEM_6"}:
            return 0.03
        return 0.0
    return 0.0


def _quality_bonus(quality: str) -> float:
    q = str(quality or "").lower().strip()
    if q == "high":
        return 0.04
    if q == "medium":
        return 0.01
    if q == "low":
        return -0.06
    return 0.0


def _period_bonus(fiscal_period: str, target_periods: set[str]) -> float:
    if not target_periods:
        return 0.0
    p = str(fiscal_period or "").strip()
    if not p:
        return 0.0
    if p in target_periods:
        return 0.12
    year = _period_year(p)
    if year is None:
        return 0.0
    if any(_period_year(tp) == year for tp in target_periods):
        return 0.04
    return -0.01


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tokenize_for_retrieval(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9][a-z0-9&._-]*", str(text or "").lower())
    return [t for t in tokens if len(t) > 2 and t not in _RETRIEVAL_STOPWORDS]


def _lexical_score(query: str, text: str) -> float:
    query_terms = list(dict.fromkeys(_tokenize_for_retrieval(query)))
    if not query_terms:
        return 0.0
    doc_counts = Counter(_tokenize_for_retrieval(text))
    if not doc_counts:
        return 0.0
    matched = [term for term in query_terms if doc_counts.get(term, 0) > 0]
    if not matched:
        return 0.0
    term_recall = len(matched) / max(len(query_terms), 1)
    clipped_density = sum(min(doc_counts[term], 3) for term in matched) / max(len(query_terms) * 3, 1)
    return round(float(min(1.0, 0.75 * term_recall + 0.25 * clipped_density)), 6)


def _sentence_candidates(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|(?<=;)\s+", normalized)
    candidates = [p.strip(" \t\r\n-•") for p in parts if p.strip(" \t\r\n-•")]
    if candidates:
        return candidates
    return [normalized]


def _truncate_snippet(text: str, max_len: int = 360) -> str:
    raw = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= max_len:
        return raw
    return raw[: max(0, max_len - 1)].rstrip() + "…"


def _supporting_snippet(
    text: str,
    query: str,
    ticker: str = "",
    max_len: int = 360,
) -> tuple[str, list[str]]:
    sentences = _sentence_candidates(text)
    if not sentences:
        return "", []

    query_terms = list(dict.fromkeys(_tokenize_for_retrieval(query)))
    ticker_term = str(ticker or "").lower().strip()
    best_sentence = sentences[0]
    best_score = -1.0
    best_terms: list[str] = []

    for idx, sentence in enumerate(sentences):
        sentence_terms = set(_tokenize_for_retrieval(sentence))
        matched = [term for term in query_terms if term in sentence_terms]
        score = len(matched) / max(len(query_terms), 1) if query_terms else 0.0
        if ticker_term and ticker_term in sentence.lower():
            score += 0.15
        # Prefer earlier sentences when support is otherwise tied.
        score -= idx * 0.001
        if score > best_score:
            best_score = score
            best_sentence = sentence
            best_terms = matched

    snippet = _truncate_snippet(best_sentence, max_len=max_len)
    if best_terms:
        return snippet, best_terms

    sentence_terms = [
        term
        for term in _tokenize_for_retrieval(best_sentence)
        if len(term) > 3 and term not in _SUPPORTING_TERM_STOPWORDS
    ]
    fallback_terms = list(dict.fromkeys(sentence_terms))
    if ticker_term and ticker_term not in fallback_terms:
        fallback_terms.insert(0, ticker_term)
    return snippet, fallback_terms[:12]


def _date_rank_bonus(value: Any, rank_by_date: dict[str, int], total_dates: int) -> float:
    key = str(value or "")
    if not key or total_dates <= 0:
        return 0.0
    rank = rank_by_date.get(key, total_dates)
    if total_dates <= 1:
        return 0.08
    return round(0.08 * (1.0 - (rank / max(total_dates - 1, 1))), 6)


def _cap_ranked_candidates(
    candidates: list[dict[str, Any]],
    *,
    top_k: int,
    max_per_filing: int | None,
    max_per_section: int | None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    filing_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    filing_cap = max(0, int(max_per_filing or 0))
    section_cap = max(0, int(max_per_section or 0))
    for candidate in candidates:
        filing_id = str(candidate.get("filing_id", ""))
        section = str(candidate.get("section", "")).upper().strip()
        if filing_cap > 0 and filing_counts[filing_id] >= filing_cap:
            continue
        if section_cap > 0 and section_counts[section] >= section_cap:
            continue
        selected.append(candidate)
        filing_counts[filing_id] += 1
        section_counts[section] += 1
        if len(selected) >= top_k:
            break
    if len(selected) >= top_k:
        return selected
    picked = {
        (str(item.get("filing_id", "")), int(item.get("chunk_order", 0) or 0))
        for item in selected
    }
    for candidate in candidates:
        key = (str(candidate.get("filing_id", "")), int(candidate.get("chunk_order", 0) or 0))
        if key in picked:
            continue
        selected.append(candidate)
        picked.add(key)
        if len(selected) >= top_k:
            break
    return selected


def search_filings_lexical_fallback(
    ticker: str,
    query: str,
    top_k: int = 8,
    form_type: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    section_allowlist: list[str] | None = None,
    strict_sections: bool = False,
    retrieval_profile: str | None = None,
    target_periods: list[str] | None = None,
    max_per_filing: int | None = None,
    max_per_section: int | None = None,
    return_diagnostics: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Fast DuckDB lexical fallback for local filing chunks."""
    top_k = min(max(int(top_k or 8), 1), 20)
    fetch_k = max(top_k * 50, 200)
    params: list[Any] = [ticker.upper()]
    sql = """
        SELECT
            c.filing_id,
            c.ticker,
            c.section,
            c.part,
            c.section_instance,
            c.quality,
            c.chunk_text,
            c.chunk_order,
            m.form_type,
            m.fiscal_period,
            m.filing_date
        FROM filing_chunks c
        JOIN filings_metadata m ON c.filing_id = m.filing_id
        WHERE c.ticker = ?
    """
    if form_type:
        sql += " AND m.form_type = ?"
        params.append(form_type.upper().replace(" ", ""))
    if date_start:
        sql += " AND m.filing_date >= CAST(? AS DATE)"
        params.append(date_start)
    if date_end:
        sql += " AND m.filing_date <= CAST(? AS DATE)"
        params.append(date_end)
    sql += " ORDER BY m.filing_date DESC NULLS LAST, c.filing_id DESC, c.chunk_order ASC LIMIT ?"
    params.append(fetch_k)

    conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    profile = _normalize_profile(retrieval_profile)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        (
            filing_id,
            row_ticker,
            section,
            part,
            section_instance,
            quality,
            text,
            chunk_order,
            row_form_type,
            fiscal_period,
            filing_date,
        ) = row
        text_value = str(text or "")
        lexical_score = _lexical_score(query, text_value)
        candidates.append(
            {
                "text": text_value,
                "filing_id": str(filing_id or ""),
                "ticker": str(row_ticker or ticker).upper(),
                "section": str(section or ""),
                "part": str(part or ""),
                "section_instance": int(section_instance or 1),
                "quality": str(quality or ""),
                "form_type": str(row_form_type or ""),
                "fiscal_period": str(fiscal_period or ""),
                "filing_date": str(filing_date or ""),
                "chunk_order": int(chunk_order or 0),
                "score": round(float(lexical_score), 6),
            }
        )

    candidates = _dedupe_candidates(candidates)
    raw_candidates = list(candidates)
    candidates, allowset, used_section_fallback = _apply_section_allowlist(
        candidates=candidates,
        section_allowlist=section_allowlist,
        strict_sections=bool(strict_sections),
        min_section_hits=3,
    )
    section_filtered_candidates = list(candidates)

    dates = sorted(
        {str(item.get("filing_date", "")) for item in raw_candidates if str(item.get("filing_date", "")).strip()},
        reverse=True,
    )
    rank_by_date = {date: idx for idx, date in enumerate(dates)}
    target_set = {str(x).strip() for x in (target_periods or []) if str(x).strip()}
    for candidate in candidates:
        section = str(candidate.get("section", "")).upper().strip()
        quality = str(candidate.get("quality", ""))
        lexical_score = float(candidate.get("score", 0.0) or 0.0)
        section_bonus = _section_bonus(section, profile)
        allow_bonus = 0.12 if allowset and section in allowset else 0.0
        period_bonus = _period_bonus(str(candidate.get("fiscal_period", "")), target_set)
        quality_bonus = _quality_bonus(quality)
        recency_bonus = _date_rank_bonus(candidate.get("filing_date"), rank_by_date, len(dates))
        mixed_penalty = 0.12 if _is_mixed_like(section, quality) else 0.0
        final_score = lexical_score + section_bonus + allow_bonus + period_bonus + quality_bonus + recency_bonus - mixed_penalty
        candidate["final_score"] = round(float(final_score), 6)
        candidate["score_breakdown"] = {
            "backend": "duckdb_lexical",
            "lexical_score": round(float(lexical_score), 6),
            "section_bonus": round(float(section_bonus + allow_bonus), 6),
            "period_bonus": round(float(period_bonus), 6),
            "quality_bonus": round(float(quality_bonus), 6),
            "recency_bonus": round(float(recency_bonus), 6),
            "mixed_penalty": round(float(mixed_penalty), 6),
            "final_score": round(float(final_score), 6),
        }

    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.get("final_score", 0.0) or 0.0),
            str(item.get("filing_date", "")),
            -int(item.get("chunk_order", 0) or 0),
        ),
        reverse=True,
    )
    selected = _cap_ranked_candidates(
        ranked,
        top_k=top_k,
        max_per_filing=max_per_filing,
        max_per_section=max_per_section,
    )

    out: list[dict[str, Any]] = []
    for candidate in selected[:top_k]:
        item = dict(candidate)
        section = str(item.get("section", "")).upper()
        item["section_fallback"] = bool(allowset) and used_section_fallback and section not in allowset
        item["source_kind"] = "document"
        item["retrieval_profile"] = profile
        item["retrieval_backend"] = "duckdb_lexical"
        item["retrieval_raw_hit_count"] = len(raw_candidates)
        item["section_filtered_hit_count"] = len(section_filtered_candidates)
        supporting_snippet, supporting_terms = _supporting_snippet(
            text=str(item.get("text", "")),
            query=query,
            ticker=str(item.get("ticker", ticker.upper())),
        )
        item["supporting_snippet"] = supporting_snippet
        item["supporting_terms"] = supporting_terms
        out.append(item)

    if return_diagnostics:
        return _diagnostic_envelope(
            items=out,
            ticker=ticker,
            query=query,
            section_allowlist=section_allowlist,
            strict_sections=strict_sections,
            fetch_k=fetch_k,
            raw_candidates=raw_candidates,
            section_filtered_candidates=section_filtered_candidates,
            used_section_fallback=used_section_fallback,
            backend="duckdb_lexical",
        )
    return out


def _select_with_profile(
    candidates: list[dict[str, Any]],
    candidate_embs: np.ndarray,
    query_emb: np.ndarray,
    top_k: int,
    lambda_mult: float,
    retrieval_profile: str,
    target_periods: list[str] | None,
    max_per_filing: int | None,
    max_per_section: int | None,
    query: str | None = None,
) -> list[dict[str, Any]]:
    if not candidates or top_k <= 0:
        return []

    sim_to_query = np.array([_cosine(query_emb, emb) for emb in candidate_embs], dtype=float)
    lexical_scores = [
        _lexical_score(str(query or ""), str(c.get("text", "")))
        for c in candidates
    ]
    selected_idx: list[int] = []
    remaining = set(range(len(candidates)))
    filing_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    profile = _normalize_profile(retrieval_profile)
    target_set = {str(x).strip() for x in (target_periods or []) if str(x).strip()}
    filing_cap = max(0, int(max_per_filing or 0))
    section_cap = max(0, int(max_per_section or 0))
    non_mixed_profiles = {"fact_support", "trend_support", "comparison_support"}

    while remaining and len(selected_idx) < top_k:
        best_idx: int | None = None
        best_score = -1e9
        rank_pos = len(selected_idx)
        has_non_mixed_remaining = any(
            not _is_mixed_like(str(candidates[i].get("section", "")), str(candidates[i].get("quality", "")))
            for i in remaining
        )

        for i in remaining:
            filing_id = str(candidates[i].get("filing_id", ""))
            section = str(candidates[i].get("section", "")).upper().strip() or "<EMPTY>"
            if filing_cap > 0 and filing_counts[filing_id] >= filing_cap:
                continue
            if section_cap > 0 and section_counts[section] >= section_cap:
                continue

            diversity = 0.0
            if selected_idx:
                diversity = max(_cosine(candidate_embs[i], candidate_embs[j]) for j in selected_idx)
            vector_score = lambda_mult * sim_to_query[i] - (1.0 - lambda_mult) * diversity

            quality = str(candidates[i].get("quality", ""))
            fiscal_period = str(candidates[i].get("fiscal_period", ""))
            group = _section_group(section)
            section_bonus = _section_bonus(section, profile)
            period_bonus = _period_bonus(fiscal_period, target_set)
            quality_bonus = _quality_bonus(quality)
            lexical_bonus = _LEXICAL_BONUS_WEIGHT * lexical_scores[i]

            mixed_penalty = 0.0
            if _is_mixed_like(section, quality):
                mixed_penalty += 0.16
                if profile in non_mixed_profiles:
                    mixed_penalty += 0.07
                if rank_pos < 2 and has_non_mixed_remaining:
                    mixed_penalty += 0.35

            repetition_penalty = 0.09 * filing_counts[filing_id] + 0.11 * section_counts[section]
            diversity_bonus = 0.0
            if profile in {"summary", "risk_summary"}:
                if group_counts[group] == 0:
                    diversity_bonus += 0.08
                elif group_counts[group] == 1:
                    diversity_bonus += 0.03

            score = (
                vector_score
                + section_bonus
                + period_bonus
                + lexical_bonus
                + quality_bonus
                + diversity_bonus
                - mixed_penalty
                - repetition_penalty
            )
            candidates[i]["final_score"] = round(float(score), 6)
            candidates[i]["score_breakdown"] = {
                "vector_score": round(float(vector_score), 6),
                "section_bonus": round(float(section_bonus), 6),
                "period_bonus": round(float(period_bonus), 6),
                "lexical_bonus": round(float(lexical_bonus), 6),
                "quality_bonus": round(float(quality_bonus), 6),
                "diversity_bonus": round(float(diversity_bonus), 6),
                "mixed_penalty": round(float(mixed_penalty), 6),
                "repetition_penalty": round(float(repetition_penalty), 6),
                "final_score": round(float(score), 6),
            }

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break
        selected_idx.append(best_idx)
        filing_id = str(candidates[best_idx].get("filing_id", ""))
        section = str(candidates[best_idx].get("section", "")).upper().strip() or "<EMPTY>"
        group = _section_group(section)
        filing_counts[filing_id] += 1
        section_counts[section] += 1
        group_counts[group] += 1
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_idx]


class SearchFilingsInput(BaseModel):
    """Input schema for the search_filings tool."""

    ticker: str = Field(description="Stock ticker symbol, e.g. AAPL")
    query: str = Field(description="Natural-language search query about the filing content")
    top_k: int = Field(default=8, description="Number of chunks to return (max 20)")
    form_type: Optional[str] = Field(
        default=None,
        description="Filter by filing form type: '10-K' or '10-Q'. Omit to search all forms.",
    )
    date_start: Optional[str] = Field(
        default=None,
        description="Only include filings filed on or after this date (YYYY-MM-DD)",
    )
    date_end: Optional[str] = Field(
        default=None,
        description="Only include filings filed on or before this date (YYYY-MM-DD)",
    )
    section_allowlist: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional section whitelist (e.g. ['ITEM_1A','ITEM_7']). "
            "When set, retrieval prefers these sections first."
        ),
    )
    strict_sections: bool = Field(
        default=False,
        description="If true, only return allowlisted sections; otherwise fallback to all sections if evidence is insufficient.",
    )
    retrieval_profile: Optional[str] = Field(
        default=None,
        description=(
            "Optional retrieval profile: default | fact_support | trend_support | "
            "comparison_support | summary | risk_summary."
        ),
    )
    target_periods: Optional[list[str]] = Field(
        default=None,
        description="Optional fiscal period hints for reranking (e.g. ['2025-12-31']).",
    )
    max_per_filing: Optional[int] = Field(
        default=None,
        description="Optional hard cap of chunks per filing_id in the final top-k.",
    )
    max_per_section: Optional[int] = Field(
        default=None,
        description="Optional hard cap of chunks per section label in the final top-k.",
    )
    return_diagnostics: bool = Field(
        default=False,
        description="Return retrieval diagnostics envelope instead of the legacy list result.",
    )


@tool("search_filings", args_schema=SearchFilingsInput)
def search_filings(
    ticker: str,
    query: str,
    top_k: int = 8,
    form_type: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    section_allowlist: list[str] | None = None,
    strict_sections: bool = False,
    retrieval_profile: str | None = None,
    target_periods: list[str] | None = None,
    max_per_filing: int | None = None,
    max_per_section: int | None = None,
    return_diagnostics: bool = False,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Search SEC 10-K / 10-Q filing text for a company.

    Uses vector retrieval plus task-aware reranking:
    - dedupe highly similar chunks
    - section-aware and period-aware soft biases
    - MMR-style diversity with filing/section repetition control
    """
    top_k = min(max(top_k, 1), 20)
    allowed_ids = _filing_ids_in_date_range(ticker, date_start, date_end, form_type)
    if allowed_ids is not None and not allowed_ids:
        if return_diagnostics:
            return _diagnostic_envelope(
                items=[],
                ticker=ticker,
                query=query,
                section_allowlist=section_allowlist,
                strict_sections=strict_sections,
                fetch_k=0,
                raw_candidates=[],
                section_filtered_candidates=[],
                used_section_fallback=False,
            )
        return []

    model = _get_model()
    coll = _get_collection()

    where_clauses: list[dict[str, Any]] = [{"ticker": ticker.upper()}]
    if form_type and allowed_ids is None:
        where_clauses.append({"form_type": form_type.upper().replace(" ", "")})
    where = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}

    overfetch = max(2, int(settings.rag_overfetch_multiplier))
    fetch_k = min(max(top_k * overfetch, top_k + 12), 200)
    if allowed_ids is not None:
        fetch_k = min(max(fetch_k, top_k * 10), 300)

    query_vector = model.encode([query], convert_to_numpy=True)
    results = coll.query(
        query_embeddings=query_vector.tolist(),
        n_results=fetch_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    candidates: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        fid = str(meta.get("filing_id", ""))
        if allowed_ids is not None and fid not in allowed_ids:
            continue
        score = 1.0 / (1.0 + float(dist))
        candidates.append(
            {
                "text": doc,
                "filing_id": fid,
                "ticker": meta.get("ticker", ticker.upper()),
                "section": meta.get("section", ""),
                "part": meta.get("part", ""),
                "section_instance": int(meta.get("section_instance", 1) or 1),
                "quality": meta.get("quality", ""),
                "form_type": meta.get("form_type", ""),
                "fiscal_period": meta.get("fiscal_period", ""),
                "chunk_order": int(meta.get("chunk_order", 0) or 0),
                "score": round(score, 4),
            }
        )

    candidates = _dedupe_candidates(candidates)
    raw_candidates = list(candidates)
    if not candidates:
        if return_diagnostics:
            return _diagnostic_envelope(
                items=[],
                ticker=ticker,
                query=query,
                section_allowlist=section_allowlist,
                strict_sections=strict_sections,
                fetch_k=fetch_k,
                raw_candidates=[],
                section_filtered_candidates=[],
                used_section_fallback=False,
            )
        return []

    candidates, allowset, used_section_fallback = _apply_section_allowlist(
        candidates=candidates,
        section_allowlist=section_allowlist,
        strict_sections=bool(strict_sections),
        min_section_hits=3,
    )
    if not candidates:
        if return_diagnostics:
            return _diagnostic_envelope(
                items=[],
                ticker=ticker,
                query=query,
                section_allowlist=section_allowlist,
                strict_sections=strict_sections,
                fetch_k=fetch_k,
                raw_candidates=raw_candidates,
                section_filtered_candidates=[],
                used_section_fallback=used_section_fallback,
            )
        return []
    section_filtered_candidates = list(candidates)

    lambda_mult = min(max(float(settings.rag_mmr_lambda), 0.0), 1.0)
    selected: list[dict[str, Any]] = []
    if candidates:
        c_embs = model.encode([c["text"] for c in candidates], convert_to_numpy=True)
        selected = _select_with_profile(
            candidates=candidates,
            candidate_embs=c_embs,
            query_emb=query_vector[0],
            top_k=top_k,
            lambda_mult=lambda_mult,
            retrieval_profile=_normalize_profile(retrieval_profile),
            target_periods=target_periods,
            max_per_filing=max_per_filing,
            max_per_section=max_per_section,
            query=query,
        )

    if len(selected) < top_k:
        # Final safety fill for strict caps that may over-prune.
        picked = {
            (str(c.get("filing_id", "")), int(c.get("chunk_order", 0)))
            for c in selected
        }
        for c in candidates:
            key = (str(c.get("filing_id", "")), int(c.get("chunk_order", 0)))
            if key in picked:
                continue
            selected.append(c)
            picked.add(key)
            if len(selected) >= top_k:
                break

    out: list[dict[str, Any]] = []
    for c in selected[:top_k]:
        item = dict(c)
        section = str(item.get("section", "")).upper()
        item["section_fallback"] = bool(allowset) and used_section_fallback and section not in allowset
        item["source_kind"] = "document"
        item["retrieval_profile"] = _normalize_profile(retrieval_profile)
        item["final_score"] = round(float(item.get("final_score", item.get("score", 0.0))), 4)
        item["retrieval_raw_hit_count"] = len(raw_candidates)
        item["section_filtered_hit_count"] = len(section_filtered_candidates)
        if "score_breakdown" not in item:
            item["score_breakdown"] = {
                "vector_score": round(float(item.get("score", 0.0)), 6),
                "section_bonus": 0.0,
                "period_bonus": 0.0,
                "lexical_bonus": 0.0,
                "quality_bonus": 0.0,
                "diversity_bonus": 0.0,
                "mixed_penalty": 0.0,
                "repetition_penalty": 0.0,
                "final_score": round(float(item.get("final_score", item.get("score", 0.0))), 6),
            }
        supporting_snippet, supporting_terms = _supporting_snippet(
            text=str(item.get("text", "")),
            query=query,
            ticker=str(item.get("ticker", ticker.upper())),
        )
        item["supporting_snippet"] = supporting_snippet
        item["supporting_terms"] = supporting_terms
        out.append(item)
    if return_diagnostics:
        return _diagnostic_envelope(
            items=out,
            ticker=ticker,
            query=query,
            section_allowlist=section_allowlist,
            strict_sections=strict_sections,
            fetch_k=fetch_k,
            raw_candidates=raw_candidates,
            section_filtered_candidates=section_filtered_candidates,
            used_section_fallback=used_section_fallback,
        )
    return out
