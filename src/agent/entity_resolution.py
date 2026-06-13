"""Company entity resolution for user queries.

This module intentionally only answers "which company or companies are
mentioned?". Higher-level routing lives in methodology/query understanding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from config import settings
from src.agent.types import AgentDomainModel


class ResolvedCompany(AgentDomainModel):
    ticker: str
    canonical_name: str = ""
    matched_text: str = ""
    match_type: str = "alias"
    confidence: float = 1.0
    position: int = 0


class EntityResolutionResult(AgentDomainModel):
    resolved_companies: list[ResolvedCompany] = []
    unresolved_mentions: list[str] = []
    confidence: float = 0.0
    ambiguity: bool = False


@dataclass(frozen=True)
class _CompanyEntry:
    ticker: str
    name: str
    aliases: tuple[str, ...]


_COMPANIES: tuple[_CompanyEntry, ...] = (
    _CompanyEntry("AAPL", "Apple", ("aapl", "apple", "apple inc", "苹果")),
    _CompanyEntry("AMZN", "Amazon", ("amzn", "amazon", "amazon.com", "亚马逊")),
    _CompanyEntry("NVDA", "NVIDIA", ("nvda", "nvidia", "nvidia corp", "英伟达")),
    _CompanyEntry("MSFT", "Microsoft", ("msft", "microsoft", "微软")),
    _CompanyEntry("GOOGL", "Alphabet", ("googl", "google", "alphabet", "谷歌")),
    _CompanyEntry("TSLA", "Tesla", ("tsla", "tesla", "特斯拉")),
    _CompanyEntry("JPM", "JPMorgan Chase", ("jpm", "jpmorgan", "jp morgan", "jpmorgan chase", "摩根大通")),
    _CompanyEntry("JNJ", "Johnson & Johnson", ("jnj", "johnson", "johnson & johnson", "j&j", "强生")),
)

_TARGET_TICKERS = {str(ticker).upper() for ticker in getattr(settings, "target_tickers", [])}
_TICKER_TO_ENTRY: dict[str, _CompanyEntry] = {entry.ticker: entry for entry in _COMPANIES}
_ALIAS_TO_TICKER: dict[str, str] = {
    alias.lower(): entry.ticker
    for entry in _COMPANIES
    for alias in entry.aliases
}
_GENERIC_REFERENCES = ("这家公司", "这个公司", "这只股票", "它", "this company", "the company")


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _add_company(
    out: list[ResolvedCompany],
    *,
    ticker: str,
    matched_text: str,
    match_type: str,
    confidence: float,
    position: int = 0,
) -> None:
    ticker = str(ticker or "").upper().strip()
    if not ticker:
        return
    if ticker not in _TICKER_TO_ENTRY and ticker not in _TARGET_TICKERS:
        return
    if any(item.ticker == ticker for item in out):
        return
    entry = _TICKER_TO_ENTRY.get(ticker)
    out.append(
        ResolvedCompany(
            ticker=ticker,
            canonical_name=entry.name if entry else ticker,
            matched_text=str(matched_text or ticker),
            match_type=match_type,
            confidence=round(float(confidence), 3),
            position=int(position),
        )
    )


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (0 if ca == cb else 1)))
        previous = current
    return previous[-1]


def _candidate_words(normalized_query: str) -> list[str]:
    words = re.findall(r"[a-z][a-z0-9&.]{2,}", normalized_query)
    return list(dict.fromkeys(words))


def _best_alias_fuzzy(word: str) -> tuple[str, str, float] | None:
    if len(word) < 4:
        return None
    best: tuple[str, str, float] | None = None
    for alias, ticker in _ALIAS_TO_TICKER.items():
        if re.search(r"[\u4e00-\u9fff]", alias) or len(alias) < 4 or " " in alias:
            continue
        distance = _levenshtein(word, alias)
        max_distance = 1 if max(len(word), len(alias)) <= 5 else 2
        if distance > max_distance:
            continue
        confidence = max(0.55, 1.0 - (distance / max(len(word), len(alias))))
        if best is None or confidence > best[2]:
            best = (ticker, alias, confidence)
    return best


def _iter_seed_values(parsed_companies: Iterable[Any] | None, comparison_target: Any | None) -> list[str]:
    values: list[str] = []
    for item in list(parsed_companies or []):
        text = str(item or "").strip()
        if text:
            values.append(text)
    target = str(comparison_target or "").strip()
    if target:
        values.append(target)
    return values


def resolve_companies(
    raw_query: str,
    *,
    parsed_companies: Iterable[Any] | None = None,
    comparison_target: Any | None = None,
) -> EntityResolutionResult:
    normalized = _normalize_text(raw_query)
    resolved: list[ResolvedCompany] = []
    unresolved: list[str] = []

    for value in _iter_seed_values(parsed_companies, comparison_target):
        value_norm = _normalize_text(value)
        ticker = _ALIAS_TO_TICKER.get(value_norm)
        if not ticker:
            cleaned = re.sub(r"[^A-Za-z]", "", value).upper()
            ticker = cleaned if cleaned in _TARGET_TICKERS or cleaned in _TICKER_TO_ENTRY else ""
        if ticker:
            _add_company(
                resolved,
                ticker=ticker,
                matched_text=value,
                match_type="parsed",
                confidence=1.0,
                position=max(0, normalized.find(value_norm)),
            )
        elif value_norm and value_norm not in unresolved:
            unresolved.append(value)

    matches: list[tuple[int, int, str, str]] = []
    for alias in sorted(_ALIAS_TO_TICKER, key=len, reverse=True):
        ticker = _ALIAS_TO_TICKER[alias]
        if re.search(r"[\u4e00-\u9fff]", alias):
            position = normalized.find(alias)
            matched = position >= 0
        else:
            match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized)
            matched = bool(match)
            position = match.start() if match else -1
        if matched:
            matches.append((position, -len(alias), alias, ticker))
    for _position, _length_key, alias, ticker in sorted(matches):
        _add_company(resolved, ticker=ticker, matched_text=alias, match_type="alias", confidence=1.0, position=_position)

    for token in re.findall(r"\b[A-Za-z]{1,5}\b", str(raw_query or "")):
        ticker = token.upper()
        if ticker in _TARGET_TICKERS or ticker in _TICKER_TO_ENTRY:
            _add_company(
                resolved,
                ticker=ticker,
                matched_text=token,
                match_type="ticker",
                confidence=1.0,
                position=max(0, normalized.find(token.lower())),
            )

    for word in _candidate_words(normalized):
        if any(str(item.matched_text).lower() == word or item.ticker.lower() == word for item in resolved):
            continue
        fuzzy = _best_alias_fuzzy(word)
        if fuzzy:
            ticker, _alias, confidence = fuzzy
            _add_company(
                resolved,
                ticker=ticker,
                matched_text=word,
                match_type="fuzzy",
                confidence=confidence,
                position=max(0, normalized.find(word)),
            )

    if not resolved:
        for marker in _GENERIC_REFERENCES:
            if marker in normalized and marker not in unresolved:
                unresolved.append(marker)

    confidence = 0.0
    if resolved:
        resolved.sort(key=lambda item: (item.position if item.position >= 0 else 10**9, item.ticker))
        confidence = round(sum(item.confidence for item in resolved) / len(resolved), 3)
    ambiguity = bool(unresolved and not resolved)
    return EntityResolutionResult(
        resolved_companies=resolved,
        unresolved_mentions=unresolved,
        confidence=confidence,
        ambiguity=ambiguity,
    )


def companies_to_tickers(companies: Iterable[Mapping[str, Any] | ResolvedCompany]) -> list[str]:
    tickers: list[str] = []
    for item in companies:
        ticker = item.ticker if isinstance(item, ResolvedCompany) else str(item.get("ticker", ""))
        ticker = str(ticker or "").upper().strip()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers
