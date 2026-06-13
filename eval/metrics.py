"""Evaluation metrics for the financial-analysis agent.

Six metrics:
  1. answer_accuracy   — LLM-as-judge (or keyword fallback)
  2. numerical_accuracy — extract numbers, compare with ±1% tolerance
  3. retrieval_hit_rate — did we retrieve chunks from the expected sources?
  4. citation_accuracy  — do generated citations match expected sources?
  5. tool_success_rate  — were all tool invocations error-free?
  6. latency            — wall-clock seconds
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── 1. answer_accuracy ────────────────────────────────────────────

def answer_accuracy(
    answer: str,
    expected_answer: str,
    key_numbers: list[float] | None = None,
) -> float:
    """Keyword-overlap score: what fraction of expected keywords appear in the answer?

    Falls back to numerical check if expected_answer is a bare number.
    Returns 0.0-1.0.
    """
    if not answer or not expected_answer:
        return 0.0

    # If expected is a single number, delegate to numerical check
    try:
        num = float(expected_answer.replace(",", ""))
        return 1.0 if _number_in_text(num, answer) else 0.0
    except ValueError:
        pass

    keywords = _extract_keywords(expected_answer)
    if not keywords:
        return 0.0

    hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    return hits / len(keywords)


def _extract_keywords(text: str) -> list[str]:
    """Split expected answer into meaningful keywords (>1 char, no stopwords)."""
    stopwords = {"的", "是", "和", "在", "了", "有", "与", "从", "到", "约", "等",
                 "a", "the", "is", "and", "or", "of", "to", "in", "for", "with"}
    tokens = re.findall(r"[\w%]+", text)
    return [t for t in tokens if len(t) > 1 and t.lower() not in stopwords]


# ── 2. numerical_accuracy ────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _parse_numbers(text: str) -> list[float]:
    """Extract all numeric values from text."""
    nums = []
    for m in _NUM_RE.finditer(text):
        try:
            nums.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return nums


def _number_in_text(expected: float, text: str, tolerance: float = 0.01) -> bool:
    """Check if expected number appears in text within tolerance."""
    found = _parse_numbers(text)
    if expected == 0:
        return 0.0 in found
    for f in found:
        if f == 0:
            continue
        ratio = abs(f - expected) / abs(expected)
        if ratio <= tolerance:
            return True
        # Also match scaled versions (billions, millions)
        for scale in [1e9, 1e6, 1e3, 100, 1]:
            scaled = expected / scale
            if scaled != 0 and abs(f - scaled) / abs(scaled) <= tolerance:
                return True
        # Match percentage representation: 0.3537 shown as "35.37%"
        if 0 < abs(expected) < 1:
            pct_val = expected * 100
            if abs(f - pct_val) / abs(pct_val) <= tolerance:
                return True
    return False


def numerical_accuracy(
    answer: str,
    key_numbers: list[float],
) -> float:
    """Fraction of expected key numbers found in the answer (±1% tolerance).

    Returns 0.0-1.0.
    """
    if not key_numbers:
        return 1.0  # no numbers to check → pass
    hits = sum(1 for n in key_numbers if _number_in_text(n, answer))
    return hits / len(key_numbers)


# ── 3. retrieval_hit_rate ─────────────────────────────────────────

def retrieval_hit_rate(
    retrieved_docs: list[dict[str, Any]],
    expected_citations: list[str],
) -> float:
    """Fraction of expected citation sources found among retrieved docs.

    expected_citations are partial strings like "AAPL 10-K".
    """
    if not expected_citations:
        return 1.0
    doc_strings = []
    for d in retrieved_docs:
        parts = [
            d.get("ticker", ""),
            d.get("form_type", ""),
            d.get("fiscal_period", ""),
            d.get("section", ""),
        ]
        doc_strings.append(" ".join(parts).upper())

    hits = 0
    for ec in expected_citations:
        ec_upper = ec.upper()
        if any(ec_upper in ds for ds in doc_strings):
            hits += 1
    return hits / len(expected_citations)


# ── 4. citation_accuracy ─────────────────────────────────────────

def citation_accuracy(
    citations: list[dict[str, Any]],
    expected_citations: list[str],
) -> float:
    """Fraction of expected sources referenced in the generated citations."""
    if not expected_citations:
        return 1.0
    cite_strings = []
    for c in citations:
        parts = [
            c.get("ticker", c.get("source", "")),
            c.get("form_type", c.get("filing_type", "")),
            c.get("fiscal_period", c.get("period", "")),
        ]
        cite_strings.append(" ".join(parts).upper())

    hits = 0
    for ec in expected_citations:
        ec_upper = ec.upper()
        if any(ec_upper in cs for cs in cite_strings):
            hits += 1
    return hits / len(expected_citations)


# ── 5. tool_success_rate ──────────────────────────────────────────

def tool_success_rate(tool_results: list[dict[str, Any]]) -> float:
    """Fraction of tool invocations that completed without error."""
    if not tool_results:
        return 1.0
    ok = sum(1 for tr in tool_results if "error" not in tr)
    return ok / len(tool_results)


# ── 6. latency (just a pass-through, measured externally) ─────────

def latency_score(elapsed_seconds: float, budget: float = 60.0) -> float:
    """1.0 if under budget, linearly decays to 0.0 at 3× budget."""
    if elapsed_seconds <= budget:
        return 1.0
    if elapsed_seconds >= budget * 3:
        return 0.0
    return 1.0 - (elapsed_seconds - budget) / (budget * 2)


def event_alignment_accuracy(
    event_results: list[dict[str, Any]],
    expected_event_type: str | None = None,
    expected_latest_n: int | None = None,
    required: bool = False,
) -> float:
    """Check whether event rows align with expected event filters.

    Returns 1.0 for non-required tasks when no event evidence is expected.
    """
    rows: list[dict[str, Any]] = []
    for item in event_results or []:
        data = item.get("data", {}) if isinstance(item, dict) else {}
        rows.extend(data.get("events", []) if isinstance(data, dict) else [])

    if not rows:
        return 0.0 if required else 1.0

    checks: list[bool] = [True]
    if expected_event_type:
        exp = str(expected_event_type).upper().replace("-", "")
        checks.append(
            any(str(r.get("event_type", "")).upper().replace("-", "") == exp for r in rows)
        )
    if expected_latest_n:
        checks.append(len(rows) <= max(1, int(expected_latest_n)))
    passed = sum(1 for x in checks if x)
    return passed / len(checks)


def event_numeric_accuracy(
    output: dict[str, Any],
    key_numbers: list[float],
    required: bool = False,
) -> float:
    """Numerical accuracy based on the market_reaction output block."""
    market = output.get("market_reaction", {}) if isinstance(output, dict) else {}
    events = market.get("events", []) if isinstance(market, dict) else []
    if not events:
        return 0.0 if required and key_numbers else (1.0 if not key_numbers else 0.0)
    text_parts: list[str] = []
    for event in events:
        for key in ("return_1d", "return_3d", "return_5d", "return_10d"):
            text_parts.append(str(event.get(key, "")))
    return numerical_accuracy(" ".join(text_parts), key_numbers)


def event_tool_success_rate(tool_results: list[dict[str, Any]], required: bool = False) -> float:
    """Success rate for query_event_price_window calls."""
    calls = [tr for tr in (tool_results or []) if tr.get("tool") == "query_event_price_window"]
    if not calls:
        return 0.0 if required else 1.0
    ok = sum(1 for tr in calls if "error" not in tr)
    return ok / len(calls)


def market_reaction_block_presence_rate(output: dict[str, Any], required: bool = False) -> float:
    """Whether output includes a non-empty market_reaction block."""
    market = output.get("market_reaction", {}) if isinstance(output, dict) else {}
    if not isinstance(market, dict) or not market:
        return 0.0 if required else 1.0
    has_title = bool(str(market.get("title", "")).strip())
    has_events = isinstance(market.get("events", []), list)
    return 1.0 if has_title and has_events else 0.0


# ── 7. v2 structured correctness metrics ─────────────────────────


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _norm_upper(value: Any) -> str:
    return _norm_text(value).upper()


def _norm_event_type(value: Any) -> str:
    return _norm_upper(value).replace("-", "")


def _norm_metric(value: Any) -> str:
    text = _norm_text(value).lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "sales": "revenue",
        "revenues": "revenue",
        "total_revenue": "revenue",
        "net_income_loss": "net_income",
        "net_income_attributable_to_common_stockholders": "net_income",
        "earnings_per_share": "eps",
        "diluted_eps": "eps",
        "gross_margin_pct": "gross_margin",
        "gross_profit_margin": "gross_margin",
        "operating_income_margin": "operating_margin",
        "operating_margin_pct": "operating_margin",
    }
    return aliases.get(text, text)


def _norm_period_type(value: Any) -> str:
    text = _norm_text(value).lower()
    aliases = {
        "q": "quarterly",
        "quarter": "quarterly",
        "quarterly": "quarterly",
        "fy": "annual",
        "year": "annual",
        "annual": "annual",
        "yearly": "annual",
    }
    return aliases.get(text, text)


def _norm_unit(value: Any) -> str:
    text = _norm_text(value).lower().replace(" ", "_")
    aliases = {
        "$": "usd",
        "dollar": "usd",
        "dollars": "usd",
        "usd_millions": "usd",
        "usd_thousands": "usd",
        "percentage": "ratio",
        "percent": "ratio",
        "%": "ratio",
        "pct": "ratio",
        "share": "shares",
        "per_share": "usd_per_share",
    }
    return aliases.get(text, text)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    text = _norm_text(value).replace(",", "")
    if not text:
        return None
    try:
        return float(text.rstrip("%"))
    except ValueError:
        return None


def _field_matches(actual: Any, expected: Any, *, normalizer=_norm_upper) -> bool:
    expected_norm = normalizer(expected)
    if not expected_norm:
        return True
    return normalizer(actual) == expected_norm


def _failure(
    failure_type: str,
    expected: dict[str, Any],
    actual: dict[str, Any] | None,
    message: str,
) -> dict[str, Any]:
    return {
        "type": failure_type,
        "expected": expected,
        "actual": actual or {},
        "message": message,
    }


def _candidate_snapshot(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not candidate:
        return {}
    keys = (
        "ticker",
        "metric",
        "period_type",
        "period_end",
        "event_type",
        "event_date",
        "fiscal_period",
        "value",
        "unit",
        "source_kind",
        "form_type",
        "section",
    )
    return {k: candidate.get(k) for k in keys if candidate.get(k) not in (None, "")}


def _numeric_identity_matches(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        _field_matches(candidate.get("ticker"), expected.get("ticker"))
        and _field_matches(candidate.get("metric"), expected.get("metric"), normalizer=_norm_metric)
    )


def _numeric_period_matches(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        _field_matches(candidate.get("period_type"), expected.get("period_type"), normalizer=_norm_period_type)
        and _field_matches(candidate.get("period_end"), expected.get("period_end"), normalizer=_norm_text)
    )


def _numeric_unit_matches(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    expected_unit = _norm_unit(expected.get("unit"))
    if not expected_unit:
        return True
    actual_unit = _norm_unit(candidate.get("unit"))
    return actual_unit == expected_unit


def _numeric_value_matches(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    expected_value = _to_float(expected.get("expected_value", expected.get("value")))
    if expected_value is None:
        return True
    actual_value = _to_float(candidate.get("value", candidate.get("actual_value")))
    if actual_value is None:
        return False
    tolerance_pct = _to_float(expected.get("tolerance_pct"))
    tolerance_pct = 0.01 if tolerance_pct is None else max(0.0, tolerance_pct)
    if expected_value == 0:
        return abs(actual_value) <= tolerance_pct
    return abs(actual_value - expected_value) / abs(expected_value) <= tolerance_pct


def _dedupe_dicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = repr(sorted((str(k), repr(v)) for k, v in row.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _collect_numeric_candidates(output: dict[str, Any], trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect structured numeric evidence from all trace/output locations."""
    out: list[dict[str, Any]] = []
    trace_output = _as_dict(trace.get("output"))
    sources = [
        trace.get("numeric_evidence"),
        trace.get("numeric_citations"),
        trace_output.get("numeric_evidence"),
        output.get("numeric_evidence"),
        output.get("numbers"),
    ]
    for source in sources:
        for raw in _as_list(source):
            if not isinstance(raw, dict):
                continue
            candidate = dict(raw)
            if "value" not in candidate:
                for alias in ("actual_value", "display_value", "amount"):
                    if alias in candidate:
                        candidate["value"] = candidate.get(alias)
                        break
            out.append(candidate)
    return _dedupe_dicts(out)


def numeric_fact_accuracy(
    output: dict[str, Any],
    trace: dict[str, Any],
    expected_facts: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Validate exact numeric facts against structured evidence.

    A fact only passes when ticker, metric, period, unit, and value all match the
    expected v2 benchmark fields. Empty expected_facts are neutral for backward
    compatibility with legacy cases.
    """
    facts = [f for f in _as_list(expected_facts) if isinstance(f, dict)]
    if not facts:
        return 1.0, []

    candidates = _collect_numeric_candidates(_as_dict(output), _as_dict(trace))
    passed = 0
    failures: list[dict[str, Any]] = []
    for fact in facts:
        id_matches = [c for c in candidates if _numeric_identity_matches(c, fact)]
        if not id_matches:
            failures.append(
                _failure(
                    "missing_expected_fact",
                    fact,
                    None,
                    f"Expected {_norm_upper(fact.get('ticker'))} {_norm_metric(fact.get('metric'))} but found no matching numeric evidence",
                )
            )
            continue

        period_matches = [c for c in id_matches if _numeric_period_matches(c, fact)]
        if not period_matches:
            failures.append(
                _failure(
                    "wrong_period",
                    {"period_type": fact.get("period_type"), "period_end": fact.get("period_end"), **fact},
                    _candidate_snapshot(id_matches[0]),
                    (
                        f"Expected {_norm_upper(fact.get('ticker'))} {_norm_metric(fact.get('metric'))} "
                        f"for {fact.get('period_end')} but matched {id_matches[0].get('period_end')}"
                    ),
                )
            )
            continue

        unit_matches = [c for c in period_matches if _numeric_unit_matches(c, fact)]
        if not unit_matches:
            failures.append(
                _failure(
                    "wrong_unit_or_scale",
                    fact,
                    _candidate_snapshot(period_matches[0]),
                    (
                        f"Expected unit {_norm_unit(fact.get('unit')) or '(any)'} but matched "
                        f"{_norm_unit(period_matches[0].get('unit')) or '(missing)'}"
                    ),
                )
            )
            continue

        value_matches = [c for c in unit_matches if _numeric_value_matches(c, fact)]
        if not value_matches:
            failures.append(
                _failure(
                    "wrong_value",
                    fact,
                    _candidate_snapshot(unit_matches[0]),
                    (
                        f"Expected value {fact.get('expected_value', fact.get('value'))} "
                        f"but matched {unit_matches[0].get('value')}"
                    ),
                )
            )
            continue

        passed += 1

    return passed / len(facts), failures


def period_accuracy(
    output: dict[str, Any],
    trace: dict[str, Any],
    expected_facts: list[dict[str, Any]],
    expected_events: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Validate that numeric and event records bind to the expected period/date."""
    checks: list[tuple[bool, dict[str, Any] | None]] = []
    failures: list[dict[str, Any]] = []

    candidates = _collect_numeric_candidates(_as_dict(output), _as_dict(trace))
    for fact in [f for f in _as_list(expected_facts) if isinstance(f, dict)]:
        id_matches = [c for c in candidates if _numeric_identity_matches(c, fact)]
        if not id_matches:
            failure = _failure(
                "missing_expected_fact",
                fact,
                None,
                f"Expected {_norm_upper(fact.get('ticker'))} {_norm_metric(fact.get('metric'))} period evidence was missing",
            )
            failures.append(failure)
            checks.append((False, failure))
            continue
        if _numeric_period_matches(id_matches[0], fact) or any(_numeric_period_matches(c, fact) for c in id_matches):
            checks.append((True, None))
            continue
        failure = _failure(
            "wrong_period",
            {"period_type": fact.get("period_type"), "period_end": fact.get("period_end"), **fact},
            _candidate_snapshot(id_matches[0]),
            (
                f"Expected {_norm_upper(fact.get('ticker'))} {_norm_metric(fact.get('metric'))} "
                f"for {fact.get('period_end')} but matched {id_matches[0].get('period_end')}"
            ),
        )
        failures.append(failure)
        checks.append((False, failure))

    events = _collect_event_candidates(_as_dict(output), _as_dict(trace))
    for expected_event in [e for e in _as_list(expected_events) if isinstance(e, dict)]:
        matches = _event_identity_matches(events, expected_event, include_event_date=False)
        if not matches:
            failure = _failure(
                "missing_expected_event",
                expected_event,
                None,
                f"Expected event for {_norm_upper(expected_event.get('ticker'))} was missing",
            )
            failures.append(failure)
            checks.append((False, failure))
            continue
        date_matches = [e for e in matches if _field_matches(e.get("event_date"), expected_event.get("event_date"), normalizer=_norm_text)]
        if date_matches:
            checks.append((True, None))
            continue
        failure = _failure(
            "wrong_event_date",
            expected_event,
            _candidate_snapshot(matches[0]),
            f"Expected event_date {expected_event.get('event_date')} but matched {matches[0].get('event_date')}",
        )
        failures.append(failure)
        checks.append((False, failure))

    if not checks:
        return 1.0, []
    passed = sum(1 for ok, _ in checks if ok)
    return passed / len(checks), failures


def _collect_citation_candidates(trace: dict[str, Any]) -> list[dict[str, Any]]:
    trace = _as_dict(trace)
    output = _as_dict(trace.get("output"))
    raw_sources = [
        trace.get("citations"),
        trace.get("text_citations"),
        trace.get("document_citations"),
        output.get("text_evidence"),
    ]
    out: list[dict[str, Any]] = []
    for source in raw_sources:
        for raw in _as_list(source):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if "form_type" not in item and "filing_type" in item:
                item["form_type"] = item.get("filing_type")
            if "fiscal_period" not in item and "period" in item:
                item["fiscal_period"] = item.get("period")
            if "source_kind" not in item:
                item["source_kind"] = "document"
            out.append(item)
    return _dedupe_dicts(out)


def _citation_structure_matches(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        _field_matches(candidate.get("source_kind"), expected.get("source_kind"), normalizer=lambda x: _norm_text(x).lower())
        and _field_matches(candidate.get("ticker"), expected.get("ticker"))
        and _field_matches(candidate.get("form_type"), expected.get("form_type"))
        and _field_matches(candidate.get("section"), expected.get("section"))
        and _field_matches(candidate.get("fiscal_period"), expected.get("fiscal_period"), normalizer=_norm_text)
    )


def _citation_term_matches(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    terms = [_norm_text(t).lower() for t in _as_list(expected.get("must_include_terms")) if _norm_text(t)]
    if not terms:
        return True
    snippet = " ".join(
        _norm_text(candidate.get(k)).lower()
        for k in ("text_snippet", "snippet", "content", "summary", "title")
    )
    if not snippet:
        return False
    mode = _norm_text(expected.get("must_include_mode") or "any").lower()
    hits = [term in snippet for term in terms]
    return all(hits) if mode == "all" else any(hits)


def citation_required_accuracy(
    trace: dict[str, Any],
    expected_text_evidence: list[dict[str, Any]],
    requires_citation: bool,
) -> tuple[float, list[dict[str, Any]]]:
    """Validate required document citations.

    If requires_citation is false and there is no expected text evidence, the
    case is neutral. Required citation cases cannot pass just because the old
    expected_citations list was empty.
    """
    expected_rows = [e for e in _as_list(expected_text_evidence) if isinstance(e, dict)]
    citations = _collect_citation_candidates(_as_dict(trace))
    if not expected_rows:
        if requires_citation and not citations:
            return 0.0, [
                _failure(
                    "missing_citation",
                    {"requires_citation": True},
                    None,
                    "Citation is required but no citation evidence was produced",
                )
            ]
        return 1.0, []

    passed = 0
    failures: list[dict[str, Any]] = []
    for expected in expected_rows:
        if not citations:
            failures.append(
                _failure(
                    "missing_citation",
                    expected,
                    None,
                    "Expected document citation but citation list was empty",
                )
            )
            continue

        structural = [c for c in citations if _citation_structure_matches(c, expected)]
        if not structural:
            same_doc = [
                c for c in citations
                if _field_matches(c.get("ticker"), expected.get("ticker"))
                and _field_matches(c.get("form_type"), expected.get("form_type"))
            ]
            reason = "retrieval_wrong_section" if same_doc else "missing_citation"
            failures.append(
                _failure(
                    reason,
                    expected,
                    _candidate_snapshot((same_doc or citations)[0]),
                    (
                        f"Expected citation {_norm_upper(expected.get('ticker'))} "
                        f"{_norm_upper(expected.get('form_type'))} {expected.get('section', '')} "
                        "but no matching citation was found"
                    ),
                )
            )
            continue

        term_matches = [c for c in structural if _citation_term_matches(c, expected)]
        if not term_matches:
            failures.append(
                _failure(
                    "missing_citation",
                    {**expected, "must_include_terms": expected.get("must_include_terms", [])},
                    _candidate_snapshot(structural[0]),
                    "Citation matched the source but did not include required evidence terms",
                )
            )
            continue

        passed += 1

    return passed / len(expected_rows), failures


def _collect_event_candidates(output: dict[str, Any], trace: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    market = _as_dict(output.get("market_reaction"))
    for raw in _as_list(market.get("events")):
        if isinstance(raw, dict):
            events.append(dict(raw))

    trace = _as_dict(trace)
    for result in _as_list(trace.get("event_results")):
        if not isinstance(result, dict):
            continue
        data = _as_dict(result.get("data"))
        for raw in _as_list(data.get("events")):
            if isinstance(raw, dict):
                events.append(dict(raw))

    normalized: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        if "event_type" not in row and "form_type" in row:
            row["event_type"] = row.get("form_type")
        returns = _as_dict(row.get("returns"))
        for key, value in returns.items():
            row.setdefault(key, value)
        normalized.append(row)
    return _dedupe_dicts(normalized)


def _event_identity_matches(
    events: list[dict[str, Any]],
    expected: dict[str, Any],
    *,
    include_event_date: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        type_expected = _norm_event_type(expected.get("event_type"))
        type_ok = not type_expected or type_expected == "ANY" or _norm_event_type(event.get("event_type")) == type_expected
        ok = (
            _field_matches(event.get("ticker"), expected.get("ticker"))
            and type_ok
            and _field_matches(event.get("fiscal_period"), expected.get("fiscal_period"), normalizer=_norm_text)
        )
        if include_event_date:
            ok = ok and _field_matches(event.get("event_date"), expected.get("event_date"), normalizer=_norm_text)
        if ok:
            out.append(event)
    return out


def _event_return_matches(event: dict[str, Any], expected: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    expected_returns = _as_dict(expected.get("returns"))
    if not expected_returns:
        return True, {}
    tolerance_abs = _to_float(expected.get("tolerance_abs"))
    tolerance_abs = 0.0005 if tolerance_abs is None else max(0.0, tolerance_abs)
    actual: dict[str, Any] = {}
    for key, expected_value_raw in expected_returns.items():
        expected_value = _to_float(expected_value_raw)
        if expected_value is None:
            continue
        actual_value = _to_float(event.get(key))
        actual[key] = event.get(key)
        if actual_value is None or abs(actual_value - expected_value) > tolerance_abs:
            return False, actual
    return True, actual


def event_window_accuracy(
    output: dict[str, Any],
    trace: dict[str, Any],
    expected_events: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Validate event-date and return-window correctness."""
    expected_rows = [e for e in _as_list(expected_events) if isinstance(e, dict)]
    if not expected_rows:
        return 1.0, []

    events = _collect_event_candidates(_as_dict(output), _as_dict(trace))
    passed = 0
    failures: list[dict[str, Any]] = []
    for expected in expected_rows:
        identity_matches = _event_identity_matches(events, expected, include_event_date=False)
        if not identity_matches:
            failures.append(
                _failure(
                    "missing_expected_event",
                    expected,
                    None,
                    f"Expected event for {_norm_upper(expected.get('ticker'))} was missing",
                )
            )
            continue

        date_matches = _event_identity_matches(events, expected, include_event_date=True)
        if not date_matches:
            failures.append(
                _failure(
                    "wrong_event_date",
                    expected,
                    _candidate_snapshot(identity_matches[0]),
                    f"Expected event_date {expected.get('event_date')} but matched {identity_matches[0].get('event_date')}",
                )
            )
            continue

        return_ok = False
        last_actual: dict[str, Any] = {}
        for event in date_matches:
            return_ok, last_actual = _event_return_matches(event, expected)
            if return_ok:
                break
        if not return_ok:
            failures.append(
                _failure(
                    "wrong_event_return",
                    expected,
                    {**_candidate_snapshot(date_matches[0]), **last_actual},
                    f"Expected event returns {expected.get('returns', {})} within tolerance",
                )
            )
            continue

        passed += 1

    return passed / len(expected_rows), failures


def correctness_summary(record_metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize per-record structured correctness failures."""
    failure_reasons = [
        r for r in _as_list(record_metrics.get("failure_reasons"))
        if isinstance(r, dict) and _norm_text(r.get("type"))
    ]
    thresholds = {
        "answer_accuracy": 0.70,
        "numerical_accuracy": 0.90,
        "numeric_fact_accuracy": 0.90,
        "period_accuracy": 0.90,
        "citation_required_accuracy": 0.85,
        "event_window_accuracy": 0.90,
    }
    below_threshold = [
        name for name, threshold in thresholds.items()
        if name in record_metrics and float(record_metrics.get(name, 0.0) or 0.0) < threshold
    ]
    primary = ""
    if failure_reasons:
        primary = str(failure_reasons[0].get("type", ""))
    elif below_threshold:
        primary = f"{below_threshold[0]}_below_threshold"
    return {
        "correctness_pass": not failure_reasons and not below_threshold,
        "failure_reasons": failure_reasons,
        "primary_failure_type": primary,
    }
