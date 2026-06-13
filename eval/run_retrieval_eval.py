"""Labeled RAG retrieval evaluation for filing evidence correctness."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tools.search_filings import _is_mixed_like, search_filings


PRIMARY_K = 2
PRECISION_K = 5


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().upper()


def _norm_form(value: Any) -> str:
    return _norm(value).replace(" ", "")


def _snippet(result: dict[str, Any]) -> str:
    return str(
        result.get("supporting_snippet")
        or result.get("text_snippet")
        or result.get("text")
        or ""
    )


def _term_supported(result: dict[str, Any], terms: list[str]) -> bool:
    if not terms:
        return True
    haystack = _snippet(result).lower()
    supporting_terms = {str(t).lower() for t in result.get("supporting_terms", []) if str(t).strip()}
    for term in terms:
        needle = str(term or "").lower().strip()
        if needle and (needle in haystack or needle in supporting_terms):
            return True
    return False


def _expected_period(case: dict[str, Any]) -> str:
    period = str(case.get("expected_fiscal_period") or "").strip()
    if period.upper() in {"", "UNKNOWN", "N/A", "NONE"}:
        return ""
    return _norm(period)


def _matches_expected(
    result: dict[str, Any],
    case: dict[str, Any],
    *,
    require_terms: bool = True,
) -> bool:
    if _norm(result.get("ticker")) != _norm(case.get("ticker")):
        return False
    if _norm_form(result.get("form_type")) != _norm_form(case.get("expected_form_type")):
        return False
    if _norm(result.get("section")) != _norm(case.get("expected_section")):
        return False
    expected_period = _expected_period(case)
    if expected_period and _norm(result.get("fiscal_period")) != expected_period:
        return False
    if require_terms and not _term_supported(result, list(case.get("must_include_terms", []) or [])):
        return False
    return True


def _profile_for_case(case: dict[str, Any]) -> str:
    category = str(case.get("category", "")).lower()
    if "risk" in category or "competition" in category:
        return "risk_summary"
    if "trend" in category or "liquidity" in category or "mda" in category:
        return "trend_support"
    return "summary"


def evaluate_case(case: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    top3 = results[:3]
    top5 = results[:PRECISION_K]
    top1 = results[0] if results else {}
    terms = list(case.get("must_include_terms", []) or [])
    expected_section = _norm(case.get("expected_section"))
    expected_period = _expected_period(case)

    recall_at_3 = 1.0 if any(_matches_expected(r, case) for r in top3) else 0.0
    recall_at_5 = 1.0 if any(_matches_expected(r, case) for r in top5) else 0.0
    section_accuracy = 1.0 if top1 and _norm(top1.get("section")) == expected_section else 0.0
    if expected_period:
        period_match_rate = 1.0 if any(_norm(r.get("fiscal_period")) == expected_period for r in top5) else 0.0
    else:
        period_match_rate = 1.0

    precision_denominator = max(len(top5), 1)
    citation_precision = sum(1 for r in top5 if _matches_expected(r, case, require_terms=False)) / precision_denominator
    mixed_primary = any(
        _is_mixed_like(str(r.get("section", "")), str(r.get("quality", "")))
        for r in results[:PRIMARY_K]
    )
    wrong_primary_section = bool(top1) and _norm(top1.get("section")) != expected_section
    mixed_section_primary_rate = 1.0 if (mixed_primary or wrong_primary_section) else 0.0
    snippet_support_rate = 1.0 if any(_matches_expected(r, case, require_terms=True) for r in top5) else 0.0

    failure_reasons: list[str] = []
    if recall_at_5 == 0.0:
        failure_reasons.append("missing_expected_evidence")
    if section_accuracy == 0.0:
        failure_reasons.append("wrong_primary_section")
    if period_match_rate == 0.0:
        failure_reasons.append("wrong_period")
    if snippet_support_rate == 0.0:
        failure_reasons.append("snippet_unsupported")
    if mixed_primary:
        failure_reasons.append("mixed_section_primary")

    return {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "query": case.get("query", ""),
        "ticker": case.get("ticker", ""),
        "expected": {
            "form_type": case.get("expected_form_type", ""),
            "section": case.get("expected_section", ""),
            "fiscal_period": case.get("expected_fiscal_period", ""),
            "must_include_terms": terms,
        },
        "metrics": {
            "recall_at_3": recall_at_3,
            "recall_at_5": recall_at_5,
            "section_accuracy": section_accuracy,
            "period_match_rate": period_match_rate,
            "citation_precision": citation_precision,
            "mixed_section_primary_rate": mixed_section_primary_rate,
            "snippet_support_rate": snippet_support_rate,
        },
        "failure_reasons": failure_reasons,
        "top_results": [
            {
                "rank": i + 1,
                "ticker": r.get("ticker", ""),
                "form_type": r.get("form_type", ""),
                "fiscal_period": r.get("fiscal_period", ""),
                "section": r.get("section", ""),
                "quality": r.get("quality", ""),
                "final_score": r.get("final_score", r.get("score", 0.0)),
                "score_breakdown": r.get("score_breakdown", {}),
                "snippet": _snippet(r)[:500],
            }
            for i, r in enumerate(top5)
        ],
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "recall_at_3",
        "recall_at_5",
        "section_accuracy",
        "period_match_rate",
        "citation_precision",
        "mixed_section_primary_rate",
        "snippet_support_rate",
    ]
    if not records:
        return {name: 0.0 for name in metric_names} | {"case_count": 0, "pass": False}
    summary = {
        name: round(
            sum(float(r["metrics"].get(name, 0.0)) for r in records) / len(records),
            4,
        )
        for name in metric_names
    }
    summary["case_count"] = len(records)
    summary["pass"] = (
        summary["recall_at_5"] >= 0.85
        and summary["section_accuracy"] >= 0.80
        and summary["mixed_section_primary_rate"] <= 0.10
        and summary["snippet_support_rate"] >= 0.85
    )
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(r.get("category", "")) for r in records}):
        rows = [r for r in records if str(r.get("category", "")) == category]
        by_category[category] = summarize_records(rows) if len(rows) != len(records) else {}
    summary["by_category"] = by_category
    return summary


def load_gold(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        missing = [
            key
            for key in (
                "id",
                "category",
                "query",
                "ticker",
                "expected_form_type",
                "expected_section",
                "expected_fiscal_period",
                "must_include_terms",
            )
            if key not in item
        ]
        if missing:
            raise ValueError(f"{path}:{line_no} missing required fields: {missing}")
        if not isinstance(item.get("must_include_terms"), list) or not item["must_include_terms"]:
            raise ValueError(f"{path}:{line_no} must_include_terms must be a non-empty list")
        records.append(item)
    return records


def run_retrieval_eval(gold_path: Path, top_k: int = 5, limit: int | None = None) -> dict[str, Any]:
    cases = load_gold(gold_path)
    if limit is not None:
        cases = cases[: max(0, limit)]
    records: list[dict[str, Any]] = []
    for case in cases:
        payload = {
            "ticker": str(case["ticker"]),
            "query": str(case["query"]),
            "top_k": max(top_k, 5),
            "form_type": str(case["expected_form_type"]),
            "retrieval_profile": _profile_for_case(case),
            "target_periods": [str(case["expected_fiscal_period"])],
            "max_per_filing": 2,
            "max_per_section": 2,
        }
        results = search_filings.invoke(payload)
        records.append(evaluate_case(case, list(results or [])))
    summary = summarize_records(records)
    failed = [r for r in records if r["failure_reasons"]]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold_path": str(gold_path),
        "summary": summary,
        "records": records,
        "failed_cases": failed[:20],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        "# Retrieval Gold Eval",
        "",
        f"- pass: {'PASS' if summary.get('pass') else 'FAIL'}",
        f"- case_count: {summary.get('case_count', 0)}",
        f"- recall_at_3: {summary.get('recall_at_3', 0.0):.2%}",
        f"- recall_at_5: {summary.get('recall_at_5', 0.0):.2%}",
        f"- section_accuracy: {summary.get('section_accuracy', 0.0):.2%}",
        f"- period_match_rate: {summary.get('period_match_rate', 0.0):.2%}",
        f"- citation_precision: {summary.get('citation_precision', 0.0):.2%}",
        f"- mixed_section_primary_rate: {summary.get('mixed_section_primary_rate', 0.0):.2%}",
        f"- snippet_support_rate: {summary.get('snippet_support_rate', 0.0):.2%}",
        "",
        "## Failed Cases",
    ]
    failed = report.get("failed_cases", [])
    if not failed:
        lines.append("- none")
        return "\n".join(lines) + "\n"

    for case in failed[:20]:
        expected = case.get("expected", {})
        lines.extend(
            [
                "",
                f"### {case.get('id', '')}",
                f"- query: {case.get('query', '')}",
                f"- reasons: {', '.join(case.get('failure_reasons', []))}",
                (
                    "- expected: "
                    f"{case.get('ticker', '')} {expected.get('form_type', '')} "
                    f"{expected.get('fiscal_period', '')} {expected.get('section', '')}"
                ),
            ]
        )
        for result in case.get("top_results", [])[:3]:
            lines.append(
                "- actual: "
                f"rank={result.get('rank')} {result.get('ticker')} {result.get('form_type')} "
                f"{result.get('fiscal_period')} {result.get('section')} "
                f"score={result.get('final_score')} snippet={result.get('snippet', '')[:180]}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run labeled retrieval gold evaluation.")
    parser.add_argument("--gold", default="eval/retrieval_gold.jsonl")
    parser.add_argument("--out-json", default="eval/retrieval_report.json")
    parser.add_argument("--out-md", default="eval/retrieval_report.md")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    report = run_retrieval_eval(Path(args.gold), top_k=args.top_k, limit=args.limit)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
