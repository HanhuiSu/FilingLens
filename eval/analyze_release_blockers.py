#!/usr/bin/env python3
"""Analyze Phase 5 blocker failures for numerical accuracy and output stability.

Input:
  - report_phase5_formal.json
  - trace directory

Output:
  - blocker_failure_attribution.json
  - blocker_failure_attribution.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any


ALLOWED_ERROR_TYPES = {
    "wrong_value",
    "wrong_period",
    "wrong_unit_or_scale",
    "llm_rephrased_number",
    "missing_numeric_evidence",
    "missing_citation",
    "wrong_event_date",
    "wrong_event_return",
    "missing_expected_fact",
    "missing_expected_event",
    "retrieval_wrong_section",
    "output_missing_field",
    "output_wrong_shape",
    "fallback_rendering_issue",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        row = json.loads(raw)
        qid = str(row.get("id", ""))
        if qid:
            out[qid] = row
    return out


def _read_trace(trace_dir: Path, trace_id: str) -> dict[str, Any]:
    path = trace_dir / f"{trace_id}.json"
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _reason_list(trace: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for x in trace.get("unsupported_claims", []) or []:
        if not isinstance(x, dict):
            continue
        reason = str(x.get("reason", "")).strip()
        if reason:
            out.append(reason)
    return out


def _classify_numeric_failure(record: dict[str, Any], trace: dict[str, Any]) -> tuple[str, str, str]:
    reasons = _reason_list(trace)
    reasons_blob = "|".join(reasons)
    numeric_evidence = trace.get("numeric_evidence", []) or []
    numeric_citations = trace.get("numeric_citations", []) or []
    answer_preview = str(record.get("answer_preview", "") or trace.get("final_answer", ""))

    if "numeric_claim_unit_mismatch" in reasons_blob:
        return (
            "wrong_unit_or_scale",
            "numeric claim unit mismatched structured evidence",
            "numeric_unit_normalization",
        )
    if "numeric_claim_value_mismatch" in reasons_blob:
        return (
            "wrong_value",
            "numeric claim value mismatched structured evidence",
            "numeric_hard_gate",
        )
    if (
        "numeric_claim_period_mismatch" in reasons_blob
        or "period_type_mismatch" in reasons_blob
        or "quarter_mismatch" in reasons_blob
        or "year_mismatch" in reasons_blob
        or "period_consistency" in reasons_blob
        or "no_common_period_for_same_period_comparison" in reasons_blob
    ):
        return (
            "wrong_period",
            "period consistency gate rejected numeric claims",
            "numeric_period_binding",
        )
    if "estimation_word_detected" in reasons_blob:
        return (
            "llm_rephrased_number",
            "numeric wording rejected by estimation phrase rule",
            "disable_model_numeric_generation",
        )
    if "no_numeric_citation_for_time_check" in reasons_blob or (numeric_evidence and not numeric_citations):
        return (
            "missing_numeric_evidence",
            "numeric evidence existed but no numeric citations bound to final claims",
            "deterministic_numeric_claim_binding",
        )
    if ("亿美元" in answer_preview or "billion" in answer_preview.lower()) and "6812.7" in answer_preview:
        return (
            "wrong_unit_or_scale",
            "scaled display drifted from raw structured value",
            "numeric_display_raw_value_only",
        )
    return (
        "wrong_value",
        "numeric statement diverged from expected key numbers",
        "deterministic_numeric_claim_binding",
    )


def _classify_output_failure(record: dict[str, Any], trace: dict[str, Any]) -> tuple[str, str, str]:
    output = trace.get("output", {})
    task_type = str(trace.get("task_type", record.get("actual_task_type", "")))
    view = output.get("view", {}) if isinstance(output, dict) else {}

    if task_type == "fact_qa":
        headline = view.get("headline_metric", {}) if isinstance(view, dict) else {}
        if not isinstance(headline, dict) or not headline:
            return (
                "output_missing_field",
                "fact_qa headline_metric missing or empty",
                "output_default_blocks",
            )

    if task_type == "company_comparison":
        table = view.get("comparison_table", {}) if isinstance(view, dict) else {}
        rows = table.get("rows", []) if isinstance(table, dict) else []
        if not rows:
            return (
                "output_wrong_shape",
                "comparison_table rows empty",
                "comparison_placeholder_row",
            )
        if not str(view.get("comparison_basis_line", "")).strip():
            return (
                "output_missing_field",
                "comparison_basis_line missing",
                "output_default_blocks",
            )

    if str(view.get("kind", "")) != task_type:
        return (
            "fallback_rendering_issue",
            "view.kind mismatched task_type",
            "task_type_template_alignment",
        )

    return (
        "output_wrong_shape",
        "output payload shape drifted from task contract",
        "programmatic_output_assembler",
    )


def _fix_bucket_for_failure_type(error_type: str) -> str:
    return {
        "wrong_value": "numeric_hard_gate",
        "wrong_period": "numeric_period_binding",
        "wrong_unit_or_scale": "numeric_unit_normalization",
        "missing_expected_fact": "deterministic_numeric_claim_binding",
        "missing_citation": "citation_required_gate",
        "retrieval_wrong_section": "retrieval_section_binding",
        "wrong_event_date": "event_date_binding",
        "wrong_event_return": "event_window_return_binding",
        "missing_expected_event": "event_tool_contract",
    }.get(error_type, "structured_correctness_gate")


def _row_common(
    record: dict[str, Any],
    baseline_item: dict[str, Any],
) -> dict[str, Any]:
    qid = str(record.get("id", ""))
    expected_task = str(record.get("expected_task_type", baseline_item.get("expected_task_type", "")))
    actual_task = str(record.get("actual_task_type", ""))
    expected_key_numbers = baseline_item.get("key_numbers", [])
    return {
        "question_id": qid,
        "task_type": actual_task or expected_task,
        "expected": {
            "task_type": expected_task,
            "key_numbers": expected_key_numbers,
        },
        "actual": {
            "task_type": actual_task,
            "answer_preview": str(record.get("answer_preview", "")),
        },
    }


def build_failure_rows(
    report: dict[str, Any],
    trace_dir: Path,
    baseline_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in report.get("records", []):
        qid = str(record.get("id", ""))
        trace = _read_trace(trace_dir, str(record.get("trace_id", "")))
        common = _row_common(record, baseline_map.get(qid, {}))

        numerical_accuracy = float(record.get("metrics", {}).get("numerical_accuracy", 1.0) or 0.0)
        output_contract_ok = bool(record.get("signals", {}).get("output_contract_ok", True))
        structured_reasons = [
            x for x in (record.get("failure_reasons", []) or [])
            if isinstance(x, dict) and str(x.get("type", "")).strip()
        ]

        if structured_reasons:
            for reason in structured_reasons:
                err = str(reason.get("type", ""))
                rows.append(
                    {
                        **common,
                        "expected": reason.get("expected", common.get("expected", {})),
                        "actual": {
                            **common.get("actual", {}),
                            **(reason.get("actual", {}) if isinstance(reason.get("actual"), dict) else {}),
                        },
                        "error_type": err,
                        "root_cause": str(reason.get("message", "")) or f"structured correctness failure: {err}",
                        "fix_bucket": _fix_bucket_for_failure_type(err),
                    }
                )
            continue

        if numerical_accuracy < 1.0:
            err, root, bucket = _classify_numeric_failure(record, trace)
            rows.append(
                {
                    **common,
                    "error_type": err,
                    "root_cause": root,
                    "fix_bucket": bucket,
                }
            )

        if not output_contract_ok:
            err, root, bucket = _classify_output_failure(record, trace)
            rows.append(
                {
                    **common,
                    "error_type": err,
                    "root_cause": root,
                    "fix_bucket": bucket,
                }
            )

    for row in rows:
        if row["error_type"] not in ALLOWED_ERROR_TYPES:
            original = row["error_type"]
            row["error_type"] = "output_wrong_shape"
            row["root_cause"] = f"error type normalized from unsupported label: {original}"
            row["fix_bucket"] = "classification_normalization"
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "task_type",
        "expected",
        "actual",
        "error_type",
        "root_cause",
        "fix_bucket",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["expected"] = json.dumps(payload.get("expected", {}), ensure_ascii=False)
            payload["actual"] = json.dumps(payload.get("actual", {}), ensure_ascii=False)
            writer.writerow(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze release blockers for numeric/output failures.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/archive/baselines/v1.0.0/report_phase5_formal.json"),
        help="Path to report_phase5_formal.json",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("data/traces"),
        help="Trace directory (contains {trace_id}.json)",
    )
    parser.add_argument(
        "--baseline-jsonl",
        type=Path,
        default=Path("docs/archive/baselines/pre_change_20260415/baseline_questions_25.jsonl"),
        help="Baseline question set with key_numbers",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval/reports/release_blockers/blocker_failure_attribution.json"),
        help="Output json path",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("eval/reports/release_blockers/blocker_failure_attribution.csv"),
        help="Output csv path",
    )
    args = parser.parse_args()

    report = _load_json(args.report)
    baseline_map = _load_jsonl_map(args.baseline_jsonl)
    rows = build_failure_rows(report=report, trace_dir=args.trace_dir, baseline_map=baseline_map)
    error_counts = Counter(str(r.get("error_type", "")) for r in rows)
    root_counts = Counter(str(r.get("root_cause", "")) for r in rows)

    payload = {
        "report_path": str(args.report),
        "trace_dir": str(args.trace_dir),
        "row_count": len(rows),
        "allowed_error_types": sorted(ALLOWED_ERROR_TYPES),
        "counts_by_error_type": dict(sorted(error_counts.items(), key=lambda x: (-x[1], x[0]))),
        "top_root_causes": [
            {"root_cause": cause, "count": count}
            for cause, count in root_counts.most_common(10)
        ],
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(args.out_csv, rows)

    print(f"[ok] rows={len(rows)}")
    print(f"[ok] json={args.out_json}")
    print(f"[ok] csv={args.out_csv}")


if __name__ == "__main__":
    main()
