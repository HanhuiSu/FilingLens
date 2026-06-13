#!/usr/bin/env python3
"""Phase 5 formal acceptance runner (baseline-fixed, API-driven).

This script executes the locked 25-question baseline through /chat + trace files,
computes core + auxiliary metrics, and writes V1 release artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from config import settings
from eval.metrics import (
    answer_accuracy,
    citation_accuracy,
    citation_required_accuracy,
    correctness_summary,
    event_window_accuracy,
    latency_score,
    numeric_fact_accuracy,
    numerical_accuracy,
    period_accuracy,
    retrieval_hit_rate,
    tool_success_rate,
)
from src.agent.nodes import OUTPUT_PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "docs" / "archive" / "baselines" / "pre_change_20260415" / "baseline_questions_25.jsonl"
DEFAULT_BASELINE_REPORT = ROOT / "docs" / "archive" / "baselines" / "pre_change_20260415" / "benchmark_report_current.json"
DEFAULT_OUTPUT_DIR = ROOT / "eval" / "reports" / "phase5_release_eval"

NUMERIC_TASKS = {"fact_qa", "trend_analysis", "company_comparison"}
TIME_KEYWORDS = ("q1", "q2", "q3", "q4", "季度", "财年", "全年", "年度", "最近", "latest", "same_period", "同口径")
RISK_KEYWORDS = ("风险", "risk", "risk factor", "litigation", "regulatory", "监管")
ESTIMATION_TERMS = ("estimated", "estimate", "infer", "推测", "估算", "大概")
MIXED_SECTIONS = {"", "MIXED", "UNKNOWN"}
FORMAL_METRIC_NAMES = (
    "answer_accuracy",
    "numerical_accuracy",
    "numeric_fact_accuracy",
    "period_accuracy",
    "citation_required_accuracy",
    "event_window_accuracy",
    "retrieval_hit_rate",
    "citation_accuracy",
    "tool_success_rate",
    "latency_score",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _safe_git_rev() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "not-a-git-repo"


def _probe_api(base_url: str) -> dict[str, Any]:
    result = {"base_url": base_url, "ok": False, "status_code": None, "error": ""}
    try:
        r = httpx.get(f"{base_url}/health", timeout=10.0)
        result["status_code"] = r.status_code
        result["ok"] = r.status_code == 200 and r.json().get("status") == "ok"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _probe_llm(base_url: str) -> dict[str, Any]:
    result = {"base_url": base_url, "ok": False, "status_code": None, "model_count": 0, "error": ""}
    try:
        r = httpx.get(
            f"{base_url}/models",
            timeout=15.0,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )
        result["status_code"] = r.status_code
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            result["model_count"] = len(models) if isinstance(models, list) else 0
            result["ok"] = result["model_count"] > 0
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _build_context_snapshot(api_base: str, llm_base: str, baseline_file: Path) -> dict[str, Any]:
    return {
        "captured_at_utc": _now_utc(),
        "project_root": str(ROOT),
        "baseline_file": str(baseline_file),
        "baseline_file_exists": baseline_file.exists(),
        "git_commit": _safe_git_rev(),
        "runtime": {
            "python": "3.x",
        },
        "config": {
            "llm_base_url": settings.llm_base_url,
            "llm_reasoning_model": settings.llm_reasoning_model,
            "llm_fast_model": settings.llm_fast_model,
            "embedding_model_name": settings.embedding_model_name,
            "rag_index_version": settings.rag_index_version,
            "rag_collection_v1": settings.rag_collection_v1,
            "rag_collection_v2": settings.rag_collection_v2,
            "rag_mixed_fallback": settings.rag_mixed_fallback,
            "retrieval_top_k": settings.retrieval_top_k,
            "rag_mmr_lambda": settings.rag_mmr_lambda,
            "api_host": settings.api_host,
            "api_port": settings.api_port,
            "output_protocol_version": OUTPUT_PROTOCOL_VERSION,
        },
        "service_probe": {
            "api": _probe_api(api_base),
            "llm": _probe_llm(llm_base),
        },
    }


def _read_trace_file(trace_id: str, timeout_s: float = 5.0) -> dict[str, Any]:
    trace_path = settings.traces_dir / f"{trace_id}.json"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if trace_path.exists():
            return json.loads(trace_path.read_text(encoding="utf-8"))
        time.sleep(0.1)
    raise FileNotFoundError(f"Trace file not found within {timeout_s}s: {trace_path}")


def _call_chat(client: httpx.Client, api_base: str, query: str, timeout_s: float) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    resp = client.post(f"{api_base}/chat", json={"query": query}, timeout=timeout_s)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    return resp.json(), elapsed


def _validate_smoke_contract(chat_body: dict[str, Any], trace_body: dict[str, Any]) -> dict[str, bool]:
    output = chat_body.get("output", {}) if isinstance(chat_body, dict) else {}
    view = output.get("view", {}) if isinstance(output, dict) else {}
    return {
        "chat_has_output": isinstance(output, dict) and bool(output),
        "chat_output_protocol_ok": str(output.get("protocol_version", "")) == OUTPUT_PROTOCOL_VERSION,
        "chat_output_view_kind": bool(str(view.get("kind", "")).strip()),
        "trace_has_numeric_evidence": isinstance(trace_body.get("numeric_evidence", None), list),
        "trace_has_text_evidence": isinstance(trace_body.get("text_evidence", None), list),
        "trace_has_limitations_output": isinstance(trace_body.get("output", {}), dict),
        "trace_has_retrieval_policy": isinstance(trace_body.get("retrieval_policy", None), dict),
        "trace_has_retrieval_debug": isinstance(trace_body.get("retrieval_debug", None), dict),
    }


def _run_smoke(api_base: str, timeout_s: float) -> dict[str, Any]:
    smoke_cases = [
        {"id": "smoke_fact", "query": "Apple 最近一个季度的营收是多少？"},
        {"id": "smoke_comparison", "query": "比较 Apple 和 Microsoft 最近一个季度的营收。"},
        {"id": "smoke_summary", "query": "根据最近的 10-K，总结 Tesla 的主要风险因素。"},
    ]
    out_cases: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_s) as client:
        for case in smoke_cases:
            row: dict[str, Any] = {"id": case["id"], "query": case["query"]}
            try:
                chat_body, elapsed = _call_chat(client, api_base, case["query"], timeout_s=timeout_s)
                trace_id = str(chat_body.get("trace_id", ""))
                trace_http = client.get(f"{api_base}/trace/{trace_id}", timeout=timeout_s)
                trace_http.raise_for_status()
                trace_body = trace_http.json()
                contract = _validate_smoke_contract(chat_body, trace_body)
                row.update(
                    {
                        "ok": True,
                        "elapsed_s": round(elapsed, 2),
                        "trace_id": trace_id,
                        "task_type": chat_body.get("task_type", ""),
                        "checks": contract,
                    }
                )
            except Exception as exc:
                row.update({"ok": False, "error": str(exc)})
            out_cases.append(row)

    return {
        "generated_at_utc": _now_utc(),
        "api_base": api_base,
        "cases": out_cases,
        "all_passed": all(c.get("ok") for c in out_cases),
    }


def _has_estimation_language(answer: str) -> bool:
    text = (answer or "").lower()
    return any(t in text for t in ESTIMATION_TERMS)


def _evidence_separation_ok(trace: dict[str, Any]) -> bool:
    output = trace.get("output", {}) if isinstance(trace, dict) else {}
    numeric_cards = output.get("numeric_evidence", []) if isinstance(output, dict) else []
    text_cards = output.get("text_evidence", []) if isinstance(output, dict) else []
    numeric_citations = trace.get("numeric_citations", []) if isinstance(trace, dict) else []
    text_citations = trace.get("text_citations", []) if isinstance(trace, dict) else []

    for c in numeric_citations:
        if str(c.get("source_kind", "")).lower() != "structured":
            return False
    for c in text_citations:
        if str(c.get("source_kind", "")).lower() != "document":
            return False

    for card in numeric_cards:
        if not all(str(card.get(k, "")).strip() for k in ("evidence_id", "ticker", "metric", "period_end", "source_tool")):
            return False
    for card in text_cards:
        if not all(str(card.get(k, "")).strip() for k in ("evidence_id", "ticker", "form_type", "fiscal_period", "section")):
            return False
        if not str(card.get("text_snippet", "")).strip():
            return False

    numeric_ids = {str(x.get("evidence_id", "")) for x in numeric_cards}
    text_ids = {str(x.get("evidence_id", "")) for x in text_cards}
    if numeric_ids & text_ids:
        return False
    return True


def _comparison_basis_aligned(trace: dict[str, Any], answer: str) -> bool:
    output = trace.get("output", {}) if isinstance(trace, dict) else {}
    if str(trace.get("task_type", "")) != "company_comparison":
        return True
    comparison_basis = str(output.get("comparison_basis", "")).strip()
    view = output.get("view", {}) if isinstance(output, dict) else {}
    basis_line = str(view.get("comparison_basis_line", "")).strip()
    if not comparison_basis or not basis_line:
        return False
    return basis_line in (answer or "")


def _output_contract_ok(trace: dict[str, Any], answer: str) -> bool:
    output = trace.get("output", {})
    if not isinstance(output, dict):
        return False
    if output.get("protocol_version") != OUTPUT_PROTOCOL_VERSION:
        return False
    task_type = str(trace.get("task_type", ""))
    view = output.get("view", {}) if isinstance(output.get("view", {}), dict) else {}
    if str(view.get("kind", "")) != task_type:
        return False

    summary = str(output.get("summary", ""))
    if task_type == "fact_qa":
        if len(summary) > 180:
            return False
        if not isinstance(view.get("headline_metric", {}), dict) or not view.get("headline_metric"):
            return False
    if task_type == "company_comparison":
        table = view.get("comparison_table", {}) if isinstance(view.get("comparison_table", {}), dict) else {}
        if not str(view.get("comparison_basis_line", "")).strip():
            return False
        if not table.get("columns") or not table.get("rows"):
            return False
    if task_type == "report_summary":
        if not str(view.get("risk_and_limits", "")).strip():
            return False
        if not output.get("limitations"):
            return False
    return bool((answer or "").strip())


def _mixed_primary(trace: dict[str, Any]) -> bool:
    text_citations = trace.get("text_citations", []) if isinstance(trace, dict) else []
    if not text_citations:
        return False
    primary = text_citations[:2]
    return any(str(c.get("section", "")).upper().strip() in MIXED_SECTIONS for c in primary)


def _text_citation_duplicate_rate(trace: dict[str, Any]) -> float:
    text_citations = trace.get("text_citations", []) if isinstance(trace, dict) else []
    if not text_citations:
        return 0.0
    seen: set[tuple[str, str, str, str, int]] = set()
    dup = 0
    for c in text_citations:
        key = (
            str(c.get("ticker", "")),
            str(c.get("filing_id", c.get("form_type", ""))),
            str(c.get("fiscal_period", c.get("period", ""))),
            str(c.get("section", "")),
            int(c.get("chunk_order", 0) or 0),
        )
        if key in seen:
            dup += 1
            continue
        seen.add(key)
    return dup / max(len(text_citations), 1)


def _data_reconciliation_from_tool_results(tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    provider_counts: dict[str, int] = {}
    conflict_count = 0
    fallback_yfinance_fact_count = 0
    sec_fact_count = 0
    for tr in tool_results or []:
        data = tr.get("data", {}) if isinstance(tr, dict) else {}
        ctx = data.get("period_context", {}) if isinstance(data, dict) else {}
        summary = ctx.get("source_summary", {}) if isinstance(ctx, dict) else {}
        for provider, count in (summary.get("provider_counts", {}) or {}).items():
            provider_counts[str(provider)] = provider_counts.get(str(provider), 0) + int(count or 0)
        conflict_count += int(summary.get("conflict_count", 0) or 0)
        fallback_yfinance_fact_count += int(summary.get("fallback_yfinance_fact_count", 0) or 0)
        sec_fact_count += int(summary.get("sec_fact_count", 0) or 0)
    total_selected = sec_fact_count + fallback_yfinance_fact_count
    return {
        "provider_counts": provider_counts,
        "conflict_count": conflict_count,
        "conflict_rate": round(conflict_count / max(total_selected, 1), 4),
        "fallback_yfinance_fact_count": fallback_yfinance_fact_count,
        "sec_fact_count": sec_fact_count,
    }


def _risk_item_1a_priority(trace: dict[str, Any], question: str) -> bool:
    text = (question or "").lower()
    if not any(k in text for k in RISK_KEYWORDS):
        return True
    text_citations = trace.get("text_citations", []) if isinstance(trace, dict) else []
    if not text_citations:
        return False
    top = text_citations[0]
    return str(top.get("section", "")).upper().strip() == "ITEM_1A"


def _format_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _trace_contract_ok(trace: dict[str, Any]) -> bool:
    if not isinstance(trace, dict):
        return False
    output = trace.get("output", {})
    if not isinstance(output, dict):
        return False
    required_trace_lists = ("numeric_evidence", "text_evidence")
    if not all(isinstance(trace.get(k), list) for k in required_trace_lists):
        return False
    if not isinstance(trace.get("retrieval_policy"), dict):
        return False
    if not isinstance(trace.get("retrieval_debug"), dict):
        return False
    return str(output.get("protocol_version", "")) == OUTPUT_PROTOCOL_VERSION


def _legacy_expected_text_evidence(item: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for raw in item.get("expected_citations", []) or []:
        text = str(raw or "").strip()
        if not text:
            continue
        tokens = text.replace("/", " ").split()
        ticker = tokens[0].upper() if tokens else ""
        form_type = next((t.upper() for t in tokens if t.upper().replace("-", "") in {"10K", "10Q", "8K"}), "")
        form_type = form_type.replace("10K", "10-K").replace("10Q", "10-Q").replace("8K", "8-K")
        section = next((t.upper() for t in tokens if t.upper().startswith("ITEM_")), "")
        fiscal_period = next((t for t in tokens if re.fullmatch(r"\d{4}", t)), "")
        out.append(
            {
                "source_kind": "document",
                "ticker": ticker,
                "form_type": form_type,
                "section": section,
                "fiscal_period": fiscal_period,
                "must_include_terms": [],
            }
        )
    return out


def _expected_text_evidence(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows = item.get("expected_text_evidence", [])
    if isinstance(rows, list) and rows:
        return [x for x in rows if isinstance(x, dict)]
    return _legacy_expected_text_evidence(item)


def _requires_citation(item: dict[str, Any]) -> bool:
    if "requires_citation" in item:
        return bool(item.get("requires_citation"))
    return bool(item.get("expected_text_evidence") or item.get("expected_citations"))


def _record_error_failure(item: dict[str, Any], message: str) -> dict[str, Any]:
    if item.get("expected_events"):
        return {
            "type": "missing_expected_event",
            "expected": {"expected_events": item.get("expected_events", [])},
            "actual": {},
            "message": message,
        }
    if _requires_citation(item):
        return {
            "type": "missing_citation",
            "expected": {"requires_citation": True},
            "actual": {},
            "message": message,
        }
    return {
        "type": "missing_expected_fact",
        "expected": {"expected_facts": item.get("expected_facts", [])},
        "actual": {},
        "message": message,
    }


def _append_threshold_failure_if_needed(
    metrics: dict[str, float],
    failure_reasons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failure_reasons:
        return failure_reasons
    if float(metrics.get("answer_accuracy", 1.0)) < 0.70:
        return [
            {
                "type": "missing_expected_fact",
                "expected": {"answer_accuracy_min": 0.70},
                "actual": {"answer_accuracy": metrics.get("answer_accuracy")},
                "message": "Answer accuracy is below the correctness gate threshold and no structured evidence match explains the failure.",
            }
        ]
    return failure_reasons


def _dedupe_failure_reasons(failure_reasons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for reason in failure_reasons:
        key = repr(sorted((str(k), repr(v)) for k, v in reason.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(reason)
    return out


def _run_formal_eval(
    baseline_items: list[dict[str, Any]],
    api_base: str,
    timeout_s: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_s) as client:
        for i, item in enumerate(baseline_items, start=1):
            qid = item.get("id", f"q{i:03d}")
            question = item.get("question", "")
            row: dict[str, Any] = {
                "id": qid,
                "question": question,
                "expected_task_type": item.get("expected_task_type", ""),
                "category": item.get("category", ""),
            }
            try:
                chat_body, elapsed = _call_chat(client, api_base, question, timeout_s=timeout_s)
                trace_id = str(chat_body.get("trace_id", ""))
                trace = _read_trace_file(trace_id, timeout_s=8.0)
                output = trace.get("output", chat_body.get("output", {})) if isinstance(trace, dict) else {}
                output = output if isinstance(output, dict) else {}
                answer = str(trace.get("final_answer", chat_body.get("answer", "")))
                citations = trace.get("citations", [])
                retrieved_docs = trace.get("retrieved_docs", [])
                tool_results = trace.get("tool_results", [])
                actual_task_type = str(trace.get("task_type", chat_body.get("task_type", "")))
                actual_tools = list(trace.get("selected_tools", chat_body.get("used_tools", [])))
                expected_facts = item.get("expected_facts", []) if isinstance(item.get("expected_facts", []), list) else []
                expected_text = _expected_text_evidence(item)
                expected_events = item.get("expected_events", []) if isinstance(item.get("expected_events", []), list) else []
                requires_citation = _requires_citation(item)

                m_answer = answer_accuracy(answer, item.get("expected_answer", ""), item.get("key_numbers", []))
                m_numerical = numerical_accuracy(answer, item.get("key_numbers", []))
                m_numeric_fact, numeric_failures = numeric_fact_accuracy(output, trace, expected_facts)
                m_period, period_failures = period_accuracy(output, trace, expected_facts, expected_events)
                m_citation_required, citation_failures = citation_required_accuracy(
                    trace,
                    expected_text,
                    requires_citation=requires_citation,
                )
                m_event_window, event_failures = event_window_accuracy(output, trace, expected_events)
                m_retrieval = retrieval_hit_rate(retrieved_docs, item.get("expected_citations", []))
                m_citation = citation_accuracy(citations, item.get("expected_citations", []))
                m_tool = tool_success_rate(tool_results)
                m_latency = latency_score(elapsed)
                metrics = {
                    "answer_accuracy": round(m_answer, 4),
                    "numerical_accuracy": round(m_numerical, 4),
                    "numeric_fact_accuracy": round(m_numeric_fact, 4),
                    "period_accuracy": round(m_period, 4),
                    "citation_required_accuracy": round(m_citation_required, 4),
                    "event_window_accuracy": round(m_event_window, 4),
                    "retrieval_hit_rate": round(m_retrieval, 4),
                    "citation_accuracy": round(m_citation, 4),
                    "tool_success_rate": round(m_tool, 4),
                    "latency_score": round(m_latency, 4),
                }
                failure_reasons = [
                    *numeric_failures,
                    *period_failures,
                    *citation_failures,
                    *event_failures,
                ]
                failure_reasons = _dedupe_failure_reasons(failure_reasons)
                failure_reasons = _append_threshold_failure_if_needed(metrics, failure_reasons)
                correctness = correctness_summary({**metrics, "failure_reasons": failure_reasons})

                empty_answer = not answer.strip()
                basis_aligned = _comparison_basis_aligned(trace, answer)
                evidence_sep_ok = _evidence_separation_ok(trace)
                output_contract_ok = _output_contract_ok(trace, answer)
                trace_contract_ok = _trace_contract_ok(trace)
                estimation_flag = _has_estimation_language(answer)
                mixed_primary = _mixed_primary(trace)
                text_dup_rate = _text_citation_duplicate_rate(trace)
                risk_item_1a_ok = _risk_item_1a_priority(trace, question)
                data_reconciliation = _data_reconciliation_from_tool_results(tool_results)

                row.update(
                    {
                        "ok": True,
                        "trace_id": trace_id,
                        "elapsed_s": round(elapsed, 2),
                        "actual_task_type": actual_task_type,
                        "task_type_correct": actual_task_type == item.get("expected_task_type", ""),
                        "tools_correct": set(item.get("expected_tools", [])).issubset(set(actual_tools)),
                        "metrics": metrics,
                        "failure_reasons": failure_reasons,
                        "correctness_pass": correctness["correctness_pass"],
                        "primary_failure_type": correctness["primary_failure_type"],
                        "data_reconciliation": data_reconciliation,
                        "requires_citation": requires_citation,
                        "expected_fact_count": len(expected_facts),
                        "expected_text_evidence_count": len(expected_text),
                        "expected_event_count": len(expected_events),
                        "signals": {
                            "empty_answer": empty_answer,
                            "comparison_basis_aligned": basis_aligned,
                            "evidence_separation_correct": evidence_sep_ok,
                            "output_contract_ok": output_contract_ok,
                            "trace_contract_ok": trace_contract_ok,
                            "has_estimation_language": estimation_flag,
                            "mixed_primary": mixed_primary,
                            "text_citation_duplicate_rate": round(text_dup_rate, 4),
                            "risk_item_1a_priority_ok": risk_item_1a_ok,
                        },
                        "answer_preview": answer[:220],
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "ok": False,
                        "error": str(exc),
                        "elapsed_s": 0.0,
                        "metrics": {name: 0.0 for name in FORMAL_METRIC_NAMES},
                        "failure_reasons": [_record_error_failure(item, str(exc))],
                        "correctness_pass": False,
                        "primary_failure_type": _record_error_failure(item, str(exc))["type"],
                        "data_reconciliation": {
                            "provider_counts": {},
                            "conflict_count": 0,
                            "conflict_rate": 0.0,
                            "fallback_yfinance_fact_count": 0,
                            "sec_fact_count": 0,
                        },
                        "requires_citation": _requires_citation(item),
                        "expected_fact_count": len(item.get("expected_facts", []) if isinstance(item.get("expected_facts", []), list) else []),
                        "expected_text_evidence_count": len(_expected_text_evidence(item)),
                        "expected_event_count": len(item.get("expected_events", []) if isinstance(item.get("expected_events", []), list) else []),
                        "signals": {
                            "empty_answer": True,
                            "comparison_basis_aligned": False,
                            "evidence_separation_correct": False,
                            "output_contract_ok": False,
                            "trace_contract_ok": False,
                            "has_estimation_language": False,
                            "mixed_primary": False,
                            "text_citation_duplicate_rate": 0.0,
                            "risk_item_1a_priority_ok": False,
                        },
                    }
                )
            records.append(row)

    def _avg(metric: str) -> float:
        vals = [r["metrics"].get(metric, 0.0) for r in records if r.get("ok")]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    total = len(records)
    ok_count = sum(1 for r in records if r.get("ok"))
    errors = total - ok_count
    empty_rate = sum(1 for r in records if r["signals"]["empty_answer"]) / max(total, 1)
    cmp_rows = [r for r in records if r.get("expected_task_type") == "company_comparison"]
    cmp_align_rate = (
        sum(1 for r in cmp_rows if r["signals"]["comparison_basis_aligned"]) / len(cmp_rows)
        if cmp_rows else 1.0
    )
    sep_rate = sum(1 for r in records if r["signals"]["evidence_separation_correct"]) / max(total, 1)
    output_ok_rate = sum(1 for r in records if r["signals"]["output_contract_ok"]) / max(total, 1)
    trace_ok_rate = sum(1 for r in records if r["signals"].get("trace_contract_ok")) / max(total, 1)
    mixed_primary_rate = sum(1 for r in records if r["signals"]["mixed_primary"]) / max(total, 1)
    estimation_rate = sum(1 for r in records if r["signals"]["has_estimation_language"]) / max(total, 1)
    avg_text_dup = statistics.fmean(r["signals"]["text_citation_duplicate_rate"] for r in records) if records else 0.0
    risk_rows = [r for r in records if any(k in r.get("question", "").lower() for k in RISK_KEYWORDS)]
    risk_item_1a_rate = (
        sum(1 for r in risk_rows if r["signals"]["risk_item_1a_priority_ok"]) / len(risk_rows)
        if risk_rows else 1.0
    )
    data_reconciliation = _aggregate_data_reconciliation(records)

    by_task: dict[str, dict[str, Any]] = {}
    for task in sorted({str(r.get("expected_task_type", "")) for r in records}):
        task_rows = [r for r in records if str(r.get("expected_task_type", "")) == task]
        if not task_rows:
            continue
        failed_ids_by_reason: dict[str, list[str]] = defaultdict(list)
        for r in task_rows:
            for reason in r.get("failure_reasons", []) or []:
                if isinstance(reason, dict) and reason.get("type"):
                    failed_ids_by_reason[str(reason["type"])].append(str(r.get("id", "")))
        by_task[task] = {
            "count": len(task_rows),
            "errors": sum(1 for r in task_rows if not r.get("ok")),
            "metrics": {
                metric: round(sum(r["metrics"].get(metric, 0.0) for r in task_rows) / len(task_rows), 4)
                for metric in FORMAL_METRIC_NAMES
            },
            "failed_ids_by_reason": dict(sorted(failed_ids_by_reason.items())),
            "signals": {
                "empty_answer_rate": round(sum(1 for r in task_rows if r["signals"]["empty_answer"]) / len(task_rows), 4),
                "evidence_separation_correctness": round(sum(1 for r in task_rows if r["signals"]["evidence_separation_correct"]) / len(task_rows), 4),
                "output_contract_ok_rate": round(sum(1 for r in task_rows if r["signals"]["output_contract_ok"]) / len(task_rows), 4),
                "trace_contract_ok_rate": round(sum(1 for r in task_rows if r["signals"].get("trace_contract_ok")) / len(task_rows), 4),
            },
        }

    summary = {
        "total": total,
        "ok": ok_count,
        "errors": errors,
        "average_latency_s": round(sum(r.get("elapsed_s", 0.0) for r in records) / max(total, 1), 2),
        "task_type_accuracy": round(sum(1 for r in records if r.get("task_type_correct")) / max(total, 1), 4),
        "tool_selection_accuracy": round(sum(1 for r in records if r.get("tools_correct")) / max(total, 1), 4),
        "average_metrics": {
            metric: _avg(metric)
            for metric in FORMAL_METRIC_NAMES
        },
        "auxiliary_metrics": {
            "empty_answer_rate": round(empty_rate, 4),
            "comparison_basis_alignment_rate": round(cmp_align_rate, 4),
            "evidence_separation_correctness": round(sep_rate, 4),
            "output_protocol_stability_rate": round(output_ok_rate, 4),
            "trace_contract_ok_rate": round(trace_ok_rate, 4),
            "mixed_primary_rate": round(mixed_primary_rate, 4),
            "estimation_language_rate": round(estimation_rate, 4),
            "avg_text_citation_duplicate_rate": round(avg_text_dup, 4),
            "risk_item_1a_priority_rate": round(risk_item_1a_rate, 4),
        },
        "data_reconciliation": data_reconciliation,
        "by_expected_task_type": by_task,
    }
    return {"generated_at_utc": _now_utc(), "summary": summary, "records": records}


def _aggregate_data_reconciliation(records: list[dict[str, Any]]) -> dict[str, Any]:
    provider_counts: dict[str, int] = {}
    conflict_count = 0
    fallback_yfinance_fact_count = 0
    sec_fact_count = 0
    for record in records:
        summary = record.get("data_reconciliation", {}) or {}
        for provider, count in (summary.get("provider_counts", {}) or {}).items():
            provider_counts[str(provider)] = provider_counts.get(str(provider), 0) + int(count or 0)
        conflict_count += int(summary.get("conflict_count", 0) or 0)
        fallback_yfinance_fact_count += int(summary.get("fallback_yfinance_fact_count", 0) or 0)
        sec_fact_count += int(summary.get("sec_fact_count", 0) or 0)
    total_selected = sec_fact_count + fallback_yfinance_fact_count
    return {
        "provider_counts": provider_counts,
        "conflict_count": conflict_count,
        "conflict_rate": round(conflict_count / max(total_selected, 1), 4),
        "fallback_yfinance_fact_count": fallback_yfinance_fact_count,
        "sec_fact_count": sec_fact_count,
    }


def _subset_stats(records: list[dict[str, Any]], ids: set[str], pass_fn) -> dict[str, Any]:
    rows = [r for r in records if r.get("id") in ids]
    if not rows:
        return {"count": 0, "passed": 0, "failed_ids": [], "pass_rate": 0.0}
    passed_rows = [r for r in rows if pass_fn(r)]
    return {
        "count": len(rows),
        "passed": len(passed_rows),
        "failed_ids": [r.get("id") for r in rows if not pass_fn(r)],
        "pass_rate": round(len(passed_rows) / len(rows), 4),
    }


def _run_specialized_regression(items: list[dict[str, Any]], formal: dict[str, Any]) -> dict[str, Any]:
    records = formal.get("records", [])
    item_map = {str(x.get("id", "")): x for x in items}

    time_ids = {
        qid for qid, item in item_map.items()
        if any(k in str(item.get("question", "")).lower() for k in TIME_KEYWORDS)
    }
    evidence_ids = {qid for qid, item in item_map.items() if str(item.get("expected_task_type", "")) in NUMERIC_TASKS or str(item.get("expected_task_type", "")) == "report_summary"}
    retrieval_ids = {
        qid for qid, item in item_map.items()
        if str(item.get("category", "")) in {"summary", "comparison", "hybrid"}
    }
    output_ids = set(item_map.keys())

    time_stats = _subset_stats(
        records,
        time_ids,
        lambda r: r.get("ok") and not r["signals"]["empty_answer"] and not any(
            x in str(r.get("answer_preview", "")).lower() for x in ("无法严格同口径比较", "time period is ambiguous")
        ),
    )
    evidence_stats = _subset_stats(
        records,
        evidence_ids,
        lambda r: r.get("ok") and r["signals"]["evidence_separation_correct"] and not r["signals"]["has_estimation_language"],
    )
    retrieval_stats = _subset_stats(
        records,
        retrieval_ids,
        lambda r: r.get("ok") and not r["signals"]["mixed_primary"] and r["signals"]["risk_item_1a_priority_ok"] and r["signals"]["text_citation_duplicate_rate"] == 0.0,
    )
    output_stats = _subset_stats(
        records,
        output_ids,
        lambda r: r.get("ok") and r["signals"]["output_contract_ok"],
    )

    return {
        "generated_at_utc": _now_utc(),
        "time_semantics": time_stats,
        "evidence_binding": evidence_stats,
        "retrieval_quality": retrieval_stats,
        "output_protocol": output_stats,
    }


def _run_spot_check(formal: dict[str, Any]) -> dict[str, Any]:
    records = {str(r.get("id", "")): r for r in formal.get("records", [])}
    spot_ids = ["fq01", "fq14", "tr01", "tr04", "cp01", "cp02", "sm01", "sm06", "hy01", "hy02"]
    rows: list[dict[str, Any]] = []
    for qid in spot_ids:
        r = records.get(qid, {})
        task = str(r.get("expected_task_type", ""))
        credible = bool(r.get("ok")) and not r.get("signals", {}).get("empty_answer", True)
        numeric_threshold = 0.95 if task == "fact_qa" else 0.85
        numeric_ok = True if task not in NUMERIC_TASKS else float(r.get("metrics", {}).get("numeric_fact_accuracy", 0.0)) >= numeric_threshold
        period_ok = float(r.get("metrics", {}).get("period_accuracy", 0.0)) >= 0.9
        if r.get("requires_citation"):
            citation_ok = float(r.get("metrics", {}).get("citation_required_accuracy", 0.0)) >= 0.85
        else:
            citation_ok = bool(r.get("signals", {}).get("evidence_separation_correct"))
        comp_ok = True if task != "company_comparison" else bool(r.get("signals", {}).get("comparison_basis_aligned"))
        summary_ok = True if task != "report_summary" else bool(r.get("signals", {}).get("output_contract_ok"))
        limits_ok = bool(r.get("signals", {}).get("output_contract_ok"))
        demo_ready = all([credible, numeric_ok, period_ok, citation_ok, comp_ok, summary_ok, limits_ok])
        rows.append(
            {
                "id": qid,
                "task_type": task,
                "credible_conclusion": credible,
                "numeric_correct": numeric_ok,
                "period_correct": period_ok,
                "citations_credible": citation_ok,
                "comparison_basis_ok": comp_ok,
                "summary_style_ok": summary_ok,
                "limitations_ok": limits_ok,
                "primary_failure_type": r.get("primary_failure_type", ""),
                "demo_ready": demo_ready,
            }
        )

    pass_rate = sum(1 for r in rows if r["demo_ready"]) / max(len(rows), 1)
    demo_candidates = [r["id"] for r in rows if r["demo_ready"]][:5]
    return {
        "generated_at_utc": _now_utc(),
        "checklist": rows,
        "pass_rate": round(pass_rate, 4),
        "demo_candidates": demo_candidates,
    }


def _before_after_summary(formal: dict[str, Any], baseline_report_path: Path) -> dict[str, Any]:
    baseline = json.loads(baseline_report_path.read_text(encoding="utf-8")) if baseline_report_path.exists() else {}
    b_summary = baseline.get("summary", {})
    b_avg = b_summary.get("average_metrics", {})
    c_summary = formal.get("summary", {})
    c_avg = c_summary.get("average_metrics", {})
    c_aux = c_summary.get("auxiliary_metrics", {})

    deltas = {}
    for key in FORMAL_METRIC_NAMES:
        if key in b_avg and key in c_avg:
            deltas[key] = round(float(c_avg[key]) - float(b_avg[key]), 4)

    return {
        "generated_at_utc": _now_utc(),
        "baseline_reference": str(baseline_report_path),
        "baseline_sample_size": b_summary.get("total", 0),
        "current_sample_size": c_summary.get("total", 0),
        "baseline_average_metrics": b_avg,
        "current_average_metrics": c_avg,
        "metric_deltas": deltas,
        "current_auxiliary_metrics": c_aux,
        "notes": [
            "Baseline frozen snapshot has only 3 samples; interpret deltas with caution.",
            "Current formal report is evaluated on the fixed 25-question baseline set.",
        ],
    }


def _evaluate_gate(
    formal: dict[str, Any],
    specialized: dict[str, Any],
    spot_check: dict[str, Any],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    summary = formal.get("summary", {})
    avg = summary.get("average_metrics", {})
    aux = summary.get("auxiliary_metrics", {})
    by_task = summary.get("by_expected_task_type", {})
    records = formal.get("records", [])

    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _avg_metric(name: str, default: float = 0.0) -> float:
        return _float(avg.get(name), default)

    def _task_metric(task: str, metric: str, default: float = 1.0) -> float:
        if task not in by_task:
            return default
        return _float(by_task.get(task, {}).get("metrics", {}).get(metric), default)

    def _subset_average_metric(rows: list[dict[str, Any]], metric: str, default: float = 1.0) -> float:
        if not rows:
            return default
        return sum(_float(r.get("metrics", {}).get(metric), 0.0) for r in rows) / len(rows)

    required_citation_rows = [r for r in records if r.get("requires_citation")]
    event_rows = [r for r in records if int(r.get("expected_event_count", 0) or 0) > 0]
    required_citation_accuracy = _subset_average_metric(required_citation_rows, "citation_required_accuracy", default=1.0)
    required_retrieval_hit_rate = _subset_average_metric(required_citation_rows, "retrieval_hit_rate", default=_avg_metric("retrieval_hit_rate", 1.0))
    required_event_window_accuracy = _subset_average_metric(event_rows, "event_window_accuracy", default=1.0)
    event_window_all_pass = all(_float(r.get("metrics", {}).get("event_window_accuracy"), 0.0) >= 0.90 for r in event_rows)
    spot_rows = spot_check.get("checklist", []) if isinstance(spot_check.get("checklist", []), list) else []
    demo_cases_all_ready = bool(spot_rows) and all(bool(r.get("demo_ready")) for r in spot_rows)

    def _check(gate_name: str, name: str, ok: bool, actual: Any = None, threshold: str = "") -> dict[str, Any]:
        row = {"gate": gate_name, "name": name, "pass": bool(ok)}
        if actual is not None:
            row["actual"] = actual
        if threshold:
            row["threshold"] = threshold
        return row

    contract_checks = [
        _check("contract_gate", "baseline_25_replayed", summary.get("total", 0) == 25 and summary.get("errors", 0) == 0, summary.get("total"), "25 cases, 0 errors"),
        _check("contract_gate", "smoke_health_and_contract", bool(smoke.get("all_passed")), smoke.get("all_passed"), "true"),
        _check("contract_gate", "output_protocol_stable", _float(aux.get("output_protocol_stability_rate")) >= 0.98, aux.get("output_protocol_stability_rate"), ">=0.98"),
        _check("contract_gate", "empty_answer_controlled", _float(aux.get("empty_answer_rate"), 1.0) <= 0.05, aux.get("empty_answer_rate"), "<=0.05"),
        _check("contract_gate", "trace_contract_ok", _float(aux.get("trace_contract_ok_rate")) >= 0.98, aux.get("trace_contract_ok_rate"), ">=0.98"),
    ]
    correctness_checks = [
        _check("correctness_gate", "overall_numerical_accuracy", _avg_metric("numerical_accuracy") >= 0.90, avg.get("numerical_accuracy"), ">=0.90"),
        _check("correctness_gate", "fact_qa_numeric_fact_accuracy", _task_metric("fact_qa", "numeric_fact_accuracy") >= 0.95, _task_metric("fact_qa", "numeric_fact_accuracy"), ">=0.95"),
        _check("correctness_gate", "trend_analysis_numeric_fact_accuracy", _task_metric("trend_analysis", "numeric_fact_accuracy") >= 0.85, _task_metric("trend_analysis", "numeric_fact_accuracy"), ">=0.85"),
        _check("correctness_gate", "company_comparison_numeric_fact_accuracy", _task_metric("company_comparison", "numeric_fact_accuracy") >= 0.85, _task_metric("company_comparison", "numeric_fact_accuracy"), ">=0.85"),
        _check("correctness_gate", "period_accuracy", _avg_metric("period_accuracy") >= 0.90, avg.get("period_accuracy"), ">=0.90"),
        _check("correctness_gate", "answer_accuracy", _avg_metric("answer_accuracy") >= 0.70, avg.get("answer_accuracy"), ">=0.70"),
        _check("correctness_gate", "required_event_window_accuracy", required_event_window_accuracy >= 0.90 and event_window_all_pass, round(required_event_window_accuracy, 4), "all required event cases >=0.90"),
    ]
    retrieval_checks = [
        _check("retrieval_gate", "citation_required_accuracy", required_citation_accuracy >= 0.85, round(required_citation_accuracy, 4), ">=0.85 over required citation cases"),
        _check("retrieval_gate", "retrieval_hit_rate", required_retrieval_hit_rate >= 0.85, round(required_retrieval_hit_rate, 4), ">=0.85 over required citation cases"),
        _check("retrieval_gate", "retrieval_not_mixed_primary", _float(aux.get("mixed_primary_rate"), 1.0) <= 0.10, aux.get("mixed_primary_rate"), "<=0.10"),
    ]
    demo_checks = [
        _check("demo_gate", "fixed_demo_cases_all_ready", demo_cases_all_ready, spot_check.get("pass_rate"), "10/10 demo_ready"),
        _check(
            "demo_gate",
            "specialized_regression_smoke",
            _float(specialized.get("output_protocol", {}).get("pass_rate")) >= 0.95,
            specialized.get("output_protocol", {}).get("pass_rate"),
            ">=0.95 output protocol specialized pass rate",
        ),
    ]

    all_checks = [*contract_checks, *correctness_checks, *retrieval_checks, *demo_checks]
    contract_pass = all(c["pass"] for c in contract_checks)
    correctness_pass = all(c["pass"] for c in correctness_checks)
    retrieval_pass = all(c["pass"] for c in retrieval_checks)
    demo_pass = all(c["pass"] for c in demo_checks)
    overall_pass = contract_pass and correctness_pass and retrieval_pass and demo_pass
    failed = [c["name"] for c in all_checks if not c["pass"]]
    release_status = (
        "V1 reliability-ready"
        if overall_pass
        else "V1 demo-ready, not reliability-ready"
        if contract_pass and retrieval_pass and demo_pass and not correctness_pass
        else "V1 blocked before demo release"
    )

    return {
        "generated_at_utc": _now_utc(),
        "contract_pass": contract_pass,
        "correctness_pass": correctness_pass,
        "retrieval_pass": retrieval_pass,
        "demo_pass": demo_pass,
        "overall_pass": overall_pass,
        "release_status": release_status,
        "gate_summary": {
            "contract_pass": contract_pass,
            "correctness_pass": correctness_pass,
            "retrieval_pass": retrieval_pass,
            "demo_pass": demo_pass,
            "overall_pass": overall_pass,
            "release_status": release_status,
        },
        "gate_checks": all_checks,
        "failed_checks": failed,
        "known_limitations": [
            "answer_accuracy remains a weak keyword-style secondary signal; numeric_fact_accuracy, period_accuracy, citation_required_accuracy, and event_window_accuracy are the release blockers.",
            "Baseline before/after metric deltas use a 3-sample frozen baseline snapshot.",
            "Legacy cases without v2 expected fields are neutral for the corresponding structured metric until they are migrated.",
        ],
        "post_v1_backlog": [
            "Expand formal baseline to 50+ samples with balanced categories.",
            "Freeze human-reviewed v2 labels for every benchmark case, including source sections and event returns.",
        ],
    }


def _render_formal_summary_md(formal: dict[str, Any]) -> str:
    s = formal["summary"]
    avg = s["average_metrics"]
    aux = s["auxiliary_metrics"]
    gate_summary = s.get("gate_summary", {})
    reason_counts = Counter(
        str(reason.get("type", "unknown"))
        for record in formal.get("records", [])
        for reason in (record.get("failure_reasons", []) or [])
        if isinstance(reason, dict)
    )
    lines = [
        "# Phase 5 Formal Regression Summary",
        "",
        f"- Generated (UTC): {formal['generated_at_utc']}",
        f"- Total: {s['total']} | OK: {s['ok']} | Errors: {s['errors']}",
        f"- Average latency: {s['average_latency_s']}s",
        "",
    ]
    if gate_summary:
        lines.extend(
            [
                "## Gate Summary",
                "",
                f"- overall_pass: {'PASS' if gate_summary.get('overall_pass') else 'FAIL'}",
                f"- contract_pass: {'PASS' if gate_summary.get('contract_pass') else 'FAIL'}",
                f"- correctness_pass: {'PASS' if gate_summary.get('correctness_pass') else 'FAIL'}",
                f"- retrieval_pass: {'PASS' if gate_summary.get('retrieval_pass') else 'FAIL'}",
                f"- demo_pass: {'PASS' if gate_summary.get('demo_pass') else 'FAIL'}",
                f"- release_status: {gate_summary.get('release_status', '(unknown)')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Core Metrics",
            "",
            f"- answer_accuracy: {_format_pct(avg.get('answer_accuracy', 0.0))}",
            f"- numerical_accuracy: {_format_pct(avg.get('numerical_accuracy', 0.0))}",
            f"- retrieval_hit_rate: {_format_pct(avg.get('retrieval_hit_rate', 0.0))}",
            f"- citation_accuracy: {_format_pct(avg.get('citation_accuracy', 0.0))}",
            f"- tool_success_rate: {_format_pct(avg.get('tool_success_rate', 0.0))}",
            f"- latency_score: {_format_pct(avg.get('latency_score', 0.0))}",
            "",
            "## Correctness Metrics",
            "",
            f"- numeric_fact_accuracy: {_format_pct(avg.get('numeric_fact_accuracy', 0.0))}",
            f"- period_accuracy: {_format_pct(avg.get('period_accuracy', 0.0))}",
            f"- citation_required_accuracy: {_format_pct(avg.get('citation_required_accuracy', 0.0))}",
            f"- event_window_accuracy: {_format_pct(avg.get('event_window_accuracy', 0.0))}",
            "",
            "## Failed Cases By Reason",
            "",
        ]
    )
    lines.extend(
        [f"- {reason}: {count}" for reason, count in reason_counts.most_common()]
        if reason_counts
        else ["- (none)"]
    )
    lines.extend(
        [
            "",
            "## Data Reconciliation",
            "",
            f"- sec_fact_count: {s.get('data_reconciliation', {}).get('sec_fact_count', 0)}",
            f"- fallback_yfinance_fact_count: {s.get('data_reconciliation', {}).get('fallback_yfinance_fact_count', 0)}",
            f"- conflict_count: {s.get('data_reconciliation', {}).get('conflict_count', 0)}",
            f"- conflict_rate: {_format_pct(float(s.get('data_reconciliation', {}).get('conflict_rate', 0.0)))}",
            "",
            "## Auxiliary Metrics",
            "",
            f"- empty_answer_rate: {_format_pct(aux.get('empty_answer_rate', 0.0))}",
            f"- comparison_basis_alignment_rate: {_format_pct(aux.get('comparison_basis_alignment_rate', 0.0))}",
            f"- evidence_separation_correctness: {_format_pct(aux.get('evidence_separation_correctness', 0.0))}",
            f"- output_protocol_stability_rate: {_format_pct(aux.get('output_protocol_stability_rate', 0.0))}",
            f"- trace_contract_ok_rate: {_format_pct(aux.get('trace_contract_ok_rate', 0.0))}",
            f"- mixed_primary_rate: {_format_pct(aux.get('mixed_primary_rate', 0.0))}",
            f"- estimation_language_rate: {_format_pct(aux.get('estimation_language_rate', 0.0))}",
            f"- avg_text_citation_duplicate_rate: {_format_pct(aux.get('avg_text_citation_duplicate_rate', 0.0))}",
            "",
            "## By Task Type",
            "",
        ]
    )
    for task, row in s["by_expected_task_type"].items():
        lines.append(f"### {task}")
        lines.append(f"- count: {row['count']} | errors: {row['errors']}")
        lines.append(f"- numerical_accuracy: {_format_pct(row['metrics'].get('numerical_accuracy', 0.0))}")
        lines.append(f"- numeric_fact_accuracy: {_format_pct(row['metrics'].get('numeric_fact_accuracy', 0.0))}")
        lines.append(f"- period_accuracy: {_format_pct(row['metrics'].get('period_accuracy', 0.0))}")
        lines.append(f"- citation_required_accuracy: {_format_pct(row['metrics'].get('citation_required_accuracy', 0.0))}")
        lines.append(f"- event_window_accuracy: {_format_pct(row['metrics'].get('event_window_accuracy', 0.0))}")
        lines.append(f"- empty_answer_rate: {_format_pct(row['signals']['empty_answer_rate'])}")
        lines.append(f"- output_contract_ok_rate: {_format_pct(row['signals']['output_contract_ok_rate'])}")
        failed_ids = row.get("failed_ids_by_reason", {})
        if failed_ids:
            lines.append(f"- failed_ids_by_reason: {json.dumps(failed_ids, ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_specialized_md(specialized: dict[str, Any]) -> str:
    lines = ["# Phase 5 Specialized Regression", ""]
    for key in ("time_semantics", "evidence_binding", "retrieval_quality", "output_protocol"):
        s = specialized.get(key, {})
        lines.append(f"## {key}")
        lines.append(f"- count: {s.get('count', 0)}")
        lines.append(f"- passed: {s.get('passed', 0)}")
        lines.append(f"- pass_rate: {_format_pct(float(s.get('pass_rate', 0.0)))}")
        lines.append(f"- failed_ids: {', '.join(s.get('failed_ids', [])) if s.get('failed_ids') else '(none)'}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_spot_md(spot: dict[str, Any]) -> str:
    rows = spot.get("checklist", [])
    lines = [
        "# Phase 5 Spot-Check",
        "",
        f"- pass_rate: {_format_pct(float(spot.get('pass_rate', 0.0)))}",
        f"- demo_candidates: {', '.join(spot.get('demo_candidates', [])) if spot.get('demo_candidates') else '(none)'}",
        "",
        "| id | task_type | credible | numeric | period | citation | basis | summary | limits | failure | demo_ready |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['task_type']} | {'Y' if r['credible_conclusion'] else 'N'} | "
            f"{'Y' if r['numeric_correct'] else 'N'} | {'Y' if r['period_correct'] else 'N'} | "
            f"{'Y' if r['citations_credible'] else 'N'} | "
            f"{'Y' if r['comparison_basis_ok'] else 'N'} | {'Y' if r['summary_style_ok'] else 'N'} | "
            f"{'Y' if r['limitations_ok'] else 'N'} | {r.get('primary_failure_type') or ''} | "
            f"{'Y' if r['demo_ready'] else 'N'} |"
        )
    return "\n".join(lines).strip() + "\n"


def _render_before_after_md(before_after: dict[str, Any]) -> str:
    lines = [
        "# Phase 5 Before / After Summary",
        "",
        f"- baseline_reference: `{before_after['baseline_reference']}`",
        f"- baseline_sample_size: {before_after['baseline_sample_size']}",
        f"- current_sample_size: {before_after['current_sample_size']}",
        "",
        "## Key Metric Deltas (Current - Baseline)",
        "",
    ]
    deltas = before_after.get("metric_deltas", {})
    if not deltas:
        lines.append("- (no comparable metrics found)")
    else:
        for k, v in deltas.items():
            sign = "+" if v >= 0 else ""
            lines.append(f"- {k}: {sign}{v:.4f}")
    lines.extend(
        [
            "",
            "## Current Auxiliary Signals",
            "",
        ]
    )
    for k, v in before_after.get("current_auxiliary_metrics", {}).items():
        lines.append(f"- {k}: {_format_pct(float(v)) if isinstance(v, (float, int)) else v}")
    lines.extend(["", "## Notes", ""])
    for n in before_after.get("notes", []):
        lines.append(f"- {n}")
    return "\n".join(lines).strip() + "\n"


def _render_gate_md(gate: dict[str, Any]) -> str:
    lines = [
        "# Phase 5 V1 Release Gate",
        "",
        f"- overall_pass: {'PASS' if gate.get('overall_pass') else 'FAIL'}",
        f"- contract_pass: {'PASS' if gate.get('contract_pass') else 'FAIL'}",
        f"- correctness_pass: {'PASS' if gate.get('correctness_pass') else 'FAIL'}",
        f"- retrieval_pass: {'PASS' if gate.get('retrieval_pass') else 'FAIL'}",
        f"- demo_pass: {'PASS' if gate.get('demo_pass') else 'FAIL'}",
        f"- release_status: {gate.get('release_status', '(unknown)')}",
        "",
        "## Gate Checks",
        "",
    ]
    for c in gate.get("gate_checks", []):
        details = []
        if "actual" in c:
            details.append(f"actual={c['actual']}")
        if c.get("threshold"):
            details.append(f"threshold={c['threshold']}")
        suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {c.get('gate', 'gate')}.{c['name']}: {'PASS' if c['pass'] else 'FAIL'}{suffix}")
    lines.extend(["", "## Failed Checks", ""])
    failed = gate.get("failed_checks", [])
    lines.append(f"- {', '.join(failed) if failed else '(none)'}")
    lines.extend(["", "## Known Limitations", ""])
    for x in gate.get("known_limitations", []):
        lines.append(f"- {x}")
    lines.extend(["", "## Post-V1 Backlog", ""])
    for x in gate.get("post_v1_backlog", []):
        lines.append(f"- {x}")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 formal release acceptance")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--baseline-report", default=str(DEFAULT_BASELINE_REPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--api-base", default="http://127.0.0.1:8080")
    parser.add_argument("--llm-base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    baseline_report_path = Path(args.baseline_report)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = _build_context_snapshot(args.api_base, args.llm_base, baseline_path)
    _write_json(out_dir / "evaluation_context_snapshot.json", context)

    smoke = _run_smoke(args.api_base, timeout_s=args.timeout)
    _write_json(out_dir / "smoke_results.json", smoke)

    items = _load_jsonl(baseline_path)
    formal = _run_formal_eval(items, api_base=args.api_base, timeout_s=args.timeout)

    specialized = _run_specialized_regression(items, formal)
    _write_json(out_dir / "specialized_regression.json", specialized)
    _write_md(out_dir / "specialized_regression.md", _render_specialized_md(specialized))

    spot = _run_spot_check(formal)
    _write_json(out_dir / "spot_check.json", spot)
    _write_md(out_dir / "spot_check.md", _render_spot_md(spot))

    before_after = _before_after_summary(formal, baseline_report_path=baseline_report_path)
    _write_json(out_dir / "before_after_summary.json", before_after)
    _write_md(out_dir / "before_after_summary.md", _render_before_after_md(before_after))

    gate = _evaluate_gate(formal, specialized, spot, smoke)
    formal.setdefault("summary", {})["gate_summary"] = gate.get("gate_summary", {})
    _write_json(out_dir / "report_phase5_formal.json", formal)
    _write_md(out_dir / "report_phase5_summary.md", _render_formal_summary_md(formal))
    _write_json(out_dir / "v1_release_gate.json", gate)
    _write_md(out_dir / "v1_release_gate.md", _render_gate_md(gate))

    print(json.dumps(
        {
            "output_dir": str(out_dir),
            "smoke_all_passed": smoke.get("all_passed"),
            "formal_total": formal.get("summary", {}).get("total"),
            "formal_errors": formal.get("summary", {}).get("errors"),
            "contract_pass": gate.get("contract_pass"),
            "correctness_pass": gate.get("correctness_pass"),
            "retrieval_pass": gate.get("retrieval_pass"),
            "demo_pass": gate.get("demo_pass"),
            "gate_pass": gate.get("overall_pass"),
            "release_status": gate.get("release_status"),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
