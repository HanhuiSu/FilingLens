#!/usr/bin/env python3
# ruff: noqa: E402
"""Eval runner — execute benchmark, compute metrics, output report.

Usage:
  PYTHONPATH=/home/hui/agent python eval/run_eval.py [OPTIONS]

Options:
  --benchmark PATH    Path to benchmark JSONL (default: eval/benchmark.jsonl)
  --output PATH       Path to save JSON report (default: eval/report.json)
  --limit N           Only run first N questions (for quick smoke tests)
  --category CAT      Only run questions in this category
  --ids ID1,ID2,...   Only run specific question IDs (comma-separated)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import (
    answer_accuracy,
    citation_accuracy,
    event_alignment_accuracy,
    event_numeric_accuracy,
    event_tool_success_rate,
    latency_score,
    market_reaction_block_presence_rate,
    numerical_accuracy,
    retrieval_hit_rate,
    tool_success_rate,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_benchmark(path: Path, limit: int | None = None,
                   category: str | None = None,
                   ids: set[str] | None = None) -> list[dict]:
    items: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if category and item.get("category") != category:
                continue
            if ids and item.get("id") not in ids:
                continue
            items.append(item)
            if limit and len(items) >= limit:
                break
    return items


def run_single(agent, item: dict) -> dict[str, Any]:
    """Run the agent on one benchmark item and compute all metrics."""
    question = item["question"]
    t0 = time.time()
    try:
        result = agent.invoke({"user_query": question})
        elapsed = time.time() - t0
        error = None
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("Agent failed on %s: %s", item.get("id"), exc)
        result = {}
        error = str(exc)

    answer = result.get("final_answer", "")
    tool_results = result.get("tool_results", [])
    retrieved_docs = result.get("retrieved_docs", [])
    citations = result.get("citations", [])
    actual_task_type = result.get("task_type", "")
    actual_tools = result.get("selected_tools", [])
    output = result.get("output", {}) if isinstance(result.get("output", {}), dict) else {}
    event_results = result.get("event_results", [])
    data_reconciliation = _data_reconciliation_from_tool_results(tool_results)

    expected_answer = item.get("expected_answer", "")
    key_numbers = item.get("key_numbers", [])
    expected_citations = item.get("expected_citations", [])
    expected_task_type = item.get("expected_task_type", "")
    expected_tools = item.get("expected_tools", [])
    expected_event_type = item.get("expected_event_type")
    expected_latest_n = item.get("expected_latest_n")
    expects_market_reaction = bool(item.get("expects_market_reaction")) or item.get("category") == "event_reaction"

    # Compute metrics
    m_answer = answer_accuracy(answer, expected_answer, key_numbers) if not error else 0.0
    m_numerical = numerical_accuracy(answer, key_numbers) if not error else 0.0
    m_retrieval = retrieval_hit_rate(retrieved_docs, expected_citations) if not error else 0.0
    m_citation = citation_accuracy(citations, expected_citations) if not error else 0.0
    m_tool = tool_success_rate(tool_results) if not error else 0.0
    m_latency = latency_score(elapsed)
    m_event_alignment = (
        event_alignment_accuracy(
            event_results=event_results,
            expected_event_type=expected_event_type,
            expected_latest_n=expected_latest_n,
            required=expects_market_reaction,
        )
        if not error
        else 0.0
    )
    m_event_numeric = (
        event_numeric_accuracy(
            output=output,
            key_numbers=key_numbers,
            required=expects_market_reaction,
        )
        if not error
        else 0.0
    )
    m_event_tool = event_tool_success_rate(tool_results, required=expects_market_reaction) if not error else 0.0
    m_market_block = market_reaction_block_presence_rate(output, required=expects_market_reaction) if not error else 0.0

    task_type_correct = actual_task_type == expected_task_type if expected_task_type else True
    tools_correct = set(expected_tools).issubset(set(actual_tools)) if expected_tools else True

    return {
        "id": item.get("id", "?"),
        "category": item.get("category", ""),
        "question": question,
        "answer_preview": answer[:200] if answer else "(error)",
        "error": error,
        "elapsed_s": round(elapsed, 1),
        "task_type_actual": actual_task_type,
        "task_type_correct": task_type_correct,
        "tools_correct": tools_correct,
        "metrics": {
            "answer_accuracy": round(m_answer, 4),
            "numerical_accuracy": round(m_numerical, 4),
            "retrieval_hit_rate": round(m_retrieval, 4),
            "citation_accuracy": round(m_citation, 4),
            "tool_success_rate": round(m_tool, 4),
            "latency_score": round(m_latency, 4),
            "event_alignment_accuracy": round(m_event_alignment, 4),
            "event_numeric_accuracy": round(m_event_numeric, 4),
            "event_tool_success_rate": round(m_event_tool, 4),
            "market_reaction_block_presence_rate": round(m_market_block, 4),
        },
        "data_reconciliation": data_reconciliation,
    }


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


def aggregate(results: list[dict]) -> dict[str, Any]:
    """Compute aggregate stats from individual results."""
    if not results:
        return {}

    total = len(results)
    errors = sum(1 for r in results if r.get("error"))

    metric_names = [
        "answer_accuracy", "numerical_accuracy", "retrieval_hit_rate",
        "citation_accuracy", "tool_success_rate", "latency_score",
        "event_alignment_accuracy", "event_numeric_accuracy",
        "event_tool_success_rate", "market_reaction_block_presence_rate",
    ]
    averages: dict[str, float] = {}
    for m in metric_names:
        vals = [r["metrics"][m] for r in results if not r.get("error")]
        averages[m] = round(sum(vals) / len(vals), 4) if vals else 0.0

    task_type_acc = sum(1 for r in results if r.get("task_type_correct")) / total
    tools_acc = sum(1 for r in results if r.get("tools_correct")) / total
    avg_latency = sum(r["elapsed_s"] for r in results) / total
    data_reconciliation = _aggregate_data_reconciliation(results)

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"count": 0, "metrics": {m: [] for m in metric_names}}
        categories[cat]["count"] += 1
        if not r.get("error"):
            for m in metric_names:
                categories[cat]["metrics"][m].append(r["metrics"][m])

    cat_summary: dict[str, dict] = {}
    for cat, data in categories.items():
        cat_summary[cat] = {
            "count": data["count"],
            "metrics": {
                m: round(sum(vs) / len(vs), 4) if vs else 0.0
                for m, vs in data["metrics"].items()
            },
        }

    return {
        "total": total,
        "errors": errors,
        "task_type_accuracy": round(task_type_acc, 4),
        "tool_selection_accuracy": round(tools_acc, 4),
        "average_latency_s": round(avg_latency, 1),
        "average_metrics": averages,
        "data_reconciliation": data_reconciliation,
        "by_category": cat_summary,
    }


def _aggregate_data_reconciliation(results: list[dict]) -> dict[str, Any]:
    provider_counts: dict[str, int] = {}
    conflict_count = 0
    fallback_yfinance_fact_count = 0
    sec_fact_count = 0
    for result in results:
        summary = result.get("data_reconciliation", {}) or {}
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


def print_report(summary: dict, results: list[dict]) -> None:
    """Print a readable table to the terminal."""
    print("\n" + "=" * 70)
    print("  EVALUATION REPORT")
    print("=" * 70)
    print(f"  Total: {summary['total']}  |  Errors: {summary['errors']}  "
          f"|  Avg latency: {summary['average_latency_s']}s")
    print(f"  Task-type accuracy: {summary['task_type_accuracy']:.1%}")
    print(f"  Tool-selection accuracy: {summary['tool_selection_accuracy']:.1%}")
    print()

    print("  Aggregate Metrics:")
    for m, v in summary["average_metrics"].items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"    {m:25s} {bar} {v:.1%}")
    print()

    recon = summary.get("data_reconciliation", {})
    print("  Data Reconciliation:")
    print(f"    sec_fact_count              : {recon.get('sec_fact_count', 0)}")
    print(f"    fallback_yfinance_fact_count: {recon.get('fallback_yfinance_fact_count', 0)}")
    print(f"    conflict_count              : {recon.get('conflict_count', 0)}")
    print(f"    conflict_rate               : {float(recon.get('conflict_rate', 0.0)):.1%}")
    print()

    print("  By Category:")
    print(f"    {'Category':<15} {'Count':>5}  {'Ans':>5}  {'Num':>5}  "
          f"{'Ret':>5}  {'Cite':>5}  {'Tool':>5}  {'Lat':>5}")
    print("    " + "-" * 62)
    for cat, data in summary["by_category"].items():
        m = data["metrics"]
        print(f"    {cat:<15} {data['count']:>5}  "
              f"{m['answer_accuracy']:>5.0%}  "
              f"{m['numerical_accuracy']:>5.0%}  "
              f"{m['retrieval_hit_rate']:>5.0%}  "
              f"{m['citation_accuracy']:>5.0%}  "
              f"{m['tool_success_rate']:>5.0%}  "
              f"{m['latency_score']:>5.0%}")
    print()

    # Show first few results
    print("  Sample Results (first 5):")
    print(f"    {'ID':<8} {'Cat':<12} {'Ans%':>5} {'Num%':>5} {'Time':>6} {'Err'}")
    print("    " + "-" * 50)
    for r in results[:5]:
        err_flag = "ERR" if r.get("error") else "ok"
        print(f"    {r['id']:<8} {r['category']:<12} "
              f"{r['metrics']['answer_accuracy']:>5.0%} "
              f"{r['metrics']['numerical_accuracy']:>5.0%} "
              f"{r['elapsed_s']:>5.1f}s {err_flag}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run FilingLens evaluation")
    parser.add_argument("--benchmark", default=str(ROOT / "eval" / "benchmark.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "eval" / "report.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--ids", default=None, help="Comma-separated IDs")
    args = parser.parse_args()

    ids_set = set(args.ids.split(",")) if args.ids else None
    items = load_benchmark(Path(args.benchmark), args.limit, args.category, ids_set)
    logger.info("Loaded %d benchmark items", len(items))

    from src.agent.graph import compile_agent
    agent = compile_agent()
    logger.info("Agent compiled")

    results: list[dict] = []
    for i, item in enumerate(items, 1):
        logger.info("[%d/%d] %s: %s", i, len(items), item.get("id"), item["question"][:60])
        r = run_single(agent, item)
        results.append(r)
        logger.info("  → ans=%.0f%% num=%.0f%% time=%.1fs %s",
                     r["metrics"]["answer_accuracy"] * 100,
                     r["metrics"]["numerical_accuracy"] * 100,
                     r["elapsed_s"],
                     "ERROR" if r.get("error") else "ok")

    summary = aggregate(results)
    report = {"summary": summary, "results": results}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    logger.info("Report saved to %s", out_path)

    print_report(summary, results)


if __name__ == "__main__":
    main()
