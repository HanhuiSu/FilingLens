#!/usr/bin/env python3
"""End-to-end verification — 5 typical questions via the FastAPI /chat endpoint."""

from __future__ import annotations

import time

import httpx

API = "http://localhost:8080"

QUESTIONS = [
    ("fact_qa", "Apple 最近一个季度的营收是多少？"),
    ("trend", "分析 Nvidia 最近 4 个季度营收变化趋势。"),
    ("comparison", "比较 Microsoft 和 Google 最近几个季度的营业利润率。"),
    ("summary", "根据最近的 10-K，总结 Tesla 的主要风险因素。"),
    ("hybrid", "结合财报文本和财务指标，概括 Amazon 最近几个季度的经营表现。"),
]


def main():
    client = httpx.Client(timeout=300)

    # Health check
    r = client.get(f"{API}/health")
    print(f"Health: {r.json()}")
    assert r.status_code == 200

    results = []
    for i, (cat, q) in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Q{i} [{cat}]: {q}")
        print(f"{'='*70}")
        t0 = time.time()
        r = client.post(f"{API}/chat", json={"query": q})
        elapsed = time.time() - t0

        if r.status_code != 200:
            print(f"ERROR {r.status_code}: {r.text[:200]}")
            results.append({"q": q, "cat": cat, "status": r.status_code, "error": r.text[:200]})
            continue

        body = r.json()
        trace_id = body.get("trace_id", "")
        answer = body.get("answer", "")
        task_type = body.get("task_type", "")
        used_tools = body.get("used_tools", [])
        citations = body.get("citations", [])

        print(f"  task_type:  {task_type}")
        print(f"  tools:      {used_tools}")
        print(f"  citations:  {len(citations)}")
        print(f"  trace_id:   {trace_id}")
        print(f"  time:       {elapsed:.1f}s")
        print("\n  --- Answer (first 600 chars) ---")
        print(f"  {answer[:600]}")

        # Verify trace endpoint
        tr = client.get(f"{API}/trace/{trace_id}")
        trace_ok = tr.status_code == 200
        if trace_ok:
            trace_data = tr.json()
            print("\n  --- Trace ---")
            print(f"  route:      {trace_data.get('data_route')}")
            print(f"  companies:  {trace_data.get('companies')}")
            print(f"  #tools:     {len(trace_data.get('tool_results', []))}")

        results.append({
            "q": q,
            "cat": cat,
            "status": 200,
            "task_type": task_type,
            "tools": used_tools,
            "citations_count": len(citations),
            "answer_len": len(answer),
            "trace_ok": trace_ok,
            "elapsed_s": round(elapsed, 1),
        })

    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Cat':<12} {'Status':>6} {'Type':<20} {'Tools':<45} {'Cite':>4} {'Time':>6} {'Trace'}")
    print("-" * 110)
    for i, r in enumerate(results, 1):
        if "error" in r:
            print(f"{i:<4} {r['cat']:<12} {r['status']:>6} ERROR")
        else:
            print(f"{i:<4} {r['cat']:<12} {r['status']:>6} {r['task_type']:<20} "
                  f"{str(r['tools']):<45} {r['citations_count']:>4} {r['elapsed_s']:>5.1f}s "
                  f"{'✓' if r['trace_ok'] else '✗'}")
    print(f"{'='*70}")

    all_ok = all(r.get("status") == 200 for r in results)
    all_traced = all(r.get("trace_ok", False) for r in results if r.get("status") == 200)
    print(f"\nAll 200: {'YES' if all_ok else 'NO'}")
    print(f"All traced: {'YES' if all_traced else 'NO'}")


if __name__ == "__main__":
    main()
