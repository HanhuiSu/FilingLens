#!/usr/bin/env python3
"""End-to-end smoke test for the financial-analysis agent.

Requires:
  - vLLM server running on localhost:8000
  - DuckDB populated at data/db/financial.duckdb
  - ChromaDB vectorstore at data/vectorstore/

Usage:
  PYTHONPATH=/home/hui/agent python scripts/test_agent_e2e.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.graph import compile_agent

QUESTIONS = [
    "Apple 2024 年 Q3 的营收是多少？",
    "分析 Nvidia 最近 4 个季度营收变化趋势。",
    "根据最近的 10-K，总结 Tesla 的主要风险因素。",
]


def main():
    agent = compile_agent()
    print(f"Agent compiled. Testing {len(QUESTIONS)} questions...\n")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"{'='*60}")
        print(f"Q{i}: {q}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            result = agent.invoke({"user_query": q})
            elapsed = time.time() - t0

            print(f"\n[task_type]  {result.get('task_type', '?')}")
            print(f"[route]      {result.get('data_route', '?')}")
            print(f"[companies]  {result.get('companies', [])}")
            print(f"[tools]      {result.get('selected_tools', [])}")
            print(f"[#docs]      {len(result.get('retrieved_docs', []))}")
            print(f"[#results]   {len(result.get('tool_results', []))}")
            print(f"[citations]  {len(result.get('citations', []))}")
            print(f"[time]       {elapsed:.1f}s")
            print("\n--- Answer ---")
            answer = result.get("final_answer", "(no answer)")
            if len(answer) > 1500:
                print(answer[:1500] + "\n... (truncated)")
            else:
                print(answer)
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"ERROR after {elapsed:.1f}s: {exc}")
            import traceback
            traceback.print_exc()
        print()


if __name__ == "__main__":
    main()
