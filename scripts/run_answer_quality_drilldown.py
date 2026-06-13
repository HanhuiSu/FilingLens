#!/usr/bin/env python3
# ruff: noqa: E402
"""Generate a markdown drilldown for methodology answer quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_methodology_eval import run_methodology_eval


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n```"


def render_drilldown(report: dict[str, Any]) -> str:
    lines = [
        "# Methodology Answer Quality Drilldown",
        "",
        f"- mode: {report.get('mode', '')}",
        f"- benchmark: `{report.get('benchmark_path', '')}`",
        f"- generated_at: {report.get('generated_at', '')}",
        "",
        "## Summary",
        "",
        _json_block(report.get("summary", {})),
    ]
    for record in report.get("records", []) or []:
        actual = dict(record.get("actual", {}) or {})
        contract = dict(actual.get("answer_contract", {}) or {})
        lines.extend(
            [
                "",
                f"## {record.get('id', '')}: {record.get('query', '')}",
                "",
                "### QueryUnderstanding",
                _json_block(actual.get("query_understanding", {})),
                "",
                "### Active Dimensions",
                ", ".join(actual.get("active_dimensions", []) or []) or "None",
                "",
                "### DimensionStatus",
                _json_block(actual.get("dimension_statuses", {})),
                "",
                "### Final Answer Preview",
                str(actual.get("answer_preview", "") or ""),
                "",
                "### Evidence Contract Result",
                _json_block(contract),
                "",
                "### Caveats Surfaced",
                _json_block(
                    {
                        "caveat_visibility_rate": dict(contract.get("metrics", {}) or {}).get("caveat_visibility_rate"),
                        "violations": [
                            item
                            for item in contract.get("violations", []) or []
                            if isinstance(item, dict) and item.get("type") == "caveat_not_visible"
                        ],
                    }
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer-quality drilldown for methodology benchmark.")
    parser.add_argument("--benchmark", default="eval/methodology_answer_benchmark.jsonl")
    parser.add_argument("--out-md", default="docs/methodology_answer_quality_drilldown.md")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = run_methodology_eval(Path(args.benchmark), mode="answer", limit=args.limit)
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_drilldown(report), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
