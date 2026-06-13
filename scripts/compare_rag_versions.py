#!/usr/bin/env python3
"""Compare RAG baseline reports for v1/v2 index versions and evaluate gates."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

GATE_THRESHOLDS = {
    "chunks_mixed_ratio_max": 0.20,
    "citation_mixed_ratio_max": 0.40,
    "chunks_duplicate_ratio_max": 0.03,
}


def _run_baseline(index_version: str, skip_tool_probe: bool) -> tuple[Path, dict[str, Any]]:
    script = ROOT / "scripts" / "report_rag_baseline.py"
    cmd = [sys.executable, str(script)]
    if skip_tool_probe:
        cmd.append("--skip-tool-probe")
    cmd.extend(["--suffix", index_version])

    env = os.environ.copy()
    env["RAG_INDEX_VERSION"] = index_version
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"baseline failed for {index_version}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    all_output = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"Wrote baseline report:\s*(.+)", all_output)
    if not match:
        raise RuntimeError(
            f"cannot locate baseline report path for {index_version}; output:\n{all_output}"
        )
    path = Path(match.group(1).strip())
    if not path.is_file():
        raise RuntimeError(f"baseline report not found for {index_version}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return path, data


def _delta(v2: float, v1: float) -> float:
    return round(v2 - v1, 6)


def _gate_eval(v2_report: dict[str, Any]) -> dict[str, Any]:
    db = v2_report.get("db_metrics", {})
    trace = v2_report.get("trace_metrics", {})

    checks = [
        {
            "name": "chunks_mixed_ratio",
            "value": float(db.get("chunks_mixed_ratio", 0.0)),
            "threshold_max": GATE_THRESHOLDS["chunks_mixed_ratio_max"],
        },
        {
            "name": "citation_mixed_ratio",
            "value": float(trace.get("citation_mixed_ratio", 0.0)),
            "threshold_max": GATE_THRESHOLDS["citation_mixed_ratio_max"],
        },
        {
            "name": "chunks_duplicate_ratio",
            "value": float(db.get("chunks_duplicate_ratio", 0.0)),
            "threshold_max": GATE_THRESHOLDS["chunks_duplicate_ratio_max"],
        },
    ]

    for item in checks:
        item["passed"] = item["value"] <= item["threshold_max"]

    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "checks": checks,
        "note": (
            "fact_qa / report_summary correctness still requires benchmark or human spot-check."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline on v1/v2 index versions and produce rollout comparison."
    )
    parser.add_argument(
        "--with-tool-probe",
        action="store_true",
        help="Enable live search_filings probes for each version.",
    )
    args = parser.parse_args()

    skip_tool_probe = not args.with_tool_probe

    v1_path, v1 = _run_baseline("v1", skip_tool_probe=skip_tool_probe)
    v2_path, v2 = _run_baseline("v2", skip_tool_probe=skip_tool_probe)

    v1_db = v1.get("db_metrics", {})
    v2_db = v2.get("db_metrics", {})
    v1_trace = v1.get("trace_metrics", {})
    v2_trace = v2.get("trace_metrics", {})

    comparison = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v1_report": str(v1_path),
            "v2_report": str(v2_path),
            "tool_probe_enabled": args.with_tool_probe,
        },
        "summary": {
            "chunks_mixed_ratio": {
                "v1": v1_db.get("chunks_mixed_ratio", 0.0),
                "v2": v2_db.get("chunks_mixed_ratio", 0.0),
                "delta_v2_minus_v1": _delta(
                    float(v2_db.get("chunks_mixed_ratio", 0.0)),
                    float(v1_db.get("chunks_mixed_ratio", 0.0)),
                ),
            },
            "chunks_duplicate_ratio": {
                "v1": v1_db.get("chunks_duplicate_ratio", 0.0),
                "v2": v2_db.get("chunks_duplicate_ratio", 0.0),
                "delta_v2_minus_v1": _delta(
                    float(v2_db.get("chunks_duplicate_ratio", 0.0)),
                    float(v1_db.get("chunks_duplicate_ratio", 0.0)),
                ),
            },
            "chunks_duplicate_ratio_global": {
                "v1": v1_db.get("chunks_duplicate_ratio_global", 0.0),
                "v2": v2_db.get("chunks_duplicate_ratio_global", 0.0),
                "delta_v2_minus_v1": _delta(
                    float(v2_db.get("chunks_duplicate_ratio_global", 0.0)),
                    float(v1_db.get("chunks_duplicate_ratio_global", 0.0)),
                ),
            },
            "citation_mixed_ratio": {
                "v1": v1_trace.get("citation_mixed_ratio", 0.0),
                "v2": v2_trace.get("citation_mixed_ratio", 0.0),
                "delta_v2_minus_v1": _delta(
                    float(v2_trace.get("citation_mixed_ratio", 0.0)),
                    float(v1_trace.get("citation_mixed_ratio", 0.0)),
                ),
            },
            "citation_diversity_avg": {
                "v1": v1_trace.get("citation_diversity_avg", 0.0),
                "v2": v2_trace.get("citation_diversity_avg", 0.0),
                "delta_v2_minus_v1": _delta(
                    float(v2_trace.get("citation_diversity_avg", 0.0)),
                    float(v1_trace.get("citation_diversity_avg", 0.0)),
                ),
            },
        },
        "gate": _gate_eval(v2),
    }

    comparison["recommendation"] = (
        "switch_default_to_v2" if comparison["gate"]["passed"] else "keep_v1_and_investigate"
    )

    out_dir = ROOT / "data" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"rag_rollout_compare_{ts}.json"
    out_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote rollout comparison report: {out_path}")
    print(f"Recommendation: {comparison['recommendation']}")


if __name__ == "__main__":
    main()
