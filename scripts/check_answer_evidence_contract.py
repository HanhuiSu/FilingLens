#!/usr/bin/env python3
# ruff: noqa: E402
"""Post-hoc check of a stored trace against the answer evidence contract."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from src.agent.answer_contract import check_answer_evidence_contract


def _load_trace_from_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace_by_id(trace_id: str, api_base: str = "") -> dict[str, Any]:
    local_path = settings.traces_dir / f"{trace_id}.json"
    if local_path.exists():
        return _load_trace_from_path(local_path)
    if not api_base:
        raise FileNotFoundError(f"Trace {trace_id} not found in {settings.traces_dir}")
    url = f"{api_base.rstrip('/')}/trace/{trace_id}"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - explicit user-provided local/API URL
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc answer evidence contract validation for a trace.")
    parser.add_argument("--trace", type=Path, default=None, help="Path to a trace JSON file.")
    parser.add_argument("--trace-id", default="", help="Trace id to resolve from local traces dir or API.")
    parser.add_argument("--api-base", default="", help="Optional API base for trace-id lookup if local trace is missing.")
    args = parser.parse_args()
    if not args.trace and not args.trace_id:
        parser.error("Provide --trace or --trace-id.")
    trace = _load_trace_from_path(args.trace) if args.trace else _load_trace_by_id(args.trace_id, args.api_base)
    result = check_answer_evidence_contract(trace)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()
