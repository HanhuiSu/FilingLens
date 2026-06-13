#!/usr/bin/env python3
"""Run all Phase-2 data scripts in order (SEC download may take a long time)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_SCRIPTS = [
    "download_filings.py",
    "parse_filings.py",
    "download_prices.py",
    "load_financial_data.py",
    "load_sec_companyfacts.py",
    "build_filing_events.py",
    "build_event_price_windows.py",
    "chunk_filings.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run phase-2 data pipeline scripts in order.")
    parser.add_argument(
        "--build-both-indexes",
        action="store_true",
        help="Build both Chroma collections (v1 and v2) for gray rollout.",
    )
    parser.add_argument(
        "--with-baseline-report",
        action="store_true",
        help="Generate baseline report at the end (report_rag_baseline.py --skip-tool-probe).",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    for name in BASE_SCRIPTS:
        path = ROOT / "scripts" / name
        print(f"\n>>> Running {name}\n")
        r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), env=env)
        if r.returncode != 0:
            print(f"Pipeline stopped: {name} exited {r.returncode}")
            sys.exit(r.returncode)

    build_script = ROOT / "scripts" / "build_vectorstore.py"
    if args.build_both_indexes:
        for version in ("v1", "v2"):
            print(f"\n>>> Running build_vectorstore.py --index-version {version}\n")
            r = subprocess.run(
                [sys.executable, str(build_script), "--index-version", version],
                cwd=str(ROOT),
                env=env,
            )
            if r.returncode != 0:
                print(f"Pipeline stopped: build_vectorstore.py({version}) exited {r.returncode}")
                sys.exit(r.returncode)
    else:
        print("\n>>> Running build_vectorstore.py\n")
        r = subprocess.run([sys.executable, str(build_script)], cwd=str(ROOT), env=env)
        if r.returncode != 0:
            print(f"Pipeline stopped: build_vectorstore.py exited {r.returncode}")
            sys.exit(r.returncode)

    if args.with_baseline_report:
        report_script = ROOT / "scripts" / "report_rag_baseline.py"
        print("\n>>> Running report_rag_baseline.py --skip-tool-probe\n")
        r = subprocess.run(
            [sys.executable, str(report_script), "--skip-tool-probe"],
            cwd=str(ROOT),
            env=env,
        )
        if r.returncode != 0:
            print(f"Pipeline stopped: report_rag_baseline.py exited {r.returncode}")
            sys.exit(r.returncode)

    print("\nPhase 2 pipeline finished.")


if __name__ == "__main__":
    main()
