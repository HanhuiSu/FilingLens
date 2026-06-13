#!/usr/bin/env python3
"""Generate baseline quality metrics for filing corpus / citations / retrieval."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import duckdb

ROOT = Path(__file__).resolve().parent.parent


def _safe_ratio(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _db_metrics(db_path: Path) -> dict:
    con = duckdb.connect(str(db_path), read_only=True)
    total, mixed = con.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN section = 'MIXED' THEN 1 ELSE 0 END) AS mixed
        FROM filing_chunks
        """
    ).fetchone()
    uniq_global = con.execute(
        "SELECT COUNT(DISTINCT md5(chunk_text)) FROM filing_chunks"
    ).fetchone()[0]
    total_wf, dup_wf = con.execute(
        """
        WITH grouped AS (
            SELECT filing_id, md5(chunk_text) AS h, COUNT(*) AS c
            FROM filing_chunks
            GROUP BY 1, 2
        )
        SELECT
            SUM(c) AS total_rows,
            SUM(CASE WHEN c > 1 THEN c - 1 ELSE 0 END) AS duplicated_rows
        FROM grouped
        """
    ).fetchone()
    section_rows = con.execute(
        """
        SELECT section, COUNT(*) AS c
        FROM filing_chunks
        GROUP BY section
        ORDER BY c DESC
        LIMIT 20
        """
    ).fetchall()
    con.close()

    total_i = int(total or 0)
    mixed_i = int(mixed or 0)
    uniq_global_i = int(uniq_global or 0)
    total_wf_i = int(total_wf or 0)
    dup_wf_i = int(dup_wf or 0)
    return {
        "chunks_total": total_i,
        "chunks_mixed": mixed_i,
        "chunks_mixed_ratio": round(_safe_ratio(mixed_i, total_i), 6),
        "chunks_unique_text_global": uniq_global_i,
        "chunks_duplicate_ratio_global": round(_safe_ratio(total_i - uniq_global_i, total_i), 6),
        "chunks_duplicate_ratio": round(_safe_ratio(dup_wf_i, total_wf_i), 6),
        "section_top20": [{"section": s, "count": int(c)} for s, c in section_rows],
    }


def _trace_metrics(traces_dir: Path) -> dict:
    files = sorted(traces_dir.glob("*.json"))
    citation_total = 0
    citation_mixed = 0
    diversity_scores: list[float] = []
    with_citations = 0

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        cits = data.get("citations", []) or []
        if cits:
            with_citations += 1
            uniq = {(
                c.get("filing_id", ""),
                c.get("section", ""),
            ) for c in cits}
            diversity_scores.append(_safe_ratio(len(uniq), len(cits)))
        for c in cits:
            citation_total += 1
            if str(c.get("section", "")).upper() == "MIXED":
                citation_mixed += 1

    return {
        "trace_files": len(files),
        "trace_with_citations": with_citations,
        "citation_total": citation_total,
        "citation_mixed": citation_mixed,
        "citation_mixed_ratio": round(_safe_ratio(citation_mixed, citation_total), 6),
        "citation_diversity_avg": round(mean(diversity_scores), 6) if diversity_scores else 0.0,
    }


def _tool_diversity_probe() -> dict:
    """Optional live probe for top-k evidence diversity; skipped if model/index unavailable."""
    try:
        from src.tools.search_filings import search_filings
    except Exception as exc:
        return {"status": "skipped", "reason": f"import_failed: {exc}"}

    probes = [
        ("AAPL", "主要风险因素和管理层讨论"),
        ("MSFT", "recent revenue trend and risk factors"),
        ("AMZN", "cash flow and operating performance discussion"),
        ("NVDA", "supply chain risks and demand outlook"),
    ]
    rows = []
    try:
        for ticker, query in probes:
            result = search_filings.invoke({"ticker": ticker, "query": query, "top_k": 8})
            uniq_pairs = {(
                r.get("filing_id", ""),
                r.get("section", ""),
            ) for r in result}
            mixed = sum(1 for r in result if str(r.get("section", "")).upper() == "MIXED")
            rows.append(
                {
                    "ticker": ticker,
                    "query": query,
                    "returned": len(result),
                    "unique_filing_section": len(uniq_pairs),
                    "diversity_ratio": round(_safe_ratio(len(uniq_pairs), len(result)), 6),
                    "mixed_ratio": round(_safe_ratio(mixed, len(result)), 6),
                }
            )
    except Exception as exc:
        return {"status": "skipped", "reason": f"probe_failed: {exc}"}

    avg_div = mean([r["diversity_ratio"] for r in rows]) if rows else 0.0
    avg_mixed = mean([r["mixed_ratio"] for r in rows]) if rows else 0.0
    return {
        "status": "ok",
        "avg_diversity_ratio": round(avg_div, 6),
        "avg_mixed_ratio": round(avg_mixed, 6),
        "probes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG baseline quality report.")
    parser.add_argument(
        "--skip-tool-probe",
        action="store_true",
        help="Skip live search_filings probes (useful in offline / model-unavailable environments).",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional output filename suffix, e.g. 'v1' or 'v2'.",
    )
    args = parser.parse_args()

    from config import settings

    now = datetime.now(timezone.utc)
    report = {
        "generated_at_utc": now.isoformat(),
        "db_path": str(settings.duckdb_path),
        "rag_index_version": settings.rag_index_version,
        "db_metrics": _db_metrics(settings.duckdb_path),
        "trace_metrics": _trace_metrics(settings.traces_dir),
        "tool_probe": {"status": "skipped", "reason": "disabled_by_flag"}
        if args.skip_tool_probe
        else _tool_diversity_probe(),
    }

    reports_dir = settings.data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = str(args.suffix or "").strip()
    suffix = f"_{suffix}" if suffix else ""
    out_path = reports_dir / f"rag_baseline_{ts}{suffix}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote baseline report: {out_path}")


if __name__ == "__main__":
    main()
