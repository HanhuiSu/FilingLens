#!/usr/bin/env python3
"""Split processed filing text into section-aware chunks and store in DuckDB."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from src.db.queries import FilingChunkRow, delete_chunks_for_filing, get_connection, insert_chunks_batch


_HEADER_V2 = re.compile(
    r"(?m)^===\s*SECTION:(?P<section>[^|=]+?)\s*\|\s*PART:(?P<part>[^|=]+?)\s*\|"
    r"\s*INSTANCE:(?P<instance>\d+)\s*\|\s*QUALITY:(?P<quality>[^=]+?)\s*===\s*$"
)
_HEADER_V1 = re.compile(r"(?m)^===\s*(?P<section>[A-Z0-9_]+)\s*===\s*$")


def _splitter() -> RecursiveCharacterTextSplitter:
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except Exception:
        # Offline-safe fallback when tiktoken encoding assets are unavailable.
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )


def _default_quality(section: str) -> str:
    if section.startswith("ITEM_"):
        return "high"
    if section in {"PREFACE", "APPENDIX"}:
        return "medium"
    return "low"


def _parse_processed_sections(text: str) -> list[dict[str, Any]]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    matches_v2 = list(_HEADER_V2.finditer(text))
    if matches_v2:
        sections: list[dict[str, Any]] = []
        for i, m in enumerate(matches_v2):
            start = m.end()
            end = matches_v2[i + 1].start() if i + 1 < len(matches_v2) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            section = m.group("section").strip().upper()
            part = m.group("part").strip().upper()
            instance = int(m.group("instance"))
            quality = m.group("quality").strip().lower()
            if quality not in {"high", "medium", "low"}:
                quality = _default_quality(section)
            sections.append(
                {
                    "section": section,
                    "part": part or None,
                    "section_instance": instance,
                    "quality": quality,
                    "body": body,
                }
            )
        if sections:
            return sections

    matches_v1 = list(_HEADER_V1.finditer(text))
    if matches_v1:
        counters: dict[str, int] = {}
        sections: list[dict[str, Any]] = []
        for i, m in enumerate(matches_v1):
            start = m.end()
            end = matches_v1[i + 1].start() if i + 1 < len(matches_v1) else len(text)
            body = text[start:end].strip()
            if not body:
                continue
            section = m.group("section").strip().upper()
            counters[section] = counters.get(section, 0) + 1
            sections.append(
                {
                    "section": section,
                    "part": None,
                    "section_instance": counters[section],
                    "quality": _default_quality(section),
                    "body": body,
                }
            )
        if sections:
            return sections

    body = text.strip()
    if not body:
        return []
    return [
        {
            "section": "UNKNOWN",
            "part": None,
            "section_instance": 1,
            "quality": "low",
            "body": body,
        }
    ]


def main() -> None:
    conn = get_connection()
    rows_db = conn.execute(
        """
        SELECT filing_id, ticker, processed_path, form_type, COALESCE(fiscal_period, '') AS fp
        FROM filings_metadata
        WHERE processed_path IS NOT NULL AND processed_path != ''
        """
    ).fetchall()

    splitter = _splitter()
    for filing_id, ticker, processed_path, form_type, _fp in rows_db:
        p = Path(processed_path)
        if not p.is_file():
            print(f"Skip missing processed: {p}")
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if len(text.strip()) < 50:
            continue

        sections = _parse_processed_sections(text)
        delete_chunks_for_filing(conn, filing_id)
        batch: list[FilingChunkRow] = []
        safe_fid = filing_id.replace("-", "_")
        chunk_i = 0
        for sec in sections:
            chunk_texts = splitter.split_text(sec["body"])
            for chunk in chunk_texts:
                if len(chunk.strip()) < 20:
                    continue
                cid = f"{safe_fid}_{chunk_i}"
                batch.append(
                    FilingChunkRow(
                        chunk_id=cid,
                        filing_id=filing_id,
                        ticker=ticker,
                        section=sec["section"],
                        part=sec.get("part"),
                        section_instance=int(sec.get("section_instance") or 1),
                        quality=sec.get("quality"),
                        chunk_text=chunk,
                        chunk_order=chunk_i,
                    )
                )
                chunk_i += 1

        if batch:
            insert_chunks_batch(conn, batch)
        print(f"  {ticker} {filing_id}: {len(batch)} chunks")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
