#!/usr/bin/env python3
"""Parse SEC full-submission files into cleaned, section-aware filing text.

This parser explicitly extracts the primary 10-K / 10-Q document from EDGAR
full-submission bundles and avoids indexing attachment-heavy sections
(e.g. EX-101 XBRL taxonomy linkbases) as core narrative corpus.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings
from src.db.queries import get_connection, update_filing_processed_path


_DOC_BLOCK_RE = re.compile(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>")
_DOC_TYPE_RE = re.compile(r"(?im)^\s*<TYPE>\s*([^\r\n<]+)")
_DOC_TEXT_RE = re.compile(r"(?is)<TEXT>(.*?)</TEXT>")

# Match PART I / PART II...
_PART_HEADING = re.compile(r"(?im)^\s*PART[\s\u00A0]+([IVX]+)\b")
# Match ITEM headings in plain text (10-K / 10-Q)
_ITEM_HEADING = re.compile(
    r"(?im)^\s*(?:ITEM|Item)[\s\u00A0]+(\d+[A-Za-z]?)\s*[.\u00b7\-–—]?\s*",
    re.MULTILINE,
)

_XBRL_MARKERS = (
    "xbrl.sec.gov",
    "xbrl.org",
    "fasb.org/us-gaap",
    "xmlns:",
    "linkbase",
    "schemaRef",
    "xbrli:",
    "ix:",
)


def _normalize_space(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00A0", " ")
        .replace("\u2007", " ")
        .replace("\u202F", " ")
    )


def _looks_technical(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    if any(m in low for m in _XBRL_MARKERS):
        return True
    if re.match(r"^[a-z][a-z0-9_.-]*:[a-z0-9_.-]+(\s+[a-z][a-z0-9_.-]*:[a-z0-9_.-]+){2,}$", low):
        return True
    if len(s) >= 80 and re.search(r"(?:\w+\.){2,}\w+", s) and sum(ch in "<>{}[];=" for ch in s) >= 4:
        return True
    alnum = sum(ch.isalnum() for ch in s)
    if len(s) >= 60 and alnum / max(len(s), 1) < 0.35:
        return True
    if s.upper().startswith(("EX-101", "XBRL TAXONOMY", "IDEA: XBRL")):
        return True
    return False


def _extract_document_blocks(raw: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    for m in _DOC_BLOCK_RE.finditer(raw):
        block = m.group(1)
        m_type = _DOC_TYPE_RE.search(block)
        doc_type = (m_type.group(1).strip().upper() if m_type else "")
        m_text = _DOC_TEXT_RE.search(block)
        doc_text = m_text.group(1) if m_text else block
        blocks.append({"type": doc_type, "text": doc_text})
    return blocks


def _select_primary_document(raw: str, form_type: str) -> str:
    blocks = _extract_document_blocks(raw)
    target = form_type.upper().strip()
    if blocks:
        for b in blocks:
            if b["type"] == target:
                return b["text"]
        for b in blocks:
            if b["type"] in {"10-K", "10-Q"}:
                return b["text"]
    return raw


def _html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove hidden inline-XBRL heavy blocks.
    for tag_name in ("ix:header", "ix:hidden"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    for tag in soup.find_all(style=True):
        attrs = getattr(tag, "attrs", None) or {}
        style = str(attrs.get("style", "")).replace(" ", "").lower()
        if "display:none" in style:
            tag.decompose()

    text = _normalize_space(soup.get_text(separator="\n"))
    lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s) > 500:
            s = s[:500] + " …"
        if _looks_technical(s):
            continue
        lines.append(s)
    return "\n".join(lines)


def _part_for_position(part_marks: list[tuple[int, str]], pos: int) -> str | None:
    current: str | None = None
    for p, part in part_marks:
        if p <= pos:
            current = part
        else:
            break
    return current


def _split_sections(text: str) -> list[dict[str, Any]]:
    """Return section blocks with section/part/instance/quality metadata."""
    text = _normalize_space(text)
    part_marks = [(m.start(), f"PART_{m.group(1).upper()}") for m in _PART_HEADING.finditer(text)]
    item_matches = list(_ITEM_HEADING.finditer(text))
    if not item_matches:
        return [
            {
                "section": "UNKNOWN",
                "part": _part_for_position(part_marks, 0),
                "instance": 1,
                "quality": "low",
                "body": text.strip(),
            }
        ]

    sections: list[dict[str, Any]] = []
    first_start = item_matches[0].start()
    prefix = text[:first_start].strip()
    if len(prefix) >= 200:
        sections.append(
            {
                "section": "PREFACE",
                "part": _part_for_position(part_marks, 0),
                "instance": 1,
                "quality": "medium",
                "body": prefix,
            }
        )

    instance_counter: dict[str, int] = {}
    for idx, match in enumerate(item_matches):
        section = f"ITEM_{match.group(1).upper()}"
        start = match.start()
        end = item_matches[idx + 1].start() if idx + 1 < len(item_matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        instance_counter[section] = instance_counter.get(section, 0) + 1
        sections.append(
            {
                "section": section,
                "part": _part_for_position(part_marks, start),
                "instance": instance_counter[section],
                "quality": "high",
                "body": body,
            }
        )

    if not sections:
        return [
            {
                "section": "UNKNOWN",
                "part": _part_for_position(part_marks, 0),
                "instance": 1,
                "quality": "low",
                "body": text.strip(),
            }
        ]
    return sections


def _fiscal_period_guess(text: str, form_type: str) -> str | None:
    t = text[:50_000].upper()
    m = re.search(
        r"FISCAL\s+(?:YEAR\s+)?(?:QUARTER\s+)?(?:ENDED|ENDING)\s+([A-Z]+\s+\d{1,2},\s*\d{4})",
        t,
    )
    if m:
        return m.group(1).strip()
    m2 = re.search(r"THREE\s+MONTHS\s+ENDED\s+([A-Z]+\s+\d{1,2},\s*\d{4})", t)
    if m2:
        return m2.group(1).strip()
    if "FORM 10-K" in t or form_type == "10-K":
        y = re.search(r"FISCAL\s+YEAR\s+ENDED\s+.*?(\d{4})", t)
        if y:
            return f"FY{y.group(1)}"
    return None


def process_one(
    filing_id: str,
    ticker: str,
    form_type: str,
    local_path: Path,
    out_dir: Path,
) -> tuple[Path, str | None]:
    raw = local_path.read_text(encoding="utf-8", errors="ignore")
    primary_doc = _select_primary_document(raw, form_type=form_type)
    text = _html_to_text(primary_doc)
    if len(text.strip()) < 500:
        # Fallback for non-HTML looking documents.
        text = re.sub(r"<[^>]+>", " ", primary_doc)
        text = _normalize_space(re.sub(r"\s+", " ", text))

    sections = _split_sections(text)
    fiscal = _fiscal_period_guess(text, form_type)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = filing_id.replace("/", "_")
    out_txt = out_dir / f"{safe_name}.txt"

    parts = []
    for sec in sections:
        sec_name = sec["section"]
        part = sec.get("part") or "UNKNOWN_PART"
        instance = int(sec.get("instance") or 1)
        quality = sec.get("quality") or "low"
        body = sec.get("body", "").strip()
        if not body:
            continue
        header = f"=== SECTION:{sec_name} | PART:{part} | INSTANCE:{instance} | QUALITY:{quality} ==="
        parts.append(f"{header}\n{body}")
    out_txt.write_text("\n\n".join(parts), encoding="utf-8")

    dbg = out_dir / f"{safe_name}.sections.json"
    dbg.write_text(
        json.dumps(
            [
                {
                    "section": s.get("section"),
                    "part": s.get("part"),
                    "instance": int(s.get("instance") or 1),
                    "quality": s.get("quality"),
                    "chars": len(s.get("body", "")),
                }
                for s in sections
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_txt, fiscal


def main() -> None:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT filing_id, ticker, form_type, local_path
        FROM filings_metadata
        ORDER BY ticker, filing_date NULLS LAST, filing_id
        """
    ).fetchall()

    for filing_id, ticker, form_type, local_path in rows:
        lp = Path(local_path)
        if not lp.is_file():
            print(f"Skip missing file: {lp}")
            continue
        out_dir = settings.processed_dir / ticker
        try:
            out_path, fiscal = process_one(filing_id, ticker, form_type, lp, out_dir)
        except Exception as exc:
            print(f"Error {filing_id}: {exc}")
            continue
        update_filing_processed_path(conn, filing_id, str(out_path.resolve()), fiscal)
        print(f"Parsed {ticker} {filing_id} -> {out_path.name}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
