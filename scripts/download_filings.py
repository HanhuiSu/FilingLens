#!/usr/bin/env python3
"""Download 10-K and 10-Q from SEC EDGAR via sec-edgar-downloader, then index paths in DuckDB."""

from __future__ import annotations

import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import sec_edgar_identity, settings
from sec_edgar_downloader import Downloader

from src.db.queries import FilingMetadataRow, get_connection, seed_companies, upsert_filing_metadata
from src.utils_data import load_companies_yaml

SEC_FILINGS_ROOT = "sec-edgar-filings"
FULL_SUBMISSION = "full-submission.txt"


def _filing_date_from_file(path: Path) -> date | None:
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:200_000]
    except OSError:
        return None
    # Common patterns in EDGAR full-submission
    for pat in (
        r"FILED AS OF DATE:\s*(\d{8})",
        r"<ACCEPTANCE-DATETIME>(\d{8})",
        r"CONFORMED PERIOD OF REPORT:\s*(\d{8})",
    ):
        m = re.search(pat, head, re.I)
        if m:
            d = m.group(1)
            try:
                return date(int(d[:4]), int(d[4:6]), int(d[6:8]))
            except ValueError:
                continue
    return None


def _sec_viewer_url(cik: str, accession: str) -> str:
    acc_nodash = accession.replace("-", "")
    cik_num = str(int(cik))  # strip leading zeros for URL
    return (
        f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik_num}"
        f"&accession_number={acc_nodash}&xbrl_type=v"
    )


def scan_and_index_filings(conn, ticker_to_cik: dict[str, str]) -> int:
    base = settings.raw_dir / SEC_FILINGS_ROOT
    if not base.is_dir():
        return 0
    rows: list[FilingMetadataRow] = []
    for ticker in settings.target_tickers:
        t = ticker.upper()
        cik = ticker_to_cik.get(t, ticker_to_cik.get(ticker, ""))
        for form in ("10-K", "10-Q"):
            form_dir = base / t / form
            if not form_dir.is_dir():
                continue
            for acc_dir in sorted(form_dir.iterdir()):
                if not acc_dir.is_dir():
                    continue
                filing_id = acc_dir.name
                local = acc_dir / FULL_SUBMISSION
                if not local.is_file():
                    continue
                fd = _filing_date_from_file(local)
                url = _sec_viewer_url(cik, filing_id) if cik else ""
                rows.append(
                    FilingMetadataRow(
                        filing_id=filing_id,
                        ticker=t,
                        form_type=form,
                        fiscal_period=None,
                        filing_date=fd,
                        source_url=url or None,
                        local_path=str(local.resolve()),
                    )
                )
    if rows:
        upsert_filing_metadata(conn, rows)
    return len(rows)


def main() -> None:
    name, email = sec_edgar_identity()
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection()
    companies = load_companies_yaml(settings.companies_yaml)
    tickers = {t.upper() for t in settings.target_tickers}
    seed_companies(conn, [c for c in companies if c.ticker in tickers])

    dl = Downloader(name, email, download_folder=settings.raw_dir)
    after = date.today() - timedelta(days=365 * settings.data_years)
    limit_k = max(4, settings.data_years + 1)
    limit_q = max(12, settings.data_years * 4 + 2)

    mapping = {k.upper(): v for k, v in dl.ticker_to_cik_mapping.items()}

    for ticker in settings.target_tickers:
        t = ticker.upper()
        time.sleep(0.35)
        n1 = dl.get("10-K", t, after=after, limit=limit_k)
        print(f"  {t} 10-K: downloaded {n1} new filings")
        time.sleep(0.35)
        n2 = dl.get("10-Q", t, after=after, limit=limit_q)
        print(f"  {t} 10-Q: downloaded {n2} new filings")

    n = scan_and_index_filings(conn, mapping)
    print(f"Indexed {n} filings in filings_metadata.")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
