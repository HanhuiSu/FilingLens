# Data Sources

The agent separates document evidence, structured financial facts, and price/event data. Numeric evidence is provenance-aware so reviewers can see whether a value came from SEC XBRL facts, yfinance fallback, computed metrics, or event-window price data.

## SEC Filing Text

SEC 10-K and 10-Q filings are downloaded, parsed, sectioned, chunked, embedded, and stored in ChromaDB.

Used for:

- report summaries
- risk summaries
- business summaries
- MD&A/revenue trend explanations
- document citations
- retrieval gold eval

Document citations include ticker, form type, fiscal period, section, chunk order, quality, supporting snippet, and retrieval score breakdown when available.

## SEC Companyfacts Structured Facts

SEC Companyfacts is the preferred structured source for financial facts when available.

Provider metadata:

- `source_provider=sec_companyfacts`
- `confidence=high`
- `extraction_method=xbrl_companyfacts`
- `source_tag=<US-GAAP tag>`
- `source_filing_id=<SEC accession number>`

Canonical metrics are mapped from US-GAAP tags through `data/metric_mappings.yaml`.

## yfinance Structured Fallback

yfinance remains a fallback for structured financial facts when SEC Companyfacts coverage is missing.

Provider metadata:

- `source_provider=yfinance`
- `confidence=medium`
- `extraction_method=api_statement`
- `source_tag=<statement row name>`

The system must not present yfinance fallback as SEC filing evidence. When an answer uses yfinance structured facts, output limitations state that the structured facts are sourced from yfinance fallback, not SEC XBRL original facts.

## Price History And Event Windows

yfinance price history is used for:

- close/adjusted close/volume queries
- filing event windows
- post-filing 1/3/5/10 trading-day returns

Event-window evidence uses:

- `source_provider=event_price_window`
- `confidence=high`
- `extraction_method=event_price_window`

Event windows are precomputed from filing dates and trading-day price data in DuckDB.

## Precedence And Reconciliation

For the same `(ticker, period_end, period_type, metric)`:

1. Use SEC Companyfacts if available.
2. Fall back to yfinance only when SEC structured facts are unavailable.
3. If SEC and yfinance both exist and differ beyond tolerance, prefer SEC and expose `reconciliation_warning=sec_yfinance_value_mismatch`.

`period_context.source_summary` reports provider counts, SEC fact counts, fallback yfinance counts, and conflict counts/rates.

## Local Asset Status

Run:

```bash
make data-status
```

The command reports:

- DuckDB path and table counts
- companies, filings, chunks, financial facts, prices, events, event windows
- financial facts provider distribution
- Chroma path, configured RAG index version, collections, and vector counts

It does not require vLLM or the embedding model.
