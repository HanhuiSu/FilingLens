"""Centralised configuration powered by pydantic-settings.

Reads from environment variables / .env file.  Every module should
``from config import settings`` and use ``settings.xxx`` to access values.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Local LLM (vLLM OpenAI-compatible API) ----
    llm_provider: str = "local"  # local | api
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "not-needed"
    llm_reasoning_model: str = "Qwen/Qwen3-8B-AWQ"
    llm_fast_model: str = "Qwen/Qwen3-8B-AWQ"
    llm_enable_thinking: bool | None = None
    llm_thinking_budget: int | None = None
    llm_reasoning_effort: str | None = None
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 1
    llm_classify_timeout_seconds: float = 25.0
    llm_classify_max_retries: int = 0
    llm_classify_fallback_enabled: bool = True
    analyst_draft_enabled: bool = True
    analyst_draft_max_attempts: int = 2
    analyst_draft_max_tokens: int = 1800
    semantic_query_parser_mode: str = "off"  # off | shadow | validated
    research_planner_mode: str = "expanded"  # off | shadow | validated | expanded
    research_planner_timeout_seconds: float = 25.0
    research_planner_fallback_for_timeout: bool = True

    # ---- Embedding (sentence-transformers, loaded in-process) ----
    # Default GPU for bulk indexing; set EMBEDDING_DEVICE=cpu if vLLM already fills VRAM.
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 256

    # ---- SEC EDGAR ----
    # Downloader API requires separate name + email; if blank, parsed from sec_edgar_user_agent.
    sec_edgar_company_name: str = ""
    sec_edgar_email: str = ""
    sec_edgar_user_agent: str = "YourName your-email@example.com"

    # ---- Data pipeline ----
    data_years: int = 3
    companies_yaml: Path = _PROJECT_ROOT / "data" / "companies.yaml"

    # ---- Paths ----
    data_dir: Path = _PROJECT_ROOT / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "vectorstore"

    @property
    def duckdb_path(self) -> Path:
        return self.data_dir / "db" / "financial.duckdb"

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    # ---- Target companies ----
    target_tickers: list[str] = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
        "TSLA", "JPM", "JNJ", "META", "AMD",
        "INTC", "AVGO", "QCOM", "ORCL", "CRM",
        "ADBE", "NOW", "COST", "WMT", "NKE",
        "MCD", "BAC", "GS", "MS", "V",
        "PFE", "UNH", "MRK", "LLY", "XOM",
    ]

    # ---- RAG ----
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 8
    rag_index_version: str = "v1"  # v1 | v2
    rag_collection_v1: str = "filing_chunks"
    rag_collection_v2: str = "filing_chunks_v2"
    rag_mixed_fallback: bool = True
    rag_mmr_lambda: float = 0.75
    rag_overfetch_multiplier: int = 8

    # ---- FastAPI ----
    api_host: str = "0.0.0.0"
    api_port: int = 8080


settings = Settings()


def sec_edgar_identity() -> tuple[str, str]:
    """Return (company_name, email) for sec_edgar_downloader.Downloader."""
    name = settings.sec_edgar_company_name.strip()
    email = settings.sec_edgar_email.strip()
    if name and email:
        return name, email
    # Parse "Name email@domain.com" from legacy single field
    parts = settings.sec_edgar_user_agent.strip().split()
    if len(parts) >= 2 and "@" in parts[-1]:
        return " ".join(parts[:-1]), parts[-1]
    raise ValueError(
        "Set SEC_EDGAR_COMPANY_NAME + SEC_EDGAR_EMAIL, or SEC_EDGAR_USER_AGENT as 'Name email@example.com'"
    )
