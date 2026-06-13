"""LLM helpers for agent nodes."""

from __future__ import annotations

import json
import logging
import re

from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)

def _get_llm(
    reasoning: bool = True,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> ChatOpenAI:
    model = settings.llm_reasoning_model if reasoning else settings.llm_fast_model
    extra_body = {}
    api_provider = str(settings.llm_provider or "").lower() == "api"
    if api_provider:
        if settings.llm_enable_thinking is not None:
            extra_body["enable_thinking"] = settings.llm_enable_thinking
        if settings.llm_thinking_budget is not None:
            extra_body["thinking_budget"] = settings.llm_thinking_budget
        if settings.llm_reasoning_effort:
            extra_body["reasoning_effort"] = settings.llm_reasoning_effort

    kwargs = {"extra_body": extra_body} if extra_body else {}
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.llm_timeout_seconds if timeout is None else timeout,
        max_retries=settings.llm_max_retries if max_retries is None else max_retries,
        **kwargs,
    )

def _parse_json_response(text: str) -> dict:
    """Best-effort extraction of a JSON object from LLM output."""
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("_parse_json_response: JSON decode failed: %s", exc)
    return {}
