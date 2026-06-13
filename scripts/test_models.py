"""Verify that LLM and Embedding models are accessible and working.

Run after starting the vLLM server:
    python scripts/test_models.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# Qwen3 thinking mode produces <think>...</think> blocks that consume
# tokens.  We need generous max_tokens for even simple answers.
_CHAT_MAX_TOKENS = 512
_TOOL_MAX_TOKENS = 1024


def _banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ------------------------------------------------------------------
# Test 1: LLM basic chat completion
# ------------------------------------------------------------------
def test_llm_chat():
    _banner("Test 1 — LLM Chat Completion")
    from openai import OpenAI

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=settings.llm_reasoning_model,
        messages=[{"role": "user", "content": "What is Apple Inc.'s ticker symbol? Reply in one word."}],
        max_tokens=_CHAT_MAX_TOKENS,
    )
    elapsed = time.time() - t0
    msg = resp.choices[0].message
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""

    print(f"  Answer  : {content.strip()[:200]}")
    if reasoning:
        print(f"  Thinking: {reasoning.strip()[:120]}...")
    print(f"  Latency : {elapsed:.2f}s")
    print(f"  Tokens  : prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}")
    assert content.strip(), "Empty content from LLM (thinking may have consumed all tokens)"
    print("  ✓ PASSED")


# ------------------------------------------------------------------
# Test 2: LLM tool calling
# ------------------------------------------------------------------
def test_llm_tool_calling():
    _banner("Test 2 — LLM Tool Calling")
    from openai import OpenAI

    client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_financial_data",
                "description": "Query structured financial data for a company",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"},
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Financial metrics to query",
                        },
                    },
                    "required": ["ticker", "metrics"],
                },
            },
        }
    ]

    t0 = time.time()
    resp = client.chat.completions.create(
        model=settings.llm_reasoning_model,
        messages=[{"role": "user", "content": "What was NVIDIA's revenue last quarter?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=_TOOL_MAX_TOKENS,
    )
    elapsed = time.time() - t0
    msg = resp.choices[0].message

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        print(f"  Tool called : {tc.function.name}")
        print(f"  Arguments   : {tc.function.arguments}")
        print(f"  Latency     : {elapsed:.2f}s")
        args = json.loads(tc.function.arguments)
        assert "ticker" in args, "Missing 'ticker' in tool call args"
        print("  ✓ PASSED")
    else:
        content = (msg.content or "")[:300]
        print("  WARNING: No tool_calls in response.")
        print(f"  Content: {content}")
        finish = resp.choices[0].finish_reason
        print(f"  Finish reason: {finish}")
        if "<tool_call>" in content:
            print("  → Model generated <tool_call> text but parser didn't extract it.")
            print("    Try: --tool-call-parser qwen3_xml")
        print("  ✗ FAILED")
        return False
    return True


# ------------------------------------------------------------------
# Test 3: Embedding model
# ------------------------------------------------------------------
def test_embedding():
    _banner("Test 3 — Embedding Model (sentence-transformers)")
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    model = SentenceTransformer(
        settings.embedding_model_name,
        device=settings.embedding_device,
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.2f}s on {settings.embedding_device}")

    texts = [
        "Apple reported revenue of $94.9 billion in Q3 2024.",
        "Risk factors include supply chain disruptions.",
        "The company's gross margin improved to 46.3%.",
    ]
    t0 = time.time()
    embeddings = model.encode(texts)
    encode_time = time.time() - t0

    print(f"  Encoded {len(texts)} texts in {encode_time:.3f}s")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Embedding dtype: {embeddings.dtype}")
    assert embeddings.shape[0] == len(texts), "Wrong number of embeddings"
    assert embeddings.shape[1] > 0, "Zero-dimension embeddings"
    print("  ✓ PASSED")
    return embeddings.shape[1]


# ------------------------------------------------------------------
# Test 4: LangGraph + ChatOpenAI integration
# ------------------------------------------------------------------
def test_langgraph_integration():
    _banner("Test 4 — LangGraph + ChatOpenAI Integration")
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_reasoning_model,
        max_tokens=_CHAT_MAX_TOKENS,
    )

    t0 = time.time()
    resp = llm.invoke("Say 'hello' in exactly one word.")
    elapsed = time.time() - t0

    content = resp.content or ""
    print(f"  Response: {content.strip()[:200]}")
    print(f"  Latency : {elapsed:.2f}s")
    assert content.strip(), "Empty response"
    print("  ✓ PASSED")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("FilingLens — Model Verification")
    print(f"LLM endpoint : {settings.llm_base_url}")
    print(f"LLM model    : {settings.llm_reasoning_model}")
    print(f"Embedding    : {settings.embedding_model_name} ({settings.embedding_device})")

    results = {}

    try:
        test_llm_chat()
        results["llm_chat"] = "✓"
    except Exception as e:
        results["llm_chat"] = f"✗ {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        ok = test_llm_tool_calling()
        results["llm_tool_calling"] = "✓" if ok else "✗"
    except Exception as e:
        results["llm_tool_calling"] = f"✗ {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        dim = test_embedding()
        results["embedding"] = f"✓ (dim={dim})"
    except Exception as e:
        results["embedding"] = f"✗ {e}"
        print(f"  ✗ FAILED: {e}")

    try:
        test_langgraph_integration()
        results["langgraph"] = "✓"
    except Exception as e:
        results["langgraph"] = f"✗ {e}"
        print(f"  ✗ FAILED: {e}")

    _banner("Summary")
    all_pass = True
    for name, status in results.items():
        flag = "PASS" if status.startswith("✓") else "FAIL"
        if flag == "FAIL":
            all_pass = False
        print(f"  [{flag}] {name}: {status}")

    if all_pass:
        print("\n  All tests passed! Ready for Phase 2.")
    else:
        print("\n  Some tests failed. Fix issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
