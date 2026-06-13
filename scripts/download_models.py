"""Download Qwen3-8B (AWQ 4-bit) and Qwen3-Embedding-0.6B from HuggingFace."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Download models from HuggingFace Hub")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-8B-AWQ", help="LLM model ID")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B", help="Embedding model ID")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM download")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding download")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    if not args.skip_llm:
        print(f"\n{'='*60}")
        print(f"Downloading LLM: {args.llm_model}")
        print(f"{'='*60}")
        snapshot_download(
            repo_id=args.llm_model,
            repo_type="model",
        )
        print(f"✓ LLM model downloaded: {args.llm_model}")

    if not args.skip_embedding:
        print(f"\n{'='*60}")
        print(f"Downloading Embedding: {args.embedding_model}")
        print(f"{'='*60}")
        snapshot_download(
            repo_id=args.embedding_model,
            repo_type="model",
        )
        print(f"✓ Embedding model downloaded: {args.embedding_model}")

    print("\nAll models downloaded. They are cached in ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
