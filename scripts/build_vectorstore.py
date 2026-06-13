#!/usr/bin/env python3
"""Embed filing_chunks with Qwen3-Embedding and upsert into ChromaDB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings


def _meta_str(v) -> str:
    if v is None:
        return ""
    return str(v)[:500]


def _collection_name(index_version: str) -> str:
    if str(index_version).lower() == "v2":
        return settings.rag_collection_v2
    return settings.rag_collection_v1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vector index from DuckDB filing_chunks")
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for embedding (default: settings.embedding_device, e.g. cuda or cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="chunks per Chroma add batch (default: settings.embedding_batch_size)",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=None,
        help="forward batch inside model.encode; lower avoids OOM on 12GB GPU (default: 8 cuda, 48 cpu)",
    )
    parser.add_argument(
        "--index-version",
        choices=["v1", "v2"],
        default=settings.rag_index_version,
        help="Target vector index version. v1 keeps legacy collection, v2 builds improved corpus.",
    )
    args = parser.parse_args()

    device = args.device or settings.embedding_device
    index_version = args.index_version
    collection_name = _collection_name(index_version)
    batch_size = args.batch_size if args.batch_size is not None else settings.embedding_batch_size
    if args.encode_batch_size is not None:
        encode_bs = args.encode_batch_size
    else:
        encode_bs = 8 if str(device).lower().startswith("cuda") else 48
    import duckdb

    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(settings.duckdb_path))
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.filing_id, c.ticker, c.section, c.part, c.section_instance, c.quality,
               c.chunk_text, c.chunk_order,
               m.form_type, m.fiscal_period
        FROM filing_chunks c
        JOIN filings_metadata m ON c.filing_id = m.filing_id
        ORDER BY c.ticker, c.filing_id, c.chunk_order
        """
    ).fetchall()
    conn.close()

    if not rows:
        print("No filing_chunks rows; run chunk_filings.py first.")
        return

    print(
        f"Embedding model={settings.embedding_model_name} device={device} index={index_version} "
        f"collection={collection_name} chroma_batch={batch_size} encode_batch={encode_bs}"
    )
    if str(device).lower().startswith("cuda"):
        print("Tip: stop vLLM / other GPU jobs first if you see CUDA OOM.")
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    model = SentenceTransformer(
        settings.embedding_model_name,
        device=device,
        trust_remote_code=True,
    )

    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    coll = client.create_collection(
        name=collection_name,
        metadata={"description": f"SEC filing text chunks ({index_version})"},
    )

    bs = batch_size
    total = 0
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        ids = [r[0] for r in batch]
        docs = [r[7] for r in batch]
        metadatas = [
            {
                "filing_id": _meta_str(r[1]),
                "ticker": _meta_str(r[2]),
                "section": _meta_str(r[3]),
                "part": _meta_str(r[4]),
                "section_instance": int(r[5]) if r[5] is not None else 1,
                "quality": _meta_str(r[6]),
                "chunk_order": int(r[8]),
                "form_type": _meta_str(r[9]),
                "fiscal_period": _meta_str(r[10]),
            }
            for r in batch
        ]
        emb = model.encode(
            docs,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=encode_bs,
        )
        vectors = emb.tolist()
        coll.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=vectors)
        total += len(batch)
        print(f"  Embedded {total}/{len(rows)}")

    print(f"Done. Chroma collection '{collection_name}' has {coll.count()} vectors.")


if __name__ == "__main__":
    main()
