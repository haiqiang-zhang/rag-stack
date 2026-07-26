"""Convert dragonball JSONL files to RAG-Stack parquet datasets.

Raw-corpus format: chunk_size is a search-space dimension, so the pipeline
chunks at eval time — the corpus is stored un-chunked and QA carries no
chunk-level retrieval_gt.

Outputs:
    raw_corpus{lang}.parquet    — un-chunked docs {path, contents, doc_id, domain, language}
    qa{lang}.parquet            — QA without retrieval_gt
    qa{lang}_sample_{N}.parquet — sampled from the existing qa{lang}.parquet

Usage:
    # Full conversion: raw_corpus_en.parquet + qa_en.parquet
    python datasets/dragonball/convert_dragonball_to_parquet.py --lang en
    # Sample 400 queries from the existing qa_en.parquet (qids stay stable)
    python datasets/dragonball/convert_dragonball_to_parquet.py --lang en --qa-sample 400

qa{lang}.parquet qids are random UUIDs minted at build time; sample files and
past runs reference them, so an existing qa{lang}.parquet is never rebuilt
unless --force is passed.
"""

import argparse
import json
import uuid
from pathlib import Path
from typing import List

import pandas as pd

DATA_DIR = Path(__file__).parent
DOCS_PATH = DATA_DIR / "dragonball_docs.jsonl"
QUERIES_PATH = DATA_DIR / "dragonball_queries.jsonl"


def load_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def filter_lang(records: List[dict], lang: str | None) -> List[dict]:
    if lang is None:
        return records
    return [r for r in records if r.get("language") == lang]


def build_raw_corpus(docs: List[dict]) -> pd.DataFrame:
    """Raw un-chunked corpus: {path, contents, doc_id, domain, language}."""
    return pd.DataFrame([
        {
            "path": f"doc_{d['doc_id']}",
            "contents": d["content"],
            "doc_id": d["doc_id"],
            "domain": d.get("domain", ""),
            "language": d.get("language", ""),
        }
        for d in docs
    ])


def build_qa(queries: List[dict]) -> pd.DataFrame:
    """QA without retrieval_gt (chunk ids don't exist until eval-time chunking)."""
    rows = []
    for q in queries:
        gt = q["ground_truth"]
        rows.append({
            "qid": str(uuid.uuid4()),
            "query": q["query"]["content"],
            "generation_gt": [gt["content"]],
            "question_type": q["query"].get("query_type", ""),
            "references": gt.get("references", []),
            "keypoints": gt.get("keypoints", []),
            "domain": q.get("domain", ""),
            "lang": q.get("language", ""),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert dragonball JSONL to parquet")
    parser.add_argument("--lang", type=str, default=None, help="Filter by language (zh/en). Default: all")
    parser.add_argument("--qa-sample", type=int, default=None, help="Sample N queries from qa{lang}.parquet into qa{lang}_sample_{N}.parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing qa{lang}.parquet (mints NEW qids — invalidates existing sample files and past runs)")
    args = parser.parse_args()

    lang_suffix = f"_{args.lang}" if args.lang else ""
    qa_full_path = DATA_DIR / f"qa{lang_suffix}.parquet"

    # --qa-sample: draw from the full QA parquet so qids stay stable —
    # every sample size remains a subset of the same qa{lang}.parquet.
    # (qa_en_sample_400.parquet == qa_en.parquet.sample(400, random_state=42))
    if args.qa_sample:
        if not qa_full_path.exists():
            raise SystemExit(
                f"{qa_full_path.name} not found — run without --qa-sample first "
                f"to build the full QA, then sample from it."
            )
        qa_df = pd.read_parquet(qa_full_path)
        sampled = qa_df.sample(n=args.qa_sample, random_state=args.seed).reset_index(drop=True)
        out_path = DATA_DIR / f"qa{lang_suffix}_sample_{args.qa_sample}.parquet"
        sampled.to_parquet(out_path)
        print(f"Sampled {len(sampled)}/{len(qa_df)} queries from {qa_full_path.name} (seed={args.seed}) -> {out_path.name}")
        return

    # Raw corpus: deterministic (no UUIDs), safe to overwrite.
    docs = filter_lang(load_jsonl(DOCS_PATH), args.lang)
    raw_df = build_raw_corpus(docs)
    raw_path = DATA_DIR / f"raw_corpus{lang_suffix}.parquet"
    raw_df.to_parquet(raw_path)
    print(f"Saved {raw_path.name} ({len(raw_df)} docs)")

    # Full QA: qids are freshly minted UUIDs, so never silently rebuild.
    if qa_full_path.exists() and not args.force:
        print(f"Skipped {qa_full_path.name} (exists; pass --force to rebuild with NEW qids)")
        return
    queries = filter_lang(load_jsonl(QUERIES_PATH), args.lang)
    qa_df = build_qa(queries)
    qa_df.to_parquet(qa_full_path)
    print(f"Saved {qa_full_path.name} ({len(qa_df)} queries, no retrieval_gt)")


if __name__ == "__main__":
    main()
