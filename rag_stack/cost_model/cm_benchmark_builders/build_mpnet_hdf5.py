"""Build an mpnet (all-mpnet-base-v2, 768-d) ANN dataset for the FAISS cost-model
benchmark — generic over (dataset_name, corpus, qa, chunk_size).

Mirrors build_dragonball_bgem3.py but uses the SAME embedding the e2e benchmark
uses (`huggingface_all_mpnet_base_v2` → sentence-transformers/all-mpnet-base-v2),
so the per-dataset workload profiles it produces (IVF cell-imbalance,
HNSW ndis/nhops predictor) are keyed `<dataset_name>__huggingface_all_mpnet_base_v2`
— EXACTLY what the e2e CM looks up. `train` = corpus-chunk base vectors,
`test` = QA query vectors. ann-benchmarks hdf5 layout (no GT — latency benchmark).

    python -m rag_stack.cost_model.cm_benchmark_builders.build_mpnet_hdf5 \
        --dataset-name dragonball_en --corpus datasets/dragonball/raw_corpus_en.parquet \
        --qa datasets/dragonball/qa_en_sample_400.parquet --chunk-size 128 \
        --out datasets/cm_benchmark/dragonball_en-mpnet-768.hdf5
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-mpnet-base-v2"   # = rag-stack alias huggingface_all_mpnet_base_v2
EMB_LABEL = "huggingface_all_mpnet_base_v2"          # the profile-key embedding string (matches e2e)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", required=True, help="e2e dataset_name (profile key)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--chunk-overlap", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    corpus = pd.read_parquet(args.corpus)
    qa = pd.read_parquet(args.qa)

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size,
                                              chunk_overlap=args.chunk_overlap)
    chunks = []
    for txt in corpus["contents"].astype(str):
        chunks.extend(splitter.split_text(txt))
    print(f"[{args.dataset_name}] chunked {len(corpus)} docs → {len(chunks)} chunks "
          f"(chunk_size={args.chunk_size})", flush=True)

    print(f"loading {MODEL} on {DEV} ...", flush=True)
    model = SentenceTransformer(MODEL, device=DEV)

    print("embedding corpus chunks (base) ...", flush=True)
    base = model.encode(chunks, batch_size=128, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
    print("embedding QA queries (test) ...", flush=True)
    test = model.encode(qa["query"].astype(str).tolist(), batch_size=128,
                        normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("train", data=base)
        f.create_dataset("test", data=test)
        f.attrs["embedding_model"] = EMB_LABEL
        f.attrs["dataset_name"] = args.dataset_name
        f.attrs["dim"] = base.shape[1]
    print(f"wrote {args.out}: train {base.shape}, test {test.shape}", flush=True)


if __name__ == "__main__":
    main()
