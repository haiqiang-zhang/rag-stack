"""Build the dragonball + bge-m3 ANN dataset for the cost-model benchmark.

Chunks the dragonball corpus (raw_corpus_en.parquet, the REAL production corpus,
108 docs / 963k chars) with recursive character splitting and embeds with
BAAI/bge-m3 (1024-d dense, normalized). `train` = corpus-chunk base vectors,
`test` = the 400 QA query vectors. ann-benchmarks hdf5 layout, mirroring
glove-100-angular.hdf5.

CHUNK_SIZE=128 (was 512) yields ~10.4k chunks (vs 2731 at 512), pushing the
corpus OUT of the degenerate sub-L3 IVF regime: the flat index is then ~42MB >
L3 (16.8MB) so the scan is DRAM-bound — the regime the cost model is calibrated
for — instead of a fully cache-resident near-full-scan. Smaller chunks are real,
distinct passages (not duplicates); for a latency benchmark only the count +
dimensionality matter.

    conda run -n rag-stack python -m rag_stack.cost_model.cm_benchmark_builders.build_dragonball_bgem3
    → datasets/cm_benchmark/dragonball-bgem3-1024.hdf5
"""
import argparse
import os
import h5py
import numpy as np
import pandas as pd
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(ROOT, "datasets/cm_benchmark/dragonball-bgem3-1024.hdf5")
MODEL = "BAAI/bge-m3"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE, CHUNK_OVERLAP = 128, 16   # ~10.4k chunks (>10k); see module docstring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    corpus = pd.read_parquet(os.path.join(ROOT, "datasets/dragonball/raw_corpus_en.parquet"))
    qa = pd.read_parquet(os.path.join(ROOT, "datasets/dragonball/qa_en_sample_400.parquet"))

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for txt in corpus["contents"].astype(str):
        chunks.extend(splitter.split_text(txt))
    print(f"chunked {len(corpus)} docs → {len(chunks)} chunks", flush=True)

    print(f"loading {MODEL} on {DEV} ...", flush=True)
    model = SentenceTransformer(MODEL, device=DEV)

    print("embedding corpus chunks (base) ...", flush=True)
    base = model.encode(chunks, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
    print("embedding QA queries (test) ...", flush=True)
    test = model.encode(qa["query"].astype(str).tolist(), batch_size=64,
                       normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("train", data=base)
        f.create_dataset("test", data=test)
        f.attrs["embedding_model"] = "bge_m3"
        f.attrs["dataset_name"] = "dragonball_en"
        f.attrs["dim"] = base.shape[1]
    print(f"wrote {args.out}: train {base.shape}, test {test.shape}", flush=True)


if __name__ == "__main__":
    main()
