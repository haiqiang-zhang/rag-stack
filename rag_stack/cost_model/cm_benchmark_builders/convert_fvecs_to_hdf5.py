"""Convert the legacy .fvecs cm_benchmark datasets (sift1M / gist1M / deep1m) to
the unified flat hdf5 layout (ann-benchmarks: train=base, test=query, learn,
neighbors=groundtruth) so the WHOLE datasets/cm_benchmark/ is one format — the
same `train`/`test` hdf5 as glove/cifar/dragonball. The IVF + HNSW benchmarks
prefer these hdf5 (FaissBenchmarkRunner falls back to the fvecs auto-download
only if the hdf5 is absent).

    conda run -n rag-stack python -m rag_stack.cost_model.cm_benchmark_builders.convert_fvecs_to_hdf5
"""
import os

import h5py
from faiss.contrib.vecs_io import fvecs_read, ivecs_read

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA_DIR = os.path.join(ROOT, "datasets/cm_benchmark")
SETS = [("sift1M", "sift"), ("gist1M", "gist"), ("deep1m", "deep")]


def main():
    for subdir, prefix in SETS:
        base = os.path.join(DATA_DIR, subdir, prefix)
        if not os.path.exists(base + "_base.fvecs"):
            print(f"skip {subdir}: no {prefix}_base.fvecs")
            continue
        out = os.path.join(DATA_DIR, f"{subdir}.hdf5")
        with h5py.File(out, "w") as f:
            f.create_dataset("train", data=fvecs_read(base + "_base.fvecs"))
            f.create_dataset("test", data=fvecs_read(base + "_query.fvecs"))
            if os.path.exists(base + "_learn.fvecs"):
                f.create_dataset("learn", data=fvecs_read(base + "_learn.fvecs"))
            if os.path.exists(base + "_groundtruth.ivecs"):
                f.create_dataset("neighbors", data=ivecs_read(base + "_groundtruth.ivecs"))
            f.attrs["dataset_name"] = subdir
        with h5py.File(out, "r") as f:
            print(f"wrote {out}: train {f['train'].shape}, test {f['test'].shape}"
                  + (f", neighbors {f['neighbors'].shape}" if "neighbors" in f else ""))


if __name__ == "__main__":
    main()
