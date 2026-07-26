"""Shared dataset preflight for FAISS cost-model benchmarks.

The IVF and HNSW accuracy harnesses use the same `datasets/cm_benchmark`
HDF5 files. Keep download/materialization policy here, at the cost-model layer,
instead of under either concrete FAISS model implementation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import faiss
import numpy as np
from faiss.contrib import datasets as faiss_datasets
from faiss.contrib.vecs_io import fvecs_write, ivecs_write

TEXMEX_BASE = "ftp://ftp.irisa.fr/local/texmex/corpus"
ANNB_BASE = "http://ann-benchmarks.com"
CM_BENCHMARK_DIR = "datasets/cm_benchmark"


FAISS_DATASETS = {
    "sift1m": dict(
        kind="faiss",
        factory=lambda: faiss_datasets.DatasetSIFT1M(),
        d=128,
        subdir="sift1M",
        texmex="sift.tar.gz",
    ),
    "gist1m": dict(
        kind="faiss",
        factory=lambda: faiss_datasets.DatasetGIST1M(),
        d=960,
        subdir="gist1M",
        texmex="gist.tar.gz",
    ),
    "deep1m": dict(
        kind="fvecs",
        annb="deep-image-96-angular",
        d=96,
        subdir="deep1m",
        prefix="deep",
        n=1_000_000,
    ),
    "bigann1m": dict(
        kind="faiss",
        factory=lambda: faiss_datasets.DatasetBigANN(nb_M=1),
        d=128,
        subdir="bigann",
        texmex=None,
    ),
    "music100": dict(
        kind="faiss",
        factory=lambda: faiss_datasets.DatasetMusic100(),
        d=100,
        subdir="music-100",
        texmex=None,
    ),
    "glove": dict(
        kind="hdf5",
        d=100,
        subdir="glove-100-angular",
        embedding="glove",
        annb="glove-100-angular",
        annb_direct=True,
    ),
    "cifar10": dict(
        kind="hdf5",
        d=384,
        subdir="cifar10-dinov2small-384",
        embedding="dinov2_small",
        build_cmd=[
            "rag_stack/cost_model/cm_benchmark_builders/build_cifar10_dinov2.py",
            "--out",
            "{out}",
        ],
    ),
    "dragonball": dict(
        kind="hdf5",
        d=1024,
        subdir="dragonball-bgem3-1024",
        embedding="bge_m3",
        requires=[
            "datasets/dragonball/raw_corpus_en.parquet",
            "datasets/dragonball/qa_en_sample_400.parquet",
        ],
        prepare_cmd=[
            ["datasets/dragonball/convert_dragonball_to_parquet.py", "--lang", "en"],
            [
                "datasets/dragonball/convert_dragonball_to_parquet.py",
                "--lang",
                "en",
                "--qa-sample",
                "400",
            ],
        ],
        build_cmd=[
            "rag_stack/cost_model/cm_benchmark_builders/build_dragonball_bgem3.py",
            "--out",
            "{out}",
        ],
    ),
    "dragonball_en": dict(
        kind="hdf5",
        d=768,
        subdir="dragonball_en-mpnet-768",
        embedding="huggingface_all_mpnet_base_v2",
        requires=[
            "datasets/dragonball/raw_corpus_en.parquet",
            "datasets/dragonball/qa_en_sample_400.parquet",
        ],
        prepare_cmd=[
            ["datasets/dragonball/convert_dragonball_to_parquet.py", "--lang", "en"],
            [
                "datasets/dragonball/convert_dragonball_to_parquet.py",
                "--lang",
                "en",
                "--qa-sample",
                "400",
            ],
        ],
        build_cmd=[
            "rag_stack/cost_model/cm_benchmark_builders/build_mpnet_hdf5.py",
            "--dataset-name",
            "dragonball_en",
            "--corpus",
            "datasets/dragonball/raw_corpus_en.parquet",
            "--qa",
            "datasets/dragonball/qa_en_sample_400.parquet",
            "--chunk-size",
            "128",
            "--out",
            "{out}",
        ],
    ),
    "eli5": dict(
        kind="hdf5",
        d=768,
        subdir="eli5-mpnet-768",
        embedding="huggingface_all_mpnet_base_v2",
        requires=[
            "datasets/eli5/corpus.parquet",
            "datasets/eli5/qa_test.parquet",
        ],
        prepare_cmd=["datasets/eli5/load_eli5_dataset.py"],
        build_cmd=[
            "rag_stack/cost_model/cm_benchmark_builders/build_mpnet_hdf5.py",
            "--dataset-name",
            "eli5",
            "--corpus",
            "datasets/eli5/corpus.parquet",
            "--qa",
            "datasets/eli5/qa_test.parquet",
            "--out",
            "{out}",
        ],
    ),
    "triviaqa": dict(
        kind="hdf5",
        d=768,
        subdir="triviaqa-mpnet-768",
        embedding="huggingface_all_mpnet_base_v2",
        requires=[
            "datasets/triviaqa/corpus.parquet",
            "datasets/triviaqa/qa_test.parquet",
        ],
        prepare_cmd=["datasets/triviaqa/load_triviaqa_dataset.py"],
        build_cmd=[
            "rag_stack/cost_model/cm_benchmark_builders/build_mpnet_hdf5.py",
            "--dataset-name",
            "triviaqa",
            "--corpus",
            "datasets/triviaqa/corpus.parquet",
            "--qa",
            "datasets/triviaqa/qa_test.parquet",
            "--out",
            "{out}",
        ],
    ),
}

DEFAULT_BENCHMARK_DATASETS = [
    ("eli5", "huggingface_all_mpnet_base_v2", "eli5-mpnet-768.hdf5"),
    ("triviaqa", "huggingface_all_mpnet_base_v2", "triviaqa-mpnet-768.hdf5"),
    ("sift1m", "raw", "sift1M.hdf5"),
    ("gist1m", "raw", "gist1M.hdf5"),
    ("deep1m", "raw", "deep1m.hdf5"),
    ("glove", "glove", "glove-100-angular.hdf5"),
]

SIFT1M_DEFAULT_DIR = CM_BENCHMARK_DIR
SIFT1M_URL = f"{TEXMEX_BASE}/sift.tar.gz"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _hdf5_path(spec: dict, data_dir: str, dataset: str) -> str:
    return os.path.join(data_dir, f"{spec.get('subdir', dataset)}.hdf5")


def cm_benchmark_hdf5_path(dataset: str, data_dir: str = CM_BENCHMARK_DIR) -> str:
    """Return the canonical cm_benchmark HDF5 path for a registry dataset key."""
    if dataset not in FAISS_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}; available: {list(FAISS_DATASETS)}"
        )
    return _hdf5_path(FAISS_DATASETS[dataset], data_dir, dataset)


def cm_benchmark_hdf5_file_path(
    hdf5_filename: str,
    data_dir: str = CM_BENCHMARK_DIR,
) -> str:
    """Return a path under cm_benchmark for an explicit HDF5 filename."""
    return os.path.join(data_dir, hdf5_filename)


def cm_benchmark_key_for_hdf5(hdf5_filename: str) -> str:
    """Return the registry key that materializes an explicit HDF5 filename."""
    want = os.path.basename(hdf5_filename)
    for key, spec in FAISS_DATASETS.items():
        got = os.path.basename(_hdf5_path(spec, "", key))
        if got == want:
            return key
    raise ValueError(
        f"Unknown cm_benchmark HDF5 {hdf5_filename!r}; "
        f"available: {[os.path.basename(_hdf5_path(s, '', k)) for k, s in FAISS_DATASETS.items()]}"
    )


def cm_benchmark_dim(dataset: str) -> int:
    """Return the vector dimension for a cm_benchmark registry dataset key."""
    if dataset not in FAISS_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}; available: {list(FAISS_DATASETS)}"
        )
    return int(FAISS_DATASETS[dataset]["d"])


def cm_benchmark_dim_for_hdf5(hdf5_filename: str) -> int:
    """Return the vector dimension for an explicit cm_benchmark HDF5 file."""
    return cm_benchmark_dim(cm_benchmark_key_for_hdf5(hdf5_filename))


def _download_annb_hdf5_direct(spec: dict, out: str) -> str:
    """Download an ann-benchmarks HDF5 file as-is."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{ANNB_BASE}/{spec['annb']}.hdf5"
    print(f"Downloading {spec['annb']}.hdf5 from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    part = out + ".part"
    with urllib.request.urlopen(req) as r, open(part, "wb") as o:
        shutil.copyfileobj(r, o, length=1 << 20)
    os.replace(part, out)
    print(f"Download complete: {out}")
    return out


def _as_cmd_list(cmds) -> list[list[str]]:
    if not cmds:
        return []
    if isinstance(cmds[0], (list, tuple)):
        return [list(cmd) for cmd in cmds]
    return [list(cmds)]


def _missing_required_inputs(spec: dict, root: str) -> list[str]:
    return [
        rel
        for rel in spec.get("requires", [])
        if not os.path.exists(os.path.join(root, rel))
    ]


def _ensure_builder_inputs(spec: dict) -> None:
    root = _repo_root()
    missing = _missing_required_inputs(spec, root)
    if missing and spec.get("prepare_cmd"):
        print(
            f"Preparing source datasets for {spec.get('subdir', '<unknown>')}: {missing}",
            flush=True,
        )
        for cmd_tmpl in _as_cmd_list(spec.get("prepare_cmd")):
            cmd = [sys.executable, os.path.join(root, cmd_tmpl[0])]
            cmd.extend(cmd_tmpl[1:])
            subprocess.run(cmd, cwd=root, check=True)
        missing = _missing_required_inputs(spec, root)
    if missing:
        raise FileNotFoundError(
            "Missing source files for cm_benchmark builder: " + ", ".join(missing)
        )


def _run_hdf5_builder(spec: dict, out: str) -> str:
    cmd_tmpl = spec.get("build_cmd")
    if not cmd_tmpl:
        raise ValueError(
            f"{spec.get('subdir', '<unknown>')}: no download source or builder"
        )
    _ensure_builder_inputs(spec)
    root = _repo_root()
    cmd = [sys.executable, os.path.join(root, cmd_tmpl[0])]
    cmd.extend(arg.format(out=out) for arg in cmd_tmpl[1:])
    print(f"Building {out} via {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=root, check=True)
    if not os.path.exists(out):
        raise FileNotFoundError(f"builder completed but did not create {out}")
    return out


def ensure_texmex(
    subdir: str,
    tarball: str,
    data_dir: str,
    download_url: str | None = None,
) -> None:
    """Download and extract a TEXMEX tarball into ``<data_dir>/<subdir>/``."""
    dest = os.path.join(data_dir, subdir)
    if os.path.isdir(dest) and any(f.endswith(".fvecs") for f in os.listdir(dest)):
        return

    os.makedirs(dest, exist_ok=True)
    url = download_url or f"{TEXMEX_BASE}/{tarball}"
    tar_path = os.path.join(dest, tarball)

    if not os.path.exists(tar_path):
        print(f"Downloading {tarball} from {url} ...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")

    print(f"Extracting {tarball}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                member.name = os.path.basename(member.name)
                tar.extract(member, dest)
    os.remove(tar_path)
    print("Extraction complete.")


def ensure_annb_fvecs(spec: dict, data_dir: str) -> str:
    """Materialize an ann-benchmarks dataset as fvecs for legacy callers."""
    dest = os.path.join(data_dir, spec["subdir"])
    base = os.path.join(dest, spec["prefix"])
    if os.path.exists(base + "_base.fvecs"):
        return base
    os.makedirs(dest, exist_ok=True)

    url = f"{ANNB_BASE}/{spec['annb']}.hdf5"
    h5_path = os.path.join(dest, f"{spec['annb']}.hdf5")
    if not os.path.exists(h5_path):
        print(f"Downloading {spec['annb']}.hdf5 from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        tmp = h5_path + ".part"
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out, length=1 << 20)
        os.replace(tmp, h5_path)
        print("Download complete.")

    print(
        f"Converting {spec['annb']} to fvecs "
        f"(subset {spec['n']:,}, recomputing L2 GT)..."
    )
    import h5py

    with h5py.File(h5_path, "r") as f:
        n = min(spec["n"] or f["train"].shape[0], f["train"].shape[0])
        xb = np.ascontiguousarray(f["train"][:n], dtype="float32")
        xq = np.ascontiguousarray(f["test"][:], dtype="float32")
    faiss.omp_set_num_threads(mp.cpu_count())
    flat = faiss.IndexFlatL2(xb.shape[1])
    flat.add(xb)
    _, gt = flat.search(xq, 100)

    fvecs_write(base + "_base.fvecs", xb)
    fvecs_write(base + "_query.fvecs", xq)
    fvecs_write(base + "_learn.fvecs", xb[: min(n, 200_000)])
    ivecs_write(base + "_groundtruth.ivecs", gt.astype("int32"))
    os.remove(h5_path)
    print("Conversion complete.")
    return base


def ensure_cm_benchmark_dataset(
    dataset: str,
    data_dir: str = CM_BENCHMARK_DIR,
    download_url: str | None = None,
) -> str:
    """Ensure one cm_benchmark dataset exists as a unified HDF5 file."""
    if dataset not in FAISS_DATASETS:
        raise ValueError(
            f"Unknown dataset {dataset!r}; available: {list(FAISS_DATASETS)}"
        )
    spec = FAISS_DATASETS[dataset]
    out = _hdf5_path(spec, data_dir, dataset)
    if os.path.exists(out):
        return out
    if spec.get("annb_direct"):
        return _download_annb_hdf5_direct(spec, out)
    if spec.get("build_cmd"):
        return _run_hdf5_builder(spec, out)
    if spec.get("annb") or spec.get("texmex"):
        return ensure_hdf5(spec, data_dir, dataset, download_url)
    raise ValueError(f"{dataset}: no auto-download or builder is configured")


def ensure_cm_benchmark_hdf5(
    hdf5_filename: str,
    data_dir: str = CM_BENCHMARK_DIR,
    download_url: str | None = None,
) -> str:
    """Ensure one cm_benchmark dataset exists, addressed by HDF5 filename."""
    key = cm_benchmark_key_for_hdf5(hdf5_filename)
    return ensure_cm_benchmark_dataset(key, data_dir, download_url)


def ensure_cm_benchmark_datasets(
    datasets: list[str] | tuple[str, ...],
    data_dir: str = CM_BENCHMARK_DIR,
    download_url: str | None = None,
) -> dict[str, str]:
    """Preflight a set of cm_benchmark datasets before running a benchmark."""
    paths = {}
    for dataset in dict.fromkeys(datasets):
        paths[dataset] = ensure_cm_benchmark_dataset(dataset, data_dir, download_url)
    return paths


def ensure_cm_benchmark_hdf5_files(
    hdf5_filenames: list[str] | tuple[str, ...],
    data_dir: str = CM_BENCHMARK_DIR,
    download_url: str | None = None,
) -> dict[str, str]:
    """Preflight a set of cm_benchmark datasets by explicit HDF5 filename."""
    paths = {}
    for filename in dict.fromkeys(hdf5_filenames):
        paths[filename] = ensure_cm_benchmark_hdf5(filename, data_dir, download_url)
    return paths


def ensure_hdf5(
    spec: dict,
    data_dir: str,
    dataset: str,
    download_url: str | None = None,
) -> str:
    """Acquire and materialize a dataset as the unified flat HDF5 layout."""
    import h5py

    out = _hdf5_path(spec, data_dir, dataset)
    if os.path.exists(out):
        return out
    os.makedirs(data_dir, exist_ok=True)

    if spec.get("annb"):
        url = f"{ANNB_BASE}/{spec['annb']}.hdf5"
        src = out + ".annb-src"
        if not os.path.exists(src):
            print(f"Downloading {spec['annb']}.hdf5 ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            part = src + ".part"
            with urllib.request.urlopen(req) as r, open(part, "wb") as o:
                shutil.copyfileobj(r, o, length=1 << 20)
            os.replace(part, src)
        print(
            f"Materializing {dataset} hdf5 "
            f"(subset {spec.get('n') or 'all'}, recomputing L2 GT)..."
        )
        with h5py.File(src, "r") as f:
            n = min(spec.get("n") or f["train"].shape[0], f["train"].shape[0])
            xb = np.ascontiguousarray(f["train"][:n], dtype="float32")
            xq = np.ascontiguousarray(f["test"][:], dtype="float32")
        faiss.omp_set_num_threads(mp.cpu_count())
        flat = faiss.IndexFlatL2(xb.shape[1])
        flat.add(xb)
        _, gt = flat.search(xq, 100)
        with h5py.File(out, "w") as f:
            f.create_dataset("train", data=xb)
            f.create_dataset("test", data=xq)
            f.create_dataset("learn", data=xb[: min(n, 200_000)])
            f.create_dataset("neighbors", data=gt.astype("int32"))
            f.attrs["dataset_name"] = spec.get("subdir", dataset)
        os.remove(src)
    elif spec.get("texmex"):
        faiss_datasets.set_dataset_basedir(data_dir.rstrip("/") + "/")
        ensure_texmex(spec.get("subdir", dataset), spec["texmex"], data_dir, download_url)
        print(f"Materializing {dataset} hdf5 from TEXMEX fvecs ...")
        ds = spec["factory"]()
        with h5py.File(out, "w") as f:
            f.create_dataset("train", data=np.ascontiguousarray(ds.get_database()))
            f.create_dataset("test", data=np.ascontiguousarray(ds.get_queries()))
            f.create_dataset("learn", data=np.ascontiguousarray(ds.get_train()))
            f.create_dataset("neighbors", data=np.ascontiguousarray(ds.get_groundtruth()))
            f.attrs["dataset_name"] = spec.get("subdir", dataset)
        shutil.rmtree(os.path.join(data_dir, spec.get("subdir", dataset)), ignore_errors=True)
    else:
        raise ValueError(f"{dataset}: no annb/texmex source; cannot auto-build hdf5")
    print(f"Materialized {out}")
    return out
