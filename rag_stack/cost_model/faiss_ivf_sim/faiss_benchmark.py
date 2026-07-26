from .types import BenchmarkResult
from .imbalance import cell_imbalance_np1, compute_cell_imbalance

import faiss
from faiss.contrib.vecs_io import fvecs_read, ivecs_read
from faiss.contrib import datasets as faiss_datasets
import numpy as np

import ctypes
import ctypes.util
import glob as globmod
import multiprocessing as mp
import os
import platform
import sys
import time

from rag_stack.cost_model.cm_benchmark_datasets import (
    CM_BENCHMARK_DIR,
    FAISS_DATASETS,
    SIFT1M_DEFAULT_DIR,
    SIFT1M_URL,
    cm_benchmark_hdf5_path,
    ensure_annb_fvecs,
    ensure_cm_benchmark_dataset,
    ensure_texmex,
)


def _detect_faiss_blas_backend():
    """Inspect the FAISS native extension to determine which BLAS it links against.

    Returns "OpenBLAS", "MKL", "Accelerate", or None.
    """
    import importlib
    import subprocess

    # Find the FAISS native .so / .dylib
    so_path = None
    for mod_name in ("faiss._swigfaiss_avx2", "faiss._swigfaiss_avx512", "faiss._swigfaiss"):
        try:
            mod = importlib.import_module(mod_name)
            p = getattr(mod, "__file__", None)
            if p and os.path.exists(p):
                so_path = p
                break
        except ImportError:
            continue

    is_mac = platform.system() == "Darwin"

    # Linux: check /proc/pid/maps for loaded libraries
    if not is_mac:
        try:
            with open(f"/proc/{os.getpid()}/maps") as f:
                maps = f.read()
            if "libmkl" in maps:
                return "MKL"
            if "libopenblas" in maps:
                return "OpenBLAS"
        except Exception:
            pass

    # Inspect the .so/.dylib dependencies
    if so_path:
        try:
            cmd = ["otool", "-L", so_path] if is_mac else ["ldd", so_path]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
            out_lower = out.lower()
            if "libmkl" in out_lower or "mkl" in out_lower:
                return "MKL"
            if "libopenblas" in out_lower or "openblas" in out_lower:
                return "OpenBLAS"
            if "Accelerate" in out or "vecLib" in out:
                return "Accelerate"
        except Exception:
            pass

    return None


def _load_openblas():
    """Try to load the OpenBLAS library, searching common install locations."""
    is_mac = platform.system() == "Darwin"
    so_ext = ".dylib" if is_mac else ".so"
    candidates = []

    found = ctypes.util.find_library("openblas")
    if found:
        candidates.append(found)

    prefix = sys.prefix
    ext_glob = "libopenblas*.dylib" if is_mac else "libopenblas*.so*"
    candidates += globmod.glob(os.path.join(prefix, "lib", ext_glob))

    # numpy.libs bundled OpenBLAS
    try:
        np_dir = os.path.dirname(np.__file__)
        for d in [os.path.join(np_dir, ".libs"), os.path.join(np_dir, ".dylibs")]:
            candidates += globmod.glob(os.path.join(d, ext_glob))
        sp_np_libs = os.path.join(os.path.dirname(np_dir), "numpy.libs")
        candidates += globmod.glob(os.path.join(sp_np_libs, ext_glob))
    except Exception:
        pass

    if is_mac:
        for d in ["/usr/local/lib", "/opt/homebrew/lib",
                   "/usr/local/opt/openblas/lib", "/opt/homebrew/opt/openblas/lib"]:
            candidates += globmod.glob(os.path.join(d, "libopenblas*.dylib"))
    else:
        for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
                   "/usr/lib64", "/usr/lib", "/usr/local/lib", "/usr/local/lib64"]:
            candidates += globmod.glob(os.path.join(d, "libopenblas*.so*"))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and conda_prefix != prefix:
        candidates += globmod.glob(os.path.join(conda_prefix, "lib", ext_glob))

    candidates.append(f"libopenblas{so_ext}")

    for path in dict.fromkeys(candidates):
        try:
            lib = ctypes.CDLL(path)
            lib.openblas_get_config.restype = ctypes.c_char_p
            lib.openblas_get_config()  # verify
            return lib
        except (OSError, AttributeError):
            continue
    return None


def _load_mkl():
    """Try to load the MKL runtime library."""
    is_mac = platform.system() == "Darwin"
    so_ext = ".dylib" if is_mac else ".so"
    candidates = []

    found = ctypes.util.find_library("mkl_rt")
    if found:
        candidates.append(found)

    prefix = sys.prefix
    candidates += globmod.glob(os.path.join(prefix, "lib", f"libmkl_rt{so_ext}*"))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and conda_prefix != prefix:
        candidates += globmod.glob(os.path.join(conda_prefix, "lib", f"libmkl_rt{so_ext}*"))
    candidates.append(f"libmkl_rt{so_ext}")

    for path in dict.fromkeys(candidates):
        try:
            lib = ctypes.CDLL(path)
            lib.mkl_get_max_threads.restype = ctypes.c_int
            lib.mkl_get_max_threads()  # verify
            return lib
        except (OSError, AttributeError):
            continue
    return None


def _classify_openblas(lib) -> str:
    """Determine OpenBLAS threading model: 'OpenBLAS_openmp' or 'OpenBLAS_pthreads'."""
    config = lib.openblas_get_config().decode()
    if "USE_OPENMP=1" in config or " USE_OPENMP" in config:
        return "OpenBLAS_openmp"
    return "OpenBLAS_pthreads"


def _find_blas_lib():
    """Detect which BLAS FAISS actually links, then load that library.

    Step 1: Inspect the FAISS .so/.dylib to determine the backend.
    Step 2: Load only the matching library.
    Fallback: If inspection fails, try both and pick whichever loads.

    Returns (backend_name, ctypes_handle) where backend_name is one of:
      "OpenBLAS_pthreads" - independent thread pool, must set explicitly
      "OpenBLAS_openmp"   - shares OMP runtime, faiss.omp_set_num_threads controls it
      "MKL"               - global cap, MKL_DYNAMIC auto-adjusts in nested regions
      "Accelerate"        - GCD-managed, no user control
      None                - detection failed
    """
    detected = _detect_faiss_blas_backend()

    if detected == "OpenBLAS":
        lib = _load_openblas()
        if lib:
            return (_classify_openblas(lib), lib)
    elif detected == "MKL":
        lib = _load_mkl()
        if lib:
            return ("MKL", lib)
    elif detected == "Accelerate":
        return ("Accelerate", None)

    # Inspection inconclusive — try both as fallback
    if detected is None:
        lib = _load_openblas()
        if lib:
            return (_classify_openblas(lib), lib)
        lib = _load_mkl()
        if lib:
            return ("MKL", lib)

    return (None, None)


# Module-level cache so we only probe once.
_blas_backend, _blas_lib = _find_blas_lib()


def _set_blas_num_threads(n: int) -> None:
    """Pin BLAS threads to avoid oversubscription with FAISS OMP threads.

    Backend-specific behavior:

    OpenBLAS_pthreads:
        Independent global thread pool. Multiple OMP threads calling BLAS
        simultaneously ALL contend on this single pool, causing catastrophic
        lock contention (measured: 12 threads 6.5x SLOWER than 1 thread).
        Fix: always set pool=1 so each OMP thread runs single-threaded BLAS.
        Total parallelism = OMP threads (no BLAS-level parallelism).

    OpenBLAS_openmp:
        Shares OpenMP runtime with FAISS. Nested parallelism is off by
        default, so BLAS inside OMP parallel region auto-degrades to 1 thread.
        No-op: faiss.omp_set_num_threads() already controls everything.

    MKL:
        MKL_Set_Num_Threads(n) sets global upper bound. MKL_DYNAMIC (default
        on) auto-detects nested OMP regions and drops to single-threaded.
        Safe to set to n — MKL handles nesting intelligently.

    Accelerate:
        Thread count managed by GCD. No user-facing API.
    """
    if _blas_backend == "OpenBLAS_pthreads" and _blas_lib is not None:
        # MUST be 1: pthreads pool contention is catastrophic with concurrent callers
        _blas_lib.openblas_set_num_threads(1)
    elif _blas_backend == "OpenBLAS_openmp":
        pass  # Already controlled by faiss.omp_set_num_threads()
    elif _blas_backend == "MKL" and _blas_lib is not None:
        _blas_lib.MKL_Set_Num_Threads(ctypes.c_int(n))
    elif _blas_backend == "Accelerate":
        pass  # GCD-managed


class FaissBenchmarkRunner:
    """Runs actual FAISS benchmarks on SIFT1M dataset."""

    # Cache flush array size (must be larger than all cache levels combined)
    # M3 Pro: L2 ~36MB + SLC ~12MB = ~48MB, use 64MB to be safe
    CACHE_FLUSH_SIZE_MB = 64

    def __init__(
        self,
        num_threads: int = 1,
        dataset: str = "sift1m",
        data_dir: str = CM_BENCHMARK_DIR,
        download_url: str | None = None,
    ):
        """Load a FAISS-recommended dataset (faiss.contrib.datasets) for benchmarking.

        ``dataset`` is a key of :data:`FAISS_DATASETS` (e.g. ``"sift1m"`` d=128,
        ``"gist1m"`` d=960). SIFT1M/GIST1M auto-download from the TEXMEX corpus;
        others must be staged under ``<data_dir>/<subdir>/``. All carry ground truth.
        """
        if dataset not in FAISS_DATASETS:
            raise ValueError(
                f"Unknown dataset {dataset!r}; available: {list(FAISS_DATASETS)}")
        spec = FAISS_DATASETS[dataset]

        self.dataset_name = dataset
        self.num_threads = num_threads
        self._cache_flush_array = None

        # Prefer the unified flat hdf5 (ann-benchmarks layout: train/test[/learn/
        # neighbors]) — the project standard (glove/cifar/dragonball + converted
        # sift/gist/deep). Falls back to the fvecs/faiss auto-download below.
        h5 = cm_benchmark_hdf5_path(dataset, data_dir)
        if not os.path.exists(h5):
            # Auto-download / generate the standard cm_benchmark hdf5 before
            # falling through to the legacy fvecs/faiss loaders.
            try:
                h5 = ensure_cm_benchmark_dataset(dataset, data_dir, download_url)
            except ValueError:
                pass
        if os.path.exists(h5):
            import h5py
            print(f"Loading {dataset} dataset (hdf5: {h5})...")
            with h5py.File(h5, "r") as f:
                self.xb = np.asarray(f["train"][:], dtype=np.float32)
                self.xq = np.asarray(f["test"][:], dtype=np.float32)
                self.xt = (np.asarray(f["learn"][:], dtype=np.float32)
                           if "learn" in f else self.xb[:200_000])
                self.gt = np.asarray(f["neighbors"][:]) if "neighbors" in f else None
        elif spec["kind"] == "faiss":
            # Point faiss.contrib.datasets at our dir, auto-download TEXMEX (SIFT/
            # GIST) into the subdir the Dataset class expects, then read via faiss.
            faiss_datasets.set_dataset_basedir(data_dir.rstrip("/") + "/")
            if spec.get("texmex"):
                ensure_texmex(spec["subdir"], spec["texmex"], data_dir, download_url)
            print(f"Loading {dataset} dataset (faiss.contrib.datasets)...")
            ds = spec["factory"]()
            self.xb = ds.get_database()
            self.xq = ds.get_queries()
            self.xt = ds.get_train()
            self.gt = ds.get_groundtruth()
        elif spec["kind"] == "fvecs":
            # Plain fvecs (ann-benchmarks datasets like deep1m are fetched+converted
            # ONCE by ensure_annb_fvecs, then read here exactly like sift/gist).
            base = ensure_annb_fvecs(spec, data_dir)
            print(f"Loading {dataset} dataset (fvecs)...")
            self.xb = fvecs_read(base + "_base.fvecs")
            self.xq = fvecs_read(base + "_query.fvecs")
            self.xt = fvecs_read(base + "_learn.fvecs")
            self.gt = ivecs_read(base + "_groundtruth.ivecs")
        else:
            raise ValueError(f"Unknown dataset kind {spec['kind']!r}")

        self.nb, self.d = self.xb.shape
        self.nq, _ = self.xq.shape

        print(f"  Database: {self.nb:,} vectors, {self.d} dimensions")
        print(f"  Queries: {self.nq:,} vectors")

    def _flush_cache(self):
        """
        Flush CPU cache by reading through a large array.

        This evicts FAISS data structures from cache, simulating cold-start
        behavior in RAG systems where queries are sparse.
        """
        if self._cache_flush_array is None:
            # Lazily allocate the flush array
            size = self.CACHE_FLUSH_SIZE_MB * 1024 * 1024 // 4  # float32
            self._cache_flush_array = np.random.randn(size).astype(np.float32)

        # Read through the array to evict cache (use sum to prevent optimization)
        _ = np.sum(self._cache_flush_array)

    def get_or_train_index(self, index_name: str, cache_dir: str = "indices") -> faiss.Index:
        """Load index from cache or train a new one."""
        os.makedirs(cache_dir, exist_ok=True)
        # Include dataset+d+N in the path: the SAME factory string (e.g.
        # "IVF1024,PQ32x4fs") builds a DIFFERENT index per dimension/dataset.
        index_path = os.path.join(
            cache_dir, f"{self.dataset_name}_n{self.nb}_d{self.d}_{index_name}.faissindex")

        if os.path.exists(index_path):
            print(f"  Loading cached index from {index_path}")
            return faiss.read_index(index_path)

        print(f"  Training index {index_name}...")
        ncpu = mp.cpu_count()
        faiss.omp_set_num_threads(ncpu)
        _set_blas_num_threads(ncpu)
        index = faiss.index_factory(self.d, index_name)
        index.train(self.xt)
        index.add(self.xb)
        faiss.write_index(index, index_path)
        return index

    def cell_imbalance(self, index_name: str) -> float:
        """nprobe=1 IVF cell-imbalance constant for ``index_name`` (cached index).

        Thin wrapper over :func:`imbalance.cell_imbalance_np1`; see that module
        for the data-property rationale. Holds the parent index alive (SWIG).
        """
        index = self.get_or_train_index(index_name)
        return cell_imbalance_np1(index)

    def cell_imbalance_by_nprobe(
        self, index_name: str, nprobe_values: list[int], n_query_sample: int = 2000,
    ) -> dict[int, float]:
        """Per-nprobe IVF imbalance for ``index_name`` on this dataset's queries.

        Thin wrapper over :func:`imbalance.compute_cell_imbalance` — the single
        source of truth shared with the (future) production path. Uses the
        cached index and the dataset's own query set as the assignment sample.
        """
        index = self.get_or_train_index(index_name)
        return compute_cell_imbalance(index, self.xq, nprobe_values, n_query_sample)

    def benchmark_index(
        self,
        index_name: str,
        nprobe_values: list[int],
        k: int = 1,
        warmup: bool = True,
        batch_size: int = 1,
        parallel_mode: int = 0,
    ) -> list[BenchmarkResult]:
        """Run benchmark for an index across multiple nprobe values.

        Args:
            index_name: FAISS index factory string.
            nprobe_values: List of nprobe values to test.
            k: Number of nearest neighbors to return.
            warmup: Whether to run warmup iterations. If False, cache is explicitly
                    flushed before each search to simulate cold-start RAG latency.
            batch_size: Number of queries per batch. When batch_size > 1, FAISS
                        uses BLAS-optimized matrix multiplication for coarse
                        quantization (when batch >= 20). The benchmark reports
                        per-batch latency and throughput.
            parallel_mode: FAISS parallel_mode for IVF search.
                           0=inter-query (default), 1=intra-query.
        """
        index = self.get_or_train_index(index_name)
        faiss.omp_set_num_threads(self.num_threads)
        _set_blas_num_threads(self.num_threads)
        index.parallel_mode = parallel_mode  # type: ignore[union-attr]

        results = []

        assert batch_size <= self.nq, "Batch size must be smaller or equal to total number of queries for benchmarking"

        if warmup:
            # Global warmup: iterate over all nprobe values to bring index data
            # into cache. This models steady-state where cache contains mixed data
            # from various nprobe settings (not artificially hot for one nprobe).
            print("  Warming up (warm cache mode)...")
            warmup_queries = self.xq[:batch_size]
            for _ in range(20):
                for nprobe_val in nprobe_values:
                    index.nprobe = nprobe_val  # type: ignore[union-attr]
                    index.search(warmup_queries, k=k)  # type: ignore[call-arg]
        else:
            # Cold cache mode - pre-allocate flush array
            print(f"  Cold cache mode (flushing {self.CACHE_FLUSH_SIZE_MB}MB before each search)...")
            self._flush_cache()  # Pre-allocate

        print(f"  Benchmarking {index_name} (batch_size={batch_size}, parallel_mode={parallel_mode}, warmup={warmup})...")

        # Determine how many full batches we can run
        num_batches = self.nq // batch_size
        # Multiple measurement rounds for stability (take median to remove outliers)
        num_rounds = 3

        for nprobe in nprobe_values:
            index.nprobe = nprobe  # type: ignore[union-attr]

            round_latencies = []
            all_I = None

            for round_idx in range(num_rounds):
                total_elapsed = 0.0
                round_I = []

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    batch_queries = self.xq[start_idx:end_idx]

                    if not warmup:
                        self._flush_cache()

                    t0 = time.perf_counter()
                    D, I = index.search(batch_queries, k=k)  # type: ignore[call-arg]
                    total_elapsed += time.perf_counter() - t0
                    round_I.append(I)

                latency_ms = total_elapsed * 1000 / num_batches
                round_latencies.append(latency_ms)

                if round_idx == num_rounds - 1:
                    all_I = np.vstack(round_I)

            total_queries = num_batches * batch_size

            # Median of rounds for stability (removes load spike outliers)
            latency_ms = float(np.median(round_latencies))
            qps = batch_size / (latency_ms / 1000)
            # Recall needs ground truth (SIFT only); synthetic data → NaN.
            recall = (
                (self.gt[:total_queries, :k] == all_I).sum() / (total_queries * k)
                if self.gt is not None else float("nan")
            )

            results.append(BenchmarkResult(
                index_name=index_name,
                nprobe=nprobe,
                recall_at_k=recall,
                latency_ms=latency_ms,
                qps=qps,
                num_queries=total_queries,
                batch_size=batch_size,
            ))

            print(f"    nprobe={nprobe:3d}: recall@{k}={recall:.4f}, "
                  f"latency={latency_ms:.4f}ms/batch, QPS={qps:,.0f}")

        return results
