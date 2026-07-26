---
title: FAISS IVF Cost Model
parent: Cost Model (RAG-CM)
nav_order: 4
---

# FAISS IVF Cost Model

Analytical roofline-based cost model for FAISS IVF vector search — one unified
`faiss_ivf` family covering IVF-PQ and IVF-Flat (selected per trial by the
`index_type` search dim). Models each pipeline stage (coarse quantizer, LUT
build, PQ/flat scan, heap selection, refinement) with per-operator compute and
memory costs.

## Quick Usage

```python
from rag_stack.cost_model.faiss_ivf_sim.ivfpq_config import IVFPQConfig
from rag_stack.cost_model.faiss_ivf_sim.cost_model import IVFPQCostModel
from rag_stack.cost_model.faiss_ivf_sim.coefficients import load_coefficients, build_hardware_from_coefficients

# Load calibrated hardware profile
coefficients = load_coefficients("calibration_profiles/faiss_ivf/epyc_9124.json")
hw = build_hardware_from_coefficients(coefficients, active_threads=1)

# IVF-PQ: compressed codes scanned via a per-query distance LUT
cfg = IVFPQConfig(d=128, N=1_000_000, nlist=1024, nprobe=16, M=32, nbits=4,
                  k=10, batch_size=1, parallel_mode=0)
result = IVFPQCostModel(cfg, hw, warm_cache=True).estimate()
print(f"Latency: {result.latency_ms:.4f} ms, QPS: {result.throughput_qps:.0f}")

# IVF-Flat: exact full-vector scan — pass index_type="flat" and drop the PQ
# knobs (M / nbits / refinement / precomputed table are ignored / forced off)
flat_cfg = IVFPQConfig(d=128, N=1_000_000, index_type="flat",
                       nlist=1024, nprobe=16, k=10, batch_size=1, parallel_mode=0)
flat_result = IVFPQCostModel(flat_cfg, hw, warm_cache=True).estimate()
```

Key parameters:

- `parallel_mode`: only `0` (FAISS inter-query mode) is currently supported.
  Mode 1 fails closed because the frozen standalone calibration grid does not
  measure its cache, synchronization, or scheduling behavior.
- `warm_cache`: `True` for throughput benchmarks (data stays in cache), `False` for cold-start RAG latency modeling
- `index_type`: `"pq"` (default) or `"flat"` — one config class covers both;
  flat prices exact full-vector scans (no LUT, no precomputed table, no
  refinement), with `flat_dtype_bytes` selecting the stored precision
  (4 = FP32 default; 2 = SQ16, 1 = SQ8)

## CLI — simulator benchmark

The module CLI provides the `benchmark` command. See
[Calibration](#calibration) for fitting coefficients:

```bash
python -m rag_stack.cost_model.faiss_ivf_sim benchmark [options]
```

It compares simulator predictions against actual FAISS results. For each
nprobe value, prints actual vs predicted latency with error percentage and a
per-operator breakdown (compute time, memory time, bottleneck):

```bash
python -m rag_stack.cost_model.faiss_ivf_sim benchmark \
    --hardware epyc_9124 \
    --index-configs "IVF1024,PQ32x4fs" \
    --nprobe-values 1 4 16 64 \
    --num-threads 0 \
    --batch-sizes 1 1000 \
    --parallel-modes 0
```

Options:

- `--hardware`: Exact CPU profile key (default `epyc_9124`); resolves
  `calibration_profiles/faiss_ivf/<cpu>.json` and never falls back to raw preset coefficients
- `--calibrated`: Path to calibrated coefficients JSON (overrides `--hardware`)
- `--config`: Path to rag_stack YAML config (overrides `--hardware`)
- `--num-threads`: FAISS thread counts to sweep (default: `0 1` = all cores AND single-thread; `0` = all cores)
- `--parallel-modes`: Must be `0` (the only standalone-calibrated mode)
- `--batch-sizes`: Query batch sizes (default: `1 1000 10000`). FAISS uses BLAS when batch size >= 20
- `--index-configs`: FAISS index factory strings to benchmark
- `--nprobe-values`: nprobe values to test (default: `1 2 4 6 8 12 16 24 32 48 64 128`)
- `--k`: Number of neighbors (default: 1)
- `--cold-cache`: Use cold cache model (default: warm cache)
- `--datasets`: Benchmark datasets (default: `sift1m deep1m gist1m`)
- `--data-dir`: Dataset directory (default: the shared `CM_BENCHMARK_DIR` benchmark cache)
- `--download-url`: URL to download the dataset tarball
- `--actuals-csv`: Reuse a previously measured actuals CSV instead of re-running FAISS
- `--output-dir`: Output directory for plots (default: `tmp_outputs/faiss_sim_bench`)

## Calibration

Cost-model coefficients are produced directly, reproducibly, under
[benchmarks/rag_cm_accuracy/](https://github.com/haiqiang-zhang/rag-stack/tree/main/benchmarks/rag_cm_accuracy) — or via the
unified `rag_stack calibrate ivf --config <box.yaml>` entry point (see
[the calibration workflow](calibration.md)):

- **CPU-specific model coefficients** — all runtime-consumed values are fit
  directly (scipy
  `differential_evolution`, wide bounds) against real multi-dataset FAISS
  component measurements. Raw CPU facts such as topology, peak throughput,
  cache hierarchy, and STREAM bandwidth remain in the hardware spec. Runtime
  selects the independently deployable profile by its exact CPU hardware key;
  the profile does not retain source paths, timestamps, or raw-file hashes. The
  reproducible drivers are
  [benchmarks/rag_cm_accuracy/ivf/bench.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/benchmarks/rag_cm_accuracy/ivf/bench.py)
  and [benchmarks/rag_cm_accuracy/hnsw/bench.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/benchmarks/rag_cm_accuracy/hnsw/bench.py)
  (`python benchmarks/rag_cm_accuracy/ivf/bench.py measure|predict
  [--system <cpu> | --config <yaml>]`). These validation drivers never launch
  calibration on a profile miss; `predict` requires the reviewed profile to
  exist. Use the staged `rag_stack calibrate measure/fit` lifecycle to create
  it.
- **Per-dataset workload predictor** (HNSW ndis/nhops) — trained per dataset:
  `python -m rag_stack.cost_model.faiss_hnsw_sim.train --dataset-hdf5 <f> --dataset-name <d> --embedding <e>`
  → `calibration_profiles/faiss_hnsw/<dataset>__<embedding>.joblib`.

The measurement helpers remain available: `collect_hnsw_benchmarks`
(faiss_hnsw_sim/calibrator.py), `parse_index_config` and
`register_hardware_from_yaml` (faiss_ivf_sim/calibrator.py).

Raw hardware presets (`SYSTEM_PRESETS` in
[system_presets.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/cost_model/faiss_ivf_sim/system_presets.py)):
`m3_pro`, `epyc_7313`, `epyc_9124`, `ultra7_265k`. New presets can be added
there, but pricing additionally requires an exact latest-schema CPU profile.

Output: a JSON file with calibrated coefficients. Production resolves it only
from `system.cpu.hardware_key` (or the CPU preset string) as
`calibration_profiles/faiss_ivf/<cpu>.json`. Pipeline YAML cannot inject a
different coefficient file:

```yaml
algo_search_space:
  vectordb:
  - name: faiss_ivf_index
    db_type: faiss_ivf
```

## Index string anatomy — `IVF{nlist},PQ{M}x{nbits}{suffix}`

- **IVF1024**: Inverted File Index with `nlist=1024` Voronoi cells. At query time only the closest `nprobe` cells are scanned, trading recall for speed.
- **PQ{M}**: Product Quantization that splits each vector into **M** sub-vectors and quantizes each independently. Larger M = higher accuracy but more memory/compute (e.g. `PQ16` → 16 sub-vectors, `PQ32` → 32, `PQ64` → 64).
- **x{nbits}**: Bits per sub-quantizer code. `x4` means 4 bits (16 centroids per sub-vector) instead of the default 8 bits (256 centroids). Lower bits = smaller index + faster scan at some accuracy cost.
- **fs** (fast-scan): Uses SIMD-optimized PQ distance computation with lookup tables packed for vectorized processing (`PQFastScan`). Significantly faster on modern CPUs.
- **fsr** (fast-scan with residual): a FAISS shape that is not currently priced;
  the cost model fails closed because its standalone grid measures only the
  default non-residual FastScan path.
- **IVF{nlist},Flat**: IVF-Flat — exact full vectors per cell, no quantization (`index_type="flat"`).
- Configs without the `x{nbits}` suffix (e.g. `PQ16`, `PQ32`) use the default 8-bit encoding and standard (non-fast-scan) PQ.
