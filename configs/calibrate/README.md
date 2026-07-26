# Calibration Hardware Specs

Files in this directory are input-only hardware specs for calibration commands.
They provide stable hardware keys and physical CPU/GPU/interconnect numbers;
they must not contain fitted coefficients, E2E corrections, or PD/assembly
scalars.
They are read with the normal RAG-Stack YAML loader and validated with the
system-only subset of `ConfigValidator`.

Generated profiles are written by the calibrators under `calibration_profiles/`.
Users should not hand-edit generated profile JSON/joblib files.

For a new machine, write to isolated roots first:

```bash
rag_stack calibrate status --config configs/calibrate/<box>.yaml \
  --data-dir /path/to/stage/calibration_data \
  --profile-dir /path/to/stage/calibration_profiles
```

Both flags apply to every calibrate component. Relative explicit paths are
resolved before the dispatcher changes working directory; defaults remain the
repository `calibration_data/` and `calibration_profiles/`. See
[`docs/rag_cm/calibration.md`](../../docs/rag_cm/calibration.md) for the
dependency order and promotion checklist.

The system spec participates in these independently keyed profile layers:

- **CPU system profiles** are keyed by the CPU/system spec, for example
  `calibration_profiles/faiss_ivf/<system>.json` and
  `calibration_profiles/faiss_hnsw/<system>.json`.
- **Corpus × embedding workload profiles** are keyed by data and embedding
  model, for
  example `calibration_profiles/faiss_ivf/imbalance/<dataset>__<embedding>.json`
  and `calibration_profiles/faiss_hnsw/<dataset>__<embedding>.joblib`.
- **GPU/component and runtime profiles** are keyed by the declared GPU plus
  exact component/model/precision/topology identity under
  `calibration_profiles/llm_sim/` or the component-specific directory.
- **CPU retrieval-worker profiles** are keyed by the declared CPU under
  `calibration_profiles/retrieval_worker/`; they currently contain standalone
  multi-list RRF service evidence.

Retrieval-worker readiness is conditional and is not included in the default
`core` scope. For a pipeline with multi-list RRF (`K>1`):

```bash
rag_stack calibrate status --config configs/calibrate/<box>.yaml \
  --scope retrieval-worker \
  --data-dir /path/to/stage/calibration_data \
  --profile-dir /path/to/stage/calibration_profiles

rag_stack calibrate retrieval-worker --config configs/calibrate/<box>.yaml \
  --data-dir /path/to/stage/calibration_data \
  --profile-dir /path/to/stage/calibration_profiles
```

The command measures and fits the production RRF CPU operator into the exact
CPU-keyed profile. Pipelines with `K=1` do not need it.

Files in `configs/calibrate/` define only the system spec/key. They do not
define any of those generated records. RAG E2E results are regression evidence,
never a calibration profile or fit source, and there is no assembly/PD
calibration stage. `system.assembly` and the old `system.serving` calibration
override are not valid system-spec fields.

For a new FAISS CPU system, add a YAML file with an inline CPU spec:

```yaml
system:
  cpu:
    hardware_key: my_cpu
    num_cores: 32
    peak_flops: 3.8e12
    peak_int_ops: 1.9e12
    mem_bandwidth: 3.0e11
    mem_bandwidth_1t: 3.8e10
    memory_tiers:
      - {name: L3, bandwidth: 1.0e12, bandwidth_1t: 6.7e10, capacity: 16777216}
      - {name: ddr, bandwidth: 3.0e11, bandwidth_1t: 3.8e10, capacity: 512000000000}
    topology: {type: numa, sockets: 2, cores_per_socket: 16}
  gpu: H100_GPU
  available_gpus: [cuda:0]
```

Then run:

```bash
python -m rag_stack calibrate ivf --config configs/calibrate/my_cpu.yaml
python -m rag_stack calibrate hnsw --config configs/calibrate/my_cpu.yaml
```

The resulting profiles are saved as:

```text
calibration_profiles/faiss_ivf/my_cpu.json
calibration_profiles/faiss_hnsw/my_cpu.json
```
