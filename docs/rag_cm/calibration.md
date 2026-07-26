# Calibrating RAG-CM on a new machine

This is the operational calibration guide. All commands below use the supported
`rag_stack calibrate` CLI. A successful campaign produces immutable raw
measurements under `calibration_data/` and runtime profiles under
`calibration_profiles/`.

Runtime profiles are self-contained and portable. Selection uses the exact
hardware/component key, and validation uses only the profile's coefficients,
fit grids, held-out diagnostics, and content schema. Profiles do not store or
verify calibration source paths, raw-file hashes, timestamps, commits, or
measurement manifests. The complete `calibration_data/` tree may be absent on
a deployment machine without affecting profile loading or pricing.

Calibration never requires measuring every deployment. The empirical serving
curve has only this identity:

```text
hardware system × performance stage × resident batch
```

TP, PP, placement, timeout, token counts and workflow visits remain analytical
or trace-driven inputs. They are not additional calibration identities. Do not
add deployment-specific residuals or fitted knobs to make individual replay
points match.

## 1. What must be calibrated

RAG-CM combines five kinds of evidence:

| Evidence | Identity | Purpose |
|---|---|---|
| CPU component | CPU hardware × component | IVF/HNSW physical service |
| Workload | corpus × embedding model | IVF imbalance and HNSW traversal work |
| GPU component | GPU hardware × component × model | Embedding, reranker and generator physical work |
| Stage service | hardware system × stage × batch | Runtime overhead at a real engine boundary |
| Replay validation | full pipeline + trace | Validate end-to-end prediction; never fit coefficients |

For vLLM, the component profile contains independently fitted physical
compute/memory efficiency and has zero per-call overhead. Runtime overhead is
owned only by the stage-service curve:

- a collocated engine cycle receives one collocated stage correction;
- a direct P/D deployment uses independently measured prefill and decode
  engine curves plus physical KV/NIXL transfer;
- collocated prefill and decode overhead must never be added separately to the
  same mixed scheduler cycle.

The sparse stage batch design is fixed:

- measured batches: `1, 8, 32, 64, 128, 256`;
- fitted anchors: `1, 8, 32, 64, 256`;
- held-out validation: `128`;
- interpolation is allowed inside the measured range;
- extrapolation outside the range is fail-closed.

Generator component calibration separately includes resident batch 128 in its
fit evidence and holds batch 256 out for validation. This ensures that the two
largest production batches are covered without creating a large calibration
grid.

## 2. Prepare an isolated campaign

Complete the repository [installation](../evaluators.md) first,
including `git submodule update --init --recursive` and the editable
`RAG-Stack-Evaluator` package in the same environment as RAG-Stack. The host
owns calibration commands and profile fitting; the evaluator owns the
runtime nodes, measured vLLM subprocesses and instrumentation that those
commands measure. The relevant import namespaces are
`rag_stack_evaluator.static_rag_evaluator` and
`rag_stack_evaluator.vllm_instrumentation`. They are supplied by the pinned
submodule, not by modules under `rag_stack/`.

The top-level FlashRAG provider is optional for this campaign; install it
only when evaluating FlashRAG pipelines. See [FlashRAG integration](../flashrag.md)
for that backend.
Keep the submodule revision recorded by this checkout when reproducing
measurements: a different runtime implementation can change the measured
service boundary.

Start from the repository root in the project Conda environment:

```bash
conda activate rag-stack
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"

RUN_ROOT="$PWD/tmp_outputs/calibration/$(hostname)-$(date +%Y%m%d)"
DATA_DIR="$RUN_ROOT/calibration_data"
PROFILE_DIR="$RUN_ROOT/calibration_profiles"
mkdir -p "$DATA_DIR" "$PROFILE_DIR"
```

Keep these two roots for the entire campaign. Do not measure into one root and
fit into another. Do not edit published JSON by hand.

Every command accepts the same global root options in any position:

```bash
--data-dir "$DATA_DIR" --profile-dir "$PROFILE_DIR"
```

The process-wide alternatives are `RAG_STACK_CALIBRATION_DATA_DIR` and
`RAG_STACK_CALIBRATION_PROFILE_DIR`. Explicit CLI roots take precedence;
relative CLI paths resolve from the invoking directory, while relative
environment paths resolve from the repository root. Runtime pricing uses the
profile root only. Keep one profile generation for the complete run, rather
than setting a different root for each candidate or hardware-replay config.

Raw measurements are append/freeze evidence. Reuse them for CPU-only refits.
Use `--force` only when intentionally replacing an invalid or partial
measurement cohort.

## 3. Verify the hardware configuration

A calibration YAML must describe the actual machine, not the machine implied
by its filename. Check at least:

```bash
nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
nvidia-smi topo -m
lscpu
numactl --hardware
```

Verify:

- visible GPU IDs, model and memory;
- CPU model, sockets, physical cores and NUMA layout;
- GPU-to-GPU and GPU-to-CPU topology;
- driver/CUDA compatibility with the repository pins;
- memory and storage capacity needed by model and corpus calibration.

The YAML must contain stable hardware identities:

```yaml
system:
  performance_source: measured
  system_hardware: my_host_gpu_system
  gpu: A100_80GB_GPU
  cpu:
    hardware_key: my_cpu_system
    # measured CPU specification follows
  available_gpus: ["cuda:0", "cuda:1"]
  interconnect:
    # measured topology follows
```

`system_hardware` identifies the complete host, accelerator and interconnect
boundary used by stage curves. A preset GPU name alone is insufficient.

For the audited SGS A100×8 host, the hardware-only starting point is:

```text
configs/calibrate/sgs_a100_8x.yaml
```

It declares `system_hardware: sgs_a100x8_epyc7742`. Change it only if the
current machine is physically different, and document the authoritative
measurements used for every change.

Production pipeline configurations must use strict mode:

```yaml
system:
  rag_cm:
    calibration_mode: strict
```

Do not put profile-directory selectors in individual run configs. Select an
isolated campaign through the two CLI root options, then publish the resulting
profile root as one reviewed generation.

## 4. Inspect readiness before measuring

Use a hardware-only config for component scopes and the complete RAG config for
pipeline readiness:

```bash
HW_CFG=configs/calibrate/sgs_a100_8x.yaml
FULL_CFG=configs/rag_stack/your_full_pipeline.yaml

rag_stack calibrate plan \
  --config "$HW_CFG" \
  --scope core \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"

rag_stack calibrate status \
  --config "$FULL_CFG" \
  --scope pipeline \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

`plan` prints missing work in dependency order and exits successfully.
`status` is the readiness gate:

- exit `0`: every selected item is ready;
- exit `2`: at least one required item is missing or invalid;
- `--json`: emit parseable JSON with the same exit-code contract.

Pipeline status examines every selectable component. If
`system.cm_search_space.placement_policy` permits `disaggregated`, a
collocated-only stage profile is incomplete; all six collocated/direct-P/D
stage profiles are required.

Always rerun `plan` after completing a step. It prints executable next actions
and explicitly labels any blocker that still needs a user-supplied physical
fact.

## 5. Run the calibration campaign

The following order avoids repeating expensive work.

### 5.1 CPU IVF and HNSW profiles

```bash
rag_stack calibrate ivf \
  --config "$HW_CFG" \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

For a built-in CPU, or when a valid HNSW profile already supplies the reviewed
hardware parameters, the normal HNSW command is:

```bash
rag_stack calibrate hnsw \
  --config "$HW_CFG" \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

For a custom CPU without an existing HNSW profile, all five independently
measured HNSW hardware parameters are mandatory. The one command below
measures and fits:

```bash
rag_stack calibrate hnsw \
  --config "$HW_CFG" \
  --hnsw-simd-efficiency VALUE \
  --hnsw-dram-random-access-latency-ns VALUE \
  --hnsw-scattered-bw-ceiling VALUE \
  --hnsw-numa-local-bytes VALUE \
  --hnsw-numa-remote-bw-ceiling VALUE \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

These five values are all-or-nothing. Measure them; do not copy them from a
different CPU. Until they are supplied, `calibrate plan` emits the safe raw
measurement step but marks fitting as blocked.

The `all-faiss` shortcut is appropriate only when the HNSW parameters are
already resolvable; it calibrates FAISS components, not GPU or serving stages.

### 5.2 Corpus × embedding workload profiles

This step uses the real corpus and query sample. It may load embedding models,
encode the corpus and build FAISS indexes, but it does not construct the
optimizer, quality evaluator or measured performance provider.

```bash
WORKLOAD_PROJECT="$RUN_ROOT/workload-project"
QA_DATA=/absolute/path/to/qa.parquet
CORPUS_DATA=/absolute/path/to/corpus.parquet

rag_stack calibrate workload \
  --config "$FULL_CFG" \
  --project-dir "$WORKLOAD_PROJECT" \
  --qa-data "$QA_DATA" \
  --corpus-data "$CORPUS_DATA" \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

On a resume, the workload project reuses its frozen `data/` directory. Keep
passing the same reviewed paths. A global embedding cache hit is reused.

Expected profile families are:

```text
calibration_profiles/faiss_ivf/imbalance/<corpus>__<embedding>.json
calibration_profiles/faiss_hnsw/<corpus>__<embedding>.joblib
```

### 5.3 Embedding and reranker GPU components

Run only the exact components selected by pipeline inventory. Query embedding
profiles are component-agnostic because the embedding model belongs to the
vectordb configuration:

```bash
rag_stack calibrate llm \
  --config "$HW_CFG" \
  --cost-model retrieval_encode_sim \
  --module huggingface_all_mpnet_base_v2 \
  --batch-sizes 1,8,32,64,128,256 \
  --seq-lens 16,32,64,128,256 \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

Example reranker:

```bash
rag_stack calibrate llm \
  --config "$HW_CFG" \
  --cost-model passage_reranker_sim \
  --module tart \
  --batch-sizes 1,8,32,64,128,256 \
  --seq-lens 128,256,512 \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

Do not assume that a model-map entry proves an acquisition runner exists.
Follow pipeline inventory; an unsupported selectable component remains
fail-closed until it has a real runner and exact profile.

### 5.4 Generator component profiles

Calibrate the models reachable by the full pipeline. With `--model`, each
command measures one model; without it, the calibrator schedules its canonical
model family across available GPUs.

```bash
rag_stack calibrate generator \
  --config "$HW_CFG" \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

Repeat for each generator and query-expansion model reported missing by
`calibrate plan`. Keep the default sparse fit and held-out validation grids
unless changing the calibration contract deliberately.

### 5.5 Collocated vLLM stage curves

Measurement freezes independent component and complete collocated-engine
telemetry. Fitting is CPU-only:

```bash
rag_stack calibrate generator-tp measure \
  --config "$HW_CFG" \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"

rag_stack calibrate generator-tp fit \
  --config "$HW_CFG" \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

Despite its compatibility name, `generator-tp` does not calibrate each TP/PP
deployment. It publishes one sparse hardware × stage × batch generation.
TP/PP remains analytical.

### 5.6 Direct disaggregated P/D curves

Run this section only when the production search space permits disaggregated
placement. It requires the generator component and collocated campaigns above.

```bash
rag_stack calibrate generator-disagg measure \
  --config "$HW_CFG" \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"

rag_stack calibrate generator-disagg fit \
  --config "$HW_CFG" \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

This measures real independent prefill/decode engines at batches
`1,8,32,64,128,256` and binds physical transfer evidence. It does not infer P/D
curves by subtracting collocated out=1/out=N measurements.

### 5.7 Other reachable components

Pipeline inventory emits exact commands when these are required:

```bash
rag_stack calibrate retrieval-worker --config "$FULL_CFG" \
  --data-dir "$DATA_DIR" --profile-dir "$PROFILE_DIR"

rag_stack calibrate llmlingua2 --config "$FULL_CFG" \
  --data-dir "$DATA_DIR" --profile-dir "$PROFILE_DIR"
```

Do not run components that the pipeline cannot select.

## 6. Validate and activate

The final strict gate is the full pipeline status:

```bash
rag_stack calibrate status \
  --config "$FULL_CFG" \
  --scope pipeline \
  --data-dir "$DATA_DIR" \
  --profile-dir "$PROFILE_DIR"
```

Do not activate the campaign unless this exits `0`.

Then run a small CM-versus-measured replay using the same saved pipeline,
system configuration and trace. Compare prediction ranking and end-to-end QPS;
do not fit new coefficients from replay error. A replay mismatch should first
be classified as one of:

- wrong or stale profile identity;
- missing batch coverage;
- component/stage boundary mismatch;
- trace/workload mismatch;
- measured implementation overhead or bug;
- unsupported deployment semantics.

After review, publish the complete `PROFILE_DIR` atomically as the production
`calibration_profiles` generation. Runtime needs only that profile tree;
it does not read or verify `DATA_DIR`. Preserve `DATA_DIR` only when you want
to repeat a CPU-only refit later.

## 7. Time budget

The following five-hour planning budget assumes a prepared machine with
models and datasets already cached:

| Serving task | Target ceiling |
|---|---:|
| Generator component family | 120 min |
| Collocated stage campaign | 90 min |
| Direct P/D campaign, if enabled | 60 min |
| Fit, status and bounded replay | 20 min |
| Total | 290 min |

This is not a universal guarantee for complete first-time hardware migration.
CPU microbenchmarks, full-corpus embedding, index construction, uncached model
downloads and optional component campaigns depend on the machine and dataset.
Prepare/cache them before starting serving calibration, and record
preparation time separately from serving-calibration time.

## 8. Troubleshooting

### `status` exits 2

This is normal before calibration. Run `calibrate plan`, execute the printed
commands in order, then rerun status.

### `system.system_hardware` is missing

Add a stable identity for the exact host + GPU + interconnect system. Do not
reuse another machine's identity merely because the GPU model matches.

### Pipeline is ready for collocated but not disaggregated

The config permits `placement_policy: disaggregated`, but only the collocated
pair exists. Run `generator-disagg measure` and `generator-disagg fit` to
publish the complete six-profile generation.

### Embedding profile is missing

Run the inventory-emitted `retrieval_encode_sim` command. The publisher stores
the record under the component identity `agnostic`, matching runtime lookup.

### Workload command is expensive

A cache miss really encodes corpus/query vectors and builds data-dependent
profiles. Reuse the same workload project and embedding cache. It must not
initialize the optimizer or quality evaluation.

### HNSW custom CPU refuses to fit

Provide all five measured HNSW hardware parameters. There is intentionally no
generic custom-CPU fallback.

### A profile exists but is invalid

Do not clamp values or edit JSON. Preserve the evidence, identify the failed
contract, and remeasure only the owning component or stage. Rerun `status`
before publishing.

## 9. Completion checklist

- [ ] Hardware YAML matches the physical machine.
- [ ] `system_hardware`, CPU key and GPU key are stable and exact.
- [ ] One isolated data root and profile root were used throughout.
- [ ] CPU component profiles are valid.
- [ ] Every corpus × embedding workload profile is present.
- [ ] Every reachable embedding/reranker/compressor profile is valid.
- [ ] Every reachable generator model has component evidence including B128
      and held-out B256 coverage.
- [ ] Collocated stage curves cover B1/8/32/64/256 with held-out B128.
- [ ] If disaggregated placement is selectable, all direct P/D curves exist.
- [ ] Full pipeline `calibrate status` exits 0.
- [ ] Bounded measured replay was used only for validation, not coefficient fit.
- [ ] Preparation time and serving-calibration time were recorded separately.
