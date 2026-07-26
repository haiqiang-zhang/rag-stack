---
title: User Guide
nav_order: 2
---

# RAG-Stack User Guide

Run the installation steps in the [README](https://github.com/haiqiang-zhang/rag-stack/blob/main/README.md#installation) first.
RAG-Stack and its pinned [RAG-Stack-Evaluator](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/README.md)
submodule must be installed in the same environment. That provider owns static
quality evaluation, measured execution, static metrics, dataset utilities, and
vLLM instrumentation. The separate top-level `FlashRAG/` submodule provides
the optional FlashRAG backend through the same host evaluator API.

After updating this checkout or switching branches, synchronize the pinned
submodules and reinstall the editable packages for your selected environment.
For the CUDA 12 environment described in the README:

```bash
conda activate rag-stack
git submodule update --init --recursive
uv pip install --torch-backend cu128 \
  -e 'RAG-Stack-Evaluator[cu12]' -e '.[cu12]'
# For NVIDIA driver >=580, use cu130 and cu13 for both projects.

# Optional: only for global.eval_backend: flashrag.
uv pip install -e FlashRAG
```

Use the CPU installation variant in the README for API/CPU development. See the
[evaluator integration guide](evaluators.md) for migration steps and package
ownership. Run these commands from the host repository root. The submodule
update follows
this branch's recorded commits, including the top-level FlashRAG
fork; do not replace them with unrelated checkouts or a package-index FlashRAG
release. FlashRAG is optional for `static_gt`.

This is the top-level entry point for *using* RAG-Stack. Everything runs
through the `rag_stack` command, which drives the Controller
([rag_stack/controller.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/controller.py)) — the component that
executes the decoupled MOBO loop:

```bash
conda run -n rag-stack python -m rag_stack <subcommand> [flags]
```

| Subcommand | Purpose |
|---|---|
| `optimize` | Run a RAG optimization — full MOBO, hardware replay, or system transfer (**this page**) |
| `eval-one` | Measure one saved pipeline/system JSON pair on real hardware — see [Evaluate one saved configuration](eval_one.md) |
| `quality-sweep` | Pre-compute a reusable quality store for hardware replay (**this page**) |
| `calibrate` | Calibrate the cost model to your hardware — see [Calibrating the cost model](rag_cm/calibration.md) |
| `benchmark` / `pareto-cache` / `recompute-metrics` | Optimizer benchmark harness — see [optimizer.md](optimizer.md) |

What the three config axes (`eval_backend` / `performance_source` /
`rag_ir_mode`) mean is covered in
[experimental_functions.md](experimental_functions.md); the config file
syntax is documented in [config_schema.md](config_schema.md)
and the annotated key-by-key reference
[configs/rag_stack/config_reference.yaml](https://github.com/haiqiang-zhang/rag-stack/blob/main/configs/rag_stack/config_reference.yaml).
This page is about *operating* runs: launching, monitoring, resuming,
replaying, transferring — plus the settings and cache layout every run
inherits.

## Choosing an evaluation backend

Keep the same six-section optimization config and `python -m rag_stack`
commands after the evaluator extraction. Select execution through these keys:

| `global.eval_backend` | `system.performance_source` | Execution |
|---|---|---|
| `static_gt` (default) | `cost_model` | The evaluator produces quality and, in dynamic mode, the trace consumed by the host cost model. |
| `static_gt` | `measured` | The evaluator executes the deployment on assigned GPUs and records measured performance and quality. |
| `flashrag` | `cost_model` | The evaluator runs the optional FlashRAG backend and supplies traces for host cost-model replay. Native component hooks capture the serving trace. |

`flashrag` with `performance_source: measured` is unsupported and rejected by
validation. FlashRAG metric names and pipeline methods belong to that backend;
changing `eval_backend` alone does not convert a static configuration. See the
[configuration schema](config_schema.md) and the
[FlashRAG integration guide](flashrag.md) for its
settings, supported methods, trace contract, and artifacts.

For custom Python integrations, use the host-owned `rag_stack.evaluation` API,
registry, and tracing hooks. Construct any selected backend with
`rag_stack.evaluation.registry.create_quality_evaluator`. The
`rag-stack-evaluator` distribution provides `static_gt`, while the separate
FlashRAG distribution provides `flashrag`. Independent plugins register through
the same entry-point
mechanism and return the same result contract. Host dataset management remains
`rag_stack.dataset_manager.DatasetManager`. See the
[Evaluator Extension Guide](evaluator_extensibility.md) and
[Evaluator API Reference](evaluator_api.md).

## Standard mode: a full optimization run

An optimization run takes one YAML config (search space + evaluation settings
+ hardware) and produces a Pareto frontier of RAG pipeline configurations.
[`config_reference.yaml`](https://github.com/haiqiang-zhang/rag-stack/blob/main/configs/rag_stack/config_reference.yaml) is an
annotated template: adapt the dataset paths and hardware before running it.
Its inline `system.cpu` example needs an explicit, new `hardware_key`, or
replace the entire CPU block with a supported preset such as `epyc_9124`.
Use the preset only when it describes your target, and complete the required
[calibration](rag_cm/calibration.md) for strict mode. The config is self-contained:
`dataset.qa` and `dataset.corpus` name the source parquets. Save the prepared
config as `/path/to/run.yaml`, then launch:

```bash
conda run -n rag-stack python -m rag_stack optimize \
    --config /path/to/run.yaml
```

A fresh project directory `<config-stem>_<timestamp>` is auto-created under
`tmp_outputs/rag_stack_projects/` (root configurable via
`rag_optimization_project_dir` in
[`.rag_stack_settings.json`](#rag_stack_settingsjson); don't pass
`--project-dir` for new runs). The run then executes the MOBO loop:
Phase 1 both-objective init →
Phase 2 MOBO iterations → Pareto extraction.

Common variations:

```bash
# Override data / iteration count from the CLI
conda run -n rag-stack python -m rag_stack optimize \
    --config /path/to/run.yaml \
    --qa-data datasets/dragonball/qa_en_sample_10.parquet \
    --corpus-data datasets/dragonball/raw_corpus_en.parquet \
    --n-iterations 20

# Measured performance (real on-GPU vLLM timing) — complete configs live in
# configs/baselines/; any cost-model config can also be flipped with
# --performance-source measured if it carries the measured system keys
# (system.system_design_space, incl. available_gpus).
conda run -n rag-stack python -m rag_stack optimize \
    --config configs/baselines/sgs_3090x4_smac_measured.yaml
```

### Monitoring a run

Long runs should be detached. Every run writes a persistent log to
`<project>/logs/optimize_<timestamp>.log` (all loggers, including vLLM
subprocess output) with a stable `optimize_latest.log` symlink — watch that
instead of the console:

```bash
PYTHONUNBUFFERED=1 nohup conda run -n rag-stack --no-capture-output python -u \
    -m rag_stack optimize --config /path/to/run.yaml \
    > /tmp/optimize_console.log 2>&1 &
tail -f tmp_outputs/rag_stack_projects/<run-dir>/logs/optimize_latest.log
```

(`PYTHONUNBUFFERED=1 ... --no-capture-output python -u` is needed because
`conda run` buffers stdout.) On SIGINT/SIGTERM the CLI reaps every descendant
process (vLLM EngineCore workers included) before exiting, so no GPU memory is
left pinned.

### What a run produces

```
<project>/
    rag_stack_config.yaml          ← copy of the run config
    config.freeze.json             ← immutable input YAML snapshot from first startup
    summary.json                   ← run status + PID lock
    data/                          ← in-project QA/corpus copy (resume-stable)
    logs/optimize_*.log            ← persistent run logs (+ _latest symlink)
    evaluations/eval_NNNN/         ← per-eval pipeline config, performance,
                                     quality metrics, execution DAG
    all_evaluations.csv            ← flattened table of every evaluation
```

### Resuming / recovering an existing project

Rerunning `optimize` against an existing project directory loads the
checkpoint from `evaluations/` + `all_evaluations.csv`, warm-starts the
optimizer, and continues from Phase 2:

```bash
conda run -n rag-stack python -m rag_stack optimize \
    --project-dir tmp_outputs/rag_stack_projects/<run-dir>
```

Resume semantics:

- **`--config` is optional on resume** — the in-project `rag_stack_config.yaml`
  is used. Configuration freezes before the first run starts. Passing a
  different config or changing `--performance-source`, `--cm-calibration-mode`
  or `--n-iterations` is rejected; create a new project for those changes.
- **Backend enforcement** — API, CLI and direct Controller calls share the
  same project lock and frozen input snapshot. Failure, completion and reset
  do not unlock it. Changes made to the project YAML outside these entry
  points are rejected at the next run/evaluation check. The snapshot records
  raw YAML without resolving credentials or hashing experiment data. Older
  started projects without a snapshot adopt their retained YAML on the first
  guarded resume; their earlier contents cannot be reconstructed from it.
- **Data pinning** — the in-project `data/` copy always wins over `--qa-data` /
  `--corpus-data`, keeping quality comparable across resumes.
- **PID lock** — `summary.json` records the owning PID; a second `optimize` on
  the same project refuses to start while that process is alive.
- **Crash recovery** — a killed run (e.g. SIGKILL, node reboot) can leave
  `summary.json` at `status: running`. Once the recorded PID is gone, the next
  `optimize` treats it as a stale checkpoint and resumes it — no manual
  cleanup needed. Runs that ended as `failed` or `completed` resume the same
  way.

### Measured baselines with a remote vLLM judge

For measured baselines, run the DeepEval judge on a separate GPU host so the
measured pipeline owns the benchmark GPUs. The judge is an OpenAI-compatible
vLLM server; the baseline gets its address from environment variables, not from
the YAML config.

On the judge host, start Qwen with two data-parallel replicas, each using TP=4
(8 GPUs total):

```bash
conda activate rag-stack
cd /path/to/rag-stack

JUDGE_MODEL=Qwen/Qwen3-32B \
JUDGE_SERVED_MODEL_NAME=Qwen/Qwen3-32B \
JUDGE_HOST=0.0.0.0 \
JUDGE_PORT=8000 \
JUDGE_TP=4 \
JUDGE_DP=2 \
JUDGE_DP_LOCAL=2 \
JUDGE_MAX_MODEL_LEN=16384 \
JUDGE_MAX_NUM_SEQS=16 \
JUDGE_GPU_MEMORY_UTILIZATION=0.90 \
JUDGE_DTYPE=bfloat16 \
bash RAG-Stack-Evaluator/scripts/start_vllm_judge_server.sh -- --disable-custom-all-reduce
```

Equivalent raw vLLM command:

```bash
vllm serve Qwen/Qwen3-32B \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name Qwen/Qwen3-32B \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --data-parallel-size-local 2 \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --disable-custom-all-reduce
```

Then run the functional A100x8 MOBO measured baseline on the measured host:

```bash
conda activate rag-stack
cd /path/to/rag-stack

JUDGE_BASE_URL=http://<judge-host>:8000/v1 \
JUDGE_MODEL=Qwen/Qwen3-32B \
JUDGE_MAX_CONCURRENCY=32 \
JUDGE_DEEPEVAL_MAX_CONCURRENT=32 \
JUDGE_MAX_TOKENS=1024 \
JUDGE_TIMEOUT=1200 \
JUDGE_STRUCTURED_OUTPUT_MODE=prompt_json \
DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=1800 \
DEEPEVAL_TASK_GATHER_BUFFER_SECONDS_OVERRIDE=1800 \
bash scripts/run_measured_baseline_mobo.sh
```

Use the same judge environment with `scripts/run_measured_baseline_smac.sh` or
`scripts/run_measured_baseline_sobol.sh` for the SMAC and Sobol baselines.

### `optimize` flag reference

| Flag | Default | Meaning |
|---|---|---|
| `--config` | — | Config YAML for a new project. Optional on resume; a different config is rejected after first startup. |
| `--project-dir` | auto | **Omit for new runs** (auto-created, see above). Pass it explicitly only to resume an existing project. |
| `--qa-data` / `--corpus-data` | config `dataset.qa` / `dataset.corpus` | Source parquets. Ignored on resume. |
| `--n-iterations` | config `global.n_iterations` | Set the TOTAL GT-call budget before first startup; persisted and frozen with the config. |
| `--performance-source` | config value | `cost_model` \| `measured` — set before first startup, then frozen with the project config. |
| `--mode` | `full` | `full` = standard MOBO; `hardware_replay` = compute Pareto from a quality store (see [below](#quality-store-and-hardware-replay)). |
| `--quality-store` | — | In normal `full` mode, an exact-config read/write cache for quality + frozen trace; it never seeds optimizer observations or candidates. In `hardware_replay`, the replay source (required unless `--from-project` is given). |
| `--from-project` | — | Synthesize a store at `<project>/quality_store/` from a finished project's GT evals, then replay it. Used by [system transfer](#system-transfer). |
| `--n-refine-iterations` | 0 | Optimizer refinement after replay; new GT evals are written back to the store. |
| `--hardware-config` | — | Optional JSON representation of the target hardware; it must match the config's `system` block (see [Step 2c](#step-2c-target-hardware)). |

## Quality store and hardware replay

Quality Store decouples expensive quality (GT) evaluation from cheap
performance (CM) simulation. Evaluate quality once, then instantly compute
Pareto frontiers for different hardware configurations.

It can also accelerate an otherwise normal optimization without changing its
search. Pass `--quality-store` with the default `--mode full`: the optimizer
selects candidates exactly as before, and only after selection does RAG-Stack
look up the candidate's fully enriched algorithm config. An exact hit reuses
that record's quality metrics and frozen v2 trace; a miss runs the normal GT
evaluation and appends the resulting pair to the store. Top-level `system`
deployment settings are deliberately excluded from the key, while fixed
algorithm semantics (for example `use_chat_template` and a pinned chunker) are
included. A record is eligible only when its complete quality protocol also
matches the run: metric definitions, judge model/prompt settings, aggregation,
query subset, and other `eval_backend_setting` semantics are compared exactly.
On optimizer steps that request only the minimal metric budget, a full cached
record is projected to that same minimal view before it reaches the optimizer;
the cache therefore cannot reveal auxiliary metrics that a cache miss would
not have computed.

```bash
conda run -n rag-stack python -m rag_stack optimize \
    --config /path/to/run.yaml \
    --quality-store quality_stores/my_project
```

Stores are bound to the exact corpus + QA content. Adding records from another
project is accepted only when both identities match; quality metrics and trace
are committed and replaced as one checksummed record, never mixed across evals.

### Step 1: Create a Quality Store

Run quality evaluation for the configs in the search space with
`quality-sweep`. Results are git-trackable and reusable across hardware.
`--config`, `--qa-data`, `--corpus-data` and `--store-path` are all required:

```bash
# Exhaustive sweep with grid discretization (n_grid=5 for continuous params)
conda run -n rag-stack python -m rag_stack quality-sweep \
    --config /path/to/run.yaml \
    --qa-data datasets/dragonball/qa_en_sample_100.parquet \
    --corpus-data datasets/dragonball/raw_corpus_en.parquet \
    --store-path quality_stores/dragonball_reference \
    --n-grid 5 --strategy exhaustive

# Sobol sampling (for large search spaces)
conda run -n rag-stack python -m rag_stack quality-sweep \
    --config /path/to/run.yaml \
    --qa-data datasets/dragonball/qa_en_sample_100.parquet \
    --corpus-data datasets/dragonball/raw_corpus_en.parquet \
    --store-path quality_stores/dragonball_reference_sobol \
    --n-configs 50 --strategy sobol
```

**Strategy choice:**

- `exhaustive` (default) — grid discretization + cartesian product. Only viable
  for small search spaces (< ~1000 total combinations); `--n-grid` controls the
  grid density for continuous parameters (default 5).
- `sobol` — quasi-random sampling for large/hierarchical search spaces;
  `--n-configs` sets the sample count.

Alternatively, synthesize a store from an already-finished optimization
project. The web/API "save quality store" operation supports either creating a
new store or appending a project to an existing same-dataset store. For an
ad-hoc replay, pass `--from-project <old_project_dir>` to the replay commands
below instead of `--quality-store`.

The store is saved to `quality_stores/dragonball_reference/` with:

- `manifest.json` — schema, exact corpus/QA identity, config hashes, token stats
- `configs/` — fully enriched algorithm configs (no `system` block)
- `quality/` — GT evaluation metrics per config
- `traces/` — frozen v2 quality trace paired with each metric record
- `pipeline_configs/` — resolved source pipeline configs
- `records/` — atomic commit markers, provenance, completeness, and checksums

### Step 2a: Hardware Replay

Compute Pareto frontier for a specific hardware config using pre-computed
quality data:

```bash
conda run -n rag-stack python -m rag_stack optimize \
    --config /path/to/run.yaml \
    --qa-data datasets/dragonball/qa_en_sample_100.parquet \
    --corpus-data datasets/dragonball/raw_corpus_en.parquet \
    --mode hardware_replay \
    --quality-store quality_stores/dragonball_reference
```

This runs CM for each quality-evaluated config and computes direct Pareto.
Small deployment spaces may complete in seconds. A cost-model-owned system
space with tens of thousands of exact placement/batch candidates per pipeline
can take substantially longer; progress is written after every source point so
an operational transfer can be monitored from its project log and summary.

### Step 2b: Hardware Replay + Refinement

Optionally let the optimizer suggest additional configs to improve the Pareto
frontier. New quality evaluations are written back to the store for future
reuse:

```bash
conda run -n rag-stack python -m rag_stack optimize \
    --config /path/to/run.yaml \
    --qa-data datasets/dragonball/qa_en_sample_100.parquet \
    --corpus-data datasets/dragonball/raw_corpus_en.parquet \
    --mode hardware_replay \
    --quality-store quality_stores/dragonball_reference \
    --n-refine-iterations 20
```

### Step 2c: Target hardware

Define the replay target in the config's `system` block. Copy your validated
source config, update its CPU/GPU identities, physical interconnect, and
`cm_search_space` for the target, and keep the dataset and quality protocol
consistent with the store:

```bash
cp /path/to/run.yaml /tmp/replay_target.yaml
# Edit /tmp/replay_target.yaml for your dataset and target system, including
# the required CPU hardware_key or preset and the calibration policy.

conda run -n rag-stack python -m rag_stack optimize \
    --config /tmp/replay_target.yaml \
    --mode hardware_replay \
    --quality-store quality_stores/dragonball_reference
```

The optional `--hardware-config <json>` argument must serialize the same target
hardware as the YAML `system` block. It does not independently override that
block: a mismatch is rejected to keep assembly pricing and pipeline policy on
one target. Omit it when the config already defines the target. For a target
with incomplete calibration, set `system.rag_cm.calibration_mode` to
`uncalibrated_spec` explicitly and treat its results as exploratory, as described
in the [schema](config_schema.md).

## System Transfer

System transfer creates a new RAG Optimize project from a previous project and
a new system YAML. Only the YAML's `system` block is read; dataset, pipeline
search space, quality settings, and data are inherited from the source project.
The new project then runs `hardware_replay` from the source project: first CM
recomputes performance on the new system for the source quality points, then
optional refinement warm-starts the optimizer from those replayed points and
runs additional global optimization iterations.

Every imported quality trace and transfer/provenance artifact uses the one
canonical, versionless structure, while target performance is always computed
by production CM. Replay is always dynamic; a missing or malformed trace
aborts the transfer instead of falling back to aggregate/static replay. Every
source quality point is repriced on the target and the complete
quality/performance history is warm-started into the optimizer before
refinement.

Replay time depends on the inherited workflow. For large dynamic sweeps, CM
uses the canonical trace and the same frozen stage-batch evidence to build a
stratified analytical shortlist; only shortlisted exact deployments enter the
authoritative DES, and only DES results are eligible for target-system
selection. Phase 2a commits each source observation independently, so an
interrupted sweep resumes at the first uncommitted observation instead of
repricing the completed prefix.

Dedicated transfer targets live under `configs/system_transfer/`. Files under
`configs/calibrate/` remain pure calibration hardware specs and are not shown in
the system-transfer picker.

A transfer target may declare its default Phase 2b budget outside the system
block:

```yaml
system_transfer:
  n_refine_iterations: 10
  generator_backend: vllm
  rag_config_overrides:
    optimizer:                 # complete section replacement (not a deep merge)
      type: agent_qnehvi
      params: {seed: 43, n_both_init: 10}
    eval_backend_setting:      # complete section replacement
      combined_quality:
        mode: mean
        metrics: [deepeval_answer_correctness]
      metrics:
      - metric_name: deepeval_answer_correctness
        model: openai/deepseek/deepseek-v4-flash
      - metric_name: deepeval_faithfulness
        model: openai/deepseek/deepseek-v4-flash
system:
  rag_cm:
    calibration_mode: uncalibrated_spec
  performance_source: cost_model
  # physical hardware and cm_search_space ...
```

`system_transfer.n_refine_iterations` must be a non-negative integer. An
explicit API value overrides the YAML default for that transfer; targets without
the metadata retain the server's compatibility default of 5. Transfer metadata
is orchestration policy, not RAG configuration: the generated
`rag_stack_config.yaml` contains only the normal six sections. The resolved
budget and whether it came from the YAML or an API override are recorded in the
API response and `<new_project>/system_transfer.json`.

`system_transfer.generator_backend` selects the refinement generation
transport. `vllm` projects inherited API modules onto local chat,
`vllm_api` retains the API modules and validates their endpoint, and `inherit`
keeps the source choice. Configs that omit the field use `inherit` for backward
compatibility.

`system_transfer.rag_config_overrides` may contain only `optimizer` and
`eval_backend_setting`. Each supplied block replaces the complete inherited
section; it is not recursively merged, so retired optimizer flags or diagnostic
metrics cannot leak in from the source project. Omitting the block preserves the
historical source-inheritance behavior. Because Phase 2a reuses source quality,
an evaluator override may add or remove diagnostic metrics but may not change
`combined_quality`, any other aggregation/sampling setting, or the evaluator
definition of a metric used by the combined objective. Such a change is rejected
before the target project is created. Metric `description` is display/prompt
metadata and may change.

An optimizer override has exactly `type` plus optional mapping `params`; unknown
top-level optimizer keys are rejected. Static-GT `metrics` must use one
homogeneous representation—either `list[str]` or `list[dict]`. Dict entries use
`metric_name` (the old `metric` alias, mixed forms, empty names, and duplicate
declarations are rejected).

`uncalibrated_spec` is the intended mode for a physical target whose CM
campaign is incomplete: exact matching profiles are still used when available,
while genuinely missing components fall back to the declared physical spec.
Malformed or identity-mismatched calibration evidence still fails closed; the
mode does not turn predicted values into calibrated measurements.

For `generator_backend: vllm`, the inherited target search space is projected
from `vllm_api` to `vllm`, API transport fields are removed, and
`use_chat_template: true` is made explicit. The source project, QualityStore
records, and canonical traces remain frozen for replay provenance. This mode
needs no `RAG_LLM_BASE_URL`, HTTP model gateway, or `vllm serve` process; Phase
2b loads the selected model through local `vllm.LLM.chat()`.

Optimizer exports that store configs and traces in separate trees
must first be converted to the standard project layout. The source export is
never modified and the destination is published atomically only after all CSV,
quality, canonical-trace, and checksum checks pass:

```bash
conda run -n rag-stack python -m rag_stack convert-optimizer-export \
    --source-project /path/to/optimizer_export \
    --output-project tmp_outputs/rag_stack_projects/<seed>_source
```

The conversion manifest uses the canonical artifact structure. Its separate
`format_version: 1` identifies only the converter layout and is orthogonal to
CM.

From the web UI:

1. Start the API backend and webapp as shown in the
   [web app guide](apps.md).
2. Open **RAG Optimize**.
3. Click **New Project**.
4. Set **Project kind** to **System transfer**.
5. Choose a source project and a system YAML. Optionally override its refine
   iteration default, then create.

The web form leaves the override blank and displays the selected YAML's
effective default (20 for `sgs_a100_8x.yaml`). Only a value entered by the user
is sent as an explicit override; a blank field inherits the YAML value.

The UI copies the source project's `rag_stack_config.yaml`, replaces `system`,
applies any complete optimizer/evaluator section overrides, projects the chosen
generation transport, validates the resulting six-section config, and launches
the run. A complete source `data/` snapshot is reused;
converted projects without one resolve `dataset.qa` and `dataset.corpus` under
the configured `datasets/` root instead of pre-copying the large corpus in the
API layer. The synthesized quality store lives at
`<new_project>/quality_store/`; transfer-only metadata remains in
`<new_project>/system_transfer.json`.

Pure CLI equivalent, after you have prepared the target project directory with
the copied source `data/` and a `rag_stack_config.yaml` whose `system` block has
already been replaced:

```bash
conda run -n rag-stack python -m rag_stack optimize \
    --project-dir tmp_outputs/rag_stack_projects/dragonball_a100_transfer \
    --mode hardware_replay \
    --from-project tmp_outputs/rag_stack_projects/dragonball_baseline \
    --n-refine-iterations 10
```

For direct CLI replay, omitting `--n-refine-iterations` means Phase 2a only
(`0` refinement iterations). The server passes a selected transfer YAML's
resolved default explicitly because transfer metadata is deliberately not
copied into the six-section project config.

A successful run writes target evaluations incrementally and finishes with an
atomic `system_transfer_result.json`. The result records the source/repriced
counts, requested/completed refinement count, target integer eval IDs, source
algorithm hashes and cost-model implementation version. The project is marked
completed only when the landed refinement count exactly matches the request.

Interrupted transfers resume in the same target project. The canonical
`system_transfer_state.json` freezes the ordered source hashes, their
QualityStore commit identities, the target config, and CM. Its refinement
budget is a cumulative target: resume may increase it, but may never decrease
it. Before a completed budget is extended, every old refinement marker and its
checksummed ledger must validate exactly under the old budget; only then is the
state atomically reopened for the larger target. On restart the CLI reuses the existing local QualityStore
only when its source manifest and source artifacts still match; it never
re-imports or merges over that store. Phase 2a continues at the first missing
source observation, and Phase 2b warm-starts from every committed replay plus
landed refinement before running exactly `requested - landed` more iterations.
Complete eval/CSV records interrupted just before their small commit marker are
promoted after checksum validation. An incomplete uncommitted tail is rolled
back and retried; a non-prefix history or any source/target/checksum mismatch
fails closed. Resetting the target project deletes its local `quality_store/`,
transfer state, evaluations, and final result so the next launch is a genuinely
fresh import.

## System settings

### `.rag_stack_settings.json`

Sticky, per-checkout settings live in `.rag_stack_settings.json` at the repo
root (gitignored; also editable from the web UI Settings page). To use a
custom path, set the environment variable before starting:

```bash
export RAG_STACK_SETTINGS_PATH=/path/to/my_settings.json
```

Keys the CLI reads:

| Key | Used by | Meaning |
|---|---|---|
| `rag_optimization_project_dir` | `optimize` | Root under which new per-run project dirs `<config-stem>_<timestamp>` are created. Empty → `tmp_outputs/rag_stack_projects/`. |
| `optimizer_benchmark_dir` | `benchmark` / `pareto-cache` / `recompute-metrics` | Shared default output dir for the benchmark subcommands. Empty → `tmp_outputs/optimizer_benchmark_results/`. |

### `.env` (API keys & secrets)

API keys and secrets (e.g. `DEEPSEEK_API_KEY` for the LLM judge) are stored in
a `.env` file at the project root (path configurable in the web UI Settings
under **Keys & Secrets**; gitignored). `rag_stack` loads `.env` with
`override=True`, so credential and cache values in that file win over shell
values. Explicit process-level routing and concurrency settings are preserved:
`RAG_LLM_BASE_URL`, `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_MAX_CONCURRENCY`,
`JUDGE_DEEPEVAL_MAX_CONCURRENT`, `CM_SWEEP_WORKERS`, and `CM_DES_WORKERS`.
Keep cache-root settings stable throughout a campaign (next section).

### Cache directories

Shared model and runtime caches default to the repository's `.cache/` folder;
chunked corpora and FlashRAG artifacts stay in the project directory. vLLM's
torch.compile artifacts alone can use GBs per model/shape and exhaust a small
home-directory quota. `rag_stack` sets these defaults at import time
([rag_stack/\_\_init\_\_.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/__init__.py)), so every entry point
(optimizer, measured harness, replay scripts) and every subprocess they spawn
inherits them.

| Cache | Env var | Default |
|---|---|---|
| Embedding + FAISS artifact caches | `RAG_STACK_CACHE_DIR` | `<repo>/.cache/rag_stack` |
| Chunked-corpus cache | — | `<project>/data/chunks` |
| FlashRAG materialized datasets, corpora, and indexes | Project directory | `<project>/_flashrag/` |
| vLLM compile + engine caches | `VLLM_CACHE_ROOT` | `<repo>/.cache/vllm` |
| Triton JIT kernels | `TRITON_CACHE_DIR` | `<repo>/.cache/triton` |
| FlashInfer JIT workspace | `FLASHINFER_WORKSPACE_BASE` | `<repo>` (tool appends `.cache/flashinfer`) |
| LlamaIndex cache | `LLAMA_INDEX_CACHE_DIR` | `<repo>/.cache/llama_index` |
| HuggingFace hub (model weights) | `HF_HOME` | `<repo>/.cache/huggingface` |

**To relocate any of them**: export the env var before launching (or put it in
your shell profile) — the in-repo defaults are `setdefault`-only and never
override your environment. Do **not** put `RAG_STACK_*` overrides in `.env`
mid-campaign: `rag_stack` loads `.env` with `override=True`, and re-pointing a
cache root between runs silently orphans every previously built FAISS index.

Each worktree defaults to its own repo-local cache. To reuse warm embeddings
and FAISS from another worktree, export its absolute cache path before launch,
for example `RAG_STACK_CACHE_DIR=/path/to/warm-repo/.cache/rag_stack`.
Chunked corpora remain project-local by design. `HF_HOME` holds model weights
(hundreds of GB), so export one shared path when worktrees should also share
model downloads. Compile/JIT caches remain worktree-local unless explicitly
relocated.

The evaluator extraction keeps the host's existing `RAG_STACK_CACHE_DIR`
default. FlashRAG materialization uses its separate project-local
`_flashrag/` tree; the selected evaluation `run_dir` holds runtime outputs,
with `<project>/_flashrag_run/` used when callers omit it. These paths remain
stable after the move. See the FlashRAG provider's
[artifacts and lifecycle](flashrag.md#artifacts-and-lifecycle)
for content identities and index reuse.
