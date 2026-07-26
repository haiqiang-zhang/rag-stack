---
title: Config Schema
nav_order: 4
---

# RAG-Stack Config Schema

This document defines the YAML config format that drives every
`python -m rag_stack optimize` run. The fully annotated template
companion is [`configs/rag_stack/config_reference.yaml`](https://github.com/haiqiang-zhang/rag-stack/blob/main/configs/rag_stack/config_reference.yaml);
this file explains the *concepts and syntax rules* behind it. Adapt the template
to your dataset and target hardware before running it. In particular, its inline
`system.cpu` needs a new `hardware_key` or replacement with a supported preset.

A config is parsed by [`rag_stack/config_validator.py`](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/config_validator.py)
(`ConfigValidator`) and resolved by
[`rag_stack/performance_context.py`](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/performance_context.py) +
[`rag_stack/controller.py`](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/controller.py). Validation collects
**all** errors and reports them at once before any evaluation starts.

---

## 1. Core concept

A config is a **search-space specification**, not a single pipeline. The
optimizer samples points from it and evaluates each on two decoupled objectives:

- **quality** — LLM-judged / metric ground-truth evaluation of the RAG answers.
- **performance** — latency/throughput, from one of two sources (cost model or
  real on-GPU measurement).

This is the optimizer's input format. The resolved evaluator format is defined
separately in the [evaluator input contract](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/docs/input_contract.md).
Replay uses `algo_config.json` and `system_config.json` with explicit
`--qa`/`--corpus` arguments; dataset selection and training/test splits are
recorded in `RUN.md`, outside both JSON files. See [replay usage](replay.md).

The single most important syntax rule follows directly from this:

> **A list-valued field is a search dimension; a scalar is a fixed constant.**

```yaml
top_k: [1, 2, 4, 8]          # the optimizer chooses one of these per trial
top_k: 8                     # always 8 — not searched
model: [Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct]   # 2-way choice
model: Qwen/Qwen2.5-7B-Instruct                               # fixed
```

A single-value list (`top_k: [8]`) is not a dimension either — it is silently
collapsed to the fixed value.

There are three sweep-spec value forms beyond the plain list:

- **`{range: [lo, hi]}`** — a **continuous** parameter:
  ```yaml
  temperature: {range: [0.0, 1.0]}   # continuous ∈ [0.0, 1.0]
  temperature: [0.0, 0.3, 0.7, 1.0]  # discrete 4-way choice
  ```
- **`{ordered: [...]}`** — an **ordered categorical**, declared inline at the
  knob whose values it orders. String choices are UNORDERED by default
  (components, index families); a knob with a semantic order that string sort
  would scramble (model-size ladders: `1.5B < 14B < 3B < 7B`
  lexicographically) wraps its values in `ordered:` and the dimension is built
  `is_ordered=True` in **declaration order** — all optimizers read the same
  flag (numeric-kernel treatment + index spacing in the shared search-space
  ops, SMAC `OrdinalHyperparameter` for ≥3 values, Ax ordinal encoding, greedy
  probes the trade-off endpoints first):
  ```yaml
  - component: vllm
    model:
      ordered:          # YAML value order IS the semantic order (size ladder)
      - Qwen/Qwen2.5-1.5B-Instruct
      - Qwen/Qwen2.5-3B-Instruct
      - Qwen/Qwen2.5-7B-Instruct
      - Qwen/Qwen2.5-14B-Instruct
  ```
  Numeric lists are auto-ordered already (wrapping one is a harmless no-op);
  `{ordered: <scalar>}` and single-value `ordered` lists fail the build loudly.
- Fields that are *inherently* lists (e.g. `algo_search_space.vectordb:`,
  `node_lines:`, `modules:`, `memory_tiers:`) are structure, not sweep dims.

---

## 2. The six sections

The schema is a fixed set of six top-level keys. Order is convention, not
enforced. Pre-6-section top-level keys and renamed knobs are hard-rejected by
the validator with a message pointing at the current location.

| # | Section | Purpose |
|---|---------|---------|
| 1 | `global` | run-level switches (quality backend, CM engine mode, iteration budget) |
| 2 | `dataset` | source QA + corpus parquets |
| 3 | `system` | hardware spec + performance source + per-mode deployment search spaces |
| 4 | `optimizer` | optimizer type + params |
| 5 | `eval_backend_setting` | everything backend-specific *except* the algo search space |
| 6 | `algo_search_space` | the RAG algorithm dims (corpus chunker + vectordb + pipeline) |

---

## 3. Section reference

### 1 · `global`

```yaml
global:
  eval_backend: static_gt   # static_gt (default) | flashrag — routes the quality evaluator
  rag_ir_mode: dynamic      # dynamic (default) | static — which CM engine prices a trial
  n_iterations: 100         # total GT-call budget (Phase 1 init + Phase 2 MOBO)
```

- `eval_backend` picks the quality backend implemented by the
  [evaluator submodule](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/README.md) (see §5).
  `static_gt` supports both performance sources; `flashrag` supports
  `cost_model` only. Measured execution requires `static_gt`.
- `rag_ir_mode` selects the CM engine for `static_gt` under
  `performance_source: cost_model`:
  - `dynamic` (default) — the DYNAMIC engine: replay each eval's recorded
    trace + workflow schema (real per-call token counts).
  - `static` — the STATIC engine: config-driven aggregate RAGO estimate, no
    trace needed.

  FlashRAG uses its recorded trace for CM replay regardless of this setting;
  keep its monitor enabled. Measured execution does not use the CM engine.

### 2 · `dataset`

```yaml
dataset:
  dataset_name: dragonball_en                           # REQUIRED — explicit dataset id
  qa: datasets/dragonball/qa_en_sample_100.parquet      # QA parquet (carries retrieval_gt)
  corpus: datasets/dragonball/raw_corpus_en.parquet     # corpus parquet
```

- `dataset_name` is **mandatory** — a non-empty string naming the dataset. It
  keys the IVF cell-imbalance profile (`dataset_name__embedding`) so the cost
  model loads the right data-aware scan-count curve. NEVER derived from a file
  path (the in-project corpus copy has a fixed basename and abs paths differ
  across machines); two runs over the same corpus must share one `dataset_name`.
- Repo-relative paths. CLI `--qa-data` / `--corpus-data` override these.
- If the corpus is **raw** and the chunker is a search dim
  (`algo_search_space.corpus.chunker`), it is re-chunked per eval. If it is an
  already-segmented final corpus, omit the chunker block and it goes straight
  to the vectordb.

### 3 · `system`

```yaml
system:
  performance_source: cost_model    # cost_model (default) | measured
  rag_cm:
    calibration_mode: strict        # strict (default) | uncalibrated_spec
  cpu: { ... }                      # cost-model CPU spec (cores, flops, memory_tiers, topology)
  gpu: A100_80GB_GPU                # GenZ preset name, OR a custom {hardware_key, Flops, ...} block
  interconnect:                     # DECLARATIVE comm fabric (cost-model only). Per device-pair,
    gpu_gpu: {bandwidth_gbps: 450, latency_us: 0.4, type: nvlink4}   # decimal, one-direction GB/s;
    gpu_cpu: {bandwidth_gbps: 54, latency_us: 0.8, type: pcie5_host} #   no preset table. `type` is a
                                    #   FREE-TEXT LABEL only (not resolved to a number). KV handoff
                                    #   (prefill→decode) uses gpu_gpu; vectors/text use gpu_cpu.
                                    #   Omitted pair / no bandwidth_gbps → gpu.ICN fallback.
  performance_objectives:
    selection: max_throughput       # reduces the RAGO sweep to one perf scalar:
                                    #   min_latency (default) | max_throughput |
                                    #   min_ttft | min_tpot | constrained
  retrieval:                        # search-time FAISS runtime knobs (BOTH perf modes)
    faiss_num_threads: null         # int = pin | [ints] = search dim | null/omitted
    #                               # = DERIVE min(batch, physical cores) — the single
    #                               # policy shared by CM and measured.
    # faiss_indexing_thread: 30     # BUILD-time OMP threads (kmeans/PQ/HNSW graph).
    #                               # int >= 1 = pin | null/omitted = DEFAULT cpu_count-2.
    #                               # NOT a search dim / NOT in the cost model (build is
    #                               # outside CM measurement); restored to 1 after build
    #                               # so it never pollutes the search-thread anchor.
    # faiss_ivf_parallel_mode: 0    # scalar PIN only (0=inter-query, 1=intra-query).
    #                               # CM-managed deployment axis — never an optimizer
    #                               # dim; unpinned, the CM sweeps {0,1} per batch.
  cm_search_space: { ... }          # cost-model RAGO deployment sweep (see below)
  system_design_space: { ... }      # optimizer-owned deployment dims (see §4)
```

- **`gpu` / `cpu` — `hardware_key` is the SINGLE calibration locator.** A bare
  preset key (`gpu: H100_GPU`, `cpu: epyc_9124`) locates ceff/meff + every
  calibration profile by itself. **A known preset key may NOT also carry explicit
  specs — it is a hard config error** (it would silently shift the base the
  calibration was fit against). For NEW hardware, give a dict with a fresh
  `hardware_key` + all specs (`gpu: {hardware_key: my_h200, flops_tflops: …}`,
  `cpu: {hardware_key: my_epyc, num_cores: …}`); the specs flow to GenZ/FAISS and
  the calibration profiles are saved/located under that new key. (`interconnect`
  is exempt from the preset-exclusive rule — it has no calibration profile and no
  preset table; `{type: nvlink4, bandwidth_gbps: 450, latency_us: 0.4}` with numbers
  written directly stays valid, `type` being a documentary label only.)
- **`interconnect` feeds both physical communication models.**
  `gpu_gpu.bandwidth_gbps` and `gpu_gpu.latency_us` are physical topology facts.
  They drive the PD/stage communication model and GenZ's analytical TP/PP model.
  Production does not load `_tp_pp_efficiency`, add a fitted NCCL collective
  residual, or apply a fitted PP penalty. TP and PP remain deployment choices
  priced from analytical compute, memory and physical communication; they are not
  calibration-profile identities. Legacy TP/PP fields that remain in an old
  `calibration_profiles/llm_sim/<gpu>.json` artifact are ignored by production
  pricing and must not be used as calibration evidence.
- **Calibration profiles** are operator-local and use the hardware identity of
  the resource that executes the operator: `faiss_ivf/<cpu>.json`,
  `faiss_hnsw/<cpu>.json`, and `retrieval_worker/<cpu>.json` are CPU-keyed;
  `llm_sim/<gpu>.json` and `compressor_llmlingua2/<gpu>.json` are GPU-keyed.
  A non-hardware workload specialization goes in its named workload namespace
  (for example `faiss_ivf/imbalance/<dataset>__<embedding>.json`). There is no
  assembly/E2E/PD scalar profile and no `system.assembly` coefficient override:
  placement, communication, batching, and PD behavior are simulated from the
  declared topology and component service models.
- **The production CM implementation is `rag_stack.cost_model`.** All
  serialized CM inputs and outputs use one canonical structure, validated by
  exact required fields, types, identities and hashes; artifacts carry no
  schema-version field and there is no version negotiation. Deployable
  calibration artifacts live under the canonical `calibration_profiles` root.
- **`system.rag_cm.calibration_mode`** controls missing-profile policy for the
  cost model. Omit it, or set `strict`, for the production default: every exact
  calibration required by a candidate must exist and validate.
  `uncalibrated_spec` is an explicit new-hardware mode. An exact valid component
  profile is still used when present; if the exact component profile is absent,
  the component uses its analytical physical-hardware specification instead.
  It never borrows a profile from another hardware/model identity and never
  treats malformed, stale, or identity-mismatched evidence as merely absent.
  Empirical-only LLMLingua2 and multi-list RRF wrappers use labelled
  hardware-roofline lower bounds when absent; their Python/host wrapper cost is
  not implied by the hardware specification. Treat this mode as exploratory,
  not calibrated evidence.

  This is the only supported key in the `system.rag_cm` block.
  Per-config `component_calibration_store` and `calibration_store` fields are
  unsupported and fail validation. Remove them instead of pointing them at a
  directory. Calibration uses the canonical
  `calibration_profiles` root, with an optional process-wide
  `RAG_STACK_CALIBRATION_PROFILE_DIR` override for an isolated campaign.

  `calibration_profiles/stage_batch` has a stronger fairness rule in
  `uncalibrated_spec` mode. The stage correction is enabled only when the exact
  hardware file contains one valid complete six-profile generation
  (`generator`, `query_expansion`, and both direct P/D roles for each). If that
  generation is absent or incomplete, **none** of its stage-batch curves are
  used for any deployment candidate. This prevents a collocated candidate from
  receiving a calibrated correction while a disaggregated candidate is priced
  without the corresponding correction.

  The optimize CLI may override and persist this field in the project config:

  ```bash
  python -m rag_stack optimize \
    --config configs/rag_stack/sgs_a100x8_dragonball_agent_s45.yaml \
    --cm-calibration-mode uncalibrated_spec
  ```

  Passing `--cm-calibration-mode strict` explicitly switches a resumed project
  back to strict mode. The setting changes CM pricing policy only; measured
  performance remains measured.
- **`cpu.memory_tiers`** should stay monotonic (`L1 > L2 > L3 > DDR` bandwidth)
  — the roofline model assumes faster upper tiers (not validator-enforced).
  Optional `cpu.topology` block (validated): `uniform` (default) / `numa`
  (needs `sockets` × `cores_per_socket == num_cores`) / `heterogeneous`
  (needs `p_cores.count + e_cores.count == num_cores`). NUMA and heterogeneous
  are mutually exclusive.
- **`system.retrieval`** holds search-time retrieval runtime knobs (never part
  of the index-build signature). `faiss_num_threads`: an int pins, a list is an
  optimizer search dim, and `null` (or omitting the key) means DERIVE
  `min(batch, physical cores)` — the one policy shared by CM and measured; an
  explicit `null` just documents that intent in the YAML.
  `faiss_ivf_parallel_mode` is **system design space managed by the
  cost model** — never an optimizer dim: unpinned, the CM prices both modes
  per batch inside its own deployment sweep and keeps the best (per-batch
  argmin is exactly the global sweep result — the mode consumes no shared
  resource); a scalar pins it, a list is rejected (an explicit grid belongs in
  `cm_search_space.faiss_ivf_parallel_mode`). Retrieval uses `batch_size_request`;
  it does not expose a separate batch axis. (Calibration profiles were fit at
  1 thread, so `faiss_num_threads > 1` is roofline extrapolation in CM mode.)
- **`cm_search_space`** — the RAGO deployment sweep used by the cost model:
  ```yaml
  cm_search_space:
    placement_policy: [disaggregated, collocated]
    min_num_gpus: 2
    max_num_gpus: 4
    max_num_gpus_per_stage: 4
    gpus_per_server: 4
    max_batch_size_request: 256     # positive power of 2
    no_microbatching: true          # request cohort shared by non-decode stages
    allow_decode_collocation: true  # may decode share a GPU group with prefill/etc.
  ```
  Three deployment axes are **default-ON** — the CM sweeps them even when the
  keys are omitted; an explicit list overrides the grid, a scalar pin
  suppresses the sweep:
  - `cm_search_space.batch_size_decode` — decode-engine concurrency
    (deployed `max_num_seqs`), agentic rows only, constraint `decode ≥
    request`. Default: powers of 2 ≤ `max_batch_size_request`. Pin with the
    scalar `system.batch_size_decode`.
  - `cm_search_space.dynamic_batch_timeout_s` — light-stage dynamic-batching
    wait cap. Default `[0.002, 0.01, 0.05]`. Priced with no effect in the
    saturated steady-state model; carried as a deployment-runtime axis. Pin with
    `system.batching.dynamic_timeout_s`.
  - `cm_search_space.faiss_ivf_parallel_mode` — FAISS IVF query-time threading
    (0=inter-query, 1=intra-query). Default `[0, 1]`, priced per batch inside
    the vector-search stage sweep with the best mode kept (single-mode when no
    `faiss_ivf` index — HNSW ignores the knob). Pin with the scalar
    `system.retrieval.faiss_ivf_parallel_mode`. Never an optimizer dim.

### 4 · `optimizer`

```yaml
optimizer:
  type: agent_qnehvi           # from OPTIMIZER_REGISTRY
  params:
    seed: 44
    n_both_init: 10            # joint-objective init points before BO
    # ... optimizer-specific params
```

Registry (`rag_stack/optimizer/__init__.py`):

| `type` | What |
|--------|------|
| `sobol` | pure Sobol quasi-random baseline (no surrogate; hierarchy-aware) |
| `greedy` | budget-aware two-mode greedy baseline (see below) |
| `ax_qnehvi` | qLogNEHVI via Ax (single-task GP per outcome) |
| `ax_qnehvi_mixedgp` | qLogNEHVI with a mixed categorical+continuous kernel |
| `smac_mobo` | SMAC3 ParEGO (random-forest scalarized MOBO) |
| `agent_qnehvi` | **main optimizer** — LLM-agent-guided qLogNEHVI |
| `agent_qnehvi_slo` | `agent_qnehvi` + suggest-time SLO gating via the static cost model |
| `greedy_then_hv_anchor_agent_qnehvi` | greedy first pass, then `agent_qnehvi` with HV-band prompt discipline and expected-HV-gain pool ranking |

Common params:

- `seed` — RNG seed.
- `objectives` — list of `{objective_name, ref_point, cost}` for `performance`
  and `quality` (SMAC/Ax HV reference points; `cost` is the decoupled
  eval-cost ratio).
- **Init-budget naming differs by optimizer** (a known wart):
  - `ax_qnehvi` / `ax_qnehvi_mixedgp` / `agent_qnehvi` use **`n_both_init`**.
  - `smac_mobo` uses **`n_gt_init`** (aliased to `n_both_init` internally).
- `agent_qnehvi` constructs an LLM Agent for structured initialization and,
  by default, every post-init BO round. `agent_model` selects the model
  (default `deepseek-v4-flash`). Its mechanism switches are listed below.
- `composite_objectives.quality.sub_metrics` is project/result metadata used
  for expanded CSV columns. It is auto-derived from
  `eval_backend_setting.metrics` when descriptions are present and is not an
  optimizer constructor control. The Agent reads the observed metric payload
  and the evaluator metric definitions directly.

`agent_qnehvi` mechanism switches belong in `optimizer.params`. All default
to `true` and require YAML booleans, not quoted strings. They control search
after initialization. The default 10-point initialization requests five
Sobol points followed by five Agent-designed points; rejected, duplicate,
or missing Agent proposals use the existing Sobol fallback.

| Parameter | Effect when `false` |
| --- | --- |
| `enable_stage_signals` | Hide stage quality diagnostics and performance attribution from all Agent evidence; continue computing and recording them. |
| `enable_agent` | Disable post-init Agent calls, raw Agent proposals, and Agent-seeded DLS; retain the initialization Agent. |
| `enable_dls` | Skip all DLS sources, polishing, and DLS restart scoring. |
| `enable_doe` | Disable Pareto-tension crossover candidates and their dedicated evidence feedback. |
| `enable_forced` | Disable the stagnation-triggered gap-design and forced-selection branch; retain ordinary Agent/pool arbitration. |

These switches do not disable quality reuse or change which quality metrics
are computed and recorded. `enable_agent: false` retains the initialization
Agent; `enable_stage_signals: false` changes only the evidence exposed to the
Agent. To compare mechanisms, hold initialization and evaluation settings fixed.

`greedy` params (the AutoRAG-style baseline ladder — pure black-box, no CM
side information; the TOTAL GT budget `n_iterations` is auto-injected from
`global.n_iterations` / benchmark `gt_budget` and split across pipeline
stages proportionally to `log2(1 + #options)`, largest-remainder integerized;
a method-level shape switch like `rag_dataflow` forks into independent
per-branch chains):

```yaml
optimizer:
  type: greedy
  params:
    search_mode: forward       # forward = AutoRAG-faithful single pass |
                               # lookback (default) = + backward/forward
                               #   refinement passes (geometric budget split)
    beam_width: 3              # breeding slots per branch, by ROLE: both front
                               #   ends (span anchors) + interior points by
                               #   exclusive-HV contribution (the knee).
                               #   3 = fast end + accurate end + knee (default);
                               #   2 = ends only; 1 = plain single chain
    stage_weight: log          # log (default) | uniform (allocation ablation)
    pass_budget_fraction: 0.6  # γ — lookback only: each pass gets γ of the
                               #   branch's remaining budget
    max_passes: 8              # lookback pass cap
    # per_stage_cap: 4         # fallback cap, used only when no n_iterations
                               #   budget is injected
```

### 5 · `eval_backend_setting`

Everything backend-specific **except** the algo search space. Keys are
interpreted per `global.eval_backend`. The host resolves the search space and
passes these settings through the host-owned `rag_stack.evaluation` interface.
Backend packages contribute implementations and capability descriptors through
Python entry points. The host owns registration, result validation, and generic
tracing hooks; configuration validation queries `rag_stack.evaluation.registry`.
The bundled `rag_stack_evaluator` package supplies `static_gt`. The separate
FlashRAG package supplies `flashrag`; both use the same host interface as
third-party backends.

Install the host and evaluator together as described in the
[README](https://github.com/haiqiang-zhang/rag-stack/blob/main/README.md#installation). The extraction does not introduce new
top-level config sections or rename `static_gt`, `flashrag`, or
`eval_backend_setting`. See the [evaluator integration guide](evaluators.md)
for installation and migration details.

**static_gt:**

Static quality and measured execution are provided by
`rag_stack_evaluator.static_rag_evaluator`. The same metric definitions apply
to both performance sources; local metrics do not require an LLM judge.

```yaml
eval_backend_setting:
  queries_per_trial: 80                 # measured-mode subsample (head N of qa.parquet); omit = all
  combined_quality:                     # how sub-metrics fold into the single `quality` objective
    mode: mean                          # mean | weighted (weighted takes {metric: weight})
    metrics: [deepeval_answer_correctness]
  metrics:                              # the full metric set (target + auxiliaries)
  - metric_name: deepeval_answer_correctness
    model: openai/deepseek/deepseek-v4-flash
    description: GEval factual correctness vs gold answer. THE business target.
  - metric_name: deepeval_context_recall
    model: openai/deepseek/deepseek-v4-flash
    description: Retrieval ceiling.
  - metric_name: retrieval_token_recall # local token-overlap metric (no LLM judge)
```

- **`combined_quality`** defines the scalar `quality` objective the optimizer
  maximizes (usually just the business-target metric). All other `metrics` are
  auxiliary signals kept for surrogate reconstruction / stage diagnosis.
- **`metrics`** is the full vocabulary, validated against the backend's metric
  set. Each LLM-judged metric needs `model`; local metrics (e.g.
  `retrieval_token_*`) need none. `description` is the one recognized
  meta-field — it feeds the agent optimizer's auto-derived
  `composite_objectives.quality.sub_metrics` (§4); all other keys are
  forwarded to the metric function as kwargs.
- `performance_only: true` is a standalone measured-replay/benchmark switch —
  it skips ALL quality scoring and is rejected in an optimize config.

**flashrag:**

The optional backend is implemented by
`flashrag.integrations.rag_stack` in the top-level `FlashRAG` submodule. RAG-Stack resolves the
search space and forwards this block. The evaluator owns backend validation,
translation, pipeline execution, scoring, and trace production. See the
[FlashRAG guide](flashrag.md) for installation and
runtime contracts.

```yaml
eval_backend_setting:
  metrics:
  - metric_name: em
  - metric_name: f1
  - metric_name: rouge-l
  flashrag:                             # FlashRAG framework settings
    framework: vllm
    monitor: full
    gpu_memory_utilization: 0.85
    max_retrieval_num: 5
    model2path: { e5: intfloat/e5-base-v2 }
```

FlashRAG uses its own metric vocabulary (`em`, `sub_em`, `f1`, `acc`, `recall`,
`precision`, `bleu`, `rouge_score`, `rouge-1`, `rouge-2`, `rouge-l`, `llm_judge`,
`input_tokens`, `retrieval_recall`, `retrieval_precision`). The adapter accepts
`rouge_l` as an alias for `rouge-l`; static metric names are not translated.
It has no `prompt_maker` or power-of-two `top_k` requirement.

Use `global.eval_backend: flashrag` with
`system.performance_source: cost_model`; measured FlashRAG execution is not
supported. Component tracing is mandatory: the FlashRAG fork records exact
native component input/output through RAG-Stack's global hooks. Legacy
`monitor: full` and `monitor: light` settings are accepted; `monitor: off` is
rejected because dynamic CM replay requires a serving trace.

The evaluator materializes FlashRAG data and indexes under
`<project>/_flashrag/`. These are separate from the static evaluator's shared
`RAG_STACK_CACHE_DIR`. The resolved `pipeline_runtime.mode` selects the method;
`flashrag.pipeline_type` is only the fallback for direct evaluator calls that
do not supply a resolved mode. See §6 for optimizer-facing method selection.

### 6 · `algo_search_space`

The RAG-algorithm dimensions: the corpus chunker and the pipeline. The
**pipeline shape depends on the backend**.

```yaml
algo_search_space:
  corpus:                               # OPTIONAL — only when sweeping chunking
    chunker:
      chunk_size: [128, 256, 512, 1024, 2048]
      component: [character, recursivecharacter]   # any registered chunker (token,
                                                   # sentence, sentencewindow, …)
  pipeline:
    # static_gt → {node_lines: [...]}                      (sequential only), or
    #             {rag_dataflow: [...], node_lines: [...]}  (sequential+react co-mode)
    #   rag_dataflow entries may carry the branch's SUB-CONFIG; the declared
    #   hierarchy is wired into Ax dependents by the search-space builder.
    #   When react is reachable, its branch MUST declare max_iter — the react
    #   loop's only round cap:
    #     rag_dataflow:
    #     - sequential
    #     - react:
    #         max_iter: [3, 5]   # REQUIRED on the react branch
    # flashrag  → {methods: [{mode, node_lines, ...}, ...]}
    node_lines:
    - node_line_name: retrieve_node_line
      nodes:
      - stage: semantic_retrieval
        top_k: [1, 2, 4, 8, 16]         # static_gt: must be powers of 2 (RAGO cost model)
        modules:
        - component: vectordb
          # SCALAR reference — one pinned index; query-time knobs sit flat:
          #   vectordb: faiss_ivf_index
          #   nprobe: [8, 16, 32, 64]
          # DICT-FORM selector — the index FAMILY is a search dim; each
          # family's query-time knob is declared INSIDE its family (ownership
          # by nesting, compiled into Ax gating: the non-chosen family's block
          # params + query knob are inactive per trial, and the resolver drops
          # the unreferenced block from the runtime config):
          vectordb:
            faiss_ivf_index:
              nprobe: [8, 16, 32, 64]     # IVF query-time
            faiss_hnsw_index:
              ef_search: [32, 64, 128]    # HNSW query-time
      - stage: passage_reranker
        optional: true                  # → the optimizer gets an enable/disable dim
        top_k: [1, 2, 4, 8]
        modules:
        - {component: sentence_transformer_reranker, bits: f32}
        - {component: flag_embedding_reranker, bits: f32}
    - node_line_name: post_retrieve_node_line
      nodes:
      - stage: prompt_maker         # required for static_gt sequential blocks
        modules:
        - component: fstring
          prompt: "Question: {query}\n Passage: {retrieved_contents}\n Answer:"
      - stage: generator            # required (≥1 generator)
        modules:
        - component: vllm
          model: [Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct]
          temperature: {range: [0.0, 1.0]}
```

Structure & rules:

- A **pipeline** is a list of **node_lines**; each node_line has **nodes**; each
  node has **modules**. User-settable `stage` values: `query_expansion`,
  `semantic_retrieval`, `lexical_retrieval`, `hybrid_retrieval`,
  `passage_reranker`, `passage_filter`, `passage_compressor`, `prompt_maker`,
  `generator`. Input aliases `dense_retrieval` and `sparse_retrieval` normalize
  to `semantic_retrieval` and `lexical_retrieval` before optimization.
- **Sequential hybrid retrieval** uses one `semantic_retrieval` / `vectordb`
  node, one `lexical_retrieval` / `bm25` node, then one `hybrid_retrieval` /
  `hybrid_rrf` node. All three are required, non-optional nodes; downstream
  passage processing, prompting, and generation follow fusion. Each node's
  `top_k` is an independent positive-integer search parameter. The module
  parameter `hybrid_rrf.weight` is the positive RRF rank constant (default
  `60`), not a dense/lexical mixture weight. `rrf_k` remains a deprecated
  alias and cannot be swept alongside `weight`. ReAct, iterative hybrid,
  multiple retrieval branches, and `hybrid_cc` are rejected explicitly.
  This configuration requires `static_gt` and
  `system.performance_source: measured`: BM25 has no analytical performance
  model, and the hybrid dense/lexical join has no validated analytical
  scheduling/cost binding. The existing local dense RRF model does not provide
  that join policy. Workflow planning still describes lexical retrieval and fusion when
  CM support is unavailable; missing cost is never treated as zero. See
  `configs/rag_stack/hybrid_sequential.yaml` and
  [the hybrid retrieval guide](hybrid_retrieval.md).
- **`optional: true`** on a node turns it into an enable/disable search dim
  (the optimizer may skip it).
- **Module choice within a node** is itself a search dim: listing multiple
  `modules` makes the optimizer pick one, and that module's own list-valued
  params are searched conditionally (hierarchy via Ax `dependents` — inactive
  branches cost no search budget).
- **vectordb reference** is by `name` into `algo_search_space.vectordb`
  (§6a). A scalar pins one index. Sweeping the index FAMILY uses the
  **dict-form selector** `vectordb: {<family>: {<its query-time knobs>}}` —
  compiled into real Ax gating: the selector gates each family's block params
  and its nested query knob, so the non-chosen family is inactive per trial
  (clean GP inputs, no eval budget on inert knobs) and the resolver drops the
  unreferenced block from the runtime config. A flat multi-family list
  (`vectordb: [fam1, fam2]` + flat `nprobe:`/`ef_search:`) is **rejected** —
  it leaves the other family's knobs as always-active phantom dims.
- **Query-time retrieval knobs live on the module** (never in the vectordb
  block): flat next to a scalar reference, nested per family in the dict-form
  selector. Declaring `nprobe`/`ef_search` inside a vectordb block is not
  rejected, but it silently becomes part of the index **build signature** —
  the same on-disk index would be rebuilt per value instead of served as-is.
  `system.retrieval.faiss_ivf_parallel_mode` is IVF-only at runtime but
  stays an always-active **system** dim — algo selectors never gate
  system-space knobs (layering); HNSW simply ignores it.
- **Legacy native generation:** native `vllm` modules default to
  `generation_protocol: chat_completions`. To reproduce runs that used raw-string
  generation, select `generation_protocol: legacy_completion` under `static_gt`
  with `performance_source: cost_model`. Raw strings then use `LLM.generate`;
  structured messages still use `LLM.chat`. Omit `use_chat_template` or set it
  to `false` in this mode. Apply the protocol to both final-generator and
  query-expansion modules. API and measured execution reject the legacy
  protocol, and quality/fingerprint identities separate it from chat.
- **static_gt requirements** (enforced by the validator): ≥1 `generator`, ≥1
  retrieval node, retrieval/reranker `top_k` a power of 2 when using
  `performance_source: cost_model` (RAGO constraint), and ≥1
  `prompt_maker` for sequential blocks — a react block needs none (it builds
  its own agentic prompt). flashrag drops the `prompt_maker` and power-of-two
  requirements.
- **flashrag pipeline** is `{methods: [...]}`; each method block has a `mode`
  (`sequential`, `corag`, `ircot`, `flare`, …) plus method-specific knobs
  (`max_iter`, `threshold`, `look_ahead_steps`, …). The optimizer's
  `pipeline.mode` choice activates exactly one method's subtree per trial.
  Supported modes are maintained by the evaluator and exposed through
  `rag_stack.evaluation.registry.supported_pipeline_modes("flashrag")`; the host
  forwards one resolved configuration to the evaluator, not the search space.

### 6a · `algo_search_space.vectordb`

A **list** of index definitions, inside `algo_search_space` (the vectordb
section is algo search space — index family/build knobs move retrieval
quality). The retrieval module references one (scalar) or several (dict-form
selector, §6) of these by `name`; at resolve time the referenced, scalarified
block(s) are re-exposed at runtime `config["vectordb"]` for the evaluators,
and unreferenced blocks are dropped.

```yaml
algo_search_space:
  vectordb:
  - name: faiss_ivf_index
    db_type: faiss_ivf
    embedding_model: huggingface_all_mpnet_base_v2   # may be a SWEEP LIST;
    collection_name: my_index                        #   embedding_dim is derived per model
    nlist_factor: [1, 2, 4, 8]        # SHARED: nlist = clamp(round(factor·sqrt(N)), 1, N); N = per-eval chunk count
    index_type:                       # SEARCH DIM (hierarchical): optimizer picks pq or flat;
      pq:                             #   each type's exclusive knobs are active ONLY when chosen.
        M: [8, 16, 32, 48, 96]        #   PQ sub-quantizers (must divide EVERY candidate model's dim)
        nbits: [4, 8]                 #   bits per code
        # dsub: [4, 8]                #   ALTERNATIVE to raw M: target sub-vector dim D/M;
        #                             #   M is derived per model (a divisor of D, capped at 64)
      flat: {}                        #   IVF-Flat: exact full vectors, no quantization knobs
    # FAISS roofline calibration resolves AUTOMATICALLY and exclusively from
    # system.cpu.hardware_key: calibration_profiles/faiss_ivf/<cpu>.json.
  - name: faiss_hnsw_index
    db_type: faiss_hnsw
    embedding_model: huggingface_all_mpnet_base_v2
    collection_name: my_hnsw_index
    M: [16, 32, 48]                   # build-time graph degree
    ef_construction: [100, 200, 400]  # build-time
```

Rules:

- **`name` / `db_type` / `embedding_model` are required.** `faiss_ivf` and
  `faiss_hnsw` are the cost-modeled families; other runtime `db_type`s
  (chroma, milvus, …) exist but have no cost-model binding, so CM mode
  rejects them.
- **`embedding_model` is a legal search dim** (list = the optimizer picks the
  encoder per trial). **`embedding_dim` is derived** from the model for the
  known fixed-dim local models — omit it there (an explicit value that
  conflicts with the derived dim is rejected). Only configurable-dim models
  (openai/mock) still require an explicit `embedding_dim`.
- **Only build-time params live here.** Query-time knobs (`nprobe`,
  `ef_search`) go on the **retrieval module** (§6), so the same on-disk index
  serves every value without a rebuild. When the module sweeps the index
  FAMILY, each query knob is declared **inside its family** in the dict-form
  selector — ownership by nesting, compiled into Ax gating.
- **No `path` for static-GT FAISS stores.** Their indexes are content-addressed
  (chunk-hash + index params) and live in ONE shared global cache
  (`<repo>/.cache/rag_stack/faiss`, override `RAG_STACK_CACHE_DIR`), reused
  across every run/seed — the resolver ignores any per-project `path`, so omit
  it. (Non-faiss stores like Chroma still declare their project-local `path`.)
  FlashRAG uses its own project-local materialization and index lifecycle; see
  the [FlashRAG guide](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/docs/flashrag.md#cache-and-artifact-layout).
- `parallel_mode` is **not** allowed here — it is a runtime threading knob
  (`system.retrieval.faiss_ivf_parallel_mode`), not an index-build parameter.
- `db_type: faiss_ivf` covers **both** IVF index types via a **nested
  `index_type`** block: each key (`pq` / `flat`) holds that type's exclusive
  knobs (`pq.M` or `pq.dsub`, `pq.nbits` for IVF-PQ; IVF-Flat has none). The
  optimizer chooses among the keys, and — because it's a hierarchical
  (dependent) search dim — only the chosen type's knobs are active per trial
  (Ax never suggests `M`/`nbits` for a flat arm). Shared knobs
  (`nlist_factor`, …) stay at the vectordb level. A single key pins that
  type; a bare scalar `index_type: pq` also works (then `M`/`nbits` sit at
  top level).
- For IVF, `nlist_factor` expresses `nlist` as a factor of `sqrt(N)` so it
  scales with the per-eval chunk count and can never exceed `N` (untrainable
  k-means). A fixed `nlist: [...]` list is also accepted (see the measured
  baselines). Applies to both `index_type: pq` and `flat`.
- `${PROJECT_DIR}` (and any `${ENV_VAR}`) is interpolated at load time;
  `PROJECT_DIR` is set to the project root before the config is read.

---

## 4. Performance source & system-space owner

Two ORTHOGONAL keys (both under `system:`):

- **`performance_source`** (omit → `cost_model`) — WHERE performance numbers
  come from: `cost_model` (analytical RAGO/GenZ/FAISS models, no GPUs needed)
  or `measured` (real vLLM deployment, timed; needs
  `global.eval_backend: static_gt`).
- **`system_space_owner`** (omit → derived: `cost_model` → `cost_model`,
  `measured` → `optimizer`) — WHO searches the deployment design space:
  - `cost_model`: no optimizer-facing deployment dims; the CM sweeps every
    deployment internally per trial and selects the best (the decoupled
    search). Uses `system.cm_search_space`.
  - `optimizer`: the optimizer samples deployment dims from
    `system.system_design_space`; exactly ONE deployment is priced
    (cost_model) or launched (measured) per trial.

Legal matrix: `(cost_model, cost_model)` = decoupled CM search;
`(optimizer, measured)` = classic measured mode; `(optimizer, cost_model)` =
**joint-space search priced by the CM** (e.g. the RQ1 baseline);
`(cost_model, measured)` = rejected (a measured run cannot realize a sweep).

Under `system_space_owner: optimizer` the design space needs the real
**`available_gpus`** list (it lives INSIDE `system_design_space`):

```yaml
system:
  performance_source: measured        # or cost_model (joint-space CM pricing)
  system_design_space:
    batch_size_request: [1, 8, 16, 32]   # request cohort for retrieval/prefill/rerank
    batch_size_decode: [8, 16, 32, 64]   # generator decode continuous-batch cap
    dynamic_batch_timeout_s: [0.002, 0.01, 0.05]  # non-vLLM dynamic batching wait
    vllm_kv_cache_dtype: auto
    available_gpus: [cuda:0, cuda:1, cuda:2, cuda:3]
```

`cm_search_space.{min,max}_num_gpus` / `max_num_gpus_per_stage` still bound
the per-trial GPU layout. `batch_size_request` and `vllm_max_num_seqs` are
mutually exclusive (the global batch drives the vLLM concurrency knob).

The GPU layout itself is driven by ONE of two mutually exclusive mechanisms:

1. **`gpu_layout` categorical** (default): auto-synthesized from
   `cm_search_space` bounds × `available_gpus`; per-engine GPU counts,
   collocation cuts, TP/PP, `gpu_memory_utilization` and serving mode are
   **derived from the partition, not searched**. Listing derived knobs
   (`vllm_gpu_memory_utilization`, `tensor_parallel_vllm`, `serving_mode`, …)
   in `system_design_space` is silently ignored (warned); per-stage batch dims
   (`batch_size_generator`, …) are a hard error. `batch_size_decode` is the
   only separate batch-like knob (decode-engine continuous batching);
   `dynamic_batch_timeout_s` is a global count-or-timeout wait for non-vLLM
   batched stages.
2. **`deployment_preset` categorical**: named practitioner templates declared
   under `system.deployment_presets` and listed in
   `system_design_space.deployment_preset` (suppresses `gpu_layout`; raw
   `cut_after_*`/`num_gpus_*`/`pipeline_parallel_*` dims are rejected
   alongside it):

   ```yaml
   system:
     deployment_presets:
       collocated_tp4: {placement: collocated, num_gpus: {generator: 4}}
       disagg_2p2d:    {placement: pd_disagg,
                        num_gpus: {generator_prefill: 2, generator_decode: 2}}
       disagg_1p2d:    {placement: pd_disagg,        # asymmetric PD (role-qualified)
                        num_gpus: {generator_prefill: 1, generator_decode: 2}}
     system_design_space:
       deployment_preset: [collocated_tp4, disagg_2p2d, disagg_1p2d]
       batch_size_request: [16, 64, 256]
       available_gpus: [cuda:0, cuda:1, cuda:2, cuda:3]
   ```

   `placement` ∈ `collocated` (every GPU stage in one resource group) |
   `pd_disagg` (decode split off after `generator_prefill`) |
   `disaggregated` (each sized stage its own group). Under `pd_disagg`,
   auxiliary stages ride the prefill-side group at 1 slot unless sized — an
   optional `aux: ride | left` key controls that placement. `num_gpus` keys
   are engine names or role-qualified stage names (asymmetric PD); values
   must be **powers of two**. TP is derived (`chips / pipeline_parallel`). A
   preset that cannot fit the trial's pipeline shape within the
   `cm_search_space` bounds penalizes that trial (never silently projected).

Quality is deployment-independent, so trials that differ only in system dims
reuse the first trial's judge scores + recorded trace (quality memoization;
every GT eval dir carries a `quality_memo.json` marker whose `hit` flag says
whether it was reused).

---

## 5. Syntax cheat-sheet

| Want | Write |
|------|-------|
| Fixed value | scalar: `top_k: 8` |
| Discrete choice | list: `top_k: [1, 2, 4, 8]` |
| Continuous range | `temperature: {range: [0.0, 1.0]}` |
| Ordered categorical | `model: {ordered: [small, medium, large]}` |
| Disable-able stage | `optional: true` on the node |
| Module choice | multiple entries under `modules:` |
| Choose index family | dict-form selector: `vectordb: {ivf_name: {nprobe: [...]}, hnsw_name: {ef_search: [...]}}` |
| Env / project path | `${PROJECT_DIR}/data/qa.parquet`, `${ANY_ENV_VAR}` |
| Quality scalar | `eval_backend_setting.combined_quality` |
| Stage diagnosis for the agent | per-metric `description` (feeds auto-derived `sub_metrics`) |

---

## 6. Validation & failure modes

`ConfigValidator(config).validate()` runs before optimization and surfaces all
errors together. Common rejections:

- Any legacy top-level key (`data`, `pipeline`, `node_lines`, `corpus`,
  `vectordb`, `gt_evaluation`, `n_iterations`) → migration error pointing at
  the 6-section location.
- `algo_search_space.pipeline` shape mismatched to the backend
  (`node_lines` for static_gt, `methods` for flashrag), or a react branch
  without `max_iter`.
- static_gt missing a `generator` / retrieval node, a sequential block
  missing `prompt_maker`, or a non-power-of-two `top_k`.
- `M` not dividing a candidate embedding dim (IVF-PQ); an `embedding_dim`
  that conflicts with the model-derived dim; a missing `embedding_dim` for a
  configurable-dim model.
- Unknown metric name for the backend.
- A module with no cost-model mapping (CM mode).
- `system_space_owner: optimizer` (e.g. measured mode) without
  `system_design_space` / its `available_gpus`.
- A flat multi-family `vectordb: [fam1, fam2]` module reference (use the
  dict-form selector).

By default the validator also `snapshot_download`s any configured HuggingFace
model that is not cached locally (`check_model_cache=True`).

---

## 7. Minimal static_gt example

Adapt the data paths and hardware to your target. The default calibration mode
is `strict`, so the exact component/workload profiles required by the selected
candidates must exist under `calibration_profiles` before evaluation.

```yaml
global:
  eval_backend: static_gt
  n_iterations: 30
dataset:
  dataset_name: dragonball_en
  qa: datasets/dragonball/qa_en_sample_100.parquet
  corpus: datasets/dragonball/raw_corpus_en.parquet
system:
  cpu: epyc_9124                     # Replace with your target's hardware identity.
  gpu: A100_80GB_GPU
  performance_objectives: { selection: max_throughput }
  cm_search_space:
    placement_policy: [disaggregated, collocated]
    min_num_gpus: 2
    max_num_gpus: 4
    max_num_gpus_per_stage: 4
    gpus_per_server: 4
    max_batch_size_request: 256
    no_microbatching: true
optimizer:
  type: agent_qnehvi
  params: { seed: 42, n_both_init: 10 }
eval_backend_setting:
  combined_quality: { mode: mean, metrics: [deepeval_answer_correctness] }
  metrics:
  - { metric_name: deepeval_answer_correctness, model: openai/deepseek/deepseek-v4-flash,
      description: GEval factual correctness vs gold answer. }
algo_search_space:
  vectordb:
  - name: faiss_ivf_index
    db_type: faiss_ivf
    embedding_model: huggingface_all_mpnet_base_v2   # embedding_dim derived (768)
    collection_name: my_index
    nlist_factor: [1, 2, 4, 8]        # shared
    index_type:                       # hierarchical search dim: pq vs flat
      pq:
        M: [8, 16, 32, 48, 96]
        nbits: [4, 8]
      flat: {}
  pipeline:
    node_lines:
    - node_line_name: retrieve_node_line
      nodes:
      - stage: semantic_retrieval
        top_k: [4, 8, 16]
        modules:
        - { component: vectordb, vectordb: faiss_ivf_index, nprobe: [8, 16, 32, 64] }
    - node_line_name: post_retrieve_node_line
      nodes:
      - stage: prompt_maker
        modules:
        - { component: fstring, prompt: "Question: {query}\n Passage: {retrieved_contents}\n Answer:" }
      - stage: generator
        modules:
        - { component: vllm, model: [Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct],
            temperature: {range: [0.0, 1.0]} }
```

---

See also: [`configs/rag_stack/config_reference.yaml`](https://github.com/haiqiang-zhang/rag-stack/blob/main/configs/rag_stack/config_reference.yaml)
(annotated canonical example), and the measured baselines under
[`configs/baselines/`](https://github.com/haiqiang-zhang/rag-stack/tree/main/configs/baselines).
