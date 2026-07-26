---
title: Experimental Functions
nav_order: 3
---

# Experimental Functions — how a trial is scored

Every evaluated config gets two objective scores — **quality** (ground-truth
metrics on a QA dataset) and **performance** (latency / throughput) — computed
by deliberately decoupled subsystems, so quality can be evaluated once and
replayed against different hardware (see
[Quality store and hardware replay](user_guide.md#quality-store-and-hardware-replay)).

Which subsystem computes each objective is selected by **three orthogonal YAML
switches**:

| Axis | YAML key | Values | Default |
|------|----------|--------|---------|
| 1. Quality backend | `global.eval_backend` | `static_gt` \| `flashrag` | `static_gt` |
| 2. Performance source | `system.performance_source` | `cost_model` \| `measured` | `cost_model` |
| 3. rag_ir mode (CM engine) | `global.rag_ir_mode` | `static` \| `dynamic` | `dynamic` |

Evaluator installation, import migration, and submodule updates are covered in
[Evaluator setup and migration](evaluators.md). The static evaluator and
FlashRAG are separate provider packages; RAG-Stack consumes their scores and
shared trace contract through the same host-owned interface.

## ① Quality backend — `global.eval_backend`

- **`static_gt`** (default) — the static evaluator supplied by the
  [RAG-Stack-Evaluator submodule](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/tree/main/rag_stack_evaluator/static_rag_evaluator).
  **Sequential + ReAct** over ONE shared `algo_search_space.pipeline.node_lines`
  (no `methods` blocks): the sequential path runs the node lines (retrieval →
  rerank/compress → generation) per query; adding the flat dataflow knob
  `pipeline.rag_dataflow` lets the optimizer pick the ReAct agentic loop per
  trial. A branch entry carries that branch's SUB-CONFIG — the declared
  hierarchy is wired into Ax `dependents` (nothing knob-specific is
  hardcoded):

  ```yaml
  pipeline:
    rag_dataflow:
    - sequential
    - react:
        max_iter: [3, 5]   # REQUIRED on the react branch: max Thought/Action
    #                      # rounds — the loop's only cap
    node_lines: [...]      # ONE shared stage list for both dataflows
  ```

  The react loop drives ONLY `semantic_retrieval` / `hybrid_retrieval`, an
  **optional** `passage_reranker`, and the `generator`
  (`REACT_ACTIVE_STAGES` in
  [rag_search_space.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/search_space/rag_search_space.py)). A
  sequential trial deactivates the react sub-knobs, AND a react trial
  deactivates every sequential-only stage (`query_expansion`,
  `passage_filter`, `passage_compressor`, `prompt_maker`) — no search
  dimensions are spent on dead knobs, and the resolver strips gated-off
  stages so the cost model never provisions them. Scores with the metric
  functions under `static_rag_evaluator/evaluation/`. Zero FlashRAG
  dependency.
- **`flashrag`** — independent FlashRAG provider in the host's top-level
  `FlashRAG/` submodule. Install it with `uv pip install -e FlashRAG` from the
  host root. Its adapter, `flashrag.integrations.rag_stack`, registers a backend
  descriptor and uses the same `Evaluator` protocol as every other provider.
  It owns data conversion, indexing, configuration translation, and native
  pipeline execution. Framework components record serving I/O through the host's
  global hooks; the adapter returns quality scores and the completed canonical
  trace for dynamic cost-model replay. The declared methods include sequential,
  iterative, and agentic pipelines; query
  `rag_stack.evaluation.supported_pipeline_modes("flashrag")` for the installed
  provider's current modes. `monitor: full` and `monitor: light` are aliases for exact capture;
  `monitor: off` is rejected because the backend requires a complete trace.
  The separate `static_gt` ReAct implementation does not need FlashRAG.
  See [FlashRAG integration](flashrag.md) for its contract and configuration.

```yaml
global:
  eval_backend: static_gt   # or: flashrag
```

## ② Performance source — `system.performance_source`

- **`cost_model`** (default) — analytical simulation for the performance
  objective. Cost-model replay itself does not launch GPU inference; a fresh
  quality evaluation may still require GPUs or a remote model API. `RAGCMAssembly` ([assembly.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/cost_model/assembly.py)) wraps
  RAGO's sweep + placement over per-stage cost models (LLM stages via GenZ,
  FAISS via the calibrated roofline model) and prices each trial through the
  static or dynamic engine ([engines.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/cost_model/engines.py)) per
  `global.rag_ir_mode` (axis ③); the selection policy
  (`system.performance_objectives.selection`, e.g. `max_throughput`) reduces
  the sweep to one score.
- **`measured`** — deploys **real vLLM engines** on
  `system.system_design_space.available_gpus` and times them
  ([measured runtime](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/tree/main/rag_stack_evaluator/static_rag_evaluator/measured)).
  GPU layout is derived, not searched; every trial tears down all vLLM
  processes to prevent GPU-memory leaks. Requires `global.eval_backend:
  static_gt`, the GPU bounds in `system.cm_search_space.{min,max}_num_gpus`,
  and a `system.system_design_space` block (with `available_gpus`) — see
  [configs/baselines/](https://github.com/haiqiang-zhang/rag-stack/tree/main/configs/baselines) for complete examples.

```yaml
system:
  performance_source: measured     # or: cost_model (default)
  cm_search_space:
    min_num_gpus: 4
    max_num_gpus: 4
    max_num_gpus_per_stage: 2
  system_design_space:
    batch_size_request: [1, 8, 16, 32]   # one global batch (no per-stage dims)
    vllm_kv_cache_dtype: auto
    available_gpus: [cuda:0, cuda:1, cuda:2, cuda:3]
```

## ③ rag_ir mode — `global.rag_ir_mode`

Only meaningful with `performance_source: cost_model`: which CM **engine**
prices a trial, and therefore what data it consumes?

- **`dynamic`** (default) — the DYNAMIC engine
  (`RAGCMAssembly.evaluate_dynamic`): replays the quality run's recorded
  **quality trace + workflow schema**. This pair is the **frozen CM input
  contract** ([rag_stack/rag_ir/](https://github.com/haiqiang-zhang/rag-stack/tree/main/rag_stack/rag_ir); full reference:
  [rag_cm/README.md](rag_cm/README.md)) — the CM accepts
  exactly these two artifacts, with no side channels:
  - The **quality trace** (canonical envelope:
    `{queries: [{question_id, calls}], provenance?}`) —
    exactly one complete ordered invocation sequence per dataset question,
    with component calls (generate / retrieve / rerank / …) and
    real token counts. Field whitelists are enforced at every layer (unknown
    keys are errors); both backends produce it. Retired envelopes are not
    accepted.
  - The [`WorkflowSchema`](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/rag_ir/workflow_schema.py), whose
    `dataflow_type` (`sequential` | `agentic`) the engine dispatches on and
    which carries the corpus statistics (`corpus_stats`: token stats +
    per-vectordb index sizes) as a first-class field.

  One trace-driven path prices sequential and react/iterative trials. It first
  uses trace-aware analytical estimates to shortlist system configurations
  according to the performance objective, then closed-loop simulation predicts
  both throughput and mean latency for retained candidates. See the
  [CM architecture guide](rag_cm/architecture.md) for the roles of schema,
  trace, and internal summary statistics.
- **`static`** — the STATIC engine (`RAGCMAssembly.evaluate_static`): offline
  `TokenStats` estimates + config-driven aggregate RAGO inputs. Works without
  ever running a quality evaluation; sequential dataflow only (an agentic
  trial has no static shape to price).

```yaml
global:
  rag_ir_mode: dynamic     # default
```

## Valid combinations

| | `performance_source: cost_model` | `performance_source: measured` |
|---|---|---|
| **`eval_backend: static_gt`** | ✅ default; CM engine per `rag_ir_mode` | ✅ measured baselines ([configs/baselines/](https://github.com/haiqiang-zhang/rag-stack/tree/main/configs/baselines)) |
| **`eval_backend: flashrag`** | ✅ always trace-replay | ❌ rejected by the config validator |

All three keys are validated up front
([config_validator.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/config_validator.py)) — unknown values fail
fast before any evaluation starts. The config follows a fixed **6-section
schema** (`global` / `dataset` / `system` / `optimizer` /
`eval_backend_setting` / `algo_search_space`) — concepts and syntax rules are
documented in [config_schema.md](config_schema.md), and the annotated
reference for every key is
[configs/rag_stack/config_reference.yaml](https://github.com/haiqiang-zhang/rag-stack/blob/main/configs/rag_stack/config_reference.yaml).
