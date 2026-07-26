---
layout: home
title: RAG-Stack Documentation
nav_order: 1
permalink: /
---

# RAG-Stack Documentation

Everything beyond the minimal installation and first-run command in the
[main README](https://github.com/haiqiang-zhang/rag-stack#readme).

- **[User guide](user_guide.md)** — the top-level entry point for using
  RAG-Stack via the `rag_stack` command: launching a run from a config,
  monitoring, resuming / recovering an existing project dir, quality stores &
  hardware replay, system transfer, and the system settings + cache-directory
  layout every run inherits.
- **[Evaluator setup and migration](evaluators.md)** — install the pinned
  static evaluator and optional top-level FlashRAG provider, update existing installs,
  and migrate evaluator imports and runtime scripts.
- **[Evaluator Extension Guide](evaluator_extensibility.md)** — register
  an external backend, capture RAG-IR with host-owned hooks, and validate on CPU.
- **[Evaluator API Reference](evaluator_api.md)** — result contracts, registration,
  tracing semantics, and validation rules for every backend.
- **[Evaluator input contract](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/docs/input_contract.md)** —
  resolved algorithm/deployment dictionaries and external QA/corpus inputs.
- **[Replay a configuration](replay.md)** — two JSON files plus explicit dataset
  arguments for quality, measured performance or fresh-trace CM replay.
- **[Project CM preview](cm_preview.md)** — execute a small query cohort and
  inspect the real compiled physical path, source-call mappings and blockers.
- **[FlashRAG integration](flashrag.md)** — install the independent framework
  provider and trace its native pipeline through the shared evaluator hooks.
- **[Evaluate one saved configuration](eval_one.md)** — measure a resolved
  pipeline and system configuration on real hardware.
- **[Experimental functions](experimental_functions.md)** — how a trial gets
  its two objective scores: the three evaluation switches (`eval_backend` /
  `performance_source` / `rag_ir_mode`), the sequential+react co-mode, the
  frozen CM input contract (quality trace + workflow schema), and the valid
  combinations.
- **[Config schema](config_schema.md)** — concepts and syntax rules
  for the 6-section run config (`global` / `dataset` / `system` / `optimizer`
  / `eval_backend_setting` / `algo_search_space`).
- **[Sequential hybrid retrieval](hybrid_retrieval.md)** — configure dense,
  BM25, and RRF in sequence, inspect recorded calls, and check runtime capabilities.
- **[Annotated config reference](https://github.com/haiqiang-zhang/rag-stack/blob/main/configs/rag_stack/config_reference.yaml)**
  — every config key, annotated inline.
- **[Cost model (RAG-CM)](rag_cm/README.md)** — the CM's public entry points
  (`evaluate_dynamic` / `evaluate_static` / `evaluate_fixed`), the frozen
  input contract (canonical quality trace + workflow schema + corpus stats),
  fixed-mode testing, and legacy-trace conversion.
  - **[Architecture and workload representation](rag_cm/architecture.md)** —
    schema and trace responsibilities, analytical shortlisting, simulation,
    and configured batch choices.
  - **[Compiled path contract](rag_cm/compiled_plan.md)** — physical compilation,
    source-call mappings, in-memory execution and saved preview descriptions.
  - **[Calibration and hardware migration](rag_cm/calibration.md)** — staged
    readiness inventory; the CPU-system, workload, GPU-component, runtime, and
    retrieval-worker profile layers; forbidden E2E/assembly fitting; and the
    current H100 fixed-point regression guard.
  - **[FAISS IVF cost model](rag_cm/faiss_ivf_cost_model.md)** — the
    roofline-based vector-search cost model (IVF-PQ + IVF-Flat): quick
    usage, the simulator-benchmark CLI, calibration, and index-string
    anatomy.
- **[Optimizer benchmark](optimizer.md)** — racing the registered optimizers
  on the historical-surrogate stand-in: the matrix-sweep YAML, output layout,
  and the reference Pareto cache (`pareto-cache` / `recompute-metrics`).
- **[Web app](apps.md)** — setup, dev servers, and configuration for the
  `apps/` SPA + its FastAPI backend (`rag_stack/server/`).
