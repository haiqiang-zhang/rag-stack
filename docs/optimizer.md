---
title: Optimizer Benchmark
nav_order: 6
---

# Optimizer Benchmark

Compare the registered optimizers against a historical surrogate of a RAG
optimization run. The benchmark predicts quality from previously measured
sub-metrics and does not run a RAG pipeline or a quality judge for each
candidate. CPU fitting is the default. Agent-based optimizers can still call
their configured LLM provider for search guidance. Runtime depends on the
optimizer and the selected performance mode.

- **Optimizer types** (`OPTIMIZER_REGISTRY` in
  [optimizer/\_\_init\_\_.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/optimizer/__init__.py)): `sobol`,
  `greedy`, `ax_qnehvi`, `ax_qnehvi_mixedgp`, `smac_mobo`, `agent_qnehvi`,
  `agent_qnehvi_slo`, `greedy_then_hv_anchor_agent_qnehvi` (what each one is:
  [config_schema.md §4](config_schema.md#4-optimizer)).
- **Benchmark function**: exactly one is registered —
  `rag_stack_historical_surrogate`
  ([synthetic_functions/rag_stack_historical_surrogate.py](https://github.com/haiqiang-zhang/rag-stack/blob/main/rag_stack/optimizer/benchmark/synthetic_functions/rag_stack_historical_surrogate.py)).

## The benchmark function — `rag_stack_historical_surrogate`

Exposes the **real RAG search space** (parsed from a rag-stack YAML config via
the same `RagSearchSpace` / `PerformanceContext` the live Controller uses) and
scores each suggested config with:

- **Quality — GP surrogate on real historical data.** One independent GP per
  sub-metric (`MixedSingleTaskGP` when the space has unordered categorical
  dims), trained on `(config, sub-metric)` rows pooled from finished optimizer
  runs' `all_evaluations.csv`. The default pool is the four runs under
  `benchmarks/historical_surrogate/`
  (approximately 195 rows; one shared corpus, judge, and search space, so pooling is a
  clean coverage gain). Each query returns the posterior **mean** (so the
  function is deterministic by default), then folds the sub-metrics into the
  scalar `combined` via the same host-side `aggregate_quality_metrics` used
  by the live controller.
- **Performance — cost model by default.** With `performance_mode: cost_model`,
  call `RAGCMAssembly.evaluate_static` on the resolved pipeline using the
  YAML's hardware and installed calibration profiles. With
  `performance_mode: historical_surrogate`, predict the historical
  `performance` column with a gradient-boosted model instead. The latter
  avoids cost-model evaluation for each candidate and is intended for
  optimizer-method studies.

In cost-model mode, performance uses the same assembly path as a live
CM-backed run. Quality remains a fitted prediction: benchmark results do not
replace fresh evaluator measurements. In historical-surrogate mode, both
objectives are predictions. Compare hypervolume only with matching objective
scales, aggregation rules, benchmark parameters, and reference points.

The default historical pool is not included in this distribution. Set
`historical_eval_dirs` to compatible local runs and provide their shared
`data/qa.parquet` and `data/corpus.parquet` inputs.

The host retains this optimizer benchmark and quality aggregation; evaluator
implementations and metrics are maintained in the `RAG-Stack-Evaluator`
submodule. Install the host and its pinned evaluator together using the
[evaluator installation guide](evaluators.md).

Constructor `params` accepted in the sweep YAML:

| Param | Default | Meaning |
|---|---|---|
| `yaml_path` | required | rag-stack YAML — defines search space, cost-model hardware, and the `combined_quality` aggregation rule |
| `historical_eval_dirs` | 4-run pool | project dirs whose `all_evaluations.csv` to pool; shared `data/qa.parquet` + `data/corpus.parquet` must sit next to the run dirs |
| `sub_metrics` | 7-metric set (deepeval × 5 + retrieval recall/ndcg) | sub-metric columns the surrogate models |
| `stochastic` / `noise_scale` / `rng_seed` | `false` / `1.0` / — | inject GP-posterior-σ-scaled noise per evaluate (reproducible per `rng_seed`) so optimizers can be discriminated on noisy landscapes |
| `surrogate_difficulty` | `0.0` | 0–1 blend weight of a per-metric GBM layer that restores the local ruggedness (categorical phase transitions, threshold effects) the GP smooths away |
| `use_gpu` | `false` | fit/query the quality GPs on CUDA when available |
| `performance_mode` | `cost_model` | choose live CM predictions or `historical_surrogate` predictions fitted from past performance values |
| `invalid_objective_penalty` | `false` | turn cost-model evaluation exceptions into the configured penalty objectives; resolution errors still propagate |
| `invalid_performance` / `invalid_quality` | `0.0` / `0.0` | penalty scores when enabled; quality is clamped to `[0, 1]` |

The predictive models are refit on construction. Fitted models are not cached
to disk; host dataset and cost-model scratch artifacts can be reused under
`tmp_outputs/_surrogate_scratch/historical_surrogate/`.

## Matrix-sweep YAML (optimizers × function variants × seeds)

One config runs the full cartesian product, writes one CSV per combination,
and (unless opted out via `defaults.auto_analyze: false`) auto-generates the
comparison plots:

```bash
python -m rag_stack benchmark \
    --config configs/optimizer_benchmark/sweep_agent_historical_surrogate.yaml
# (--plot exists but only affects the legacy single-run config shape;
#  matrix sweeps plot via defaults.auto_analyze)
```

Schema (abridged from
[sweep_agent_historical_surrogate.yaml](https://github.com/haiqiang-zhang/rag-stack/blob/main/configs/optimizer_benchmark/sweep_agent_historical_surrogate.yaml)):

```yaml
defaults:
  gt_budget: 50                 # total GT evals per (optimizer, seed)
  checkpoint_interval: 5
  output_dir: tmp_outputs/optimizer_benchmark_results/agent_historical
  seeds: [42, 43, 44]
  auto_analyze: true            # default true; set false to keep CSV-only

optimizers:
  - type: ax_qnehvi_mixedgp
    name: ax_mixedgp_golden     # REQUIRED per-experiment id — CSV prefix +
    params: {n_both_init: 10}   #   experiments.yaml key; enables same-type ablations

  - type: agent_qnehvi
    name: agent_mo_inject
    params:
      n_both_init: 10
      # Agent proposals and all search mechanisms are enabled by default.
      # See config_schema.md for constructor-level mechanism switches.
    gt_budget: 30               # per-optimizer override

synthetic_functions:
  - type: rag_stack_historical_surrogate
    name: rag_historical_surrogate   # = subfolder + pareto-cache key
    params:
      # yaml_path must define a search space the pooled historical CSV rows
      # decode under (rows that fail to encode are silently skipped)
      yaml_path: configs/rag_stack/config_reference.yaml
      # historical_eval_dirs omitted → default 4-run pool under
      # benchmarks/historical_surrogate/
      sub_metrics: [deepeval_answer_correctness, deepeval_answer_relevancy]
      performance_mode: cost_model

# Optional: drop specific (optimizer, function) combinations
# skip:
#   - optimizer: ax_qnehvi_mixedgp
#     synthetic_function: rag_historical_surrogate
```

`synthetic_functions` stays a list even though only one function class
exists — different `params` variants of the surrogate (e.g. another
`yaml_path`, a `surrogate_difficulty` ablation) are separate entries with
distinct `name`s, and each gets its own subfolder + Pareto cache.

A legacy single-run shape is also accepted: one `optimizer:` block + one
`synthetic_function:` block + top-level `gt_budget` / `seed`.

Key points:

- The Agent may use its first response to request targeted evidence from the
  complete evaluation ledger before proposing candidates. This read-only
  follow-up is always available, limited to one additional LLM request, and
  has no configuration switch; qLogNEHVI still decides which candidate is
  evaluated.
- Every post-initialization optimization iteration consults the Agent exactly
  once. A stagnation-triggered forced iteration uses the specialized
  maximum-gap Agent call instead of an additional ordinary call; there is no
  cadence or warmup parameter.
- The single forced safety channel becomes due after three consecutive
  ordinary evaluations add no realized normalized hypervolume. It freezes that
  episode's objective normalization, joins the gap-directed Agent proposals
  with the same round's DOE OAT/pair cells, and selects one legal candidate by
  exact posterior expected coverage of the largest empty Pareto rectangle.
  Sources have no quota or weight. A valid forced evaluation resets the
  episode; an invalid or unscored attempt is blacklisted without consuming it.
- Each run produces `{optimizer_name}__{function_name}.csv` under
  `<output_dir>/<function_name>/` — the **seed is a column** inside the CSV
  (all seeds of one experiment accumulate in the same file), not part of the
  filename. `optimizer_name` is the required `name` field.
- Per-optimizer override keys (`gt_budget` / `checkpoint_interval` / `output_dir`) win over `defaults`.
- Seeds live only in `defaults.seeds` — the matrix stays rectangular.
- Failures are isolated per run: a failing optimizer logs a traceback and the sweep continues; final summary reports `N/N succeeded`.
- Validation is fail-fast: unknown `type`, duplicate `name`, missing `name`, or unrecognized keys at the optimizer level raise before any run starts.

## Re-plot only (CSVs already exist)

```bash
python -m rag_stack.optimizer.benchmark.analyze tmp_outputs/optimizer_benchmark_results/
```

## Reference Pareto cache

The historical surrogate has no analytical true Pareto, so the reference used
to compute `hv_diff` and `igd` (regret metrics, lower = better) is stored
**per-benchmark-name** alongside the run outputs — seeded with NSGA-II on the
very first build, then refreshed from the pooled evaluation trails:

```
<output_dir>/<name>/
    experiments.yaml              ← function config snapshot + per-optimizer entries
    _pareto_cache.npy              ← frozen reference (n_pareto × n_obj)
    <optimizer>__<name>.csv        ← per-iteration snapshot (seed = a column)
    <optimizer>__<name>__evals.csv ← per-eval ground-truth trail
```

`<name>` is the YAML's `synthetic_functions[].name` field — different params
under the same function class get different `name`s and therefore separate
caches. `BenchmarkRunner` lazily loads the frozen cache (falling back to
`compute_true_pareto` on a miss); at run time the snapshot CSVs record the
live `hv` only — `hv_diff` / `igd` stay `nan` until the backfill below fills
them against the cached reference.

### Initial build (brand-new benchmark, no eval CSVs yet)

Use `--no-pool` for the very first build before any sweep has run — there's
nothing to pool from, so the reference is computed with pymoo NSGA-II on the
surrogate (~1–2 min per function):

```bash
python -m rag_stack pareto-cache --name rag_historical_surrogate --no-pool
```

`--name` is repeatable to build several benchmarks in one call.

### Refresh after running new benchmark sweeps (default mode)

After a sweep finishes (new optimizers / seeds / runs producing fresh
`*__<name>__evals.csv` files in the subfolder), the **default** invocation
rebuilds the reference from those evals **and** auto-backfills `hv_diff` /
`igd` in every snapshot CSV so historical chart numbers match the new cache:

```bash
python -m rag_stack pareto-cache --name rag_historical_surrogate
```

The eval CSVs to pool are read from `<output_dir>/<name>/*__<name>__evals.csv`
— `--name` and `--output-dir` together fully locate them, no separate
`--evals-dir` to set.

The default output dir comes from `.rag_stack_settings.json`'s
`optimizer_benchmark_dir` field (see
[System settings](user_guide.md#system-settings)) (or
`tmp_outputs/optimizer_benchmark_results` when empty / file missing) and is
shared by all three subcommands:

```jsonc
// .rag_stack_settings.json — sticky default, applies to all benchmark subcommands
{
  "optimizer_benchmark_dir": "/my/results/root",
  ...
}
```

To override per call, pass `--output-dir /my/results/root` to either
`pareto-cache` or `recompute-metrics`.

End to end, per `--name`:

1. Reads `<output_dir>/<name>/experiments.yaml` to get the function `type` +
   `params`, instantiates it.
2. Pools every `*__<name>__evals.csv` in `<output_dir>/<name>/`,
   non-dominates the pooled points, and writes them as the new
   `<output_dir>/<name>/_pareto_cache.npy` — in this default mode the pool
   **replaces** the previous reference; NSGA-II runs only under `--no-pool`.
3. Calls `recompute_metrics_from_evals` on the same subfolder to backfill
   historical snapshot CSVs in place (originals saved to
   `<file>.bak_pre_recompute`).

### Backfill only (cache already correct)

If the cache is fine but historical snapshot CSVs are stale relative to it
(e.g. after manually editing the `.npy`), run the backfill standalone — also
`--name`-keyed:

```bash
python -m rag_stack recompute-metrics --name rag_historical_surrogate

# preview without writing:
python -m rag_stack recompute-metrics --name rag_historical_surrogate --dry-run
# skip the .bak_pre_recompute backups:
python -m rag_stack recompute-metrics --name rag_historical_surrogate --no-backup
```

Notes:

- The default refresh is **pool-only**: once real evaluation trails exist they
  ARE the reference; NSGA-II is only the bootstrap for a brand-new benchmark
  (`--no-pool`).
- The pool is scoped to a single `<name>` subfolder, so different function
  configs never cross-mix as long as they each get a distinct `name`.
- To rebuild from scratch, delete the per-name subfolder's cache
  (`rm <output_dir>/<name>/_pareto_cache.npy`) and rerun with `--no-pool`.
