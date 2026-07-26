---
title: Evaluate One Saved Configuration
---

# Evaluate one saved configuration

Use `eval-one` to measure one resolved RAG pipeline on real hardware. It runs
neither optimization nor the cost model. It is deliberately
**performance-only**: existing quality is preserved and never recomputed.

The command has two required inputs:

```bash
python -m rag_stack eval-one --performance-only \
  PIPELINE_CONFIG.json SYSTEM_CONFIG.json
```

`--perf-only` is a shorter alias. Performance-only is also the default, so
the original two-path command remains valid; the explicit flag is recommended
for CM-versus-measured replay scripts.

- `pipeline_config.json` is the fully resolved RAG pipeline.
- `system_config.json` is its concrete placement, topology, and batching.

`algo_config.json` is not an input. In some exported bundles it is simply a
copy of `pipeline_config.json`.

For the dataset-independent two-config contract, use [replay](replay.md) and
pass QA/corpus explicitly. `eval-one` supports legacy project artifacts that
embed dataset metadata.

## Prerequisites

Complete the [evaluator installation guide](evaluators.md), including the pinned
`RAG-Stack-Evaluator` submodule. The `eval-one` CLI remains a RAG-Stack command;
measured execution, dataset evaluation, and vLLM instrumentation are provided by
`rag_stack_evaluator.*`. After updating the host, synchronize its dependencies:

```bash
git submodule update --init --recursive
```

If dependency requirements changed, repeat the editable installation for your
selected CUDA stack. The optional top-level FlashRAG provider installation is not required: this command
uses the static measured provider. The saved pipeline must reference available
datasets and models, and its system configuration must match the allocated local
devices. Relative dataset paths are searched in the current working directory,
the repository root, and then the directory containing `pipeline_config.json`.

## Example

From the repository root, with the `rag-stack` Conda environment available:

```bash
conda run --no-capture-output -n rag-stack python -u -m rag_stack eval-one \
  --performance-only \
  /path/to/saved/pipeline_config.json \
  /path/to/saved/system_config.json
```

That is the complete command. It automatically:

- reads the QA and corpus paths from `pipeline_config.json`;
- overlays `eval_backend_setting.performance_only=true` in memory;
- converts any saved `vllm_api` component or generator backend to local vLLM
  chat in memory, without contacting the quality-stage endpoint;
- measures performance without running quality metrics, an optimizer agent,
  or an LLM judge;
- uses the full saved QA dataset rather than a smoke subset;
- uses exactly the concrete devices in `system_config.layout.available_devices`
  or `system_config.hardware.available_gpus`; when both are present, they must
  match;
- preserves the input JSON files unchanged;
- persists the measured performance execution trace;
- publishes QPS only after the measured window passes saturation and
  admissibility checks.

The saved deployment's devices and models must be available on the measurement
host. Its models use local vLLM chat serving, so the command needs no external
vLLM endpoint or quality-judge API key.

## Output

The CLI prints the result directory when it finishes. By default it uses a
stable, config-specific directory under:

```text
tmp_outputs/rag_stack_projects/eval_one_<config>_<hash>/
```

Reusing this directory lets later invocations reuse dataset, chunk, embedding,
and vector-index caches. The main files are:

```text
result.json                              latest result
runs/<run-id>/result.json                immutable result for this invocation
runs/<run-id>/performance_execution_trace.json
inputs/pipeline_config.json              exact input copy
inputs/system_config.json                exact input copy
logs/eval_one_latest.log
```

A successful result has:

```json
{
  "status": "ok",
  "performance_only": true,
  "quality_mode": "preserve_existing",
  "performance_mode": "measured",
  "qps": 12.34,
  "performance_score": 12.34
}
```

Read it with:

```bash
jq '{status, qps, performance_score}' \
  tmp_outputs/rag_stack_projects/eval_one_<config>_<hash>/result.json
```

To select the directory yourself, add the optional flag:

```bash
python -m rag_stack eval-one PIPELINE_CONFIG.json SYSTEM_CONFIG.json \
  --output-dir tmp_outputs/rag_stack_projects/my_measured_point
```

Exit status `0` means the QPS is publishable. Exit status `2` means the saved
deployment is invalid or the measured window is inadmissible; any positive QPS
in such a diagnostic window must not replace the Pareto performance value.
Exit status `1` means the command failed before producing a valid measurement.

## CM result replay

For each CM/optimizer candidate, pass its resolved `pipeline_config.json` and
`system_config.json` to `eval-one --performance-only`. Retain the candidate's
existing quality value and replace only its predicted performance with `qps`
from a result whose `status` is `ok`. An `inadmissible` result is diagnostic
only and must not replace the CM value.

The input `pipeline_config.json` may still contain quality metrics. They are
ignored by the in-memory performance-only overlay, and the original JSON is
copied byte-for-byte without modification.
