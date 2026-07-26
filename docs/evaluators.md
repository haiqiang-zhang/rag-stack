---
title: Evaluator Setup and Migration
nav_order: 3
---

# Evaluator setup and migration

This guide describes RAG-Stack's evaluator integration. RAG-Stack
resolves the optimizer search space and selects an evaluator through its own
uniform interface and registry. The `RAG-Stack-Evaluator` submodule provides
`static_gt`; the separate top-level `FlashRAG` submodule provides `flashrag`.
Independently installed plugins provide other
backends through the same interface. Keep the host and the provider packages
needed by your configuration installed in the same environment.

## Ownership and local checkout

```text
rag-stack/
  rag_stack/                         # Optimization, CM, configuration, shared contracts
    evaluation/                      # Uniform API, registry, public component hooks
  RAG-Stack-Evaluator/                # The evaluator checkout pinned by the host
    rag_stack_evaluator/
      backends.py                    # Provider descriptors and implementation adapters
      base.py                        # Compatibility import of the host BaseEvaluator
      static_rag_evaluator/          # Static quality and measured execution
      vllm_instrumentation/           # Runtime and calibration instrumentation
    scripts/                         # Static evaluator runtime utilities
  FlashRAG/                          # Independently installed framework provider
    flashrag/integrations/rag_stack/  # flashrag backend descriptor and adapter
```

Develop each provider in its top-level submodule checkout. An additional clone
can cause editable installs to resolve a different revision from the host's
gitlink. Both integrations consume host-owned contracts; install their editable
packages in the same environment as the compatible host.

The distribution version is currently `0.0.1` for both projects; the parent
**Git commit and submodule gitlink** identify the actual compatible source
revisions. Framework patches and native component hooks are part of the
FlashRAG fork pinned directly by the host.

## Install the selected runtime

Complete the environment and GenZ setup in the
[README](https://github.com/haiqiang-zhang/rag-stack/blob/main/README.md#installation)
first. Run the following from the host root:

```bash
git submodule update --init --recursive
conda activate rag-stack
```

Choose exactly one runtime installation:

```bash
# CPU/API development and lightweight tests.
uv pip install --torch-backend cpu -e 'RAG-Stack-Evaluator[test]' -e .
```

```bash
# GPU stack for NVIDIA driver 525-579.
uv pip install --torch-backend cu128 \
  -e 'RAG-Stack-Evaluator[cu12]' -e '.[cu12]'
```

```bash
# GPU stack for NVIDIA driver >=580.
uv pip install --torch-backend cu130 \
  -e 'RAG-Stack-Evaluator[cu13]' -e '.[cu13]'
```

The CUDA extras own vLLM, NIXL, and the related framework pins. Install the
host and evaluator with the same CUDA extra and Torch backend. CPU/API setup
covers development interfaces; local vLLM models and measured execution need
the GPU runtime and a suitable device allocation.

For `global.eval_backend: flashrag`, additionally install:

```bash
uv pip install -e FlashRAG
```

The host pins `FlashRAG/` separately from `RAG-Stack-Evaluator/`. Installing
FlashRAG adds its `flashrag` entry point to the host registry. The default
`static_gt` backend, including its ReAct dataflow, does not require FlashRAG.

### CUDA toolkit and NIXL details

Before launching local or subprocess vLLM engines, the evaluator validates an
explicit `CUDA_HOME` and discovers a compatible Conda/system toolkit. The cu13
extra supplies a complete CUDA 13 toolkit for hosts without `/usr/local/cuda`.
The cu12 extra supplies matching precompiled FlashInfer cubins; kernels that
still require source JIT need a compatible CUDA 12 compiler toolkit. These
launch checks are implemented by the evaluator.

Both NIXL 1.2.0 backend wheels install a `nixl_ep` extension. If that import
fails after a GPU installation, reinstall the backend matching your selected
stack, then check the import:

```bash
# For cu12; use nixl-cu13==1.2.0 for the cu13 installation.
uv pip install --reinstall --no-deps nixl-cu12==1.2.0
python -c 'import torch, nixl_ep'
```

## Updating an existing installation

Preserve intentional host or submodule edits before changing their pinned
revisions. Update the host, synchronize its submodule URLs, and check out
the commits recorded by that host:

```bash
git status --short
git pull --ff-only
git submodule sync --recursive
git submodule update --init --recursive

# Reinstall using your existing runtime choice; cu12 is shown here.
uv pip install --torch-backend cu128 \
  -e 'RAG-Stack-Evaluator[cu12]' -e '.[cu12]'

# Repeat only when the FlashRAG backend is used.
uv pip install -e FlashRAG
```

Resolve evaluator updates through the host gitlink. Advancing the evaluator
independently with `git submodule update --remote` may select a revision that
the current host has not integrated. When developing evaluator changes,
commit them in the nested repository and record that commit in the host.
Publish dependencies before publishing a host gitlink that refers to them.

This interface migration adds evaluator entry-point metadata. Reinstall the
provider even if it is already installed in editable mode: changing its source
checkout alone does not refresh installed entry points. From the host root,
refresh local package metadata without resolving or changing runtime dependencies:

```bash
python -m pip install --no-deps -e RAG-Stack-Evaluator -e .
# When using FlashRAG, refresh its separate entry point too.
python -m pip install --no-deps -e FlashRAG
```

Apply the same editable reinstall to an external plugin when its entry-point
metadata changes. Every worker must use an environment with the provider's
entry points installed.

Restart an existing CLI worker or development API after reinstalling or moving
packages so it loads the new module locations. For the API's evaluator reload
paths, see [Web app](apps.md).

## Import and script migration

Old host evaluator namespaces have been removed. Update custom scripts and
plugins to the owning package:

| Previous location | Current location |
| --- | --- |
| `rag_stack.static_rag_evaluator` | `rag_stack_evaluator.static_rag_evaluator` |
| `rag_stack.flashrag_quality_evaluator` or `rag_stack_evaluator.flashrag_quality_evaluator` | `flashrag.integrations.rag_stack` |
| `rag_stack.vllm_instrumentation` | `rag_stack_evaluator.vllm_instrumentation` |
| `rag_stack.base_evaluator.BaseEvaluator` | `rag_stack.evaluation.api.BaseEvaluator` (inheritance is optional; `Evaluator` is structural) |
| `rag_stack_evaluator.api` | `rag_stack.evaluation.api` |
| `rag_stack_evaluator.factory` | `rag_stack.evaluation.registry` |
| `rag_stack_evaluator.tracing` | `rag_stack.evaluation.tracing` |
| FlashRAG monitor-to-trace conversion | Native component hooks using `rag_stack.evaluation.trace_stage` |
| `scripts/start_vllm_judge_server.sh` | `RAG-Stack-Evaluator/scripts/start_vllm_judge_server.sh` |
| `scripts/check_vllm_judge_server.py` | `RAG-Stack-Evaluator/scripts/check_vllm_judge_server.py` |
| `scripts/vllm_multimodel_gateway.py` | `RAG-Stack-Evaluator/scripts/vllm_multimodel_gateway.py` |

The host dataset owner remains
`rag_stack.dataset_manager.DatasetManager`; it extends the evaluator's
`DatasetEvalManager` with host-managed cost-model integration. Cost-model
code and reusable profiles use `rag_stack.cost_model` and
`calibration_profiles/`. The calibration dispatcher remains
`python -m rag_stack calibrate`; it calls evaluator-owned instrumentation
where physical runtime measurements are needed.

## Backend and input contracts

For adding an external scorer or execution framework, see
[Evaluator Extension Guide](evaluator_extensibility.md), including
backend registration, public tracing hooks, and the FlashRAG integration. The
[Evaluator API Reference](evaluator_api.md) defines their host-owned contract.

| Quality backend | Pipeline configuration | Performance support |
| --- | --- | --- |
| `static_gt` | Shared `node_lines`; sequential or ReAct dataflow | Cost model or measured execution |
| `flashrag` | Host search-space `methods` resolved to one concrete method | Cost-model replay from recorded traces |
| Registered external backend | Descriptor-declared node lines or methods | Dynamic cost-model replay for supported components |

YAML backend selection stays at `global.eval_backend`, and backend options
stay under `eval_backend_setting`. The host resolves all search choices before
calling the evaluator. The evaluator receives a concrete in-memory pipeline
mapping, with one selected module per node. It does not accept an optimizer
search space as its evaluation input.

For callers that already own a dataset manager and a resolved configuration:

```python
from rag_stack.evaluation.registry import create_quality_evaluator

evaluator = create_quality_evaluator(
    dataset_manager=dataset_manager,
    project_dir=dataset_manager.project_dir,
    backend="static_gt",
)
quality = evaluator.evaluate(
    resolved_pipeline_config,
    run_dir=run_dir,
    metrics_override=None,
    on_trace_ready=None,
)
```

`metrics_override` replaces scoring metrics for that call without mutating the
input configuration. When a trace is produced, `on_trace_ready` receives the
same canonical envelope returned as `quality.trace`; callback
failures are advisory. FlashRAG publishes the completed serving trace before
calling its native quality evaluator. All registered backends return the shared `EvaluationResult`;
provider adapters translate older engine dictionaries at their implementation
boundary. Reserved metadata keys are not optimization metrics.

Backend capabilities come from `supported_pipeline_modes(backend)` and
`supported_metrics(backend)` in `rag_stack.evaluation.registry`. FlashRAG uses its
native metric names; `rouge_l` is accepted as an alias and normalized to
`rouge-l`. The FlashRAG provider always returns a complete dynamic trace.
The legacy `monitor: full` and `monitor: light` settings select the same exact
capture; `monitor: off` is rejected.

Measured execution uses the evaluator's `MeasuredProvider` with a fully
resolved pipeline and physical `system_config`; the host optimizer and
`eval-one` CLI prepare those inputs. See [Experimental functions](experimental_functions.md),
[Config schema](config_schema.md), and [Evaluate one saved configuration](eval_one.md).
The pinned evaluator documents the detailed
[static/measured contract](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/README.md)
and the host documents the independent [FlashRAG contract](flashrag.md).

## Data, caches, and import checks

Existing datasets and quality stores remain caller-managed. The host defaults
`RAG_STACK_CACHE_DIR` to `<host-checkout>/.cache/rag_stack` unless overridden.
FlashRAG owns JSONL/index caches under `<project>/_flashrag` and uses
`<project>/_flashrag_run` when no run directory is provided. Reuse the intended
project and cache roots; extracting the package does not require copying
internal datasets into the evaluator repository.

After an upgrade, these checks inspect import locations and lightweight
metadata without starting a model:

```bash
git submodule status --recursive
python -c 'import rag_stack_evaluator; print(rag_stack_evaluator.__file__)'
python -c 'from rag_stack.evaluation.registry import supported_pipeline_modes; print(supported_pipeline_modes("flashrag"))'
# Only after installing the optional top-level FlashRAG provider:
python -c 'import flashrag; print(flashrag.__file__)'
```

Both package paths should resolve inside this checkout. An old sibling
clone in an editable-install path means the environment must be reinstalled
from the intended provider directories. A missing `rag_stack_evaluator` import
usually means recursive submodules or the evaluator editable install are
missing; a missing `flashrag` import requires the optional top-level FlashRAG installation.
A successful import check does not validate GPU kernels, model downloads,
dataset availability, or a complete optimization run.
