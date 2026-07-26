# Project CM preview

Open a project in Apps and select **CM Preview** to execute a small query cohort
and inspect the physical path compiled by RAG-CM. The preview runs the selected
evaluator, retains its canonical RAG-IR, and invokes the same physical compiler
used by fixed-deployment CM execution. The browser displays that compiler's
output.

## Select the scope

Choose an existing evaluation's resolved pipeline, or supply every active
parameter of a new configuration point. The latter uses the project's formal
Ax search space and normal pipeline resolver. A search space is not itself a
pipeline, and previewing one point does not validate its other choices.

The deployment must also be concrete. Optimizer-owned deployment parameters
use the normal system resolver. For a project whose CM ordinarily searches
deployments, provide an explicit resolved `system_config`; the preview does
not silently choose a layout from the sweep.

The default cohort is the first eight QA rows. Explicit question IDs select
and order a different cohort. QA and corpus paths are invocation inputs,
separate from `algo_config.json` and `system_config.json`. The preview displays
the selected IDs and source so its scope remains visible after reopening it.

By default, the evaluator runs retrieval, fusion and generation while skipping
all scoring, including node-level judges. Serving parameters are unchanged.
This produces observed calls without requiring a quality score. **Reuse trace**
is an explicit alternative for an existing evaluation: it recompiles the saved
RAG-IR, records its source, and does not claim to have executed a new evaluation.
Reuse requires `provenance.trace_origin: observed`. Older traces without this
producer marker and reconstructed fallback traces require a fresh execution.

## Inspect the result

The query view places recorded calls next to compiled operations. Selecting an
operation highlights its source calls. The compiler supplies:

- physical workers, including the calls combined into a retrieval worker;
- LLM prefill and decode roles, admission policies and replica assignments;
- communication operations and their dependencies;
- device placement, batch policies and calibration bindings.

`source_call_ids` and `source_call_indices` preserve the mapping back to the
observed RAG-IR. Indices are zero-based positions in a query's original `calls`
array, not its possibly sparse or repeated `step_idx` values.
An observed call may map to several operations, for example
prefill and decode, or share a worker with other calls. Replica-pair transfers
represent alternatives selected by the serving policy, not simultaneous work
on every replica.

A successful compilation means CM has bound an executable path for this
configuration, deployment and observed cohort. It is not a latency or QPS
prediction. Analysis and simulation execute the compiled path separately.
Small-cohort previews also do not establish that future queries take the same
conditional branches or number of loop iterations.

Unsupported operations are explicit blockers. In particular, a valid serial
dense/BM25/RRF evaluation can produce RAG-IR while CM reports that BM25 pricing
or hybrid physical scheduling is unavailable. The raw trace remains accessible;
unsupported calls are never omitted or assigned zero cost.

## Compiler contract

The CM owns workload compilation and deployment binding:

```python
from rag_stack.cost_model.compiled import describe

plan = assembly.compile_fixed(algo_config, system_config, trace=rag_ir)
preview = describe(plan)              # Display projection for Apps / inspection
analysis = assembly.analyze_plan(plan)
simulation = assembly.simulate_plan(plan)

```

The lower-level `compile_workload(resolved_pipeline, rag_ir)` and
`bind_plan(workload, ...)` separate the observed workload from a deployment's
physical resources and cost models. Both the preview and normal fixed dynamic
CM execution use these contracts. The retained simulator inputs drive
simulation; analytical execution consumes the plan's source calls and bound
latency functions.

`describe(plan)` produces the `rag-cm.compiled-plan.v1` display projection. Its
`queries`, `resources`, `deployment`, `calibration`, workload diagnostics and
source-call mappings describe the compiled path. Apps saves this projection as
`compiled_description.json` for users and developers to inspect later.

The executable `CompiledCMPlan` object retains its graph and bound service
functions in the process that compiled it. Analysis and simulation consume that
object directly. The saved description is only a preview: it is never imported
as execution input or used to restore, recompile, analyze or simulate a path.
A changed pipeline requires recompilation; a changed deployment or calibration
requires rebinding through the normal runtime interfaces.

## Job and artifact ownership

Each preview lives under `<project>/cm_previews/<preview_id>/`. It owns its
worker process, runtime workspace, log, configuration snapshots, RAG-IR and
`compiled_description.json`. The project stores a run ID, schema/package
versions and an event note; experiment-file checksums are not part of acceptance.

The worker saves the display projection after compilation. Apps can reopen and
download it after a server restart. Opening or downloading a preview only reads
these saved display records; it never invokes the evaluator, compiler, analysis
or simulation. The UI labels the compiled path **Display only**. No executable
`compiled_plan.json` or restoration API is provided.

If a historical job has no saved description, Apps shows that the preview is
unavailable while retaining its recorded RAG-IR and job history. Creating a new
preview is an explicit action; reusing an observed trace remains an option for
that new job.

Jobs are asynchronous and their history can be reopened. Cancellation targets
only the recorded preview worker session. Worker exit also releases its owned
model subprocesses, including evaluator caches retained for later evaluations.
A vanished worker becomes
`interrupted`, retaining its artifacts. A compiler blocker becomes `blocked`;
unexpected runtime failures become `failed`. Neither path updates optimizer
evaluation counts, checkpoints or Pareto records.

The API uses `/api/rag-project/projects/{project}/cm-previews`:

| Method | Suffix | Purpose |
| --- | --- | --- |
| GET | `/options` | Formal configuration controls and dataset availability |
| POST | empty | Create an isolated preview |
| GET | empty | List preview job history |
| GET | `/{preview_id}` | State, log tail, observed trace and saved compiled-path display description |
| POST | `/{preview_id}/cancel` | Cancel the owned preview worker |
| GET | `/{preview_id}/artifacts/{name}` | Download persisted inputs, traces, job records or the display description |

The detail response exposes `compiled_path_status`: `available` when a saved
description can be shown, `unavailable` when a completed job has no readable
description, or `null` when no compiled path was produced. `compiled_plan`
carries this display projection when available; its internal field name does
not make it an executable input. Apps polls only queued or running jobs.

Creation requires exactly one of `eval_id`, `parameters` (flat active Ax
parameters), or `algo_config`. Additional fields include `system_config`,
`qa_data`, `corpus_data`, `dataset_name`, `sample_count`, `question_ids`,
`reuse_trace` and `skip_judge`. Unknown request fields are rejected. Reused
traces keep their original data provenance; dataset overrides require a fresh
run.

See the [evaluator API](evaluator_api.md), [resolved-config replay](replay.md)
and [CM execution-path contract](rag_cm/execution_path.md) for the surrounding
interfaces.
