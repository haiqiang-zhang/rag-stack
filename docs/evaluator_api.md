# Evaluator API Reference

An evaluator supplies two observations for each resolved candidate: scalar
quality scores and RAG-IR, the pair `(workflow, trace)`. RAG-Stack owns search,
objective selection, caching, and performance modeling. An external backend owns
pipeline execution, tracing its serving components, and quality scoring.

## 1. Select an extension boundary

| Need | Extension point | What remains reusable |
| --- | --- | --- |
| Score the existing static pipeline with another judge | Generation/retrieval metric adapter | Static execution, instrumentation, aggregation |
| Run and score a different pipeline/evaluation framework | `Evaluator` protocol and `BackendSpec` | Optimizer, quality store, RAG-CM for supported components |
| Capture an external pipeline's component I/O | Global tracing hooks or explicit `TraceSession` | Canonical RAG-IR validation and replay |

The interfaces, registration logic, hooks, and RAG-IR contracts belong to
RAG-Stack. `rag-stack-evaluator` provides `static_gt`; the separately installed
FlashRAG package provides `flashrag`. Other packages are peers of both providers. All backends
implement the same `Evaluator` protocol and register a `BackendSpec` through
the same host registry. Plugins do not need to import or install
`rag-stack-evaluator` to implement this contract.

## 2. Public Python API

```python
from rag_stack.evaluation.api import EvaluationResult, Evaluator, publish_trace
from rag_stack.evaluation.registry import BackendSpec, EvaluatorContext, register_backend
from rag_stack.evaluation import (
    capture_trace, trace_query, trace_batch, trace_stage,
    get_trace_session, TraceSession, summarize_items, summarize_lists,
)
```

These host imports do not load model runtimes or any backend implementation.
Trace/workflow validation uses `rag_stack.rag_ir`. Install the selected backend
provider separately when running its implementation.

### Construction context

| `EvaluatorContext` field | Meaning |
| --- | --- |
| `dataset_manager` | Host-owned active QA/corpus data and chunk cache; do not construct a second owner |
| `project_dir` | Project directory, if supplied |
| `options` | Selected backend's `eval_backend_setting.backend_options` dictionary |
| `dataset` | Legacy direct-construction argument; optional |

A backend factory accepts one context and returns any object implementing:

```python
def evaluate(self, config, run_dir=None, metrics_override=None,
             on_trace_ready=None) -> EvaluationResult:
    ...
```

`config` is a resolved candidate, with one selected component and concrete knob
values. Apply those choices in the external runner so it executes the pipeline
selected by the optimizer. Honor `metrics_override` per call without
changing the caller's config or default metric list. `run_dir` is the destination
for per-evaluation artifacts. Let execution/scoring errors propagate so the host
can record failure; do not turn them into successful zero scores.

### Configuration and dataset ownership

For `static_gt`, the authoritative accepted fields and examples are in the
[evaluator input contract](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/docs/input_contract.md).
Algorithm and deployment settings are separate resolved dictionaries, persisted
as `algo_config.json` and `system_config.json`. Neither selects a dataset.
Bind QA/corpus through `DatasetManager(..., qa_data_path=..., corpus_data_path=...)`
or the evaluator's dataset argument; `evaluate(config, ...)` uses that binding.
The caller chooses the training/test cohort and records it in its run notes.

The optimizer's YAML and its `algo_search_space` are host inputs. They are not
the evaluator's resolved configuration. A caller with a resolved static
pipeline and a raw corpus uses
`rag_stack.evaluator_contract.dataset_owner_config_from_resolved(config)` to
build the host dataset owner's internal document. Pass that document to
`DatasetManager.config`, and the original resolved pipeline to `evaluate`.
Chunk/token statistics are derived from the supplied data, not restored from
archived training statistics. The [replay CLI](replay.md) performs this adaptation.

Quality evaluation consumes the algorithm mapping; measured execution consumes
both mappings through `MeasuredProvider`. RAG-CM consumes validated workflow
and serving-trace data from RAG-IR plus the concrete deployment. An evaluator
trace records component work and dependencies; measured timing/QPS and judge
calls are not inputs to the CM workload. Configurations, QA/corpus and RAG-IR
therefore have distinct ownership at each boundary.

### Result contract

```python
EvaluationResult(
    metrics={"answer_quality": 0.83},
    trace=trace,
    workflow=workflow,  # WorkflowSchema or its JSON dictionary; optional
)
```

* `metrics`: named, finite numeric scalar scores. Booleans, NaN and infinity are
  rejected. The interface imposes no universal `[0, 1]` range. Configure objective
  direction and aggregation for each metric's meaning.
* `trace`: one complete invocation for every active QA question, joined by its
  stable question ID. Every backend using dynamic RAG-IR must return it;
  missing/extra/duplicate requests fail validation. Query order may differ from
  dataset order.
* `workflow`: `rag_stack.rag_ir.WorkflowSchema` or its dictionary form. Omit it
  only when the resolved host configuration already describes the runner's
  structure. Supply it explicitly when the adapter owns orchestration.

`normalize_evaluation_result` is the compatibility boundary. It translates the
result to existing `__execution_dag__` / `__workflow_schema__` fields, preserves
the trace object's identity, validates metrics and trace, and serializes the
workflow into `trace.provenance.workflow_schema`. The host removes metadata from
optimizer objectives and binds the workflow when pricing live or cached traces.
Every registered backend returns `EvaluationResult`. Older engine dictionaries
are translated within their provider adapter before crossing this boundary;
they are not a second plugin interface. The provider drops unavailable `None`
metrics instead of treating them as successful numeric scores.

Two optional fields preserve existing execution/cache behavior:

| Field | Meaning |
| --- | --- |
| `runtime_stats` | Aggregate execution statistics for a backend advertising `supports_static_ir`; cannot replace a per-query trace in dynamic mode |
| `metadata` | Internal cache metadata with reserved `__`-prefixed keys; cannot replace `__execution_dag__` or `__workflow_schema__` |

Ordinary dynamic backends leave both fields unset. `trace`, when supplied, must
always be canonical; aggregate statistics belong in `runtime_stats`. The static
backend implements its aggregate mode through this same typed contract. The
host's internal flat cache serialization and legacy-cache reader are separate
from the public evaluator protocol.

For aggregate mode, normalization copies `runtime_stats` and stores an explicit
workflow under the copy's `provenance.workflow_schema`, preserving the caller's
statistics. A workflow without trace/statistics receives an otherwise empty
aggregate carrier so cache replay retains it. Existing embedded workflow
metadata is validated and must agree with the explicit workflow, just as it
must for canonical traces.

`QualityStore` always requests dynamic tracing from every backend, including
when the project selects `global.rag_ir_mode: static` for cost modeling. Store
records require canonical per-query traces so they can be replayed on another
hardware configuration; aggregate-only results cannot be stored there.

### Early trace publication

Finish serving before calling judges, then optionally publish:

```python
trace = recorder.finish(expected_question_ids=question_ids)
publish_trace(trace, workflow=workflow, on_trace_ready=on_trace_ready)
scores = score_predictions(predictions)
return EvaluationResult(metrics=scores, trace=trace, workflow=workflow)
```

This lets the host overlap cost modeling with slow judging. `publish_trace`
validates and attaches the workflow **before** the callback. It logs callback
failures and continues scoring; the host can retry cost modeling afterward.
Return the same trace object and treat its contents as read-only after publication.
A callback is optional; an adapter can simply return the completed result.

## 3. Instrument component boundaries

The global hooks are importable from `rag_stack.evaluation` in any backend or
component module. They use the session active in the current execution context,
so a recorder does not need to be threaded through component function arguments.
No inheritance, monkeypatch, or framework-specific callback base class is required.

### Global hook API

| Hook | Behavior |
| --- | --- |
| `capture_trace(session=None)` | Activate a new session, or an explicitly shared session, for the enclosed calls; return it from the context manager |
| `trace_query(question_id)` | Bind one stable request ID in the active session |
| `trace_batch(question_ids)` | Bind a batch in input/output order in the active session |
| `trace_stage(stage, *, model_id)` | Wrap a component in the current request/batch and yield its `input` / `output` hooks |
| `get_trace_session()` | Return the active `TraceSession`, or `None` outside capture |

```python
# Component module: no recorder parameter.
from rag_stack.evaluation import trace_stage

def generate_answer(generator, model_id, prompt, prompt_ids):
    with trace_stage("generator", model_id=model_id) as call:
        call.input(token_ids=prompt_ids, text=prompt)
        answer = generator.generate(prompt_ids)
        call.output(token_ids=answer.token_ids, text=answer.text)
    return answer
```

```python
# Evaluator module: activate once, then call existing component functions.
from rag_stack.evaluation import capture_trace, trace_query

with capture_trace() as session:
    with trace_query(question_id):
        answer = generate_answer(generator, model_id, rendered_prompt, prompt_ids)
trace = session.finish(expected_question_ids=[question_id])
```

`capture_trace` activates recording but does not finalize or publish a result.
Call `finish` after all request scopes and workers complete. Recording hooks
outside an active session raise `RuntimeError`; a stage also requires a query or
batch scope. For instrumentation that is deliberately optional, explicitly check
`get_trace_session() is not None` before entering the hook.

### Explicit session API

Existing adapters may continue passing a `TraceSession` explicitly:

```python
recorder = TraceSession()
with recorder.query(question_id):
    # Existing code renders/tokenizes the actual model prompt.
    prompt_ids = tokenizer.encode(rendered_prompt)
    with recorder.stage("generator", model_id=model_id) as call:
        call.input(token_ids=prompt_ids, text=rendered_prompt)
        answer = generator.generate(prompt_ids)
        call.output(token_ids=answer.token_ids, text=answer.text)
trace = recorder.finish(expected_question_ids=[question_id])
```

The snippet assumes the runner already supplies its tokenizer, generator, and
output token IDs; it does not construct them. Record the actual chat template,
retrieval context, and tokens used by the model. Generated token IDs exclude the
prompt. For streaming, aggregate the completed response before recording output.

### Hook payloads

| Input/output argument | Single request | Batch |
| --- | --- | --- |
| Actual token IDs | `token_ids=[1, 2]` | `token_ids=[[1, 2], [3]]` |
| Exact runtime token count | `token_count=2` | `token_counts=[2, 1]` |
| Text with the component's explicit tokenizer callable | `text=prompt, tokenizer=encode` | `texts=prompts, tokenizer=encode` |
| Optional exact bytes | `byte_count=12` | `byte_counts=[12, 8]` |
| Structured item summary | `profile=summary` | `profiles=[summary_for_q1, summary_for_q2]` |

For token data, choose exactly one token representation. Optional `text`/`texts` alongside
explicit tokens provides UTF-8 byte sizes; raw text and token IDs are not retained
in the trace. Text alone fails: the hooks never download a tokenizer or estimate
counts from whitespace. Missing byte sizes remain absent so CM fallback semantics
are preserved. For a one-request batch, `token_ids` is a flat list.

Both hooks are required. Use `output(token_count=0)` for a non-generative
component, or record its structured output profile as described below.
Counts must come from the relevant component/tokenizer. For retrieval,
output token count is zero; retrieved text belongs in the downstream generator
prompt's input count and in the quality scorer's prediction record.

```python
with recorder.batch(question_ids):
    with recorder.stage("generator", model_id=model_id) as call:
        call.input(token_counts=prompt_counts, texts=prompts)
        answers = generate_batch(prompts)
        call.output(token_counts=completion_counts, texts=answers)
```

This records one call per question, without embedding runtime batch shape in
RAG-IR. Payload lengths must match the batch exactly.

### Structured inputs and outputs

Use profiles for non-token data such as ranked document lists. The hooks store
them as optional `input_profile` / `output_profile` call fields. Profiles can
accompany token and byte measurements. A profile-only payload records zero
tokens; bytes remain absent unless measured explicitly. Providing text still
requires exact token counts or the component's tokenizer.

| Profile field | Meaning |
| --- | --- |
| `items` | Required nonnegative integer: number of item occurrences |
| `list_sizes` | Optional list of input-list sizes; its sum must equal `items` |
| `list_unique_items` | Optional distinct counts per list, aligned with `list_sizes` and bounded by each list's size |
| `unique_items` | Optional distinct count across all items, bounded by `items` and the per-list union bounds when available |

`summarize_items(ids)` computes `items` and `unique_items`.
`summarize_lists(list_of_ids)` also computes `list_sizes` and
`list_unique_items`. They use the actual hashable item IDs and retain only the
counts. For example, `[["a", "a"], ["a", "b"]]` has four occurrences, list
sizes `[2, 2]`, per-list distinct counts `[1, 2]`, and two distinct items overall.
Missing statistics remain omitted; do not replace them with configured top-k.
Profiles reject unknown fields, inconsistent counts, raw IDs, and timing data.

The following instrumentation belongs at the real fusion call inside a
component, with an active capture and query scope:

```python
from rag_stack.evaluation import trace_stage, summarize_lists, summarize_items

with trace_stage("retrieval_fusion", model_id="rrf_pure") as call:
    call.input(profile=summarize_lists(ids))
    fused_ids, fused_scores = rrf_pure(ids, scores, rrf_k, top_k)
    call.output(profile=summarize_items(fused_ids))
```

Here `rrf_pure` denotes the existing merge implementation and `ids` contains
its actual input lists. The model identity must describe the implementation
executed; another implementation of RRF can perform different work. The bundled
static evaluator records this boundary in its dense, lexical, and hybrid paths; do not
wrap an instrumented boundary a second time.
For the known `rrf_pure` implementation, validation also checks that the output
is unique and does not exceed the available input union when these counts are
provided.

For a batched call, use `trace_batch(question_ids)` and supply one profile per
request in the same order: `call.input(profiles=input_summaries)` and
`call.output(profiles=output_summaries)`. These are per-request summaries,
not an aggregate over the serving batch. Hooks snapshot profiles at capture;
normalization also copies nested profiles defensively.

### Stage semantics

| Trace stage | Boundary to wrap |
| --- | --- |
| `query_expansion` | Expansion model call; exact expansion prompt/completion |
| `semantic_retrieval_encode` | Query embedding forward pass |
| `semantic_retrieval_vectorsearch` | Vector index search; separate from encoding |
| `lexical_retrieval` | Lexical retrieval operation |
| `retrieval_fusion` | An actual result-list merge, with the executed implementation and structured input/output summaries |
| `passage_reranker` | Reranking operation, using its component's accounting convention |
| `passage_compressor` | Compression operation |
| `generator` | Each generator call, including intermediate agent iterations |

Semantic retrieval is always split into encoding and vector search. Judge calls
are excluded: they measure quality, not candidate serving cost. Instrument each
boundary once. Public hooks target uninstrumented external components; wrapping
an already instrumented static component can double-count calls, and deferred
legacy text records are rejected by public finalization.

### Ordering, concurrency, and failures

Each query has ordered calls with monotonically increasing `step_idx` and a
terminal generator call. Different queries can run in concurrent asyncio tasks
or threads. Async tasks inherit the current context; enter a separate
`trace_query` / `trace_batch` scope in each worker. New thread workers explicitly
activate the shared session:

```python
def worker(session, question):
    with capture_trace(session):
        with trace_query(question.id):
            return run_pipeline(question)  # Components use trace_stage().
```

Nested `capture_trace()` scopes create independent sessions and restore the
outer session on exit. Explicit `session.query()` / `.batch()` scopes also bind
the active session, so global and explicit hooks interoperate. ContextVars
restore the enclosing session and IDs on scope exit. Overlapping
stages for the same query are rejected because the current IR cannot represent
parallel branches within one request.

Exceptions, missing hooks, repeated hooks, unfinished scopes, and wrong question
coverage make the session unpublishable. `finish` freezes it against further
writes. Repeating `finish` with the same IDs returns the same trace object.

## 4. Describe the workflow and connect observed calls

A canonical trace is JSON:

```json
{
  "queries": [{
    "question_id": "turn-1",
    "calls": [{
      "stage": "generator", "step_idx": 0,
      "input_tokens": 128, "output_tokens": 24, "model_id": "model-name"
    }]
  }]
}
```

The companion `WorkflowSchema` describes `dataflow_type` (`sequential` or
`agentic`), pipeline nodes and dependencies, stage names, component-presence
flags, and optional corpus/runtime statistics. Performance stages differ from trace stages: for
example one `generator` call is priced using `generator_prefill` and
`generator_decode`.

```python
from rag_stack.rag_ir import build_workflow_schema
workflow = build_workflow_schema(resolved_config, trace=trace)
```

Build the schema after finishing the trace. The evaluator records whether a
merge actually ran; query expansion alone does not imply RRF. The trace-aware
builder includes observed fusion stages without inventing additional calls.
It does not import or construct cost models. RAG-CM separately checks support
and derives algorithm work and hardware costs from the calls it can price.

Each structural node declares `node_id`, `stage`, `components`, named `inputs`,
and `operations`. `execution_order` preserves node order; `execution_policy`
currently accepts only `sequential`. Newly captured calls have query-local
`call_id` values. Pass `node_id` and `input_refs` to `trace_stage()` to associate
an operation with its configured node and actual producers. After a single-query
stage exits successfully, `call.call_id` exposes its recorded ID. A batched stage
exposes `call.call_ids` in query order and accepts one input-reference mapping per
query. References must point to earlier calls in the same query.

For hybrid fusion, `input_refs` has `dense` and `lexical` ports pointing to each
producer's latest result. The input profile's list sizes must match those
producers' output counts. The host validates these relationships on result
publication, cache loading, and replay. Existing traces without identities remain
readable, but cannot satisfy a hybrid workflow's producer contract. See
[sequential hybrid retrieval](hybrid_retrieval.md) for the full configuration and
module boundaries.

For custom orchestration, construct an explicit schema describing the actual
runner. Its stages and presence flags must match the host's declared pipeline
operations. Explicit corpus statistics are retained; omitted corpus
statistics are supplied from the host's resolved dataset metadata. Every observed
trace stage must have a corresponding declared operation; configured conditional
stages may remain unobserved. Complete
instrumentation of executed components remains the adapter's responsibility.
The workflow travels in trace provenance and in persisted resolved configs, including cache
replay through the host controller. Direct RAG-CM callers must supply the
matching workflow explicitly. Quality-store identity includes the selected backend and backend options.
Version the external runner/judge and include those identities in options.

Registration does not implement new hardware cost models. A new unsupported
component, tokenizer/model mismatch, or an arbitrary parallel dependency graph
requires separate IR/CM work. The current trace is an ordered invocation, not a
timestamped runtime profile. `supports_measured` is descriptive metadata; a plugin
does not automatically receive a physical measurement runner. Dynamic integration
can use `performance_source: cost_model` with `rag_ir_mode: dynamic` only when
all observed components have supported models. The bundled static evaluator also
provides measured execution; sequential BM25 hybrid currently requires
`performance_source: measured` because BM25 has no analytical cost model.
Current RRF pricing covers the supported local merge inside a retrieval worker.
Capturing a fusion call does not establish support for arbitrary parallel
lexical/dense branches or every fusion implementation.
`supports_static_ir=True` advertises an additional aggregate-statistics mode;
the host checks this capability rather than identifying a backend by name.

## 5. Register and select a backend

```python
# my_eval/backend.py
from rag_stack.evaluation.registry import BackendSpec

BACKEND = BackendSpec(
    name="my_eval",
    factory=MyEvaluator,  # MyEvaluator(context) implements evaluate(...)
    metrics=("answer_quality",),
    pipeline_modes=("sequential",),
    pipeline_style="node_lines",
    supports_static_ir=False,
)
```

For scripts embedding RAG-Stack, call `register_backend(BACKEND)` before creating
the controller. For CLI and subprocess discovery, install an entry-point plugin:

```toml
[project.entry-points."rag_stack.evaluators"]
my_eval = "my_eval.backend:BACKEND"
```

The entry point resolves to a `BackendSpec` object with the same name. Duplicate
registrations and ambiguous installed names fail. Discovery lists package
metadata without loading all plugins; selecting a backend loads its descriptor.
This discovery path is identical for `static_gt`, `flashrag`, and external
plugins; the host registry has no built-in implementation imports. Installing
`rag-stack-evaluator` supplies `static_gt`; installing the top-level `FlashRAG`
package supplies `flashrag`. Keep model imports
inside each backend factory/runtime. `metrics` and `pipeline_modes`
can also be lazy zero-argument callables returning tuples.

Entry points are installed distribution metadata. Reinstall a provider after
adding or changing them, including an existing editable install. Refresh the
bundled provider from the host root with
`python -m pip install --no-deps -e RAG-Stack-Evaluator -e .`; refresh the
optional FlashRAG provider with `python -m pip install --no-deps -e FlashRAG`.
Changing source files alone does not update discovery in a new worker.

```yaml
global:
  eval_backend: my_eval
  rag_ir_mode: dynamic
system:
  performance_source: cost_model
eval_backend_setting:
  metrics: [answer_quality]
  backend_options:
    runner_revision: your-pinned-revision
    # Additional fields are defined by your adapter.
```

This is an overlay on a complete host config, not a standalone optimization
configuration. Keep the dataset, system, optimizer and algorithm search-space
sections. `pipeline_style: node_lines` accepts the normal node-line representation
(and method blocks for advertised modes); `methods` requires explicit method
blocks. Backend metadata drives validator and registry APIs. Use YAML for plugin
options; the visual editor has no custom plugin-specific controls. The generic
scaffold does not invent valid options for an external backend.

Every factory receives `EvaluatorContext.options` from `backend_options`.
The old provider-level `rag_stack_evaluator.factory` module forwards to the host
and translates its legacy `flashrag_options` argument. New integrations use
`rag_stack.evaluation.registry` and the uniform `backend_options` argument.

## 6. Ownership and migration

| Public facility | Canonical host module |
| --- | --- |
| `Evaluator`, optional `BaseEvaluator`, `EvaluationResult`, `publish_trace` | `rag_stack.evaluation.api` |
| `BackendSpec`, construction context, discovery and registration | `rag_stack.evaluation.registry` |
| Global hooks and `TraceSession` component I/O hooks | `rag_stack.evaluation.tracing`, also exported from `rag_stack.evaluation` |
| `WorkflowSchema` and trace schemas | `rag_stack.rag_ir` |

The earlier `rag_stack_evaluator.api`, `.factory`, and `.tracing` imports are
compatibility forwarding modules. New integrations import the host modules.
Use this reference for the shared API, [FlashRAG integration](flashrag.md) for
the independent FlashRAG provider, and the evaluator repository documentation
for the static provider.

The public interface, registry, and hooks can be imported and used without
importing backend implementations. Existing host dataset and component utilities
still depend on `rag-stack-evaluator`.

## 7. Validate a backend

Contract tests belong in the host repository; backend implementation tests
belong in their provider. Run the generic interface and hook tests from the
host root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest tests/contracts/test_evaluator_plugins.py \
  tests/contracts/test_public_tracing.py tests/contracts/test_global_tracing.py -q
```

These tests exercise registration, result validation, question coverage,
component accounting, nesting, concurrency, and failures. Validate each new
backend with its actual pipeline entry point as well: a valid descriptor alone
does not show that component calls are being captured. The
[FlashRAG integration](flashrag.md) documents this boundary for the bundled
independent framework provider.
