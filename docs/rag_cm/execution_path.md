---
title: Automatic Compilation and Algorithm Models
parent: Cost Model (RAG-CM)
nav_order: 2
---

# Automatic compilation and algorithm models

The evaluator records the operations its pipeline actually executes in
RAG-IR. Its generic compiler preserves those recorded calls as execution paths.
RAG-CM checks pricing support and invokes
their algorithm and system models. For RRF, the trace contains a
`retrieval_fusion` stage with observed input-list sizes, union cardinality,
and output size. Query expansion alone never causes the compiler to insert RRF.

## Use RAG-CM

Given a resolved `pipeline_config`, a hardware `SystemConfig`, and the
evaluator's `schema` and canonical `trace` envelope:

```python
from rag_stack.cost_model import RAGCMAssembly

result = RAGCMAssembly(hardware).evaluate_dynamic(
    pipeline_config,
    workflow_schema=schema,
    trace=trace,
)
```

The same automatic compilation applies to trace-driven fixed-deployment
evaluation with `evaluate_fixed`. The [entry-point guide](README.md#the-three-public-entry-points)
describes configuration and hardware setup.

The static evaluator records its dense-retrieval `rrf_pure` calls automatically,
including the single-list fast path. Custom evaluator authors instrument the
real fusion boundary with the [global hooks](../evaluator_api.md). Users do
not instantiate `RRFAlgorithmModel` or construct algorithm workloads to run CM.
Old traces without fusion observations remain unchanged; regenerate them with
an instrumented evaluator to include fusion cost.

## From observed RRF to a compiled execution path

The workflow schema and resolved pipeline select component models. The
quality trace supplies invocation order and per-call input/output sizes.
RAG-CM performs the following steps internally:

| Step | Responsibility | Result |
| --- | --- | --- |
| Capture execution | The evaluator hooks actual component input and output | Ordered calls with implementation identity and structured I/O profiles |
| Bind components | `dispatch_manifest` expands dense retrieval and binds observed `rrf_pure` calls | Separate encode, vector-search, and fusion model bindings |
| Prepare and compile the workload | `prepare_trace_workload` validates and samples the trace, then calls `compile_execution_paths` | An explicit operation path for each retained request |
| Describe fusion work | `RRFCostModel` reads each recorded I/O profile and invokes `RRFAlgorithmModel.build` | A hardware-independent operator work profile |
| Price and schedule operations | CPU service modeling, analytical evaluation, and simulation consume the derived work and paths | Deployment throughput and latency estimates |

For the supported local dense-retrieval adapter, the evaluator records:

```text
query_expansion → semantic_retrieval_encode → semantic_retrieval_vectorsearch
                → retrieval_fusion → generator
```

The global hook around the real fusion function is small:

```python
from rag_stack.evaluation import trace_stage, summarize_lists, summarize_items

with trace_stage("retrieval_fusion", model_id="rrf_pure") as call:
    call.input(profile=summarize_lists(input_ids))
    output_ids, output_scores = rrf_pure(input_ids, input_scores, 60, top_k)
    call.output(profile=summarize_items(output_ids))
```

This runs within the evaluator's `capture_trace()` and `trace_query(qid)`
scopes. `rrf_pure` is the static evaluator implementation; a different
implementation must identify itself and supply a matching cost model.
The hooks summarize existing materialized IDs without storing them in the
trace. For example, `[["a", "a", "b"], ["b", "c"]]` produces:

```json
{"items": 5, "list_sizes": [3, 2], "list_unique_items": [2, 2], "unique_items": 3}
```

The output profile records the actual emitted count. The profiles do not
infer bytes, tokens, latency, or union size from `top_k`. Profile-only calls
carry zero token work and omit unknown byte fields. Use
`build_workflow_schema(config, trace=trace)` to include the observed fusion
stage when constructing an explicit workflow schema.

The original quality trace is preserved. Internally,
`TraceWorkload.traces` retains the producer's calls and
`TraceWorkload.execution_paths` contains the compiled paths used by analysis
and simulation. These fields support inspection of workload preparation;
calling the compiler manually is not part of normal evaluation.

The runtime carries these paths in `PreparedExecutionPaths`, available as
`TraceWorkload.prepared_paths`. Construction validates and copies the selected
calls once. Candidate analysis, analytical walks and DES builders borrow that
same snapshot read-only; they do not validate or deep-copy it again per candidate.
Spawned DES tasks carry only the selected paths, not the original unsampled
envelope. Physical lowering and pricing still run for each distinct deployment.

Low-level entrypoints that accept raw call lists continue to validate and copy
them. Internal callers may also provide `prepared_paths`, which must own the
exact `traces` list supplied to that call; a mismatched snapshot is rejected.
The ownership check is constant-time. Consumers, including injected pricing
callbacks, must not mutate the snapshot or its nested profiles. A new or changed
trace requires preparation of a new snapshot. This is an in-memory contract,
independent of saved preview descriptions.

Compilation preserves fusion calls and nested profiles without duplication.
Recompiling a compiled path has the same result. Repeated retrieval visits
remain separate, and input shapes may differ across queries and visits.

Fusion workload comes from each call's profiles. No fusion call is fabricated
when a pipeline concatenates results, uses another merge, or supplies an older
trace. A declared RRF call with missing profiles fails pricing with an explicit
error. Generic profiles may omit distinct counts; in that case the algorithm
model marks its union estimate as an upper-bound assumption. The built-in
producer supplies exact counts from the real inputs.

RRF has a separate CPU service cost. In simulation, local search and RRF
execute within the same physical retrieval worker, sharing its queue and
batching timeout. RRF becomes a named CPU operation after the backend chunks;
making it explicit does not add a queue or permit overlap with the next visit
in that worker.

The current CM system adapter covers local dense-retrieval fusion, immediately
following its encode/search visit. Generic RAG-IR also represents serial
dense/BM25 hybrid execution, including each local merge and the final hybrid
merge with explicit producer references. Its compiler does not impose CM's
local-worker adjacency restriction. BM25 analytical cost modeling is unavailable,
so CM rejects lexical/hybrid pipelines before pricing. The evaluator and measured
PE can execute those pipelines independently; see
[sequential hybrid retrieval](../hybrid_retrieval.md). Parallel joins remain a
separate extension.

The current downstream reranker and compressor adapters assume the passage
count configured by `semantic_retrieval.top_k`. If recorded fusion outputs
differ from that count, CM binding rejects the configuration instead of pricing
the wrong downstream workload. This restriction belongs to CM; measured PE does
not use it to reject a valid saved workflow. Fusion followed directly by generation supports
observed output sizes that vary between calls, including empty outputs.

## Component-model developer interface

This interface is for developers adding or maintaining an algorithm model.
RAG-CM invokes it after deriving the component's workload. It is not the
entry point for evaluating a user's RAG pipeline.

The interface in `rag_stack.cost_model.algorithm` has one method:

```python
class AlgorithmModel(Protocol):
    def build(self, config, workload: AlgorithmWorkload) -> OperatorWorkProfile:
        ...
```

An implementation registers with `@register_algorithm_model("component_name")`.
Importing its module performs registration. No inheritance is required.
Registration makes the model discoverable; connecting a new component also
requires model dispatch, workload derivation, and a system service model.
Adding an algorithm model alone does not add a pipeline execution path.

| Value | Contents |
| --- | --- |
| `config` | Algorithm and implementation parameters, such as output top-k and record widths |
| `AlgorithmWorkload.inputs` | Named input ports, each containing a `DataProfile(items, bytes, properties)` |
| `AlgorithmWorkload.statistics` | Additional workload characteristics, such as the union size of overlapping candidate lists |
| `OperatorWorkProfile.phases` | Each phase's operation counts, logical accessed bytes, optional unique/working-set bytes, access block size, compute mode, and parallel fraction |
| `OperatorWorkProfile.dependencies` | Required ordering between named phases |
| `OperatorWorkProfile.outputs` | Data profiles delivered to downstream components |
| `OperatorWorkProfile.metadata` | Implementation identity, estimation basis, and assumptions |

The method receives no hardware specification and returns no latency or QPS.
Cache placement, bandwidth, resource allocation, and execution timing belong
to system modeling. Dependencies allow execution when predecessors complete;
they do not imply that the system runs independent phases concurrently.

`DataProfile.bytes=None` means unknown volume. It must not be replaced with
zero. Operation counts and byte volumes must be finite and nonnegative;
phase names must be unique and their dependency graph must be acyclic.

### Built-in RRF implementation

`RRFCostModel.build_work_profile(call)` constructs the derived
`AlgorithmWorkload` and invokes `RRFAlgorithmModel.build` internally.
The output describes alignment, ranking, score accumulation, sorting, and
output materialization for the supported `rrf_pure` implementation. It models
the implementation's full sorts before truncation. Each phase keeps different
operation kinds separate; comparisons and floating-point divisions are not
treated as interchangeable FLOPs.

If union cardinality is unavailable, the model uses a disjoint-list upper
bound and records that assumption. Its byte estimates describe logical
scalar/reference storage, excluding Python object overhead. These are work
estimates, not measured CPU memory traffic or execution times.

The system adapter
reads the profile's `input_items` demand and applies the existing standalone
CPU calibration profile for this implementation. The service unit is one
fusion visit. The detailed phase estimates remain available for inspection;
they are not individually converted to measured phase times by this adapter.
Introducing a new implementation can require a new model and calibration.
The [calibration guide](calibration.md) describes profile selection and policy.

The single-list Python fast path and the multi-list pandas implementation
have different work profiles. The existing standalone calibration covers
multi-list fusion only. Strict calibrated pricing rejects single-list calls
until a matching service profile is available. Explicit `uncalibrated_spec`
mode uses a hardware-specification lower bound for this path; it excludes
Python dispatch and allocation overhead and is not a calibrated prediction.

## RAGO table compatibility

This conversion is an internal adapter step performed by assembly.
RAG-CM stores separate `semantic_retrieval_vectorsearch` and
`retrieval_fusion` component tables. Their batch axes differ:

- A search-table batch counts backend queries.
- A fusion-table batch counts retrieval visits, each merging `K` lists.

The external RAGO candidate generator currently accepts one CPU retrieval
table. `to_rago_sweep_dfs(..., retrieval_fanout=K, max_retrieval_servers=N)`
converts the separate tables at that boundary. If one fusion visit costs
`f` milliseconds, a search row
with `B` backend queries receives a compatibility charge of `B*f/K`.
Here `f` is the mean observed fusion demand per retrieval visit, including
visits without fusion. This coarse table supports candidate construction;
analysis and simulation price each actual call from its own profile.
RAGO then applies the request's total retrieval multiplicity. The adapter
receives `K`, rather than the total `R*K`.

Search replication is expanded before adding fusion. For a one-server row
with batch `B` and search time `t`, an unmeasured `n`-server row uses batch
`n*B` and the same search time, then adds `n*B*f/K` fusion work. Existing
multi-server search rows are preserved. This keeps local fusion service
shared when search servers are replicated. Assembly supplies a server-count
bound covering the candidate space; the adapter also includes its next
power-of-two boundary so RAGO's preprocessing cannot trigger further automatic
replication of the combined table.

The adapter copies the search table and validates that fusion cost scales
linearly with visit batch size. It rejects incompatible tables. It also
excludes fusion from RAGO's GPU-stage and placement lists. The original
component tables remain separate, and analysis and simulation continue to
price the explicit fusion calls.
