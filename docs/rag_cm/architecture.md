---
title: Architecture and Workload Representation
parent: Cost Model (RAG-CM)
nav_order: 1
---

# RAG-CM architecture

RAG-CM predicts the serving throughput and latency of a RAG workload on
declared hardware. It first analyzes system configurations and shortlists them
according to the optimization objective. It then simulates the retained
configurations using the recorded quality trace. This page follows that
trace-driven execution path; the [CM overview](README.md) lists the public APIs.

## Component models, component tables, and system candidates

A **component cost model** is a function that estimates the execution time of
one stage, such as generator prefill or vector search. Its inputs include the selected model or index, workload shape
(such as token lengths), batch size, hardware, and parallelism. The function
uses hardware specifications and component calibration profiles.

A **component cost table** contains sampled outputs of that function. For a
fixed component and representative workload shape, RAG-CM evaluates a grid of
batch sizes and device counts, recording the TP/PP choice and predicted
service time for each row. TP and PP denote tensor and pipeline parallelism.
Time units are stage-specific: the prefill example below describes a complete
prefill batch. A decode component's per-step estimate must be combined with
its generated length when estimating a complete decode burst.

For example, a generator-prefill table for a fixed model with **676 input
tokens per call** could have the following structure. **The time values below
are illustrative, not measurements or RAG-CM predictions.**

| Component batch size (calls) | GPUs (count) | TP | PP | Batch service time (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 1 | 1 | 3.0 |
| 8 | 1 | 1 | 1 | 12.0 |
| 1 | 2 | 2 | 1 | 2.5 |
| 8 | 2 | 2 | 1 | 7.0 |

The 12 ms entry is the estimated time for that **entire prefill batch**. It
does not include retrieval, decoding, repeated generation calls, or queueing,
and is therefore neither request latency nor system throughput.

A **system candidate table** has a different unit: each row is a complete
deployment configuration. It specifies stage placement, exact devices,
parallelism, request/decode batch settings, and batching timeout. Analytical
and simulated system metrics are attached to those rows. RAGO uses component
tables to construct candidate configurations; the two evaluation stages use
the underlying component cost functions to price those configurations.

![RAG-CM architecture: workflow schema and system configuration support candidate construction; trace-derived statistics feed component tables, encoding, and communication; compiled execution paths and shared component service models feed analytical estimation and closed-loop simulation.](../assets/rag_cm_architecture.svg)

*The retained trace is compiled into explicit execution paths for both
evaluation stages. Component tables support candidate construction; they are
not the simulator's workload or service-time input. Both stages evaluate
component service functions for the candidate.*

## Schema, trace, and the internal summary

| Representation | Contents | Role |
| --- | --- | --- |
| Workflow schema | Workflow type, participating stages, dataset and prompt statistics, and vector-index cardinalities | Supplies stage structure and workload information not present in each call |
| Quality trace | One ordered call sequence per request, including stage, sequence position, token counts, implementation identity, optional bytes, and structured I/O profiles | Preserves actual calls, individual sizes, repeated invocations, and call order |
| Compiled execution paths | Validated copies of the evaluator's calls and their nested I/O profiles | Gives analysis and simulation the same observed operation order without inventing stages |
| Internal trace summary | Representative token lengths, mean invocation counts, and observed fusion profiles derived inside RAG-CM | Supplies component workload and scalar inputs for component tables, candidate feasibility, encoding, and communication |

The summary is computed from the trace, not supplied as a third RAG-IR input
or stored as part of the dynamic workflow schema. Schema statistics include
indexed-vector counts, chunk-length means and quantiles, mean question and
reference-answer lengths, and fixed prompt-template length. These quantities
come from the corpus, evaluation dataset, and prompt template; a call trace
does not establish the size of the complete vector index. The
[input contract](README.md#the-frozen-input-contract) lists the schema fields.

The evaluator records each actual RRF merge as `retrieval_fusion`, with
input-list sizes, distinct counts, and output size captured at the call
boundary. The [execution-path compiler](execution_path.md) preserves this
stage; query expansion never creates a fusion call by itself. Search and
fusion have separate component models, tables, and CPU service demands.
CM derives fusion work from the recorded profiles and uses a service model
for the identified implementation. Users provide RAG-IR through the existing
CM entry points; the algorithm-model interface is for component developers.
The compiler preserves repeated retrieval visits;
it does not flatten all retrieval rounds into one merge or infer parallel
branches from an ordered trace.

Consider the two ReAct requests below. Each pair is `(input tokens, output
tokens)` for a generation call. Query encoding consumes 8 and 7 tokens in the
first request's retrieval rounds, and 5 tokens in the second request's round.

| Request | Generation calls, in execution order | Retrieval rounds |
| --- | --- | ---: |
| Query 1 | `(180, 16)` → `(800, 14)` → `(1400, 48)` | 2 |
| Query 2 | `(200, 10)` → `(800, 48)` | 1 |

The internal summary has concrete consumers:

| Derived quantity | Value for this trace | Where it is used |
| --- | --- | --- |
| Generator input tokens per call | `3380 / 5 = 676` | Representative prefill shape in component tables and concrete-device feasibility checks |
| Generated tokens per request | `136 / 2 = 68` | Representative decode length in component tables and feasibility checks, including intermediate generations |
| Generation calls per request | `5 / 2 = 2.5` | Repeated-prefill factor in RAGO's coarse component calculations; generator prefill/decode transfer count |
| Dense retrieval visits per request | `3 / 2 = 1.5` | Retrieval/encoding component-table workload sizing; combined with configured subquery expansion per visit |
| Retrieval-query tokens | `floor((8 + 7 + 5) / 3) = 6` | Representative length used by the encoding service function in analysis and simulation |
| Generator input tokens per request | `676 × 2.5 = 1690` | Generator KV-transfer volume in the analytical communication model when prefill and decode are separated |

The query-length mean is computed from retrieval calls' recorded
`input_tokens`. In this ReAct trace, encoding and vector search record the
same query text with the same tokenizer. Thus the three retrieval records
carry 8, 7, and 5 query tokens. The vector-search tensor input shown as
"1 vec" in the trace visualization is a different quantity.

The last row is calculated from the summary's first and third rows. Multiplying
1,690 by the model's KV bytes per token gives the estimated transfer bytes per
request. Link bandwidth and transfer overhead then determine a communication
latency and a throughput bound. Simulation models individual transfer events
using the trace's call shapes.

Workload preparation validates the trace and retains a deterministic stride
sample when it exceeds 2,048 requests. It derives the summary and compiles
execution paths from that retained cohort. `TraceWorkload.traces` preserves
the producer's calls; `TraceWorkload.prepared_paths` carries its owned
`execution_paths` snapshot through analysis and simulation. Candidate walks and
DES builders borrow these calls read-only without revalidating or deep-copying
them. All representations retain the same requests and source IDs.
Token means are stored as
integers; invocation factors have a minimum of one. The summary's
`num_retrievals` field denotes generator prefill/decode handoffs, while
`retrieval_invocations_per_query` denotes dense retrieval visits.

## Stage 1: analytical estimation and shortlisting

**Construct candidates.** RAGO combines component tables with the configured
placement, resource, and batch search space. RAG-CM resolves and checks the
resulting concrete device/parallelism choices. Representative lengths affect
component pricing and memory checks; component-table availability can
constrain resource counts. A fixed deployment constructs one candidate
directly without a sweep.

RAGO also produces coarse system QPS and latency during candidate construction.
For trace-driven evaluation these values are discarded before analytical
scoring. They do not rank the shortlist. The component tables therefore
support candidate construction and feasibility, rather than supplying the
final analytical or simulated system metrics.

RAGO's external table format has one CPU retrieval slot. At that boundary,
the adapter folds the independent fusion table into a copy of the search
table using their declared batch units. RAG-CM retains separate original
tables and execution calls. This compatibility conversion is described in
[RRF table lowering](execution_path.md#rago-table-compatibility).

**Analyze each configuration.** `DynamicEngine.analyze` evaluates generation
calls with their individual input/output lengths and the concrete candidate's
devices, parallelism, and batch settings. Repeated calls contribute repeatedly
to resource demand. The prefix-cache approximation uses earlier generation
calls in the same request. Reranker pricing also uses trace call shapes;
encoding uses the representative length described above. The 676-token mean
does not replace the five actual generator input lengths during this analysis.

**Compute system metrics.** The model sums call costs into average service
demand per request on each resource. It accounts for batching, replicas, and
stages sharing a resource, then derives capacity bounds. The smallest relevant
capacity limits throughput; a concurrency and batch-fill approximation can
reduce it further. Request execution time and a queueing approximation yield
analytical latency. Communication contributes latency and a separate bandwidth
capacity bound.

For example, without batching or replication, 10 ms of retrieval service and
20 ms of generation service per request imply capacities of 100 and 50
requests/s on two independent resources: the bottleneck is 50 requests/s.
If both stages use one serial resource, their 30 ms combined demand gives
about 33 requests/s. Per-call time alone does not determine system throughput.

**Shortlist by the objective.** `hybrid_screening.screen_des_candidates` reads
the `dynamic_analytical_*` scores from analysis. `max_throughput` favors higher
throughput; `min_latency` favors lower mean latency. TTFT/TPOT policies use
their corresponding analytical fields. Weighted policies retain candidates
from the weighted score and each positive-weight metric; constrained policies
retain candidates from the objective and constraint rankings. Final constraints
are checked after simulation. The default objective is `min_latency`.

Shortlisting preserves every concrete device/TP/PP topology within each
workflow and generator/query-expansion P/D stratum. For each ranking it retains
five request/decode batch parents per topology, or one for sweeps of at least
4,096 rows, plus a global top-ten parent rescue. All timeout siblings of a
retained parent are kept together. Compound policies use the union of these
sets. Sweeps below 256 rows, fixed deployments, and policies without an
analytical ranking contract retain all candidates. `CM_DES_EXACT_ALL=1` also
requests exhaustive simulation.

## Stage 2: closed-loop simulation and final selection

`closed_loop_sim.apply_closed_loop_simulation` receives the explicit shortlist,
the same trace cohort, and the frozen calibration policy. It consumes compiled
execution paths with an explicit RRF CPU operation inside the originating
retrieval worker. Search and local fusion share that worker's queue and batch
timeout while retaining separate service costs. For each retained
configuration it models queues, batching, prefix/KV caches, resource sharing,
and communication events. Component service functions supply execution costs;
the simulator determines when those services execute and which calls share a
batch. It does not replay the representative component cost table.

The simulator increases offered concurrency to obtain saturated serving
throughput and the corresponding mean request latency. Modeled host CPU
core-seconds capacity may additionally cap throughput without changing GPU
service times. The returned `qps` and `latency_s` contain these final results.
TTFT/TPOT fields remain analytical estimates, rather than simulated per-token
observations.

Final selection applies the same configured objective to simulated candidates.
Screened rows remain in the output as `simulator_status=screened_out`, with
zero QPS and infinite latency. Selection considers only `simulator_status=ok`
rows.

## Batch settings remain configuration choices

A candidate with `batch_size_request=8` retains that setting when the trace
averages 2.5 generation calls per request. The invocation count describes
repeated work; it does not divide or replace the configured batch.

RAGO adjusts its coarse table lookup to account for repeated calls and
extrapolates costs beyond the sampled table range. That calculation does not
change the configured request-batch grid or prescribe runtime batching.
`no_microbatching: true` gives stages before generator decode one shared
request-batch axis. Generator decode has its own concurrency setting,
`batch_size_decode`. Simulation determines batching of ready calls within
those configured limits.

## Implementation responsibilities

| Module or method | Responsibility |
| --- | --- |
| `workload.prepare_trace_workload` | Validate/sample the trace, derive summary statistics, and compile the shared execution paths |
| `rag_ir.execution_path.compile_execution_paths` | Validate and copy observed calls and structured I/O while preserving invocation order |
| `rrf.RRFCostModel` | Price the independent CPU RRF component and generate its per-visit batch table |
| `rago_adapter.to_rago_sweep_dfs` | Lower separate search/fusion tables into RAGO's external retrieval-table format |
| `assembly.RAGCMAssembly` | Coordinate inputs, construct candidates, and select the final configuration |
| `dynamic_pipeline.evaluate_candidates` | Freeze calibration policy and orchestrate analysis, shortlisting, and simulation |
| `engines.DynamicEngine.analyze` | Compute trace-aware analytical scores and resource-demand diagnostics |
| `hybrid_screening.screen_des_candidates` | Select candidates using analytical scores and the performance objective |
| `closed_loop_sim.apply_closed_loop_simulation` | Simulate retained candidates and report final throughput and mean latency |
| `selection.apply_selection` | Apply the objective and constraints to completed results |
