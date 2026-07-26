---
title: Sequential Hybrid Retrieval
nav_order: 4.1
---

# Sequential hybrid retrieval

The `static_gt` evaluator supports this serial pipeline:

```text
query expansion (optional) → dense retrieval → BM25 → HybridRRF
    → reranker (optional) → compressor (optional) → prompt → generator
```

Dense retrieval and BM25 run in the declared order for each request. HybridRRF
consumes both completed candidate lists. Parallel dense/BM25 execution is a
separate extension and is not enabled by this configuration.

## Configure and run

Use the complete example `configs/rag_stack/hybrid_sequential.yaml` as a starting
point. Adjust its dataset, cached models, index paths, and hardware settings to
the target machine. It uses the existing six-section configuration format and
`system.performance_source: measured`. The three retrieval nodes are:

```yaml
node_lines:
  - node_line_name: rag
    nodes:
      - stage: semantic_retrieval
        top_k: [16, 32]
        modules:
          - component: vectordb
            vectordb: dense
      - stage: lexical_retrieval
        top_k: [16, 32]
        modules:
          - component: bm25
      - stage: hybrid_retrieval
        top_k: [8, 16]
        modules:
          - component: hybrid_rrf
            weight: [30, 60]
```

Each node has its own candidate limit. `weight` is the existing RRF rank
constant, not a coefficient blending dense and lexical scores. Scalar values,
choice lists, and supported range expressions retain their normal search-space
meaning. The optimizer resolves each choice before execution.

This integration requires one dense, one lexical, and one hybrid node, in that
order. Their dependencies cannot be optional. It rejects ReAct hybrid execution,
repeated retrieval nodes, `hybrid_cc`, and parallel policies. Aliases are
normalized at the configuration boundary; persisted plans use canonical stage
names. Standalone evaluator lexical execution retains its existing behavior;
lexical-only host optimization is outside this integration.

Measured execution supports BM25 and RRF as CPU operations. They do not receive
GPU model allocations. Dense encoding and generation use their configured
devices. BM25 analytical cost modeling is currently unavailable, so selecting
`performance_source: cost_model` fails explicitly. The analytical adapter also
lacks a validated scheduling binding for the dense/lexical join; the existing
local dense RRF model does not supply that policy. A missing component model
must never become zero cost. Valid execution and a valid RAG-IR trace do not
imply that CM can price that pipeline.

## Candidate and scoring contract

Dense and lexical results retain their separate provenance. HybridRRF consumes
those lists and publishes the active candidate IDs, scores, and contents used by
downstream stages. The evaluator also copies a neutral retrieval snapshot before
reranking or compression. Retrieval metrics use this snapshot, so hybrid quality
measures the fused retrieval results; generation uses the final downstream
context. Mutating downstream candidates cannot rewrite the retrieval snapshot.

The bundled dense and BM25 implementations each perform their own local RRF
merge, including a single-list fast path. HybridRRF then merges the two final
retrieval lists. The trace records every actual merge once. It does not infer
fusion calls from the configured number of expanded queries.

## Contracts between modules

| Module | Responsibility |
| --- | --- |
| RAG-Stack configuration and search | Normalize aliases, validate dependencies and capabilities, resolve search choices |
| RAG-Stack-Evaluator | Execute components, preserve candidate provenance, compute metrics, record real operation inputs and outputs |
| RAG-IR | Validate and serialize workflow structure, ordered calls, identities, producer references, and I/O profiles |
| Performance evaluation (PE) | Run the supported measured pipeline using its physical resources; retain per-request serial ordering |
| Cost model (CM) | Check pricing support, derive algorithm work from observed calls, and apply available system models |
| Apps | Render the backend plan and observed quality execution; preserve configuration semantics during editing |

`WorkflowSchema.nodes` declares stable `node_id`, `stage`, `components`, named
`inputs`, and `operations`. `execution_order` lists every node in order and
`execution_policy` is `sequential`. Describing a workflow does not import or
construct a cost model, optimizer, or evaluator runtime.

Every newly recorded call has a query-local `call_id`. Optional `node_id` and
`input_refs` fields identify the configured node and earlier producer calls in
the same query. The retrieval/fusion contract requires these identities and
references; other existing stage adapters may omit them. A hybrid call has two
ports, `dense` and `lexical`, each referring to its
producer's latest result. Its input list sizes must match those producers'
actual output counts. Raw candidate IDs remain in evaluator data; the public
trace contains only references and aggregate I/O profiles.

The same validation applies at evaluator result normalization, quality-store
loading, replay, and the observed-trace API. Legacy traces remain readable for
their original workflows, but an incomplete dense-only trace cannot stand in
for a hybrid run. Recollect it to obtain missing observations and identities.
See the [evaluator API](evaluator_api.md) for the public hooks.

## Planned and observed views

The editor requests the structural plan from `POST /api/rag-project/plan`.
Moving a node on the canvas changes its layout only; explicit order controls
change pipeline execution order. The plan includes capability errors before a
run starts. It does not claim that configured operations have executed.

An evaluation's observed view reads its persisted quality execution trace from
`GET /api/rag-project/evaluations/{eval_id}/trace?project={name}`. It displays actual calls,
producer references, and observed counts per query. Missing observations remain
unavailable. This view is not a measured timing timeline or a CM prediction.
