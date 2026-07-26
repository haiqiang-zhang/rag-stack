# FlashRAG integration

FlashRAG is an independent evaluator provider, pinned as the host's top-level
`FlashRAG/` submodule. Its adapter lives in `flashrag.integrations.rag_stack`.
It runs native FlashRAG pipelines, records serving component I/O through
RAG-Stack's global hooks, and returns the same `EvaluationResult` as other
backends. `RAG-Stack-Evaluator` provides `static_gt` separately.

## Installation and registration

Install the compatible host environment first, then install the pinned provider
from the RAG-Stack root:

```bash
git submodule update --init FlashRAG
uv pip install -e FlashRAG
```

FlashRAG's `setup.py` registers the adapter with the host:

```python
entry_points={
    "rag_stack.evaluators": [
        "flashrag = flashrag.integrations.rag_stack:BACKEND",
    ],
}
```

New plugins may equivalently declare this entry-point group in `pyproject.toml`,
as shown in the [extension guide](evaluator_extensibility.md).

Reinstall the editable package after an entry-point change; editing source files
alone does not refresh installed discovery metadata. Use the top-level checkout
recorded by the host gitlink. An older install pointing to
`RAG-Stack-Evaluator/FlashRAG` must be reinstalled from `FlashRAG/`.

`static_gt` remains the default backend and does not require FlashRAG. FlashRAG
supports dynamic cost-model replay; it does not provide the measured execution
runner. Local models still require their normal framework dependencies, model
files, and suitable hardware.

## The extension boundary

```text
RAG-Stack resolves a candidate and constructs the selected provider
    → flashrag.integrations.rag_stack
    → capture_trace around native pipeline.run(..., do_eval=False)
    → native component calls use trace_query / trace_batch / trace_stage
    → finish and publish the complete serving trace
    → native pipeline.evaluator.evaluate(...) outside capture
    → EvaluationResult(metrics, trace, workflow)
    → host validation, optimizer, and RAG-CM
```

The adapter owns dataset conversion, index orchestration, configuration
translation, pipeline selection, and result aggregation. RAG-Stack owns the
protocol, backend registry, hooks, and RAG-IR validation. Framework component
hooks record calls as they execute; no monitor-log conversion is required to
construct RAG-IR.

The adapter activates capture once around the serving invocation with native
quality evaluation deferred (`do_eval=False`). It finalizes and publishes the
trace, then invokes the native FlashRAG evaluator outside capture. Query scopes
associate calls with stable dataset IDs; repeated agent iterations remain
ordered calls in each query trace. Judge calls are excluded from serving cost.
The workflow describes all configured component stages, including conditional
branches that a particular run skips. Observed calls must belong to that
structure; an undeclared executed component is rejected.

The trace stores counts and optional byte sizes, not prompt text or token IDs.
The native accounting boundaries are:

| Path | Recorded I/O |
| --- | --- |
| Hugging Face generation | Non-padding input IDs after native tokenization/truncation; generated IDs including EOS, excluding batch padding and any decoder-start ID |
| vLLM generation | Exact prompt IDs submitted to the engine and completion IDs returned by its native output API |
| BM25 | Query IDs from the BM25 tokenizer after its own text processing; zero output tokens |
| Dense query encoder | Native non-padding model input; zero output tokens and embedding byte size |
| FAISS search | Zero tokens; exact input embedding and returned index/score-array byte sizes |
| Cross-encoder reranker | Actual query/document-pair token counts summed per owning query for each model call; zero output tokens |

Dense retrieval records encoding and vector search as separate stages. Native
minibatches retain stable question attribution; no whitespace-based estimate or
fallback tokenizer is used for the public trace.

The hook API is the same one available to any other evaluator:

```python
from rag_stack.evaluation import trace_stage

# Inside a framework component, under its caller's capture and query scope:
with trace_stage("generator", model_id=model_id) as call:
    call.input(token_ids=prompt_ids, text=rendered_prompt)
    response = generate(prompt_ids)
    call.output(token_ids=response.token_ids, text=response.text)
```

This illustrates the component boundary; `generate` and its token IDs come from
the framework implementation. The [extension guide](evaluator_extensibility.md)
shows the outer capture, query scopes, finalization, and backend registration.
The [API reference](evaluator_api.md) defines token accounting and concurrency.

### Instrumented component paths

The pinned fork captures exact I/O for these native paths:

| Component | Instrumented runtime path |
| --- | --- |
| Generator | Hugging Face causal and encoder-decoder models; vLLM |
| Lexical retrieval | BM25 through `bm25s` |
| Dense retrieval | Native `Encoder` followed by FAISS search |
| Reranker | Native cross-encoder reranker |

A registered method still needs instrumented components for the concrete
configuration selected. Retrieval-cache shortcuts, sentence-transformer/CLIP
encoders, sparse or multimodal retrieval, multi-retriever routing, external
search APIs, FiD generation, refiners, OpenAI generation, and bi-encoder
reranking are unsupported. These paths must fail
before a result is returned rather than produce an incomplete trace. Add exact
component accounting before using those paths with the RAG-Stack backend. A method appearing in capability metadata
does not grant instrumentation support to every framework/model combination.
`adaptive_rag` is not registered because its serving-time complexity judge is
not represented by the current stage vocabulary. CoRAG currently requires
`framework: vllm`. Beam search and multiple
completions are also unsupported. For vLLM, token truncation must occur before
the captured call; its internal `truncate_prompt_tokens` option is rejected.

The current RAG-IR contract requires each query to end with a real generator
call. At an iteration limit, native IRCOT, Self-Ask, SearchR1, ReAct, or A-RAG
execution can end after retrieval instead. Those executions fail trace
finalization; the adapter does not add a synthetic final call or a fallback
trace. An advertised method is dispatchable, but every concrete execution must
still satisfy this terminal-call requirement. The sequential CPU validation
case finishes with generation and is unaffected.

## Construct and call the backend

```python
from rag_stack.evaluation import create_quality_evaluator

evaluator = create_quality_evaluator(
    dataset_manager=dataset_manager,
    project_dir=dataset_manager.project_dir,
    backend="flashrag",
    backend_options={"framework": "vllm"},
)
result = evaluator.evaluate(
    resolved_pipeline_config,
    run_dir=run_dir,
    metrics_override=None,
    on_trace_ready=None,
)
scores = result.metrics
trace = result.trace
```

Supply the caller-owned dataset manager. It provides `qa_data`, `corpus_data`,
`project_dir`, corpus resolution, and dataset activation. The adapter does not
construct a static evaluator or import `rag_stack_evaluator` to acquire data.

`metrics_override` replaces metrics for that evaluation without changing the
caller's configuration. The provider returns a complete canonical trace with
exact question coverage and finite metric values. `on_trace_ready`, if supplied,
receives the same trace object returned in the result before quality scoring
begins. The adapter calls the native FlashRAG evaluator after trace publication,
allowing RAG-CM to overlap with scoring and keeping judge calls outside capture.

## Resolved candidate and configuration

`evaluate` accepts one concrete pipeline, not an optimizer search space. The
host resolves `algo_search_space.pipeline.methods` before invoking the provider.

| Field | Meaning |
| --- | --- |
| `node_lines` | Concrete retrieval and generation nodes, each with one selected module |
| `vectordb` | Resolved embedding and index settings; the adapter uses the first store |
| `corpus_runtime.chunker` | A concrete chunker, or `{}` for pre-chunked data |
| `pipeline_runtime.mode` | The selected native method |
| Other `pipeline_runtime` keys | Resolved method options such as iteration limits |
| `eval_backend_setting.metrics` | FlashRAG metric names or mappings containing `metric_name` |
| `eval_backend_setting.backend_options` | Provider construction options |
| `eval_backend_setting.flashrag` | Existing FlashRAG settings block, retained for compatibility |

QA rows use `qid`, `query`, and accepted answers in `generation_gt`; corpus rows
use `doc_id` and `contents`. The adapter materializes FlashRAG datasets with
stable IDs. Retrieval metrics use FlashRAG's semantics; legacy passage-ID
`retrieval_gt` is not automatically translated.

Query the installed descriptor for authoritative capabilities:

```python
from rag_stack.evaluation import supported_metrics, supported_pipeline_modes

metric_names = supported_metrics("flashrag")
method_names = supported_pipeline_modes("flashrag")
```

The provider declares sequential, iterative, and agentic methods. Names and
configuration belong to FlashRAG; selecting a different backend does not
translate another evaluator's metric vocabulary. `rouge_l` is accepted as an
alias for FlashRAG's `rouge-l`.

The following is an overlay on a complete optimization configuration:

```yaml
global:
  eval_backend: flashrag
  rag_ir_mode: dynamic
system:
  performance_source: cost_model
eval_backend_setting:
  metrics: [em, f1, rouge-l]
  backend_options:
    framework: vllm
    gpu_memory_utilization: 0.85
    model2path:
      e5: intfloat/e5-base-v2
```

Keep the dataset, hardware, optimizer, and method search space appropriate to
your workload. The legacy `monitor: full` and `monitor: light` settings are aliases for the
same exact capture, and recording
is now provided by the global hooks. `monitor: off` is rejected: this registered
backend requires a complete serving trace.

`pipeline_runtime.mode` selects the concrete method. Per-call FlashRAG settings
can override constructor options; the constructor's `dataset_name` determines
the materialized QA name. Without that option, the QA content hash supplies the
name. Reusing a fixed name requires keeping its QA content consistent.

## Artifacts and lifecycle

The caller selects writable `project_dir` and `run_dir` locations. The provider
stores materialized data and indexes under the project:

```text
<project_dir>/
  _flashrag/
    datasets/<dataset-name>/test.jsonl
    corpus/<corpus-hash>/corpus.jsonl
    indexes/<corpus-hash>/...
  _flashrag_run/                     # Default when run_dir is omitted.
```

Corpus paths incorporate content hashes; index reuse also depends on retrieval
method and index settings. These files are separate from static-provider caches
under `RAG_STACK_CACHE_DIR`.

Capture and query scopes are restored when execution ends, including failures.
The provider owns its runtime lifecycle; RAG-Stack owns optimization scheduling
and validation of returned results. A failed or incomplete invocation must not
be reported as a successful quality score.

## CPU validation

Run a native pipeline in a separate CPU environment from the host root:

```bash
python -m venv /tmp/flashrag-rag-stack-cpu
source /tmp/flashrag-rag-stack-cpu/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r FlashRAG/requirements-rag-stack-cpu.txt
python -m pip install --no-deps -e . -e FlashRAG
python FlashRAG/scripts/rag_stack_cpu.py --output-dir /tmp/flashrag-rag-stack
```

The script uses FlashRAG's native BM25 retriever, `SequentialPipeline`, a
pretrained `sshleifer/tiny-gpt2` generator, and FlashRAG's EM/F1 implementations.
It computes new responses and scores while the component hooks record their
actual serving I/O. The model and tokenizer are downloaded on first use. The
small model and corpus exercise the integration; their scores are not benchmark
results or evidence of production-model quality.

Provider and native component tests are also available:

```bash
python -m pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest FlashRAG/tests/test_rag_stack_integration.py \
  FlashRAG/tests/test_native_rag_stack_tracing.py -q
```

Provider tests cover registration, native pipeline invocation, returned scores,
question coverage, and the shared tracing contract. Native component tests use
small local Transformers models/tokenizers and CPU BM25/FAISS calls to verify
batching and exact token accounting. The vLLM test uses a mocked native-output
contract; it does not run a vLLM engine on CPU. These checks do not establish calibrated
component costs or GPU performance.
