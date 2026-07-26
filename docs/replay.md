# Replay a resolved configuration

Run one algorithm/system pair with caller-selected QA and corpus files:

```bash
python -m rag_stack replay configs/point/algo_config.json configs/point/system_config.json \
  --qa datasets/example/qa.parquet \
  --corpus datasets/example/corpus.parquet \
  --mode quality \
  --output-dir tmp_outputs/replays/example-quality
```

The two JSON files describe one concrete pipeline and deployment. They contain
no dataset selection or optimizer search space. Dataset paths are required
command-line arguments, resolved relative to the current directory. The same
pair can be reused with a different training or test cohort. Record the
original cohort and any preparation/sampling in the campaign's `RUN.md`.
The evaluator treats QA as the supplied evaluation cohort; it does not infer a
training/test split or train models.

| Mode | Execution | Result |
| --- | --- | --- |
| `quality` | Run the selected evaluator and configured metrics | Scalar scores and available per-question score vectors |
| `measured` (default) | Use the existing local vLLM projection and `MeasuredProvider`, without judge calls | Measured score, QPS and runtime measurements |
| `cm` | Run a fresh dynamic evaluator trace with scoring disabled, validate RAG-IR, call `RAGCMAssembly.evaluate_fixed` | CM prediction for the supplied deployment |

Measured mode preserves configured generation parameters while projecting API
generation modules onto the local measured deployment. It requires `static_gt`.
CM supports the same component set as the production cost model; unsupported
operations fail explicitly. CM mode executes the real pipeline to collect its
workload, so it may load models or contact configured generation endpoints.
Neither measured nor CM results are inferred from a previous benchmark score.
FAISS CM replay also requires `--dataset-name <corpus-id>` to select the existing
corpus/embedding workload profiles. Use the identity recorded in `RUN.md`; do
not infer it from a file name. This is invocation metadata, kept outside both
JSON files. The host binds it only to the dataset owner and CM's runtime
context. Calibration/workload profiles must be available in the runtime
environment; the command does not recreate a historical calibrated environment.
Quality mode requires an explicit, non-empty metric list. Some historical
performance-only configurations contain no scoring protocol; those cannot
reproduce quality until a caller supplies one. The command reports this rather
than returning an empty successful quality result.

`--dry-run` checks the resolved input shape and the existence of the external
Parquet paths without loading models or creating an output directory. It does
not validate Parquet contents, model availability, credentials or deployment
feasibility. The evaluator validates the data when execution begins.

The host adapts the selected pipeline into the dataset owner's schema using
`dataset_owner_config_from_resolved`. This lets a raw corpus use the selected
chunker and recomputes token statistics for the supplied data. The evaluator
still receives the resolved pipeline, never the owner's search-space-shaped
internal document. See the
[input contract](https://github.com/haiqiang-zhang/RAG-Stack-Evaluator/blob/main/docs/input_contract.md) for exact fields,
QA/corpus schemas and the Python API.

Each invocation uses a fresh temporary project, removed when it finishes. The
output contains `<mode>_result.json` and `RUN.md`; the latter records the command,
dataset arguments and any failure. The default destination is a new directory
under `tmp_outputs/replays/`. Exit status is nonzero for execution errors or an
inadmissible measured result. No file hashes, verification receipts, archived
trace or copied source files are needed.

The older [`eval-one`](eval_one.md) command consumes legacy project artifacts
that embed dataset metadata. Use `replay` for the dataset-independent benchmark
archive contract.
