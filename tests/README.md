# Tests

The default host suite uses synthetic inputs, temporary directories and fake
services. It does not launch models or require a running benchmark. Install the
repository and evaluator dependencies, then install the test extra with
`uv pip install -e '.[test]'`.

```bash
# All host tests, or one domain
python -m pytest
python -m pytest tests/contracts
python -m pytest tests/optimizer

# Concrete evaluator implementation: a separate project and test package
(cd RAG-Stack-Evaluator && python -m pytest)

# Frontend configuration transformations
npm --prefix apps test
```

`pyproject.toml` limits default collection to `tests/`, enables importlib import
mode and rejects unknown markers. Unrelated DeepEval/LangSmith pytest plugins
are disabled so installing a model evaluation dependency does not change this
suite's startup. Tests that use AnyIO run on asyncio.

On hosts whose Conda native libraries need an explicit search path, prefix the
command with `LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`.
This is a process-local environment fix; do not patch the system libraries.

## Ownership

| Directory | Coverage |
| --- | --- |
| `contracts/` | Public evaluator API, backend ownership, RAG-IR, tracing, pipeline and PE/CM boundaries |
| `config/` | Configuration validation, search-space decoding and deployment settings |
| `cost_model/` | CM components, simulation, placement and workload pricing |
| `calibration/` | Profile schemas, calibration inputs, fitting and publication |
| `optimizer/` | Search algorithms, observation accounting, budgets and exports |
| `evaluation/` | Host controller, quality storage, checkpoints and result publication |
| `data/` | Host dataset preparation and dataset-manager integration |
| `runtime/` | Host clients, CLI lifecycle and environment routing |
| `server/` | Registry endpoints and app-facing project operations |

Concrete retrieval, quality scoring, measured serving and backend process tests
belong in `RAG-Stack-Evaluator/tests/`; keep one copy there. Host tests cover the
public boundary and host consumers. Frontend tests live in `apps/tests/`.

## Adding or changing tests

- Test observable behavior and interface contracts, including rejected invalid
  inputs. Prefer fixtures that express the relevant settings over snapshots of
  a particular historical run.
- Use `tmp_path` and `monkeypatch` for files, caches and environment variables.
  Never depend on a user's credentials, live endpoint or workstation cache.
- Use a fresh subprocess for cold-import and process-global initialization
  checks. Do not require that another test has not imported a module.
- Put reusable fixtures in the domain's `conftest.py` or a support module;
  do not import another test module as a fixture library.
- Use `from tests import REPO_ROOT` when a host test needs checked-in inputs.
  Avoid assumptions about the test file's nesting depth.
- Keep model launches, data downloads and manual performance diagnostics outside
  pytest. They are explicit commands.

File checksums are not an experiment acceptance criterion. Preserve schema,
configuration, sample membership, measurement and numerical checks instead.
