---
title: Web App
nav_order: 7
---

# RAG-Stack Web App

The web app lives in [`apps/`](https://github.com/haiqiang-zhang/rag-stack/tree/main/apps):
a Vite + React 19 + shadcn/ui SPA, installable as a PWA from a production build.
Its FastAPI backend lives in
[`rag_stack/server/`](https://github.com/haiqiang-zhang/rag-stack/tree/main/rag_stack/server).

## Setup

Complete the host and pinned evaluator installation in the
[evaluator guide](evaluators.md) first. Run the backend in the same
`rag-stack` Conda environment used by the command-line tools. From the dev
repository root, install the frontend dependencies:

```bash
cd apps
npm install
```

The frontend selects a Python backend through the host registry. Static
quality and measured execution use `RAG-Stack-Evaluator`; selecting
`global.eval_backend: flashrag` requires the independently installed top-level
FlashRAG provider in the API server's environment:

```bash
# From the repository root, with the rag-stack environment active.
git submodule update --init --recursive
uv pip install -e FlashRAG
```

`static_gt` remains the default backend. The visual editor gets metric names
and supported FlashRAG methods from the host registry and each selected provider's capability metadata.
Choosing a backend in the UI does not install its dependencies.

Verify discovery using the same Python environment that will run the API:

```bash
python -c 'from rag_stack.evaluation import available_backends; print(available_backends())'
```

The result must include `static_gt` after installing `RAG-Stack-Evaluator`.
If validation reports `Unknown global.eval_backend: 'static_gt'` and the
supported-backend list is empty, the API environment has no evaluator entry
points installed. An editable checkout can import successfully while its
installed metadata is stale. Follow
[Updating an existing installation](evaluators.md#updating-an-existing-installation)
to refresh that metadata, then restart the API. The UI displays the backend's
registry and validation results directly.

## Usage

Start the backend from the repository root:

```bash
conda activate rag-stack
uvicorn rag_stack.server.cost_model_api:app \
  --host 127.0.0.1 --port 8000 \
  --reload --reload-dir rag_stack \
  --reload-dir RAG-Stack-Evaluator/rag_stack_evaluator \
  --reload-dir FlashRAG/flashrag
```

The provider reload directories pick up Python changes in the editable
submodules. Omit the FlashRAG reload directory if that checkout is absent. Restart the backend after changing
installed dependencies or updating a submodule revision.

In a separate terminal, start the frontend from `apps/`:

```bash
conda activate rag-stack
cd apps
npm run dev -- --host 127.0.0.1 --port 2333
```

Open <http://localhost:2333>. In the connection screen, use
`http://localhost:8000` as the backend URL. The browser saves this URL and its
optional access token for later connections.

For PWA installation testing during development, use this frontend command
instead:

```bash
npm run dev:pwa -- --host 127.0.0.1 --port 2333
```

To build the frontend assets:

```bash
npm run build
```

The build writes `apps/dist/`. Hosting those static files still requires a
reachable API backend; the Python evaluator does not run in the browser.

## First evaluation

A newly created project opens **Start eval**, where its configuration can be
edited and saved before starting the first run. Select the QA and corpus files
as run inputs, validate the configuration, and save any changes before launch.
The evaluator's algorithm/system configuration contract is unchanged: training
and evaluation datasets are supplied separately at runtime.

The project configuration becomes read-only once a run starts, including runs
that fail before producing an evaluation. Prior optimizer state, evaluations,
run logs, and initialized run data also keep it read-only after a reset. Create
a new project to change a configuration with run history. Standalone CM preview
snapshots remain available and do not count as optimizer run history.

`GET /api/rag-project/config/{project_name}` returns the parsed `content`, exact
`raw_content`, `editable`, and `read_only_reason`. `PUT /api/rag-project/config`
accepts `{project_name, content}` with YAML text, validates it with the canonical
server schema without downloading models, and atomically saves eligible drafts.
The save and launch endpoints share a project lock so concurrent requests cannot
change configuration during first-run startup. An invalid config returns HTTP
400; a project that has already started returns HTTP 409.

This is a rag-stack core constraint shared by the API, CLI and Controller.
Before launching the evaluator, `config.freeze.json` records the exact input
YAML, including unexpanded environment placeholders. It survives reset and
failed startup. Subsequent runs reject config replacement, changed policy
overrides, or a project YAML that differs from that snapshot. No experiment
data hashes are involved. Use a new project for a different configuration.

## Project CM preview

Open a project and select **CM Preview** to run a concrete pipeline on a small
query cohort and inspect its real RAG-CM compilation. The view links observed
RAG-IR calls to physical workers, engines, communication and device bindings.
Previews have their own progress, logs, cancellation and downloadable artifacts;
their saved descriptions can be reopened after a server restart for inspection
only. CM execution never consumes these files, and viewing them does not run
the evaluator or compiler. Previews do not add optimizer evaluations. Unsupported CM operations retain their
observed trace and show the compiler's blocker. See the
[CM preview guide](cm_preview.md) for selection, reuse and interface contracts.

## Cost Models

`/cost-models` shows the backend's read-only dispatch catalog. Its rows come
from `GET /api/cost-model/catalog` (`rag-stack.cost-model-catalog.v1`), which
projects the production taxonomy, stage/component dispatch and CM support
rules. The browser does not construct its own stage expansion or pricing map.
Dense retrieval includes separate encode and vector-search bindings; the index
family selects the latter. Observed local RRF fusion has its own binding and
physical-worker constraints. Unsupported lexical/hybrid operations show their
CM blocker instead of being described as free pass-through operations.

The catalog describes possible bindings. A project's **CM Preview** supplies
the observed calls, concrete workers, prefill/decode engines, communication,
devices and selected pricing profiles. It uses the same
[compiled-path contract](rag_cm/compiled_plan.md) as RAG-CM execution.

The model estimator reports **component estimates**. The estimate endpoints
return `scope: "component_estimate"`; the sweep endpoint returns
`scope: "component_sweep"`. These results describe component service demand,
without project trace compilation, queueing or deployment simulation. Invalid
input, unsupported operations and missing calibration produce explicit errors.
An unavailable browser form does not fall back to another component's form.

Calibration lists are **artifact inventories**, not project coverage reports.
The API reads the same `calibration_profile_root()` and
`calibration_data_root()` as the runtime, including the
`RAG_STACK_CALIBRATION_PROFILE_DIR` and `RAG_STACK_CALIBRATION_DATA_DIR`
environment overrides. A file count alone cannot establish that hardware,
model, precision, corpus and scheduler requirements match a particular plan.
Use CM Preview to inspect that binding, and the
[calibration guide](rag_cm/calibration.md) for calibration commands.

The calibration launcher obtains its command choices and supported fields
from the backend's public CLI descriptors. Workload calibration requires a
project directory, with QA/corpus supplied as separate runtime inputs.
Unsupported or duplicate structured options fail before a job is launched.
Shared profile files are read-only in the per-component view: removing one
displayed record must not delete other components' calibration records.

## Sequential hybrid retrieval

The Pipeline editor represents hybrid retrieval as three ordered nodes:
`semantic_retrieval` → `lexical_retrieval` → `hybrid_retrieval`. Choose
`vectordb`, `bm25`, and `hybrid_rrf` as their respective modules and configure
each node's `top_k` independently. The BM25 and fusion parameter fields come
from the backend registry. **RRF rank constant** edits the existing `weight`
parameter; it is the positive rank smoothing constant, with a default of 60.

Nodes execute sequentially in their numbered order. Dragging a node or node
line changes its position on the canvas. Use the inspector's **Execute
earlier** and **Execute later** controls to change execution order. The editor
generates the connecting arrows from that order, and backend validation checks
that dense and lexical retrieval precede fusion. Numeric values, candidate
lists, and selector objects such as `{"range": [8, 64]}` remain structured
values when switching between the visual editor and YAML. The vector-index
field accepts a scalar reference or a JSON candidate selector.

Expand **Planned operations** in the Pipeline tab to inspect the backend's
logical node identities, input producers, operation stages, and sequential
execution order. This view comes from `POST /api/rag-project/plan`. Planned
operations describe the configuration; they do not assert that any call has
already run or supply runtime timings.

The registry and plan expose evaluator, measured-execution, and cost-model
support separately. Missing capability information is shown as **unknown**.
A registry request failure is displayed, and the editor preserves the loaded
configuration without supplying a fallback component list. BM25 currently has
no analytical cost model, so the hybrid pipeline requires measured performance
evaluation. An unavailable model is never displayed as zero cost. ReAct hybrid,
parallel retrieval, and `hybrid_cc` are outside the supported sequential
hybrid contract.

For an evaluated candidate, expand **Observed execution** below its resolved
pipeline. Select a question to inspect the actual recorded calls, including
BM25 and each RRF invocation, with `node_id`, `call_id`, producer references,
and input/output candidate counts. The view reads the saved quality trace
through `GET /api/rag-project/evaluations/{eval_id}/trace?project={name}`.
Dense-local, lexical-local, and cross-retriever fusion remain separate calls.
Legacy traces without identities or profiles show **not recorded**; missing
trace artifacts show an unavailable message. The app never reconstructs
observed calls or candidate counts from a configured pipeline.

## Connecting to a remote development host

With both services bound to loopback on the remote machine, forward their
ports from the computer running your browser:

```bash
ssh -N \
  -L 2333:127.0.0.1:2333 \
  -L 8000:127.0.0.1:8000 \
  USER@HOST
```

Open <http://localhost:2333> and enter `http://localhost:8000` as the backend
URL. Both forwards are needed for this direct API connection.

Vite also proxies `/api` requests to `http://localhost:8000` by default. To
use that development proxy, enter `http://localhost:2333` as the backend URL
in the connection screen. In this arrangement, only the frontend port needs
to be forwarded. The proxy target can be changed through `APP_API_URL` when
starting Vite.

## Configuration

| Setting | Purpose |
| --- | --- |
| `APP_PORT` | Vite development port; defaults to `2333`, and `--port` takes precedence. |
| `APP_API_URL` | Target for the Vite development server's `/api` proxy; defaults to `http://localhost:8000`. |
| `VITE_API_URL` | Initial browser-side backend URL; defaults to `http://localhost:8000`. A URL saved in the connection screen takes precedence. Set it before starting Vite or building assets. |
| `RAG_STACK_API_TOKEN` | Optional bearer token required by the Python API when set. Enter the same token in the connection screen. |
| `RAG_STACK_SETTINGS_PATH` | Optional path to the backend settings file. |

`APP_API_URL` does not change the browser's saved backend URL. If requests
still go to an old address, update the URL in the connection screen.

Settings are stored in `.rag_stack_settings.json` at the project root by
default. To use a custom path, set the environment variable before starting
the server:

```bash
export RAG_STACK_SETTINGS_PATH=/path/to/my_settings.json
```

API keys and secrets are stored in a `.env` file (default: project root). The
`.env` file path can be configured in the Settings UI under
**Keys & Secrets**. Both `.rag_stack_settings.json` and `.env` are in
`.gitignore`.

See the [user guide's System settings section](user_guide.md#system-settings)
for the settings reference and cache-directory layout. Backend-specific
configuration and the evaluator migration are described in the
[evaluator guide](evaluators.md).
