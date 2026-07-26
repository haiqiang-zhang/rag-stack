# CM compiled paths

A compiled preview is a real fixed-deployment CM compilation. The evaluator
first executes the selected pipeline and supplies canonical RAG-IR. CM then
validates that workload and binds it to the resolved deployment, hardware and
available pricing models. Compilation does not run the discrete-event simulator
and does not claim a throughput or latency prediction.

```python
from rag_stack.cost_model.assembly import RAGCMAssembly
from rag_stack.cost_model.compiled import (
    CompilationBlocked, describe,
)
from rag_stack.system_layout import cm_fixed_pipeline_config
from rag_stack.types import SystemConfig

hardware = SystemConfig.from_dict(
    cm_fixed_pipeline_config(algo_config, system_config)["system"]
)
assembly = RAGCMAssembly(hardware)
try:
    plan = assembly.compile_fixed(algo_config, system_config, trace=rag_ir)
except CompilationBlocked as exc:
    blocked = exc.to_dict()
else:
    preview = describe(plan)
    # Explicit, separate operations; compilation does not perform either.
    analytical = assembly.analyze_plan(plan)
    simulation = assembly.simulate_plan(plan)
```

`algo_config` is the concrete evaluator pipeline. The caller binds the actual
QA/corpus externally and supplies the resulting corpus statistics through the
existing workflow-schema contract. Training/test selection is not inferred from
the configuration. Reusing statistics from another dataset is invalid. See the
[execution-path contract](execution_path.md) and [architecture](architecture.md).

## Ownership and phase boundaries

`rag_stack.cost_model.compiled.compile_workload(resolved_pipeline, rag_ir)`
returns a `CompiledWorkload`: the validated, selected query cohort, source
question identities, ordered calls, workload summary and normalization record.
It accepts the canonical envelope; the compiler does not accept an archived
performance trace or reconstruct executed calls from configuration stages.
Its `prepared_paths` owns the validated call snapshot; `execution_paths` exposes
the same read-only list. Optimizer phase boundaries pass this runtime ownership
object through analysis and simulation, avoiding repeated logical-path
validation and copying. It does not change deployment or calibration binding.

`bind_plan(workload, hardware=..., candidate_row=...)` returns a
`CompiledCMPlan`. Its physical lowering is the production `build_sim_inputs`
implementation. The resulting simulator graph retains source invocations on
both worker items and LLM calls:

- Dense encode/search and observed local fusion remain one physical worker
  visit. Multiple visits remain distinct; absent fusion is never inserted.
- Each LLM call maps to prefill and decode work. Unified serving uses one
  engine; disaggregated serving retains its independent engines and explicit
  pair-specific KV transfers.
- Host boundaries, concrete devices, replica pools, admission gates, batching
  limits and frozen scheduler calibration come from the same bound objects
  consumed by simulation.
- An operation without a supported physical or pricing policy blocks
  compilation. In particular, BM25 currently reports `unsupported_component`;
  it is never dropped or assigned a zero service time.

Ordinary fixed dynamic evaluation binds this same plan before analysis. The
analytical method reads its logical-call projection and retained service
functions; simulation consumes its retained simulator inputs. Changing batch,
layout or another simulation-identity field after compilation is rejected.
Optimizer sweeps retain their existing analytical screening: shortlisted
candidates use the same physical compiler without materializing every rejected
candidate's complete graph. Static, aggregate-only estimation has no observed
query path and is outside this preview contract.

## Browser response contract

`describe(plan)` describes the live in-memory compiled path. The API sends this
description to the browser as JSON and may save it for later viewing. This
display projection is never an execution input. It returns `schema_version: "rag-cm.compiled-plan.v1"` with:

| Field | Meaning |
| --- | --- |
| `status` | `compiled`; this does not mean simulation or execution was validated |
| `workload` | Summary, normalization and original source question identities |
| `queries` | One ordered physical operation list per retained source query |
| `resources` | Actual workers, engines, devices, capacities and initial population |
| `deployment`, `hardware` | Bound deployment and hardware facts |
| `calibration`, `pricing_binding` | Frozen scheduler generation, component/LLM bindings and complete component profile values |
| `prediction` | `null`; predictions require a separate analysis/simulation call |

Every operation has `op_id`, `kind`, `name`, `source_call_ids`,
`source_call_indices`, `depends_on`, `resource_ids` and `details`. The kinds are
`worker`, `prefill`, `decode` and `transfer`. Source indices refer to that query's
original `queries[i].calls` array, indexed from zero. They are not the producer's
`step_idx` values, which may contain gaps or duplicates. Normalization preserves
call order and never removes calls. Legacy calls without IDs retain indices and an empty
ID list. No synthetic evaluator call IDs are invented. Transfer alternatives
for replica pairs declare `selection: "replica_pair"`; their decode operation
uses `dependency_policy: "selected_replica_pair"`, not an all-links barrier.
Worker operator samples are obtained from the actual batch operator callable;
`single_item_service_s` describes that sample, not end-to-end latency or a
prediction at the deployment's saturated batch.

Failures use `CompilationBlocked.to_dict()`:

```json
{
  "status": "blocked",
  "phase": "compile_workload",
  "code": "unsupported_component",
  "message": "The selected operation has no supported CM service model.",
  "details": {}
}
```

The caller should retain the real evaluator trace when compilation is blocked.

## Runtime object and saved preview

A `CompiledCMPlan` is a runtime object. Its graph and service functions remain
in the process that compiled it, and `analyze_plan(plan)` and
`simulate_plan(plan)` consume that object directly. There is no executable
serialization or restoration API.

`describe(plan)` provides a display projection for users and developers. Apps
persists it as `compiled_description.json` alongside the preview's configuration
snapshots, observed RAG-IR, job status and logs. It can be reopened or downloaded
after a server restart. Opening or downloading it only reads the saved
projection; these actions never invoke the evaluator, compiler, analysis or
simulation. The projection does not contain executable functions or restoration
inputs, and no runtime entrypoint consumes it. There is no executable
`compiled_plan.json` artifact.

Changing the pipeline requires compilation of a new workload; changing the
system or calibration requires a new binding through the runtime interfaces.
Lazy service functions check the profiles bound to the live object before and
after pricing. A changed profile blocks reuse with `pricing_binding_changed`,
including when its single-item price is unchanged but another batch would use
a different curve. These checks protect the live runtime object. A saved preview
remains a historical display of the path compiled at that time.
