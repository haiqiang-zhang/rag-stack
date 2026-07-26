import { useState } from "react";
import type {
  CMCompiledPlan,
  CMOperation,
  CMRagIR,
  CMResource,
  CMTraceCall,
} from "@/api/cmPreview";
import {
  jsonText,
  operationSources,
  rawQueryForCompiledQuery,
} from "./cmPreviewModel";

export function JsonDetails({
  title,
  value,
  open = false,
}: {
  title: string;
  value: unknown;
  open?: boolean;
}) {
  return (
    <details
      open={open}
      className="min-w-0 rounded-md border bg-background px-3 py-2"
    >
      <summary className="cursor-pointer text-xs font-medium">{title}</summary>
      <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-all font-mono text-[11px] text-muted-foreground">
        {jsonText(value)}
      </pre>
    </details>
  );
}

function ResourceTable({
  title,
  resources,
}: {
  title: string;
  resources: CMResource[];
}) {
  return (
    <section className="min-w-0 rounded-lg border bg-card p-4">
      <h3 className="mb-3 text-sm font-semibold">{title}</h3>
      {resources.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No {title.toLowerCase()} in this compiled path.
        </p>
      ) : (
        <div className="overflow-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="p-2">Resource / role</th>
                <th className="p-2">Devices</th>
                <th className="p-2">Batching / capacity</th>
              </tr>
            </thead>
            <tbody>
              {resources.map((resource) => (
                <tr key={resource.id} className="border-t align-top">
                  <td className="p-2">
                    <div className="font-mono">{resource.id}</div>
                    <div>{resource.name}</div>
                    {resource.role !== undefined && (
                      <div className="text-muted-foreground">
                        {String(resource.role)}
                      </div>
                    )}
                  </td>
                  <td className="p-2 font-mono">
                    {jsonText(resource.devices)}
                  </td>
                  <td className="space-y-1 p-2">
                    {Object.entries(resource)
                      .filter(
                        ([key]) =>
                          !["id", "name", "devices", "role"].includes(key),
                      )
                      .map(([key, value]) => (
                        <div key={key}>
                          <span className="text-muted-foreground">{key}: </span>
                          <code className="break-all">
                            {JSON.stringify(value)}
                          </code>
                        </div>
                      ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function RawCall({
  call,
  index,
  highlighted,
}: {
  call: CMTraceCall;
  index: number;
  highlighted: boolean;
}) {
  return (
    <div
      className={`rounded-md border p-3 ${highlighted ? "border-primary bg-primary/5" : "bg-background"}`}
    >
      <div className="flex flex-wrap items-baseline gap-2 text-xs">
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]">
          index {index} · step {call.step_idx ?? "not recorded"}
        </span>
        <span className="font-semibold">
          {call.stage ?? "Stage not recorded"}
        </span>
        <span className="break-all font-mono text-muted-foreground">
          {call.call_id ?? "No call ID (legacy trace)"}
        </span>
      </div>
      {call.node_id && (
        <p className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
          Node: {call.node_id}
        </p>
      )}
      <div className="mt-2">
        <JsonDetails title="Recorded call fields" value={call} />
      </div>
    </div>
  );
}

function Operation({
  operation,
  selected,
  onSelect,
}: {
  operation: CMOperation;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      className={`rounded-md border p-3 ${selected ? "border-primary bg-primary/5" : "bg-background"}`}
    >
      <button
        type="button"
        onClick={onSelect}
        aria-pressed={selected}
        className="flex w-full flex-wrap items-baseline gap-2 text-left text-xs hover:text-primary"
      >
        <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]">
          {operation.kind}
        </span>
        <span className="font-semibold">{operation.name}</span>
        <span className="break-all font-mono text-muted-foreground">
          {operation.op_id}
        </span>
      </button>
      <dl className="mt-2 space-y-1 break-all text-[11px]">
        <div>
          <dt className="inline text-muted-foreground">Source call IDs: </dt>
          <dd className="inline font-mono">
            {operation.source_call_ids.length
              ? operation.source_call_ids.join(", ")
              : "None recorded"}
          </dd>
        </div>
        <div>
          <dt className="inline text-muted-foreground">
            Source call indices:{" "}
          </dt>
          <dd className="inline font-mono">
            {operation.source_call_indices.length
              ? operation.source_call_indices.join(", ")
              : "None"}
          </dd>
        </div>
        <div>
          <dt className="inline text-muted-foreground">Depends on: </dt>
          <dd className="inline font-mono">
            {operation.depends_on.join(", ") || "None"}
          </dd>
        </div>
        <div>
          <dt className="inline text-muted-foreground">Resources: </dt>
          <dd className="inline font-mono">
            {operation.resource_ids.join(", ") || "None"}
          </dd>
        </div>
      </dl>
      <div className="mt-2">
        <JsonDetails
          title={
            operation.kind === "transfer"
              ? "Communication details"
              : "Service / token details"
          }
          value={operation.details}
        />
      </div>
    </div>
  );
}

/** Displays only compiler output. No front-end config-to-plan projection. */
export function CMCompiledPlanView({
  plan,
  ragIR,
  compiledPathStatus,
}: {
  plan: CMCompiledPlan | null;
  ragIR: CMRagIR | null;
  compiledPathStatus?: "available" | "unavailable" | null;
}) {
  const [queryIndex, setQueryIndex] = useState(0);
  const [selectedOpId, setSelectedOpId] = useState<string | null>(null);
  const rawQueries = ragIR?.queries ?? [];
  const validPlan =
    plan?.schema_version === "rag-cm.compiled-plan.v1" ? plan : null;
  const queries = validPlan?.queries ?? rawQueries;
  const safeIndex = Math.min(queryIndex, Math.max(0, queries.length - 1));
  const query = queries[safeIndex];
  const rawQuery =
    validPlan && query
      ? rawQueryForCompiledQuery(rawQueries, query.question_id, safeIndex)
      : rawQueries[safeIndex];
  const calls = rawQuery?.calls ?? [];
  const operations = validPlan?.queries[safeIndex]?.operations ?? [];
  const selectedOperation = operations.find(
    (operation) => operation.op_id === selectedOpId,
  );
  const highlightedIndices = new Set(
    selectedOperation
      ? operationSources(selectedOperation, calls).map((entry) => entry.index)
      : [],
  );

  return (
    <div className="space-y-4">
      {!plan && compiledPathStatus === "unavailable" && <p role="status" className="rounded-lg border bg-card p-4 text-sm text-muted-foreground">No saved compiled path preview is available for this job. Create a new preview to inspect its compiled path. The recorded RAG-IR and job history remain available.</p>}
      {plan && !validPlan && (
        <p
          role="alert"
          className="rounded-md border border-amber-500/40 p-3 text-xs"
        >
          This compiled path response format is not supported by this client.
          Update the app to view the compiler output.
        </p>
      )}
      {validPlan && (
        <div className="rounded-lg border bg-card p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-sm font-semibold">RAG-CM compiled path</h3>
            <span className="rounded-full border px-2 py-0.5 text-[10px] text-muted-foreground">Display only</span>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {validPlan.queries.length} compiled queries ·{" "}
            {validPlan.resources.n_cards} GPU cards ·{" "}
            {validPlan.resources.cpu_cores} CPU cores. Compilation completed;
            latency and throughput have not been predicted.
          </p>
        </div>
      )}
      {(queries.length > 0 || ragIR) && (
        <section className="rounded-lg border bg-card p-4">
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <label className="text-xs font-medium">
              Query
              <select
                aria-label="Preview query"
                value={safeIndex}
                onChange={(event) => {
                  setQueryIndex(Number(event.target.value));
                  setSelectedOpId(null);
                }}
                className="ml-2 max-w-72 rounded-md border bg-background px-2 py-1.5 font-mono text-xs"
              >
                {queries.map((item, index) => (
                  <option key={index} value={index}>
                    {String(item.question_id)}
                  </option>
                ))}
              </select>
            </label>
            <span className="text-xs text-muted-foreground">
              {calls.length} recorded calls
              {validPlan ? ` · ${operations.length} compiled operations` : ""}
            </span>
          </div>
          {validPlan && (
            <p className="mb-3 text-xs text-muted-foreground">
              Select an operation to highlight its source calls. Dependencies
              and resource assignments come from the compiler.
            </p>
          )}
          <div className="grid min-w-0 gap-4 xl:grid-cols-2">
            <div className="min-w-0">
              <h4 className="mb-2 text-xs font-semibold">
                Observed RAG-IR calls
              </h4>
              <div className="max-h-[640px] space-y-2 overflow-auto pr-1">
                {calls.map((call, index) => (
                  <RawCall
                    key={index}
                    call={call}
                    index={index}
                    highlighted={highlightedIndices.has(index)}
                  />
                ))}
                {calls.length === 0 && (
                  <p className="py-3 text-xs text-muted-foreground">
                    No recorded calls are available for this query.
                  </p>
                )}
              </div>
            </div>
            <div className="min-w-0">
              <h4 className="mb-2 text-xs font-semibold">
                Compiled operations
              </h4>
              <div className="max-h-[640px] space-y-2 overflow-auto pr-1">
                {operations.map((operation) => (
                  <Operation
                    key={operation.op_id}
                    operation={operation}
                    selected={selectedOpId === operation.op_id}
                    onSelect={() =>
                      setSelectedOpId(
                        selectedOpId === operation.op_id
                          ? null
                          : operation.op_id,
                      )
                    }
                  />
                ))}
                {!validPlan && compiledPathStatus !== "unavailable" && (
                  <p className="py-3 text-xs text-muted-foreground">
                    No compiled path is available. Recorded calls and
                    downloadable RAG-IR remain available if compilation is
                    blocked.
                  </p>
                )}
                {validPlan && operations.length === 0 && (
                  <p className="py-3 text-xs text-muted-foreground">
                    The compiler returned no operations for this query.
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>
      )}
      {validPlan && (
        <>
          <div className="grid min-w-0 gap-4 2xl:grid-cols-2">
            <ResourceTable
              title="Workers"
              resources={validPlan.resources.workers}
            />
            <ResourceTable
              title="Engines"
              resources={validPlan.resources.engines}
            />
          </div>
          <div className="grid min-w-0 gap-3 lg:grid-cols-2">
            <JsonDetails
              title="Resolved deployment and batching"
              value={validPlan.deployment}
            />
            <JsonDetails
              title="Calibration and pricing bindings"
              value={{
                calibration: validPlan.calibration,
                pricing: validPlan.pricing_binding,
                hardware: validPlan.hardware,
              }}
            />
            <JsonDetails
              title="Workload summary and normalization"
              value={validPlan.workload}
            />
            <JsonDetails title="Complete compiler response" value={validPlan} />
          </div>
        </>
      )}
      {ragIR && <JsonDetails title="Complete raw RAG-IR" value={ragIR} />}
    </div>
  );
}
