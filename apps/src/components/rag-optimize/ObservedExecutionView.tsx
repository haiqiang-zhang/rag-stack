import { useEffect, useState } from "react";
import { fetchEvaluationTrace } from "@/api/ragProject";
import type { EvaluationTrace, TraceItemProfile } from "@/api/ragProject";
import { useProject } from "@/context/ProjectContext";
import { stageColor } from "@/lib/stages";

function profileText(profile?: TraceItemProfile): string {
  if (!profile) return "not recorded";
  const parts = [`${profile.items} items`];
  if (profile.unique_items !== undefined) parts.push(`${profile.unique_items} unique`);
  if (profile.list_sizes) parts.push(`lists [${profile.list_sizes.join(", ")}]`);
  return parts.join(" · ");
}

export function ObservedExecutionView({ evalId }: { evalId: number }) {
  const { currentProject } = useProject();
  const [open, setOpen] = useState(false);
  const [result, setResult] = useState<EvaluationTrace | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [queryIndex, setQueryIndex] = useState(0);
  useEffect(() => {
    setResult(null);
    setError(null);
    setQueryIndex(0);
    if (!open || currentProject === null) return;
    let cancelled = false;
    setLoading(true);
    fetchEvaluationTrace(currentProject, evalId)
      .then((next) => { if (!cancelled) setResult(next); })
      .catch((err: unknown) => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [currentProject, evalId, open]);
  const queries = result?.trace?.queries ?? [];
  const query = queries[queryIndex];
  const counts = new Map<string, number>();
  for (const call of query?.calls ?? []) counts.set(call.stage, (counts.get(call.stage) ?? 0) + 1);
  return (
    <details className="rounded-lg border border-border bg-card p-4" open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer text-xs font-semibold">Observed execution · Eval #{evalId}</summary>
      {open && <div className="mt-3 space-y-3 text-xs">
        <p className="text-muted-foreground">Recorded quality-evaluation calls and candidate counts, in execution order.</p>
        {loading && <p className="text-muted-foreground">Loading recorded calls…</p>}
        {error && <p role="alert" className="text-destructive">Trace unavailable: {error}</p>}
        {result && !result.available && <p className="text-muted-foreground">{result.reason ?? "No recorded trace is available for this evaluation."}</p>}
        {result?.available && queries.length === 0 && <p className="text-muted-foreground">The trace contains no queries.</p>}
        {query && <>
          <label className="flex items-center gap-2">Query
            <select aria-label="Recorded query" value={queryIndex} onChange={(event) => setQueryIndex(Number(event.target.value))}
              className="max-w-xs rounded border border-border bg-background px-2 py-1 font-mono text-xs">
              {queries.map((item, index) => <option key={index} value={index}>{String(item.question_id)}</option>)}
            </select>
            <span className="text-muted-foreground">{query.calls.length} recorded calls</span>
          </label>
          <div className="flex flex-wrap gap-2">
            {[...counts].map(([stage, count]) => <span key={stage} className="rounded border border-border px-2 py-1" style={{ color: stageColor(stage) }}>{stage}: {count}</span>)}
          </div>
          <div className="max-h-[440px] overflow-auto">
            <table className="w-full border-collapse text-left text-[11px]">
              <thead><tr className="border-b border-border text-muted-foreground">
                <th className="p-2">Step / operation</th><th className="p-2">Node / call</th>
                <th className="p-2">Input calls</th><th className="p-2">Input candidates</th><th className="p-2">Output candidates</th>
              </tr></thead>
              <tbody>{query.calls.map((call, index) => <tr key={call.call_id ?? index} className="border-b border-border/60 align-top">
                <td className="p-2"><span style={{ color: stageColor(call.stage) }}>{call.step_idx}. {call.stage}</span>
                  {call.model_id && <div className="mt-1 font-mono text-muted-foreground">{call.model_id}</div>}</td>
                <td className="p-2 font-mono">{call.node_id ?? "not recorded"}<div className="mt-1 text-muted-foreground">{call.call_id ?? "not recorded"}</div></td>
                <td className="p-2 font-mono">{call.input_refs === undefined ? "not recorded" : Object.entries(call.input_refs).length === 0 ? "none" : Object.entries(call.input_refs).map(([port, ids]) => <div key={port}>{port}: {ids.join(", ") || "none"}</div>)}</td>
                <td className="p-2">{profileText(call.input_profile)}</td>
                <td className="p-2">{profileText(call.output_profile)}</td>
              </tr>)}</tbody>
            </table>
          </div>
        </>}
      </div>}
    </details>
  );
}
