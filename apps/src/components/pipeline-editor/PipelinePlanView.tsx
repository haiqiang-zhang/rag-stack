import { useEffect, useState } from "react";
import yaml from "js-yaml";
import { fetchPipelinePlan } from "@/api/registry";
import type { PipelinePlan } from "@/api/registry";
import { CapabilityStatus } from "./panels/ModuleParamForm";

/** Config-derived operations are never presented as observed execution. */
export function PipelinePlanView({ yamlText }: { yamlText: string }) {
  const [plan, setPlan] = useState<PipelinePlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setPlan(null);
    setError(null);
    const timer = setTimeout(async () => {
      try {
        const config = yaml.load(yamlText);
        if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("Enter a configuration mapping.");
        const next = await fetchPipelinePlan(config as Record<string, unknown>);
        if (!cancelled) setPlan(next);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 600);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [yamlText]);
  const byId = new Map(plan?.workflow?.nodes?.map((node) => [node.node_id, node]));
  return (
    <details className="shrink-0 border-b border-border bg-card px-3.5 py-2 text-xs">
      <summary className="cursor-pointer font-medium">
        Planned operations{loading ? " · loading" : error || !plan?.workflow ? " · unavailable" : ` · ${plan.workflow.execution_policy}`}
      </summary>
      <div className="mt-2 max-h-56 overflow-auto">
        <p className="mb-2 text-[11px] text-muted-foreground">Backend plan for this configuration. Actual calls are recorded during evaluation.</p>
        {error && <p role="alert" className="text-destructive">{error}</p>}
        {plan && <>
          <CapabilityStatus capabilities={plan.capabilities} />
          {plan.errors?.map((message, i) => <p key={i} className="text-amber-700">{message}</p>)}
          <ol className="space-y-2">
            {plan.workflow?.execution_order?.map((id, index) => {
              const node = byId.get(id);
              if (!node) return null;
              return <li key={id} className="rounded border border-border p-2">
                <div className="font-medium">{index + 1}. {node.stage} <span className="font-mono text-muted-foreground">{node.components.join(" / ")}</span></div>
                <div className="mt-1 font-mono text-[10px]">{node.operations.join(" → ") || "No instrumented operation declared"}</div>
                {Object.entries(node.inputs).map(([port, producers]) => (
                  <div key={port} className="mt-1 text-[10px] text-muted-foreground">{port} ← {producers.join(", ") || "query input"}</div>
                ))}
              </li>;
            })}
          </ol>
        </>}
      </div>
    </details>
  );
}
