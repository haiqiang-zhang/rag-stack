import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowRight, Database, Info, Loader2, RefreshCw, Search, TriangleAlert } from "lucide-react";
import { fetchCostModelCatalog } from "@/api/costModel";
import type { CostModelCatalog, CostModelInfo } from "@/types/costModel";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { groupByFamily } from "./families";
import { filterCatalogStages, groupBindings, type BindingGroup } from "./catalogModel";

interface CostModelMappingProps {
  models: CostModelInfo[];
  onOpenModel?: (modelName: string) => void;
}

function BindingRow({ binding, onOpenModel }: {
  binding: BindingGroup;
  onOpenModel?: (name: string) => void;
}) {
  const blocked = binding.status === "unsupported";
  return (
    <div className={cn("grid gap-3 p-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)]", blocked && "bg-amber-500/5")}>
      <div className="flex flex-wrap content-start gap-1.5">
        {binding.components.map((component) => (
          <span key={component} className="max-w-full break-all rounded bg-muted px-1.5 py-0.5 font-mono text-[11px]">{component}</span>
        ))}
      </div>
      <div className="min-w-0 space-y-1">
        <span className="block text-[10px] uppercase tracking-wide text-muted-foreground lg:hidden">Normalized stage</span>
        <p className="break-words font-mono text-[11px]">{binding.dispatch_stage}</p>
        {binding.db_type && <p className="break-words text-[11px] text-muted-foreground">db_type: {binding.db_type}</p>}
      </div>
      <div className="min-w-0 space-y-2">
        {binding.cost_model ? (
          <button type="button" onClick={() => onOpenModel?.(binding.cost_model!)} disabled={!onOpenModel}
            className="group flex max-w-full items-center gap-1.5 rounded text-left text-xs font-medium text-primary outline-none hover:underline focus-visible:ring-2 focus-visible:ring-ring">
            <span className="break-all font-mono">{binding.cost_model}</span>
            {onOpenModel && <ArrowRight className="size-3 shrink-0" />}
          </button>
        ) : <span className="text-xs text-muted-foreground">No cost-model binding</span>}
        <Badge variant="outline" className={cn("text-[10px] font-normal", blocked && "border-amber-500/30 text-amber-700 dark:text-amber-300")}>
          {blocked ? "CM blocked" : binding.status === "priced" ? "Binding available" : "Unpriced"}
        </Badge>
      </div>
      <div className="min-w-0 space-y-1.5">
        {binding.performance_stages.map((phase) => (
          <div key={phase.name} className="flex flex-wrap items-baseline gap-x-1.5">
            <span className="break-all font-mono text-[11px]">{phase.name}</span>
            <span className="text-[10px] text-muted-foreground">{phase.resource}</span>
          </div>
        ))}
        {binding.reason && <p className={cn("text-xs leading-relaxed text-muted-foreground", blocked && "text-amber-800 dark:text-amber-200")}>{binding.reason}</p>}
        {!binding.performance_stages.length && !binding.reason && <span className="text-xs text-muted-foreground">No modeled cost phase</span>}
      </div>
    </div>
  );
}

/** Renders backend descriptors. Actual worker/device bindings belong to CM Preview. */
export function CostModelCatalogView({ catalog, models, onOpenModel, search = "" }: CostModelMappingProps & {
  catalog: CostModelCatalog;
  search?: string;
}) {
  const stages = filterCatalogStages(catalog, search);
  const registered = new Set(models.map((model) => model.name));
  const openRegisteredModel = onOpenModel
    ? (name: string) => { if (registered.has(name)) onOpenModel(name); }
    : undefined;
  return (
    <div className="space-y-8">
      <div className="overflow-hidden rounded-xl border">
        <div className="hidden grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.2fr)] gap-3 bg-muted/40 px-4 py-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground lg:grid">
          <span>Component</span><span>Normalized RAG-IR stage</span><span>Cost model</span><span>CM cost phases / constraints</span>
        </div>
        {stages.length === 0 ? (
          <p className="p-8 text-center text-sm text-muted-foreground">No bindings match your search.</p>
        ) : stages.map((stage) => (
          <section key={stage.stage} className="border-t first:border-t-0">
            <div className="flex flex-wrap items-center gap-2 border-b bg-muted/20 px-4 py-2.5">
              <h3 className="font-mono text-xs font-semibold">{stage.stage}</h3>
              {!stage.user_selectable && <span className="text-[10px] text-muted-foreground">Internal stage</span>}
            </div>
            <div className="divide-y divide-border/60">
              {groupBindings(stage.bindings).map((binding, index) => (
                <BindingRow key={index} binding={binding} onOpenModel={openRegisteredModel} />
              ))}
            </div>
          </section>
        ))}
      </div>
      {!search.trim() && (
        <section className="space-y-5">
          <div>
            <h2 className="text-lg font-semibold tracking-tight">Registered cost models</h2>
            <p className="mt-1 text-sm text-muted-foreground">Inspect model inputs and saved profiles. Project compatibility is checked when CM binds a compiled path.</p>
          </div>
          {groupByFamily(models).map((family) => (
            <div key={family.key} className="space-y-3">
              <h3 className="text-xs font-medium text-muted-foreground">{family.label}</h3>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {family.items.map((model) => (
                  <button key={model.name} type="button" onClick={() => onOpenModel?.(model.name)} disabled={!onOpenModel}
                    className="group flex min-w-0 flex-col gap-2 rounded-lg border p-4 text-left transition-colors hover:border-primary/40 hover:bg-muted/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                    <span className="flex w-full items-start justify-between gap-2">
                      <span className="break-all font-mono text-xs font-semibold">{model.name}</span>
                      <ArrowRight className="size-3.5 shrink-0 text-muted-foreground group-hover:text-primary" />
                    </span>
                    <span className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">{model.description}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
          {!models.length && <p className="text-sm text-muted-foreground">No cost models are registered.</p>}
        </section>
      )}
    </div>
  );
}

export function CostModelMapping({ models, onOpenModel }: CostModelMappingProps) {
  const [catalog, setCatalog] = useState<CostModelCatalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try { setCatalog(await fetchCostModelCatalog()); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  const counts = useMemo(() => ({ stages: catalog?.stages.length ?? 0, models: models.length }), [catalog, models]);

  if (loading && !catalog) return (
    <div role="status" className="flex flex-1 items-center justify-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />Loading CM bindings…</div>
  );

  return (
    <div className="min-w-0 flex-1 overflow-auto">
      <div className="mx-auto max-w-[1320px] space-y-6 px-4 py-7 pb-16 sm:px-8">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="space-y-1.5">
            <h2 className="text-xl font-semibold tracking-tight">RAG-IR to cost models</h2>
            <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">Explore the bindings used by RAG-CM. Exact operations, workers, devices, and calibration matches are resolved in your project’s CM Preview.</p>
          </div>
          <Button variant="outline" size="sm" onClick={() => void load()} disabled={loading} aria-label="Refresh cost-model catalog">
            <RefreshCw className={cn("size-3.5", loading && "animate-spin")} />Refresh
          </Button>
        </div>
        <div className="grid gap-3 rounded-xl border bg-muted/15 p-4 sm:grid-cols-4">
          {[
            ["Resolved pipeline", "A concrete configuration"],
            ["Observed RAG-IR", "Real evaluator calls and dependencies"],
            ["CM compile + bind", "Cost models, deployment, calibration"],
            ["Compiled path", "Shared by analysis and simulation"],
          ].map(([label, detail], index) => (
            <div key={label} className="flex items-start gap-2.5">
              <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border bg-background text-[10px] text-muted-foreground">{index + 1}</span>
              <div><p className="text-xs font-medium">{label}</p><p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{detail}</p></div>
            </div>
          ))}
        </div>
        <div className="flex items-start gap-2 text-xs leading-relaxed text-muted-foreground"><Info className="mt-0.5 size-3.5 shrink-0" /><p>A binding or a saved profile does not establish that a project is ready. CM Preview validates the observed workload against its system and calibration.</p></div>
        {error && <div role="alert" className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive"><TriangleAlert className="mt-0.5 size-4 shrink-0" /><div><p className="font-medium">Could not load CM bindings</p><p className="mt-1 break-words text-xs">{error}</p><Button variant="outline" size="sm" className="mt-3" onClick={() => void load()}>Try again</Button></div></div>}
        {catalog && <>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-xs text-muted-foreground"><Database className="size-3.5" />{counts.stages} stages<span className="text-border">/</span>{counts.models} registered models</div>
            <div className="relative w-full sm:max-w-xs"><Search className="pointer-events-none absolute left-3 top-2.5 size-3.5 text-muted-foreground" /><Input aria-label="Filter cost-model bindings" placeholder="Find a stage, component, or model…" className="h-9 pl-9 text-xs" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
          </div>
          <CostModelCatalogView catalog={catalog} models={models} search={search} onOpenModel={onOpenModel} />
        </>}
      </div>
    </div>
  );
}
