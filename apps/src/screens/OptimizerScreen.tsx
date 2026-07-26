/**
 * OptimizerScreen — top-level page for the Optimizer route.
 *
 * Three sub-tabs (rendered as shadcn <Tabs>):
 *   - Results     — view past benchmark CSVs (function picker + chart grid + summary table)
 *   - Run         — config browser + YAML editor + launch + jobs. Sweep-aware.
 *   - Optimizers  — registry of available optimizer types with description + matching configs
 *
 * Layout follows the same typography-driven, no-card-chrome pattern used in
 * Settings / Quality Stores. Section headers use h2 + description + <Separator>.
 *
 * Sweep awareness: if a config's parsed YAML has `optimizers:` as an array,
 * we detect it as a sweep, show how many optimizer × seed runs the backend
 * will execute, and label the entry in the side list accordingly. Single
 * configs render as before.
 */
import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import yaml from "js-yaml";
import {
  Loader2,
  Play,
  RefreshCw,
  Zap,
  GitBranch,
  Hash,
  Layers as LayersIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PageHeader, PageHeaderTitle } from "@/components/ui/PageHeader";
import { SidePanel, SidePanelHeader } from "@/components/ui/SidePanel";
import { SideItem, SideItemDot } from "@/components/ui/SideItem";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

import { useOptimizerRegistry } from "@/hooks/useOptimizerRegistry";
import { useOptimizerBenchmark } from "@/hooks/useOptimizerBenchmark";
import {
  fetchOptimizerConfigs,
  launchBenchmarkRun,
  fetchBenchmarkJobs,
  saveOptimizerConfig,
  killOptimizerJob,
} from "@/api/costModelClient";
import { deleteOptimizerBenchmarkResult } from "@/api/optimizer";
import type {
  BenchmarkConfig,
  BenchmarkJob,
  OptimizerBenchmarkSummary,
} from "@/types/optimizer";

import { BenchmarkChartGrid } from "@/components/optimizer/BenchmarkChartGrid";
import { SummaryTable } from "@/components/optimizer/SummaryTable";
import { EvalDistributionGrid } from "@/components/optimizer/EvalDistributionGrid";
import { useEvalDistribution } from "@/hooks/useEvalDistribution";
import { ConfigViewer } from "@/components/ConfigViewer";
import { ConfigEditor } from "@/components/ConfigEditor";
import { JobList, type Job } from "@/components/JobList";

type SubTab = "results" | "run" | "optimizers";

// ── Per-benchmark selection persistence ───────────────────────────────
// Stored shape: { [functionName]: optimizer[] }. Each benchmark's row
// selection in the SummaryTable is restored on page load and filtered
// against the optimizers currently present on disk — entries pointing
// at deleted CSVs are silently dropped.
const SELECTION_STORAGE_KEY = "optimizer.benchmark.selected_optimizers_v1";

function readSelectionStorage(): Record<string, string[]> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(SELECTION_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, string[]>;
    }
  } catch {
    /* corrupt JSON or quota-exceeded — fall through to default */
  }
  return {};
}

function writeSelectionStorage(map: Record<string, string[]>): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(SELECTION_STORAGE_KEY, JSON.stringify(map));
  } catch {
    /* localStorage may be disabled (incognito quota etc.) — ignore */
  }
}

// ── Sweep-vs-single classification ────────────────────────────────────

interface ConfigShape {
  kind: "sweep" | "single";
  optimizerCount: number;
  seedCount: number;
  syntheticFunctionCount: number;
  totalRuns: number;
  optimizers: Array<{ type: string; name?: string }>;
  syntheticFunctions: Array<{ type: string }>;
  iterations: number | null;
}

function classifyConfig(content: Record<string, unknown>): ConfigShape {
  const defaults = (content.defaults as Record<string, unknown>) ?? {};
  const seedsRaw =
    (defaults.seeds as unknown[] | undefined) ??
    (content.seeds as unknown[] | undefined) ??
    (content.seed != null ? [content.seed] : []);
  const seedCount = Array.isArray(seedsRaw) ? Math.max(seedsRaw.length, 1) : 1;
  const iterations =
    (defaults.n_iterations as number | undefined) ??
    (content.n_iterations as number | undefined) ??
    null;

  const optimizersList = content.optimizers;
  const synthFnsList = content.synthetic_functions;

  if (Array.isArray(optimizersList) && optimizersList.length > 0) {
    const synFns = Array.isArray(synthFnsList) && synthFnsList.length > 0
      ? (synthFnsList as Array<Record<string, unknown>>).map((f) => ({
          type: String(f.type ?? ""),
        }))
      : [{ type: String((content.synthetic_function as any)?.type ?? content.synthetic_function ?? "") }];
    const opts = (optimizersList as Array<Record<string, unknown>>).map((o) => ({
      type: String(o.type ?? ""),
      name: o.name ? String(o.name) : undefined,
    }));
    return {
      kind: "sweep",
      optimizerCount: opts.length,
      seedCount,
      syntheticFunctionCount: synFns.length,
      totalRuns: opts.length * synFns.length * seedCount,
      optimizers: opts,
      syntheticFunctions: synFns,
      iterations,
    };
  }

  return {
    kind: "single",
    optimizerCount: 1,
    seedCount,
    syntheticFunctionCount: 1,
    totalRuns: seedCount,
    optimizers: [
      { type: String((content.optimizer as any)?.type ?? "") },
    ],
    syntheticFunctions: [
      {
        type: String(
          (content.synthetic_function as any)?.type ??
            content.synthetic_function ??
            "",
        ),
      },
    ],
    iterations,
  };
}

// ── Section header (matches Settings / Quality Stores) ────────────────

function SectionHeader({
  title,
  description,
  action,
  className,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-4", className)}>
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
        {description && (
          <p className="text-sm text-muted-foreground">{description}</p>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

// ── Main screen ───────────────────────────────────────────────────────

export default function OptimizerScreen({ hideHeader = false }: { hideHeader?: boolean } = {}) {
  const { optimizers, loading: regLoading, error: regError } =
    useOptimizerRegistry();
  const [subTab, setSubTab] = useState<SubTab>("results");
  // Log regret = log-scale y-axis. Default ON because convergence usually
  // spans 2–3 orders of magnitude and linear hides the late-stage progress.
  const [logRegret, setLogRegret] = useState(true);
  const [selectedOptimizer, setSelectedOptimizer] = useState<string | null>(
    null,
  );
  const [selectedFunction, setSelectedFunction] = useState<string | undefined>(
    undefined,
  );
  const {
    rows,
    functions,
    summary,
    loading: benchLoading,
    refreshing: benchRefreshing,
    error: _benchError,
    refetch: refetchBench,
  } = useOptimizerBenchmark(selectedFunction);

  useEffect(() => {
    if (!selectedFunction && functions.length > 0) {
      setSelectedFunction(functions[0]);
    }
  }, [functions, selectedFunction]);

  // Optimizer names visible across all loaded benchmark rows, sorted —
  // mirrors BenchmarkChartGrid's optimizer ordering so chart colors and
  // table dots line up.
  //
  // Filter to `selectedFunction` defensively: useOptimizerBenchmark uses
  // stale-while-revalidate, so right after the user switches benchmark
  // the prior benchmark's rows linger until the async fetch resolves.
  // Without this filter, the restore effect below would run with the
  // PREVIOUS benchmark's optimizer list and clobber the new benchmark's
  // saved selection in localStorage. Empty allOptimizers → effect early-
  // returns → writeback waits until fresh rows arrive.
  const allOptimizers = useMemo(() => {
    const matching = selectedFunction
      ? rows.filter((r) => r.synthetic_function === selectedFunction)
      : rows;
    const set = new Set(matching.map((r) => r.optimizer));
    return Array.from(set).sort();
  }, [rows, selectedFunction]);

  // Selection set drives both the chart legend and the table checkboxes.
  // Persisted per-benchmark in localStorage so the user doesn't have to
  // re-pick each visit. Optimizers that no longer exist on disk (e.g. CSV
  // was deleted) are silently dropped on restore. Empty restored set →
  // default to "all selected".
  const [selectedOptimizers, setSelectedOptimizers] = useState<Set<string>>(
    () => new Set(),
  );
  const allOptimizersKey = allOptimizers.join("|");
  useEffect(() => {
    if (!selectedFunction || allOptimizers.length === 0) return;
    const stored = readSelectionStorage();
    const saved = stored[selectedFunction];
    let next: Set<string>;
    if (Array.isArray(saved) && saved.length > 0) {
      const allSet = new Set(allOptimizers);
      const filtered = saved.filter((s) => allSet.has(s));
      next = filtered.length > 0 ? new Set(filtered) : new Set(allOptimizers);
    } else {
      next = new Set(allOptimizers);
    }
    setSelectedOptimizers(next);
    // Write back the cleaned set so deleted-but-stored entries don't linger.
    stored[selectedFunction] = Array.from(next);
    writeSelectionStorage(stored);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allOptimizersKey, selectedFunction]);

  // Wrapper used by SummaryTable / chart toggles. Persists every change to
  // the current function's slot so a refresh restores exactly what was
  // checked. Bypasses the restore effect by writing directly — no double-
  // write race when selectedFunction changes.
  const setSelectedOptimizersPersist = useCallback(
    (next: Set<string>) => {
      setSelectedOptimizers(next);
      if (!selectedFunction) return;
      const stored = readSelectionStorage();
      stored[selectedFunction] = Array.from(next);
      writeSelectionStorage(stored);
    },
    [selectedFunction],
  );

  // Eval-distribution payload (per-eval (q,s) samples + true Pareto for the
  // currently selected synthetic function). Gated: the fetch only fires once
  // the section's anchor scrolls into view so the page paints + stays
  // responsive while the user works with the chart + table above. Reading
  // hundreds of evals CSVs synchronously on the backend thread can saturate
  // FastAPI's thread pool — keeping this fetch off the critical path keeps
  // tab switches and other endpoints snappy.
  const evalDistAnchorRef = useRef<HTMLDivElement | null>(null);
  const [evalDistVisible, setEvalDistVisible] = useState(false);
  useEffect(() => {
    if (evalDistVisible) return;
    const el = evalDistAnchorRef.current;
    if (!el || typeof IntersectionObserver === "undefined") {
      // Fallback: if IO isn't available, just enable on mount (next tick)
      // so we're no worse than the old behavior.
      const id = setTimeout(() => setEvalDistVisible(true), 0);
      return () => clearTimeout(id);
    }
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setEvalDistVisible(true);
          io.disconnect();
        }
      },
      // 200px rootMargin = start fetching just before the user scrolls to it
      { rootMargin: "200px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [evalDistVisible]);

  const {
    data: evalDist,
    loading: evalDistLoading,
    refetch: refetchEvalDist,
  } = useEvalDistribution(selectedFunction, { enabled: evalDistVisible });
  const [evalSeed, setEvalSeed] = useState<number | null>(null);
  useEffect(() => {
    if (evalDist && evalDist.seeds.length > 0) {
      setEvalSeed((cur) =>
        cur != null && evalDist.seeds.includes(cur) ? cur : evalDist.seeds[0],
      );
    } else {
      setEvalSeed(null);
    }
  }, [evalDist]);

  // Configs + jobs for Run tab
  const [configs, setConfigs] = useState<BenchmarkConfig[]>([]);
  const [jobs, setJobs] = useState<BenchmarkJob[]>([]);
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [editYaml, setEditYaml] = useState("");
  const [editDirty, setEditDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveOk, setSaveOk] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const loadConfigs = useCallback(async () => {
    try {
      setConfigs(await fetchOptimizerConfigs());
    } catch {
      setConfigs([]);
    }
  }, []);

  const loadJobs = useCallback(async () => {
    try {
      setJobs(await fetchBenchmarkJobs());
    } catch {
      setJobs([]);
    }
  }, []);

  useEffect(() => {
    loadConfigs();
    loadJobs();
  }, [loadConfigs, loadJobs]);

  // Poll jobs every 5s when on Run tab
  useEffect(() => {
    if (subTab !== "run") return;
    const interval = setInterval(loadJobs, 5000);
    return () => clearInterval(interval);
  }, [subTab, loadJobs]);

  // Sync editYaml when selectedConfig changes (not on config reload after save)
  const lastSaveRef = useRef(0);
  useEffect(() => {
    if (Date.now() - lastSaveRef.current < 2000) return;
    const cfg = configs.find((c) => c.filename === selectedConfig);
    if (cfg) {
      setEditYaml(yaml.dump(cfg.content, { indent: 2, lineWidth: 120 }));
      setEditDirty(false);
      setSaveOk(false);
    }
  }, [selectedConfig, configs]);

  const handleSaveConfig = async () => {
    if (!selectedConfig) return;
    try {
      setSaving(true);
      setSaveOk(false);
      setSaveError(null);
      await saveOptimizerConfig(selectedConfig, editYaml);
      lastSaveRef.current = Date.now();
      setSaveOk(true);
      setEditDirty(false);
      loadConfigs();
      setTimeout(() => setSaveOk(false), 2000);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  };

  const handleLaunch = async (filename: string) => {
    try {
      setLaunching(true);
      await launchBenchmarkRun(filename);
      loadJobs();
    } catch {
      /* ignore */
    } finally {
      setLaunching(false);
    }
  };

  // Configs matching selected optimizer (Optimizers tab)
  const matchingConfigs = useMemo(
    () => configs.filter((c) => c.optimizer_type === selectedOptimizer),
    [configs, selectedOptimizer],
  );

  // Unfiltered results (Optimizers tab uses these)
  const {
    rows: allRows,
    summary: allSummary,
    refetch: refetchAllBench,
  } = useOptimizerBenchmark(undefined);

  const handleDeleteBenchmarkRow = useCallback(
    async (row: OptimizerBenchmarkSummary) => {
      await deleteOptimizerBenchmarkResult(row.optimizer, row.synthetic_function);
      // Refetch the trajectory CSVs (Results + Optimizers tabs) AND the
      // eval-distribution payload, so the chart grid, summary table, and
      // eval panels stay in sync after a deletion.
      refetchBench();
      refetchAllBench();
      refetchEvalDist();
    },
    [refetchBench, refetchAllBench, refetchEvalDist],
  );
  const optimizerRows = useMemo(
    () => allRows.filter((r) => r.optimizer === selectedOptimizer),
    [allRows, selectedOptimizer],
  );
  const optimizerSummary = useMemo(
    () => allSummary.filter((s) => s.optimizer === selectedOptimizer),
    [allSummary, selectedOptimizer],
  );

  if (regLoading) {
    return (
      <div className="flex flex-1 items-center justify-center gap-3 bg-background">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }
  if (regError) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 bg-background p-5">
        <span className="text-[15px] font-semibold text-destructive">
          Failed to connect to API
        </span>
        <span className="text-xs text-muted-foreground">{regError.message}</span>
        <div className="mt-3 rounded-lg bg-muted p-3">
          <span className="break-all font-mono text-xs text-muted-foreground">
            uvicorn rag_stack.server.cost_model_api:app --port 8000 --reload --reload-dir rag_stack
          </span>
        </div>
      </div>
    );
  }

  return (
    <Tabs
      value={subTab}
      onValueChange={(v) => setSubTab(v as SubTab)}
      className="flex flex-1 min-w-0 flex-col gap-0 bg-background"
    >
      {/* Header — title left, tab triggers right. When embedded in the
          Benchmark page, the title + mode toggle live in the parent, so we
          render just the sub-tabs in a slim bar. */}
      {hideHeader ? (
        <div className="flex items-center justify-end border-b border-border bg-card px-4 py-1.5">
          <TabsList>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="run">Run</TabsTrigger>
            <TabsTrigger value="optimizers">Optimizers</TabsTrigger>
          </TabsList>
        </div>
      ) : (
        <PageHeader>
          <PageHeaderTitle icon={Zap} title="Optimizer" />
          <TabsList>
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="run">Run</TabsTrigger>
            <TabsTrigger value="optimizers">Optimizers</TabsTrigger>
          </TabsList>
        </PageHeader>
      )}

      {/* ── Results ───────────────────────────────────────────── */}
      <TabsContent value="results" className="flex-1 min-w-0 overflow-auto m-0">
        <div className="mx-auto max-w-[1200px] px-8 pt-10 pb-20">
          <SectionHeader
            title="Benchmark Results"
            description="HV-diff and IGD across optimizers, plotted vs cumulative GT calls and wall time. One chart-grid per benchmark function."
            action={
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <Label htmlFor="log-regret-toggle" className="text-xs font-medium">
                    Log regret
                  </Label>
                  <Switch
                    id="log-regret-toggle"
                    checked={logRegret}
                    onCheckedChange={setLogRegret}
                  />
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    refetchBench();
                    refetchEvalDist();
                  }}
                  aria-label="Reload"
                  disabled={benchRefreshing}
                  className="size-8"
                >
                  <RefreshCw
                    className={cn(
                      "size-4 text-muted-foreground",
                      benchRefreshing && "animate-spin",
                    )}
                  />
                </Button>
              </div>
            }
          />
          <Separator className="my-6" />

          {functions.length > 0 && (
            <div className="mb-6 flex flex-wrap gap-1.5">
              {functions.map((fn) => (
                <button
                  key={fn}
                  type="button"
                  onClick={() => setSelectedFunction(fn)}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                    selectedFunction === fn
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border bg-card text-muted-foreground hover:text-foreground",
                  )}
                >
                  {fn}
                </button>
              ))}
            </div>
          )}

          {benchLoading ? (
            <div className="flex items-center justify-center p-10">
              <Loader2 className="size-5 animate-spin text-muted-foreground" />
            </div>
          ) : rows.length === 0 ? (
            <p className="text-sm text-muted-foreground">No benchmark results found.</p>
          ) : (
            <div className="space-y-6">
              <BenchmarkChartGrid
                rows={rows}
                yScale={logRegret ? "log" : "linear"}
                selected={selectedOptimizers}
              />
              <SummaryTable
                summary={summary.filter(
                  (s) => !selectedFunction || s.synthetic_function === selectedFunction,
                )}
                optimizers={allOptimizers}
                selected={selectedOptimizers}
                onSelectedChange={setSelectedOptimizersPersist}
                onDelete={handleDeleteBenchmarkRow}
              />
              <div ref={evalDistAnchorRef}>
                {!evalDistVisible ? (
                  <div className="py-6 text-center text-xs text-muted-foreground">
                    Eval distribution loads on scroll…
                  </div>
                ) : evalDistLoading && !evalDist ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="size-4 animate-spin text-muted-foreground" />
                    <span className="ml-2 text-xs text-muted-foreground">
                      Loading eval distribution + true Pareto…
                    </span>
                  </div>
                ) : evalDist && evalSeed != null ? (
                  <EvalDistributionGrid
                    data={evalDist}
                    seed={evalSeed}
                    onSeedChange={setEvalSeed}
                    selected={selectedOptimizers}
                    rows={rows}
                  />
                ) : null}
              </div>
            </div>
          )}
        </div>
      </TabsContent>

      {/* ── Run ───────────────────────────────────────────────── */}
      <TabsContent value="run" className="flex-1 min-w-0 overflow-hidden m-0">
        <div className="flex flex-1 min-w-0 overflow-hidden h-full">
          <SidePanel className="w-[260px]">
            <SidePanelHeader>
              <span className="text-[10px] font-semibold tracking-[1px] text-muted-foreground">
                CONFIGS ({configs.length})
              </span>
              <Button type="button" variant="ghost" size="icon" onClick={loadConfigs} className="size-6">
                <RefreshCw className="size-3 text-muted-foreground" />
              </Button>
            </SidePanelHeader>
            <div className="flex-1 overflow-auto">
              {configs.map((c) => {
                const isActive = selectedConfig === c.filename;
                const shape = classifyConfig(c.content);
                return (
                  <SideItem
                    key={c.filename}
                    active={isActive}
                    onClick={() => setSelectedConfig(c.filename)}
                  >
                    <SideItemDot active={isActive} />
                    <div className="flex min-w-0 flex-1 flex-col gap-px">
                      <div className="flex items-center gap-1.5">
                        <span
                          className={cn(
                            "truncate text-xs",
                            isActive
                              ? "font-semibold text-foreground"
                              : "text-muted-foreground",
                          )}
                        >
                          {c.filename}
                        </span>
                        {shape.kind === "sweep" && (
                          <Badge
                            variant="secondary"
                            className="h-4 shrink-0 px-1.5 text-[9px] font-semibold"
                          >
                            sweep ×{shape.totalRuns}
                          </Badge>
                        )}
                      </div>
                      <span className="truncate text-[10px] text-muted-foreground">
                        {shape.kind === "sweep"
                          ? `${shape.optimizerCount} optimizer${
                              shape.optimizerCount === 1 ? "" : "s"
                            } · ${shape.seedCount} seed${
                              shape.seedCount === 1 ? "" : "s"
                            }`
                          : `${c.optimizer_type} · ${c.synthetic_function}`}
                      </span>
                    </div>
                  </SideItem>
                );
              })}
              {configs.length === 0 && (
                <p className="p-3.5 text-xs text-muted-foreground/60">
                  No configs found. Configure the directory in Settings.
                </p>
              )}
            </div>
          </SidePanel>

          <div className="flex-1 min-w-0 overflow-auto">
            <div className="mx-auto max-w-[960px] px-8 pt-10 pb-20">
              {!selectedConfig ? (
                <div className="flex h-full items-center justify-center pt-20">
                  <p className="text-sm text-muted-foreground">
                    Select a config to view and launch.
                  </p>
                </div>
              ) : (
                (() => {
                  const cfg = configs.find((c) => c.filename === selectedConfig);
                  if (!cfg) return null;
                  const shape = classifyConfig(cfg.content);
                  return (
                    <>
                      <SectionHeader
                        title={cfg.filename}
                        description={
                          shape.kind === "sweep"
                            ? `Sweep config — runs ${shape.totalRuns} jobs in one launch.`
                            : "Single benchmark run."
                        }
                        action={
                          <div className="flex items-center gap-2">
                            {saveOk && (
                              <span className="text-[11px] text-emerald-600">
                                Saved
                              </span>
                            )}
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={handleSaveConfig}
                              disabled={!editDirty || saving}
                            >
                              {saving ? <Loader2 className="size-4 animate-spin" /> : "Save"}
                            </Button>
                            <Button
                              type="button"
                              onClick={() => handleLaunch(cfg.filename)}
                              disabled={launching}
                            >
                              {launching ? (
                                <Loader2 className="size-4 animate-spin" />
                              ) : (
                                <>
                                  <Play className="size-[13px]" />
                                  Launch
                                </>
                              )}
                            </Button>
                          </div>
                        }
                      />
                      <Separator className="my-6" />

                      {/* Sweep / single breakdown */}
                      <div className="mb-6 grid grid-cols-2 gap-x-8 gap-y-2 sm:grid-cols-4">
                        <BreakdownStat
                          icon={<GitBranch className="size-3.5" />}
                          label="Optimizers"
                          value={shape.optimizerCount}
                        />
                        <BreakdownStat
                          icon={<LayersIcon className="size-3.5" />}
                          label="Functions"
                          value={shape.syntheticFunctionCount}
                        />
                        <BreakdownStat
                          icon={<Hash className="size-3.5" />}
                          label="Seeds"
                          value={shape.seedCount}
                        />
                        <BreakdownStat
                          icon={<Play className="size-3.5" />}
                          label="Total runs"
                          value={shape.totalRuns}
                          accent
                        />
                      </div>

                      {shape.kind === "sweep" && shape.optimizers.length > 0 && (
                        <div className="mb-6">
                          <p className="mb-2 text-xs font-semibold text-muted-foreground">
                            Optimizers in this sweep
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {shape.optimizers.map((o, i) => (
                              <Badge
                                key={i}
                                variant="outline"
                                className="font-mono text-[10px]"
                              >
                                {o.name ?? o.type}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {saveError && (
                        <div className="mb-4 rounded-md border border-destructive/20 bg-destructive/5 p-3">
                          <p className="text-[11px] font-semibold text-destructive">
                            Save failed
                          </p>
                          <p className="mt-1 font-mono text-[11px] text-destructive">
                            {(() => {
                              try {
                                const parsed = JSON.parse(saveError);
                                return String(parsed.detail || saveError);
                              } catch {
                                return saveError;
                              }
                            })()}
                          </p>
                        </div>
                      )}

                      <div>
                        <p className="mb-2 text-xs font-semibold text-muted-foreground">
                          YAML
                        </p>
                        {/* Fixed height — ConfigEditor (CodeMirror-backed)
                            needs a definite height to enable internal
                            scroll; without it the editor collapses. */}
                        <div className="h-[480px] overflow-hidden rounded-md border border-border">
                          <ConfigEditor
                            value={editYaml}
                            onChangeText={(t) => {
                              setEditYaml(t);
                              setEditDirty(true);
                              setSaveOk(false);
                              setSaveError(null);
                            }}
                          />
                        </div>
                      </div>
                    </>
                  );
                })()
              )}

              <Separator className="my-8" />

              <SectionHeader
                title="Recent Runs"
                description="Polled every 5 seconds while this tab is open."
              />
              <Separator className="my-6" />

              <JobList
                jobs={jobs as Job[]}
                onRefresh={loadJobs}
                onKill={async (id) => {
                  await killOptimizerJob(id);
                  loadJobs();
                }}
                renderTitle={(job) => {
                  const cfg = job.config as Record<string, unknown>;
                  if (cfg && Array.isArray(cfg.optimizers)) {
                    const opts = cfg.optimizers as Array<{ type?: string; name?: string }>;
                    return `sweep (${opts.length} optimizers)`;
                  }
                  const opt = cfg?.optimizer;
                  const optType =
                    typeof opt === "object" && opt
                      ? ((opt as any).type ?? "")
                      : "";
                  const sf = cfg?.synthetic_function;
                  const sfType =
                    typeof sf === "string"
                      ? sf
                      : typeof sf === "object" && sf
                        ? ((sf as any).type ?? "")
                        : "";
                  return `${optType} · ${sfType}`;
                }}
              />
            </div>
          </div>
        </div>
      </TabsContent>

      {/* ── Optimizers (registry) ─────────────────────────────── */}
      <TabsContent value="optimizers" className="flex-1 min-w-0 overflow-hidden m-0">
        <div className="flex flex-1 min-w-0 overflow-hidden h-full">
          <SidePanel className="w-[260px]">
            <SidePanelHeader>
              <span className="text-[10px] font-semibold tracking-[1px] text-muted-foreground">
                ALGORITHMS ({optimizers.length})
              </span>
            </SidePanelHeader>
            <div className="flex-1 overflow-auto">
              {optimizers.map((o) => {
                const isActive = selectedOptimizer === o.key;
                return (
                  <SideItem
                    key={o.key}
                    active={isActive}
                    onClick={() => setSelectedOptimizer(o.key)}
                  >
                    <SideItemDot active={isActive} />
                    <span
                      className={cn(
                        "flex-1 truncate text-xs",
                        isActive
                          ? "font-semibold text-foreground"
                          : "text-muted-foreground",
                      )}
                    >
                      {o.key}
                    </span>
                  </SideItem>
                );
              })}
            </div>
          </SidePanel>

          <div className="flex-1 min-w-0 overflow-auto">
            <div className="mx-auto max-w-[1100px] px-8 pt-10 pb-20">
              {!selectedOptimizer ? (
                <div className="flex h-full items-center justify-center pt-20">
                  <p className="text-sm text-muted-foreground">
                    Select an algorithm to view its description, configs, and past
                    benchmark results.
                  </p>
                </div>
              ) : (
                (() => {
                  const info = optimizers.find((o) => o.key === selectedOptimizer);
                  return (
                    <>
                      {info && (
                        <>
                          <SectionHeader
                            title={info.key}
                            description={info.description}
                            action={
                              <Badge variant="outline" className="font-mono text-[10px]">
                                {info.class_name}
                              </Badge>
                            }
                          />
                          <Separator className="my-6" />
                        </>
                      )}

                      {/* Matching configs */}
                      <div className="mb-8 space-y-3">
                        <h3 className="text-sm font-semibold">
                          Benchmark Configurations{" "}
                          <span className="text-muted-foreground">
                            ({matchingConfigs.length})
                          </span>
                        </h3>
                        {matchingConfigs.length === 0 ? (
                          <p className="text-xs text-muted-foreground">
                            No configs use this optimizer yet.
                          </p>
                        ) : (
                          <ul className="divide-y divide-border rounded-md border border-border">
                            {matchingConfigs.map((c) => (
                              <li key={c.filename} className="p-3">
                                <div className="mb-1.5 flex items-center gap-2">
                                  <Badge
                                    variant="secondary"
                                    className="font-mono text-[10px]"
                                  >
                                    {c.filename}
                                  </Badge>
                                  <span className="text-[11px] text-muted-foreground">
                                    {c.synthetic_function} · {c.n_iterations} iters
                                    · seed {c.seed}
                                  </span>
                                </div>
                                <ConfigViewer config={c.content} />
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>

                      {/* Past results */}
                      <div className="space-y-3">
                        <h3 className="text-sm font-semibold">Past Results</h3>
                        {optimizerRows.length === 0 ? (
                          <p className="text-xs text-muted-foreground">
                            No benchmark results recorded for this algorithm.
                          </p>
                        ) : (
                          <div className="space-y-6">
                            <BenchmarkChartGrid
                              rows={optimizerRows}
                              yScale={logRegret ? "log" : "linear"}
                            />
                            <SummaryTable
                              summary={optimizerSummary}
                              optimizers={selectedOptimizer ? [selectedOptimizer] : []}
                              onDelete={handleDeleteBenchmarkRow}
                            />
                          </div>
                        )}
                      </div>
                    </>
                  );
                })()
              )}
            </div>
          </div>
        </div>
      </TabsContent>
    </Tabs>
  );
}

// ── Helpers ───────────────────────────────────────────────────────────

function BreakdownStat({
  icon,
  label,
  value,
  accent,
}: {
  icon: React.ReactNode;
  label: string;
  value: number | string;
  accent?: boolean;
}) {
  return (
    <div>
      <div className="flex items-center gap-1.5 text-xs uppercase tracking-wider text-muted-foreground">
        {icon}
        {label}
      </div>
      <div
        className={cn(
          "mt-1 text-2xl font-semibold tabular-nums",
          accent ? "text-primary" : "text-foreground",
        )}
      >
        {value}
      </div>
    </div>
  );
}
