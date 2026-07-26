/**
 * EvalDistributionGrid — per-(optimizer, seed) panels showing every evaluated
 * (quality, speed) point colored by iteration order, with the true Pareto
 * front (red curve) and the seed's found Pareto front (black markers + line)
 * overlaid.
 *
 * Web port of `rag_stack/optimizer/benchmark/plot_eval_distribution.py`.
 *
 * UI:
 *   - shadcn `<Select>` for the seed picker
 *   - bare grid (no Card chrome) of small ECharts panels — same typography-
 *     driven pattern used elsewhere in the redesign
 *   - panel order matches `optimizers` (alphabetical), filtered by `selected`
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, Loader2, Pause, Play, RotateCcw } from "lucide-react";
import EChartsView from "@/components/charts/EChartsView";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import type {
  EvalDistributionPayload,
  OptimizerBenchmarkRow,
} from "@/types/optimizer";

const TEXT_TERTIARY = "#9CA3AF";
const BORDER_DEFAULT = "#E2E6EF";
const BORDER_SUBTLE = "#F0F2F7";
const TRUE_PARETO_COLOR = "#DC2626"; // crimson
const FOUND_PARETO_COLOR = "#111827"; // near-black
// CM = many cheap-only points; muted gray so they don't visually shout.
// GT = few real ground-truth points; amber so the diamonds pop.
const CM_EVAL_COLOR = "#94A3B8"; // slate-400
const GT_EVAL_COLOR = "#F59E0B"; // amber-500

/** Adaptive axis-label formatter — keeps ~3 sig figs, falls back to
 *  exponential for very small / very large values. Matches the helper
 *  used in BenchmarkChartGrid so axis numbers look the same everywhere. */
function fmtAxisLabel(v: number | string): string {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return String(v);
  if (n === 0) return "0";
  const abs = Math.abs(n);
  if (abs < 1e-3 || abs >= 1e4) return n.toExponential(1);
  const s = n.toPrecision(3);
  return s.includes(".") ? s.replace(/\.?0+$/, "") || "0" : s;
}

interface Props {
  data: EvalDistributionPayload;
  /** Currently picked seed; controlled by the parent. */
  seed: number;
  onSeedChange: (s: number) => void;
  /** Whitelist of optimizer names to render. Empty = render all. */
  selected?: Set<string>;
  /** Trajectory rows (already filtered to the current function on the
   *  parent). Used to surface per-(optimizer, seed) hv_diff + IGD in each
   *  panel header, snapped to the budget the progress slider is showing. */
  rows?: OptimizerBenchmarkRow[];
}

/** Adaptive number formatter for HV-diff / IGD display. Mirrors the
 *  SummaryTable's `fmt` so the panel header reads the same as the table
 *  cell at the rightmost slider position. */
function fmtMetric(x: number | undefined | null): string {
  if (x == null || !Number.isFinite(x)) return "—";
  if (x === 0) return "0";
  const abs = Math.abs(x);
  if (abs >= 1e5) return x.toExponential(3);
  if (abs < 1e-2) return x.toExponential(3);
  if (abs >= 100) return x.toFixed(2);
  if (abs >= 10) return x.toFixed(3);
  if (abs >= 1) return x.toFixed(4);
  const decimals = Math.min(6, -Math.floor(Math.log10(abs)) + 4);
  const s = x.toFixed(decimals);
  return s.includes(".") ? s.replace(/\.?0+$/, "") || "0" : s;
}

/** Non-dominated mask in 2D max-max under (quality, speed). */
function paretoFront2D(points: [number, number][]): [number, number][] {
  if (points.length === 0) return [];
  const idxByQDesc = points
    .map((_, i) => i)
    .sort((a, b) => points[b][0] - points[a][0]);
  const out: [number, number][] = [];
  let bestS = -Infinity;
  for (const i of idxByQDesc) {
    const [, s] = points[i];
    if (s > bestS) {
      out.push(points[i]);
      bestS = s;
    }
  }
  // Sort by speed for the line plot.
  out.sort((a, b) => a[1] - b[1]);
  return out;
}

/** A "GT eval" actually queried the ground-truth function. The cheap
 *  phases (cheap_batch / cheap_bo / main_cheap) only paid the cost-model;
 *  only `main_both` issues a real GT call. We split them into two series
 *  with different symbols so the user can tell at a glance which points
 *  cost the optimizer a GT budget unit. */
function isGtPhase(phase: string): boolean {
  return phase === "main_both";
}

function buildOption(
  evals: {
    iteration: number;
    obj_values: Record<string, number>;
    phase: string;
  }[],
  truePareto: Array<Record<string, number>>,
  optName: string,
  xName: string,
  yName: string,
  axisBounds: { xLo: number; xHi: number; yLo: number; yHi: number },
): any {
  if (evals.length === 0) {
    return {
      title: {
        text: `${optName}\n(no data for this seed)`,
        left: "center",
        top: "center",
        textStyle: { color: TEXT_TERTIARY, fontSize: 12, fontWeight: "normal" },
      },
    };
  }

  // Split by phase so GT evals get their own series with a distinct
  // symbol. Both still carry iteration in dim 2 for the shared visualMap.
  const cmPoints: [number, number, number][] = [];
  const gtPoints: [number, number, number][] = [];
  for (const e of evals) {
    const tup: [number, number, number] = [
      e.obj_values[xName],
      e.obj_values[yName],
      e.iteration,
    ];
    if (isGtPhase(e.phase)) gtPoints.push(tup);
    else cmPoints.push(tup);
  }

  // Found Pareto must come from GT-only points. The cheap phases only paid
  // the cost model so their value isn't a real ground-truth measurement;
  // feeding them into the front would produce a phantom Pareto built from
  // never-evaluated (x, y) pairs. Only `main_both` issued a GT call.
  const gtXY: [number, number][] = gtPoints.map(([x, y]) => [x, y]);
  const found = paretoFront2D(gtXY);

  const series: any[] = [
    {
      name: "CM evals",
      type: "scatter",
      data: cmPoints,
      symbol: "circle",
      symbolSize: 5,
      itemStyle: { color: CM_EVAL_COLOR, opacity: 0.55 },
      large: true,
      largeThreshold: 500,
      progressive: 1500,
      progressiveThreshold: 2000,
      z: 2,
    },
    {
      name: "GT evals",
      type: "scatter",
      data: gtPoints,
      symbol: "diamond",
      symbolSize: 9,
      itemStyle: {
        color: GT_EVAL_COLOR,
        opacity: 1,
        borderColor: "#ffffff",
        borderWidth: 1.2,
      },
      z: 3,
    },
    {
      name: "True Pareto",
      type: "line",
      data: truePareto.map((p) => [p[xName], p[yName]]),
      showSymbol: false,
      lineStyle: { color: TRUE_PARETO_COLOR, width: 1.5, opacity: 0.9 },
      itemStyle: { color: TRUE_PARETO_COLOR },
      z: 4,
      tooltip: { show: false },
    },
    {
      name: "Found Pareto",
      type: "line",
      data: found,
      symbol: "circle",
      symbolSize: 6,
      lineStyle: { color: FOUND_PARETO_COLOR, width: 1, opacity: 0.85 },
      itemStyle: {
        color: FOUND_PARETO_COLOR,
        borderColor: "#fff",
        borderWidth: 1,
      },
      z: 5,
    },
  ];

  return {
    title: { text: "", show: false },
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(17,24,39,0.92)",
      borderColor: "transparent",
      textStyle: { color: "#F9FAFB", fontSize: 11 },
      formatter: (params: any) => {
        if (
          params.seriesName === "CM evals" ||
          params.seriesName === "GT evals"
        ) {
          const [x, y, it] = params.value;
          const tag = params.seriesName === "GT evals" ? "GT" : "CM";
          return `<b>${tag} · iter ${it}</b><br/>${xName} = ${x.toFixed(4)}<br/>${yName} = ${y.toFixed(4)}`;
        }
        const [x, y] = params.value;
        return `<b>${params.seriesName}</b><br/>${xName} = ${x.toFixed(4)}<br/>${yName} = ${y.toFixed(4)}`;
      },
    },
    legend: {
      type: "scroll",
      // Bottom of the chart, below the xAxis name. With the grid bottom
      // bumped to 64 there's room for: tick labels (~12) + nameGap (24) +
      // nameText (~12) + legend (~14) = ≤62px. Pinning the legend at
      // bottom: 4 keeps it off the xAxis label.
      bottom: 4,
      textStyle: { fontSize: 10, color: TEXT_TERTIARY },
      itemWidth: 10,
      itemHeight: 8,
      data: ["CM evals", "GT evals", "True Pareto", "Found Pareto"],
    },
    grid: { left: 64, right: 16, top: 12, bottom: 64 },
    xAxis: {
      type: "value",
      name: xName,
      nameLocation: "middle",
      nameGap: 24,
      nameTextStyle: { fontSize: 10, color: TEXT_TERTIARY },
      min: axisBounds.xLo,
      max: axisBounds.xHi,
      axisLine: { lineStyle: { color: BORDER_DEFAULT } },
      axisLabel: {
        fontSize: 10,
        color: TEXT_TERTIARY,
        formatter: fmtAxisLabel,
      },
      splitLine: { lineStyle: { color: BORDER_SUBTLE } },
    },
    yAxis: {
      type: "value",
      name: yName,
      nameLocation: "middle",
      nameGap: 44,
      nameTextStyle: { fontSize: 10, color: TEXT_TERTIARY },
      min: axisBounds.yLo,
      max: axisBounds.yHi,
      axisLine: { lineStyle: { color: BORDER_DEFAULT } },
      axisLabel: {
        fontSize: 10,
        color: TEXT_TERTIARY,
        formatter: fmtAxisLabel,
      },
      splitLine: { lineStyle: { color: BORDER_SUBTLE } },
    },
    series,
  };
}

// How long a single Play sweep takes, in ms. ~6 s feels like a film
// reel — short enough to feel snappy, long enough to read the path.
const PLAY_DURATION_MS = 6000;
// Update interval while playing. 60 ms ≈ 16 fps; ECharts re-renders are
// the bottleneck so finer granularity is wasted.
const PLAY_TICK_MS = 60;

export function EvalDistributionGrid({
  data,
  seed,
  onSeedChange,
  selected,
  rows,
}: Props) {
  // Per-(optimizer, seed) trajectory lookup, sorted by cumulative_gt_calls.
  // `panels` later picks the latest row whose budget is ≤ the progress
  // slider's `cap` so the displayed hv_diff / IGD line up exactly with
  // what the eval-distribution panels are visually showing.
  const trajByOptSeed = useMemo(() => {
    const map = new Map<string, OptimizerBenchmarkRow[]>();
    if (!rows) return map;
    for (const r of rows) {
      const key = `${r.optimizer}|${r.seed}`;
      let arr = map.get(key);
      if (!arr) {
        arr = [];
        map.set(key, arr);
      }
      arr.push(r);
    }
    for (const arr of map.values()) {
      arr.sort((a, b) => a.cumulative_gt_calls - b.cumulative_gt_calls);
    }
    return map;
  }, [rows]);
  // Filter optimizers via the same selection set as the SummaryTable.
  const optimizers = useMemo(() => {
    if (!selected || selected.size === 0) return data.optimizers;
    return data.optimizers.filter((o) => selected.has(o.name));
  }, [data.optimizers, selected]);

  // Seeds offered in the picker = union across the *currently checked*
  // optimizers only. Unchecking a row in the SummaryTable drops any seeds
  // that only that row had, so the picker can't strand the user on a seed
  // with no panels to render.
  const availableSeeds = useMemo(() => {
    const set = new Set<number>();
    for (const o of optimizers) {
      for (const e of o.evals) set.add(e.seed);
    }
    return Array.from(set).sort((a, b) => a - b);
  }, [optimizers]);

  // Auto-switch when the active seed disappears from the visible set
  // (e.g., user unchecked the last optimizer that ran it).
  useEffect(() => {
    if (availableSeeds.length === 0) return;
    if (!availableSeeds.includes(seed)) {
      onSeedChange(availableSeeds[0]);
    }
  }, [availableSeeds, seed, onSeedChange]);

  // ── Progress slider ─────────────────────────────────────────────────
  // Use cumulative GT calls as the canonical budget axis: it's monotonic,
  // shared across optimizers, and matches what the trajectory chart
  // already plots. The slider value is the upper-bound cap; each panel
  // shows only evals whose cumulative_gt_calls ≤ cap.
  const maxGtCalls = useMemo(() => {
    let max = 0;
    for (const o of optimizers) {
      for (const e of o.evals) {
        if (e.seed !== seed) continue;
        if (e.cumulative_gt_calls > max) max = e.cumulative_gt_calls;
      }
    }
    return max;
  }, [optimizers, seed]);

  const [progress, setProgress] = useState(maxGtCalls);
  const [playing, setPlaying] = useState(false);
  const lastTickRef = useRef<number | null>(null);

  // Snap the slider to the new ceiling whenever the underlying data
  // changes (function switch, seed switch, optimizer add/remove). Default
  // to the full run so the chart matches the static view.
  useEffect(() => {
    setProgress(maxGtCalls);
    setPlaying(false);
  }, [maxGtCalls]);

  // Auto-advance loop driving Play.
  useEffect(() => {
    if (!playing || maxGtCalls <= 0) return;
    lastTickRef.current = null;
    const id = setInterval(() => {
      const now = performance.now();
      const last = lastTickRef.current ?? now;
      lastTickRef.current = now;
      const dt = now - last;
      const stepGt = (maxGtCalls * dt) / PLAY_DURATION_MS;
      setProgress((p) => {
        const next = p + stepGt;
        if (next >= maxGtCalls) {
          setPlaying(false);
          return maxGtCalls;
        }
        return next;
      });
    }, PLAY_TICK_MS);
    return () => clearInterval(id);
  }, [playing, maxGtCalls]);

  const onTogglePlay = () => {
    if (progress >= maxGtCalls) {
      setProgress(0);
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  };

  const onResetProgress = () => {
    setPlaying(false);
    setProgress(maxGtCalls);
  };

  // x/y obj names: first two from data.obj_names. Falls back to the legacy
  // ("quality","speed") pair so older API responses keep rendering.
  const [xName, yName] = useMemo<[string, string]>(() => {
    const names = data.obj_names ?? [];
    if (names.length >= 2) return [names[0], names[1]];
    return ["quality", "speed"];
  }, [data.obj_names]);

  // Compute shared axis bounds across all optimizers + the true Pareto so
  // every panel uses the same scale.
  const axisBounds = useMemo(() => {
    let xLo = Infinity, xHi = -Infinity, yLo = Infinity, yHi = -Infinity;
    for (const o of optimizers) {
      for (const e of o.evals) {
        if (e.seed !== seed) continue;
        const x = e.obj_values?.[xName];
        const y = e.obj_values?.[yName];
        if (Number.isFinite(x)) {
          if (x < xLo) xLo = x;
          if (x > xHi) xHi = x;
        }
        if (Number.isFinite(y)) {
          if (y < yLo) yLo = y;
          if (y > yHi) yHi = y;
        }
      }
    }
    for (const p of data.true_pareto) {
      const x = p[xName];
      const y = p[yName];
      if (Number.isFinite(x)) {
        if (x < xLo) xLo = x;
        if (x > xHi) xHi = x;
      }
      if (Number.isFinite(y)) {
        if (y < yLo) yLo = y;
        if (y > yHi) yHi = y;
      }
    }
    if (!Number.isFinite(xLo) || !Number.isFinite(yLo)) {
      return { xLo: 0, xHi: 1, yLo: 0, yHi: 1 };
    }
    const xPad = (xHi - xLo) * 0.05 || 0.05;
    const yPad = (yHi - yLo) * 0.05 || 0.05;
    return {
      xLo: xLo - xPad,
      xHi: xHi + xPad,
      yLo: yLo - yPad,
      yHi: yHi + yPad,
    };
  }, [optimizers, data.true_pareto, seed, xName, yName]);

  const cap = Math.round(progress);
  const panels = useMemo(
    () =>
      optimizers.map((o) => {
        const seedEvals = o.evals.filter(
          (e) => e.seed === seed && e.cumulative_gt_calls <= cap,
        );
        let gtCount = 0;
        for (const e of seedEvals) {
          if (e.phase === "main_both") gtCount++;
        }
        // Latest trajectory row at or below `cap` for this (optimizer, seed).
        // Falls back to undefined when the rows prop wasn't supplied or
        // there's no data for this combo (e.g. seed not yet run).
        let snapHv: number | undefined;
        let snapIgd: number | undefined;
        const traj = trajByOptSeed.get(`${o.name}|${seed}`);
        if (traj && traj.length > 0) {
          let pick: OptimizerBenchmarkRow | undefined;
          for (const r of traj) {
            if (r.cumulative_gt_calls <= cap) pick = r;
            else break;
          }
          if (pick) {
            snapHv = pick.hv_diff;
            snapIgd = pick.igd;
          }
        }
        return {
          name: o.name,
          count: seedEvals.length,
          gtCount,
          cmCount: seedEvals.length - gtCount,
          hvDiff: snapHv,
          igd: snapIgd,
          option: buildOption(
            seedEvals,
            data.true_pareto,
            o.name,
            xName,
            yName,
            axisBounds,
          ),
        };
      }),
    [optimizers, seed, cap, data.true_pareto, axisBounds, xName, yName, trajByOptSeed],
  );

  if (data.optimizers.length === 0 || data.seeds.length === 0) {
    return (
      <div className="py-8 text-center text-sm text-muted-foreground">
        No eval-distribution CSVs found for {data.function}.
      </div>
    );
  }
  if (availableSeeds.length === 0) {
    return (
      <div className="py-8 text-center text-sm text-muted-foreground">
        No optimizers selected — check rows in the table to view their evals.
      </div>
    );
  }

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Eval Distribution
          </h3>
          <span className="text-[11px] text-muted-foreground/80">
            ·&nbsp;{optimizers.length} optimizer{optimizers.length === 1 ? "" : "s"} · seed {seed}
          </span>
          {data.true_pareto_pending && !data.true_pareto_error && (
            <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
              <Loader2 className="size-3 animate-spin" />
              loading true Pareto…
            </span>
          )}
          {data.true_pareto_error && (
            <TooltipProvider delayDuration={150}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="inline-flex cursor-help items-center gap-1 text-[11px] text-amber-600 dark:text-amber-400">
                    <AlertCircle className="size-3" />
                    true Pareto unavailable
                  </span>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-md">
                  <span className="font-mono text-[11px]">
                    {data.true_pareto_error}
                  </span>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Label
            htmlFor="eval-dist-seed"
            className="text-[11px] uppercase tracking-wider text-muted-foreground"
          >
            Seed
          </Label>
          <Select
            value={String(seed)}
            onValueChange={(v) => onSeedChange(Number(v))}
          >
            <SelectTrigger id="eval-dist-seed" className="h-8 w-[110px] text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {availableSeeds.map((s) => (
                <SelectItem key={s} value={String(s)} className="text-xs">
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Progress slider — scrub through GT calls to watch the run unfold. */}
      {maxGtCalls > 0 && (
        <div className="flex items-center gap-3 rounded-md border border-border bg-card px-3 py-2">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onTogglePlay}
            aria-label={playing ? "Pause" : "Play"}
            className="size-7 shrink-0"
            disabled={maxGtCalls === 0}
          >
            {playing ? (
              <Pause className="size-3.5" />
            ) : (
              <Play className="size-3.5" />
            )}
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={onResetProgress}
            aria-label="Reset"
            className="size-7 shrink-0"
            disabled={progress >= maxGtCalls && !playing}
          >
            <RotateCcw className="size-3.5" />
          </Button>
          <Slider
            value={[Math.min(progress, maxGtCalls)]}
            min={0}
            max={maxGtCalls}
            step={1}
            onValueChange={(v) => {
              setPlaying(false);
              setProgress(v[0] ?? 0);
            }}
            aria-label="Progress"
            className="flex-1"
          />
          <span className="shrink-0 font-mono text-[11px] tabular-nums text-muted-foreground">
            GT&nbsp;{cap.toLocaleString()}&nbsp;/&nbsp;{maxGtCalls.toLocaleString()}
          </span>
        </div>
      )}

      {panels.length === 0 ? (
        <p className="py-6 text-center text-xs text-muted-foreground">
          Every optimizer is hidden — toggle one back on in the table above to see
          its eval distribution.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {panels.map(({ name, option, count, cmCount, gtCount, hvDiff, igd }) => (
            <div
              key={name}
              className="rounded-md border border-border bg-card"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 border-b border-border px-3 py-2">
                <span className="font-mono text-xs font-semibold text-foreground">
                  {name}
                </span>
                <span className="flex flex-wrap items-baseline gap-x-2 text-[10px] uppercase tracking-wider text-muted-foreground">
                  {(hvDiff !== undefined || igd !== undefined) && (
                    <span className="tabular-nums">
                      <span className="text-muted-foreground/70">HV diff</span>{" "}
                      <span className="font-mono text-foreground">{fmtMetric(hvDiff)}</span>
                      <span className="mx-1 text-muted-foreground/40">·</span>
                      <span className="text-muted-foreground/70">IGD</span>{" "}
                      <span className="font-mono text-foreground">{fmtMetric(igd)}</span>
                    </span>
                  )}
                  <span>
                    {count}&nbsp;evals
                    <span className="ml-1 text-muted-foreground/70">
                      ({cmCount}&nbsp;CM · {gtCount}&nbsp;GT)
                    </span>
                    &nbsp;· seed {seed}
                  </span>
                </span>
              </div>
              <EChartsView
                option={option}
                style={{ height: 300, width: "100%" }}
                notMerge
              />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
