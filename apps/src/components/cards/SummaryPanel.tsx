import * as React from "react";
import { ChevronDown } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { StatusPill } from "@/components/ui/StatusPill";
import { cn } from "@/lib/utils";
import type { Evaluation, Summary } from "@/types";
import { isEvalFailure } from "@/types";

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function relativeTime(iso: string, now: number): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "—";
  const s = Math.max(0, Math.floor((now - t) / 1000));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function pct(n: number, d: number, digits = 1): string {
  if (!d) return "—";
  return `${((n / d) * 100).toFixed(digits)}%`;
}

function fmt(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e6)) return value.toExponential(2);
  return value.toFixed(digits);
}

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function mean(xs: number[]): number | null {
  if (!xs.length) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

// Re-render every 30s so the relative timestamp stays fresh between polls.
function useTick(intervalMs = 30_000) {
  const [now, setNow] = React.useState(() => Date.now());
  React.useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  return now;
}

// ─────────────────────────────────────────────────────────────────────────────
// Zone primitive
// ─────────────────────────────────────────────────────────────────────────────

interface ZoneProps {
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
  className?: string;
}

function Zone({ label, value, detail, className }: ZoneProps) {
  return (
    <div className={cn("flex flex-1 flex-col gap-2 px-6 py-5", className)}>
      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <div className="text-3xl font-semibold tabular-nums tracking-tight text-foreground">
        {value}
      </div>
      {detail && (
        <div className="flex flex-col gap-1 text-xs text-muted-foreground">
          {detail}
        </div>
      )}
    </div>
  );
}

// Smaller secondary zone — used in the expanded panel where density matters.
function SubZone({
  label,
  value,
  detail,
}: {
  label: string;
  value: React.ReactNode;
  detail?: React.ReactNode;
}) {
  return (
    <div className="flex flex-1 flex-col gap-1 px-6 py-4">
      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
        {label}
      </span>
      <div className="text-xl font-semibold tabular-nums tracking-tight text-foreground">
        {value}
      </div>
      {detail && (
        <div className="text-xs tabular-nums text-muted-foreground">
          {detail}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregates
// ─────────────────────────────────────────────────────────────────────────────

interface Aggregates {
  bestPerf: { value: number; evalId: number } | null;
  bestQuality: { value: number; evalId: number } | null;
  latest: Evaluation | null;
  latestPerf: number | null;
  latestQuality: number | null;
  subMetricBests: { name: string; value: number; evalId: number }[];
  qualityMean: number | null;
  qualityMedian: number | null;
  perfMean: number | null;
  perfMedian: number | null;
  /** Per-eval-id classification, source of truth for GT/CM/INVALID counts.
   *  Unlike summary.{gt,cm}_count (controller-side counters that drift on
   *  resume / mid-run crashes), these are derived from the CSV.
   *  Priority when an eval_id has multiple rows: GT > CM > DUPLICATE > OOM/INVALID. */
  totalEvals: number;
  gtEvals: number;
  cmEvals: number;
  invalidEvals: number;
  skippedEvals: number;
}

/** Live getters from EvalDataContext — scores must reflect the active quality
 *  formula + performance policy, never the raw CSV columns, or this panel
 *  drifts from the Pareto chart / navigator next to it. */
interface ScoreGetters {
  getQuality: (ev: Evaluation) => number | null;
  getPerformanceResult: (ev: Evaluation) => {
    score: number;
    feasible: boolean;
    displayScore: number;
  };
}

function computeAggregates(
  evaluations: Evaluation[],
  { getQuality, getPerformanceResult }: ScoreGetters,
): Aggregates {
  const empty: Aggregates = {
    bestPerf: null,
    bestQuality: null,
    latest: null,
    latestPerf: null,
    latestQuality: null,
    subMetricBests: [],
    qualityMean: null,
    qualityMedian: null,
    perfMean: null,
    perfMedian: null,
    totalEvals: 0,
    gtEvals: 0,
    cmEvals: 0,
    invalidEvals: 0,
    skippedEvals: 0,
  };
  if (!evaluations.length) return empty;

  // Resolve a single tier per eval_id: GT > CM > DUPLICATE > OOM/INVALID.
  // CSV writes a fresh row when an eval is upgraded (CM → GT), so the same
  // eval_id appears multiple times; per-id classification is the only honest
  // way to count.
  const tierRank: Record<string, number> = {
    GT: 4,
    CM: 3,
    DUPLICATE: 2,
    OOM: 1,
    INVALID: 1,
  };
  const idTier = new Map<number, string>();
  for (const ev of evaluations) {
    const cur = idTier.get(ev.eval_id);
    const newRank = tierRank[ev.eval_type] ?? 0;
    const curRank = cur ? tierRank[cur] ?? 0 : 0;
    if (newRank > curRank) idTier.set(ev.eval_id, ev.eval_type);
  }
  let gtEvals = 0;
  let cmEvals = 0;
  let invalidEvals = 0;  // OOM + INVALID (ran-and-failed / unrunnable)
  let skippedEvals = 0;  // DUPLICATE (known-bad, not re-run)
  for (const t of idTier.values()) {
    if (t === "GT") gtEvals += 1;
    else if (t === "CM") cmEvals += 1;
    else if (t === "DUPLICATE") skippedEvals += 1;
    else invalidEvals += 1;  // OOM or INVALID
  }

  let bestPerf: { value: number; evalId: number } | null = null;
  let bestQuality: { value: number; evalId: number } | null = null;
  let latest: Evaluation | null = null;
  const perfVals: number[] = [];
  const qualityVals: number[] = [];
  const subMetricMax: Record<string, { value: number; evalId: number }> = {};

  // For best/distribution: skip INVALID rows (no real measurement) but
  // still consider CM and GT rows. We may double-count an eval_id that
  // appears as both — that's OK because the *latest* row carries the
  // most up-to-date scalar for that eval_id.
  for (const ev of evaluations) {
    if (isEvalFailure(ev.eval_type)) continue;
    if (latest === null || ev.eval_id > latest.eval_id) latest = ev;
    // Policy-infeasible evals carry the INFEASIBLE_SCORE sentinel — they have
    // no achievable performance under the active constraint, so they belong
    // in neither the best nor the distribution stats.
    const perfRes = getPerformanceResult(ev);
    if (perfRes.feasible && Number.isFinite(perfRes.score)) {
      perfVals.push(perfRes.score);
      if (bestPerf === null || perfRes.score > bestPerf.value) {
        bestPerf = { value: perfRes.score, evalId: ev.eval_id };
      }
    }
    const q = getQuality(ev);
    if (q !== null && Number.isFinite(q)) {
      qualityVals.push(q);
      if (bestQuality === null || q > bestQuality.value) {
        bestQuality = { value: q, evalId: ev.eval_id };
      }
    }
    for (const [name, val] of Object.entries(ev.quality_metrics ?? {})) {
      if (name === "combined" || !Number.isFinite(val)) continue;
      const cur = subMetricMax[name];
      if (!cur || val > cur.value) {
        subMetricMax[name] = { value: val, evalId: ev.eval_id };
      }
    }
  }

  const subMetricBests = Object.entries(subMetricMax)
    .map(([name, v]) => ({ name, ...v }))
    .sort((a, b) => a.name.localeCompare(b.name));

  return {
    bestPerf,
    bestQuality,
    latest,
    latestPerf: latest ? getPerformanceResult(latest).displayScore : null,
    latestQuality: latest ? getQuality(latest) : null,
    subMetricBests,
    qualityMean: mean(qualityVals),
    qualityMedian: median(qualityVals),
    perfMean: mean(perfVals),
    perfMedian: median(perfVals),
    totalEvals: idTier.size,
    gtEvals,
    cmEvals,
    invalidEvals,
    skippedEvals,
  };
}

// Strip common prefixes for a tidier sub-metric label.
function prettySubMetricName(raw: string): string {
  return raw.replace(/^deepeval_/, "").replace(/_/g, " ");
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel
// ─────────────────────────────────────────────────────────────────────────────

export interface SummaryPanelProps extends ScoreGetters {
  summary: Summary | null;
  evaluations?: Evaluation[];
  /** Size of the live client-side Pareto frontier (EvalDataContext), which is
   *  the authoritative one in the UI — summary.n_pareto_points is the server's
   *  frozen default-policy frontier and can disagree with the charts. */
  paretoCount: number;
}

export function SummaryPanel({
  summary,
  evaluations = [],
  paretoCount,
  getQuality,
  getPerformanceResult,
}: SummaryPanelProps) {
  const now = useTick();
  const [open, setOpen] = React.useState(false);

  const agg = React.useMemo(
    () => computeAggregates(evaluations, { getQuality, getPerformanceResult }),
    [evaluations, getQuality, getPerformanceResult],
  );

  if (!summary) return null;

  // Prefer CSV-derived counts (truth) over summary.{cm,gt,n_invalid}_count
  // (controller counters that drift on resume / partial runs). Fall back to
  // summary only while evaluations haven't loaded yet.
  const haveEvals = evaluations.length > 0;
  const gt = haveEvals ? agg.gtEvals : summary.gt_count;
  const cm = haveEvals ? agg.cmEvals : summary.cm_count;
  const invalid = haveEvals
    ? agg.invalidEvals
    : summary.n_invalid_configs ?? 0;
  const skipped = haveEvals ? agg.skippedEvals : 0;
  // `attempts` = every unique eval_id that was tried (GT + CM + failures:
  // OOM/INVALID + DUPLICATE). `valid` = only ones that produced data (GT + CM).
  // summary.n_evaluations already includes failure+duplicate rows, so no
  // double-add on the fallback path.
  const attempts = haveEvals ? agg.totalEvals : summary.n_evaluations;
  const valid = gt + cm;
  // Client frontier once evals are in (matches the Pareto chart + export
  // button); server counter only as the pre-load placeholder.
  const pareto = haveEvals ? paretoCount : summary.n_pareto_points;
  const assembled = summary.n_assembly_results ?? 0;

  const validRate = attempts > 0 ? (valid / attempts) * 100 : 100;
  const gtShare = attempts > 0 ? (gt / attempts) * 100 : 0;
  const cmShare = attempts > 0 ? (cm / attempts) * 100 : 0;

  const since = relativeTime(summary.timestamp, now);

  return (
    <Card className="mb-4 gap-0 py-0">
      <Collapsible open={open} onOpenChange={setOpen}>
        {/* ── Top tier: always visible ─────────────────────────────────── */}
        <div className="flex flex-col divide-y sm:flex-row sm:divide-x sm:divide-y-0">
          {/* Status */}
          <div className="flex flex-1 flex-col gap-2 px-6 py-5">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Status
            </span>
            <StatusPill status={summary.status} />
            <div className="flex flex-col gap-1 text-xs text-muted-foreground">
              <span className="tabular-nums">updated {since}</span>
              {summary.pid !== undefined && summary.pid !== null && (
                <span className="tabular-nums">pid {summary.pid}</span>
              )}
            </div>
          </div>

          {/* Evaluations */}
          <div className="flex flex-[1.3] flex-col gap-2 px-6 py-5">
            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              Evaluations
            </span>
            <div className="text-3xl font-semibold tabular-nums tracking-tight text-foreground">
              {attempts.toLocaleString()}
            </div>
            <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-muted">
              {gtShare > 0 && (
                <span
                  className="bg-primary"
                  style={{ width: `${gtShare}%` }}
                />
              )}
              {cmShare > 0 && (
                <span
                  className="bg-secondary-foreground/40"
                  style={{ width: `${cmShare}%` }}
                />
              )}
            </div>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
              <span className="inline-flex items-center gap-1.5 tabular-nums">
                <span className="size-1.5 rounded-full bg-primary" />
                GT <span className="text-foreground">{gt}</span>
              </span>
              <span className="inline-flex items-center gap-1.5 tabular-nums">
                <span className="size-1.5 rounded-full bg-secondary-foreground/40" />
                CM <span className="text-foreground">{cm}</span>
              </span>
            </div>
          </div>

          {/* Pareto */}
          <Zone
            label="Pareto Front"
            value={
              <span>
                {pareto}
                <span className="ml-1.5 text-base font-normal tabular-nums text-muted-foreground">
                  / {valid}
                </span>
              </span>
            }
            detail={
              <>
                <span className="tabular-nums">
                  {pct(pareto, valid)} of valid evals
                </span>
                <span className="tabular-nums">
                  {assembled} assemblies built
                </span>
              </>
            }
          />

          {/* Health */}
          <Zone
            label="Search Health"
            value={
              <span>
                {validRate.toFixed(1)}
                <span className="text-base font-normal text-muted-foreground">
                  %
                </span>
              </span>
            }
            detail={
              <>
                <span className="tabular-nums">
                  {invalid + skipped} dropped / {attempts} tried
                </span>
                <span className="tabular-nums">
                  GT yield {pct(gt, attempts)}
                </span>
              </>
            }
          />
        </div>

        {/* ── Expanded tier ─────────────────────────────────────────────── */}
        <CollapsibleContent className="overflow-hidden">
          <div className="border-t">
            {/* Bests row */}
            <div className="flex flex-col divide-y sm:flex-row sm:divide-x sm:divide-y-0">
              <SubZone
                label="Best Performance"
                value={fmt(agg.bestPerf?.value, 3)}
                detail={
                  agg.bestPerf
                    ? `eval #${agg.bestPerf.evalId}`
                    : "no evals yet"
                }
              />
              <SubZone
                label="Best Quality"
                value={fmt(agg.bestQuality?.value, 3)}
                detail={
                  agg.bestQuality
                    ? `eval #${agg.bestQuality.evalId}`
                    : "no GT evals yet"
                }
              />
              <SubZone
                label="Latest Eval"
                value={
                  agg.latest ? `#${agg.latest.eval_id}` : "—"
                }
                detail={
                  agg.latest
                    ? `${agg.latest.eval_type} · perf ${fmt(
                        agg.latestPerf,
                        2,
                      )} · q ${fmt(agg.latestQuality, 3)}`
                    : undefined
                }
              />
              <SubZone
                label="Eval Mix"
                value={
                  <span className="tabular-nums">
                    {gt}
                    <span className="text-base font-normal text-muted-foreground">
                      {" "}/{" "}
                    </span>
                    {cm}
                  </span>
                }
                detail={`GT / CM · ${invalid} invalid${
                  skipped > 0 ? ` · ${skipped} skipped` : ""
                }`}
              />
            </div>

            {/* Quality sub-metrics */}
            {agg.subMetricBests.length > 0 && (
              <div className="border-t px-6 py-4">
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Best per Quality Sub-metric
                </div>
                <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">
                  {agg.subMetricBests.map((sm) => (
                    <div
                      key={sm.name}
                      className="flex flex-col gap-0.5"
                    >
                      <span className="truncate text-xs capitalize text-muted-foreground">
                        {prettySubMetricName(sm.name)}
                      </span>
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-base font-semibold tabular-nums text-foreground">
                          {fmt(sm.value, 3)}
                        </span>
                        <span className="text-xs tabular-nums text-muted-foreground">
                          #{sm.evalId}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Distributions */}
            {(agg.qualityMean !== null || agg.perfMean !== null) && (
              <div className="border-t px-6 py-4">
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Distributions
                </div>
                <div className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Quality</span>
                    <span className="tabular-nums text-foreground">
                      mean {fmt(agg.qualityMean, 3)}{" "}
                      <span className="text-muted-foreground">·</span> median{" "}
                      {fmt(agg.qualityMedian, 3)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Performance</span>
                    <span className="tabular-nums text-foreground">
                      mean {fmt(agg.perfMean, 2)}{" "}
                      <span className="text-muted-foreground">·</span> median{" "}
                      {fmt(agg.perfMedian, 2)}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </CollapsibleContent>

        {/* ── Toggle ─────────────────────────────────────────────────────── */}
        <div className="flex justify-end border-t px-3 py-1.5">
          <CollapsibleTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 gap-1.5 text-xs text-muted-foreground"
            >
              {open ? "Hide details" : "More metrics"}
              <ChevronDown
                className={cn(
                  "size-3.5 transition-transform",
                  open && "rotate-180",
                )}
              />
            </Button>
          </CollapsibleTrigger>
        </div>
      </Collapsible>
    </Card>
  );
}
