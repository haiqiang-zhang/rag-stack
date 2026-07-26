import { useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import { SidePanel, SidePanelHeader } from "@/components/ui/SidePanel";
import { SideItem } from "@/components/ui/SideItem";
import { Chip, ChipText } from "@/components/ui/Chip";
import { ParetoChart } from "@/components/charts/ParetoChart";
import {
  NUMERIC_RAMP,
  PROGRESS_RAMP,
  categoryBuckets,
  type MapColorBy,
} from "@/components/charts/knobColoring";
import { useEvalData } from "@/context/EvalDataContext";
import { useSelection } from "@/context/SelectionContext";
import { buildKnobDimensions, type KnobDim } from "@/utils/searchKnobs";
import { isEvalFailure, isEvalSuccess, type Evaluation } from "@/types";
import { cn } from "@/lib/utils";

type Filter = "all" | "gt" | "failed";
type PerfResult = { score: number; feasible: boolean };

/**
 * Left rail: the objective-space map (compact Pareto) on top and the full eval
 * index below. Selecting a point on the chart OR a row in the list drives the
 * selected eval for every tab.
 *
 * The map is multi-functional: `colorKey` ("none" | "progress" | a knob key,
 * driven by the Trade-off tab's impact table) recolors its points, with a
 * compact legend right under the chart. One chart, one place, extra function —
 * instead of a second full-size copy in the tab content.
 */
export function EvalNavigator({ colorKey = "none" }: { colorKey?: string }) {
  const {
    enrichedEvals,
    clientPareto,
    getQuality,
    getPerformance,
    getPerformanceResult,
    getDisplayPerformance,
    isFeasible,
    perfAxisLabel,
    sweepsReady,
    loading,
    error,
  } = useEvalData();
  const { selectedEvalId, setSelectedEvalId } = useSelection();
  const [filter, setFilter] = useState<Filter>("all");

  const list = useMemo(() => {
    const rows = enrichedEvals.filter((ev) =>
      filter === "all"
        ? true
        : filter === "gt"
          ? isEvalSuccess(ev.eval_type)
          : isEvalFailure(ev.eval_type),
    );
    return rows.sort((a, b) => b.eval_id - a.eval_id);
  }, [enrichedEvals, filter]);

  const hasEvaluations = enrichedEvals.length > 0;
  const chartLoading = loading || (hasEvaluations && !sweepsReady);
  const chartReady = !error && !chartLoading && hasEvaluations;

  // Resolve the color key into an actual dimension (cheap; memoized).
  const colorBy: MapColorBy = useMemo(() => {
    if (colorKey === "none" || colorKey === "") return null;
    if (colorKey === "progress") return "progress";
    const dims = buildKnobDimensions(enrichedEvals);
    return dims.find((d) => d.key === colorKey) ?? null;
  }, [colorKey, enrichedEvals]);

  return (
    <SidePanel className="w-[320px]">
      <SidePanelHeader>
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Evaluations · {enrichedEvals.length}
        </span>
      </SidePanelHeader>

      {/* Objective-space map (multi-functional: recolorable by knob) */}
      <div className="border-b border-border p-3">
        <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.6px] text-muted-foreground">
          Quality vs performance
          {colorBy != null && colorBy !== "progress" && (
            <span className="ml-1 normal-case tracking-normal text-foreground/70">
              · by {colorBy.label}
            </span>
          )}
          {colorBy === "progress" && (
            <span className="ml-1 normal-case tracking-normal text-foreground/70">
              · by search order
            </span>
          )}
        </div>
        <div className="h-[240px]">
          {chartReady ? (
            <ParetoChart
              evaluations={enrichedEvals}
              computedPareto={clientPareto}
              onPointClick={setSelectedEvalId}
              getQuality={getQuality}
              getPerformance={getPerformance}
              getDisplayPerformance={getDisplayPerformance}
              isFeasible={isFeasible}
              selectedEvalId={selectedEvalId}
              perfAxisLabel={perfAxisLabel}
              colorBy={colorBy}
              height={240}
              compact
            />
          ) : error ? (
            <div role="alert" className="flex h-full flex-col items-center justify-center gap-2 px-2 text-center text-[11px] text-destructive">
              <span>Unable to load evaluations.</span>
              <span>{error.message}</span>
            </div>
          ) : chartLoading ? (
            <div className="flex h-full items-center justify-center gap-2 text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              <span className="text-[11px] uppercase tracking-wide">
                Loading…
              </span>
            </div>
          ) : (
            <div className="flex h-full flex-col items-center justify-center gap-2 px-2 text-center text-[11px] text-muted-foreground">
              <span>No evaluations yet.</span>
              <span>Start an evaluation to see quality and performance.</span>
            </div>
          )}
        </div>
        {chartReady && colorBy != null && <MapColorLegend colorBy={colorBy} />}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-1.5 border-b border-border px-3 py-2">
        {(
          [
            ["all", "All"],
            ["gt", "Measured"],
            ["failed", "Failed"],
          ] as [Filter, string][]
        ).map(([key, label]) => (
          <Chip key={key} active={filter === key} onClick={() => setFilter(key)}>
            <ChipText active={filter === key}>{label}</ChipText>
          </Chip>
        ))}
      </div>

      {/* Eval index */}
      <div className="flex-1 overflow-auto py-1">
        {error && !hasEvaluations ? null : list.length === 0 ? (
          <div className="px-3 py-6 text-center text-[11px] text-muted-foreground">
            {loading ? "Loading evaluations…" : hasEvaluations ? "No evals in this filter." : "No evaluations yet."}
          </div>
        ) : (
          list.map((ev) => (
            <EvalRow
              key={ev.eval_id}
              ev={ev}
              active={ev.eval_id === selectedEvalId}
              onClick={() => setSelectedEvalId(ev.eval_id)}
              getPerformanceResult={getPerformanceResult}
              getQuality={getQuality}
            />
          ))
        )}
      </div>
    </SidePanel>
  );
}

/** Compact legend for the recolored map — the in-chart legend/visualMap is
 *  hidden in compact mode, so identity must be readable right here. */
function MapColorLegend({ colorBy }: { colorBy: Exclude<MapColorBy, null> }) {
  if (colorBy === "progress") {
    return (
      <GradientStrip
        ramp={PROGRESS_RAMP}
        left="early"
        right="late"
      />
    );
  }
  if (colorBy.numeric) {
    const nums = colorBy.distinct.map(Number).filter((v) => Number.isFinite(v));
    return (
      <GradientStrip
        ramp={NUMERIC_RAMP}
        left={nums.length ? String(Math.min(...nums)) : "low"}
        right={nums.length ? String(Math.max(...nums)) : "high"}
      />
    );
  }
  const buckets = categoryBuckets(colorBy as KnobDim);
  return (
    <div className="mt-2 flex flex-wrap gap-x-2.5 gap-y-1">
      {buckets.map((b) => (
        <span
          key={b.value}
          title={b.value}
          className="inline-flex min-w-0 items-center gap-1 text-[10px] text-muted-foreground"
        >
          <span style={{ color: b.color }} className="text-[9px] leading-none">
            {b.glyph}
          </span>
          <span className="max-w-[8rem] truncate font-mono">{b.label}</span>
        </span>
      ))}
    </div>
  );
}

function GradientStrip({
  ramp,
  left,
  right,
}: {
  ramp: string[];
  left: string;
  right: string;
}) {
  return (
    <div className="mt-2 flex items-center gap-2">
      <span className="shrink-0 font-mono text-[9px] text-muted-foreground">{left}</span>
      <span
        className="h-1.5 min-w-0 flex-1 rounded-full"
        style={{ background: `linear-gradient(to right, ${ramp.join(", ")})` }}
      />
      <span className="shrink-0 font-mono text-[9px] text-muted-foreground">{right}</span>
    </div>
  );
}

function EvalRow({
  ev,
  active,
  onClick,
  getPerformanceResult,
  getQuality,
}: {
  ev: Evaluation;
  active: boolean;
  onClick: () => void;
  getPerformanceResult: (ev: Evaluation) => PerfResult;
  getQuality: (ev: Evaluation) => number | null;
}) {
  const failed = isEvalFailure(ev.eval_type);
  const perf = failed ? null : getPerformanceResult(ev);
  const qual = getQuality(ev);
  return (
    <SideItem active={active} onClick={onClick} className="h-auto py-1.5">
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-xs font-semibold text-foreground">
            #{ev.eval_id}
          </span>
          {ev.is_pareto && (
            <span className="text-[9px] font-semibold uppercase tracking-wide text-primary">
              pareto
            </span>
          )}
          <span
            title={ev.eval_error || undefined}
            className={cn(
              "ml-auto rounded px-1 text-[9px] font-semibold uppercase",
              failed
                ? "bg-destructive/10 text-destructive"
                : ev.eval_type === "GT"
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                  : "bg-muted text-muted-foreground",
            )}
          >
            {ev.eval_type}
          </span>
        </div>
        <div className="flex items-center gap-3 font-mono text-[10px] text-muted-foreground">
          {failed ? (
            <span className="truncate">no result</span>
          ) : (
            <>
              <span>
                P {perf?.feasible ? perf.score.toFixed(2) : "—"}
              </span>
              <span>Q {qual != null ? qual.toFixed(2) : "—"}</span>
            </>
          )}
        </div>
      </div>
    </SideItem>
  );
}
