/**
 * Quality tab — per-eval quality deep-dive (the mirror of PerformanceTab).
 *
 * Owns everything the Performance tab used to mix in: how the combined score
 * is composed under the ACTIVE metric/formula, where each sub-metric sits
 * against the rest of the project, which pipeline stage the sub-metrics
 * attribute to, and reference GT text metrics when present.
 */
import { useMemo } from "react";
import { MousePointerClick } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { useEvalData } from "@/context/EvalDataContext";
import { useSelection } from "@/context/SelectionContext";
import { usePostAnalysis } from "@/context/PostAnalysisContext";
import { stageColor } from "@/lib/theme";
import type { Evaluation } from "@/types";

const ACCENT = "#0D9488";

export function QualityTab() {
  const { selectedEval, enrichedEvals, getQuality } = useEvalData();
  const { selectedEvalId } = useSelection();
  const { qualityKey, formula, weights } = usePostAnalysis();

  // Project context per sub-metric: mean + best across GT evals (dedup by id,
  // last row wins — CM→GT upgrades append rows).
  const subMetricStats = useMemo(() => {
    const byId = new Map<number, Evaluation>();
    for (const ev of enrichedEvals) byId.set(ev.eval_id, ev);
    const gts = [...byId.values()].filter(
      (ev) => ev.eval_type === "GT" && Object.keys(ev.quality_metrics).length > 0,
    );
    const stats = new Map<string, { mean: number; best: number; n: number }>();
    for (const ev of gts) {
      for (const [k, v] of Object.entries(ev.quality_metrics)) {
        if (!Number.isFinite(v)) continue;
        const s = stats.get(k) ?? { mean: 0, best: -Infinity, n: 0 };
        s.mean += v;
        s.best = Math.max(s.best, v);
        s.n += 1;
        stats.set(k, s);
      }
    }
    for (const s of stats.values()) s.mean = s.n > 0 ? s.mean / s.n : 0;
    return stats;
  }, [enrichedEvals]);

  // Rank of the selected eval among GT evals under the ACTIVE quality getter.
  const rank = useMemo(() => {
    if (!selectedEval) return null;
    const mine = getQuality(selectedEval);
    if (mine == null) return null;
    const byId = new Map<number, Evaluation>();
    for (const ev of enrichedEvals) byId.set(ev.eval_id, ev);
    const scores = [...byId.values()]
      .map((ev) => (ev.eval_type === "GT" ? getQuality(ev) : null))
      .filter((q): q is number => q != null && Number.isFinite(q));
    if (scores.length === 0) return null;
    const better = scores.filter((q) => q > mine).length;
    return { pos: better + 1, of: scores.length };
  }, [selectedEval, enrichedEvals, getQuality]);

  if (selectedEvalId == null || !selectedEval) {
    return (
      <div className="flex flex-1 min-h-0 items-center justify-center p-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <MousePointerClick className="size-7 text-muted-foreground/50" />
          <span className="max-w-xs text-[13px] text-muted-foreground">
            Pick an evaluation in the left navigator — how its quality score
            composes appears here.
          </span>
        </div>
      </div>
    );
  }

  const metrics = selectedEval.quality_metrics;
  const hasQuality = Object.keys(metrics).length > 0;
  const quality = getQuality(selectedEval);

  if (!hasQuality) {
    return (
      <div className="flex-1 min-w-0 overflow-auto p-6 pb-12">
        <EvalHeader ev={selectedEval} quality={quality} rank={rank} />
        <Card className="rounded-lg py-0">
          <CardContent className="p-6 text-center">
            <span className="text-xs text-muted-foreground">
              This eval carries no quality measurement
              {selectedEval.eval_type === "CM"
                ? " — it was scored by the cost model only. Pick a GT eval to see quality composition."
                : "."}
            </span>
          </CardContent>
        </Card>
      </div>
    );
  }

  const formulaText =
    qualityKey === "combined"
      ? `combined = ${formula.replace(/_/g, " ")}(sub-metrics)`
      : `quality = ${qualityKey} (single sub-metric view)`;
  const weightEntries =
    qualityKey === "combined" && formula === "weighted_mean"
      ? Object.entries(weights).filter(([, w]) => w > 0)
      : [];

  const subMetrics = Object.entries(metrics)
    .filter(([k]) => k !== "combined")
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className="flex-1 min-w-0 overflow-auto p-6 pb-12">
      <EvalHeader ev={selectedEval} quality={quality} rank={rank} />

      {/* ── Score composition ─────────────────────────────────────────── */}
      <Card className="rounded-lg py-0">
        <CardContent className="p-4">
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold text-foreground">
              Score composition
            </h3>
            <span className="font-mono text-[11px] text-muted-foreground">
              {formulaText}
            </span>
          </div>
          <p className="mb-3 text-xs text-muted-foreground">
            Each sub-metric for this eval, with the project mean (▮ tick) and
            best (▯ tick) across GT evals for context.
          </p>
          {weightEntries.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-1.5">
              {weightEntries.map(([k, w]) => (
                <span
                  key={k}
                  className="rounded border bg-muted/40 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground"
                >
                  {prettyMetric(k)} × {w}
                </span>
              ))}
            </div>
          )}
          <div className="flex flex-col gap-2">
            {subMetrics.map(([k, v]) => (
              <MetricBar
                key={k}
                name={k}
                value={v}
                stats={subMetricStats.get(k)}
                emphasized={qualityKey === k}
              />
            ))}
          </div>
        </CardContent>
      </Card>

      {/* ── Attribution to pipeline stages ────────────────────────────── */}
      <div className="mt-4">
        <QualityByStage metrics={metrics} />
      </div>

      {/* ── Reference GT text metrics ─────────────────────────────────── */}
      {selectedEval.gt_metrics && (
        <Card className="mt-4 rounded-lg py-0">
          <CardContent className="p-4">
            <h3 className="mb-2 text-sm font-semibold text-foreground">
              Ground-truth text metrics
            </h3>
            <div className="flex flex-wrap gap-x-6 gap-y-1.5">
              {Object.entries(selectedEval.gt_metrics).map(([k, v]) => (
                <span key={k} className="inline-flex items-baseline gap-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.6px] text-muted-foreground">
                    {k.replace(/^gt_/, "")}
                  </span>
                  <span className="font-mono text-sm font-semibold text-foreground">
                    {v.toFixed(3)}
                  </span>
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function EvalHeader({
  ev,
  quality,
  rank,
}: {
  ev: Evaluation;
  quality: number | null;
  rank: { pos: number; of: number } | null;
}) {
  return (
    <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2">
      <span className="font-mono text-lg font-semibold text-foreground">
        Eval #{ev.eval_id}
      </span>
      <Badge variant="secondary" className="font-mono text-[10px] uppercase">
        {ev.eval_type}
      </Badge>
      <span className="inline-flex items-baseline gap-1.5">
        <span className="text-[10px] font-semibold uppercase tracking-[0.6px] text-muted-foreground">
          Quality
        </span>
        <span className="font-mono text-sm font-semibold text-foreground">
          {quality != null ? quality.toFixed(3) : "—"}
        </span>
      </span>
      {rank && (
        <span className="inline-flex items-baseline gap-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-[0.6px] text-muted-foreground">
            Rank
          </span>
          <span className="font-mono text-sm font-semibold text-foreground">
            #{rank.pos}
            <span className="font-normal text-muted-foreground"> / {rank.of} GT</span>
          </span>
        </span>
      )}
    </div>
  );
}

function prettyMetric(raw: string): string {
  return raw.replace(/^deepeval_/, "").replace(/^retrieval_/, "").replace(/_/g, " ");
}

function MetricBar({
  name,
  value,
  stats,
  emphasized,
}: {
  name: string;
  value: number;
  stats?: { mean: number; best: number; n: number };
  emphasized: boolean;
}) {
  const w = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className="grid grid-cols-[10rem_1fr_auto] items-center gap-3"
      title={
        stats
          ? `${name}: ${value.toFixed(4)} · project mean ${stats.mean.toFixed(4)} · best ${stats.best.toFixed(4)} (${stats.n} GT evals)`
          : `${name}: ${value.toFixed(4)}`
      }
    >
      <span
        className={
          "truncate text-[11px] " +
          (emphasized ? "font-semibold text-foreground" : "text-muted-foreground")
        }
      >
        {prettyMetric(name)}
      </span>
      <div className="relative h-1.5 min-w-0 rounded-full bg-muted">
        <span
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${w}%`, backgroundColor: ACCENT, opacity: emphasized ? 1 : 0.75 }}
        />
        {stats && (
          <>
            <span
              className="absolute -inset-y-0.5 w-px bg-foreground/35"
              style={{ left: `${Math.max(0, Math.min(1, stats.mean)) * 100}%` }}
            />
            <span
              className="absolute -inset-y-0.5 w-px bg-foreground/70"
              style={{ left: `${Math.max(0, Math.min(1, stats.best)) * 100}%` }}
            />
          </>
        )}
      </div>
      <span className="w-12 text-right font-mono text-[11px] tabular-nums text-foreground">
        {value.toFixed(3)}
      </span>
    </div>
  );
}

// ── Quality sub-metrics, grouped by the stages they judge ──────────────────
// (moved from StageBreakdownTable — quality attribution lives on the Quality
// tab now; the Performance tab stays pure performance.)
const Q_GROUPS: {
  title: string;
  stage: string;
  metrics: [string, string][];
}[] = [
  {
    title: "Retrieval quality",
    stage: "semantic_retrieval_vectorsearch",
    metrics: [
      ["retrieval_token_recall", "token recall"],
      ["retrieval_token_precision", "token precision"],
      ["deepeval_context_precision", "context precision"],
      ["deepeval_context_recall", "context recall"],
      ["deepeval_contextual_relevancy", "contextual relevancy"],
    ],
  },
  {
    title: "Answer quality",
    stage: "generator_prefill",
    metrics: [
      ["deepeval_answer_correctness", "answer correctness"],
      ["deepeval_faithfulness", "faithfulness"],
      ["deepeval_answer_relevancy", "answer relevancy"],
    ],
  },
];

function QualityByStage({ metrics }: { metrics: Record<string, number> }) {
  const groups = Q_GROUPS.map((g) => ({
    ...g,
    rows: g.metrics.filter(([k]) => metrics[k] != null),
  })).filter((g) => g.rows.length > 0);
  if (groups.length === 0) return null;

  return (
    <Card className="rounded-lg py-0">
      <CardContent className="p-4">
        <h3 className="text-sm font-semibold text-foreground">
          Quality by stage
        </h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Per-answer metrics, attributed to the stage most responsible for
          them — same stage colors as the Performance tab.
        </p>
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          {groups.map((g) => {
            const color = stageColor(g.stage);
            return (
              <div key={g.title}>
                <div className="mb-2 flex items-center gap-2">
                  <span
                    className="size-2.5 rounded-sm"
                    style={{ backgroundColor: color }}
                  />
                  <span className="text-[11px] font-semibold text-foreground">
                    {g.title}
                  </span>
                </div>
                <div className="flex flex-col gap-1.5">
                  {g.rows.map(([k, label]) => {
                    const v = metrics[k];
                    return (
                      <div
                        key={k}
                        className="grid grid-cols-[1fr_auto] items-center gap-2"
                      >
                        <div className="flex items-center gap-2">
                          <span className="w-32 shrink-0 truncate text-[11px] text-muted-foreground">
                            {label}
                          </span>
                          <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-muted">
                            <span
                              className="block h-full rounded-full"
                              style={{
                                width: `${Math.max(0, Math.min(1, v)) * 100}%`,
                                backgroundColor: color,
                              }}
                            />
                          </span>
                        </div>
                        <span className="w-10 text-right font-mono text-[11px] tabular-nums text-foreground">
                          {v.toFixed(3)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
