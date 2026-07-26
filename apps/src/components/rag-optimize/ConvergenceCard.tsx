import { useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import EChartsView from "@/components/charts/EChartsView";
import type { EChartsOption } from "echarts";
import type { Evaluation } from "@/types";
import { isEvalFailure } from "@/types";
import { CHART_COLORS } from "@/lib/theme";
import { useEvalData } from "@/context/EvalDataContext";

const BORDER_SUBTLE = "#F0F2F7";
const BORDER_DEFAULT = "#E2E6EF";
const TEXT_TERTIARY = "#9CA3AF";
const ERROR = "#EF4444";

type PerfResult = { score: number; feasible: boolean };

/**
 * Optimizer progress over time: running-best performance & quality
 * (convergence), and the per-eval timeline. Reads the shared eval data so it
 * agrees with the sidebar Pareto and the History table.
 */
export function ConvergenceCard() {
  const { evaluations, getQuality, getPerformanceResult } = useEvalData();
  const cOpt = useMemo(
    () => buildConvergenceChart(evaluations, getQuality, getPerformanceResult),
    [evaluations, getQuality, getPerformanceResult],
  );
  const tOpt = useMemo(
    () => buildTimelineChart(evaluations, getQuality, getPerformanceResult),
    [evaluations, getQuality, getPerformanceResult],
  );

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Convergence</CardTitle>
          <p className="text-xs text-muted-foreground">
            Best performance &amp; quality found so far, vs evaluation number.
          </p>
        </CardHeader>
        <CardContent>
          <div style={{ height: 320 }}>
            <EChartsView option={cOpt} style={{ height: 320, width: "100%" }} notMerge />
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Evaluation timeline</CardTitle>
          <p className="text-xs text-muted-foreground">
            Every eval as it was tried — measured, failed, or SLO-infeasible.
          </p>
        </CardHeader>
        <CardContent>
          <div style={{ height: 320 }}>
            <EChartsView option={tOpt} style={{ height: 320, width: "100%" }} notMerge />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function buildConvergenceChart(
  evals: Evaluation[],
  getQuality: (ev: Evaluation) => number | null,
  getPerformanceResult: (ev: Evaluation) => PerfResult,
): EChartsOption {
  if (!evals.length)
    return { title: { text: "No data", left: "center", top: "center", textStyle: { color: TEXT_TERTIARY } } };
  const sorted = [...evals].sort((a, b) => a.eval_id - b.eval_id);
  let bP = -Infinity,
    bQ = -Infinity;
  const pD: number[][] = [],
    qD: number[][] = [];
  for (const ev of sorted) {
    const r = getPerformanceResult(ev);
    if (r.feasible) {
      if (r.score > bP) bP = r.score;
      if (bP > -Infinity) pD.push([ev.eval_id, bP]);
    }
    const q = getQuality(ev);
    if (q != null && q > bQ) bQ = q;
    if (bQ > -Infinity) qD.push([ev.eval_id, bQ]);
  }
  return {
    tooltip: { trigger: "axis", backgroundColor: "rgba(17,24,39,0.92)", borderColor: "transparent", textStyle: { color: "#F9FAFB" } },
    legend: { bottom: 0, textStyle: { fontSize: 11, color: TEXT_TERTIARY } },
    grid: { left: 56, right: 56, top: 16, bottom: 60 },
    xAxis: { type: "value", name: "Eval ID", nameLocation: "middle", nameGap: 30, nameTextStyle: { color: TEXT_TERTIARY }, axisLabel: { color: TEXT_TERTIARY }, axisLine: { lineStyle: { color: BORDER_DEFAULT } }, splitLine: { lineStyle: { color: BORDER_SUBTLE } } },
    yAxis: [
      { type: "value", name: "Best performance", position: "left", nameLocation: "middle", nameGap: 40, nameTextStyle: { color: TEXT_TERTIARY }, axisLabel: { color: TEXT_TERTIARY }, splitLine: { lineStyle: { color: BORDER_SUBTLE } } },
      { type: "value", name: "Best quality", position: "right", nameLocation: "middle", nameGap: 40, nameTextStyle: { color: TEXT_TERTIARY }, axisLabel: { color: TEXT_TERTIARY }, splitLine: { show: false } },
    ],
    series: [
      { name: "Performance", type: "line", data: pD, smooth: true, lineStyle: { color: CHART_COLORS[0] }, itemStyle: { color: CHART_COLORS[0] }, yAxisIndex: 0 },
      { name: "Quality", type: "line", data: qD, smooth: true, lineStyle: { color: CHART_COLORS[1] }, itemStyle: { color: CHART_COLORS[1] }, yAxisIndex: 1 },
    ],
  } as EChartsOption;
}

function buildTimelineChart(
  evals: Evaluation[],
  getQuality: (ev: Evaluation) => number | null,
  getPerformanceResult: (ev: Evaluation) => PerfResult,
): EChartsOption {
  if (!evals.length)
    return { title: { text: "No data", left: "center", top: "center", textStyle: { color: TEXT_TERTIARY } } };

  const feasiblePerf: number[][] = [];
  const infeasibleEvalIds: number[] = [];
  for (const e of evals) {
    if (isEvalFailure(e.eval_type)) continue;
    const r = getPerformanceResult(e);
    if (r.feasible) feasiblePerf.push([e.eval_id, r.score]);
    else infeasibleEvalIds.push(e.eval_id);
  }

  const series: NonNullable<EChartsOption["series"]> = [
    { name: "Performance", type: "scatter", data: feasiblePerf, symbolSize: 8, itemStyle: { color: CHART_COLORS[0] }, yAxisIndex: 0 },
    { name: "Quality", type: "scatter", data: evals.filter((e) => e.eval_type === "GT").map((e) => { const q = getQuality(e); return q != null ? [e.eval_id, q] : null; }).filter(Boolean) as number[][], symbolSize: 8, itemStyle: { color: CHART_COLORS[1] }, yAxisIndex: 1 },
    { name: "Failed", type: "scatter", data: evals.filter((e) => isEvalFailure(e.eval_type)).map((e) => [e.eval_id, 0]), symbolSize: 8, symbol: "triangle", itemStyle: { color: ERROR, opacity: 0.7 }, yAxisIndex: 0 },
  ];
  if (infeasibleEvalIds.length > 0) {
    const yMin = feasiblePerf.length > 0 ? Math.min(...feasiblePerf.map((p) => p[1])) : 0;
    series.splice(1, 0, { name: "Infeasible (SLO)", type: "scatter", data: infeasibleEvalIds.map((id) => [id, yMin]), symbolSize: 8, symbol: "diamond", itemStyle: { color: ERROR, opacity: 0.6 }, yAxisIndex: 0 });
  }

  return {
    tooltip: { trigger: "item", backgroundColor: "rgba(17,24,39,0.92)", borderColor: "transparent", textStyle: { color: "#F9FAFB" } },
    legend: { bottom: 0, textStyle: { fontSize: 11, color: TEXT_TERTIARY }, itemWidth: 10, itemHeight: 10 },
    grid: { left: 56, right: 56, top: 16, bottom: 60 },
    xAxis: { type: "value", name: "Eval ID", nameLocation: "middle", nameGap: 30, nameTextStyle: { color: TEXT_TERTIARY }, axisLabel: { color: TEXT_TERTIARY }, axisLine: { lineStyle: { color: BORDER_DEFAULT } }, splitLine: { lineStyle: { color: BORDER_SUBTLE } } },
    yAxis: [
      { type: "value", name: "Performance", position: "left", nameLocation: "middle", nameGap: 40, scale: true, nameTextStyle: { color: TEXT_TERTIARY }, axisLabel: { color: TEXT_TERTIARY }, splitLine: { lineStyle: { color: BORDER_SUBTLE } } },
      { type: "value", name: "Quality", position: "right", nameLocation: "middle", nameGap: 40, scale: true, nameTextStyle: { color: TEXT_TERTIARY }, axisLabel: { color: TEXT_TERTIARY }, splitLine: { show: false } },
    ],
    series,
  } as EChartsOption;
}
