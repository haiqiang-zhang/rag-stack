/**
 * BenchmarkChartGrid — 4-up grid of HV-diff / IGD vs (GT calls / wall time).
 *
 * Web port of apps/src/components/optimizer/BenchmarkChartGrid.tsx.
 *
 * The chart logic — per-seed step-sampling on a unioned x-grid, 5/50/95
 * percentile band rendered as a custom polygon (because ECharts' stacked
 * areaStyle is in linear data space and breaks on the log y-axis), tight-
 * fit min/max on both axes, custom tooltip formatter that filters out the
 * band's `custom`-type series, and the cross-emphasis `mouseover` handler
 * that dispatches `highlight` by series name so the band lights up with its
 * line — is preserved verbatim from the legacy version.
 *
 * Only the surrounding shell changed: each chart card is a shadcn `<Card>`
 * with `<CardContent>` instead of a Tamagui-styled YStack.
 */
import { useMemo } from "react";
import EChartsView from "@/components/charts/EChartsView";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { CHART_COLORS } from "@/lib/theme";
import type { OptimizerBenchmarkRow } from "@/types/optimizer";

// Neutral hex colors that stand in for the legacy theme tokens used inside
// the ECharts option (which takes raw color strings, not Tailwind classes).
// These match the same Gray-400 / Gray-200 / Gray-100 values the legacy
// theme.ts exported.
const TEXT_TERTIARY = "#9CA3AF";
const BORDER_DEFAULT = "#E2E6EF";
const BORDER_SUBTLE = "#F0F2F7";

interface Props {
  rows: OptimizerBenchmarkRow[];
  /**
   * Y-axis scale. `log` (default) = log10 — best for visualizing convergence
   * across orders of magnitude. `linear` = linear scale — better for early-
   * iteration absolute differences and for non-positive values.
   */
  yScale?: "log" | "linear";
  /**
   * Optional set of optimizer names that should be visible. If provided,
   * deselected optimizers are hidden from the chart via `legend.selected`,
   * keeping color/index assignments stable across selections.
   */
  selected?: Set<string>;
}

interface ChartSpec {
  title: string;
  xCol: keyof OptimizerBenchmarkRow;
  yCol: keyof OptimizerBenchmarkRow;
  xLabel: string;
  yLabel: string;
}

const CHARTS: ChartSpec[] = [
  {
    title: "HV Diff vs GT Calls",
    xCol: "cumulative_gt_calls",
    yCol: "hv_diff",
    xLabel: "Cumulative GT Calls",
    yLabel: "HV Difference (lower=better)",
  },
  {
    title: "IGD vs GT Calls",
    xCol: "cumulative_gt_calls",
    yCol: "igd",
    xLabel: "Cumulative GT Calls",
    yLabel: "IGD (lower=better)",
  },
  {
    title: "HV Diff vs Wall Time",
    xCol: "cumulative_optimizer_time_s",
    yCol: "hv_diff",
    xLabel: "Cumulative Optimizer Time (s)",
    yLabel: "HV Difference (lower=better)",
  },
  {
    title: "IGD vs Wall Time",
    xCol: "cumulative_optimizer_time_s",
    yCol: "igd",
    xLabel: "Cumulative Optimizer Time (s)",
    yLabel: "IGD (lower=better)",
  },
];

/**
 * Compact, rounded axis-label formatter shared by xAxis + yAxis.
 *   - 0 → "0"
 *   - very small / very large → exponential, 1 decimal (e.g. "1.2e-3", "3.4e+5")
 *   - mid-range → 3 significant digits (e.g. "0.00342", "1.23", "847")
 *
 * Default ECharts formatter for log axes shows raw floats like
 * "0.0034279861234" — too wide and ugly. This helper rounds consistently.
 */
function fmtAxisLabel(v: number | string): string {
  const n = typeof v === "number" ? v : Number(v);
  if (!isFinite(n)) return String(v);
  if (n === 0) return "0";
  const abs = Math.abs(n);
  if (abs < 1e-3 || abs >= 1e4) return n.toExponential(1);
  // toPrecision(3) handles "0.00342" + "1.23" + "847" all correctly.
  // Strip a trailing ".0" if toPrecision produced one (e.g. "100.").
  const s = n.toPrecision(3);
  return s.includes(".") ? s.replace(/\.?0+$/, "") || "0" : s;
}

/** Sample mean of an array (NaN on empty). */
function mean(xs: number[]): number {
  if (xs.length === 0) return NaN;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/** Sample standard deviation (n-1 denominator). 0 when n < 2. */
function stddev(xs: number[], mu: number): number {
  if (xs.length < 2) return 0;
  let s = 0;
  for (const x of xs) {
    const d = x - mu;
    s += d * d;
  }
  return Math.sqrt(s / (xs.length - 1));
}

function buildOption(
  yScale: "log" | "linear",
  rows: OptimizerBenchmarkRow[],
  spec: ChartSpec,
  optimizers: string[],
  selected?: Set<string>,
): any {
  if (rows.length === 0) {
    return {
      title: {
        text: "No data",
        left: "center",
        top: "center",
        textStyle: { color: TEXT_TERTIARY },
      },
    };
  }

  // Group by optimizer, then aggregate by xCol across seeds (mean +/- std).
  const series: any[] = [];
  // Per-(optimizer, x) std lookup, so the tooltip can show "mean ± std"
  // per hovered point — matching SummaryTable's display.
  const stdLookup = new Map<string, Map<number, number>>();
  // Track min/max x AND y across *all* optimizers so both axes can be
  // tight-fit. ECharts's default rounding gives ugly blank margins —
  //   x (type: "value") → rounds to "nice" integers (0-200 for data max 100)
  //   y (type: "log")   → rounds to decade boundaries (0.01-10 for data [0.3, 2.7])
  // We compute the exact data envelope (including the std band) and pin
  // min/max so the chart area is fully used.
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;

  for (let oi = 0; oi < optimizers.length; oi++) {
    const opt = optimizers[oi];
    const optRows = rows.filter((r) => r.optimizer === opt);
    const color = CHART_COLORS[oi % CHART_COLORS.length];
    // Whether this optimizer is currently checked in the SummaryTable.
    // Unchecked optimizers still get a series built (so color indexing
    // stays stable) but they don't contribute to the chart's x/y bounds —
    // otherwise an outlier run hidden by the user would still pin the
    // axis range and waste chart area.
    const isVisible = !selected || selected.has(opt);

    // Per-seed trajectory of (x, y), sorted by x. Cannot just `groupby(x)`
    // because for time-based x (`cumulative_optimizer_time_s`) each seed has
    // its own wall-time values — they don't align across seeds, so groupby
    // leaves every bucket with a single y and the mean line becomes a zigzag
    // through all 3 seeds' raw points.
    const seedTrajs = new Map<number, Array<[number, number]>>();
    for (const r of optRows) {
      const seed = r.seed as number;
      const xv = r[spec.xCol] as number;
      const yvRaw = r[spec.yCol] as number;
      // Clamp non-negative metrics (hv_diff, igd) to ≥ 0. Negative values
      // mean the optimizer's HV exceeded the reference Pareto's HV — a
      // reference-set artifact, not real "negative regret" (which is
      // physically impossible). Clamping at ingest avoids:
      //   1. log(<0) = NaN crashing ECharts on log y-axis
      //   2. mean − std crossing 0 → lower band visually plummeting to
      //      the axis floor when one or two seeds went negative.
      // The four metrics in CHARTS (hv_diff + igd vs gt_calls / wall time)
      // are all non-negative; bake the clamp in unconditionally.
      const yv = Number.isFinite(yvRaw) ? Math.max(0, yvRaw) : yvRaw;
      if (!seedTrajs.has(seed)) seedTrajs.set(seed, []);
      seedTrajs.get(seed)!.push([xv, yv]);
    }
    for (const traj of seedTrajs.values()) {
      traj.sort((a, b) => a[0] - b[0]);
    }

    // Common grid = union of all seeds' x values. At each grid point,
    // step-sample each seed's trajectory (last observation with x ≤ grid x),
    // then aggregate. For aligned x (iteration, gt_calls) this is identical
    // to groupby-mean; for unaligned x (wall time) it gives a smooth curve.
    const allXSet = new Set<number>();
    for (const traj of seedTrajs.values()) {
      for (const [x] of traj) allXSet.add(x);
    }
    const xVals = Array.from(allXSet).sort((a, b) => a - b);

    const meanData: [number, number][] = [];
    const upperData: [number, number][] = [];
    const lowerData: [number, number][] = [];
    const stdByX = new Map<number, number>();
    stdLookup.set(opt, stdByX);

    for (const gx of xVals) {
      const ys: number[] = [];
      for (const traj of seedTrajs.values()) {
        if (traj.length === 0 || gx < traj[0][0]) continue; // seed hasn't started yet
        let yv: number | null = null;
        for (const [x, y] of traj) {
          if (x <= gx) yv = y;
          else break;
        }
        // Keep zeros — they're meaningful for HV-diff / IGD (a zero-regret
        // run shouldn't be silently dropped). Log-axis safety is handled
        // when we set yMin below.
        if (yv !== null && Number.isFinite(yv)) ys.push(yv);
      }
      if (ys.length === 0) continue;

      // Arithmetic mean for the line + tooltip — matches SummaryTable's
      // (mean, std) verbatim so a hovered chart point reads identically
      // to the corresponding table row.
      const mu = mean(ys);
      const sd = stddev(ys, mu);
      meanData.push([gx, mu]);
      stdByX.set(gx, sd);

      // Band = ±1 SD = mean ± std. Purpose of these charts is comparing
      // run-to-run spread between MOBO optimizers — readers need to see
      // "how stable is this optimizer across seeds". ±1 SD is the textbook
      // RL/MOBO convention for that. Wide bands here are honest signal,
      // not a viz bug: when one optimizer's seeds disagree by an order of
      // magnitude, the band SHOULD be wide so the reader can't miss it.
      // Tooltip shows the same ±std value → matches SummaryTable verbatim.
      // hv_diff / igd are non-negative; clamp lower to 0 for log-axis
      // safety (negative raw values were clamped at ingest, but mu−sd can
      // still cross 0 when std > mean — happens whenever seeds disagree
      // more than the mean magnitude).
      const lo = Math.max(0, mu - sd);
      const hi = mu + sd;
      lowerData.push([gx, lo]);
      upperData.push([gx, hi]);
      // Only checked optimizers count toward the axis envelope. An outlier
      // run hidden in the table no longer pins min/max — chart re-fits to
      // whatever the user is actually looking at.
      if (isVisible) {
        if (gx < xMin) xMin = gx;
        if (gx > xMax) xMax = gx;
        if (lo < yMin) yMin = lo;
        if (hi > yMax) yMax = hi;
      }
    }

    // Mean line
    series.push({
      name: opt,
      id: opt,
      type: "line",
      data: meanData,
      symbol: "circle",
      symbolSize: 4,
      lineStyle: { width: 2, color },
      itemStyle: { color },
      emphasis: { focus: "series" },
    });

    // ±1 std band — drawn as a single closed polygon via a custom renderer.
    // The standard ECharts "stacked line + areaStyle" idiom is broken for
    // us: y-axis is log, and stack-add lives in linear data space. Custom
    // polygon = upper forward + lower reversed, closed shape.
    //
    // Naming: same `name` as the line so the legend toggles BOTH at once;
    // `id` is unique so ECharts can still distinguish them internally. The
    // `mouseover` event handler dispatches `highlight` by seriesName, which
    // also lights up the band when the line is hovered (and vice versa).
    if (lowerData.some((_, i) => upperData[i][1] !== lowerData[i][1])) {
      const upperArr = upperData.slice();
      const lowerArr = lowerData.slice();
      const fillColor = color;
      series.push({
        name: opt,
        id: `${opt}__band`,
        type: "custom",
        data: upperArr, // only drives dataIndex iteration
        renderItem: (params: any, api: any) => {
          if (params.dataIndex !== 0) return null; // draw the polygon once
          const points: [number, number][] = [];
          // Tiny positive sentinel for log-axis safety: when `lower = 0`
          // (mu − sd clamped at 0 because std > mean), log(0) maps to
          // -Infinity → api.coord returns invalid pixel y → polygon skips
          // those vertices → its bottom edge "jumps" diagonally from the
          // last valid lower point straight to the first valid upper point
          // (visible as a slanted line cutting across the band). Replacing
          // 0 with 1e-300 keeps log transform finite; the resulting pixel
          // sits far below the chart area but ECharts clips to the plot
          // rect, so visually the band just extends to the axis floor.
          const SAFE_LOG_FLOOR = 1e-300;
          for (let i = 0; i < upperArr.length; i++) {
            points.push(api.coord([upperArr[i][0], upperArr[i][1]]));
          }
          for (let i = lowerArr.length - 1; i >= 0; i--) {
            const yLo = lowerArr[i][1] > 0 ? lowerArr[i][1] : SAFE_LOG_FLOOR;
            points.push(api.coord([lowerArr[i][0], yLo]));
          }
          return {
            type: "polygon",
            shape: { points },
            style: { fill: fillColor, opacity: 0.15, stroke: "transparent" },
            // Band darkens when hovered (its line, by name, dispatches the
            // highlight action — see onEvents below).
            emphasisDisabled: false,
            emphasis: { style: { opacity: 0.32 } },
          };
        },
        tooltip: { show: false },
        z: 0, // render below the median lines
      });
    }
  }

  const fmtVal = (v: number | string): string => {
    if (typeof v !== "number" || !isFinite(v)) return String(v);
    const abs = Math.abs(v);
    if (abs !== 0 && (abs < 1e-3 || abs >= 1e4)) return v.toExponential(2);
    return v.toPrecision(4);
  };

  return {
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(17,24,39,0.92)",
      borderColor: "transparent",
      textStyle: { color: "#F9FAFB", fontSize: 11 },
      // Custom formatter: filter out custom-type series (the band polygons),
      // format numbers compactly, render one tidy row per optimizer. Default
      // ECharts tooltip dumps every series with full-precision floats and
      // covers the entire chart.
      formatter: (params: any) => {
        const rows = (Array.isArray(params) ? params : [params]).filter(
          (p: any) => p && p.seriesName && p.seriesType !== "custom",
        );
        if (rows.length === 0) return "";
        const xv = rows[0].axisValue;
        const header = `<div style="margin-bottom:4px;opacity:0.75">${spec.xLabel}: ${fmtVal(xv)}</div>`;
        const body = rows
          .map((p: any) => {
            const y = Array.isArray(p.value) ? p.value[1] : p.value;
            const x = Array.isArray(p.value) ? p.value[0] : p.axisValue;
            const sd = stdLookup.get(p.seriesName)?.get(x);
            const dot = `<span style="display:inline-block;width:8px;height:8px;background:${p.color};border-radius:50%;margin-right:6px"></span>`;
            const stdSuffix = sd !== undefined && Number.isFinite(sd) && sd > 0
              ? ` <span style="opacity:0.65">± ${fmtVal(sd)}</span>`
              : "";
            return `${dot}${p.seriesName}: <b>${fmtVal(y)}</b>${stdSuffix}`;
          })
          .join("<br/>");
        return header + body;
      },
    },
    legend: {
      type: "scroll",
      top: 0,
      left: "center",
      textStyle: { fontSize: 10, color: TEXT_TERTIARY },
      itemWidth: 10,
      itemHeight: 8,
      itemGap: 8,
      pageIconColor: TEXT_TERTIARY,
      pageIconInactiveColor: BORDER_DEFAULT,
      pageTextStyle: { color: TEXT_TERTIARY, fontSize: 10 },
      // Only the optimizers currently checked in SummaryTable get a legend
      // entry. Filtering `data` removes the row from the legend; series for
      // unselected optimizers are still in the series array (so their color
      // index stays stable across selections), but `selected` below hides
      // their lines + bands.
      data: selected ? optimizers.filter((o) => selected.has(o)) : optimizers,
      selected: selected
        ? Object.fromEntries(optimizers.map((o) => [o, selected.has(o)]))
        : undefined,
    },
    grid: { left: 60, right: 16, top: 28, bottom: 40 },
    xAxis: {
      type: "value",
      name: spec.xLabel,
      nameLocation: "middle",
      nameGap: 24,
      nameTextStyle: { fontSize: 10, color: TEXT_TERTIARY },
      axisLabel: { fontSize: 10, color: TEXT_TERTIARY, formatter: fmtAxisLabel },
      axisLine: { lineStyle: { color: BORDER_DEFAULT } },
      splitLine: { lineStyle: { color: BORDER_SUBTLE } },
      // Tight-fit to data range so the axis doesn't extend past the last
      // data point of the longest line. Default "nice" rounding produces
      // e.g. 0–200 for data max 100.
      min: isFinite(xMin) ? xMin : undefined,
      max: isFinite(xMax) ? xMax : undefined,
    },
    yAxis: {
      // ECharts uses "value" for linear axes (not "linear").
      type: yScale === "log" ? "log" : "value",
      name: spec.yLabel,
      nameLocation: "middle",
      nameGap: 46,
      nameTextStyle: { fontSize: 10, color: TEXT_TERTIARY },
      axisLabel: { fontSize: 10, color: TEXT_TERTIARY, formatter: fmtAxisLabel },
      splitLine: { lineStyle: { color: BORDER_SUBTLE } },
      // Tight-fit to the band envelope. log → multiplicative padding so the
      // band hugs the gridlines; linear → 0-based with small additive padding
      // (regret >= 0 by definition, so 0 is a meaningful floor).
      min: yScale === "log"
        ? (isFinite(yMin) && yMin > 0 ? yMin / 1.15 : undefined)
        : 0,
      max: yScale === "log"
        ? (isFinite(yMax) && yMax > 0 ? yMax * 1.15 : undefined)
        : (isFinite(yMax) && yMax > 0 ? yMax * 1.05 : undefined),
    },
    series,
  };
}

export function BenchmarkChartGrid({ rows, yScale = "log", selected }: Props) {
  const optimizers = useMemo(() => {
    const set = new Set(rows.map((r) => r.optimizer));
    return Array.from(set).sort();
  }, [rows]);

  const options = useMemo(
    () =>
      CHARTS.map((spec) => ({
        spec,
        option: buildOption(yScale, rows, spec, optimizers, selected),
      })),
    [rows, optimizers, yScale, selected],
  );

  if (rows.length === 0) return null;

  // Cross-emphasis: hovering the median line dispatches `highlight` by series
  // name, which lights up BOTH the line and its band (they share the name).
  // Mouseout dispatches `downplay` to undo the emphasis. Without this, ECharts
  // only emphasizes the exact series-index under the cursor — the band stays
  // dim because it's a separate (custom) series.
  const onEvents = {
    mouseover: (params: any, chart: any) => {
      if (params.componentType === "series" && params.seriesName && chart) {
        chart.dispatchAction({
          type: "highlight",
          seriesName: params.seriesName,
        });
      }
    },
    mouseout: (params: any, chart: any) => {
      if (params.componentType === "series" && params.seriesName && chart) {
        chart.dispatchAction({
          type: "downplay",
          seriesName: params.seriesName,
        });
      }
    },
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      {options.map(({ spec, option }) => (
        <Card key={spec.title} className="gap-0 py-0">
          <CardHeader className="border-b px-5 py-3.5">
            <CardTitle className="text-sm">{spec.title}</CardTitle>
            <CardDescription className="text-[11px]">
              Mean across seeds · band = ±1 std · {yScale} y-axis
            </CardDescription>
          </CardHeader>
          <CardContent className="px-2 py-3">
            <EChartsView
              option={option}
              style={{ height: 280, width: "100%" }}
              notMerge
              onEvents={onEvents}
            />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
