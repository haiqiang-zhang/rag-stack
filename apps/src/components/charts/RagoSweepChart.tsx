import { useCallback, useMemo } from "react";
import EChartsView from "@/components/charts/EChartsView";
import {
  buildBrush,
  buildDataZoom,
  useChartZoom,
  type ZoomState,
} from "@/components/charts/zoomControls";
import type { RagoSweepPoint, AssemblyResult } from "@/types";
import { CHART_COLORS } from "@/lib/theme";

// TODO(migrate): the original Tamagui theme exposed ACCENT/WARNING/TEXT_*
// color constants. ECharts options take raw color strings, so we hardcode
// neutrals here — keep in sync with ParetoChart.tsx if these tokens change.
const ACCENT = "#0F766E";
const WARNING = "#F59E0B";
const TEXT_PRIMARY = "#111827";
const TEXT_TERTIARY = "#6B7280";
const TEXT_MUTED = "#9CA3AF";
const BORDER_DEFAULT = "#E2E6EF";
const BORDER_SUBTLE = "#EEF1F6";

function finiteNumber(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

interface RagoSweepChartProps {
  sweepPoints: RagoSweepPoint[];
  assembly: AssemblyResult | null;
  xKey: string;
  yKey: string;
  colorKey: string;
  xLabel: string;
  yLabel: string;
  height?: number;
  /** Optional: the policy-best sweep row. Rendered as a "Best" pin when the
   *  user has manually picked a different point. */
  selectedBestRow?: RagoSweepPoint | null;
  /** Optional: the user's currently picked sweep point (clicked). Defaults
   *  to selectedBestRow. Rendered as a halo + axis crosshairs. */
  selectedSweepRow?: RagoSweepPoint | null;
  onPointClick?: (point: RagoSweepPoint) => void;
  /** See ParetoChart's identically-named prop. */
  onZoomStateChange?: (state: ZoomState) => void;
}

export function RagoSweepChart({
  sweepPoints, assembly, xKey, yKey, colorKey, xLabel, yLabel, height = 400,
  selectedBestRow,
  selectedSweepRow, onPointClick,
  onZoomStateChange,
}: RagoSweepChartProps) {
  // Resolve the "best" point once at the component level so both the chart
  // option (rendering) and the click handler (selection) reference the same
  // object. Many configs project to the same (x, y) — without this, clicking
  // the Best pin would `find` the first match, which usually isn't bestPt.
  const { onChartReady, onZoomEvent, onBrushEnd } = useChartZoom(onZoomStateChange);

  const bestPt = useMemo<RagoSweepPoint | null | undefined>(() => {
    if (selectedBestRow) return selectedBestRow;
    if (assembly?.best_hw_params) {
      const hw = assembly.best_hw_params;
      return sweepPoints.find((pt) =>
        Object.entries(hw).every(([k, v]) => String(pt[k]) === String(v))
      );
    }
    return null;
  }, [selectedBestRow, assembly, sweepPoints]);
  const selPt = selectedSweepRow ?? bestPt;

  const option = useMemo<any>(() => {
    if (sweepPoints.length === 0) {
      return { title: { text: "No RAGO sweep data", left: "center", top: "center", textStyle: { color: TEXT_MUTED, fontSize: 14, fontWeight: "normal" as const } } };
    }
    const groups: Record<string, number[][]> = {};
    for (let rowIndex = 0; rowIndex < sweepPoints.length; rowIndex++) {
      const pt = sweepPoints[rowIndex];
      const x = finiteNumber(pt[xKey]);
      const y = finiteNumber(pt[yKey]);
      if (x == null || y == null) continue;
      const category = String(pt[colorKey] ?? "unknown");
      if (!groups[category]) groups[category] = [];
      groups[category].push([x, y, rowIndex]);
    }
    if (Object.keys(groups).length === 0) {
      return {
        title: {
          text: "No finite data for selected axes",
          left: "center",
          top: "center",
          textStyle: { color: TEXT_MUTED, fontSize: 14, fontWeight: "normal" as const },
        },
        xAxis: { type: "value", name: xLabel },
        yAxis: { type: "value", name: yLabel },
        series: [],
      };
    }
    const series: any[] = Object.entries(groups).map(([name, data], i) => ({
      name, type: "scatter", data, symbolSize: 7,
      itemStyle: { color: CHART_COLORS[i % CHART_COLORS.length], opacity: 0.7 },
      emphasis: { itemStyle: { color: CHART_COLORS[i % CHART_COLORS.length], borderColor: TEXT_PRIMARY, borderWidth: 1, opacity: 1 }, scale: 2.5 },
      z: 5,
      // Keep normal per-point hit testing for common thousand-point sweeps.
      // Very large historical sweeps still get batched rendering.
      large: true,
      largeThreshold: 5000,
      progressive: 2000,
      progressiveThreshold: 5000,
    }));

    // Selection halo + axis crosshairs at the currently-selected point
    // (defaults to the best point). When selected === best, this halo
    // sits AROUND the Best marker below — they combine into one composite
    // indicator at that point. The crosshairs live on a separate empty
    // series so ECharts anchors them purely to axis values, not to any
    // host series' data dimensions.
    if (selPt) {
      const sx = finiteNumber(selPt[xKey]);
      const sy = finiteNumber(selPt[yKey]);
      const rowIndex = sweepPoints.indexOf(selPt);
      if (sx != null && sy != null) {
        series.push({
          name: "Selected",
          type: "scatter",
          data: [[sx, sy, rowIndex]],
          symbolSize: 26,
          symbol: "circle",
          itemStyle: { color: ACCENT, opacity: 0.18, borderColor: ACCENT, borderWidth: 0 },
          z: 3,
          silent: true,
        });
        series.push({
          name: "_crosshair",
          type: "line",
          data: [],
          silent: true,
          showSymbol: false,
          legendHoverLink: false,
          tooltip: { show: false },
          markLine: {
            silent: true,
            symbol: ["none", "none"],
            label: { show: false },
            lineStyle: { color: ACCENT, type: "dashed", opacity: 0.45, width: 1 },
            data: [{ xAxis: sx }, { yAxis: sy }],
          },
          z: 2,
        });
      }
    }

    // Policy-optimum reference marker — ALWAYS visible. Diamond shape +
    // soft outer glow makes it visually distinct from the category dots
    // and the ACCENT halo (which marks user inspection state). Internal
    // series name stays "Best" so the click handler keeps working; the
    // legend formatter below renders it as "★ Optimal (policy)".
    if (bestPt) {
      const bx = finiteNumber(bestPt[xKey]);
      const by = finiteNumber(bestPt[yKey]);
      const rowIndex = sweepPoints.indexOf(bestPt);
      if (bx != null && by != null) {
        // Outer glow (renders below the diamond — same trick as the halo).
        series.push({
          name: "_optimal_glow",
          type: "scatter",
          data: [[bx, by, rowIndex]],
          symbolSize: 26,
          symbol: "circle",
          itemStyle: { color: WARNING, opacity: 0.18, borderWidth: 0 },
          z: 19,
          silent: true,
          tooltip: { show: false },
        });
        series.push({
          name: "Best",
          type: "scatter",
          data: [[bx, by, rowIndex]],
          symbolSize: 14,
          symbol: "diamond",
          itemStyle: { color: WARNING, borderColor: "#fff", borderWidth: 2, opacity: 1 },
          z: 20,
        });
      }
    }
    return {
      // Keep the normal populated-chart option explicit; EChartsView replaces
      // the full option for this chart so stale empty states and stale series
      // cannot survive eval/axis/color changes.
      title: { text: "", show: false },
      tooltip: { trigger: "item", backgroundColor: "rgba(17,24,39,0.92)", borderColor: "transparent", textStyle: { color: "#F9FAFB", fontSize: 12 }, padding: [8, 12],
        formatter: (params: any) => { const d = params.data; if (!d) return ""; return [`<b>${params.seriesName}</b>`, `${xLabel}: <b>${d[0].toFixed(4)}</b>`, `${yLabel}: <b>${d[1].toFixed(4)}</b>`].join("<br/>"); },
      },
      legend: (() => {
        // Exclude UX-state series (halo, crosshair, glow, "Selected"
        // indicator) — they're feedback for user interaction, not data
        // layers, so showing them in the legend implies they're toggleable
        // categories and confuses viewers when "Selected" sits on top of
        // "Best".
        const visible = series.filter(
          (s) => !s.name.startsWith("_") && s.name !== "Selected",
        );
        // The "Best" entry is distinguished from category dots through three
        // independent visual channels — shape (diamond vs circle, auto from
        // series symbol), color (WARNING amber), and weight (semi-bold) —
        // so it reads as a different *kind* of marker without leaning on a
        // decorative glyph.
        const data = visible.map((s) =>
          s.name === "Best"
            ? {
                name: s.name,
                textStyle: {
                  color: WARNING,
                  fontWeight: 600,
                  fontSize: 10,
                },
              }
            : s.name,
        );
        return {
          type: "scroll",
          top: 4,
          left: "center",
          // Slightly larger legend marks: 10×8 was too tight for diamond
          // (its inscribed rhombus uses only ~half the box area, so it
          // looked smaller than the inscribed-circle category dots at
          // matching dimensions).
          textStyle: { fontSize: 11, color: TEXT_TERTIARY },
          itemWidth: 16,
          itemHeight: 14,
          itemGap: 12,
          padding: [2, 8],
          data,
          formatter: (name: string) => (name === "Best" ? "Optimal" : name),
        };
      })(),
      grid: { left: 56, right: 16, top: 36, bottom: 44 },
      xAxis: { type: "value", name: xLabel, nameLocation: "middle", nameGap: 24, nameTextStyle: { fontSize: 11, color: TEXT_TERTIARY }, axisLine: { lineStyle: { color: BORDER_DEFAULT } }, axisTick: { show: false }, splitLine: { lineStyle: { color: BORDER_SUBTLE } }, axisLabel: { fontSize: 10, color: TEXT_TERTIARY }, scale: true },
      yAxis: { type: "value", name: yLabel, nameLocation: "middle", nameGap: 36, nameTextStyle: { fontSize: 11, color: TEXT_TERTIARY }, axisLine: { lineStyle: { color: BORDER_DEFAULT } }, axisTick: { show: false }, splitLine: { lineStyle: { color: BORDER_SUBTLE } }, axisLabel: { fontSize: 10, color: TEXT_TERTIARY }, scale: true },
      // SHIFT+wheel zoom, plain-drag pan, SHIFT+drag for box-zoom (via the
      // brush component below). No slider chrome. The Reset affordance is
      // rendered by the host (SweepCard) in its SectionHeader so the plot
      // keeps its full size.
      dataZoom: buildDataZoom(),
      brush: buildBrush(),
      animation: true, animationDuration: 300, series,
    };
  }, [sweepPoints, assembly, xKey, yKey, colorKey, xLabel, yLabel, bestPt, selPt]);

  const handleClick = useCallback(
    (params: unknown) => {
      if (!onPointClick) return;
      const p = params as any;
      const cat = p?.seriesName;
      // Overlay clicks resolve directly to the known object — many configs
      // share the same (x, y) projection, so a (x, y) lookup would pick an
      // arbitrary one of them, NOT the actual best/selected row.
      if (cat === "Best" && bestPt) {
        onPointClick(bestPt);
        return;
      }
      if (cat === "Selected" && selPt) {
        onPointClick(selPt);
        return;
      }
      const rowIndex = Number(p?.data?.[2]);
      const match = Number.isInteger(rowIndex) ? sweepPoints[rowIndex] : null;
      if (match) onPointClick(match);
    },
    [sweepPoints, onPointClick, bestPt, selPt],
  );

  const onEvents = useMemo(() => {
    const events: Record<string, (params: unknown, chart: any) => void> = {
      datazoom: onZoomEvent,
      brushend: onBrushEnd,
    };
    if (onPointClick) events.click = handleClick;
    return events;
  }, [onPointClick, handleClick, onZoomEvent, onBrushEnd]);

  return (
    <div style={{ height, width: "100%" }}>
      <EChartsView
        option={option}
        height={height}
        style={{ height, width: "100%" }}
        // Series membership changes across evals, axes, color fields, and
        // selected rows. Replacing the option prevents stale sweep points from
        // previous views from staying on screen.
        notMerge
        onEvents={onEvents}
        onChartReady={onChartReady}
      />
    </div>
  );
}
