import { useMemo, useRef, useState, useEffect } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import type { EChartsOption } from "echarts";
import EChartsView from "@/components/charts/EChartsView";
import {
  buildBrush,
  buildDataZoom,
  useChartZoom,
  type ZoomState,
} from "@/components/charts/zoomControls";
import {
  NUMERIC_RAMP,
  PROGRESS_RAMP,
  bucketOf,
  categoryBuckets,
  type MapColorBy,
} from "@/components/charts/knobColoring";
import type { Evaluation } from "@/types";

// TODO(migrate): the original Tamagui theme exposed ACCENT/WARNING/TEXT_*
// color constants. They're consumed both by Tailwind classes (text-foreground
// etc.) AND raw ECharts option strings (which need an actual hex). The theme
// module now only exports CHART_COLORS/STAGE_COLORS, so we hardcode the
// neutral hex values below. If the design tokens shift, update here.
const ACCENT = "#0F766E";
const WARNING = "#F59E0B";
const ERROR = "#EF4444";
const TEXT_PRIMARY = "#111827";
const TEXT_SECONDARY = "#374151";
const TEXT_TERTIARY = "#6B7280";
const TEXT_MUTED = "#9CA3AF";
const BORDER_DEFAULT = "#E2E6EF";
const BORDER_SUBTLE = "#EEF1F6";

interface ParetoChartProps {
  evaluations: Evaluation[];
  /** Client-computed Pareto frontier evaluations */
  computedPareto: Evaluation[];
  onPointClick?: (evalId: number) => void;
  /** Quality value getter from PostAnalysisContext */
  getQuality: (ev: Evaluation) => number | null;
  /** Performance value getter from PerformancePolicyContext. Required — a raw
   *  ev.performance fallback here would silently ignore the active policy. */
  getPerformance: (ev: Evaluation) => number;
  /**
   * Whether an eval is SLO-feasible under the current policy. Infeasible
   * evals are kept on the chart but rendered as red triangles, with x taken
   * from `getDisplayPerformance` (the unconstrained best `optimize` value)
   * so a -1e9 INFEASIBLE_SCORE doesn't stretch the x-axis.
   */
  isFeasible: (ev: Evaluation) => boolean;
  /**
   * X-axis-friendly performance for chart positioning. For feasible evals
   * this equals the policy score; for infeasible evals this is the best
   * `optimize` value the sweep can achieve when the SLO is dropped (still
   * higher-is-better, so it lives on the same scale).
   */
  getDisplayPerformance: (ev: Evaluation) => number;
  /** Currently selected eval — gets the ACCENT halo + axis crosshairs. */
  selectedEvalId?: number | null;
  /** Self-describing x-axis title (metric + unit + direction) from the active
   *  performance policy. Defaults to a bare "Performance". */
  perfAxisLabel?: string;
  /** Trim legend / visualMap / axis names + margins for a narrow rail. */
  compact?: boolean;
  /**
   * Optional point-color encoding — this is what makes the map
   * multi-functional: null = plain map, "progress" = color by eval order,
   * a KnobDim = color by that searched knob (categorical → fixed palette +
   * shapes; numeric → sequential ramp). In compact mode the in-chart
   * legend/visualMap stays hidden — the host renders an external legend
   * (see knobColoring.categoryBuckets).
   */
  colorBy?: MapColorBy;
  height?: number;
  /**
   * Optional listener for zoom state. The chart canvas keeps its full size;
   * parents that want to render a "Reset zoom" affordance subscribe here and
   * render it in their own header chrome (e.g. SectionHeader's `right` slot).
   */
  onZoomStateChange?: (state: ZoomState) => void;
}

interface OverlapMenu {
  x: number;
  y: number;
  evals: Evaluation[];
}

export function ParetoChart({
  evaluations,
  computedPareto,
  onPointClick,
  getQuality,
  getPerformance,
  isFeasible,
  getDisplayPerformance,
  selectedEvalId,
  perfAxisLabel = "Performance",
  compact = false,
  colorBy = null,
  height = 400,
  onZoomStateChange,
}: ParetoChartProps) {
  const getDisplayX = getDisplayPerformance;
  const knobDim = colorBy !== null && colorBy !== "progress" ? colorBy : null;
  const [overlapMenu, setOverlapMenu] = useState<OverlapMenu | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const { onChartReady, onZoomEvent, onBrushEnd } = useChartZoom(onZoomStateChange);

  // Outside-click dismissal — but only when the click is OUTSIDE the menu
  // (clicking an item must NOT dismiss before the item's onPress runs).
  // Use mousedown so this fires before the click cycle completes.
  useEffect(() => {
    if (!overlapMenu) return;
    let armed = false;
    const arm = setTimeout(() => { armed = true; }, 50);
    const handler = (e: MouseEvent) => {
      if (!armed) return;
      const node = menuRef.current;
      if (node && e.target instanceof Node && node.contains(e.target)) {
        return; // click inside menu — let the item's onPress handle it
      }
      setOverlapMenu(null);
    };
    document.addEventListener("mousedown", handler);
    return () => {
      clearTimeout(arm);
      document.removeEventListener("mousedown", handler);
    };
  }, [overlapMenu]);

  // Split GT evals by feasibility — both groups are plotted, but infeasible
  // ones are *pinned* to the left edge of the feasible perf range so they
  // don't drag the x-axis to INFEASIBLE_SCORE (-1e9). Pinning preserves the
  // quality dimension so users can still see "this config has high quality
  // but violates the SLO".
  const gtEvals = useMemo(
    () =>
      evaluations.filter(
        (ev) => ev.eval_type === "GT" && getQuality(ev) != null && isFeasible(ev),
      ),
    [evaluations, getQuality, isFeasible],
  );
  const infeasibleGT = useMemo(
    () =>
      evaluations.filter(
        (ev) => ev.eval_type === "GT" && getQuality(ev) != null && !isFeasible(ev),
      ),
    [evaluations, getQuality, isFeasible],
  );

  // Use 3-decimal precision so near-overlaps (visually indistinguishable
  // points whose values differ by < 0.001) are grouped together. With 6
  // decimals two points were treated as "overlapping" only if every digit
  // matched — too strict for real data.
  const coordIndex = useMemo(() => {
    const map = new Map<string, Evaluation[]>();
    for (const ev of gtEvals) {
      const q = getQuality(ev)!;
      const p = getPerformance(ev);
      const key = `${p.toFixed(3)}_${q.toFixed(3)}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(ev);
    }
    return map;
  }, [gtEvals, getQuality, getPerformance]);


  const cmEvals = useMemo(
    () => evaluations.filter((ev) => ev.eval_type === "CM" && isFeasible(ev)),
    [evaluations, isFeasible],
  );

  const option = useMemo(() => {
    if (gtEvals.length === 0 && infeasibleGT.length === 0) return null;

    const cmData = cmEvals.map((ev) => [getPerformance(ev), 0, ev.eval_id]);

    // Infeasible points sit at their *unconstrained* best optimize-metric
    // value (via `getDisplayPerformance` → `displayScore`), on the SAME x
    // scale as feasible points. They render as red triangles so users can
    // tell at a glance "this config has high quality / decent throughput
    // but violates the SLO".
    const infeasibleData = infeasibleGT.map((ev) => [
      getDisplayX(ev),
      getQuality(ev)!,
      ev.eval_id,
    ]);

    // Client-computed Pareto line
    const paretoData = computedPareto
      .map((ev) => [getPerformance(ev), getQuality(ev)!, ev.eval_id])
      .sort((a, b) => (a[0] as number) - (b[0] as number));

    const series: any[] = [];
    const categoryLegend: string[] = [];
    let visualMap: any;
    const gtEmphasis = {
      itemStyle: { borderColor: TEXT_PRIMARY, borderWidth: 2, opacity: 1 },
      scale: 2,
    };

    if (knobDim && !knobDim.numeric) {
      // ── Categorical knob: one series per value, fixed color+shape order ──
      const buckets = categoryBuckets(knobDim);
      const byBucket = new Map<string, Evaluation[]>(buckets.map((b) => [b.value, []]));
      const missing: Evaluation[] = [];
      for (const ev of gtEvals) {
        const v = knobDim.values.get(ev.eval_id);
        if (v == null) {
          missing.push(ev);
          continue;
        }
        byBucket.get(bucketOf(buckets, v)?.value ?? "")?.push(ev);
      }
      for (const b of buckets) {
        const evs = byBucket.get(b.value)!;
        if (evs.length === 0) continue;
        categoryLegend.push(b.label);
        series.push({
          name: b.label,
          type: "scatter",
          data: evs.map((ev) => [getPerformance(ev), getQuality(ev)!, ev.eval_id]),
          symbol: b.symbol,
          symbolSize: 10,
          itemStyle: { color: b.color },
          emphasis: gtEmphasis,
          z: 5,
        });
      }
      if (missing.length > 0) {
        categoryLegend.push("knob absent");
        series.push({
          name: "knob absent",
          type: "scatter",
          data: missing.map((ev) => [getPerformance(ev), getQuality(ev)!, ev.eval_id]),
          symbol: "circle",
          symbolSize: 7,
          itemStyle: { color: TEXT_MUTED, opacity: 0.5 },
          emphasis: gtEmphasis,
          z: 4,
        });
      }
    } else {
      // ── Single GT series; visualMap (possibly invisible) drives coloring ──
      const rows =
        knobDim != null
          ? gtEvals.filter((ev) => knobDim.values.get(ev.eval_id) != null)
          : gtEvals;
      const gtData = rows.map((ev) =>
        knobDim != null
          ? [getPerformance(ev), getQuality(ev)!, Number(knobDim.values.get(ev.eval_id)), ev.eval_id]
          : [getPerformance(ev), getQuality(ev)!, ev.eval_id],
      );
      series.push({
        name: "GT Evaluations",
        type: "scatter",
        data: gtData,
        symbolSize: 10,
        // Plain map → one accent hue; colored modes let visualMap paint.
        itemStyle: colorBy === null ? { color: ACCENT, opacity: 0.85 } : undefined,
        emphasis: gtEmphasis,
        z: 5,
      });
      if (colorBy === "progress") {
        const ids = rows.map((ev) => ev.eval_id);
        visualMap = {
          type: "continuous",
          dimension: 2,
          min: ids.length ? Math.min(...ids) : 0,
          max: ids.length ? Math.max(...ids) : 1,
          precision: 0,
          orient: "horizontal",
          inRange: { color: PROGRESS_RAMP },
          text: ["late", "early"],
          textStyle: { color: TEXT_TERTIARY, fontSize: 10 },
          itemWidth: 12,
          itemHeight: 120,
          bottom: 0,
          left: "center",
          seriesIndex: series.length - 1,
          // Compact hosts render an external legend instead.
          show: !compact,
        };
      } else if (knobDim) {
        const vals = rows
          .map((ev) => Number(knobDim.values.get(ev.eval_id)))
          .filter((v) => Number.isFinite(v));
        visualMap = {
          type: "continuous",
          dimension: 2,
          min: vals.length ? Math.min(...vals) : 0,
          max: vals.length ? Math.max(...vals) : 1,
          orient: "horizontal",
          inRange: { color: NUMERIC_RAMP },
          text: [`${knobDim.label} high`, "low"],
          textStyle: { color: TEXT_TERTIARY, fontSize: 10 },
          itemWidth: 12,
          itemHeight: 120,
          bottom: 0,
          left: "center",
          seriesIndex: series.length - 1,
          show: !compact,
        };
      }
    }

    if (cmData.length > 0) {
      series.push({
        name: "CM Evaluations",
        type: "scatter",
        data: cmData,
        symbolSize: 5,
        large: true,
        largeThreshold: 200,
        itemStyle: { color: TEXT_MUTED, opacity: 0.5 },
        emphasis: {
          itemStyle: { color: TEXT_TERTIARY, borderColor: TEXT_PRIMARY, borderWidth: 1, opacity: 1 },
          scale: 2,
        },
        z: 1,
      });
    }

    if (infeasibleData.length > 0) {
      series.push({
        name: "Infeasible (SLO)",
        type: "scatter",
        data: infeasibleData,
        symbolSize: 11,
        symbol: "triangle",
        itemStyle: { color: ERROR, opacity: 0.65, borderColor: ERROR, borderWidth: 1.5 },
        emphasis: {
          itemStyle: { opacity: 1, borderColor: TEXT_PRIMARY, borderWidth: 2 },
          scale: 1.6,
        },
        z: 4,
      });
    }

    if (paretoData.length > 0) {
      series.push({
        name: "Pareto Frontier",
        type: "line",
        data: paretoData,
        smooth: false,
        symbolSize: 12,
        symbol: "diamond",
        lineStyle: { type: "dashed", width: 1.5, color: TEXT_SECONDARY },
        itemStyle: { color: TEXT_PRIMARY, borderColor: "#fff", borderWidth: 2 },
        emphasis: {
          itemStyle: { borderColor: TEXT_PRIMARY, borderWidth: 2, opacity: 1 },
          scale: 1.5,
        },
        z: 10,
      });
    }

    // Selected-eval halo + dashed axis crosshairs (matches the Hardware
    // Sweep "Selected" treatment so users learn one visual language).
    // The markLine lives on its own data-less series — when attached to
    // the halo scatter, ECharts can anchor the cross relative to that
    // series's data dims and the lines drift off the actual point.
    //
    // Search both feasible and infeasible sets so an SLO-violated eval also
    // gets the halo + crosshair treatment when selected. X coord uses the
    // *display* x (= unconstrained best for infeasible) so the halo lands on
    // the rendered triangle, not at -1e9.
    if (selectedEvalId != null) {
      const selEv =
        gtEvals.find((ev) => ev.eval_id === selectedEvalId)
        ?? infeasibleGT.find((ev) => ev.eval_id === selectedEvalId);
      if (selEv) {
        const sx = getDisplayX(selEv);
        const sy = getQuality(selEv)!;
        series.push({
          name: "Selected",
          type: "scatter",
          data: [[sx, sy]],
          symbolSize: 26,
          symbol: "circle",
          itemStyle: { color: ACCENT, opacity: 0.18, borderWidth: 0 },
          z: 3,
          silent: true,
          tooltip: { show: false },
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

    return {
      tooltip: {
        trigger: "item",
        enterable: false,
        backgroundColor: "rgba(17,24,39,0.92)",
        borderColor: "transparent",
        textStyle: { color: "#F9FAFB", fontSize: 12 },
        padding: [8, 12],
        formatter: (params: any) => {
          const d = params.data;
          if (!d || d.length < 2) return "";
          const perf = d[0] as number;
          const qual = d[1] as number;
          // eval_id is always the LAST tuple element (numeric-knob mode
          // inserts the knob value at index 2).
          const evalId = d.length >= 3 ? (d[d.length - 1] as number) : -1;
          const isCM = params.seriesName === "CM Evaluations";
          const isPareto = params.seriesName === "Pareto Frontier";
          const isInfeasible = params.seriesName === "Infeasible (SLO)";
          if (isCM) {
            return [
              `<b>Eval #${evalId}</b> <span style="color:#9CA3AF">(CM)</span>`,
              `Performance: <b>${perf.toFixed(4)}</b>`,
            ].join("<br/>");
          }
          if (isInfeasible) {
            // x = unconstrained best of `optimize` metric. Show it as
            // "Best (no SLO)" so users know this isn't the policy score.
            return [
              `<b style="color:${ERROR}">Eval #${evalId} · SLO infeasible</b>`,
              `<span style="color:#9CA3AF">No deployment satisfies the constraints.</span>`,
              `Quality: <b>${qual.toFixed(4)}</b>`,
              `Best (no SLO): <b>${perf.toFixed(4)}</b>`,
            ].join("<br/>");
          }
          const key = `${perf.toFixed(3)}_${qual.toFixed(3)}`;
          const group = coordIndex.get(key);
          const count = group?.length || 1;
          const lines: string[] = [];
          if (count > 1) {
            lines.push(`<b style="color:${WARNING}">${count} evals at this point</b>`);
            lines.push(`<span style="color:#9CA3AF">click to pick one</span>`);
          }
          lines.push(isPareto ? `<b>Pareto · Eval #${evalId}</b>` : `<b>Eval #${evalId}</b>`);
          lines.push(`Performance: <b>${perf.toFixed(4)}</b>`);
          lines.push(`Quality: <b>${qual.toFixed(4)}</b>`);
          if (knobDim) {
            const v = knobDim.values.get(evalId);
            lines.push(
              `${knobDim.label}: <b>${v != null ? String(v) : "—"}</b>`,
            );
          }
          return lines.join("<br/>");
        },
      },
      visualMap,
      legend: {
        show: !compact,
        bottom: 24,
        type: "scroll",
        textStyle: { fontSize: 11, color: TEXT_TERTIARY },
        itemWidth: 12,
        itemHeight: 8,
        selected: { "CM Evaluations": false },
        data: [
          ...(categoryLegend.length > 0 ? categoryLegend : ["GT Evaluations"]),
          "CM Evaluations",
          "Infeasible (SLO)",
          "Pareto Frontier",
          "Selected",
        ],
      },
      grid: compact
        ? { left: 40, right: 12, top: 12, bottom: 28 }
        : { left: 60, right: 20, top: 24, bottom: 100 },
      xAxis: {
        type: "value",
        name: compact ? "" : perfAxisLabel,
        nameLocation: "middle",
        nameGap: 28,
        nameTextStyle: { fontSize: 12, color: TEXT_TERTIARY },
        axisLine: { lineStyle: { color: BORDER_DEFAULT } },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: BORDER_SUBTLE } },
        axisLabel: { fontSize: compact ? 9 : 10, color: TEXT_TERTIARY, showMinLabel: false, showMaxLabel: false },
        scale: true,
      },
      yAxis: {
        type: "value",
        name: compact ? "" : "Quality",
        nameLocation: "middle",
        nameGap: 38,
        nameTextStyle: { fontSize: 12, color: TEXT_TERTIARY },
        axisLine: { lineStyle: { color: BORDER_DEFAULT } },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: BORDER_SUBTLE } },
        axisLabel: { fontSize: compact ? 9 : 10, color: TEXT_TERTIARY },
        scale: true,
      },
      // SHIFT+wheel zoom (Mapbox convention), plain-drag pan, SHIFT+drag
      // for box-zoom (via the brush component below). No slider chrome —
      // the plot canvas stays at full size. The Reset affordance lives in
      // the host's SectionHeader. `filterMode: 'none'` so the dashed Pareto
      // line / halo / crosshair stay visible when zoomed in.
      dataZoom: buildDataZoom(),
      brush: buildBrush(),
      // Disable per-point chart animation: with 200+ evals the staggered
      // appearance makes the chart feel unfinished while it draws and the
      // pareto line keeps redrawing as points settle. We render once,
      // instantly, when the data is ready.
      animation: false,
      progressive: 0,
      progressiveThreshold: Number.POSITIVE_INFINITY,
      series,
    };
  }, [gtEvals, infeasibleGT, cmEvals, computedPareto, coordIndex, getQuality, getPerformance, getDisplayX, selectedEvalId, perfAxisLabel, compact, colorBy, knobDim]);

  const onEvents = useMemo(() => {
    const events: Record<string, (params: unknown, chart: any) => void> = {
      datazoom: onZoomEvent,
      brushend: onBrushEnd,
    };
    if (onPointClick) {
      events.click = (params: unknown) => {
        const p = params as any;
        const d = p.data;
        if (!d || d.length < 2) return;
        const perf = d[0] as number;
        const qual = d[1] as number;
        const key = `${perf.toFixed(3)}_${qual.toFixed(3)}`;
        const group = coordIndex.get(key);
        // Page coordinates (clientX/clientY) — used with `position: fixed`
        // portal so we never have to worry about parent overflow / stacking.
        const wrapper = p.event;
        const native = wrapper?.event ?? wrapper;
        const x = native?.clientX ?? native?.pageX ?? wrapper?.clientX ?? 100;
        const y = native?.clientY ?? native?.pageY ?? wrapper?.clientY ?? 100;
        if (group && group.length > 1) {
          setOverlapMenu({ x, y, evals: group });
        } else if (group && group.length === 1) {
          setOverlapMenu(null);
          onPointClick(group[0].eval_id);
        } else if (d.length >= 3) {
          setOverlapMenu(null);
          // eval_id is the LAST tuple element (numeric-knob coloring inserts
          // the knob value at index 2).
          onPointClick(d[d.length - 1]);
        }
      };
    }
    return events;
  }, [onPointClick, coordIndex, onZoomEvent, onBrushEnd]);

  if (!option) return null;

  const MENU_W = 240;
  const MENU_H = 220;
  // Viewport-relative position for the fixed-portal menu. Clamp to viewport.
  let menuLeft = 0;
  let menuTop = 0;
  if (overlapMenu) {
    const vw = typeof window !== "undefined" ? window.innerWidth : 1200;
    const vh = typeof window !== "undefined" ? window.innerHeight : 800;
    menuLeft = Math.max(8, Math.min(overlapMenu.x + 12, vw - MENU_W - 8));
    menuTop = Math.max(8, Math.min(overlapMenu.y + 4, vh - MENU_H - 8));
  }

  const menuNode = overlapMenu && typeof document !== "undefined"
    ? createPortal(
        // Plain DOM positioning shim — `position: fixed` is the only thing
        // that reliably escapes parent overflow / stacking contexts.
        <div
          ref={menuRef}
          style={{
            position: "fixed",
            left: menuLeft,
            top: menuTop,
            zIndex: 9999,
          }}
        >
          <div className="w-[240px] overflow-hidden rounded-xl border border-border bg-card shadow-md">
            <div className="flex items-center justify-between gap-2 border-b border-border px-[14px] pt-[10px] pb-[6px]">
              <span className="text-[11px] font-semibold text-muted-foreground">SELECT EVALUATION</span>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-muted-foreground">{overlapMenu.evals.length} at this point</span>
                <button
                  type="button"
                  onClick={() => setOverlapMenu(null)}
                  className="cursor-pointer rounded p-0.5 hover:bg-muted"
                >
                  <X size={12} className="text-muted-foreground" />
                </button>
              </div>
            </div>
            <div className="overflow-auto" style={{ maxHeight: 180 }}>
              {overlapMenu.evals.map((ev) => (
                <div
                  key={ev.eval_id}
                  onClick={(e) => { e.stopPropagation?.(); setOverlapMenu(null); onPointClick?.(ev.eval_id); }}
                  className="flex cursor-pointer select-none items-center justify-between px-[14px] py-2 hover:bg-muted"
                >
                  <div className="flex items-center gap-2">
                    <div className="size-1.5 rounded-full" style={{ backgroundColor: ACCENT }} />
                    <span className="text-[13px] font-medium text-foreground">Eval #{ev.eval_id}</span>
                  </div>
                  <span className="text-[11px] text-muted-foreground">P:{getPerformance(ev).toFixed(2)} Q:{getQuality(ev)?.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>,
        document.body,
      )
    : null;

  return (
    <div style={{ height, width: "100%", position: "relative", overflow: "visible" }}>
      <EChartsView option={option as EChartsOption} style={{ height: "100%", width: "100%" }} onEvents={onEvents} onChartReady={onChartReady} notMerge={true} />
      {menuNode}
    </div>
  );
}
