/**
 * Web ECharts wrapper — direct echarts canvas/svg via the official `echarts`
 * package. Replaces the apps/ version that used @wuba/react-native-echarts.
 *
 * Same public API as the old wrapper so callers don't change:
 *   - <EChartsView option style notMerge onEvents />
 *   - onEvents handlers receive `(params, chart)` so cross-series emphasis
 *     via `chart.dispatchAction(...)` keeps working.
 */
import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import {
  LineChart,
  ScatterChart,
  BarChart,
  HeatmapChart,
  CustomChart,
  RadarChart,
} from "echarts/charts";
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  VisualMapComponent,
  MarkPointComponent,
  MarkLineComponent,
  BrushComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsOption } from "echarts";

echarts.use([
  CanvasRenderer,
  LineChart,
  ScatterChart,
  BarChart,
  HeatmapChart,
  CustomChart, // band polygons in BenchmarkChartGrid
  RadarChart,  // ProfileComparison cost-model profile radar
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
  VisualMapComponent,
  MarkPointComponent,
  MarkLineComponent,
  BrushComponent,
]);

/**
 * Tooltips portal into <body> by default: our charts live inside overflow
 * containers and next to sidebars that would otherwise clip the bubble or
 * cover it (e.g. the Benchmark project rail). An option that explicitly sets
 * tooltip.appendToBody / appendTo / confine keeps its own behavior.
 *
 * With appendToBody the bubble escapes container clipping but ECharts still
 * positions it relative to the chart, so near the window edge it can run off
 * screen — the injected `position` callback flips/clamps it to the viewport.
 */
function clampedTooltipPosition(getContainer: () => HTMLElement | null) {
  return (
    point: [number, number],
    _params: unknown,
    _dom: unknown,
    _rect: unknown,
    size: { contentSize: [number, number] },
  ): [number, number] => {
    const [w, h] = size.contentSize;
    const crect = getContainer()?.getBoundingClientRect();
    const cl = crect?.left ?? 0;
    const ct = crect?.top ?? 0;
    const pad = 8;
    // Container-relative coords (ECharts adds the page offset itself);
    // clamp in viewport space via the container's bounding rect.
    let x = point[0] + 14;
    if (cl + x + w > window.innerWidth - pad) x = point[0] - w - 14;
    if (cl + x < pad) x = pad - cl;
    let y = point[1] + 14;
    if (ct + y + h > window.innerHeight - pad) y = point[1] - h - 14;
    if (ct + y < pad) y = pad - ct;
    return [x, y];
  };
}

function withPortaledTooltip(
  option: EChartsOption,
  getContainer: () => HTMLElement | null,
): EChartsOption {
  const t = (option as any).tooltip;
  if (!t) return option;
  const patch = (tt: any) => {
    if (!tt) return tt;
    if (
      tt.appendToBody !== undefined ||
      tt.appendTo !== undefined ||
      tt.confine !== undefined
    ) {
      return tt; // explicit per-chart behavior wins
    }
    return {
      ...tt,
      appendToBody: true,
      position: tt.position ?? clampedTooltipPosition(getContainer),
    };
  };
  return {
    ...option,
    tooltip: Array.isArray(t) ? t.map(patch) : patch(t),
  } as EChartsOption;
}

export interface EChartsViewProps {
  option: EChartsOption;
  height?: number;
  style?: React.CSSProperties;
  notMerge?: boolean;
  /**
   * ECharts event handlers — `mouseover`, `click`, etc. Each handler receives
   * `(params, chart)` so it can call `chart.dispatchAction(...)` for
   * cross-series emphasis or programmatic actions.
   */
  onEvents?: Record<
    string,
    (params: unknown, chart: echarts.ECharts) => void
  >;
  /**
   * Fires once after `echarts.init` + initial setOption. Lets parents grab
   * the chart instance without waiting for the first user interaction —
   * needed when a global keyboard listener wants to dispatchAction (e.g.
   * Shift-key brush activation) before any data event has fired.
   */
  onChartReady?: (chart: echarts.ECharts) => void;
}

export default function EChartsView({
  option,
  height = 350,
  style,
  notMerge = false,
  onEvents,
  onChartReady,
}: EChartsViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const instanceRef = useRef<echarts.ECharts | null>(null);

  // Stable ref to the latest event handlers. Binding `chart.on(...)` directly
  // with the current handler at mount captures stale closures — common when
  // a parent's click handler relies on data that loads asynchronously
  // (e.g. RagoSweepChart's `handleClick` uses `sweepPoints`, which is empty
  // at first render and populated later). Stable wrappers read from this ref
  // each invocation, so handlers always see fresh state without rebinding.
  const eventsRef = useRef(onEvents);
  useEffect(() => {
    eventsRef.current = onEvents;
  }, [onEvents]);

  // Init + dispose
  useEffect(() => {
    if (!containerRef.current) return;
    const chart = echarts.init(containerRef.current, "light", {
      renderer: "canvas",
    });
    instanceRef.current = chart;
    chart.setOption(withPortaledTooltip(option, () => containerRef.current), notMerge);
    onChartReady?.(chart);

    // Bind stable wrappers for every event name passed at mount time.
    // The wrapper reads the latest handler from `eventsRef` so closures
    // always see fresh state; the wrapper itself is registered once, which
    // avoids `chart.off`/`chart.on` churn on every parent re-render.
    if (onEvents) {
      for (const eventName of Object.keys(onEvents)) {
        chart.on(eventName, (params: unknown) => {
          const handler = eventsRef.current?.[eventName];
          handler?.(params, chart);
        });
      }
    }

    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.dispose();
      instanceRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // init once; re-renders push new option via the effect below

  // Update option (no re-init).
  //
  // Dispatching `downplay` first clears ECharts' runtime "highlighted"
  // state — set by user click/hover and otherwise persistent across
  // setOption calls even with notMerge:true. Without this, charts that
  // re-render markLine/scatter overlays at a new coordinate (e.g. the
  // "Selected" halo + crosshair in RagoSweepChart) get the new option
  // but the visual emphasis stays anchored to the previously clicked point.
  useEffect(() => {
    const chart = instanceRef.current;
    if (!chart) return;
    chart.dispatchAction({ type: "downplay" });
    chart.setOption(withPortaledTooltip(option, () => containerRef.current), notMerge);
  }, [option, notMerge]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height, ...style }}
    />
  );
}
