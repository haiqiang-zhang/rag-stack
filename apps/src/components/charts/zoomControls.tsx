/**
 * Shared zoom-interaction primitives for our ECharts scatter charts.
 *
 * HCI choice: zero chrome on the canvas, modifier-key gated interactions.
 *
 *   - Plain mouse wheel  → page scroll (untouched).
 *   - Plain mouse drag   → pan the zoomed plot (`dataZoom-inside`).
 *   - Plain click        → select point (host's onPointClick).
 *   - SHIFT + wheel      → cursor-anchored zoom (`zoomOnMouseWheel: "shift"`).
 *   - SHIFT + drag       → box-zoom: draw a rect, release to zoom into it.
 *                          Implemented via the native `brush` component,
 *                          activated on Shift keydown via `takeGlobalCursor`
 *                          and translated into a `dataZoom` dispatch on
 *                          `brushEnd`. No toolbox icons, no visible buttons.
 *   - Reset zoom         → pill rendered by the host in its SectionHeader.
 *
 * The chart canvas keeps its original margins; no slider, no header strip.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { ECharts } from "echarts/core";
import { RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCENT = "#0F766E";

/** Inside dataZoom on both axes. No sliders — those would eat plot area. */
export function buildDataZoom() {
  return [
    {
      type: "inside" as const,
      xAxisIndex: 0,
      zoomOnMouseWheel: "shift" as const,
      moveOnMouseWheel: false,
      moveOnMouseMove: true,
      filterMode: "none" as const,
    },
    {
      type: "inside" as const,
      yAxisIndex: 0,
      zoomOnMouseWheel: "shift" as const,
      moveOnMouseWheel: false,
      moveOnMouseMove: true,
      filterMode: "none" as const,
    },
  ];
}

/**
 * Brush config — invisible by default. Activated by Shift-keydown via
 * `takeGlobalCursor`. The rectangle the user draws is converted into a
 * `dataZoom` range and the brush is cleared on completion.
 */
export function buildBrush() {
  return {
    // Empty toolbox: brush is fully driven by keyboard + drag, no icons.
    toolbox: [] as string[],
    xAxisIndex: 0,
    yAxisIndex: 0,
    brushType: "rect" as const,
    brushMode: "single" as const,
    transformable: false,
    removeOnClick: false,
    brushStyle: {
      borderColor: ACCENT,
      borderWidth: 1,
      color: "rgba(15,118,110,0.08)",
    },
    throttleType: "debounce" as const,
    throttleDelay: 0,
  };
}

export interface ZoomState {
  isZoomed: boolean;
  reset: () => void;
}

/**
 * Wires up the full zoom interaction for one chart:
 *   - `onChartReady` captures the chart instance and installs a global
 *     Shift-keydown/up listener that toggles brush mode via `takeGlobalCursor`.
 *   - `onZoomEvent` (datazoom) updates `isZoomed` so the host can show/hide
 *     its Reset pill.
 *   - `onBrushEnd` translates the user-drawn rect into a `dataZoom` dispatch
 *     and clears the brush mark.
 *   - `reset` dispatches a full restore.
 *
 * `onStateChange` lets the host render the Reset pill in its own header
 * chrome rather than inside the chart canvas.
 */
export function useChartZoom(onStateChange?: (state: ZoomState) => void) {
  const chartRef = useRef<ECharts | null>(null);
  const [isZoomed, setIsZoomed] = useState(false);

  const onChartReady = useCallback((chart: ECharts) => {
    chartRef.current = chart;
  }, []);

  // Shift held → enter brush rect mode. Release → back to default cursor
  // (so plain drag pans again). Repeated keydown events while held are a
  // no-op since `takeGlobalCursor` is idempotent.
  useEffect(() => {
    const enterBrush = () => {
      chartRef.current?.dispatchAction({
        type: "takeGlobalCursor",
        key: "brush",
        brushOption: { brushType: "rect", brushMode: "single" },
      });
    };
    const exitBrush = () => {
      chartRef.current?.dispatchAction({ type: "takeGlobalCursor", key: null });
      // Clear any half-drawn brush rect that might be lingering if the user
      // released Shift mid-drag.
      chartRef.current?.dispatchAction({ type: "brush", areas: [] });
    };
    const onDown = (e: KeyboardEvent) => {
      if (e.key === "Shift" && !e.repeat) enterBrush();
    };
    const onUp = (e: KeyboardEvent) => {
      if (e.key === "Shift") exitBrush();
    };
    // Window-blur fires when user Alt-Tabs while holding Shift — without this
    // the chart would stay stuck in brush mode after they switch back.
    const onBlur = () => exitBrush();
    window.addEventListener("keydown", onDown);
    window.addEventListener("keyup", onUp);
    window.addEventListener("blur", onBlur);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
      window.removeEventListener("blur", onBlur);
    };
  }, []);

  const onZoomEvent = useCallback((_params: unknown, chart: ECharts) => {
    chartRef.current = chart;
    const opt = chart.getOption() as { dataZoom?: Array<{ start?: number; end?: number }> };
    const dzs = opt.dataZoom ?? [];
    const zoomed = dzs.some((dz) => {
      const start = Number(dz.start ?? 0);
      const end = Number(dz.end ?? 100);
      return start > 0.5 || end < 99.5;
    });
    setIsZoomed(zoomed);
  }, []);

  const onBrushEnd = useCallback((params: unknown, chart: ECharts) => {
    const p = params as { areas?: Array<{ brushType?: string; coordRange?: number[][] }> };
    const area = p.areas?.[0];
    if (!area || area.brushType !== "rect" || !area.coordRange) return;
    const [[x0, x1], [y0, y1]] = area.coordRange;
    // Filter zero-sized brushes (a Shift-click with no drag motion can fire
    // brushEnd with a degenerate rect on some browsers).
    if (x0 === x1 || y0 === y1) {
      chart.dispatchAction({ type: "brush", areas: [] });
      return;
    }
    chart.dispatchAction({
      type: "dataZoom",
      dataZoomIndex: 0,
      startValue: Math.min(x0, x1),
      endValue: Math.max(x0, x1),
    });
    chart.dispatchAction({
      type: "dataZoom",
      dataZoomIndex: 1,
      startValue: Math.min(y0, y1),
      endValue: Math.max(y0, y1),
    });
    chart.dispatchAction({ type: "brush", areas: [] });
  }, []);

  const reset = useCallback(() => {
    chartRef.current?.dispatchAction({ type: "dataZoom", start: 0, end: 100 });
  }, []);

  // Push state changes upward so parents can render external chrome.
  useEffect(() => {
    onStateChange?.({ isZoomed, reset });
  }, [isZoomed, reset, onStateChange]);

  return { chartRef, isZoomed, onChartReady, onZoomEvent, onBrushEnd, reset };
}

/**
 * Compact pill — icon + label. No internal positioning; the parent decides
 * where it goes (typically a SectionHeader's `right` slot). Fades in/out
 * cleanly when toggled.
 */
export function ResetZoomPill({
  visible,
  onClick,
}: {
  visible: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label="Reset chart zoom"
      className={cn(
        "flex items-center gap-1.5 rounded-md",
        "border border-border bg-card/95 px-2 py-1",
        "text-[10.5px] font-medium tracking-wide text-muted-foreground",
        "shadow-sm",
        "transition-all duration-200 ease-out",
        "hover:border-foreground/25 hover:bg-card hover:text-foreground",
        visible
          ? "translate-y-0 opacity-100"
          : "pointer-events-none -translate-y-0.5 opacity-0",
      )}
    >
      <RotateCcw className="size-3" />
      Reset zoom
    </button>
  );
}
