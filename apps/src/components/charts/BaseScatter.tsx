import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import EChartsView from "@/components/charts/EChartsView";

export interface ScatterSeries {
  name: string;
  /** Each data point: [x, y, ...customData] */
  data: (number | string)[][];
  color: string;
  symbolSize?: number;
  symbol?: string;
  opacity?: number;
  /** Whether to draw a connecting line through points */
  showLine?: boolean;
}

interface BaseScatterProps {
  series: ScatterSeries[];
  xAxisLabel: string;
  yAxisLabel: string;
  title?: string;
  height?: number;
  onPointClick?: (seriesName: string, dataIndex: number, data: (number | string)[]) => void;
  /** Custom tooltip formatter */
  tooltipFormatter?: (params: Record<string, unknown>) => string;
}

/**
 * Generic ECharts scatter chart component.
 *
 * Accepts series data, axis labels, and an optional click handler.
 * ParetoChart and RagoSweepChart are thin wrappers around this.
 */
export function BaseScatter({
  series,
  xAxisLabel,
  yAxisLabel,
  title,
  height = 400,
  onPointClick,
  tooltipFormatter,
}: BaseScatterProps) {
  const option: EChartsOption = useMemo(
    () => ({
      title: title ? { text: title, left: "center", textStyle: { fontSize: 14 } } : undefined,
      tooltip: {
        trigger: "item",
        formatter: tooltipFormatter as unknown as string | undefined,
      },
      legend: {
        bottom: 0,
        data: series.map((s) => s.name),
      },
      grid: {
        left: 60,
        right: 20,
        top: title ? 40 : 20,
        bottom: 40,
      },
      xAxis: {
        type: "value",
        name: xAxisLabel,
        nameLocation: "middle",
        nameGap: 30,
        scale: true,
      },
      yAxis: {
        type: "value",
        name: yAxisLabel,
        nameLocation: "middle",
        nameGap: 40,
        scale: true,
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, zoomOnMouseWheel: "shift", moveOnMouseWheel: false, moveOnMouseMove: false },
        { type: "inside", yAxisIndex: 0, zoomOnMouseWheel: "shift", moveOnMouseWheel: false, moveOnMouseMove: false },
      ],
      series: series.map((s) => ({
        name: s.name,
        type: "scatter" as const,
        data: s.data,
        symbolSize: s.symbolSize || 8,
        symbol: s.symbol || "circle",
        itemStyle: {
          color: s.color,
          opacity: s.opacity ?? 0.8,
        },
        ...(s.showLine
          ? {
              type: "line" as const,
              smooth: false,
              lineStyle: { type: "dashed" as const, width: 1, color: s.color },
              symbolSize: s.symbolSize || 12,
              symbol: s.symbol || "diamond",
            }
          : {}),
      })),
    }),
    [series, xAxisLabel, yAxisLabel, title, tooltipFormatter]
  );

  const onEvents = useMemo(() => {
    if (!onPointClick) return undefined;
    return {
      click: (params: unknown) => {
        const p = params as { seriesName?: string; dataIndex?: number; data?: unknown };
        if (p.seriesName != null && p.dataIndex != null && p.data) {
          onPointClick(
            p.seriesName,
            p.dataIndex,
            p.data as (number | string)[]
          );
        }
      },
    };
  }, [onPointClick]);

  return (
    <div style={{ height, width: "100%" }}>
      <EChartsView
        option={option}
        style={{ height, width: "100%" }}
        onEvents={onEvents}
        notMerge={true}
      />
    </div>
  );
}
