/**
 * BenchmarkChart — actual vs predicted latency, group toggle on top, no card.
 */
import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import EChartsView from "@/components/charts/EChartsView";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { BenchmarkRow } from "@/types/costModel";

// ECharts needs raw hex; keep these local to the chart renderer.
const ACCENT = "#0F766E";
const ERROR = "#EF4444";
const BORDER_DEFAULT = "#E2E6EF";
const TEXT_TERTIARY = "#9CA3AF";

interface Props {
  rows: BenchmarkRow[];
}

export function BenchmarkChart({ rows }: Props) {
  const groups = useMemo(() => {
    const map = new Map<string, BenchmarkRow[]>();
    for (const r of rows) {
      const key = `${r.index_name}|bs=${r.batch_size}|pm=${r.parallel_mode}`;
      if (!map.has(key)) map.set(key, []);
      map.get(key)!.push(r);
    }
    return map;
  }, [rows]);

  const groupKeys = useMemo(() => Array.from(groups.keys()), [groups]);
  const [selectedGroup, setSelectedGroup] = useState(0);

  const option = useMemo<EChartsOption | null>(() => {
    const key = groupKeys[selectedGroup];
    if (!key) return null;
    const data = groups.get(key)!.sort((a, b) => a.nprobe - b.nprobe);

    return {
      tooltip: { trigger: "axis" as const },
      legend: {
        data: ["Actual", "Predicted"],
        bottom: 0,
        textStyle: { fontSize: 11, color: TEXT_TERTIARY },
      },
      xAxis: {
        type: "category" as const,
        data: data.map((r) => String(r.nprobe)),
        name: "nprobe",
        nameTextStyle: { fontSize: 11, color: TEXT_TERTIARY },
        axisLabel: { color: TEXT_TERTIARY },
        axisLine: { lineStyle: { color: BORDER_DEFAULT } },
      },
      yAxis: {
        type: "value" as const,
        name: "Latency (ms)",
        nameTextStyle: { fontSize: 11, color: TEXT_TERTIARY },
        axisLabel: { color: TEXT_TERTIARY },
        splitLine: { lineStyle: { color: BORDER_DEFAULT } },
      },
      series: [
        {
          name: "Actual",
          type: "line" as const,
          data: data.map((r) => r.actual_latency_ms),
          lineStyle: { color: ACCENT },
          itemStyle: { color: ACCENT },
        },
        {
          name: "Predicted",
          type: "line" as const,
          data: data.map((r) => r.predicted_latency_ms),
          lineStyle: { color: ERROR, type: "dashed" as const },
          itemStyle: { color: ERROR },
        },
      ],
    };
  }, [groups, groupKeys, selectedGroup]);

  if (rows.length === 0) {
    return (
      <div className="py-8 text-center">
        <span className="text-sm text-muted-foreground">
          No benchmark data available
        </span>
      </div>
    );
  }

  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between border-b border-border pb-1.5">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Actual vs Predicted Latency
        </h3>
        <span className="text-[11px] text-muted-foreground/80">
          {groupKeys.length} group{groupKeys.length === 1 ? "" : "s"}
        </span>
      </div>

      {groupKeys.length > 1 && (
        <ToggleGroup
          type="single"
          value={String(selectedGroup)}
          onValueChange={(v) => v && setSelectedGroup(Number(v))}
          variant="outline"
          size="sm"
          className="flex flex-wrap justify-start gap-1"
        >
          {groupKeys.map((key, i) => (
            <ToggleGroupItem
              key={key}
              value={String(i)}
              className="h-7 px-2 font-mono text-[10px]"
            >
              {key}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      )}

      {option && <EChartsView option={option} style={{ height: 300 }} />}
    </section>
  );
}
