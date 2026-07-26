/**
 * StageBreakdownChart — stacked per-stage latency, group toggle, no card.
 */
import React, { useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import EChartsView from "@/components/charts/EChartsView";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { BenchmarkRow } from "@/types/costModel";

const BORDER_DEFAULT = "#E2E6EF";
const BORDER_SUBTLE = "#F0F2F7";
const TEXT_TERTIARY = "#9CA3AF";

interface Props {
  rows: BenchmarkRow[];
}

const STAGES = [
  { prefix: "1_coarse_quantizer", label: "Coarse Quantizer", color: "#0F766E" },
  { prefix: "2_lut_build",        label: "LUT Build",        color: "#06B6D4" },
  { prefix: "3_pq_scan_adc",      label: "PQ Scan (ADC)",    color: "#10B981" },
  { prefix: "4_heap_selection",   label: "Heap Selection",   color: "#F59E0B" },
];

export function StageBreakdownChart({ rows }: Props) {
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
    const nprobeLabels = data.map((r) => String(r.nprobe));

    const series = STAGES.map((stage) => ({
      name: stage.label,
      type: "line" as const,
      stack: "total",
      areaStyle: { opacity: 0.35 },
      lineStyle: { width: 1.5 },
      symbol: "circle",
      symbolSize: 5,
      itemStyle: { color: stage.color },
      emphasis: { focus: "series" as const },
      data: data.map((r) => {
        const v = r[`${stage.prefix}_time_ms`];
        return typeof v === "number" ? v : parseFloat(String(v)) || 0;
      }),
    }));

    return {
      tooltip: {
        trigger: "axis" as const,
        backgroundColor: "rgba(17,24,39,0.92)",
        borderColor: "transparent",
        textStyle: { color: "#F9FAFB", fontSize: 11 },
        formatter: (params: any) => {
          const arr = Array.isArray(params) ? params : [params];
          if (!arr.length) return "";
          const idx = arr[0].dataIndex;
          const row = data[idx];
          const nprobe = arr[0].axisValue;
          let total = 0;
          let html = `<b>nprobe = ${nprobe}</b><br/>`;
          for (const stage of STAGES) {
            const time = parseFloat(String(row[`${stage.prefix}_time_ms`])) || 0;
            const bound = String(row[`${stage.prefix}_bottleneck`] ?? "");
            const tag = bound === "compute"
              ? `<span style="color:#818CF8">compute</span>`
              : `<span style="color:#FCD34D">memory</span>`;
            total += time;
            html += `<span style="color:${stage.color}">●</span> ${stage.label}: <b>${time.toFixed(4)}</b> ms ${tag}<br/>`;
          }
          html += `<br/>Total predicted: <b>${(parseFloat(String(row.predicted_latency_ms)) || 0).toFixed(4)}</b> ms`;
          return html;
        },
      },
      legend: {
        bottom: 0,
        textStyle: { fontSize: 11, color: TEXT_TERTIARY },
        itemWidth: 14,
        itemHeight: 8,
      },
      grid: { left: 60, right: 20, top: 20, bottom: 56 },
      xAxis: {
        type: "category" as const,
        data: nprobeLabels,
        name: "nprobe",
        nameLocation: "middle" as const,
        nameGap: 28,
        nameTextStyle: { fontSize: 11, color: TEXT_TERTIARY },
        axisLabel: { color: TEXT_TERTIARY, fontSize: 10 },
        axisLine: { lineStyle: { color: BORDER_DEFAULT } },
        boundaryGap: false,
      },
      yAxis: {
        type: "value" as const,
        name: "Latency (ms)",
        nameTextStyle: { fontSize: 11, color: TEXT_TERTIARY },
        axisLabel: { color: TEXT_TERTIARY, fontSize: 10 },
        splitLine: { lineStyle: { color: BORDER_SUBTLE } },
      },
      series,
    };
  }, [groups, groupKeys, selectedGroup]);

  if (rows.length === 0 || !(`1_coarse_quantizer_time_ms` in rows[0])) {
    return null;
  }

  return (
    <section className="space-y-3">
      <div className="flex items-baseline justify-between border-b border-border pb-1.5">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Predicted Latency · Per-Stage Breakdown
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

      {option && <EChartsView option={option} style={{ height: 320 }} />}
    </section>
  );
}
