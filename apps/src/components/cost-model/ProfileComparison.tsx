/**
 * ProfileComparison — radar chart comparing efficiency profiles. No card.
 */
import React, { useMemo } from "react";
import type { EChartsOption } from "echarts";
import EChartsView from "@/components/charts/EChartsView";
import { CHART_COLORS } from "@/lib/theme";
import type { CalibrationProfile } from "@/types/costModel";

const BORDER_DEFAULT = "#E2E6EF";
const TEXT_TERTIARY = "#9CA3AF";

interface Props {
  profiles: CalibrationProfile[];
}

const RADAR_KEYS = [
  { key: "compute_efficiency", label: "Compute Eff" },
  { key: "blas_efficiency", label: "BLAS Eff" },
  { key: "simd_efficiency", label: "SIMD Eff" },
  { key: "pq_fastscan_efficiency", label: "PQ Scan Eff" },
  { key: "scattered_bw_ceiling", label: "Scattered BW" },
];

export function ProfileComparison({ profiles }: Props) {
  const option = useMemo<EChartsOption | null>(() => {
    if (profiles.length < 2) return null;

    return {
      tooltip: {},
      legend: {
        data: profiles.map((p) => p.hardware_preset),
        bottom: 0,
        textStyle: { fontSize: 11, color: TEXT_TERTIARY },
      },
      radar: {
        indicator: RADAR_KEYS.map((k) => ({
          name: k.label,
          max: 1.0,
        })),
        shape: "polygon" as const,
        splitArea: { show: false },
        axisName: { color: TEXT_TERTIARY },
        splitLine: { lineStyle: { color: BORDER_DEFAULT } },
        axisLine: { lineStyle: { color: BORDER_DEFAULT } },
      },
      series: [
        {
          type: "radar" as const,
          data: profiles.map((p, i) => ({
            value: RADAR_KEYS.map((k) => p[k.key] as number),
            name: p.hardware_preset,
            lineStyle: { color: CHART_COLORS[i % CHART_COLORS.length] },
            areaStyle: {
              color: CHART_COLORS[i % CHART_COLORS.length],
              opacity: 0.12,
            },
            itemStyle: { color: CHART_COLORS[i % CHART_COLORS.length] },
          })),
        },
      ],
    };
  }, [profiles]);

  if (profiles.length < 2) {
    return (
      <div className="py-8 text-center">
        <span className="text-sm text-muted-foreground">
          Select 2+ profiles to compare
        </span>
      </div>
    );
  }

  return (
    <section className="space-y-3">
      <h3 className="border-b border-border pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Efficiency Comparison
      </h3>
      {option && <EChartsView option={option} style={{ height: 320 }} />}
    </section>
  );
}
