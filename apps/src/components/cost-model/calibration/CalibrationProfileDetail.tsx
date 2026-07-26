import React, { useMemo } from "react";
import type { EChartsOption } from "echarts";
import EChartsView from "@/components/charts/EChartsView";
import type { CalibrationProfileRecord } from "@/types/costModel";
import { CHART_COLORS } from "@/lib/theme";

// from old theme.ts
const TEXT_TERTIARY = "#9CA3AF";
const BORDER_DEFAULT = "#E2E6EF";
const BORDER_SUBTLE = "#F0F2F7";

interface Props {
  profile: CalibrationProfileRecord;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60">
        {label.toUpperCase()}
      </span>
      <span className="text-[15px] font-bold text-foreground">
        {value}
      </span>
    </div>
  );
}

export function CalibrationProfileDetail({ profile }: Props) {
  const chartOption = useMemo<EChartsOption | null>(() => {
    const grid = (Array.isArray(profile.grid) ? profile.grid : []).filter((point) => Number.isFinite(point.predicted_s) && Number.isFinite(point.measured_s));
    if (grid.length === 0) return null;

    // Group points by batch_size for color coding
    const groups: Record<number, number[][]> = {};
    let maxVal = 0;
    for (const p of grid) {
      if (!groups[p.batch_size]) groups[p.batch_size] = [];
      groups[p.batch_size].push([p.predicted_s, p.measured_s, p.seq_len, p.rel_err]);
      if (p.predicted_s > maxVal) maxVal = p.predicted_s;
      if (p.measured_s > maxVal) maxVal = p.measured_s;
    }
    maxVal *= 1.1;

    const series: any[] = Object.entries(groups).map(([bs, pts], i) => ({
      name: `batch=${bs}`,
      type: "scatter",
      data: pts,
      symbolSize: 10,
      itemStyle: { color: CHART_COLORS[i % CHART_COLORS.length], opacity: 0.85 },
    }));

    // y = x reference line
    series.push({
      name: "ideal (y=x)",
      type: "line",
      data: [[0, 0], [maxVal, maxVal]],
      symbol: "none",
      lineStyle: { color: TEXT_TERTIARY, type: "dashed", width: 1 },
      silent: true,
      tooltip: { show: false },
    });

    return {
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(17,24,39,0.92)",
        borderColor: "transparent",
        textStyle: { color: "#F9FAFB", fontSize: 12 },
        padding: [8, 12],
        formatter: (params: any) => {
          if (params.seriesName === "ideal (y=x)") return "";
          const [predicted, measured, seqLen, relErr] = params.data;
          return [
            `<b>${params.seriesName}, seq=${seqLen}</b>`,
            `predicted: <b>${predicted.toFixed(4)}s</b>`,
            `measured:  <b>${measured.toFixed(4)}s</b>`,
            `rel_err:   <b>${((relErr ?? 0) * 100).toFixed(1)}%</b>`,
          ].join("<br/>");
        },
      },
      legend: { bottom: 0, textStyle: { fontSize: 11, color: TEXT_TERTIARY } },
      grid: { left: 60, right: 20, top: 20, bottom: 60 },
      xAxis: {
        type: "value", name: "Predicted (s)", nameLocation: "middle", nameGap: 32,
        nameTextStyle: { fontSize: 12, color: TEXT_TERTIARY },
        axisLine: { lineStyle: { color: BORDER_DEFAULT } },
        splitLine: { lineStyle: { color: BORDER_SUBTLE } },
        axisLabel: { fontSize: 10, color: TEXT_TERTIARY },
      },
      yAxis: {
        type: "value", name: "Measured (s)", nameLocation: "middle", nameGap: 42,
        nameTextStyle: { fontSize: 12, color: TEXT_TERTIARY },
        axisLine: { lineStyle: { color: BORDER_DEFAULT } },
        splitLine: { lineStyle: { color: BORDER_SUBTLE } },
        axisLabel: { fontSize: 10, color: TEXT_TERTIARY },
      },
      animation: true, animationDuration: 300, series,
    } as EChartsOption;
  }, [profile.grid]);

  if (profile.error) {
    return (
      <div className="rounded-xl bg-card p-4 shadow-sm">
        <span className="text-[13px] font-semibold text-foreground">Failed to read profile</span>
        <span className="mt-1 block text-xs text-muted-foreground">{profile.error}</span>
      </div>
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-4">
      {/* Stat row */}
      <div className="flex flex-wrap gap-6 rounded-xl bg-card p-4 shadow-sm">
        {typeof profile.compute_efficiency === "number" && <Stat label="Compute eff" value={profile.compute_efficiency.toFixed(3)} />}
        {typeof profile.memory_efficiency === "number" && <Stat label="Memory eff" value={profile.memory_efficiency.toFixed(3)} />}
        {Number.isFinite(profile.fit_mape) && <Stat label="Fit MAPE" value={`${(profile.fit_mape * 100).toFixed(1)}%`} />}
        {typeof profile.correlation === "number" && <Stat label="Correlation" value={profile.correlation.toFixed(3)} />}
        <Stat label="Grid points" value={String(profile.grid?.length ?? 0)} />
        {profile.bits && <Stat label="Bits" value={profile.bits} />}
      </div>

      {profile.match_scope === "shared_model_candidate" && <p className="rounded-lg border p-3 text-xs leading-relaxed text-muted-foreground">This shared model record is a calibration candidate. CM checks the component, model, hardware, and workload before selecting it.</p>}
      {profile.linear_law != null && <div className="min-w-0 space-y-2 rounded-lg border p-4"><h3 className="text-sm font-medium">Linear cost law</h3><pre className="max-w-full overflow-auto text-xs">{JSON.stringify(profile.linear_law, null, 2)}</pre></div>}

      {/* Source card — only what the sticky header doesn't already show */}
      {(profile.cuda_name || profile.filename) && (
        <div className="flex flex-col gap-1.5 rounded-xl bg-card p-4 shadow-sm">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60">
            SOURCE
          </span>
          {profile.filename && (
            <div className="flex gap-2">
              <span className="w-[90px] text-xs text-muted-foreground">file</span>
              <span className="min-w-0 break-all font-mono text-xs text-foreground">{profile.filename}</span>
            </div>
          )}
          {profile.cuda_name && (
            <div className="flex gap-2">
              <span className="w-[90px] text-xs text-muted-foreground">cuda_name</span>
              <span className="min-w-0 break-all font-mono text-xs text-foreground">{profile.cuda_name}</span>
            </div>
          )}
        </div>
      )}

      {/* Scatter */}
      <div className="rounded-xl bg-card p-4 shadow-sm">
        <span className="mb-2 block text-[13px] font-semibold text-foreground">
          Measured vs Predicted
        </span>
        {chartOption ? (
          <div style={{ height: 360, width: "100%" }}>
            <EChartsView option={chartOption} style={{ height: 360, width: "100%" }} notMerge />
          </div>
        ) : (
          <span className="text-xs text-muted-foreground">No grid points in this profile.</span>
        )}
      </div>

      {/* Grid table */}
      {Array.isArray(profile.grid) && profile.grid.length > 0 && (
        <div className="flex flex-col gap-2 rounded-xl bg-card p-4 shadow-sm">
          <span className="text-[13px] font-semibold text-foreground">Grid points ({profile.grid.length})</span>
          <div className="max-w-full overflow-x-auto rounded-lg bg-muted p-2">
            <div className="min-w-[490px]">
            <div className="flex border-b border-border/40 py-1">
              <span className="w-[70px] text-[11px] font-semibold text-muted-foreground">batch</span>
              <span className="w-[90px] text-[11px] font-semibold text-muted-foreground">seq_len</span>
              <span className="w-[110px] text-[11px] font-semibold text-muted-foreground">measured (s)</span>
              <span className="w-[110px] text-[11px] font-semibold text-muted-foreground">predicted (s)</span>
              <span className="w-[90px] text-[11px] font-semibold text-muted-foreground">rel_err</span>
            </div>
            {profile.grid.map((pt, i) => (
              <div key={i} className="flex border-b border-border/40 py-[3px]">
                <span className="w-[70px] text-[11px] text-foreground">{pt.batch_size}</span>
                <span className="w-[90px] text-[11px] text-foreground">{pt.seq_len}</span>
                <span className="w-[110px] text-[11px] text-foreground">{typeof pt.measured_s === "number" ? pt.measured_s.toFixed(4) : "—"}</span>
                <span className="w-[110px] text-[11px] text-foreground">{typeof pt.predicted_s === "number" ? pt.predicted_s.toFixed(4) : "—"}</span>
                <span className="w-[90px] text-[11px] text-foreground">
                  {typeof pt.rel_err === "number" ? `${(pt.rel_err * 100).toFixed(1)}%` : "—"}
                </span>
              </div>
            ))}
            </div>
          </div>
        </div>
      )}
      <details className="min-w-0 rounded-lg border p-4"><summary className="cursor-pointer text-sm font-medium">Profile record</summary><pre className="mt-3 max-h-96 max-w-full overflow-auto text-xs">{JSON.stringify(profile, null, 2)}</pre></details>
    </div>
  );
}
