/**
 * Scaling explorer — "what if you resize a stage?"
 *
 * Both modes plot STAGE THROUGHPUT (requests/second), not raw batch latency:
 * req/s = batch ÷ (dec_steps × time). That makes curves comparable across
 * stages, puts them on the same scale as the Stage-load block and the e2e
 * rate, and answers the actual decision ("does a bigger batch / more chips
 * buy throughput?"). Validated against assembly.json: the derived value at
 * the operating cell reproduces the CM's reported per-stage qps.
 *
 *   - Compare stages: one line per stage at its current chip count, with the
 *     pipeline's current e2e rate as a dashed reference line.
 *   - One stage in depth: one line per chip count (light → dark shades of the
 *     stage's color) — the vertical gap between lines is what chips buy.
 */
import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { Loader2 } from "lucide-react";
import EChartsView from "@/components/charts/EChartsView";
import { stageColor } from "@/lib/theme";
import type { SweepRow } from "@/types";
import type { StageInfo } from "@/hooks/useEvalBreakdown";
import { PillTabs } from "@/components/common/PillTabs";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

// TODO(migrate): theme tokens that ECharts options need as raw hex.
// Keep in sync with ParetoChart / RagoSweepChart.
const TEXT_PRIMARY = "#111827";
const TEXT_SECONDARY = "#4B5563";
const TEXT_TERTIARY = "#6B7280";
const BORDER_SUBTLE = "#EEF1F6";

interface StageScalingChartProps {
  stages: StageInfo[];
  qpsBound: string[];
  getStageSweep: (stage: string) => SweepRow[] | undefined;
  /** End-to-end rate of the selected deployment — the dashed reference line
   *  every throughput curve is judged against. */
  e2eQps?: number;
  initialMode?: Mode;
  initialSingleStage?: string;
}

type Mode = "line" | "single";

/** One sweep row converted to the throughput domain. */
interface ThroughputPt {
  batch: number;
  chips: number;
  reqPerS: number;
  /** Wall time to finish the whole batch (dec_steps × time_ms). */
  batchMs: number;
  tp?: number;
  pp?: number;
}

function toPoints(rows: SweepRow[]): ThroughputPt[] {
  return rows
    .filter((r) => r.batch_size > 0 && r.time_ms > 0)
    .map((r) => {
      const steps = r.dec_steps && r.dec_steps > 0 ? r.dec_steps : 1;
      const batchMs = r.time_ms * steps;
      return {
        batch: r.batch_size,
        chips: r.num_chips,
        reqPerS: (r.batch_size / batchMs) * 1000,
        batchMs,
        tp: r.winning_tp,
        pp: r.winning_pp,
      };
    })
    .sort((a, b) => a.batch - b.batch);
}

/** Rows at the stage's current chip count (closest swept value otherwise). */
function rowsAtCurrentChips(rows: SweepRow[], stage: StageInfo): SweepRow[] {
  const chipValues = Array.from(
    new Set(rows.map((r) => r.num_chips).filter(Number.isFinite)),
  );
  if (chipValues.length === 0) return rows;
  if (!Number.isFinite(stage.num_chips) || stage.num_chips <= 0) {
    return rows.filter((r) => r.num_chips === Math.min(...chipValues));
  }
  const closest = chipValues.reduce((best, value) =>
    Math.abs(value - stage.num_chips) < Math.abs(best - stage.num_chips)
      ? value
      : best,
  );
  return rows.filter((r) => r.num_chips === closest);
}

/** Point on the curve nearest the stage's current batch — the marker must sit
 *  ON the line even when the exact batch wasn't swept. */
function nearestPt(pts: ThroughputPt[], batch: number): ThroughputPt | null {
  if (pts.length === 0) return null;
  if (!Number.isFinite(batch) || batch <= 0) return null;
  return pts.reduce((acc, p) =>
    Math.abs(p.batch - batch) < Math.abs(acc.batch - batch) ? p : acc,
  );
}

/**
 * Sweep rows are PER-CALL rates; the assembly's qps_<stage> is PER-REQUEST
 * (react-style dataflows call a stage several times per request, and
 * collocation sharing shifts the effective rate). Anchor the curve to the
 * deployment: scale so the operating-cell value equals the stage's real
 * per-request rate. Returns 1 when there's nothing to anchor to.
 */
function calibrationFactor(pts: ThroughputPt[], stage: StageInfo): number {
  if (!(stage.qps > 0)) return 1;
  const anchor = nearestPt(pts, stage.batch_size);
  if (!anchor || !(anchor.reqPerS > 0)) return 1;
  const f = stage.qps / anchor.reqPerS;
  return Number.isFinite(f) && f > 0 ? f : 1;
}

function scalePoints(pts: ThroughputPt[], f: number): ThroughputPt[] {
  if (f === 1) return pts;
  return pts.map((p) => ({ ...p, reqPerS: p.reqPerS * f }));
}

/** Human line for the folded-in calibration, when it is material. */
function calibrationNote(f: number): string | null {
  if (f > 0 && f < 0.95) {
    return `≈${(1 / f).toFixed(1)} calls per request folded in`;
  }
  if (f > 1.05) {
    return `×${f.toFixed(2)} pipelining/sharing gain folded in`;
  }
  return null;
}

/** Chip-count unit word — retrieval scales in servers, everything else chips. */
function chipUnit(stageName: string, n: number): string {
  const word = stageName === "semantic_retrieval_vectorsearch" ? "server" : "chip";
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/** Blend a hex color toward white: t=0 → 65% tint, t=1 → full color. */
function shadeOf(hex: string, t: number): string {
  const n = parseInt(hex.slice(1), 16);
  const mix = 0.65 * (1 - t);
  const ch = (shift: number) => {
    const v = (n >> shift) & 0xff;
    return Math.round(v + (255 - v) * mix);
  };
  return `#${((ch(16) << 16) | (ch(8) << 8) | ch(0)).toString(16).padStart(6, "0")}`;
}

const tooltipStyle = {
  backgroundColor: "rgba(17,24,39,0.92)",
  borderColor: "transparent",
  textStyle: { color: "#F9FAFB", fontSize: 11 },
  padding: [6, 10] as [number, number],
};

function throughputTooltipLines(
  stageName: string,
  pt: ThroughputPt,
  calibration: number,
): string[] {
  const isDecode = stageName.endsWith("_decode");
  const lines = [
    `<b>${stageName}</b>`,
    `${isDecode ? "Decode buffer" : "Batch"}: ${pt.batch} · ${chipUnit(stageName, pt.chips)}`,
    `Throughput: <b>${formatRate(pt.reqPerS)} req/s</b>`,
    `Batch wall time: ${formatMs(pt.batchMs)} per call`,
  ];
  if (pt.tp != null && pt.pp != null)
    lines.push(`CM picked TP${pt.tp}·PP${pt.pp}`);
  const note = calibrationNote(calibration);
  if (note) lines.push(`<span style="color:#9CA3AF">${note}</span>`);
  return lines;
}

/** Dashed "pipeline today" reference line, attached to a silent series. */
function e2eReferenceSeries(e2eQps: number) {
  if (!(e2eQps > 0)) return [];
  return [
    {
      name: "_e2e_ref",
      type: "line" as const,
      data: [],
      silent: true,
      showSymbol: false,
      legendHoverLink: false,
      tooltip: { show: false },
      markLine: {
        silent: true,
        symbol: ["none", "none"],
        lineStyle: { color: TEXT_SECONDARY, type: "dashed", width: 1.2 },
        label: {
          show: true,
          position: "insideEndTop" as const,
          formatter: `pipeline today: ${formatRate(e2eQps)} req/s`,
          fontSize: 10,
          color: TEXT_SECONDARY,
        },
        data: [{ yAxis: e2eQps }],
      },
      z: 1,
    },
  ];
}

const LOG_AXES = (yName: string): Pick<EChartsOption, "xAxis" | "yAxis"> => ({
  xAxis: {
    type: "log",
    name: "batch size (log)",
    nameGap: 22,
    nameTextStyle: { fontSize: 10, color: TEXT_TERTIARY },
    axisLabel: { fontSize: 10, color: TEXT_TERTIARY },
    axisLine: { lineStyle: { color: BORDER_SUBTLE } },
    splitLine: { lineStyle: { color: BORDER_SUBTLE, type: "dashed" } },
  },
  yAxis: {
    type: "log",
    name: yName,
    nameGap: 40,
    nameLocation: "middle",
    nameTextStyle: { fontSize: 10, color: TEXT_TERTIARY },
    axisLabel: { fontSize: 10, color: TEXT_TERTIARY, formatter: (v: number) => formatRate(v) },
    axisLine: { lineStyle: { color: BORDER_SUBTLE } },
    splitLine: { lineStyle: { color: BORDER_SUBTLE, type: "dashed" } },
  },
});

export function StageScalingChart({
  stages,
  qpsBound,
  getStageSweep,
  e2eQps = 0,
  initialMode = "line",
  initialSingleStage,
}: StageScalingChartProps) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const defaultSingleStage =
    initialSingleStage ??
    stages.find((s) => s.is_bottleneck)?.name ??
    stages[0]?.name ??
    "";
  const [singleStage, setSingleStage] = useState(defaultSingleStage);
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  useEffect(() => {
    setHidden(new Set());
  }, [stages]);

  // React to caller-provided overrides (e.g. the Stage-load row jump).
  useEffect(() => {
    setMode(initialMode);
  }, [initialMode]);
  useEffect(() => {
    if (initialSingleStage) setSingleStage(initialSingleStage);
  }, [initialSingleStage]);
  useEffect(() => {
    if (!singleStage || stages.some((s) => s.name === singleStage)) return;
    setSingleStage(
      stages.find((s) => s.is_bottleneck)?.name ?? stages[0]?.name ?? "",
    );
  }, [singleStage, stages]);

  if (stages.length === 0) {
    return (
      <span className="text-xs text-muted-foreground">
        No stage sweep data available.
      </span>
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="flex min-w-0 flex-wrap items-center gap-2">
        <PillTabs
          tabs={[
            { key: "line", label: "Compare stages" },
            { key: "single", label: "One stage in depth" },
          ]}
          active={mode}
          onSelect={(k) => setMode(k as Mode)}
        />
        <span className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">
          {mode === "line"
            ? "Throughput vs batch size, each stage at its current chip count. Dot = this deployment; a curve stuck below the dashed line can't keep up."
            : "One stage, one line per chip count — the gap between lines is what extra chips buy. Dot = this deployment."}
        </span>
      </div>

      {mode === "line" ? (
        <CompareMode
          stages={stages}
          getStageSweep={getStageSweep}
          e2eQps={e2eQps}
          hidden={hidden}
          setHidden={setHidden}
        />
      ) : (
        <SingleStageMode
          stages={stages}
          selected={singleStage}
          onSelect={setSingleStage}
          getStageSweep={getStageSweep}
          e2eQps={e2eQps}
        />
      )}
    </div>
  );
}

// ── Compare stages ───────────────────────────────────────────────────────
function CompareMode({
  stages,
  getStageSweep,
  e2eQps,
  hidden,
  setHidden,
}: {
  stages: StageInfo[];
  getStageSweep: (stage: string) => SweepRow[] | undefined;
  e2eQps: number;
  hidden: Set<string>;
  setHidden: (s: Set<string>) => void;
}) {
  // Trigger fetches for stages we don't have yet; getStageSweep is memoized
  // and re-renders us via internal setSweepTick when a fetch resolves.
  const stageRows = stages.map((s) => ({ stage: s, rows: getStageSweep(s.name) }));
  const hasPending = stageRows.some(({ rows }) => rows === undefined);

  const { series, ptIndex } = useMemo(() => {
    const out: any[] = [];
    // seriesName:batch → the point + its stage's calibration factor (tooltip).
    const ptIndex = new Map<string, { pt: ThroughputPt; f: number }>();
    for (const { stage, rows } of stageRows) {
      if (!rows || hidden.has(stage.name)) continue;
      const raw = toPoints(rowsAtCurrentChips(rows, stage));
      if (raw.length === 0) continue;
      const f = calibrationFactor(raw, stage);
      const pts = scalePoints(raw, f);
      for (const p of pts) ptIndex.set(`${stage.name}:${p.batch}`, { pt: p, f });
      const color = stageColor(stage.name);
      const marker = nearestPt(pts, stage.batch_size);
      out.push({
        name: stage.name,
        type: "line",
        showSymbol: true,
        symbolSize: 5,
        lineStyle: { width: stage.is_bottleneck ? 2.5 : 1.5, color },
        itemStyle: { color },
        data: pts.map((p) => [p.batch, p.reqPerS]),
        markPoint: marker
          ? {
              symbol: "circle",
              symbolSize: stage.is_bottleneck ? 14 : 9,
              itemStyle: { color, borderColor: "#fff", borderWidth: 1.5 },
              label: { show: false },
              tooltip: { show: false },
              data: [{ coord: [marker.batch, marker.reqPerS] }],
            }
          : undefined,
        z: 5,
      });
    }
    out.push(...e2eReferenceSeries(e2eQps));
    return { series: out, ptIndex };
  }, [stageRows, hidden, e2eQps]);

  const option: EChartsOption = useMemo(
    () => ({
      tooltip: {
        trigger: "item",
        ...tooltipStyle,
        formatter: (p: any) => {
          const hit = ptIndex.get(`${p.seriesName}:${p.data?.[0]}`);
          if (!hit) return "";
          return throughputTooltipLines(p.seriesName, hit.pt, hit.f).join("<br/>");
        },
      },
      grid: { left: 60, right: 24, top: 24, bottom: 42 },
      ...LOG_AXES("stage throughput, req/s per request (log)"),
      series,
    }),
    [series, ptIndex],
  );

  const drawable = series.some((s) => s.name !== "_e2e_ref");
  if (!drawable) {
    if (hasPending) return <ChartStatus title="Loading stage sweep data" loading />;
    if (hidden.size >= stages.length) return <ChartStatus title="All stages hidden" />;
    return <ChartStatus title="No per-stage sweep data for this evaluation" />;
  }

  return (
    <div className="flex min-w-0 flex-col gap-2.5">
      <div className="min-w-0 rounded-md border bg-card">
        <EChartsView option={option} height={320} notMerge />
      </div>
      <div className="flex min-w-0 flex-wrap gap-1.5">
        {stages.map((s) => {
          const isHidden = hidden.has(s.name);
          const hasData = (getStageSweep(s.name)?.length ?? 0) > 0;
          return (
            <Button
              key={s.name}
              type="button"
              variant="outline"
              size="xs"
              onClick={() => {
                const next = new Set(hidden);
                if (next.has(s.name)) next.delete(s.name);
                else next.add(s.name);
                setHidden(next);
              }}
              aria-pressed={!isHidden}
              className={cn(
                "h-7 max-w-[18rem] justify-start overflow-hidden px-2 text-xs",
                isHidden ? "opacity-45" : "opacity-100",
              )}
              title={
                hasData
                  ? `runs at ${chipUnit(s.name, s.num_chips)}, batch ${s.batch_size} in this deployment — click to toggle`
                  : "no sweep CSV for this stage"
              }
            >
              <span
                className="size-2 shrink-0 rounded"
                style={{ backgroundColor: stageColor(s.name) }}
              />
              <span className="min-w-0 truncate font-mono" style={{ color: TEXT_PRIMARY }}>
                {s.name}
              </span>
              <span className="shrink-0 font-mono" style={{ color: TEXT_TERTIARY }}>
                {hasData ? `${s.num_chips}c·b${s.batch_size}` : "(no sweep)"}
              </span>
            </Button>
          );
        })}
      </div>
    </div>
  );
}

// ── One stage in depth — a line per chip count ───────────────────────────
function SingleStageMode({
  stages,
  selected,
  onSelect,
  getStageSweep,
  e2eQps,
}: {
  stages: StageInfo[];
  selected: string;
  onSelect: (s: string) => void;
  getStageSweep: (stage: string) => SweepRow[] | undefined;
  e2eQps: number;
}) {
  const rows = getStageSweep(selected);
  const stage = stages.find((s) => s.name === selected);

  const { series, ptIndex, calibration } = useMemo(() => {
    const ptIndex = new Map<string, ThroughputPt>();
    if (!rows || !stage)
      return { series: [] as any[], ptIndex, calibration: 1 };
    // One calibration factor for the whole stage (call multiplicity is a
    // property of the dataflow, not of the chip count), anchored on the
    // stage's current-chips line.
    const f = calibrationFactor(toPoints(rowsAtCurrentChips(rows, stage)), stage);
    const pts = scalePoints(toPoints(rows), f);
    const chipValues = Array.from(new Set(pts.map((p) => p.chips))).sort(
      (a, b) => a - b,
    );
    const base = stageColor(stage.name);
    const out: any[] = [];
    chipValues.forEach((chips, i) => {
      const line = pts.filter((p) => p.chips === chips);
      if (line.length === 0) return;
      const color = shadeOf(base, chipValues.length === 1 ? 1 : i / (chipValues.length - 1));
      const name = chipUnit(stage.name, chips);
      for (const p of line) ptIndex.set(`${name}:${p.batch}`, p);
      const isCurrent = chips === stage.num_chips;
      const marker = isCurrent ? nearestPt(line, stage.batch_size) : null;
      out.push({
        name,
        type: "line",
        showSymbol: true,
        symbolSize: 5,
        lineStyle: { width: isCurrent ? 2.5 : 1.5, color },
        itemStyle: { color },
        endLabel: {
          show: true,
          formatter: name,
          fontSize: 10,
          color,
          distance: 6,
        },
        data: line.map((p) => [p.batch, p.reqPerS]),
        markPoint: marker
          ? {
              symbol: "circle",
              symbolSize: 14,
              itemStyle: { color, borderColor: "#fff", borderWidth: 1.5 },
              label: { show: false },
              tooltip: { show: false },
              data: [{ coord: [marker.batch, marker.reqPerS] }],
            }
          : undefined,
        z: 5,
      });
    });
    out.push(...e2eReferenceSeries(e2eQps));
    return { series: out, ptIndex, calibration: f };
  }, [rows, stage, e2eQps]);

  const option: EChartsOption = useMemo(
    () => ({
      tooltip: {
        trigger: "item",
        ...tooltipStyle,
        formatter: (p: any) => {
          const pt = ptIndex.get(`${p.seriesName}:${p.data?.[0]}`);
          if (!pt || !stage) return "";
          return throughputTooltipLines(stage.name, pt, calibration).join("<br/>");
        },
      },
      grid: { left: 60, right: 76, top: 24, bottom: 42 },
      ...LOG_AXES("stage throughput, req/s per request (log)"),
      series,
    }),
    [series, ptIndex, stage, calibration],
  );

  return (
    <div className="flex min-w-0 flex-col gap-2.5">
      <div className="flex min-w-0 flex-wrap gap-1.5 rounded-md border bg-muted/20 p-1.5">
        {stages.map((s) => (
          <Button
            key={s.name}
            type="button"
            variant={s.name === selected ? "secondary" : "ghost"}
            size="xs"
            onClick={() => onSelect(s.name)}
            className="h-7 max-w-[16rem] justify-start overflow-hidden px-2 text-xs"
          >
            <span
              className="size-2 shrink-0 rounded"
              style={{ backgroundColor: stageColor(s.name) }}
            />
            <span className="min-w-0 truncate font-mono">{s.name}</span>
            {s.is_bottleneck && (
              <span className="shrink-0 text-[9px] font-semibold uppercase text-primary">
                bound
              </span>
            )}
          </Button>
        ))}
      </div>
      {rows === undefined ? (
        <ChartStatus title="Loading stage sweep data" loading />
      ) : series.filter((s: any) => s.name !== "_e2e_ref").length === 0 ? (
        <ChartStatus title="No sweep data for this stage" />
      ) : (
        <div className="min-w-0 rounded-md border bg-card">
          <EChartsView option={option} height={320} notMerge />
        </div>
      )}
    </div>
  );
}

function ChartStatus({ title, loading }: { title: string; loading?: boolean }) {
  return (
    <div className="flex h-[320px] min-w-0 flex-col items-center justify-center gap-2 rounded-md border bg-muted/20 px-4 text-center">
      {loading && <Loader2 className="size-4 animate-spin text-muted-foreground" />}
      <span className="text-sm text-muted-foreground">{title}</span>
    </div>
  );
}

function formatRate(v: number) {
  if (!Number.isFinite(v)) return "—";
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e4) return `${(v / 1e3).toFixed(0)}k`;
  if (v >= 100) return v.toFixed(0);
  if (v >= 1) return v.toFixed(1);
  return v.toFixed(2);
}

function formatMs(ms: number) {
  if (!Number.isFinite(ms)) return "—";
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  if (ms >= 1) return `${ms.toFixed(1)}ms`;
  return `${(ms * 1000).toFixed(0)}µs`;
}
