import React from "react";
import { stageColor } from "@/lib/theme";
import { PERF_STAGE_ORDER } from "@/lib/stages";
import type { StageInfo } from "@/hooks/useEvalBreakdown";
import { Badge } from "@/components/ui/badge";
import { useStageInteraction } from "@/context/StageInteractionContext";
import { cn } from "@/lib/utils";

// Canonical left-to-right pipeline order so the bars and waterfall read in
// dataflow order regardless of best_hw_params key ordering.
function orderStages(stages: StageInfo[]): StageInfo[] {
  return [...stages].sort((a, b) => {
    const ia = PERF_STAGE_ORDER.indexOf(a.name);
    const ib = PERF_STAGE_ORDER.indexOf(b.name);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
}

interface StageBreakdownTableProps {
  stages: StageInfo[];
  ttftS: number;
  e2eS: number;
  /** Time spent queued before service — surfaced as its own waterfall segment
   *  instead of being folded into decode. */
  queueWaitS: number;
  /** End-to-end qps of the selected deployment (headline for the headroom
   *  block); bars stay meaningful without it. */
  e2eQps?: number;
}

/**
 * Two answers about the selected deployment, stacked:
 *   1. Throughput headroom — which stage caps the pipeline, who has slack.
 *   2. Latency waterfall — where one request's time actually goes.
 * Deployment shape (chips, batches, placement) lives on the layout card above.
 */
export function StageBreakdownTable({
  stages,
  ttftS,
  e2eS,
  queueWaitS,
  e2eQps = 0,
}: StageBreakdownTableProps) {
  if (stages.length === 0) {
    return (
      <span className="text-xs text-muted-foreground">
        No stage data available.
      </span>
    );
  }
  const ordered = orderStages(stages);
  return (
    <div className="grid min-w-0 grid-cols-1 gap-3">
      <StageHeadroom stages={ordered} e2eQps={e2eQps} />
      <Waterfall
        stages={ordered}
        ttftS={ttftS}
        e2eS={e2eS}
        queueWaitS={queueWaitS}
      />
    </div>
  );
}

// ── Stage load — the "who caps the pipeline, who is idle" view ─────────────
function StageHeadroom({
  stages,
  e2eQps,
}: {
  stages: StageInfo[];
  e2eQps: number;
}) {
  const rated = stages.filter((s) => s.qps > 0);
  // Reference rate for the load bars: the pipeline's real end-to-end qps.
  // Without it (older rows), the slowest stage's rate is the closest proxy.
  const reference =
    e2eQps > 0
      ? e2eQps
      : rated.length > 0
        ? Math.min(...rated.map((s) => s.qps))
        : 0;

  return (
    <div className="rounded-md border bg-muted/20 p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs font-medium text-foreground">Stage load</div>
        {e2eQps > 0 && (
          <span className="font-mono text-[11px] text-muted-foreground">
            pipeline moves {formatNumber(e2eQps)} req/s end-to-end
          </span>
        )}
      </div>
      <div className="mt-0.5 text-[11px] text-muted-foreground">
        Each stage alone could sustain the rate on its right; the bar is its
        load — how hard the pipeline drives it. Hot bars and the{" "}
        <span className="font-medium text-foreground">Bottleneck</span>-tagged
        group (stages that share the binding hardware) cap end-to-end
        throughput; near-empty bars are over-provisioned. Click a stage to
        explore resizing it.
      </div>
      <div className="mt-3 flex flex-col gap-2">
        {stages.map((s) => (
          <HeadroomRow key={s.name} stage={s} reference={reference} />
        ))}
      </div>
    </div>
  );
}

function HeadroomRow({
  stage,
  reference,
}: {
  stage: StageInfo;
  reference: number;
}) {
  const { hoveredStage, setHoveredStage, requestJump } = useStageInteraction();
  const isHovered = hoveredStage === stage.name;
  const bound = stage.is_bottleneck;
  const hasRate = stage.qps > 0;
  // Load can exceed 100% on shared hardware (batch interleaving lets the
  // pipeline drive a stage past its standalone rate) — print the true number,
  // cap only the bar fill.
  const load = hasRate && reference > 0 ? reference / stage.qps : 0;
  const fill = Math.min(1, load);

  // Secondary facts — everything the old 8-column table said, demoted to a
  // single muted line so the bar stays the headline.
  const facts: string[] = [];
  if (stage.svc_s > 0) facts.push(`service ${formatLatency(stage.svc_s)}/req`);
  if (stage.batch_size > 0) facts.push(`batch ${stage.batch_size}`);
  if (stage.tp > 0 || stage.pp > 0)
    facts.push(`TP${stage.tp || 1}·PP${stage.pp || 1}`);
  if (stage.qps_capacity > 0)
    facts.push(`best-batch ceiling ${formatNumber(stage.qps_capacity)} req/s`);

  return (
    <div
      onMouseEnter={() => setHoveredStage(stage.name)}
      onMouseLeave={() => setHoveredStage(null)}
      onClick={() => requestJump(stage.name)}
      title={
        hasRate
          ? `${stage.name} alone could sustain ${formatNumber(stage.qps)} req/s; the pipeline drives it at ${(load * 100).toFixed(0)}% of that. Click to open it in the resize view.`
          : `${stage.name}: no per-stage rate reported for this deployment`
      }
      className={cn(
        "cursor-pointer rounded-md border px-2.5 py-2 transition-colors",
        bound
          ? "border-primary/50 bg-primary/10"
          : isHovered
            ? "border-border bg-muted/60"
            : "border-transparent bg-background/60",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span
          className="size-2.5 shrink-0 rounded-full"
          style={{ backgroundColor: stageColor(stage.name) }}
        />
        <span
          className={cn(
            "min-w-0 truncate font-mono text-xs font-medium text-foreground",
            isHovered && "underline",
          )}
        >
          {stage.name}
        </span>
        {bound && (
          <Badge variant="default" className="shrink-0 rounded-md px-1.5 text-[10px]">
            Bottleneck
          </Badge>
        )}
        <span
          className="ml-auto shrink-0 font-mono text-[11px] tabular-nums text-foreground"
          title="What this stage could sustain on its own at this deployment"
        >
          {hasRate ? `alone: ${formatNumber(stage.qps)} req/s` : "—"}
        </span>
      </div>

      <div className="mt-1.5 flex items-center gap-2">
        <div className="relative h-2 min-w-0 flex-1 overflow-hidden rounded-full bg-muted">
          {hasRate && (
            <span
              className="absolute inset-y-0 left-0 rounded-full"
              style={{
                width: `${fill * 100}%`,
                backgroundColor: stageColor(stage.name),
                opacity: bound ? 1 : 0.75,
              }}
            />
          )}
        </div>
        <span
          className={cn(
            "w-14 shrink-0 text-right font-mono text-[11px] tabular-nums",
            bound ? "font-semibold text-foreground" : "text-muted-foreground",
          )}
          title="Load = end-to-end rate ÷ this stage's standalone rate"
        >
          {hasRate ? `${(load * 100).toFixed(0)}% load` : "n/a"}
        </span>
      </div>

      {facts.length > 0 && (
        <div className="mt-1 truncate text-[10px] text-muted-foreground">
          {facts.join(" · ")}
        </div>
      )}
    </div>
  );
}

function formatNumber(v: number) {
  if (!Number.isFinite(v)) return "—";
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (v >= 1e4) return `${(v / 1e3).toFixed(0)}k`;
  if (v >= 100) return v.toFixed(1);
  if (v >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function formatLatency(s: number) {
  if (!Number.isFinite(s) || s <= 0) return "—";
  if (s >= 1) return `${s.toFixed(2)}s`;
  if (s >= 0.001) return `${(s * 1000).toFixed(1)}ms`;
  return `${(s * 1e6).toFixed(0)}µs`;
}

// ── E2E latency waterfall — honest per-stage service + explicit queue wait ──
function Waterfall({
  stages,
  ttftS,
  e2eS,
  queueWaitS,
}: {
  stages: StageInfo[];
  ttftS: number;
  e2eS: number;
  queueWaitS: number;
}) {
  if (e2eS <= 0) return null;

  const WAIT_COLOR = "var(--muted-foreground)";

  // Real per-stage service time from best_hw_params svc_<stage>_s.
  const segments: Array<{
    name: string;
    seconds: number;
    color: string;
    wait?: boolean;
  }> = stages
    .filter((s) => s.svc_s > 1e-6)
    .map((s) => ({
      name: s.name,
      seconds: s.svc_s,
      color: stageColor(s.name),
    }));

  const svcSum = segments.reduce((acc, s) => acc + s.seconds, 0);
  const queue = Math.max(0, queueWaitS);
  // Anything E2E can't attribute to a stage's service time or the queue wait
  // (e.g. generator overhead, comm) becomes an explicit residual — never
  // silently loaded onto decode.
  const residual = Math.max(0, e2eS - svcSum - queue);

  if (queue > 1e-6) {
    segments.push({ name: "queue wait", seconds: queue, color: WAIT_COLOR, wait: true });
  }
  if (residual > e2eS * 0.01) {
    segments.push({ name: "overhead", seconds: residual, color: WAIT_COLOR, wait: true });
  }

  const total = segments.reduce((acc, s) => acc + s.seconds, 0) || 1;
  const ttftPct = Math.min(100, Math.max(0, (ttftS / e2eS) * 100));

  return (
    <div className="rounded-md border bg-muted/20 p-3">
      <div className="flex min-w-0 flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="text-xs font-medium text-foreground">
            Where a request spends {formatLatency(e2eS)}
          </div>
          <div className="mt-0.5 text-[11px] text-muted-foreground">
            Real service time per stage, with queue wait shown separately.
            The mark is TTFT — everything right of it is decode streaming.
          </div>
        </div>
        <div className="flex shrink-0 gap-3">
          <LabelPair label="TTFT" value={formatLatency(ttftS)} />
          <LabelPair label="Queue" value={formatLatency(queue)} />
          <LabelPair label="E2E" value={formatLatency(e2eS)} />
        </div>
      </div>

      <div className="relative mt-3 h-8">
        <div className="flex h-6 overflow-hidden rounded-md border bg-background">
          {segments.map((seg, i) => (
            <div
              key={`${seg.name}-${i}`}
              title={`${seg.name}: ${formatLatency(seg.seconds)}`}
              style={{
                flex: seg.seconds,
                backgroundColor: seg.color,
                opacity: seg.wait ? 0.35 : 0.9,
              }}
              className={cn(
                "min-w-px",
                i < segments.length - 1 && "border-r border-r-white/40",
              )}
            />
          ))}
        </div>
        {ttftPct > 0 && ttftPct < 100 && (
          <div
            className="absolute -top-0.5 bottom-1 w-px bg-foreground/60"
            style={{ left: `${ttftPct}%` }}
            title={`TTFT ${formatLatency(ttftS)}`}
          />
        )}
      </div>
      <div className="mt-2 grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
        {segments.map((seg) => (
          <div
            key={seg.name}
            className="flex min-w-0 items-center gap-2 text-[11px]"
          >
            <span
              className="size-2 shrink-0 rounded-sm"
              style={{ backgroundColor: seg.color, opacity: seg.wait ? 0.5 : 1 }}
            />
            <span className="min-w-0 flex-1 truncate font-mono text-muted-foreground">
              {seg.name}
            </span>
            <span className="shrink-0 font-mono tabular-nums text-foreground">
              {formatLatency(seg.seconds)}
            </span>
            <span className="w-9 shrink-0 text-right font-mono tabular-nums text-muted-foreground">
              {((seg.seconds / total) * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function LabelPair({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-1">
      <span className="text-[9px] font-semibold uppercase tracking-[0.8px] text-muted-foreground">
        {label}
      </span>
      <span className="font-mono text-[11px] text-foreground">{value}</span>
    </div>
  );
}
