import React from "react";
import { Settings, ChevronRight } from "lucide-react";
import { stageColor } from "@/lib/theme";
import type { PipelineNode, RagoSweepPoint } from "@/types";
import { Badge, BadgeText } from "@/components/ui/LegacyBadge";
import { Tooltip } from "@/components/ui/LegacyTooltip";
import { useStageInteraction } from "@/context/StageInteractionContext";
import { cn } from "@/lib/utils";

// from old theme.ts
const ACCENT = "#0F766E";

interface PipelineDiagramProps {
  pipeline: PipelineNode[];
  bottleneckStages: string[];
  /** When provided, show the per-stage hardware values for THIS sweep point. */
  point?: RagoSweepPoint | null;
  /** Parsed `point.collocation_strategy` — list of stage groups sharing hardware. */
  collocationGroups?: string[][];
  /** Drop the static `key=value` display rows (used in the cascade). */
  compact?: boolean;
}

export function PipelineDiagram({
  pipeline,
  bottleneckStages,
  point,
  collocationGroups,
  compact,
}: PipelineDiagramProps) {
  if (pipeline.length === 0) {
    return (
      <span className="text-xs text-muted-foreground">
        No pipeline configuration available.
      </span>
    );
  }
  const bound = new Set(bottleneckStages);
  const grouping = computeNodeGrouping(pipeline, collocationGroups);

  return (
    <div className="flex flex-wrap items-stretch gap-x-1.5 gap-y-3">
      {grouping.map((entry, gi) => (
        <React.Fragment key={`group-${gi}`}>
          {entry.kind === "single" ? (
            <NodeCard node={entry.node} bound={bound} point={point} compact={compact} />
          ) : (
            <CollocationGroup nodes={entry.nodes} bound={bound} point={point} compact={compact} sharedChips={entry.sharedChips} />
          )}
          {gi < grouping.length - 1 && <Arrow />}
        </React.Fragment>
      ))}
    </div>
  );
}

// ── Collocation grouping ────────────────────────────────────────────────
type GroupingEntry =
  | { kind: "single"; node: PipelineNode }
  | { kind: "collocated"; nodes: PipelineNode[]; sharedChips: number | null };

function computeNodeGrouping(
  pipeline: PipelineNode[],
  collocationGroups?: string[][],
): GroupingEntry[] {
  if (!collocationGroups || collocationGroups.length === 0) {
    return pipeline.map((node) => ({ kind: "single", node }));
  }
  // Each pipeline node carries its `stages: string[]`. Build a map
  // stage → node so we can locate node owners of each collocation group.
  const stageToNode = new Map<string, PipelineNode>();
  for (const node of pipeline) {
    for (const s of node.stages) stageToNode.set(s, node);
  }
  const out: GroupingEntry[] = [];
  const consumed = new Set<PipelineNode>();
  // Walk pipeline in order; for each node, find the collocation group containing
  // any of its stages. If that group spans multiple distinct nodes, render as a
  // collocated cluster (consuming all of them at this position).
  for (const node of pipeline) {
    if (consumed.has(node)) continue;
    if (node.stages.length === 0) {
      out.push({ kind: "single", node });
      consumed.add(node);
      continue;
    }
    const group = collocationGroups.find((g) =>
      g.some((s) => node.stages.includes(s)),
    );
    if (!group) {
      out.push({ kind: "single", node });
      consumed.add(node);
      continue;
    }
    const groupNodes = Array.from(
      new Set(group.map((s) => stageToNode.get(s)).filter(Boolean) as PipelineNode[]),
    );
    if (groupNodes.length <= 1) {
      out.push({ kind: "single", node });
      consumed.add(node);
      continue;
    }
    out.push({ kind: "collocated", nodes: groupNodes, sharedChips: null });
    for (const n of groupNodes) consumed.add(n);
  }
  return out;
}

function CollocationGroup({
  nodes,
  bound,
  point,
  compact,
  sharedChips,
}: {
  nodes: PipelineNode[];
  bound: Set<string>;
  point?: RagoSweepPoint | null;
  compact?: boolean;
  sharedChips: number | null;
}) {
  return (
    <div className="flex flex-col gap-1.5 rounded-xl border border-dashed border-border bg-card p-2">
      <span className="text-[9px] font-semibold uppercase tracking-[0.8px] text-muted-foreground">
        collocated
      </span>
      <div className="flex flex-wrap items-stretch gap-x-1 gap-y-1.5">
        {nodes.map((node, i) => (
          <React.Fragment key={`${node.stage}-${i}`}>
            <NodeCard node={node} bound={bound} point={point} compact={compact} />
            {i < nodes.length - 1 && <SmallArrow />}
          </React.Fragment>
        ))}
      </div>
      {sharedChips != null && (
        <span className="text-right font-mono text-[10px] text-muted-foreground">
          shared {sharedChips} chips
        </span>
      )}
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex items-center justify-center px-0.5">
      <ChevronRight className="size-3.5 text-muted-foreground" />
    </div>
  );
}

function SmallArrow() {
  return (
    <div className="flex items-center justify-center px-px">
      <ChevronRight className="size-3 text-muted-foreground/60" />
    </div>
  );
}

// ── Node card ───────────────────────────────────────────────────────────
function NodeCard({
  node,
  bound,
  point,
  compact,
}: {
  node: PipelineNode;
  bound: Set<string>;
  point?: RagoSweepPoint | null;
  compact?: boolean;
}) {
  const isBottleneck = node.stages.some((s) => bound.has(s));
  const { hoveredStage, setHoveredStage } = useStageInteraction();
  const isHovered =
    hoveredStage != null && node.stages.includes(hoveredStage);

  const onMouseEnter = () => {
    if (node.stages.length > 0) setHoveredStage(node.stages[0]);
  };
  const onMouseLeave = () => setHoveredStage(null);

  return (
    <div
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      className={cn(
        "relative flex shrink-0 flex-col rounded-[10px] bg-muted",
        compact ? "min-w-[130px] gap-1 p-2.5" : "min-w-[140px] gap-1.5 p-3",
        isBottleneck
          ? "border-[1.5px] border-primary"
          : isHovered
            ? "border border-primary/40"
            : "border border-border",
      )}
    >
      {node.override !== undefined && (
        <div className="absolute right-1.5 top-1.5 z-[2]">
          <OverrideBadge override={node.override} />
        </div>
      )}

      <span
        className={cn(
          "text-[13px] font-semibold text-foreground",
          node.override !== undefined && "pr-4",
        )}
      >
        {node.stage}
      </span>
      <span className="text-[10px] uppercase tracking-[0.5px] text-muted-foreground">
        {node.component}
      </span>

      {!compact && node.display.length > 0 && (
        <div className="mt-0.5 flex flex-col gap-0.5">
          {node.display.slice(0, 3).map((d) => (
            <span
              key={d.label}
              className="truncate font-mono text-[11px] text-muted-foreground"
            >
              <span className="font-mono text-[11px] text-muted-foreground/70">
                {d.label}=
              </span>
              {d.value}
            </span>
          ))}
        </div>
      )}

      {point && <PointStatsLines node={node} point={point} />}

      {node.stages.length > 0 && (
        <div className="mt-1 flex flex-wrap items-center gap-1">
          {node.stages.map((stage) => (
            <StagePill key={stage} stage={stage} bound={bound.has(stage)} />
          ))}
        </div>
      )}

      {isBottleneck && !compact && (
        <div className="mt-0.5 flex">
          <Badge variant="accent">
            <BadgeText variant="accent">BOUND</BadgeText>
          </Badge>
        </div>
      )}

      {/* Spine anchor: a small ACCENT bottom dot the parent uses to align the spine line. */}
      {isBottleneck && (
        <div
          className="absolute -bottom-[3px] left-1/2 size-1.5 -ml-[3px] rounded-full"
          style={{ backgroundColor: ACCENT }}
        />
      )}
    </div>
  );
}

// Per-node hardware values pulled from the selected sweep point.
function PointStatsLines({ node, point }: { node: PipelineNode; point: RagoSweepPoint }) {
  const lines = pointStatLines(node, point);
  if (lines.length === 0) return null;
  return (
    <div className="mt-0.5 flex flex-col gap-0.5">
      {lines.map((l, i) => (
        <span key={i} className="truncate font-mono text-[11px] text-muted-foreground">
          {l}
        </span>
      ))}
    </div>
  );
}

function pointStatLines(node: PipelineNode, point: RagoSweepPoint): string[] {
  const get = (k: string) => {
    const v = (point as any)[k];
    return v != null && v !== "" ? Number(v) : null;
  };
  // Generator is special — it has both prefill & decode stages so we render two lines.
  if (node.stages.includes("generator_prefill") && node.stages.includes("generator_decode")) {
    const chips = get("num_chips_generator_prefill") ?? get("num_chips_generator_decode");
    const bsPre = get("batch_size_generator_prefill");
    const bsDec = get("batch_size_decode");
    const out: string[] = [];
    if (chips != null) out.push(`${chips} chips`);
    if (bsPre != null && bsDec != null) out.push(`pre ${bsPre} · dec ${bsDec}`);
    return out;
  }
  // Retrieval uses num_servers, not num_chips.
  if (node.stages.includes("semantic_retrieval_vectorsearch")) {
    const servers = get("num_retrieval_servers");
    const bs = get("batch_size_semantic_retrieval_vectorsearch");
    const out: string[] = [];
    if (servers != null) out.push(`${servers} servers`);
    if (bs != null) out.push(`batch ${bs}`);
    return out;
  }
  // Single-stage non-LLM nodes: passage_reranker, passage_filter, query_expansion, passage_compressor.
  const singleStage = node.stages[0];
  if (!singleStage) return [];
  const chips = get(`num_chips_${singleStage}`);
  const bs = get(`batch_size_${singleStage}`);
  const out: string[] = [];
  if (chips != null) out.push(`${chips} chips`);
  if (bs != null) out.push(`batch ${bs}`);
  return out;
}

function StagePill({ stage, bound }: { stage: string; bound: boolean }) {
  const color = stageColor(stage);
  return (
    <div
      className={cn(
        "flex items-center gap-1 rounded px-1.5 py-0.5",
        bound ? "border border-primary bg-primary/10" : "border-0",
      )}
      style={
        bound
          ? undefined
          : { backgroundColor: `${color}14` }
      }
    >
      <div
        className="size-1.5 rounded-full"
        style={{ backgroundColor: bound ? ACCENT : color }}
      />
      <span
        className={cn(
          "font-mono text-[10px]",
          bound ? "text-primary" : "text-muted-foreground",
        )}
      >
        {stage}
      </span>
    </div>
  );
}

function OverrideBadge({
  override,
}: {
  override: number | Record<string, number>;
}) {
  return (
    <Tooltip content={<OverrideTooltipBody override={override} />} placement="top" delay={80}>
      <div className="flex size-[18px] cursor-help items-center justify-center rounded bg-primary/10">
        <Settings className="size-[11px] text-primary" />
      </div>
    </Tooltip>
  );
}

function OverrideTooltipBody({
  override,
}: {
  override: number | Record<string, number>;
}) {
  return (
    <div className="flex min-w-[140px] flex-col gap-1">
      <span className="text-[9px] font-semibold uppercase tracking-[0.8px] text-muted-foreground">
        max_batch_size override
      </span>
      {typeof override === "number" ? (
        <span className="font-mono text-xs text-foreground">
          {override}
        </span>
      ) : (
        <div className="flex flex-col gap-0.5">
          {Object.entries(override).map(([stage, val]) => (
            <div key={stage} className="flex justify-between gap-3">
              <span className="font-mono text-[11px] text-muted-foreground">
                {stage}
              </span>
              <span className="font-mono text-[11px] text-foreground">
                {val}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Helpers exposed for the parent (PointCard / DashboardContent) ──────

/** Parse `assembly.collocation_strategy` ("[['retrieval'], ['rerank']]") into string[][]. */
export function parseCollocationGroups(raw: unknown): string[][] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw as string[][];
  if (typeof raw !== "string") return [];
  // Python repr uses single quotes; convert to JSON.
  const jsonish = raw.replace(/'/g, '"');
  try {
    const parsed = JSON.parse(jsonish);
    if (Array.isArray(parsed)) return parsed as string[][];
  } catch {
    // ignore
  }
  return [];
}
