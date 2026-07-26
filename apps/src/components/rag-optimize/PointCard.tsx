import React from "react";
import type { PipelineNode, RagoSweepPoint } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PipelineDiagram, parseCollocationGroups } from "./PipelineDiagram";

interface PointCardProps {
  point: RagoSweepPoint;
  pipeline: PipelineNode[];
  bottleneckStages: string[];
  isPolicyDefault: boolean;
  /** Position of this deployment in the sweep (for the provenance line). */
  pointIdx: number;
  pointTotal: number;
  /** Active performance-policy name, e.g. "min_latency". */
  policySelection: string;
}

/**
 * Deployment layout — WHICH hardware assignment the numbers below describe,
 * and where it came from. The diagram shows the pipeline placed on hardware:
 * per node chips/servers + batch, collocation groups, and the bound stage.
 */
export function PointCard({
  point,
  pipeline,
  bottleneckStages,
  isPolicyDefault,
  pointIdx,
  pointTotal,
  policySelection,
}: PointCardProps) {
  const collocationGroups = parseCollocationGroups(point.collocation_strategy);
  const placement = String(point.placement_policy ?? "—");
  const numChips = Number((point as any).num_chips_e2e ?? 0);
  const scheduling = String(point.scheduling_policy ?? "—");

  return (
    <Card className="rounded-lg py-0">
      <CardContent className="space-y-3 p-4">
        <div className="min-w-0">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold text-foreground">
              Deployment layout
            </h3>
            <Badge
              variant={isPolicyDefault ? "secondary" : "outline"}
              className="rounded-md"
              title={
                isPolicyDefault
                  ? `Best deployment under the "${policySelection}" performance policy`
                  : "You picked this point by clicking the sweep chart above"
              }
            >
              {isPolicyDefault ? "policy pick" : "manual pick"}
            </Badge>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            The cost model swept {pointTotal > 0 ? pointTotal : "many"} ways to
            lay this pipeline on hardware; this is
            {pointIdx >= 0 ? ` #${pointIdx + 1} of them` : " one of them"} —{" "}
            {isPolicyDefault
              ? `the best under the “${policySelection}” policy`
              : "the one you clicked in the sweep chart"}
            . Everything below describes THIS deployment; click another sweep
            point to switch.
          </p>
        </div>

        <div className="grid min-w-0 grid-cols-1 gap-2 text-xs text-muted-foreground sm:grid-cols-3">
          <Meta
            label="Placement"
            value={placement}
            title="How stages are assigned to GPUs: collocated = stages share GPUs, disaggregated = each stage gets its own"
          />
          <Meta label="Scheduling" value={scheduling} />
          <Meta label="Total chips" value={String(numChips)} />
        </div>

        <div className="min-w-0 rounded-md border bg-muted/20 p-3">
          <PipelineDiagram
            pipeline={pipeline}
            bottleneckStages={bottleneckStages}
            point={point}
            collocationGroups={collocationGroups}
            compact
          />
        </div>
      </CardContent>
    </Card>
  );
}

function Meta({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <div className="min-w-0 rounded-md bg-muted/40 px-2.5 py-2" title={title}>
      <span className="block truncate text-[10px] font-medium uppercase text-muted-foreground">
        {label}
      </span>
      <span className="mt-0.5 block truncate font-mono text-xs text-foreground">
        {value}
      </span>
    </div>
  );
}
