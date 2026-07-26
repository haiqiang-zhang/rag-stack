import { useCallback, useEffect, useMemo, useState } from "react";
import { MousePointerClick } from "lucide-react";

import { useRagoSweep } from "@/hooks/useRagoSweep";
import { useEvalBreakdown } from "@/hooks/useEvalBreakdown";
import { useSelection } from "@/context/SelectionContext";
import { usePerformancePolicy } from "@/context/PerformancePolicyContext";
import { useEvalData } from "@/context/EvalDataContext";
import { StageInteractionProvider } from "@/context/StageInteractionContext";
import { applyPerformanceSelection } from "@/utils/performanceSelection";

import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PointCard } from "./PointCard";
import { StageBreakdownTable } from "./StageBreakdownTable";
import { StageScalingCard } from "./StageScalingCard";
import { SweepCard } from "./SweepCard";
import { CompactSummaryBar } from "./CompactSummaryBar";
import type { RagoSweepPoint } from "@/types";

/**
 * Selected-eval deployment deep-dive: the operating point, the hardware sweep
 * (kept chart), the per-stage breakdown (throughput / service / ceiling / TP·PP
 * + honest latency waterfall + quality by stage), and per-stage scaling.
 * Drilling the sweep chart overrides the operating point.
 */
export function PerformanceTab() {
  const { selectedEvalId } = useSelection();
  const { selectedEval, getDisplayPerformance } = useEvalData();
  const perfPolicy = usePerformancePolicy();
  const { sweepPoints, assembly } = useRagoSweep(selectedEvalId);

  const recomputed = useMemo(
    () => applyPerformanceSelection(sweepPoints, perfPolicy),
    [sweepPoints, perfPolicy],
  );
  const [selectedSweepPoint, setSelectedSweepPoint] =
    useState<RagoSweepPoint | null>(null);
  useEffect(() => {
    setSelectedSweepPoint(recomputed.row);
  }, [selectedEvalId, recomputed.row]);

  const isPolicyDefault = selectedSweepPoint === recomputed.row;
  const pointIdx = selectedSweepPoint
    ? sweepPoints.findIndex((pt) => pt === selectedSweepPoint)
    : -1;
  const breakdown = useEvalBreakdown(selectedEvalId, selectedSweepPoint);

  const handleSweepPointClick = useCallback(
    (pt: RagoSweepPoint) => setSelectedSweepPoint(pt),
    [],
  );

  // Policy default → the live policy getter (displayScore keeps constrained-
  // infeasible evals at their would-be value instead of the -1e9 sentinel).
  // Manual point → that point's raw qps.
  const effectivePerformance = isPolicyDefault
    ? selectedEval
      ? getDisplayPerformance(selectedEval)
      : 0
    : Number((selectedSweepPoint as any)?.qps ?? 0);

  if (selectedEvalId == null) {
    return (
      <div className="flex flex-1 min-h-0 items-center justify-center p-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <MousePointerClick className="size-7 text-muted-foreground/50" />
          <span className="max-w-xs text-[13px] text-muted-foreground">
            Pick an evaluation in the left navigator — its deployment, stage
            breakdown, and scaling appear here.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 min-w-0 overflow-auto p-6 pb-12">
      {selectedEval && (
        <div className="mb-4">
          <CompactSummaryBar
            evalId={selectedEval.eval_id}
            performance={effectivePerformance}
            performanceLabel={
              isPolicyDefault ? `PERF · ${perfPolicy.selection}` : "PERF · this point"
            }
            qpsBound={breakdown.qps_bound}
            pointPlacement={
              selectedSweepPoint
                ? String((selectedSweepPoint as any).placement_policy ?? "")
                : undefined
            }
            pointChips={
              selectedSweepPoint
                ? Number((selectedSweepPoint as any).num_chips_e2e ?? 0)
                : undefined
            }
            pointIdx={pointIdx}
            pointTotal={sweepPoints.length}
            isPolicyDefault={isPolicyDefault}
            onExpand={() => setSelectedSweepPoint(recomputed.row)}
          />
        </div>
      )}

      <SweepCard
        evalId={selectedEvalId}
        sweepPoints={sweepPoints}
        assembly={assembly}
        selectedSweepRow={selectedSweepPoint}
        selectedBestRow={recomputed.row}
        onPointClick={handleSweepPointClick}
        height={320}
      />

      <StageInteractionProvider>
        <div className="mt-4 flex flex-col gap-3.5">
          {selectedSweepPoint && (
            <PointCard
              point={selectedSweepPoint}
              pipeline={breakdown.pipeline}
              bottleneckStages={breakdown.qps_bound}
              isPolicyDefault={isPolicyDefault}
              pointIdx={pointIdx}
              pointTotal={sweepPoints.length}
              policySelection={perfPolicy.selection}
            />
          )}

          {breakdown.stages.length > 0 && (
            <div className="grid min-w-0 gap-3.5 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
              <Card className="min-w-0 rounded-lg py-0">
                <CardContent className="space-y-3 p-4">
                  <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <h3 className="text-sm font-semibold text-foreground">
                        Bottleneck &amp; latency budget
                      </h3>
                      <p className="mt-1 text-xs text-muted-foreground">
                        Which stage caps end-to-end throughput, then where a
                        single request's time actually goes.
                      </p>
                    </div>
                    <div className="flex shrink-0 flex-wrap gap-1.5 sm:justify-end">
                      <Badge variant="secondary" className="rounded-md">
                        {breakdown.stages.length} stages
                      </Badge>
                    </div>
                  </div>
                  <StageBreakdownTable
                    stages={breakdown.stages}
                    ttftS={breakdown.ttft_s}
                    e2eS={breakdown.e2e_s}
                    queueWaitS={breakdown.queue_wait_s}
                    e2eQps={Number((selectedSweepPoint as any)?.qps ?? 0)}
                  />
                </CardContent>
              </Card>

              <StageScalingCard
                stages={breakdown.stages}
                qpsBound={breakdown.qps_bound}
                getStageSweep={breakdown.getStageSweep}
                e2eQps={Number((selectedSweepPoint as any)?.qps ?? 0)}
              />
            </div>
          )}
        </div>
      </StageInteractionProvider>
    </div>
  );
}
