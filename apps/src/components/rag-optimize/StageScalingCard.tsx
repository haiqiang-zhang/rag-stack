import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StageScalingChart } from "@/components/charts/StageScalingChart";
import { useStageInteraction } from "@/context/StageInteractionContext";
import type { SweepRow } from "@/types";
import type { StageInfo } from "@/hooks/useEvalBreakdown";

interface StageScalingCardProps {
  stages: StageInfo[];
  qpsBound: string[];
  getStageSweep: (stage: string) => SweepRow[] | undefined;
  /** End-to-end rate of the selected deployment — reference line for the
   *  throughput curves. */
  e2eQps?: number;
}

export function StageScalingCard({
  stages,
  qpsBound,
  getStageSweep,
  e2eQps = 0,
}: StageScalingCardProps) {
  const { jumpToStage, consumeJump } = useStageInteraction();
  const [initialMode, setInitialMode] = useState<"line" | "single">("line");
  const [initialSingleStage, setInitialSingleStage] = useState<string | undefined>(
    undefined,
  );

  useEffect(() => {
    if (!jumpToStage) return;
    setInitialMode("single");
    setInitialSingleStage(jumpToStage);
    consumeJump();
  }, [jumpToStage, consumeJump]);

  return (
    <Card className="min-w-0 rounded-lg py-0">
      <CardContent className="space-y-3 p-4">
        <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">
              What if you resize a stage?
            </h3>
            <p className="mt-1 text-xs text-muted-foreground">
              Cost-model predictions around this deployment: each stage's
              throughput as batch size and chip count change. Start with the
              bottleneck — that's where resizing pays.
            </p>
          </div>
          {qpsBound.length > 0 && (
            <div className="flex max-w-full flex-wrap justify-start gap-1.5 sm:justify-end">
              {qpsBound.slice(0, 3).map((stage) => (
                <Badge key={stage} variant="secondary" className="max-w-[12rem] rounded-md">
                  <span className="truncate font-mono">{stage}</span>
                </Badge>
              ))}
              {qpsBound.length > 3 && (
                <Badge variant="outline" className="rounded-md">
                  +{qpsBound.length - 3}
                </Badge>
              )}
            </div>
          )}
        </div>
        <StageScalingChart
          stages={stages}
          qpsBound={qpsBound}
          getStageSweep={getStageSweep}
          e2eQps={e2eQps}
          initialMode={initialMode}
          initialSingleStage={initialSingleStage}
        />
      </CardContent>
    </Card>
  );
}
