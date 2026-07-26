import React from "react";
import { ChevronUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface CompactSummaryBarProps {
  evalId: number;
  performance: number;
  performanceLabel: string;
  qpsBound: string[];
  pointPlacement?: string;
  pointChips?: number;
  pointIdx: number;
  pointTotal: number;
  isPolicyDefault: boolean;
  onExpand: () => void;
}

export function CompactSummaryBar({
  evalId,
  performance,
  performanceLabel,
  qpsBound,
  pointPlacement,
  pointChips,
  pointIdx,
  pointTotal,
  isPolicyDefault,
  onExpand,
}: CompactSummaryBarProps) {
  const pointText =
    pointPlacement != null
      ? [
          `point ${pointIdx + 1}/${pointTotal}`,
          isPolicyDefault ? "default" : "custom",
          pointPlacement,
          pointChips != null ? `${pointChips} chips` : null,
        ]
          .filter(Boolean)
          .join(" · ")
      : null;

  return (
    <div className="flex min-h-11 w-full min-w-0 max-w-full items-center gap-3 overflow-hidden border-b border-border bg-card px-4 py-2">
      <Badge variant="secondary" className="shrink-0 font-mono text-[11px]">
        EVAL #{String(evalId).padStart(4, "0")}
      </Badge>

      <div className="flex shrink-0 items-center gap-3">
        <CompactStat label={performanceLabel} value={fmtScore(performance)} />
      </div>

      {qpsBound.length > 0 && (
        <div className="hidden min-w-0 max-w-[14rem] shrink items-center gap-1.5 rounded-md border border-primary/30 bg-primary/10 px-2 py-1 md:flex">
          <span className="shrink-0 text-[10px] font-medium uppercase text-muted-foreground">
            bound
          </span>
          <span
            className="min-w-0 truncate font-mono text-[11px] font-semibold text-primary"
            title={qpsBound.join(" + ")}
          >
            {qpsBound.join(" + ")}
          </span>
        </div>
      )}

      {pointText ? (
        <span
          className="hidden min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground sm:block"
          title={pointText}
        >
          {pointText}
        </span>
      ) : (
        <div className="min-w-0 flex-1" />
      )}

      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={onExpand}
        className="h-8 shrink-0 gap-1.5 px-2.5 text-[10px] font-semibold uppercase tracking-[0.5px]"
      >
        <ChevronUp className="size-[13px] text-muted-foreground" />
        <span className="hidden sm:inline">Expand</span>
      </Button>
    </div>
  );
}

function CompactStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline gap-[5px]">
      <span className="text-[9px] font-semibold uppercase tracking-[0.9px] text-muted-foreground">
        {label}
      </span>
      <span className="font-mono text-sm font-semibold text-foreground">
        {value}
      </span>
    </div>
  );
}

function fmtScore(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 100) return v.toFixed(1);
  if (Math.abs(v) >= 10) return v.toFixed(2);
  return v.toFixed(3);
}
