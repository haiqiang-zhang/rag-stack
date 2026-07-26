import React from "react";
import type { CalibrationProfile } from "@/types/costModel";
import { Badge, BadgeText } from "@/components/ui/LegacyBadge";
import { cn } from "@/lib/utils";

interface Props {
  profiles: CalibrationProfile[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

export function ProfileList({ profiles, selectedIndex, onSelect }: Props) {
  if (profiles.length === 0) {
    return (
      <div className="flex flex-col items-center p-5">
        <span className="text-[13px] text-muted-foreground">
          No calibration profiles found
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {profiles.map((p, i) => {
        const isActive = selectedIndex === i;
        return (
          <div
            key={p.filename}
            onClick={() => onSelect(i)}
            className={cn(
              "cursor-pointer select-none rounded-xl border p-[14px]",
              isActive
                ? "border-primary bg-primary/10"
                : "border-border bg-card hover:border-border/80",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="text-[14px] font-semibold text-foreground">
                {p.hardware_preset}
              </span>
              <Badge variant="warning">
                <BadgeText variant="warning">
                  MAPE {p.mape_latency.toFixed(1)}%
                </BadgeText>
              </Badge>
            </div>

            <div className="mt-1.5 flex gap-3">
              <span className="text-[11px] text-muted-foreground">
                Trials: {p.num_trials}
              </span>
              <span className="text-[11px] text-muted-foreground">
                Max APE: {p.max_ape_latency.toFixed(1)}%
              </span>
            </div>

            <div className="mt-1 text-[11px] text-muted-foreground">
              {p.index_configs.join(", ")}
            </div>

            <div className="mt-1 text-[10px] text-muted-foreground/60">
              {p.timestamp}
            </div>
          </div>
        );
      })}
    </div>
  );
}
