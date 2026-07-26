import React from "react";
import type { CalibrationProfileRecord } from "@/types/costModel";
import { SideItem, SideItemDot } from "@/components/ui/SideItem";

// from old theme.ts
const SUCCESS = "#10B981";
const WARNING = "#F59E0B";
const ERROR = "#EF4444";

interface Props {
  profiles: CalibrationProfileRecord[];
  selectedIndex: number | null;
  onSelect: (index: number) => void;
}

/** Traffic-light color for the MAPE: green <10%, amber <25%, red otherwise. */
export function mapeColor(mapeFraction: number): string {
  if (!Number.isFinite(mapeFraction)) return "#94A3B8";
  const pct = mapeFraction * 100;
  if (pct < 10) return SUCCESS;
  if (pct < 25) return WARNING;
  return ERROR;
}

export function CalibrationProfileList({ profiles, selectedIndex, onSelect }: Props) {
  if (profiles.length === 0) {
    return (
      <div className="flex items-center justify-center px-4 py-10">
        <span className="text-center text-xs text-muted-foreground">
          No profiles match the current filters.
        </span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-0.5 py-1">
      {profiles.map((p, i) => {
        const active = selectedIndex === i;
        const color = p.error ? ERROR : mapeColor(p.fit_mape);
        const mapePct = Number.isFinite(p.fit_mape) ? `${(p.fit_mape * 100).toFixed(1)}%` : "No fit metric";
        return (
          <SideItem key={`${p.filename}:${p.record_key ?? p.component ?? i}`} active={active} onClick={() => onSelect(i)}>
            {/* Left accent strip — MAPE traffic light */}
            <div
              className="mr-0.5 w-[3px] self-stretch rounded-[2px]"
              style={{ backgroundColor: color, opacity: active ? 1 : 0.65 }}
            />
            <SideItemDot active={active} />
            <div className="flex flex-1 flex-col gap-0.5">
              <div className="flex items-center gap-1.5">
                <span
                  className={`truncate text-xs ${
                    active
                      ? "font-bold text-foreground"
                      : "font-semibold text-muted-foreground"
                  }`}
                >
                  {p.component || p.filename}
                </span>
              </div>
              <span
                className={`truncate text-[11px] ${
                  active ? "text-muted-foreground" : "text-muted-foreground/80"
                }`}
              >
                {p.model_key}
              </span>
              <div className="flex items-center gap-1.5">
                <span className="font-mono text-[10px] text-muted-foreground">
                  {p.hardware_key}
                </span>
                <span className="text-[10px] text-muted-foreground/60">·</span>
                <span className="font-mono text-[10px] text-muted-foreground">
                  {p.bits}
                </span>
                <span className="text-[10px] text-muted-foreground/60">·</span>
                <span
                  className="font-mono text-[10px] font-semibold"
                  style={{ color }}
                >
                  {p.error ? "ERR" : mapePct}
                </span>
              </div>
            </div>
          </SideItem>
        );
      })}
    </div>
  );
}
