/**
 * StatusPill — small rounded label that color-codes a job/run status.
 *
 * Statuses: created | running | completed | failed | quality_complete.
 * For "running" the leading dot pulses; everything else uses a tiny lucide icon.
 */
import { Activity, CircleCheckBig, CircleX, CircleDot } from "lucide-react";
import type { ElementType } from "react";

interface StatusConfig {
  /** Tailwind text color class for icon + label. */
  textClass: string;
  /** Tailwind background class for the pill. */
  bgClass: string;
  /** Tailwind background class for the pulsing dot (running). */
  dotClass: string;
  label: string;
  icon: ElementType;
}

const STATUS_CONFIG: Record<string, StatusConfig> = {
  created: {
    textClass: "text-muted-foreground",
    bgClass: "bg-muted",
    dotClass: "bg-muted-foreground",
    label: "Created",
    icon: CircleDot,
  },
  running: {
    textClass: "text-amber-700 dark:text-amber-300",
    bgClass: "bg-amber-100 dark:bg-amber-500/15",
    dotClass: "bg-amber-500",
    label: "Running",
    icon: Activity,
  },
  completed: {
    textClass: "text-emerald-700 dark:text-emerald-300",
    bgClass: "bg-emerald-100 dark:bg-emerald-500/15",
    dotClass: "bg-emerald-500",
    label: "Done",
    icon: CircleCheckBig,
  },
  failed: {
    textClass: "text-destructive",
    bgClass: "bg-destructive/10",
    dotClass: "bg-destructive",
    label: "Failed",
    icon: CircleX,
  },
  quality_complete: {
    textClass: "text-emerald-700 dark:text-emerald-300",
    bgClass: "bg-emerald-100 dark:bg-emerald-500/15",
    dotClass: "bg-emerald-500",
    label: "Quality Done",
    icon: CircleCheckBig,
  },
};

interface StatusPillProps {
  status: string;
  showLabel?: boolean;
}

export function StatusPill({ status, showLabel = true }: StatusPillProps) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.created;
  const Icon = cfg.icon;
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-md px-[7px] py-[3px] ${cfg.bgClass}`}
    >
      {status === "running" ? (
        <span
          className={`size-1.5 animate-pulse rounded-full ${cfg.dotClass}`}
        />
      ) : (
        <Icon className={`size-2.5 ${cfg.textClass}`} />
      )}
      {showLabel && (
        <span className={`text-[11px] font-medium ${cfg.textClass}`}>
          {cfg.label}
        </span>
      )}
    </span>
  );
}
