/**
 * SeriesTray — the strip between the header and the charts where chart
 * series are assembled. One chip per series; the chip's glyph is a
 * miniature of what the chart will draw for it — a plain line for a single
 * run, a line inside a translucent halo for a merged group rendered as
 * mean ± std across its runs.
 *
 * Interaction: click chips to arm them; with ≥ 2 armed a "Merge" action
 * appears. A group chip's chevron expands an inline detail strip for
 * rename / remove-member / split. Grouping is always the user's explicit
 * choice — nothing is inferred from project names.
 */
import React from "react";
import { ChevronDown, ChevronUp, Eye, EyeOff, Merge, Ungroup, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { BenchSeries } from "./projectSeries";

/** Mini convergence-curve glyph; `band` adds the ±std halo behind the line. */
function SeriesGlyph({ color, band }: { color: string; band: boolean }) {
  return (
    <svg width="20" height="12" viewBox="0 0 20 12" aria-hidden className="shrink-0">
      {band && (
        <polygon
          points="1,7.5 7,4 13,2.5 19,1 19,6 13,7.5 7,9 1,12"
          fill={color}
          opacity={0.22}
        />
      )}
      <polyline
        points="1,10 7,6.5 13,5 19,3.5"
        fill="none"
        stroke={color}
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

interface SeriesTrayProps {
  series: BenchSeries[];
  armed: string[]; // series keys
  onToggleArm: (key: string) => void;
  onMerge: () => void;
  /** Hidden series keys — hidden series stay selected/grouped/colored but
   *  aren't drawn (chart-legend semantics, persists across sessions). */
  hidden: Set<string>;
  onToggleHide: (key: string) => void;
  expandedGroupId: string | null;
  onToggleExpand: (groupId: string | null) => void;
  onRenameGroup: (groupId: string, name: string) => void;
  onSplitGroup: (groupId: string) => void;
  onRemoveMember: (groupId: string, member: string) => void;
}

export function SeriesTray({
  series,
  armed,
  onToggleArm,
  onMerge,
  hidden,
  onToggleHide,
  expandedGroupId,
  onToggleExpand,
  onRenameGroup,
  onSplitGroup,
  onRemoveMember,
}: SeriesTrayProps) {
  const expanded = series.find((s) => s.group && s.group.id === expandedGroupId);
  const nArmed = armed.length;
  const hasGroups = series.some((s) => s.group);
  const showHint = series.length >= 2 && !hasGroups && nArmed === 0;

  return (
    <div className="border-b border-border bg-muted/30 px-4 py-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[10px] font-semibold uppercase tracking-[1px] text-muted-foreground">
          Series
        </span>

        {series.map((s) => {
          const isArmed = armed.includes(s.key);
          const isGroup = !!s.group;
          const isExpanded = isGroup && s.group!.id === expandedGroupId;
          const isHidden = hidden.has(s.key);
          return (
            <div
              key={s.key}
              className={cn(
                "flex items-center overflow-hidden rounded-md border bg-card transition-all",
                isArmed ? "border-primary/60 ring-1 ring-primary/50" : "border-border",
              )}
            >
              <button
                type="button"
                aria-pressed={isArmed}
                title={isGroup ? `${s.members.length} runs — click to arm for merging` : "Click to arm for merging"}
                onClick={() => onToggleArm(s.key)}
                className={cn(
                  "flex cursor-pointer items-center gap-1.5 px-2 py-1 transition-colors hover:bg-muted/60",
                  isArmed && "bg-primary/5",
                  isHidden && "opacity-45",
                )}
              >
                <SeriesGlyph color={s.color} band={isGroup && s.members.length > 1} />
                <span title={s.name} className="max-w-[180px] truncate font-mono text-[11px] text-foreground">{s.name}</span>
                {isGroup && (
                  <span className="rounded bg-muted px-1 text-[10px] font-semibold text-muted-foreground">
                    ×{s.members.length}
                  </span>
                )}
              </button>
              <button
                type="button"
                aria-pressed={isHidden}
                title={isHidden ? "Show in charts" : "Hide from charts (stays selected + keeps its color)"}
                onClick={() => onToggleHide(s.key)}
                className={cn(
                  "cursor-pointer self-stretch border-l border-border/60 px-1 transition-colors hover:bg-muted/60 hover:text-foreground",
                  isHidden ? "text-muted-foreground/70" : "text-muted-foreground",
                )}
              >
                {isHidden ? <EyeOff size={11} /> : <Eye size={11} />}
              </button>
              {isGroup && (
                <button
                  type="button"
                  title={isExpanded ? "Close" : "Rename, remove runs, or split"}
                  onClick={() => onToggleExpand(isExpanded ? null : s.group!.id)}
                  className="cursor-pointer self-stretch border-l border-border/60 px-1 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
                >
                  {isExpanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                </button>
              )}
            </div>
          );
        })}

        {nArmed >= 2 && (
          <button
            type="button"
            onClick={onMerge}
            className="flex cursor-pointer items-center gap-1.5 rounded-md border border-dashed border-primary/60 bg-primary/5 px-2 py-1 text-[11px] font-semibold text-primary transition-colors hover:bg-primary/10"
          >
            <Merge size={11} />
            Merge {nArmed} into one series
          </button>
        )}
        {nArmed === 1 && (
          <span className="text-[10px] text-muted-foreground">arm one more to merge</span>
        )}
        {showHint && (
          <span className="text-[10px] text-muted-foreground/80">
            same config, different seeds? click two runs, then merge them into one banded series
          </span>
        )}
      </div>

      {expanded && expanded.group && (
        <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-dashed border-border pt-2">
          <SeriesGlyph color={expanded.color} band={expanded.members.length > 1} />
          <Input
            value={expanded.group.name}
            onChange={(e) => onRenameGroup(expanded.group!.id, e.target.value)}
            aria-label="Series name"
            className="h-7 w-56 font-mono text-[11px]"
          />
          <div className="flex flex-wrap items-center gap-1">
            {expanded.group.members.map((m) => {
              const off = !expanded.members.includes(m); // in group but not selected
              return (
                <span
                  key={m}
                  className={cn(
                    "flex items-center gap-1 rounded bg-muted px-1.5 py-0.5 font-mono text-[10px]",
                    off ? "text-muted-foreground/60" : "text-foreground/90",
                  )}
                  title={off ? "In this series, but not selected in the project list" : undefined}
                >
                  {m}
                  <button
                    type="button"
                    title="Remove from series"
                    onClick={() => onRemoveMember(expanded.group!.id, m)}
                    className="cursor-pointer text-muted-foreground transition-colors hover:text-destructive"
                  >
                    <X size={10} />
                  </button>
                </span>
              );
            })}
          </div>
          <button
            type="button"
            onClick={() => onSplitGroup(expanded.group!.id)}
            className="ml-auto flex cursor-pointer items-center gap-1.5 rounded-md border border-border px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
          >
            <Ungroup size={11} />
            Split into single runs
          </button>
        </div>
      )}
    </div>
  );
}
