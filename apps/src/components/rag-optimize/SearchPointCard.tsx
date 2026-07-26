/**
 * Search point — THE lead card of the Config tab.
 *
 * Shows what the optimizer actually chose for this eval, one knob per row,
 * grouped by pipeline stage, with project context (how many values were
 * tried) and an optional second eval to diff against. Knobs that are fixed
 * across the whole run are folded away — they carry no decision information.
 */
import { useMemo, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import type { Evaluation } from "@/types";
import {
  buildKnobDimensions,
  formatKnobValue,
  varyingDims,
  type KnobDim,
} from "@/utils/searchKnobs";
import { cn } from "@/lib/utils";

interface SearchPointCardProps {
  evaluations: Evaluation[];
  selected: Evaluation;
  /** Second eval to diff against (rows that differ get highlighted). */
  compare: Evaluation | null;
}

export function SearchPointCard({ evaluations, selected, compare }: SearchPointCardProps) {
  const dims = useMemo(() => buildKnobDimensions(evaluations), [evaluations]);
  const varying = useMemo(() => varyingDims(dims), [dims]);
  const fixed = useMemo(() => dims.filter((d) => d.distinct.length <= 1), [dims]);

  // Rows: varying dims present in either eval, in group order.
  const rows = useMemo(
    () =>
      varying.filter(
        (d) =>
          d.values.has(selected.eval_id) ||
          (compare != null && d.values.has(compare.eval_id)),
      ),
    [varying, selected, compare],
  );
  const groups = useMemo(() => {
    const out = new Map<string, KnobDim[]>();
    for (const d of rows) {
      (out.get(d.group) ?? out.set(d.group, []).get(d.group)!).push(d);
    }
    return [...out.entries()];
  }, [rows]);

  const nDiff = useMemo(() => {
    if (!compare) return 0;
    return rows.filter(
      (d) => d.values.get(selected.eval_id) !== d.values.get(compare.eval_id),
    ).length;
  }, [rows, selected, compare]);

  return (
    <Card className="rounded-lg py-0">
      <CardContent className="p-4">
        <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-sm font-semibold text-foreground">Search point</h3>
          <span className="font-mono text-[11px] text-muted-foreground">
            {rows.length} searched knobs
            {compare && (
              <>
                {" · "}
                <span className="text-foreground">{nDiff} differ</span> vs #
                {compare.eval_id}
              </>
            )}
          </span>
        </div>
        <p className="mb-3 text-xs text-muted-foreground">
          What the optimizer chose here, out of what it tried across the run.
          {compare == null && " Pick a compare eval (top right) to diff two points."}
        </p>

        <div className="min-w-0 overflow-x-auto rounded-md border">
          <table className="w-full min-w-[560px] border-collapse text-xs">
            <thead className="bg-muted/50 text-[10px] uppercase text-muted-foreground">
              <tr>
                <th className="px-3 py-2 text-left font-medium">Knob</th>
                <th className="px-3 py-2 text-left font-medium">
                  This eval (#{selected.eval_id})
                </th>
                {compare && (
                  <th className="px-3 py-2 text-left font-medium">
                    Compare (#{compare.eval_id})
                  </th>
                )}
                <th
                  className="px-3 py-2 text-right font-medium"
                  title="Distinct values the optimizer tried for this knob across the whole run"
                >
                  Tried
                </th>
              </tr>
            </thead>
            <tbody>
              {groups.map(([group, ds]) => (
                <GroupRows
                  key={group}
                  group={group}
                  dims={ds}
                  selected={selected}
                  compare={compare}
                />
              ))}
            </tbody>
          </table>
        </div>

        {fixed.length > 0 && <FixedKnobs dims={fixed} selected={selected} />}
      </CardContent>
    </Card>
  );
}

function GroupRows({
  group,
  dims,
  selected,
  compare,
}: {
  group: string;
  dims: KnobDim[];
  selected: Evaluation;
  compare: Evaluation | null;
}) {
  const cols = compare ? 4 : 3;
  return (
    <>
      <tr className="border-t bg-muted/25">
        <td
          colSpan={cols}
          className="px-3 py-1 font-mono text-[10px] font-semibold uppercase tracking-[0.5px] text-muted-foreground"
        >
          {group.replace(/_/g, " ")}
        </td>
      </tr>
      {dims.map((d) => {
        const mine = d.values.get(selected.eval_id);
        const theirs = compare ? d.values.get(compare.eval_id) : undefined;
        const differs = compare != null && mine !== theirs;
        return (
          <tr
            key={d.key}
            className={cn("border-t", differs && "bg-primary/5")}
            title={`${d.key} — tried: ${d.distinct.map((v) => formatKnobValue(v)).join(", ")}`}
          >
            <td className="px-3 py-1.5">
              <span
                className={cn(
                  "font-mono",
                  differs ? "font-semibold text-foreground" : "text-muted-foreground",
                )}
              >
                {d.label}
              </span>
            </td>
            <ValueCell value={mine} emphasized={differs} />
            {compare && <ValueCell value={theirs} emphasized={differs} />}
            <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono tabular-nums text-muted-foreground">
              {d.distinct.length}
            </td>
          </tr>
        );
      })}
    </>
  );
}

function ValueCell({
  value,
  emphasized,
}: {
  value: string | number | boolean | undefined;
  emphasized: boolean;
}) {
  return (
    <td className="px-3 py-1.5">
      <span
        title={value != null ? String(value) : undefined}
        className={cn(
          "font-mono",
          value == null
            ? "text-muted-foreground/60"
            : emphasized
              ? "font-semibold text-foreground"
              : "text-foreground",
        )}
      >
        {formatKnobValue(value)}
      </span>
    </td>
  );
}

function FixedKnobs({ dims, selected }: { dims: KnobDim[]; selected: Evaluation }) {
  const [open, setOpen] = useState(false);
  const present = dims.filter((d) => d.values.has(selected.eval_id));
  if (present.length === 0) return null;
  return (
    <div className="mt-3 border-t pt-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 py-1 text-muted-foreground hover:text-foreground"
      >
        {open ? (
          <ChevronDown className="size-3.5" />
        ) : (
          <ChevronRight className="size-3.5" />
        )}
        <span className="font-mono text-[11px]">
          {present.length} knobs fixed for the whole run
        </span>
      </button>
      {open && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {present.map((d) => (
            <span
              key={d.key}
              title={d.key}
              className="inline-flex max-w-full items-baseline gap-1.5 rounded border bg-muted/30 px-2 py-0.5 font-mono text-[11px]"
            >
              <span className="text-muted-foreground">
                {d.group !== d.label ? `${d.group}.${d.label}` : d.label}
              </span>
              <span className="truncate font-medium text-foreground">
                {formatKnobValue(d.values.get(selected.eval_id))}
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
