import { useMemo, useState } from "react";
import { MousePointerClick } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EvalConfigView } from "./EvalConfigView";
import { SearchPointCard } from "./SearchPointCard";
import { useEvalData } from "@/context/EvalDataContext";
import { useSelection } from "@/context/SelectionContext";
import { isEvalFailure, type Evaluation } from "@/types";

/**
 * Config tab — reads top-down as a narrative:
 *   1. Search point   — the knobs the optimizer actually varied (+ diff)
 *   2. Resolved pipeline / full configuration — what those knobs became
 * The searched knobs lead because they are the decision; the resolved tree
 * is the consequence.
 */
export function ConfigTab() {
  const { selectedEval, enrichedEvals, getPerformanceResult, getQuality } =
    useEvalData();
  const { selectedEvalId } = useSelection();

  // Compare candidates: successful evals (dedup by id, latest row), Pareto
  // members first (they're the interesting reference points), then by id.
  const compareOptions = useMemo(() => {
    const byId = new Map<number, Evaluation>();
    for (const ev of enrichedEvals) byId.set(ev.eval_id, ev);
    return [...byId.values()]
      .filter(
        (ev) => !isEvalFailure(ev.eval_type) && ev.eval_id !== selectedEvalId,
      )
      .sort(
        (a, b) =>
          Number(b.is_pareto) - Number(a.is_pareto) || a.eval_id - b.eval_id,
      );
  }, [enrichedEvals, selectedEvalId]);

  const [compareId, setCompareId] = useState<string>("none");
  const compare = useMemo(
    () =>
      compareId === "none"
        ? null
        : (compareOptions.find((ev) => ev.eval_id === Number(compareId)) ?? null),
    [compareId, compareOptions],
  );

  if (!selectedEval) {
    return (
      <div className="flex flex-1 min-h-0 items-center justify-center p-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <MousePointerClick className="size-7 text-muted-foreground/50" />
          <span className="max-w-xs text-[13px] text-muted-foreground">
            Pick an evaluation in the left navigator to see exactly what it ran.
          </span>
        </div>
      </div>
    );
  }

  const failed = isEvalFailure(selectedEval.eval_type);
  const perf = getPerformanceResult(selectedEval);
  const qual = getQuality(selectedEval);

  return (
    <div className="flex-1 min-w-0 overflow-auto p-6 pb-12">
      {/* Eval header — identity, outcome, headline metrics, compare picker */}
      <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className="font-mono text-lg font-semibold text-foreground">
          Eval #{selectedEval.eval_id}
        </span>
        <span
          className={
            "rounded px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase " +
            (failed
              ? "bg-destructive/10 text-destructive"
              : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400")
          }
        >
          {selectedEval.eval_type}
        </span>
        {!failed && (
          <>
            <Metric
              label="Performance"
              value={perf.feasible ? perf.score.toFixed(3) : "infeasible"}
            />
            <Metric label="Quality" value={qual != null ? qual.toFixed(3) : "—"} />
          </>
        )}
        <div className="ml-auto">
          <Select value={compareId} onValueChange={setCompareId}>
            <SelectTrigger size="sm" className="w-56">
              <SelectValue placeholder="Compare with…" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">Compare: none</SelectItem>
              {compareOptions.map((ev) => (
                <SelectItem key={ev.eval_id} value={String(ev.eval_id)}>
                  #{ev.eval_id}
                  {ev.is_pareto ? " ★" : ""} · {ev.eval_type}
                  {ev.quality != null ? ` · q ${ev.quality.toFixed(3)}` : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {failed && (
        <Card className="mb-4 border-destructive/40 bg-destructive/5">
          <CardContent>
            <span className="mb-1 block text-[10px] font-semibold uppercase tracking-[0.5px] text-destructive">
              {selectedEval.eval_type} · no result
            </span>
            <span className="font-mono text-xs text-foreground">
              {selectedEval.eval_error || "This configuration produced no result."}
            </span>
          </CardContent>
        </Card>
      )}

      <div className="flex flex-col gap-4">
        <SearchPointCard
          evaluations={enrichedEvals}
          selected={selectedEval}
          compare={compare}
        />
        <EvalConfigView evalId={selectedEvalId} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-baseline gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-[0.6px] text-muted-foreground">
        {label}
      </span>
      <span className="font-mono text-sm font-semibold text-foreground">
        {value}
      </span>
    </span>
  );
}
