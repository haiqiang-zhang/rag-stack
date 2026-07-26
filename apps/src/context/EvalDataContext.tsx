import React, { createContext, useContext, useCallback, useMemo } from "react";
import { useSummary } from "@/hooks/useSummary";
import { useEvaluations } from "@/hooks/useEvaluations";
import { usePostAnalysis } from "@/context/PostAnalysisContext";
import { usePerformancePolicy } from "@/context/PerformancePolicyContext";
import { useSelection } from "@/context/SelectionContext";
import { computePareto } from "@/utils/pareto";
import type { Evaluation, Summary } from "@/types";

// Human-readable x-axis title per performance-selection policy, so the Pareto
// axis (sidebar + anywhere it renders) stops being a bare unitless "Performance".
const PERF_AXIS_LABEL: Record<string, string> = {
  max_throughput: "Throughput (qps) → higher is better",
  min_latency: "E2E latency (s) → lower is better",
  min_ttft: "TTFT (s) → lower is better",
  min_tpot: "TPOT (s) → lower is better",
  weighted: "Weighted performance score → higher is better",
  constrained: "Constrained performance score → higher is better",
};

interface PerfResult {
  score: number;
  feasible: boolean;
  displayScore: number;
}

interface EvalData {
  summary: Summary | null;
  /** Raw evaluations from all_evaluations.csv (fetched once, here). */
  evaluations: Evaluation[];
  /** Same list, each row tagged with the live client-Pareto membership. */
  enrichedEvals: Evaluation[];
  /** Client-computed non-dominated frontier under the active quality + policy. */
  clientPareto: Evaluation[];
  /** The eval currently selected in SelectionContext (or null). */
  selectedEval: Evaluation | null;
  loading: boolean;
  error: Error | null;
  sweepsReady: boolean;
  sweepsResolvedCount: number;
  sweepsTotalCount: number;
  getQuality: (ev: Evaluation) => number | null;
  getPerformance: (ev: Evaluation) => number;
  getPerformanceResult: (ev: Evaluation) => PerfResult;
  getDisplayPerformance: (ev: Evaluation) => number;
  isFeasible: (ev: Evaluation) => boolean;
  perfAxisLabel: string;
}

const Ctx = createContext<EvalData | null>(null);

export function useEvalData(): EvalData {
  const v = useContext(Ctx);
  if (!v) throw new Error("useEvalData must be used inside <EvalDataProvider>");
  return v;
}

/**
 * Screen-scoped provider that owns the evaluation list + client Pareto once,
 * so the sidebar navigator, Overview, History, Config and Performance tabs all
 * read ONE frontier instead of each re-fetching all_evaluations.csv and
 * recomputing a divergent Pareto. Must be mounted inside the PostAnalysis +
 * PerformancePolicy providers (and beside SweepCacheBridge) so the policy /
 * quality getters and the bulk sweep cache are live.
 */
export function EvalDataProvider({ children }: { children: React.ReactNode }) {
  const { data: summary } = useSummary();
  const { evaluations, loading, error } = useEvaluations(summary?.n_evaluations ?? 0);
  const { getQuality } = usePostAnalysis();
  const perfPolicy = usePerformancePolicy();
  const {
    getPerformance,
    getPerformanceResult,
    sweepsReady,
    sweepsResolvedCount,
    sweepsTotalCount,
    selection,
  } = perfPolicy;
  const { selectedEvalId } = useSelection();

  const isFeasible = useCallback(
    (ev: Evaluation) => getPerformanceResult(ev).feasible,
    [getPerformanceResult],
  );
  const getDisplayPerformance = useCallback(
    (ev: Evaluation) => getPerformanceResult(ev).displayScore,
    [getPerformanceResult],
  );

  const clientPareto = useMemo(
    () => computePareto(evaluations, getQuality, getPerformance, isFeasible),
    [evaluations, getQuality, getPerformance, isFeasible],
  );
  const enrichedEvals = useMemo(() => {
    const ids = new Set(clientPareto.map((e) => e.eval_id));
    return evaluations.map((ev) => ({ ...ev, is_pareto: ids.has(ev.eval_id) }));
  }, [evaluations, clientPareto]);
  const selectedEval = useMemo(
    () => enrichedEvals.find((e) => e.eval_id === selectedEvalId) ?? null,
    [enrichedEvals, selectedEvalId],
  );
  const perfAxisLabel = PERF_AXIS_LABEL[selection] ?? "Performance";

  const value = useMemo<EvalData>(
    () => ({
      summary: summary ?? null,
      evaluations,
      enrichedEvals,
      clientPareto,
      selectedEval,
      loading,
      error,
      sweepsReady,
      sweepsResolvedCount,
      sweepsTotalCount,
      getQuality,
      getPerformance,
      getPerformanceResult,
      getDisplayPerformance,
      isFeasible,
      perfAxisLabel,
    }),
    [
      summary,
      evaluations,
      enrichedEvals,
      clientPareto,
      selectedEval,
      loading,
      error,
      sweepsReady,
      sweepsResolvedCount,
      sweepsTotalCount,
      getQuality,
      getPerformance,
      getPerformanceResult,
      getDisplayPerformance,
      isFeasible,
      perfAxisLabel,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
