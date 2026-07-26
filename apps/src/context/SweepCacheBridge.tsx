import { useEffect, useMemo } from "react";
import { useSummary } from "@/hooks/useSummary";
import { useEvaluations } from "@/hooks/useEvaluations";
import { useAllRagoSweeps } from "@/hooks/useAllRagoSweeps";
import { usePerformancePolicy } from "./PerformancePolicyContext";
import { isEvalSuccess } from "@/types";

/**
 * Loads every evaluation's rago_sweep CSV and pushes the result into
 * PerformancePolicyContext so `getPerformance` can recompute scores
 * client-side under any selection policy.
 *
 * Renders nothing; mount once inside PerformancePolicyProvider.
 */
export function SweepCacheBridge() {
  const { data: summary } = useSummary();
  const { evaluations } = useEvaluations(summary?.n_evaluations ?? 0);
  const { setSweeps, setSweepsProgress } = usePerformancePolicy();

  // Dedupe: a single eval can appear in the CSV under multiple rows (e.g.
  // first as CM-only, later upgraded to GT). Without `Set`, the same id is
  // queued twice and useAllRagoSweeps' totalCount > unique fetches forever,
  // which leaves sweepsReady stuck false.
  const evalIds = useMemo(
    () =>
      Array.from(
        new Set(
          evaluations
            .filter((ev) => isEvalSuccess(ev.eval_type))
            .map((ev) => ev.eval_id),
        ),
      ),
    [evaluations],
  );

  const { sweeps, resolvedCount, totalCount } = useAllRagoSweeps(evalIds);

  useEffect(() => {
    setSweeps(sweeps);
  }, [sweeps, setSweeps]);

  useEffect(() => {
    setSweepsProgress(resolvedCount, totalCount);
  }, [resolvedCount, totalCount, setSweepsProgress]);

  return null;
}
