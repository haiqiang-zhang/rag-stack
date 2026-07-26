import { useState, useEffect } from "react";
import { useProject } from "@/context/ProjectContext";
import { fetchProjectEvaluations } from "@/api/ragProject";
import type { Evaluation, EvalType } from "@/types";

/** Server rows are pre-normalized, but guard against an older backend. */
function asEvalType(raw: string): EvalType {
  switch (raw) {
    case "GT":
    case "CM":
    case "DUPLICATE":
    case "OOM":
    case "INVALID":
      return raw;
    default:
      return "INVALID";
  }
}

/**
 * Fetch all evaluations (parsed server-side, flat dot-path config knobs),
 * re-fetching when evalCount changes (live mode).
 */
export function useEvaluations(evalCount: number) {
  const { currentProject } = useProject();
  const [state, setState] = useState<{
    project: string | null;
    evaluations: Evaluation[];
    loading: boolean;
    error: Error | null;
  }>({ project: null, evaluations: [], loading: false, error: null });

  useEffect(() => {
    if (currentProject == null) return;
    let cancelled = false;
    setState((previous) => ({
      project: currentProject,
      evaluations: previous.project === currentProject ? previous.evaluations : [],
      loading: true,
      error: null,
    }));

    (async () => {
      try {
        const rows = await fetchProjectEvaluations(currentProject);
        if (cancelled) return;
        setState({
          project: currentProject,
          evaluations: rows.map((r) => ({
            eval_id: r.eval_id,
            eval_type: asEvalType(r.eval_type),
            eval_error: r.eval_error,
            performance: r.performance,
            quality: r.quality["combined"] ?? null,
            quality_metrics: r.quality,
            is_pareto: false, // enriched client-side by EvalDataContext
            config: r.config,
            gt_metrics: r.gt_metrics,
          })),
          loading: false,
          error: null,
        });
      } catch (err) {
        if (cancelled) return;
        setState((previous) => ({
          ...previous,
          loading: false,
          error: err instanceof Error ? err : new Error(String(err)),
        }));
      }
    })();

    return () => { cancelled = true; };
  }, [evalCount, currentProject]);

  // A project switch renders before the new effect runs. Never expose the
  // previous project's rows (or errors) during that render.
  if (currentProject == null || state.project !== currentProject) {
    return { evaluations: EMPTY_EVALUATIONS, loading: currentProject != null, error: null };
  }
  return { evaluations: state.evaluations, loading: state.loading, error: state.error };
}

const EMPTY_EVALUATIONS: Evaluation[] = [];
