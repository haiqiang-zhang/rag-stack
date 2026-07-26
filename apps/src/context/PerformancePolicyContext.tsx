import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import yaml from "js-yaml";
import type { Evaluation, RagoSweepPoint } from "@/types";
import { useProject } from "@/context/ProjectContext";
import {
  applyPerformanceSelection,
  type PerformancePolicy,
  type PerformanceMetricKey,
  type SelectionPolicy,
} from "@/utils/performanceSelection";

interface PerformancePolicyState extends PerformancePolicy {
  setSelection: (s: SelectionPolicy) => void;
  setWeights: (w: Partial<Record<PerformanceMetricKey, number>>) => void;
  setConstraint: (c: Partial<Record<PerformanceMetricKey, number>>) => void;
  setOptimize: (m: PerformanceMetricKey) => void;

  /** Inject the sweep cache from `useAllRagoSweeps`. */
  setSweeps: (m: Map<number, RagoSweepPoint[]>) => void;

  /** How many per-eval sweep fetches have RESOLVED — fetched or confirmed
   *  absent. CM-only evals ship no sweep CSV, so this is deliberately not a
   *  success count; readiness must not wait on files that don't exist. */
  sweepsResolvedCount: number;
  /** Total number of sweep fetches expected (drives `sweepsReady`). */
  sweepsTotalCount: number;
  /** True once every expected sweep has resolved — chart can paint a final picture. */
  sweepsReady: boolean;
  setSweepsProgress: (resolved: number, total: number) => void;

  /**
   * Recomputed performance for an evaluation under the current policy.
   * Falls back to `ev.performance` when that eval's sweep hasn't loaded.
   * Returns `INFEASIBLE_SCORE` for `constrained` policies with no feasible
   * point — keeps Pareto math correct, but chart/table consumers should
   * use `getPerformanceResult` and check `feasible` to skip those points.
   */
  getPerformance: (ev: Evaluation) => number;

  /**
   * Same as `getPerformance` but also returns the feasibility flag and a
   * chart-friendly `displayScore` (= score for feasible; = unconstrained
   * best of the `optimize` metric for infeasible — keeps a usable x-axis
   * position instead of -1e9). Use this from chart/table code.
   */
  getPerformanceResult: (ev: Evaluation) => {
    score: number;
    feasible: boolean;
    displayScore: number;
  };

  /** True when the current policy differs from the last applied snapshot. */
  isDirty: boolean;
  /** Mark current policy as applied for this browser session. */
  save: () => Promise<void>;
  /** True while a save is in flight. */
  saving: boolean;
}

const DEFAULT_SELECTION: SelectionPolicy = "min_latency";
const DEFAULT_WEIGHTS: Partial<Record<PerformanceMetricKey, number>> = {
  latency_s: 1.0,
};
const DEFAULT_CONSTRAINT: Partial<Record<PerformanceMetricKey, number>> = {
  ttft_s: 0.5,
};
const DEFAULT_OPTIMIZE: PerformanceMetricKey = "throughput_qps";

// null default + throwing hook: a consumer mounted outside the provider would
// otherwise silently get raw ev.performance, ignoring the active policy.
const PerformancePolicyContext = createContext<PerformancePolicyState | null>(null);

/** Build the policy dict that mirrors the YAML `system.performance_objectives`
 *  shape. Omits empty/null sub-fields so the YAML stays minimal — selection-only
 *  policies don't carry weights/constraint baggage. */
function policyToYaml(p: PerformancePolicy): Record<string, unknown> {
  const out: Record<string, unknown> = { selection: p.selection };
  if (p.selection === "weighted" && p.weights && Object.keys(p.weights).length > 0) {
    out.weights = { ...p.weights };
  }
  if (p.selection === "constrained") {
    if (p.constraint && Object.keys(p.constraint).length > 0) {
      out.constraint = { ...p.constraint };
    }
    if (p.optimize) out.optimize = p.optimize;
  }
  return out;
}

/** Parse a YAML `system.performance_objectives` block into provider state.
 *  Tolerant to missing/partial blocks — falls back to defaults per field. */
function yamlToPolicy(block: unknown): PerformancePolicy {
  if (!block || typeof block !== "object") {
    return {
      selection: DEFAULT_SELECTION,
      weights: { ...DEFAULT_WEIGHTS },
      constraint: { ...DEFAULT_CONSTRAINT },
      optimize: DEFAULT_OPTIMIZE,
    };
  }
  const b = block as Record<string, unknown>;
  const sel = (typeof b.selection === "string" ? b.selection : DEFAULT_SELECTION) as SelectionPolicy;
  // Keep per-mode defaults for *the other* modes so a user toggling
  // "weighted"/"constrained" in the UI has something prefilled instead of an
  // empty form.
  const weights =
    b.weights && typeof b.weights === "object"
      ? (b.weights as Partial<Record<PerformanceMetricKey, number>>)
      : { ...DEFAULT_WEIGHTS };
  const constraint =
    b.constraint && typeof b.constraint === "object"
      ? (b.constraint as Partial<Record<PerformanceMetricKey, number>>)
      : { ...DEFAULT_CONSTRAINT };
  const optimize =
    typeof b.optimize === "string"
      ? (b.optimize as PerformanceMetricKey)
      : DEFAULT_OPTIMIZE;
  return { selection: sel, weights, constraint, optimize };
}

/** Stable string key for dirty-tracking. The save target is the YAML block,
 *  so we compare the *serialized* form — saving an identical block is a no-op. */
function policyFingerprint(p: PerformancePolicy): string {
  return JSON.stringify(policyToYaml(p));
}

export function PerformancePolicyProvider({ children }: { children: React.ReactNode }) {
  const { projectBaseUrl } = useProject();

  const [selection, setSelection] = useState<SelectionPolicy>(DEFAULT_SELECTION);
  const [weights, setWeights] = useState<Partial<Record<PerformanceMetricKey, number>>>(
    { ...DEFAULT_WEIGHTS },
  );
  const [constraint, setConstraint] = useState<
    Partial<Record<PerformanceMetricKey, number>>
  >({ ...DEFAULT_CONSTRAINT });
  const [optimize, setOptimize] = useState<PerformanceMetricKey>(DEFAULT_OPTIMIZE);
  const [sweeps, setSweeps] = useState<Map<number, RagoSweepPoint[]>>(new Map());
  const [sweepsResolvedCount, setSweepsResolvedCount] = useState(0);
  const [sweepsTotalCount, setSweepsTotalCount] = useState(0);
  const setSweepsProgress = useCallback((resolved: number, total: number) => {
    setSweepsResolvedCount(resolved);
    setSweepsTotalCount(total);
  }, []);
  // Empty projects and runs containing only failed evaluations have no sweeps
  // to fetch. Their zero-work queue is complete, not permanently loading.
  const sweepsReady = sweepsResolvedCount >= sweepsTotalCount;

  // Fingerprint of the policy as last loaded from / saved to YAML. Compared
  // against the live state to compute `isDirty`. Resets per project so dirty
  // state doesn't leak across project switches.
  const [savedFingerprint, setSavedFingerprint] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Hydrate state from project YAML whenever the project changes. js-yaml is
  // already used elsewhere for client-side YAML reads (ConfigVisualEditor),
  // so this stays consistent.
  useEffect(() => {
    if (!projectBaseUrl) {
      setSavedFingerprint(null);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${projectBaseUrl}/rag_stack_config.yaml`, {
          cache: "no-store",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        const doc = (yaml.load(text) as Record<string, unknown> | null) ?? {};
        const block = (doc.system as Record<string, unknown> | undefined)
          ?.performance_objectives;
        const next = yamlToPolicy(block);
        if (cancelled) return;
        setSelection(next.selection);
        setWeights(next.weights ?? { ...DEFAULT_WEIGHTS });
        setConstraint(next.constraint ?? { ...DEFAULT_CONSTRAINT });
        setOptimize(next.optimize ?? DEFAULT_OPTIMIZE);
        setSavedFingerprint(policyFingerprint(next));
      } catch {
        // YAML missing or unreadable — keep defaults. Treat defaults as
        // "saved" so isDirty starts false.
        if (cancelled) return;
        const fallback: PerformancePolicy = {
          selection: DEFAULT_SELECTION,
          weights: { ...DEFAULT_WEIGHTS },
          constraint: { ...DEFAULT_CONSTRAINT },
          optimize: DEFAULT_OPTIMIZE,
        };
        setSelection(fallback.selection);
        setWeights(fallback.weights!);
        setConstraint(fallback.constraint!);
        setOptimize(fallback.optimize!);
        setSavedFingerprint(policyFingerprint(fallback));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectBaseUrl]);

  const policy = useMemo<PerformancePolicy>(
    () => ({ selection, weights, constraint, optimize }),
    [selection, weights, constraint, optimize],
  );

  const isDirty =
    savedFingerprint !== null && policyFingerprint(policy) !== savedFingerprint;

  const save = useCallback(async () => {
    setSaving(true);
    try {
      setSavedFingerprint(policyFingerprint(policy));
    } finally {
      setSaving(false);
    }
  }, [policy]);

  const getPerformanceResult = useCallback(
    (ev: Evaluation): { score: number; feasible: boolean; displayScore: number } => {
      const rows = sweeps.get(ev.eval_id);
      if (!rows || rows.length === 0) {
        return { score: ev.performance, feasible: true, displayScore: ev.performance };
      }
      const res = applyPerformanceSelection(rows, policy);
      if (res.row == null) {
        return { score: ev.performance, feasible: true, displayScore: ev.performance };
      }
      // res.feasible is undefined for non-constrained policies — treat as feasible.
      return {
        score: res.score,
        feasible: res.feasible !== false,
        displayScore: res.displayScore,
      };
    },
    [sweeps, policy],
  );

  const getPerformance = useCallback(
    (ev: Evaluation): number => getPerformanceResult(ev).score,
    [getPerformanceResult],
  );

  const value = useMemo(
    () => ({
      selection,
      weights,
      constraint,
      optimize,
      setSelection,
      setWeights,
      setConstraint,
      setOptimize,
      setSweeps,
      sweepsResolvedCount,
      sweepsTotalCount,
      sweepsReady,
      setSweepsProgress,
      getPerformance,
      getPerformanceResult,
      isDirty,
      save,
      saving,
    }),
    [
      selection,
      weights,
      constraint,
      optimize,
      sweepsResolvedCount,
      sweepsTotalCount,
      sweepsReady,
      setSweepsProgress,
      getPerformance,
      getPerformanceResult,
      isDirty,
      save,
      saving,
    ],
  );

  return (
    <PerformancePolicyContext.Provider value={value}>
      {children}
    </PerformancePolicyContext.Provider>
  );
}

export function usePerformancePolicy(): PerformancePolicyState {
  const v = useContext(PerformancePolicyContext);
  if (!v) throw new Error("usePerformancePolicy must be used inside <PerformancePolicyProvider>");
  return v;
}
