import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
} from "react";
import yaml from "js-yaml";
import type { Evaluation } from "@/types";
import { applyFormula, type FormulaType } from "@/utils/pareto";
import { useProject } from "@/context/ProjectContext";

interface PostAnalysisState {
  /** Selected quality metric key: "combined" or a sub-metric name */
  qualityKey: string;
  setQualityKey: (k: string) => void;

  /** Combination formula (only relevant when qualityKey === "combined") */
  formula: FormulaType;
  setFormula: (f: FormulaType) => void;

  /** Per-metric weights (for weighted_mean formula) */
  weights: Record<string, number>;
  setWeights: (w: Record<string, number>) => void;

  /** Compute the quality value for an evaluation based on current settings */
  getQuality: (ev: Evaluation) => number | null;

  /** True when current state differs from the last applied snapshot. */
  isDirty: boolean;
  /** Mark current state as applied for this browser session. */
  save: () => Promise<void>;
  /** True while a save is in flight. */
  saving: boolean;
}

const DEFAULT_QUALITY_KEY = "combined";
const DEFAULT_FORMULA: FormulaType = "mean";
const DEFAULT_WEIGHTS: Record<string, number> = {};

// null default + throwing hook: a consumer mounted outside the provider would
// otherwise silently read a getQuality that returns null for everything.
const PostAnalysisContext = createContext<PostAnalysisState | null>(null);

interface QualityYamlBlock {
  metric: string;
  formula: FormulaType;
  weights: Record<string, number>;
}

/** Build the YAML `analysis.quality` block from provider state. Weights are
 *  only emitted when non-empty so the YAML stays compact for the default case. */
function stateToYaml(s: QualityYamlBlock): Record<string, unknown> {
  const out: Record<string, unknown> = { metric: s.metric, formula: s.formula };
  if (Object.keys(s.weights).length > 0) out.weights = { ...s.weights };
  return out;
}

function yamlToState(block: unknown): QualityYamlBlock {
  if (!block || typeof block !== "object") {
    return { metric: DEFAULT_QUALITY_KEY, formula: DEFAULT_FORMULA, weights: { ...DEFAULT_WEIGHTS } };
  }
  const b = block as Record<string, unknown>;
  return {
    metric: typeof b.metric === "string" ? b.metric : DEFAULT_QUALITY_KEY,
    formula: (typeof b.formula === "string" ? b.formula : DEFAULT_FORMULA) as FormulaType,
    weights:
      b.weights && typeof b.weights === "object"
        ? (b.weights as Record<string, number>)
        : { ...DEFAULT_WEIGHTS },
  };
}

function fingerprint(s: QualityYamlBlock): string {
  return JSON.stringify(stateToYaml(s));
}

export function PostAnalysisProvider({ children }: { children: React.ReactNode }) {
  const { projectBaseUrl } = useProject();

  const [qualityKey, setQualityKey] = useState<string>(DEFAULT_QUALITY_KEY);
  const [formula, setFormula] = useState<FormulaType>(DEFAULT_FORMULA);
  const [weights, setWeights] = useState<Record<string, number>>({ ...DEFAULT_WEIGHTS });

  const [savedFingerprint, setSavedFingerprint] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // Hydrate from YAML on project change.
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
        const block = (doc.analysis as Record<string, unknown> | undefined)?.quality;
        const next = yamlToState(block);
        if (cancelled) return;
        setQualityKey(next.metric);
        setFormula(next.formula);
        setWeights(next.weights);
        setSavedFingerprint(fingerprint(next));
      } catch {
        if (cancelled) return;
        const fallback: QualityYamlBlock = {
          metric: DEFAULT_QUALITY_KEY,
          formula: DEFAULT_FORMULA,
          weights: { ...DEFAULT_WEIGHTS },
        };
        setQualityKey(fallback.metric);
        setFormula(fallback.formula);
        setWeights(fallback.weights);
        setSavedFingerprint(fingerprint(fallback));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectBaseUrl]);

  const current = useMemo<QualityYamlBlock>(
    () => ({ metric: qualityKey, formula, weights }),
    [qualityKey, formula, weights],
  );

  const isDirty =
    savedFingerprint !== null && fingerprint(current) !== savedFingerprint;

  const save = useCallback(async () => {
    setSaving(true);
    try {
      setSavedFingerprint(fingerprint(current));
    } finally {
      setSaving(false);
    }
  }, [current]);

  const getQuality = useCallback(
    (ev: Evaluation): number | null => {
      if (!ev.quality_metrics || Object.keys(ev.quality_metrics).length === 0) {
        return null;
      }
      if (qualityKey === "combined") {
        if (formula !== "mean" || Object.keys(weights).length > 0) {
          return applyFormula(ev.quality_metrics, formula, weights);
        }
        const val = ev.quality_metrics["combined"];
        return val != null && !isNaN(val) ? val : null;
      }
      const val = ev.quality_metrics[qualityKey];
      return val != null && !isNaN(val) ? val : null;
    },
    [qualityKey, formula, weights],
  );

  const value = useMemo(
    () => ({
      qualityKey,
      setQualityKey,
      formula,
      setFormula,
      weights,
      setWeights,
      getQuality,
      isDirty,
      save,
      saving,
    }),
    [qualityKey, formula, weights, getQuality, isDirty, save, saving],
  );

  return (
    <PostAnalysisContext.Provider value={value}>
      {children}
    </PostAnalysisContext.Provider>
  );
}

export function usePostAnalysis(): PostAnalysisState {
  const v = useContext(PostAnalysisContext);
  if (!v) throw new Error("usePostAnalysis must be used inside <PostAnalysisProvider>");
  return v;
}
