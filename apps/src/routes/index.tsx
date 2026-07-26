import { createFileRoute } from "@tanstack/react-router";
import RagOptimizeScreen from "@/screens/RagOptimizeScreen";

const VALID_TABS = [
  "start",
  "overview",
  "tradeoff",
  "performance",
  "quality",
  "config",
  "cm-preview",
  "history",
] as const;
type RagSubTab = (typeof VALID_TABS)[number];

interface RagOptimizeSearch {
  /** Selected project name. Empty → no selection (project list view). */
  project?: string;
  /** Defaults to Start eval before any results, Overview afterwards. */
  tab?: RagSubTab;
  /** One-shot deep-link: eval to select on arrival (e.g. from the Benchmark
   *  Pareto scatter). Applied to SelectionContext, then stripped. */
  eval?: number;
  /** Persisted CM preview job; reopening the URL restores its result. */
  preview?: string;
}

export const Route = createFileRoute("/")({
  component: RagOptimizeScreen,
  validateSearch: (search: Record<string, unknown>): RagOptimizeSearch => {
    const out: RagOptimizeSearch = {};
    if (typeof search.project === "string" && search.project.length > 0) {
      out.project = search.project;
    }
    if (
      typeof search.tab === "string" &&
      (VALID_TABS as readonly string[]).includes(search.tab)
    ) {
      out.tab = search.tab as RagSubTab;
    }
    const ev = Number(search.eval);
    if (search.eval != null && Number.isInteger(ev) && ev >= 0) {
      out.eval = ev;
    }
    if (typeof search.preview === "string" && search.preview.length > 0) {
      out.preview = search.preview;
    }
    return out;
  },
});
