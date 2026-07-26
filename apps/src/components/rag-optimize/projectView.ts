/** Project navigation depends on available results, never on sweep loading. */
export type ProjectTab =
  | "start" | "overview" | "tradeoff" | "performance" | "quality"
  | "config" | "cm-preview" | "history";

export function projectView(evalCount: number | undefined, requested?: ProjectTab) {
  const hasEvaluations = evalCount != null && evalCount > 0;
  const tabs: { key: ProjectTab; label: string }[] = hasEvaluations
    ? [
        { key: "overview", label: "Overview" },
        { key: "tradeoff", label: "Trade-off" },
        { key: "performance", label: "Performance" },
        { key: "quality", label: "Quality" },
        { key: "config", label: "Config" },
        { key: "cm-preview", label: "CM Preview" },
        { key: "history", label: "History" },
      ]
    : [
        { key: "start", label: "Start eval" },
        { key: "cm-preview", label: "CM Preview" },
        { key: "history", label: "History" },
      ];
  const tab = tabs.some((item) => item.key === requested)
    ? requested!
    : tabs[0].key;
  return { hasEvaluations, tabs, tab, loading: evalCount == null };
}
