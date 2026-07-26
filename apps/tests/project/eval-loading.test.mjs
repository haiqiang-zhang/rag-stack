import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadTs } from "../support/load-ts.mjs";

const utils = loadTs("src/lib/utils.ts");
const searchKnobs = loadTs("src/utils/searchKnobs.ts");
const types = loadTs("src/types/index.ts", {
  "@/lib/stages": loadTs("src/lib/stages.ts"),
});
const baseData = {
  enrichedEvals: [],
  clientPareto: [],
  getQuality: (ev) => ev.quality,
  getPerformance: (ev) => ev.performance,
  getPerformanceResult: (ev) => ({ score: ev.performance, feasible: true }),
  getDisplayPerformance: (ev) => ev.performance,
  isFeasible: () => true,
  perfAxisLabel: "Performance",
  sweepsReady: true,
  loading: false,
  error: null,
};

function renderNavigator(overrides) {
  const { EvalNavigator } = loadTs("src/components/rag-optimize/EvalNavigator.tsx", {
    "@/lib/utils": utils,
    "@/types": types,
    "@/utils/searchKnobs": searchKnobs,
    "@/components/ui/SidePanel": loadTs("src/components/ui/SidePanel.tsx", { "@/lib/utils": utils }),
    "@/components/ui/SideItem": loadTs("src/components/ui/SideItem.tsx", { "@/lib/utils": utils }),
    "@/components/ui/Chip": loadTs("src/components/ui/Chip.tsx", { "@/lib/utils": utils }),
    "@/components/charts/knobColoring": loadTs("src/components/charts/knobColoring.ts", { "@/utils/searchKnobs": searchKnobs }),
    // ECharts needs a browser; assert that the real navigator chooses its
    // chart branch instead of mounting a canvas in this server renderer.
    "@/components/charts/ParetoChart": { ParetoChart: () => createElement("div", null, "Pareto chart") },
    "@/context/EvalDataContext": { useEvalData: () => ({ ...baseData, ...overrides }) },
    "@/context/SelectionContext": { useSelection: () => ({ selectedEvalId: null, setSelectedEvalId() {} }) },
  });
  return renderToStaticMarkup(createElement(EvalNavigator));
}

test("a loaded project with no evaluations shows an empty state, even without sweep data", () => {
  const html = renderNavigator({ sweepsReady: false });
  assert.match(html, /No evaluations yet/);
  assert.match(html, /Start an evaluation/);
  assert.doesNotMatch(html, /animate-spin|Loading|No evals in this filter/);
});

test("initial evaluation loading is distinct from a confirmed empty project", () => {
  const html = renderNavigator({ loading: true });
  assert.match(html, /animate-spin/);
  assert.doesNotMatch(html, /No evaluations yet|No evals in this filter/);
});

test("an evaluation request failure is visible and does not remain loading", () => {
  const html = renderNavigator({ error: new Error("HTTP 503") });
  assert.match(html, /role="alert"/);
  assert.match(html, /Unable to load evaluations/);
  assert.match(html, /HTTP 503/);
  assert.doesNotMatch(html, /animate-spin|No evaluations yet/);
});

test("successful evaluations wait for actual pending sweep fetches", () => {
  const evaluation = { eval_id: 1, eval_type: "GT", performance: 2, quality: 0.7, config: {} };
  const loading = renderNavigator({ enrichedEvals: [evaluation], sweepsReady: false });
  assert.match(loading, /animate-spin/);
  const ready = renderNavigator({ enrichedEvals: [evaluation], sweepsReady: true });
  assert.match(ready, /Pareto chart/);
  assert.doesNotMatch(ready, /animate-spin/);
});

test("zero expected sweeps are ready, including a project with only failed evaluations", () => {
  const policy = loadTs("src/context/PerformancePolicyContext.tsx", {
    "@/context/ProjectContext": { useProject: () => ({ projectBaseUrl: null }) },
    "@/utils/performanceSelection": loadTs("src/utils/performanceSelection.ts"),
  });
  function Consumer() {
    const { sweepsReady } = policy.usePerformancePolicy();
    assert.equal(sweepsReady, true);
    return createElement("span", null, String(sweepsReady));
  }
  renderToStaticMarkup(createElement(policy.PerformancePolicyProvider, null, createElement(Consumer)));
  const html = renderNavigator({
    enrichedEvals: [{ eval_id: 2, eval_type: "INVALID", eval_error: "invalid config", config: {} }],
  });
  assert.match(html, /INVALID/);
  assert.match(html, /no result/);
  assert.doesNotMatch(html, /animate-spin/);
});
