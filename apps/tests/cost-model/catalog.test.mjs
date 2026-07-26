import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadTs } from "../support/load-ts.mjs";

const helpers = loadTs("src/components/cost-model/catalogModel.ts");
const forms = loadTs("src/components/cost-model/estimateForms.ts");
const ui = new Proxy({}, { get: () => ({ children }) => createElement("span", null, children) });
const dependencies = {
  "@/api/costModel": {},
  "@/components/ui/button": ui,
  "@/components/ui/badge": ui,
  "@/components/ui/input": ui,
  "@/components/ui/label": ui,
  "@/components/ui/select": ui,
  "@/lib/utils": loadTs("src/lib/utils.ts"),
  "./families": loadTs("src/components/cost-model/families.ts"),
  "./catalogModel": helpers,
  "./estimateForms": forms,
};
const { CostModelCatalogView } = loadTs("src/components/cost-model/CostModelMapping.tsx", dependencies);
const { SimulatePanel, buildQueryExpansionConfig } = loadTs("src/components/cost-model/SimulatePanel.tsx", dependencies);
const binding = {
  component: "vectordb", dispatch_stage: "semantic_retrieval_encode", db_type: null,
  cost_model: "retrieval_encode_sim", status: "priced", reason: null,
  execution_kind: "forward", performance_stages: [{ name: "semantic_retrieval_encode", kind: "prefill", resource: "gpu" }],
  resources: ["gpu"],
};
const blocked = { ...binding, component: "bm25", dispatch_stage: "lexical_retrieval", cost_model: null,
  status: "unsupported", reason: "BM25 has no CM physical policy.", execution_kind: null, performance_stages: [], resources: [] };
const catalog = { schema_version: "rag-stack.cost-model-catalog.v1", read_only: true, scope: "dispatch_catalog",
  stages: [{ stage: "retrieval", user_selectable: true, bindings: [binding, blocked] }], contract: {} };
const models = [{ name: "retrieval_encode_sim", family: "retrieval", description: "Query encoder", capabilities: [], mapped_module_types: ["vectordb"] }];

test("binding grouping preserves backend stage, index family and unsupported constraints", () => {
  const grouped = helpers.groupBindings([binding, { ...binding, component: "other_encoder" }, blocked,
    { ...binding, db_type: "new_backend_index" }]);
  assert.equal(grouped.length, 3);
  assert.deepEqual(grouped[0].components, ["vectordb", "other_encoder"]);
  assert.equal(grouped[1].status, "unsupported");
  assert.equal(grouped[1].cost_model, null);
  assert.equal(grouped[2].db_type, "new_backend_index");
});

test("search finds unsupported constraints without hiding them as pass-through", () => {
  const matches = helpers.filterCatalogStages(catalog, "physical policy");
  assert.equal(matches.length, 1);
  assert.deepEqual(matches[0].bindings, [blocked]);
  assert.equal(helpers.filterCatalogStages(catalog, "unknown").length, 0);
  assert.equal(helpers.filterCatalogStages(catalog, "retrieval")[0].bindings.length, 2);
});

test("catalog renderer shows server bindings, explicit blocker and all registered models", () => {
  const html = renderToStaticMarkup(createElement(CostModelCatalogView, { catalog, models }));
  for (const text of ["semantic_retrieval_encode", "retrieval_encode_sim", "BM25 has no CM physical policy.", "CM blocked", "Binding available", "Registered cost models"]) assert.ok(html.includes(text), text);
  assert.ok(!html.includes("pass-through"));
  assert.ok(!html.includes("faiss_ivf"), "client must not synthesize a vector search binding");
});

test("models without a dedicated form never reuse a reranker estimate", () => {
  for (const name of ["faiss_hnsw", "retrieval_encode_sim", "rrf", "future_model"]) {
    assert.equal(forms.hasComponentEstimateForm(name), false);
    const html = renderToStaticMarkup(createElement(SimulatePanel, { model: { name } }));
    assert.ok(html.includes("No component estimate form"));
    assert.ok(!html.includes("Run Estimate"));
    assert.ok(!html.includes("ms_marco_minilm"));
  }
  assert.equal(forms.hasComponentEstimateForm("passage_reranker_sim"), true);
});

test("catalog API refuses unknown protocol versions instead of showing misleading empty data", async () => {
  const { fetchCostModelCatalog } = loadTs("src/api/costModel.ts", {
    "./base": { apiFetch: async (path) => { assert.equal(path, "/api/cost-model/catalog"); return { ...catalog, schema_version: "future" }; } },
  });
  await assert.rejects(fetchCostModelCatalog(), /not supported/);
});

test("query expansion sends the canonical model field consumed by its adapter", () => {
  const config = buildQueryExpansionConfig({ component: "hyde", llm_model: "Qwen/Qwen2.5-3B-Instruct", batch_size: 1 });
  const module = config.node_lines[0].nodes[0].modules[0];
  assert.equal(module.model, "Qwen/Qwen2.5-3B-Instruct");
  assert.equal(Object.hasOwn(module, "llm_model"), false);
});

test("current linear calibration and non-LLM records render without invented efficiencies", () => {
  const { CalibrationProfileDetail } = loadTs("src/components/cost-model/calibration/CalibrationProfileDetail.tsx", {
    "@/components/charts/EChartsView": { default: () => null },
    "@/lib/theme": { CHART_COLORS: [] },
  });
  for (const profile of [
    { filename: "linear.json", compute_efficiency: null, memory_efficiency: null, linear_law: { slope: 0.2 }, match_scope: "shared_model_candidate" },
    { filename: "hnsw.json", overhead_us: 3 },
  ]) {
    const html = renderToStaticMarkup(createElement(CalibrationProfileDetail, { profile }));
    assert.ok(html.includes(profile.filename));
    assert.ok(html.includes("Profile record"));
    assert.ok(!html.includes("COMPUTE EFF"));
    assert.ok(!html.includes("NaN"));
  }
});
