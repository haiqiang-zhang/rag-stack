import assert from "node:assert/strict";
import test from "node:test";
import { loadTs } from "../support/load-ts.mjs";

const { projectView } = loadTs("src/components/rag-optimize/projectView.ts");

test("new projects land in Start eval, including stale dashboard URLs", () => {
  for (const tab of [undefined, "overview", "config", "performance", "tradeoff", "quality"]) {
    const view = projectView(0, tab);
    assert.equal(view.tab, "start");
    assert.equal(view.loading, false);
    assert.equal(view.hasEvaluations, false);
    assert.deepEqual(view.tabs.map((item) => item.key), ["start", "cm-preview", "history"]);
  }
});

test("project discovery is loading, distinct from a confirmed empty project", () => {
  assert.equal(projectView(undefined).loading, true);
  assert.equal(projectView(0).loading, false);
});

test("the first evaluation transitions Start eval to Overview", () => {
  assert.equal(projectView(0, "start").tab, "start");
  assert.equal(projectView(1, "start").tab, "overview");
  assert.equal(projectView(1).hasEvaluations, true);
  assert.equal(projectView(1).tabs.length, 7);
});

test("CM Preview and History deep links survive the first evaluation", () => {
  for (const tab of ["cm-preview", "history"]) {
    assert.equal(projectView(0, tab).tab, tab);
    assert.equal(projectView(1, tab).tab, tab);
  }
  assert.equal(projectView(12, "performance").tab, "performance");
});

test("project configuration API addresses the project copy and keeps dataset external", async () => {
  const calls = [];
  const api = loadTs("src/api/ragProject.ts", {
    "@/lib/config": {},
    "./base": {
      apiFetch: async (...args) => calls.push(["GET", ...args]),
      apiPut: async (...args) => calls.push(["PUT", ...args]),
    },
  });
  await api.fetchProjectConfig("my first proj");
  await api.saveProjectConfig("my first proj", "global: {}\n");
  assert.deepEqual(calls, [
    ["GET", "/api/rag-project/config/my%20first%20proj"],
    ["PUT", "/api/rag-project/config", { project_name: "my first proj", content: "global: {}\n" }],
  ]);
});

test("save exposes canonical validation and immutable-run error details", async () => {
  const { apiPut } = loadTs("src/api/base.ts", {
    "@/lib/config": { getApiUrl: () => "http://unused", getAuthToken: () => null },
  });
  const fetch = globalThis.fetch;
  try {
    globalThis.fetch = async () => new Response(JSON.stringify({ detail: "Configuration is frozen after launch" }), { status: 409 });
    await assert.rejects(apiPut("/config", {}), /Configuration is frozen after launch/);
  } finally { globalThis.fetch = fetch; }
});
