import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadTs } from "../support/load-ts.mjs";

const helpers = loadTs("src/components/rag-optimize/cmPreviewModel.ts");
const { CMCompiledPlanView } = loadTs(
  "src/components/rag-optimize/CMCompiledPlanView.tsx",
  { "./cmPreviewModel": helpers },
);
const ragIR = {
  queries: [
    {
      question_id: "internal-0",
      source_question_id: "qa-7",
      calls: [
        {
          call_id: "generation-7",
          step_idx: 3,
          stage: "generator",
          node_id: "gen",
          output_tokens: 19,
        },
      ],
    },
  ],
};
const plan = {
  schema_version: "rag-cm.compiled-plan.v1",
  status: "compiled",
  prediction: null,
  workload: { summary: {}, normalization: {}, source_question_ids: ["qa-7"] },
  queries: [
    {
      question_id: "qa-7",
      operations: [
        {
          op_id: "op-3",
          name: "Generator decode",
          kind: "decode",
          source_call_ids: ["generation-7"],
          source_call_indices: [0],
          depends_on: ["op-2"],
          resource_ids: ["engine-0"],
          details: { output_tokens: 19 },
        },
      ],
    },
  ],
  resources: {
    workers: [],
    engines: [
      {
        id: "engine-0",
        name: "Generator",
        role: "decode",
        devices: [2],
        depth: 4,
        budget: 4096,
      },
    ],
    n_cards: 4,
    cpu_cores: 32,
  },
  deployment: { batching: { timeout_s: 0.01 } },
  calibration: { selection: "real-profile" },
};

test("renderer shows actual query calls, compiler mappings/resources, and an unpriced result", () => {
  const html = renderToStaticMarkup(
    createElement(CMCompiledPlanView, { plan, ragIR }),
  );
  for (const text of [
    "qa-7",
    "generation-7",
    "Generator decode",
    "op-3",
    "engine-0",
    "real-profile",
    "Source call indices",
    "index 0 · step 3",
    "latency and throughput have not been predicted",
  ])
    assert.ok(html.includes(text), text);
  assert.ok(
    html.includes("1 recorded calls"),
    "source_question_id must match the compiled query",
  );
  assert.ok(!html.includes("QPS"));
});

test("blocked compilation still renders real RAG-IR without inventing operations", () => {
  const html = renderToStaticMarkup(
    createElement(CMCompiledPlanView, { plan: null, ragIR }),
  );
  assert.ok(html.includes("generation-7"));
  assert.ok(html.includes("No compiled path is available"));
  assert.ok(!html.includes("Generator decode"));
});

test("unknown plan schemas do not silently become a current-version visualization", () => {
  const html = renderToStaticMarkup(
    createElement(CMCompiledPlanView, {
      plan: { ...plan, schema_version: "future" },
      ragIR,
    }),
  );
  assert.ok(html.includes("not supported by this client"));
  assert.ok(!html.includes("Generator decode"));
});

test("saved display projections render after reopening without an executable runtime object", () => {
  const savedProjection = JSON.parse(JSON.stringify(plan));
  const html = renderToStaticMarkup(createElement(CMCompiledPlanView, {
    plan: savedProjection, ragIR, compiledPathStatus: "available",
  }));
  assert.ok(html.includes("Display only"));
  assert.ok(html.includes("Generator decode"));
  assert.ok(html.includes("generation-7"));
  assert.ok(!html.includes("server memory"));
  assert.ok(!html.includes("restart"));
});

test("historical jobs without a saved description retain their observed trace", () => {
  const html = renderToStaticMarkup(createElement(CMCompiledPlanView, {
    plan: null, ragIR, compiledPathStatus: "unavailable",
  }));
  assert.ok(html.includes("No saved compiled path preview is available"));
  assert.ok(html.includes("Create a new preview"));
  assert.ok(html.includes("generation-7"));
  assert.ok(!html.includes("Generator decode"));
  assert.ok(!html.includes("server memory"));
});
