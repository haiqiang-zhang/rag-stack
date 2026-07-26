import assert from "node:assert/strict";
import test from "node:test";
import { loadTs } from "../support/load-ts.mjs";

const {
  buildPreviewRequest,
  parseQuestionIds,
  activeParameters,
  operationSources,
  rawQueryForCompiledQuery,
  isPreviewActive,
} = loadTs("src/components/rag-optimize/cmPreviewModel.ts");
const form = {
  source: "evaluation",
  evalId: 0,
  parameters: "{}",
  queryMode: "sample",
  sampleCount: "8",
  questionIds: "",
  reuseTrace: false,
  skipJudge: true,
  qaData: "",
  corpusData: "",
  datasetName: "",
  systemConfig: "",
  systemRequired: false,
};

test("preview requests preserve explicit source and default real eight-query scope", () => {
  assert.deepEqual(buildPreviewRequest(form), {
    eval_id: 0,
    sample_count: 8,
    reuse_trace: false,
    skip_judge: true,
  });
  assert.throws(
    () => buildPreviewRequest({ ...form, evalId: null }),
    /Select an evaluation/,
  );
  const request = buildPreviewRequest({
    ...form,
    source: "parameters",
    parameters: '{"retrieval__top_k":5}',
    qaData: " /data/qa.parquet ",
    corpusData: "/data/corpus.parquet",
    datasetName: "dragonball_en",
    systemConfig: '{"cpu":"Xeon"}',
  });
  assert.deepEqual(request.parameters, { retrieval__top_k: 5 });
  assert.equal(request.eval_id, undefined);
  assert.equal(request.qa_data, "/data/qa.parquet");
  assert.equal(request.dataset_name, "dragonball_en");
  assert.deepEqual(request.system_config, { cpu: "Xeon" });
  assert.equal(request.parameters.dataset, undefined);
});

test("explicit question IDs retain type/order and are exclusive with a sample count", () => {
  assert.deepEqual(parseQuestionIds('["001", 1, "q1"]'), ["001", 1, "q1"]);
  const request = buildPreviewRequest({
    ...form,
    queryMode: "ids",
    questionIds: '["001", 1]',
    reuseTrace: true,
  });
  assert.deepEqual(request.question_ids, ["001", 1]);
  assert.equal(request.sample_count, undefined);
  assert.equal(request.reuse_trace, true);
  for (const invalid of [
    "[]",
    '["q", "q"]',
    '[1,"1"]',
    "[null]",
    "[true]",
    "[1.2]",
    '[""]',
    "q1,q2",
    JSON.stringify(Array.from({ length: 129 }, (_, i) => i)),
  ]) {
    assert.throws(() => parseQuestionIds(invalid));
  }
});

test("invalid or ambiguous run inputs fail before starting a job", () => {
  for (const sampleCount of ["", "0", "129", "1.5", "NaN"])
    assert.throws(
      () => buildPreviewRequest({ ...form, sampleCount }),
      /Sample count/,
    );
  for (const parameters of ["[]", "null", "{bad}"])
    assert.throws(
      () => buildPreviewRequest({ ...form, source: "parameters", parameters }),
      /Point parameters/,
    );
  assert.throws(
    () =>
      buildPreviewRequest({ ...form, source: "parameters", reuseTrace: true }),
    /requires an existing evaluation/,
  );
  assert.throws(
    () =>
      buildPreviewRequest({
        ...form,
        reuseTrace: true,
        qaData: "other.parquet",
      }),
    /keeps its original dataset/,
  );
  assert.throws(
    () =>
      buildPreviewRequest({
        ...form,
        source: "parameters",
        systemRequired: true,
      }),
    /concrete system config/,
  );
  assert.throws(
    () => buildPreviewRequest({ ...form, systemConfig: "[]" }),
    /System config/,
  );
});

test("conditional point controls follow only selected Ax branches without choosing defaults", () => {
  const parameters = [
    {
      name: "mode",
      type: "choice",
      values: ["dense", "bm25"],
      dependents: { dense: ["embedding"], bm25: ["k1"] },
    },
    {
      name: "embedding",
      type: "choice",
      values: ["small", "large"],
      dependents: { large: ["precision"] },
    },
    { name: "k1", type: "range", bounds: [0.1, 2] },
    { name: "precision", type: "choice", values: ["fp16", "fp32"] },
    { name: "top_k", type: "range", bounds: [1, 20] },
  ];
  assert.deepEqual(
    activeParameters(parameters, {}).map((p) => p.name),
    ["mode", "top_k"],
  );
  assert.deepEqual(
    activeParameters(parameters, { mode: "dense", embedding: "large" }).map(
      (p) => p.name,
    ),
    ["mode", "embedding", "precision", "top_k"],
  );
  assert.deepEqual(
    activeParameters(parameters, { mode: "bm25", embedding: "large" }).map(
      (p) => p.name,
    ),
    ["mode", "k1", "top_k"],
  );
  const booleanParameters = [
    {
      name: "enabled",
      type: "choice",
      values: [true, false],
      dependents: { True: ["child"] },
    },
    { name: "child", type: "range", bounds: [1, 2] },
  ];
  assert.deepEqual(
    activeParameters(booleanParameters, { enabled: true }).map((p) => p.name),
    ["enabled", "child"],
  );
  assert.deepEqual(
    activeParameters(booleanParameters, { enabled: false }).map((p) => p.name),
    ["enabled"],
  );
});

test("point requests include server-declared constants and require explicit variable values", () => {
  const parameterSchema = [
    { name: "backend", type: "fixed", value: "static_gt" },
    { name: "top_k", type: "choice", values: [5, 10] },
  ];
  const base = { ...form, source: "parameters", parameterSchema };
  assert.throws(() => buildPreviewRequest(base), /Choose a value for top_k/);
  assert.deepEqual(
    buildPreviewRequest({ ...base, parameters: '{"top_k":10}' }).parameters,
    { top_k: 10, backend: "static_gt" },
  );
  assert.throws(
    () =>
      buildPreviewRequest({
        ...base,
        parameters: '{"top_k":10,"backend":"other"}',
      }),
    /is fixed/,
  );
});

test("CM call mapping uses compiler IDs and raw array positions across gapped or duplicate steps", () => {
  const calls = [
    { call_id: "a", stage: "generate", step_idx: 10 },
    { call_id: "b", stage: "generate", step_idx: 10 },
    { stage: "generate", step_idx: 30 },
  ];
  const operation = { source_call_ids: ["b"], source_call_indices: [2] };
  assert.deepEqual(
    operationSources(operation, calls).map((entry) => entry.index),
    [1, 2],
  );
  assert.deepEqual(
    operationSources(
      { source_call_ids: [], source_call_indices: [10, 30] },
      calls,
    ),
    [],
  );
  assert.deepEqual(
    operationSources(
      { source_call_ids: [], source_call_indices: [1] },
      calls,
    ).map((entry) => entry.call),
    [calls[1]],
  );
  const oneBasedSteps = [{ step_idx: 1 }, { step_idx: 2 }];
  assert.deepEqual(
    operationSources(
      { source_call_ids: [], source_call_indices: [1] },
      oneBasedSteps,
    ).map((entry) => entry.call),
    [oneBasedSteps[1]],
  );
  assert.deepEqual(
    operationSources(
      { source_call_ids: ["missing"], source_call_indices: [] },
      calls,
    ),
    [],
  );
});

test("only queued/running jobs enable cancellation and polling", () => {
  assert.equal(isPreviewActive("queued"), true);
  assert.equal(isPreviewActive("running"), true);
  for (const status of [
    "succeeded",
    "blocked",
    "failed",
    "cancelled",
    "interrupted",
    undefined,
  ])
    assert.equal(isPreviewActive(status), false);
});

test("repeated source question IDs retain each distinct raw invocation", () => {
  const queries = [
    {
      question_id: "invocation-1",
      source_question_id: "qa-7",
      calls: [{ call_id: "first" }],
    },
    {
      question_id: "invocation-2",
      source_question_id: "qa-7",
      calls: [{ call_id: "second" }],
    },
    { question_id: "qa-8", calls: [{ call_id: "third" }] },
  ];
  assert.equal(rawQueryForCompiledQuery(queries, "qa-7", 1), queries[1]);
  assert.equal(rawQueryForCompiledQuery(queries, "qa-7", 0), queries[0]);
  assert.equal(rawQueryForCompiledQuery(queries, "qa-7", 2), undefined);
  assert.equal(rawQueryForCompiledQuery(queries, "qa-8", 0), queries[2]);
});
