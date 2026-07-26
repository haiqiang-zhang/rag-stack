import assert from "node:assert/strict";
import test from "node:test";
import { loadTs } from "../support/load-ts.mjs";

const { yamlObjToFlow } = loadTs("src/components/pipeline-editor/hooks/useYamlToFlow.ts");
const { flowToConfig, flowToNodeLines, moveInSequence, sequenceEdges } = loadTs("src/components/pipeline-editor/hooks/useFlowToYaml.ts");

function hybridConfig() {
  return {
    global: { eval_backend: "static_gt" },
    algo_search_space: {
      vectordb: [{ name: "dense", db_type: "faiss_hnsw" }],
      pipeline: { node_lines: [
        { node_line_name: "retrieval", nodes: [
          { stage: "semantic_retrieval", top_k: [32, 64], modules: [{ component: "vectordb", vectordb: ["dense", "dense2"] }] },
          { stage: "lexical_retrieval", top_k: { range: [32, 64], n_grid: 3 }, modules: [{ component: "bm25", bm25_tokenizer: "porter_stemmer" }] },
          { stage: "hybrid_retrieval", top_k: 8, modules: [{ component: "hybrid_rrf", weight: [30, 60] }] },
        ] },
        { node_line_name: "generate", nodes: [{ stage: "generator", modules: [{ component: "vllm", temperature: 0 }] }] },
      ] },
    },
  };
}

test("sequential hybrid round-trips scalar, list, range, and selector config", () => {
  const config = hybridConfig();
  const flow = yamlObjToFlow(config);
  assert.deepEqual(flowToConfig(flow.nodes, flow.nonVisual, flow.edges), config);
  assert.deepEqual(config, hybridConfig(), "round-trip must not mutate the source");
});

test("moving nodes/groups and reversing display edges never changes execution", () => {
  const config = hybridConfig();
  const { nodes, nonVisual, edges } = yamlObjToFlow(config);
  const dragged = nodes.map((node, i) => ({ ...node, position: { x: 1000 - 400 * i, y: -100 * i } })).reverse();
  const reversedEdges = edges.map((edge) => ({ ...edge, source: edge.target, target: edge.source }));
  assert.deepEqual(flowToConfig(dragged, nonVisual, reversedEdges), config);
});

test("explicit order changes update both YAML sequence and displayed linear edges", () => {
  const { nodes, nonVisual } = yamlObjToFlow(hybridConfig());
  const reordered = moveInSequence(nodes, "node-0-1", -1);
  assert.deepEqual(flowToNodeLines(reordered)[0].nodes.map((node) => node.stage), ["lexical_retrieval", "semantic_retrieval", "hybrid_retrieval"]);
  const edges = sequenceEdges(reordered);
  assert.deepEqual(edges.map(({ source, target }) => [source, target]), [
    ["node-0-1", "node-0-0"], ["node-0-0", "node-0-2"], ["node-0-2", "node-1-0"],
  ]);
  assert.deepEqual(reordered.map((n) => n.position), nodes.map((n) => n.position));
  assert.deepEqual(nonVisual, hybridConfig());
});

test("moving a node line uses explicit order across empty lines", () => {
  const config = hybridConfig();
  config.algo_search_space.pipeline.node_lines.splice(1, 0, { node_line_name: "empty", nodes: [] });
  const { nodes } = yamlObjToFlow(config);
  const reordered = moveInSequence(moveInSequence(nodes, "group-2", -1), "group-2", -1);
  assert.deepEqual(flowToNodeLines(reordered).map((line) => line.node_line_name), ["generate", "retrieval", "empty"]);
  assert.deepEqual(sequenceEdges(reordered).map(({ source, target }) => [source, target]), [
    ["node-2-0", "node-0-0"], ["node-0-0", "node-0-1"], ["node-0-1", "node-0-2"],
  ]);
});

test("round-trip retains evaluator settings and unrecognized node parameters", () => {
  const config = hybridConfig();
  config.eval_backend_setting = { custom_metrics: ["answer_score"], parallel: false };
  config.algo_search_space.pipeline.node_lines[0].nodes[2].optional = true;
  config.algo_search_space.pipeline.node_lines[0].nodes[2].custom_policy = { tie_break: "first" };
  const { nodes, nonVisual, edges } = yamlObjToFlow(config);

  assert.deepEqual(flowToConfig(nodes, nonVisual, edges), config);
});

test("sequence boundary moves preserve the execution plan and source nodes", () => {
  const { nodes } = yamlObjToFlow(hybridConfig());
  const original = structuredClone(nodes);

  assert.strictEqual(moveInSequence(nodes, "node-0-0", -1), nodes);
  assert.strictEqual(moveInSequence(nodes, "node-0-2", 1), nodes);
  assert.strictEqual(moveInSequence(nodes, "missing", 1), nodes);
  moveInSequence(nodes, "node-0-1", -1);
  assert.deepEqual(nodes, original);
});
