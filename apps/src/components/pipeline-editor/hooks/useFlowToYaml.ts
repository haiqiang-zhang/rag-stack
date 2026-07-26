import type { Node, Edge } from "@xyflow/react";
import type {
  PipelineNodeData,
  NodeLineLabelData,
  NonVisualConfig,
} from "../types";

/**
 * Convert xyflow nodes/edges back into a `node_lines` array.
 *
 * Explicit line/node indices determine execution order. Canvas positions and
 * edges are presentation only, so dragging never edits the execution plan.
 */
export function flowToNodeLines(nodes: Node[], _edges: Edge[] = []): any[] {
  // Group + child indexing
  const groups = nodes.filter((n) => n.type === "nodeLineGroup")
    .sort((a, b) => (a.data as unknown as NodeLineLabelData).nodeLineIndex
      - (b.data as unknown as NodeLineLabelData).nodeLineIndex);
  const childrenByGroup = new Map<string, Node[]>();
  for (const g of groups) {
    const kids = nodes
      .filter((n) => n.type === "pipelineNode" && n.parentId === g.id)
      .sort((a, b) => (a.data as unknown as PipelineNodeData).nodeIndex
        - (b.data as unknown as PipelineNodeData).nodeIndex);
    childrenByGroup.set(g.id, kids);
  }

  return groups.map((group) => {
    const d = group.data as unknown as NodeLineLabelData;
    const children = childrenByGroup.get(group.id) ?? [];
    return {
      node_line_name: d.nodeLineName,
      nodes: children.map((n) => {
        const pd = n.data as unknown as PipelineNodeData;
        const node: Record<string, unknown> = {
          stage: pd.nodeType,
          ...(pd.optional ? { optional: true } : {}),
          ...pd.extraParams,
        };
        if (pd.modules && pd.modules.length > 0) node.modules = pd.modules;
        if (pd.strategy) node.strategy = pd.strategy;
        if (pd.topK !== undefined) node.top_k = pd.topK;
        return node;
      }),
    };
  });
}

/** Display the same linear sequence that is serialized to node_lines. */
export function sequenceEdges(nodes: Node[]): Edge[] {
  const groups = nodes.filter((n) => n.type === "nodeLineGroup")
    .sort((a, b) => Number(a.data.nodeLineIndex) - Number(b.data.nodeLineIndex));
  const ordered = groups.flatMap((group) => nodes
    .filter((n) => n.type === "pipelineNode" && n.parentId === group.id)
    .sort((a, b) => Number(a.data.nodeIndex) - Number(b.data.nodeIndex)));
  return ordered.slice(1).map((node, index) => ({
    id: `sequence-${ordered[index].id}-${node.id}`,
    source: ordered[index].id,
    target: node.id,
    selectable: false,
    deletable: false,
  }));
}

/** Swap explicit order with an adjacent peer, preserving canvas positions. */
export function moveInSequence(nodes: Node[], id: string, direction: -1 | 1): Node[] {
  const node = nodes.find((n) => n.id === id);
  if (!node) return nodes;
  const key = node.type === "nodeLineGroup" ? "nodeLineIndex" : "nodeIndex";
  const peers = nodes.filter((n) => n.type === node.type && n.parentId === node.parentId)
    .sort((a, b) => Number(a.data[key]) - Number(b.data[key]));
  const adjacent = peers[peers.findIndex((n) => n.id === id) + direction];
  if (!adjacent) return nodes;
  return nodes.map((n) => n.id === id ? { ...n, data: { ...n.data, [key]: adjacent.data[key] } }
    : n.id === adjacent.id ? { ...n, data: { ...n.data, [key]: node.data[key] } } : n);
}

/**
 * Merge the visual graph back into the full config (the static_gt pipeline).
 * Writes algo_search_space.pipeline.node_lines on a clone of the config base.
 * A flashrag config carries pipeline.methods instead — leave it untouched and
 * never inject node_lines alongside methods (the shapes are exclusive).
 */
export function flowToConfig(
  nodes: Node[],
  nonVisual: NonVisualConfig,
  edges: Edge[] = [],
): Record<string, unknown> {
  const config: Record<string, unknown> =
    typeof structuredClone === "function"
      ? structuredClone(nonVisual as Record<string, unknown>)
      : JSON.parse(JSON.stringify(nonVisual));

  const newNodeLines = flowToNodeLines(nodes, edges);

  const asp = (config.algo_search_space as Record<string, unknown>) ?? {};
  const pipeline = (asp.pipeline as Record<string, unknown>) ?? {};
  if (!pipeline.methods) {
    pipeline.node_lines = newNodeLines;
    asp.pipeline = pipeline;
    config.algo_search_space = asp;
  }

  return config;
}
