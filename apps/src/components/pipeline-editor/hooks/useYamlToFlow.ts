import { useMemo } from "react";
import type { Node, Edge } from "@xyflow/react";
import type {
  PipelineNodeData,
  NodeLineLabelData,
  NonVisualConfig,
} from "../types";

export interface FlowState {
  nodes: Node[];
  edges: Edge[];
  nonVisual: NonVisualConfig;
}

// ── Layout constants ─────────────────────────────────────────────────
const GROUP_X = 40;
const GROUP_Y_START = 200;
const GROUP_Y_MARGIN = 50;

const PIPELINE_NODE_W = 260;
const NODE_X_GAP = PIPELINE_NODE_W + 60;

// Padding around child nodes inside the group
const PAD_TOP = 70;   // group padding(10) + label tag(~18) + gap(~12) + breathing room
const PAD_LEFT = 24;
const PAD_RIGHT = 24;
const PAD_BOTTOM = 24;

/** Estimate rendered pixel height for one pipeline node. */
function estimateNodeHeight(modules: any[], strategy: any): number {
  let h = 36; // header
  const modCount = modules?.length || 0;
  if (modCount === 0) {
    h += 24;
  } else {
    h += modCount * 24 + (modCount - 1) * 4;
  }
  h += 18; // padding
  if (strategy?.metrics?.length) h += 26;
  return h;
}

/**
 * Convert a raw `node_lines` array into xyflow nodes + edges. This is the
 * shared graph builder used by BOTH the static_gt pipeline (one node_lines at
 * algo_search_space.pipeline.node_lines) and each FlashRAG method (its own
 * methods[i].node_lines) — so the same visual editor drives either.
 */
export function nodeLinesToFlow(nodeLines: any[]): { nodes: Node[]; edges: Edge[] } {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  const lineLastNodes: string[] = [];
  let nextGroupY = GROUP_Y_START;

  (nodeLines || []).forEach((nl, lineIdx) => {
    const lineName = nl.node_line_name || `line_${lineIdx}`;
    const lineNodes = (nl.nodes as any[]) || [];
    const groupId = `group-${lineIdx}`;

    // Step 1: Create child nodes at local positions (relative to group)
    interface ChildInfo { id: string; x: number; y: number; w: number; h: number; node: Node }
    const children: ChildInfo[] = [];
    let prevNodeId: string | null = null;

    lineNodes.forEach((node, nodeIdx) => {
      const nodeId = `node-${lineIdx}-${nodeIdx}`;
      const { stage, modules, strategy, top_k, optional, ...extraParams } = node;
      const mods = modules || [];
      const nodeH = estimateNodeHeight(mods, strategy);
      const localX = PAD_LEFT + nodeIdx * NODE_X_GAP;
      const localY = PAD_TOP;

      const childNode: Node = {
        id: nodeId,
        type: "pipelineNode",
        position: { x: localX, y: localY },
        parentId: groupId,
        extent: "parent" as const,
        data: {
          nodeType: stage,
          nodeLineName: lineName,
          nodeIndex: nodeIdx,
          modules: mods,
          topK: top_k,
          optional: optional === true,
          strategy,
          extraParams,
        } satisfies PipelineNodeData,
      };

      children.push({ id: nodeId, x: localX, y: localY, w: PIPELINE_NODE_W, h: nodeH, node: childNode });

      if (prevNodeId) {
        edges.push({ id: `e-${lineIdx}-${nodeIdx}`, source: prevNodeId, target: nodeId });
      }
      prevNodeId = nodeId;
    });

    // Step 2: Compute bounding box of all children
    let minX = Infinity, minY = Infinity, maxRight = 0, maxBottom = 0;
    for (const c of children) {
      if (c.x < minX) minX = c.x;
      if (c.y < minY) minY = c.y;
      const r = c.x + c.w;
      const b = c.y + c.h;
      if (r > maxRight) maxRight = r;
      if (b > maxBottom) maxBottom = b;
    }

    // Fallback for empty node lines
    if (children.length === 0) {
      minX = PAD_LEFT; minY = PAD_TOP;
      maxRight = PAD_LEFT + PIPELINE_NODE_W;
      maxBottom = PAD_TOP + 80;
    }

    // Step 3: Size group = bounding box + padding
    const groupWidth = maxRight + PAD_RIGHT;
    const groupHeight = maxBottom + PAD_BOTTOM;

    nodes.push({
      id: groupId,
      type: "nodeLineGroup",
      position: { x: GROUP_X, y: nextGroupY },
      data: { nodeLineName: lineName, nodeLineIndex: lineIdx } satisfies NodeLineLabelData,
      style: { width: groupWidth, height: groupHeight },
    });

    // Step 4: Add children to nodes array
    for (const c of children) {
      nodes.push(c.node);
    }

    nextGroupY += groupHeight + GROUP_Y_MARGIN;

    if (prevNodeId) lineLastNodes.push(prevNodeId);
  });

  // Inter-line edges (initial): connect each consecutive pair from the
  // YAML's array order so the loaded graph reflects the source order.
  for (let i = 0; i < lineLastNodes.length - 1; i++) {
    const firstOfNext = nodes.find(
      (n) =>
        n.type === "pipelineNode" &&
        n.parentId === `group-${i + 1}` &&
        (n.data as any).nodeIndex === 0,
    );
    if (firstOfNext) {
      edges.push({
        id: `e-line-${i}-to-${i + 1}`,
        source: lineLastNodes[i],
        target: firstOfNext.id,
      });
    }
  }

  return { nodes, edges };
}

export function yamlObjToFlow(config: Record<string, unknown>): FlowState {
  // nonVisual = the FULL config base (all 7 sections). The visual graph only
  // owns algo_search_space.pipeline.node_lines; on save we overwrite that one
  // path and keep everything else (incl. algo_search_space.corpus and a
  // flashrag pipeline.methods) intact.
  const nonVisual = (typeof structuredClone === "function"
    ? structuredClone(config)
    : JSON.parse(JSON.stringify(config))) as NonVisualConfig;

  const pipeline = ((config.algo_search_space as any)?.pipeline) ?? {};
  const { nodes, edges } = nodeLinesToFlow((pipeline.node_lines as any[]) || []);
  return { nodes, edges, nonVisual };
}

export function useYamlToFlow(config: Record<string, unknown> | null): FlowState {
  return useMemo(() => {
    if (!config) return { nodes: [], edges: [], nonVisual: {} };
    return yamlObjToFlow(config);
  }, [config]);
}
