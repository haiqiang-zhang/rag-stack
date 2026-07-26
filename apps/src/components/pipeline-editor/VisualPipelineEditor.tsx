import React, { useState, useCallback, useMemo, useRef, useEffect } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  BackgroundVariant,
  Panel,
  MarkerType,
} from "@xyflow/react";
import type { Node, Edge } from "@xyflow/react";
// xyflow CSS is loaded globally via apps-next/src/index.css

import { PipelineNodeComponent } from "./nodes/PipelineNodeComponent";
// VectorDB is not rendered as nodes — it's in nonVisual config, selectable via dropdown in modules
import { NodeLineGroup } from "./nodes/NodeLineGroup";
import { NodeInspector } from "./panels/NodeInspector";
import type { NodeTypeRegistry, PipelineNodeData, NodeLineLabelData } from "./types";
import { NODE_TYPE_COLORS } from "./constants";
import { sequenceEdges, moveInSequence } from "./hooks/useFlowToYaml";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const nodeTypes = {
  pipelineNode: PipelineNodeComponent,
  nodeLineGroup: NodeLineGroup,
};

const defaultEdgeOptions = {
  type: "smoothstep" as const,
  interactionWidth: 40,
  // Directional arrow at the target end so flow direction is unambiguous.
  // `ArrowClosed` is the filled triangle marker shipped by xyflow.
  markerEnd: { type: MarkerType.ArrowClosed, width: 18, height: 18 },
};

interface Props {
  initialNodes: Node[];
  initialEdges: Edge[];
  registry: NodeTypeRegistry;
  vectordbNames: string[];
  onFlowChange: (nodes: Node[], edges: Edge[]) => void;
}

export function VisualPipelineEditor({
  initialNodes,
  initialEdges,
  registry,
  vectordbNames,
  onFlowChange,
}: Props) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  // The canvas displays the serialized sequential order. Reordering is explicit.
  const edges = useMemo(() => sequenceEdges(nodes), [nodes]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const nextCreatedId = useRef(0);

  // Track whether user has made changes (skip initial render)
  const initialRender = useRef(true);
  useEffect(() => {
    if (initialRender.current) {
      initialRender.current = false;
      return;
    }
    onFlowChange(nodes, edges);
  }, [nodes, edges]); // deliberately exclude onFlowChange to avoid loops

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNodeId(node.id);
  }, []);

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  const handleUpdateNode = useCallback(
    (id: string, data: any) => {
      setNodes((nds) => {
        const target = nds.find((n) => n.id === id);
        // If renaming a group, also update nodeLineName on all child nodes
        if (target?.type === "nodeLineGroup" && data.nodeLineName !== (target.data as any).nodeLineName) {
          const oldName = (target.data as any).nodeLineName;
          return nds.map((n) => {
            if (n.id === id) return { ...n, data };
            if (n.type === "pipelineNode" && (n.data as any).nodeLineName === oldName) {
              return { ...n, data: { ...n.data, nodeLineName: data.nodeLineName } };
            }
            return n;
          });
        }
        return nds.map((n) => (n.id === id ? { ...n, data } : n));
      });
    },
    [setNodes]
  );

  const selectedNode = nodes.find((n) => n.id === selectedNodeId) ?? null;

  // Determine which node line is selected (from group or child node)
  const selectedLineName = useMemo(() => {
    if (!selectedNode) return null;
    if (selectedNode.type === "nodeLineGroup") {
      return (selectedNode.data as unknown as NodeLineLabelData).nodeLineName;
    }
    if (selectedNode.type === "pipelineNode") {
      return (selectedNode.data as unknown as PipelineNodeData).nodeLineName;
    }
    return null;
  }, [selectedNode]);

  // Layout constants (match useYamlToFlow)
  const PIPELINE_NODE_W = 260;
  const NODE_X_GAP = PIPELINE_NODE_W + 60;
  const NODE_PAD_LEFT = 24;
  const NODE_PAD_TOP = 70;
  const GROUP_PAD_RIGHT = 24;
  const GROUP_PAD_BOTTOM = 24;

  // Add node line (as a group)
  const handleAddNodeLine = useCallback(() => {
    const groups = nodes.filter((n) => n.type === "nodeLineGroup");
    const lineIdx = Math.max(-1, ...groups.map((n) => Number(n.data.nodeLineIndex))) + 1;
    const lineName = `node_line_${lineIdx}`;
    const groupId = `group-added-${++nextCreatedId.current}`;
    // Place below the last group
    const lastGroup = groups.sort((a, b) => a.position.y - b.position.y).at(-1);
    const groupY = lastGroup
      ? lastGroup.position.y + ((lastGroup.style?.height as number) || 180) + 50
      : 200;

    const groupWidth = NODE_PAD_LEFT + PIPELINE_NODE_W + GROUP_PAD_RIGHT;
    const groupHeight = NODE_PAD_TOP + 100 + GROUP_PAD_BOTTOM;

    const groupNode: Node = {
      id: groupId,
      type: "nodeLineGroup",
      position: { x: 40, y: groupY },
      data: { nodeLineName: lineName, nodeLineIndex: lineIdx } satisfies NodeLineLabelData,
      style: { width: groupWidth, height: groupHeight },
    };

    const firstNode: Node = {
      id: `node-added-${++nextCreatedId.current}`,
      type: "pipelineNode",
      position: { x: NODE_PAD_LEFT, y: NODE_PAD_TOP },
      parentId: groupId,
      extent: "parent" as const,
      data: {
        nodeType: "semantic_retrieval",
        nodeLineName: lineName,
        nodeIndex: 0,
        modules: [{ component: "vectordb" }],
        extraParams: {},
      } satisfies PipelineNodeData,
    };

    setNodes((nds) => [...nds, groupNode, firstNode]);
  }, [nodes, setNodes]);

  // Add node to a line
  const handleAddNodeToLine = useCallback(
    (lineName: string, nodeType: string) => {
      const lineNodes = nodes.filter(
        (n) => n.type === "pipelineNode" && (n.data as unknown as PipelineNodeData).nodeLineName === lineName
      );
      const lastNode = lineNodes.sort((a, b) => Number(a.data.nodeIndex) - Number(b.data.nodeIndex)).at(-1);
      const groupNode = nodes.find(
        (n) => n.type === "nodeLineGroup" && (n.data as unknown as NodeLineLabelData).nodeLineName === lineName
      );
      const lineIdx = groupNode ? (groupNode.data as unknown as NodeLineLabelData).nodeLineIndex : 0;
      const nodeIdx = Math.max(-1, ...lineNodes.map((n) => Number(n.data.nodeIndex))) + 1;
      const x = lastNode ? lastNode.position.x + NODE_X_GAP : NODE_PAD_LEFT;
      const y = NODE_PAD_TOP;

      const newId = `node-added-${++nextCreatedId.current}`;
      const groupId = groupNode?.id || `group-${lineIdx}`;
      const newNode: Node = {
        id: newId,
        type: "pipelineNode",
        position: { x, y },
        parentId: groupId,
        extent: "parent" as const,
        data: {
          nodeType,
          nodeLineName: lineName,
          nodeIndex: nodeIdx,
          modules: [],
          extraParams: {},
        } satisfies PipelineNodeData,
      };

      // Expand group width to fit new node
      if (groupNode) {
        const newWidth = x + PIPELINE_NODE_W + GROUP_PAD_RIGHT;
        setNodes((nds) =>
          nds.map((n) =>
            n.id === groupId ? { ...n, style: { ...n.style, width: Math.max((n.style?.width as number) || 400, newWidth) } } : n
          )
        );
      }

      setNodes((nds) => [...nds, newNode]);
    },
    [nodes, setNodes]
  );

  // Recalculate a group's size based on its current children
  const resizeGroup = useCallback((nds: Node[], groupId: string): Node[] => {
    const children = nds.filter((n) => n.type === "pipelineNode" && n.parentId === groupId);
    if (children.length === 0) {
      // Empty group: minimum size
      return nds.map((n) => n.id === groupId ? { ...n, style: { ...n.style, width: 200, height: 120 } } : n);
    }
    let maxRight = 0, maxBottom = 0;
    for (const c of children) {
      const r = c.position.x + PIPELINE_NODE_W;
      // Estimate node height from module count
      const mods = ((c.data as any).modules || []).length;
      const h = 36 + Math.max(1, mods) * 24 + 18 + ((c.data as any).strategy?.metrics?.length ? 26 : 0);
      const b = c.position.y + h;
      if (r > maxRight) maxRight = r;
      if (b > maxBottom) maxBottom = b;
    }
    const newWidth = maxRight + NODE_PAD_LEFT;
    const newHeight = maxBottom + NODE_PAD_LEFT; // reuse pad as bottom pad
    return nds.map((n) => n.id === groupId
      ? { ...n, style: { ...n.style, width: Math.max(200, newWidth), height: Math.max(120, newHeight) } }
      : n
    );
  }, []);

  // Delete selected node or selected edges
  const handleDeleteSelected = useCallback(() => {
    // Delete selected node
    if (selectedNodeId) {
      // Block deleting a node line group that still has child nodes
      const selectedNode = nodes.find((n) => n.id === selectedNodeId);
      if (selectedNode?.type === "nodeLineGroup") {
        const children = nodes.filter((n) => n.parentId === selectedNodeId);
        if (children.length > 0) {
          setToast(`Remove all nodes from "${(selectedNode.data as any).nodeLineName}" before deleting the node line`);
          setTimeout(() => setToast(null), 3000);
          return;
        }
      }
      // If deleting a pipeline node, resize its parent group
      const parentGroupId = selectedNode?.parentId;
      setNodes((nds) => {
        let updated = nds.filter((n) => n.id !== selectedNodeId);
        if (parentGroupId) {
          updated = resizeGroup(updated, parentGroupId);
        }
        return updated;
      });
      setSelectedNodeId(null);
    }
  }, [selectedNodeId, nodes, setNodes, resizeGroup]);

  const hasSelection = !!selectedNodeId;

  // Get unique node line names
  const nodeLineNames = useMemo(() => {
    const groups = nodes.filter((n) => n.type === "nodeLineGroup");
    return groups.map((n) => (n.data as unknown as NodeLineLabelData).nodeLineName);
  }, [nodes]);

  return (
    <div style={{ display: "flex", width: "100%", height: "100%" }}>
      <div style={{ flex: 1, position: "relative" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={nodeTypes}
          defaultEdgeOptions={defaultEdgeOptions}
          edgesReconnectable={false}
          nodesConnectable={false}
          fitView
          fitViewOptions={{ padding: 0.2 }}
          proOptions={{ hideAttribution: true }}
          deleteKeyCode={["Delete", "Backspace"]}
          panOnScroll
          zoomOnScroll={false}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#E2E8F0" />
          <Controls
            showInteractive={false}
            style={{ borderRadius: 8, border: "1px solid #E2E8F0" }}
          />
          <MiniMap
            style={{ borderRadius: 8, border: "1px solid #E2E8F0" }}
            maskColor="rgba(15, 23, 42, 0.05)"
            nodeColor={(n) => {
              if (n.type === "vectordbNode") return "#CBD5E1";
              if (n.type === "pipelineNode") {
                const nt = (n.data as unknown as PipelineNodeData).nodeType;
                return NODE_TYPE_COLORS[nt]?.accent || "#0F766E";
              }
              return "transparent";
            }}
          />

          {/* Toolbar */}
          <Panel position="top-left">
            <div style={toolbarStyle}>
              <button onClick={handleAddNodeLine} disabled={Object.keys(registry).length === 0} style={toolbarBtnStyle}>
                + Node Line
              </button>
              {selectedLineName && (
                <Select
                  value=""
                  onValueChange={(v) => handleAddNodeToLine(selectedLineName, v)}
                >
                  <SelectTrigger className="h-8 min-w-[120px] text-xs">
                    <SelectValue placeholder="+ Add Node" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.keys(registry).map((nt) => (
                      <SelectItem key={nt} value={nt}>
                        {nt}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              {hasSelection && (
                <button
                  onClick={handleDeleteSelected}
                  style={{ ...toolbarBtnStyle, color: "#EF4444", borderColor: "#FECACA" }}
                >
                  Delete
                </button>
              )}
              <span style={{ fontSize: 10, color: "#64748B", alignSelf: "center", padding: "0 6px" }}>
                Sequential · drag to arrange; use inspector to reorder
              </span>
            </div>
          </Panel>
        </ReactFlow>
      </div>

      {/* Inspector panel */}
      {selectedNode && (
        <NodeInspector
          node={selectedNode}
          registry={registry}
          vectordbNames={vectordbNames}
          onUpdateNode={handleUpdateNode}
          onMove={(direction) => setNodes((nds) => moveInSequence(nds, selectedNode.id, direction))}
          onClose={() => setSelectedNodeId(null)}
        />
      )}

      {/* Toast notification */}
      {toast && (
        <div style={toastStyle}>
          {toast}
        </div>
      )}
    </div>
  );
}

const toolbarStyle: React.CSSProperties = {
  display: "flex",
  gap: 6,
  background: "#fff",
  borderRadius: 10,
  padding: 6,
  border: "1px solid #E2E8F0",
  boxShadow: "0 1px 4px rgba(15, 23, 42, 0.06)",
};

const toastStyle: React.CSSProperties = {
  position: "absolute",
  bottom: 20,
  left: "50%",
  transform: "translateX(-50%)",
  background: "#1E293B",
  color: "#fff",
  fontSize: 12,
  fontWeight: 500,
  padding: "8px 16px",
  borderRadius: 8,
  boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
  zIndex: 100,
  whiteSpace: "nowrap",
};

const toolbarBtnStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 600,
  color: "#0F766E",
  background: "#F0FDFA",
  border: "1px solid #99F6E4",
  borderRadius: 6,
  padding: "5px 10px",
  cursor: "pointer",
  userSelect: "none",
};
