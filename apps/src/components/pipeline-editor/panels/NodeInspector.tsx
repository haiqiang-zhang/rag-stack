import React from "react";
import type { Node } from "@xyflow/react";
import type {
  PipelineNodeData,
  NodeLineLabelData,
  ModuleConfig,
  NodeTypeRegistry,
  VectorDbNodeData,
} from "../types";
import { NODE_TYPE_COLORS, VECTORDB_COLOR } from "../constants";
import { ModuleParamForm } from "./ModuleParamForm";
import { ConfigValueInput } from "./ConfigValueInput";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface Props {
  node: Node;
  registry: NodeTypeRegistry;
  vectordbNames?: string[];
  onUpdateNode: (id: string, data: any) => void;
  onClose: () => void;
  onMove?: (direction: -1 | 1) => void;
}

export function NodeInspector({ node, registry, vectordbNames, onUpdateNode, onClose, onMove }: Props) {
  if (node.type === "pipelineNode") {
    return (
      <PipelineNodeInspector
        node={node}
        registry={registry}
        vectordbNames={vectordbNames}
        onUpdateNode={onUpdateNode}
        onClose={onClose}
        onMove={onMove}
      />
    );
  }

  if (node.type === "nodeLineGroup") {
    return (
      <NodeLineGroupInspector
        node={node}
        onUpdateNode={onUpdateNode}
        onClose={onClose}
        onMove={onMove}
      />
    );
  }

  return null;
}

function PipelineNodeInspector({ node, registry, vectordbNames, onUpdateNode, onClose, onMove }: Props) {
  const d = node.data as unknown as PipelineNodeData;
  const colors = NODE_TYPE_COLORS[d.nodeType] || NODE_TYPE_COLORS.generator;
  const stageRegistry = registry[d.nodeType]
    ?? Object.values(registry).find((entry) => entry.aliases?.includes(d.nodeType));
  const availableModules = stageRegistry?.modules || [];

  const updateModules = (modules: ModuleConfig[]) => {
    onUpdateNode(node.id, { ...d, modules });
  };

  const handleModuleChange = (index: number, updated: ModuleConfig) => {
    const modules = [...d.modules];
    modules[index] = updated;
    updateModules(modules);
  };

  const handleRemoveModule = (index: number) => {
    updateModules(d.modules.filter((_, i) => i !== index));
  };

  const handleAddModule = (moduleType: string) => {
    const schema = stageRegistry?.component_parameters?.[moduleType] ?? {};
    const defaults = Object.fromEntries(Object.entries(schema)
      .filter(([, field]) => field.default !== undefined)
      .map(([key, field]) => [key, structuredClone(field.default)]));
    updateModules([...d.modules, { ...defaults, component: moduleType }]);
  };

  // Modules not yet added
  const unusedModules = availableModules.filter(
    (m) => !d.modules.some((mod) => mod.component === m)
  );

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <div>
          <span style={{ ...styles.badge, background: colors.bg, color: colors.accent, border: `1px solid ${colors.border}` }}>
            {d.nodeType.replace(/_/g, " ")}
          </span>
        </div>
        <button onClick={onClose} style={styles.closeBtn}>x</button>
      </div>

      <OrderControls index={d.nodeIndex} onMove={onMove} />

      {/* Optional toggle */}
      <div style={{ padding: "8px 16px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <label style={styles.label}>Optional Node</label>
        <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", userSelect: "none" }}>
          <input
            type="checkbox"
            checked={d.optional === true}
            onChange={(e) => onUpdateNode(node.id, { ...d, optional: e.target.checked })}
            style={{ accentColor: "#8B5CF6", width: 14, height: 14, cursor: "pointer" }}
          />
          <span style={{ fontSize: 11, color: d.optional ? "#8B5CF6" : "#94A3B8", fontWeight: 500 }}>
            {d.optional ? "Yes" : "No"}
          </span>
        </label>
      </div>

      {/* Top K */}
      <div style={{ padding: "8px 16px" }}>
        <label style={styles.label}>top_k</label>
        <ConfigValueInput
          label="top_k"
          value={d.topK}
          onChange={(topK) => onUpdateNode(node.id, { ...d, topK })}
          validate={(v) => v === undefined || typeof v === "number"
            || (Array.isArray(v) ? v.every((n) => typeof n === "number")
              : v !== null && typeof v === "object")}
          placeholder={'10, [5, 10], or {"range": [5, 20]}'}
          style={styles.input}
        />
      </div>

      {/* Modules */}
      <div style={{ padding: "0 16px 16px" }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "#94A3B8", letterSpacing: 0.8, marginBottom: 8 }}>
          MODULES ({d.modules.length})
        </div>

        {d.modules.map((mod, i) => (
          <ModuleParamForm
            key={`${mod.component}-${i}`}
            module={mod}
            onChange={(updated) => handleModuleChange(i, updated)}
            onRemove={() => handleRemoveModule(i)}
            vectordbNames={vectordbNames}
            schema={stageRegistry?.component_parameters?.[mod.component]}
            capabilities={stageRegistry?.component_capabilities?.[mod.component]}
          />
        ))}

        {unusedModules.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <Select value="" onValueChange={(v) => handleAddModule(v)}>
              <SelectTrigger className="h-8 w-full text-xs">
                <SelectValue placeholder="+ Add module..." />
              </SelectTrigger>
              <SelectContent>
                {unusedModules.map((m) => (
                  <SelectItem key={m} value={m} disabled={stageRegistry?.component_capabilities?.[m]?.execution_supported === false}>
                    {m}{stageRegistry?.component_capabilities?.[m]?.execution_supported === false ? " (unavailable)" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}
      </div>
    </div>
  );
}

// TODO(migrate): VectorDbInspector is unreferenced from the NodeInspector dispatch
// (no `node.type === "vectordbNode"` branch). Preserved verbatim from the original
// file for parity; consider removing if confirmed dead.
function VectorDbInspector({
  node,
  onUpdateNode,
  onClose,
}: Omit<Props, "registry">) {
  const d = node.data as unknown as VectorDbNodeData;

  const handleParamChange = (key: string, raw: string) => {
    let value: unknown = raw;
    try {
      const parsed = JSON.parse(raw);
      if (typeof parsed !== "string") value = parsed;
    } catch { /* keep string */ }

    const params = { ...d.params, [key]: value };
    onUpdateNode(node.id, { ...d, params });
  };

  const handleNameChange = (name: string) => {
    onUpdateNode(node.id, { ...d, name });
  };

  const handleDbTypeChange = (dbType: string) => {
    onUpdateNode(node.id, { ...d, dbType });
  };

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={{ ...styles.badge, background: VECTORDB_COLOR.bg, color: VECTORDB_COLOR.accent, border: `1px solid ${VECTORDB_COLOR.border}` }}>
          vectordb
        </span>
        <button onClick={onClose} style={styles.closeBtn}>x</button>
      </div>

      <div style={{ padding: "8px 16px 16px" }}>
        <label style={styles.label}>name</label>
        <input
          type="text"
          value={d.name}
          onChange={(e) => handleNameChange(e.target.value)}
          style={styles.input}
        />

        <label style={{ ...styles.label, marginTop: 8 }}>db_type</label>
        <input
          type="text"
          value={d.dbType}
          onChange={(e) => handleDbTypeChange(e.target.value)}
          style={styles.input}
        />

        <div style={{ fontSize: 10, fontWeight: 700, color: "#94A3B8", letterSpacing: 0.8, margin: "12px 0 6px" }}>
          PARAMETERS
        </div>

        {Object.entries(d.params).map(([key, value]) => (
          <div key={key} style={{ marginBottom: 4, display: "flex", alignItems: "center", gap: 6 }}>
            <label style={{ fontSize: 10, fontWeight: 500, color: "#64748B", minWidth: 100 }}>
              {key}
            </label>
            <input
              type="text"
              value={typeof value === "object" ? JSON.stringify(value) : String(value ?? "")}
              onChange={(e) => handleParamChange(key, e.target.value)}
              style={{ ...styles.input, flex: 1 }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function NodeLineGroupInspector({
  node,
  onUpdateNode,
  onClose,
  onMove,
}: Omit<Props, "registry">) {
  const d = node.data as unknown as NodeLineLabelData;

  const handleNameChange = (name: string) => {
    onUpdateNode(node.id, { ...d, nodeLineName: name });
  };

  return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <span style={{ ...styles.badge, background: "#F1F5F9", color: "#64748B", border: "1px solid #CBD5E1" }}>
          node line
        </span>
        <button onClick={onClose} style={styles.closeBtn}>x</button>
      </div>

      <OrderControls index={d.nodeLineIndex} onMove={onMove} />

      <div style={{ padding: "12px 16px" }}>
        <label style={styles.label}>Node Line Name</label>
        <input
          type="text"
          value={d.nodeLineName}
          onChange={(e) => handleNameChange(e.target.value)}
          style={styles.input}
        />

        <div style={{ fontSize: 10, color: "#94A3B8", marginTop: 12, lineHeight: 1.5 }}>
          A node line is a sequential group of processing nodes.
          Execution follows the numbered order. Moving a node on the canvas
          changes its layout; the order controls change execution.
        </div>
      </div>
    </div>
  );
}

function OrderControls({ index, onMove }: { index: number; onMove?: (direction: -1 | 1) => void }) {
  if (!onMove) return null;
  return (
    <div style={{ padding: "8px 16px", display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontSize: 11, color: "#64748B" }}>Order {index + 1}</span>
      <button type="button" onClick={() => onMove(-1)} style={styles.closeBtn} title="Execute earlier">↑</button>
      <button type="button" onClick={() => onMove(1)} style={styles.closeBtn} title="Execute later">↓</button>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 320,
    background: "#fff",
    borderLeft: "1px solid #E2E8F0",
    height: "100%",
    overflowY: "auto",
    fontSize: 12,
  },
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 16px",
    borderBottom: "1px solid #E2E8F0",
  },
  closeBtn: {
    width: 24,
    height: 24,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    border: "1px solid #E2E8F0",
    borderRadius: 6,
    background: "#F8FAFC",
    cursor: "pointer",
    fontSize: 12,
    color: "#64748B",
  },
  badge: {
    fontSize: 11,
    fontWeight: 600,
    padding: "3px 10px",
    borderRadius: 6,
    textTransform: "uppercase" as const,
    letterSpacing: 0.5,
  },
  label: {
    display: "block",
    fontSize: 10,
    fontWeight: 500,
    color: "#64748B",
    marginBottom: 4,
  },
  input: {
    width: "100%",
    fontSize: 12,
    padding: "6px 10px",
    border: "1px solid #E2E8F0",
    borderRadius: 6,
    outline: "none",
    background: "#F8FAFC",
    color: "#1E293B",
    fontFamily: "monospace",
    boxSizing: "border-box" as const,
  },
  select: {
    width: "100%",
    fontSize: 11,
    padding: "6px 10px",
    border: "1px solid #E2E8F0",
    borderRadius: 6,
    background: "#F8FAFC",
    color: "#64748B",
    cursor: "pointer",
    outline: "none",
  },
};
