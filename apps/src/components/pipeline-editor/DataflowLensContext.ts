/**
 * DataflowLensContext — lets the DataflowBar project the react branch's
 * gating onto the pipeline canvas without touching react-flow node state.
 *
 * When `lens === "react"`, PipelineNodeComponent ghosts every stage the react
 * engine strips from a trial's node_lines (stage ∉ reactActiveStages) and
 * rings the stages the loop actually drives. `lens === null` renders the
 * graph normally (the sequential truth).
 */
import { createContext, useContext } from "react";

export interface DataflowLens {
  lens: "react" | null;
  /** Stages the react loop runs each round (backend REACT_ACTIVE_STAGES). */
  reactActiveStages: ReadonlySet<string>;
}

const NO_LENS: DataflowLens = { lens: null, reactActiveStages: new Set() };

export const DataflowLensContext = createContext<DataflowLens>(NO_LENS);

export function useDataflowLens(): DataflowLens {
  return useContext(DataflowLensContext);
}
