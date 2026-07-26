import React, { createContext, useCallback, useContext, useMemo, useState } from "react";

interface StageInteraction {
  hoveredStage: string | null;
  setHoveredStage: (s: string | null) => void;
  jumpToStage: string | null;
  requestJump: (s: string) => void;
  consumeJump: () => void;
}

const Ctx = createContext<StageInteraction>({
  hoveredStage: null,
  setHoveredStage: () => {},
  jumpToStage: null,
  requestJump: () => {},
  consumeJump: () => {},
});

export function StageInteractionProvider({ children }: { children: React.ReactNode }) {
  const [hoveredStage, setHoveredStage] = useState<string | null>(null);
  const [jumpToStage, setJumpToStage] = useState<string | null>(null);

  const requestJump = useCallback((s: string) => setJumpToStage(s), []);
  const consumeJump = useCallback(() => setJumpToStage(null), []);

  const value = useMemo(
    () => ({ hoveredStage, setHoveredStage, jumpToStage, requestJump, consumeJump }),
    [hoveredStage, jumpToStage, requestJump, consumeJump],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStageInteraction() {
  return useContext(Ctx);
}
