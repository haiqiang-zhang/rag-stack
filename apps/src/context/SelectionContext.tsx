import React, { createContext, useContext, useState, useCallback } from "react";

interface SelectionState {
  selectedEvalId: number | null;
  setSelectedEvalId: (id: number | null) => void;
}

const SelectionContext = createContext<SelectionState>({
  selectedEvalId: null,
  setSelectedEvalId: () => {},
});

export function SelectionProvider({ children }: { children: React.ReactNode }) {
  const [selectedEvalId, setSelectedEvalIdRaw] = useState<number | null>(null);

  const setSelectedEvalId = useCallback((id: number | null) => {
    setSelectedEvalIdRaw(id);
  }, []);

  return (
    <SelectionContext.Provider value={{ selectedEvalId, setSelectedEvalId }}>
      {children}
    </SelectionContext.Provider>
  );
}

export function useSelection() {
  return useContext(SelectionContext);
}
