import { useMemo } from "react";
import { Dropdown, type DropdownOption } from "@/components/selectors/Dropdown";
import type { AxisOption } from "@/types";

interface AxisSelectorProps {
  label: string;
  options: AxisOption[];
  selectedKey: string;
  onSelect: (key: string) => void;
}

/**
 * Axis selector — thin wrapper around the shared Dropdown.
 */
export function AxisSelector({
  label,
  options,
  selectedKey,
  onSelect,
}: AxisSelectorProps) {
  const dropdownOptions: DropdownOption[] = useMemo(
    () => options.map((o) => ({ key: o.key, label: o.label, group: o.group })),
    [options]
  );

  return (
    <Dropdown
      label={label}
      options={dropdownOptions}
      selectedKey={selectedKey}
      onSelect={onSelect}
    />
  );
}
