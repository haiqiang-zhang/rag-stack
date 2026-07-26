import { useMemo } from "react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export interface DropdownOption {
  key: string;
  label: string;
  group?: string;
  icon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

interface DropdownProps {
  label?: string;
  options: DropdownOption[];
  selectedKey: string;
  onSelect: (key: string) => void;
  placeholder?: string;
  triggerIcon?: React.ReactNode;
}

export function Dropdown({
  label,
  options,
  selectedKey,
  onSelect,
  placeholder = "Select...",
  // TODO(migrate): triggerIcon was unused in the legacy Tamagui implementation
  // (kept as a public prop for API stability). Preserve the parameter so
  // callers don't break, but it's still not rendered anywhere.
  triggerIcon: _triggerIcon,
}: DropdownProps) {
  const hasGroups = options.some((o) => o.group);
  const groups = useMemo(() => {
    if (!hasGroups) return [["", options] as [string, DropdownOption[]]];
    return Object.entries(
      options.reduce<Record<string, DropdownOption[]>>((acc, opt) => {
        const g = opt.group || "";
        (acc[g] = acc[g] || []).push(opt);
        return acc;
      }, {})
    ) as [string, DropdownOption[]][];
  }, [options, hasGroups]);

  return (
    <div className="flex flex-col">
      {label && (
        <span className="mb-1.5 ml-0.5 text-[11px] font-medium tracking-wide text-muted-foreground">
          {label}
        </span>
      )}
      <Select value={selectedKey} onValueChange={onSelect}>
        <SelectTrigger className="w-full justify-between rounded-[10px] px-3.5 py-[9px] text-[13px] font-medium hover:border-primary focus-visible:border-primary">
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent className="min-w-[180px] max-h-[300px]">
          {groups.map(([group, items], gi) => (
            <SelectGroup key={group || gi}>
              {group !== "" && (
                <SelectLabel className="px-3.5 pt-2 pb-1 text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                  {group}
                </SelectLabel>
              )}
              {items.map((item) => (
                <SelectItem
                  key={item.key}
                  value={item.key}
                  className={cn(
                    "rounded-md px-3.5 py-2 text-[13px]",
                    item.key === selectedKey
                      ? "font-semibold text-primary"
                      : "font-normal",
                  )}
                >
                  {item.label}
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
