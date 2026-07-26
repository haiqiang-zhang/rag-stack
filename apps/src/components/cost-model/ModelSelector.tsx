import React from "react";
import { Cpu, Database, Layers } from "lucide-react";
import type { CostModelInfo } from "@/types/costModel";
import { SideItem, SideItemDot } from "@/components/ui/SideItem";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { groupByFamily } from "./families";

interface Props {
  models: CostModelInfo[];
  selected: string | null;
  onSelect: (name: string) => void;
}

const ICONS = { Cpu, Database, Layers } as const;

/**
 * Grouped cost-model list. Visual hierarchy follows shadcn sidebar pattern:
 *   - Top: panel-level title + total count
 *   - Per family: small uppercase eyebrow + count badge + icon
 *   - Items: SideItem (button-like, bg-accent on active)
 */
export function ModelSelector({ models, selected, onSelect }: Props) {
  const sections = groupByFamily(models);

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-sidebar-border px-3 py-3">
        <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Cost Models
        </p>
        <p className="mt-0.5 text-xs text-muted-foreground/80">
          {models.length} total · {sections.length}{" "}
          categor{sections.length === 1 ? "y" : "ies"}
        </p>
      </div>

      <div className="flex-1 overflow-auto pt-1 pb-2">
        {sections.map((section, sIdx) => {
          const Icon = ICONS[section.iconName];
          return (
            <div
              key={section.key}
              className={cn(sIdx > 0 && "mt-3")}
            >
              {/* Family eyebrow */}
              <div className="mb-1 flex items-center justify-between px-3 py-1">
                <div className="flex items-center gap-1.5">
                  <Icon className="size-3 text-muted-foreground/70" />
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/80">
                    {section.label}
                  </span>
                </div>
                <Badge variant="secondary" className="h-4 px-1.5 text-[10px]">
                  {section.items.length}
                </Badge>
              </div>

              {/* Items */}
              {section.items.map((m) => {
                const active = selected === m.name;
                return (
                  <SideItem
                    key={m.name}
                    active={active}
                    onClick={() => onSelect(m.name)}
                  >
                    <SideItemDot active={active} />
                    <div className="flex min-w-0 flex-1 flex-col">
                      <span
                        className={cn(
                          "truncate text-xs",
                          active
                            ? "font-semibold text-foreground"
                            : "font-medium",
                        )}
                      >
                        {m.name}
                      </span>
                      <span className="truncate text-[10px] text-muted-foreground/70">
                        {m.description || m.class_name}
                      </span>
                    </div>
                  </SideItem>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
