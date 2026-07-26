/**
 * PillTabs — thin wrapper over shadcn `<Tabs>` for cases where you want a
 * stand-alone segmented control without a corresponding `<TabsContent>`
 * pane (e.g. the control sits in a PageHeader and drives its own state).
 *
 * Adds support for an optional `actionKey` whose tab is decorated with a
 * `<Play>` glyph — used as a "run" action inside the tab strip.
 */
import { Play } from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";

export interface PillTabsProps {
  tabs: { key: string; label: string }[];
  active: string;
  onSelect: (k: any) => void;
  actionKey?: string;
}

export function PillTabs({
  tabs,
  active,
  onSelect,
  actionKey,
}: PillTabsProps) {
  return (
    <Tabs value={active} onValueChange={onSelect}>
      <TabsList>
        {tabs.map((tab) => {
          const isAction = tab.key === actionKey;
          return (
            <TabsTrigger
              key={tab.key}
              value={tab.key}
              className={cn(isAction && "gap-1.5")}
            >
              {isAction && (
                <Play
                  className={cn(
                    "size-3",
                    active === tab.key
                      ? "fill-primary text-primary"
                      : "text-muted-foreground",
                  )}
                />
              )}
              {tab.label}
            </TabsTrigger>
          );
        })}
      </TabsList>
    </Tabs>
  );
}
