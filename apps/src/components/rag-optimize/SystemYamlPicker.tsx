import { Check, Cpu, FileCode2, Server } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import type { SystemYamlInfo } from "@/api/ragProject";
import { cn } from "@/lib/utils";

interface SystemYamlPickerProps {
  items: SystemYamlInfo[];
  value: string;
  onValueChange: (value: string) => void;
  loading?: boolean;
  disabled?: boolean;
  className?: string;
}

export function SystemYamlPicker({
  items,
  value,
  onValueChange,
  loading = false,
  disabled = false,
  className,
}: SystemYamlPickerProps) {
  return (
    <div className={cn("w-full min-w-0 max-w-full overflow-hidden rounded-md border bg-background", className)}>
      <div className="flex min-w-0 items-center justify-between gap-3 border-b bg-muted/30 px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <FileCode2 className="size-4 text-muted-foreground" />
          <span className="truncate text-sm font-medium">Available YAMLs</span>
        </div>
        <Badge variant="secondary" className="h-5 shrink-0 px-1.5 text-[10px]">
          {loading ? "Loading" : `${items.length} files`}
        </Badge>
      </div>

      <ScrollArea className="h-[240px] w-full min-w-0 overflow-hidden">
        {loading ? (
          <div className="min-w-0 space-y-3 p-3">
            {[0, 1, 2].map((i) => (
              <div key={i} className="space-y-2">
                <Skeleton className="h-4 w-2/3" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-1/3" />
              </div>
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="flex h-[180px] items-center justify-center px-4 text-sm text-muted-foreground">
            No system YAMLs found
          </div>
        ) : (
          <div className="min-w-0 divide-y overflow-hidden">
            {items.map((item) => {
              const selected = item.path === value;
              const gpu = item.summary.gpu ?? "gpu";
              const cpu = item.summary.cpu ?? "cpu";
              const performanceSource = item.summary.performance_source ?? "cost_model";

              return (
                <button
                  key={item.path}
                  type="button"
                  disabled={disabled}
                  onClick={() => onValueChange(item.path)}
                  className={cn(
                    "flex w-full min-w-0 max-w-full items-start gap-3 overflow-hidden px-3 py-2.5 text-left transition-colors",
                    "hover:bg-accent/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    disabled && "cursor-not-allowed opacity-60",
                    selected && "bg-primary/5 hover:bg-primary/10",
                  )}
                >
                  <span
                    className={cn(
                      "mt-0.5 flex size-4 shrink-0 items-center justify-center rounded-full border",
                      selected
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-muted-foreground/30 text-transparent",
                    )}
                  >
                    <Check className="size-3" />
                  </span>
                  <span className="min-w-0 flex-1 overflow-hidden">
                    <span className="flex min-w-0 max-w-full items-center overflow-hidden">
                      <span className="min-w-0 flex-1 truncate font-mono text-xs font-medium text-foreground">
                        {item.filename || item.path}
                      </span>
                    </span>
                    <span className="mt-1 block max-w-full overflow-hidden break-all font-mono text-[11px] leading-4 text-muted-foreground">
                      {item.path}
                    </span>
                    <span className="mt-2 flex max-w-full flex-wrap gap-1.5 overflow-hidden">
                      <Badge variant="outline" className="h-5 max-w-full rounded-md px-1.5 text-[10px] font-normal">
                        <Server className="size-3" />
                        <span className="truncate">{gpu}</span>
                      </Badge>
                      <Badge variant="outline" className="h-5 max-w-full rounded-md px-1.5 text-[10px] font-normal">
                        <Cpu className="size-3" />
                        <span className="truncate">{cpu}</span>
                      </Badge>
                      <Badge variant="secondary" className="h-5 max-w-full rounded-md px-1.5 text-[10px] font-normal">
                        <span className="truncate">{performanceSource}</span>
                      </Badge>
                    </span>
                  </span>
                </button>
              );
            })}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
