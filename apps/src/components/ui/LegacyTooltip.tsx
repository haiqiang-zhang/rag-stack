/**
 * Legacy Tooltip — hover tooltip wrapper around Radix `<Tooltip>`.
 *
 * NOTE on filename: this file is `LegacyTooltip.tsx` (not `Tooltip.tsx`) to
 * avoid a case-only filename collision with shadcn's `tooltip.tsx` that
 * TypeScript rejects under `forceConsistentCasingInFileNames`. The exported
 * symbol (`Tooltip`) is unchanged, and `./index.ts` re-exports it under
 * that name so callers never see the rename.
 *
 * Imports `radix-ui` directly (NOT `@/components/ui/tooltip`) so the legacy
 * variant stays self-contained.
 *
 * Public API kept compatible with the legacy custom Tamagui tooltip:
 * `content`, `children`, `placement` ("top" | "bottom" | "left" | "right"),
 * `delay`, `offset`.
 */
import type { ReactNode } from "react";
import { Tooltip as TooltipPrimitive } from "radix-ui";
import { cn } from "@/lib/utils";

type Placement = "top" | "bottom" | "left" | "right";

interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  placement?: Placement;
  delay?: number;
  offset?: number;
}

export function Tooltip({
  content,
  children,
  placement = "top",
  delay = 120,
  offset = 6,
}: TooltipProps) {
  return (
    <TooltipPrimitive.Provider delayDuration={delay}>
      <TooltipPrimitive.Root>
        <TooltipPrimitive.Trigger asChild>
          <span className="inline-flex">{children}</span>
        </TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            side={placement}
            sideOffset={offset}
            className={cn(
              "z-50 w-fit rounded-md bg-foreground px-3 py-1.5 text-xs text-background",
              "animate-in fade-in-0 zoom-in-95",
              "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
              "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
              "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
            )}
          >
            {content}
            <TooltipPrimitive.Arrow className="z-50 size-2.5 translate-y-[calc(-50%_-_2px)] rotate-45 rounded-[2px] bg-foreground fill-foreground" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}
