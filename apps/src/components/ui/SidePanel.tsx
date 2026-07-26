/**
 * SidePanel — fixed-width left rail container.
 * SidePanelHeader — its sticky header strip with an optional action on the right.
 *
 * Style aligns with shadcn's sidebar primitives (`bg-sidebar` /
 * `border-sidebar-border`) so in-page rails and the global app sidebar look
 * like a single visual family.
 */
import * as React from "react";
import { cn } from "@/lib/utils";

export function SidePanel({
  className,
  children,
  ...rest
}: React.ComponentProps<"aside">) {
  return (
    <aside
      className={cn(
        "flex w-[220px] shrink-0 select-none flex-col",
        // `bg-card` (pure white) matches the PageHeader so the panel feels
        // visually continuous with the chrome above it. Using `bg-sidebar`
        // (≈98.5% lightness) creates a subtle color shift below the header
        // border that reads as a "gap".
        "border-r border-border bg-card",
        className,
      )}
      {...rest}
    >
      {children}
    </aside>
  );
}

export function SidePanelHeader({
  className,
  children,
  ...rest
}: React.ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-1 border-b border-border px-3 py-1.5",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
