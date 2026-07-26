/**
 * SideItem + SideItemDot — list rows used inside `<SidePanel>`.
 *
 * Polished to shadcn data-list aesthetic:
 *   - `<button>` element (keyboard focusable + activatable, NOT a `<div>`)
 *   - `bg-accent` active state (shadcn-canonical neutral)
 *   - `hover:bg-accent/50` hover state
 *   - `rounded-md` corners (matches shadcn list rows)
 *   - `focus-visible:ring-2 focus-visible:ring-ring` for keyboard nav
 *   - Compact 32px row, full-width inside the panel (with 6px outer margin)
 *
 * SideItemDot is a 1.5×1.5 indicator that flips between primary (active)
 * and muted-foreground/30 (inactive — faint but discoverable).
 */
import * as React from "react";
import { cn } from "@/lib/utils";

interface SideItemProps extends React.ComponentProps<"button"> {
  active?: boolean;
}

export function SideItem({
  active = false,
  className,
  children,
  type = "button",
  ...rest
}: SideItemProps) {
  return (
    <button
      type={type}
      data-active={active || undefined}
      className={cn(
        "mx-1.5 mb-0.5 flex w-[calc(100%-12px)] select-none items-center gap-2",
        "rounded-md px-2 py-1.5 text-left text-sm transition-colors outline-none",
        "focus-visible:ring-2 focus-visible:ring-ring",
        active
          ? "bg-accent text-accent-foreground"
          : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

interface SideItemDotProps extends React.ComponentProps<"span"> {
  active?: boolean;
}

export function SideItemDot({
  active = false,
  className,
  ...rest
}: SideItemDotProps) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "size-1.5 shrink-0 rounded-full transition-colors",
        active ? "bg-primary" : "bg-muted-foreground/30",
        className,
      )}
      {...rest}
    />
  );
}
