/**
 * Chip + ChipText — small clickable filter pill.
 *
 * Legacy API exposed `<Chip active>` (background) and `<ChipText active>`
 * (text color/weight) as separate styled primitives. We keep both exports
 * — typical use: `<Chip active={x}><ChipText active={x}>label</ChipText></Chip>`.
 */
import * as React from "react";
import { cn } from "@/lib/utils";

interface ChipProps extends React.ComponentProps<"div"> {
  active?: boolean;
}

export function Chip({
  active = false,
  className,
  children,
  ...rest
}: ChipProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-[3px] rounded-md px-2.5 py-[5px]",
        "cursor-pointer select-none transition-opacity hover:opacity-80",
        active ? "bg-primary/10" : "bg-muted",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

interface ChipTextProps extends React.ComponentProps<"span"> {
  active?: boolean;
}

export function ChipText({
  active = false,
  className,
  children,
  ...rest
}: ChipTextProps) {
  return (
    <span
      className={cn(
        "text-[11px]",
        active
          ? "font-semibold text-primary"
          : "font-normal text-muted-foreground",
        className,
      )}
      {...rest}
    >
      {children}
    </span>
  );
}
