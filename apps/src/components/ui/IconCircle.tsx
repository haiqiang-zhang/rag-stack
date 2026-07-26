/**
 * IconCircle — square-ish rounded tile that wraps a small icon.
 *
 * Public API: `size` ("sm" | "md" | "lg"), `variant` ("accent" | "accentSolid"
 * | "muted"), and `active` (forces accentSolid).
 */
import * as React from "react";
import { cn } from "@/lib/utils";

type Size = "sm" | "md" | "lg";
type Variant = "accent" | "accentSolid" | "muted";

interface IconCircleProps extends React.ComponentProps<"div"> {
  size?: Size;
  variant?: Variant;
  active?: boolean;
}

const SIZE_CLASS: Record<Size, string> = {
  sm: "size-7 rounded-lg",
  md: "size-9 rounded-[10px]",
  lg: "size-12 rounded-[14px]",
};

const VARIANT_CLASS: Record<Variant, string> = {
  accent: "bg-primary/10 text-primary",
  accentSolid: "bg-primary text-primary-foreground",
  muted: "bg-muted text-foreground",
};

export function IconCircle({
  size = "md",
  variant = "accent",
  active = false,
  className,
  children,
  ...rest
}: IconCircleProps) {
  const effectiveVariant: Variant = active ? "accentSolid" : variant;
  return (
    <div
      className={cn(
        "inline-flex items-center justify-center",
        SIZE_CLASS[size],
        VARIANT_CLASS[effectiveVariant],
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}
