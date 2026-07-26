import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * shadcn-canonical class merger. Use everywhere we conditionally combine
 * Tailwind utilities — `cn("px-2", isActive && "bg-accent")` — to avoid the
 * "later wins" Tailwind specificity gotcha.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
