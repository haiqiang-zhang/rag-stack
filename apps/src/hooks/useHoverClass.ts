import { useRef, useEffect } from "react";

/**
 * Adds a CSS class to the DOM element for web hover effects.
 * React Native Web doesn't forward className prop, so we add it via ref.
 *
 * Usage:
 *   const hoverRef = useHoverClass("hover-btn-primary");
 *   <Pressable ref={hoverRef} style={...}>
 */
// TODO(migrate): originally a workaround for React Native Web not forwarding
// className. In the Vite + shadcn/ui app, prefer applying the className prop
// directly on the element instead of using this ref-based hook. Kept here for
// minimal-diff porting; revisit when callers are migrated.
export function useHoverClass<T = any>(className: string) {
  const ref = useRef<T>(null);

  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current as unknown as HTMLElement;
    if (el && el.classList) {
      className.split(" ").forEach((c) => el.classList.add(c));
    }
  });

  return ref;
}

/**
 * Directly add hover class to a DOM node. For use with callback refs or
 * in places where useHoverClass hook can't be used (e.g. inside .map()).
 *
 * Usage:
 *   <Pressable ref={(r) => addHoverClass(r, "hover-btn-icon")} style={...}>
 */
export function addHoverClass(node: any, className: string) {
  if (!node) return;
  const el = node as HTMLElement;
  if (el && el.classList) {
    className.split(" ").forEach((c) => el.classList.add(c));
  }
}
