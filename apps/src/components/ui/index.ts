/**
 * Barrel re-exports for the legacy app-specific primitive components.
 *
 * NOTE: shadcn primitives (Button, Card, Dialog, Tabs, Tooltip, …) are NOT
 * re-exported here — import them directly from `@/components/ui/<name>`
 * (lowercase filenames) per shadcn convention.
 *
 * `Badge` / `BadgeText` and `Tooltip` are re-exported from `./LegacyBadge`
 * and `./LegacyTooltip` (renamed on disk to avoid case-only filename
 * collisions with shadcn's `badge.tsx` / `tooltip.tsx`); the public names
 * are unchanged.
 */
export { PageHeader } from "./PageHeader";
export { SidePanel, SidePanelHeader } from "./SidePanel";
export { SideItem, SideItemDot } from "./SideItem";
export { Chip, ChipText } from "./Chip";
export { Badge, BadgeText } from "./LegacyBadge";
export { StatusPill } from "./StatusPill";
export { IconCircle } from "./IconCircle";
export { Tooltip } from "./LegacyTooltip";
export { SectionHeader } from "./SectionHeader";
export { CascadeArrow } from "./CascadeArrow";
export { ConfirmDialog } from "./ConfirmDialog";
