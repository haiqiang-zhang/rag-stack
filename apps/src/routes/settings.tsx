import { createFileRoute } from "@tanstack/react-router";
import SettingsScreen from "@/screens/SettingsScreen";

const VALID_SECTIONS = [
  "connection",
  "generators",
  "keys",
  "rag",
  "dataset",
  "cost-model",
  "optimizer",
] as const;

type SectionKey = (typeof VALID_SECTIONS)[number];

interface SettingsSearch {
  section?: SectionKey;
}

export const Route = createFileRoute("/settings")({
  component: SettingsScreen,
  validateSearch: (search: Record<string, unknown>): SettingsSearch => {
    const s = search.section;
    if (typeof s === "string" && (VALID_SECTIONS as readonly string[]).includes(s)) {
      return { section: s as SectionKey };
    }
    return {};
  },
});
