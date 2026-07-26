import { createFileRoute } from "@tanstack/react-router";
import QualityStoresScreen from "@/screens/QualityStoresScreen";

export const Route = createFileRoute("/quality-stores")({
  component: QualityStoresScreen,
});
