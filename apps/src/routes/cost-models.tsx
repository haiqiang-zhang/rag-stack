import { createFileRoute } from "@tanstack/react-router";
import CostModelsScreen from "@/screens/CostModelsScreen";
export const Route = createFileRoute("/cost-models")({
  component: CostModelsScreen,
});
