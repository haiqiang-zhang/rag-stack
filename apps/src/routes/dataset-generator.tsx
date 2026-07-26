import { createFileRoute } from "@tanstack/react-router";
import DatasetGeneratorScreen from "@/screens/DatasetGeneratorScreen";

export const Route = createFileRoute("/dataset-generator")({
  component: DatasetGeneratorScreen,
});
