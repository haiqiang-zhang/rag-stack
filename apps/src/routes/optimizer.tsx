import { createFileRoute } from "@tanstack/react-router";
import BenchmarkScreen from "@/screens/BenchmarkScreen";

export const Route = createFileRoute("/optimizer")({
  component: BenchmarkScreen,
});
