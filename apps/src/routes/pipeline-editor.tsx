import { createFileRoute } from "@tanstack/react-router";
import PipelineEditorScreen from "@/screens/PipelineEditorScreen";

export const Route = createFileRoute("/pipeline-editor")({
  component: PipelineEditorScreen,
});
