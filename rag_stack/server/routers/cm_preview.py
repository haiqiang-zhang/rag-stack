"""Project CM previews; transport and path authorization only."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from rag_stack import cm_preview

router = APIRouter(tags=["cm-preview"])


class CreatePreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    eval_id: int | None = Field(default=None, ge=0)
    parameters: dict | None = None
    algo_config: dict | None = None
    system_config: dict | None = None
    qa_data: str | None = None
    corpus_data: str | None = None
    dataset_name: str | None = None
    sample_count: int = Field(default=8, ge=1, le=128)
    question_ids: list[str | int] | None = Field(default=None, min_length=1, max_length=128)
    reuse_trace: bool = False
    skip_judge: bool = True

    @model_validator(mode="after")
    def validate_selection(self):
        if sum(x is not None for x in (self.eval_id, self.parameters, self.algo_config)) != 1:
            raise ValueError("Choose exactly one of eval_id, parameters, or algo_config")
        if self.reuse_trace and self.eval_id is None:
            raise ValueError("Trace reuse requires eval_id")
        if self.reuse_trace and (self.qa_data or self.corpus_data):
            raise ValueError("A reused trace retains its original dataset; dataset overrides require a fresh run")
        if self.question_ids is not None and len(set(map(str, self.question_ids))) != len(self.question_ids):
            raise ValueError("Question IDs must be unique")
        return self


def _project(name: str) -> Path:
    from .rag_project import _get_project_dir
    if not name or name in {".", ".."} or any(c in name for c in ("/", "\\", "\0")):
        raise cm_preview.PreviewError("invalid_project", "Invalid project name")
    root = _get_project_dir().resolve()
    project = (root / name).resolve()
    if project.parent != root:
        raise cm_preview.PreviewError("invalid_project", "Project path is outside the project root")
    if not project.is_dir():
        raise FileNotFoundError("Project not found")
    return project


def _config(project: Path) -> dict:
    path = cm_preview._contained(project, project / "rag_stack_config.yaml")
    return yaml.safe_load(path.read_text()) or {}


def _dataset_path(project: Path, config: dict, kind: str, supplied: str | None) -> Path:
    from .rag_project import _resolve_dataset_file
    if supplied:
        path, _ = _resolve_dataset_file(supplied, kind.upper())
    else:
        names = ("qa.parquet",) if kind == "qa" else ("original_corpus.parquet", "corpus.parquet")
        path = next((cm_preview._contained(project, project / "data" / name)
                     for name in names if (project / "data" / name).is_file()), None)
        if path is None:
            ref = (config.get("dataset") or {}).get(kind)
            if not ref:
                raise cm_preview.PreviewError("dataset_required", f"Supply {kind}_data for this project",
                                              details={"field": f"{kind}_data"})
            path, _ = _resolve_dataset_file(ref, kind.upper())
    if path.suffix != ".parquet":
        raise cm_preview.PreviewError("invalid_dataset", f"{kind.upper()} must be a Parquet file")
    return path


def _data_paths(project: Path, request: dict) -> dict:
    config = _config(project)
    dataset = config.get("dataset") or {}
    if request.get("dataset_name") is None:
        request["dataset_name"] = dataset.get("dataset_name")
    if request.get("reuse_trace"):
        return request
    for kind in ("qa", "corpus"):
        path = _dataset_path(project, config, kind, request.get(f"{kind}_data"))
        request[f"{kind}_path"] = str(path)
    return request


def _raise_http(exc):
    if isinstance(exc, HTTPException):
        if isinstance(exc.detail, str):
            exc.detail = {"code": "invalid_input", "phase": "preparing", "message": exc.detail, "details": {}}
        raise exc
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(404, detail={"code": "not_found", "phase": "preparing",
                                         "message": str(exc), "details": {}}) from exc
    if isinstance(exc, cm_preview.PreviewError):
        raise HTTPException(400, detail=exc.to_dict()) from exc
    raise HTTPException(400, detail={"code": "invalid_input", "phase": "preparing",
                                     "message": str(exc), "details": {}}) from exc


@router.get("/projects/{project}/cm-previews/options")
def preview_options(project: str):
    try:
        directory = _project(project)
        config = _config(directory)
        dataset = config.get("dataset") or {}
        result = {"parameters": [], "selection_error": None,
                  "dataset": {"qa_available": False, "corpus_available": False,
                              "dataset_name": dataset.get("dataset_name")},
                  "system_config_required": True}
        for kind in ("qa", "corpus"):
            try:
                _dataset_path(directory, config, kind, None)
                result["dataset"][f"{kind}_available"] = True
            except (cm_preview.PreviewError, HTTPException):
                # Request creation provides the exact actionable missing field.
                pass
        try:
            from rag_stack.search_space import RagSearchSpace
            from rag_stack.performance_context import resolve_system_space_owner
            from ax.core.parameter import ChoiceParameter, RangeParameter
            space = RagSearchSpace.from_yaml(config, check_performance_support=False).ss
            for name, parameter in space.parameters.items():
                item = {"name": name, "value_type": parameter.parameter_type.name.lower()}
                if isinstance(parameter, ChoiceParameter):
                    item.update(type="choice", values=list(parameter.values))
                    if getattr(parameter, "is_hierarchical", False):
                        item["dependents"] = {str(k): list(v) for k, v in parameter.dependents.items()}
                elif isinstance(parameter, RangeParameter):
                    item.update(type="range", bounds=[parameter.lower, parameter.upper],
                                log_scale=parameter.log_scale)
                else:
                    item.update(type="fixed", value=parameter.value)
                result["parameters"].append(item)
            result["system_config_required"] = resolve_system_space_owner(config) != "optimizer"
        except Exception as exc:
            result["selection_error"] = {"code": "search_space_unavailable", "message": str(exc)}
        return result
    except (ValueError, OSError, HTTPException) as exc:
        _raise_http(exc)


@router.post("/projects/{project}/cm-previews", status_code=202)
def create_preview(project: str, body: CreatePreviewRequest):
    try:
        directory = _project(project)
        return cm_preview.create_preview(directory, _data_paths(directory, body.model_dump()))
    except (ValueError, OSError, HTTPException) as exc:
        _raise_http(exc)


@router.get("/projects/{project}/cm-previews")
def list_previews(project: str):
    try:
        return {"previews": cm_preview.list_previews(_project(project))}
    except (ValueError, OSError) as exc:
        _raise_http(exc)


@router.get("/projects/{project}/cm-previews/{preview_id}")
def get_preview(project: str, preview_id: str):
    try:
        return cm_preview.detail(cm_preview.preview_directory(_project(project), preview_id))
    except (ValueError, OSError) as exc:
        _raise_http(exc)


@router.post("/projects/{project}/cm-previews/{preview_id}/cancel")
def cancel_preview(project: str, preview_id: str):
    try:
        return cm_preview.cancel_preview(cm_preview.preview_directory(_project(project), preview_id))
    except (ValueError, OSError) as exc:
        _raise_http(exc)


@router.get("/projects/{project}/cm-previews/{preview_id}/artifacts/{name}")
def download_artifact(project: str, preview_id: str, name: str):
    try:
        directory = cm_preview.preview_directory(_project(project), preview_id)
        path = cm_preview.artifact_path(directory, name)
        return FileResponse(path, filename=name)
    except (ValueError, OSError) as exc:
        _raise_http(exc)
