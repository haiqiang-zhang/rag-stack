"""The sole cost model and its calibration roots own unversioned namespaces."""

from __future__ import annotations

from tests import REPO_ROOT

import ast
from pathlib import Path


RAG_STACK = REPO_ROOT / "rag_stack"


def _import_names(path: Path) -> set[str]:
    names: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_repository_packages_one_unversioned_cost_model_and_profile_root() -> None:
    assert sorted(
        path.name
        for path in RAG_STACK.glob("cost_model*")
        if path.is_dir()
    ) == ["cost_model"]
    assert sorted(
        path.name
        for path in REPO_ROOT.glob("calibration_profiles*")
        if path.is_dir()
    ) == ["calibration_profiles"]
    assert sorted(path.name for path in RAG_STACK.glob("calibration_paths*.py")) == [
        "calibration_paths.py"
    ]
    assert sorted(
        path.name for path in RAG_STACK.glob("calibration_inventory*.py")
    ) == ["calibration_inventory.py"]


def test_cost_model_implementation_and_registry_belong_to_the_public_namespace() -> None:
    import rag_stack.cost_model as cost_model
    import rag_stack.cost_model.types as types
    from rag_stack.cost_model.dispatch import validate_against_registry

    package_root = RAG_STACK / "cost_model"
    assert Path(cost_model.__file__).resolve() == package_root / "__init__.py"
    assert Path(types.__file__).resolve() == package_root / "types.py"
    assert cost_model.RAGCMAssembly.__module__ == "rag_stack.cost_model.assembly"
    assert types.CostModelResult.__module__ == "rag_stack.cost_model.types"
    assert cost_model.COST_MODEL_REGISTRY
    validate_against_registry()
    assert all(
        cls.__module__.startswith("rag_stack.cost_model.")
        for cls in cost_model.COST_MODEL_REGISTRY.values()
    )


def test_calibration_helpers_are_implemented_in_the_public_namespace() -> None:
    import rag_stack.calibration_inventory as inventory
    import rag_stack.calibration_paths as paths

    assert inventory.build_calibration_inventory.__module__ == (
        "rag_stack.calibration_inventory"
    )
    assert paths.calibration_profile_root.__module__ == "rag_stack.calibration_paths"
    assert paths.configure_calibration_roots.__module__ == "rag_stack.calibration_paths"


def test_production_imports_do_not_use_obsolete_cost_model_namespaces() -> None:
    obsolete_prefixes = (
        "rag_stack.cost_model_v",
        "rag_stack.calibration_paths_v",
        "rag_stack.calibration_inventory_v",
    )
    for path in RAG_STACK.rglob("*.py"):
        obsolete = {
            name for name in _import_names(path) if name.startswith(obsolete_prefixes)
        }
        assert not obsolete, (path, obsolete)


def test_default_calibration_roots_use_unversioned_paths(monkeypatch) -> None:
    from rag_stack.calibration_paths import (
        CALIBRATION_DATA_DIR_ENV,
        CALIBRATION_PROFILE_DIR_ENV,
        calibration_data_root,
        calibration_profile_root,
        resolve_calibration_roots,
    )

    assert CALIBRATION_DATA_DIR_ENV == "RAG_STACK_CALIBRATION_DATA_DIR"
    assert CALIBRATION_PROFILE_DIR_ENV == "RAG_STACK_CALIBRATION_PROFILE_DIR"
    monkeypatch.delenv(CALIBRATION_DATA_DIR_ENV, raising=False)
    monkeypatch.delenv(CALIBRATION_PROFILE_DIR_ENV, raising=False)
    expected = (REPO_ROOT / "calibration_data", REPO_ROOT / "calibration_profiles")
    assert calibration_data_root() == expected[0]
    assert calibration_profile_root() == expected[1]
    assert resolve_calibration_roots() == expected


def test_calibration_root_overrides_keep_measurements_and_profiles_isolated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from rag_stack.calibration_paths import (
        CALIBRATION_DATA_DIR_ENV,
        CALIBRATION_PROFILE_DIR_ENV,
        calibration_data_root,
        calibration_profile_root,
        frozen_measurement_path,
        resolve_calibration_roots,
        system_profile_path,
    )

    data_root = tmp_path / "measurements"
    profile_root = tmp_path / "profiles"
    monkeypatch.setenv(CALIBRATION_DATA_DIR_ENV, str(data_root))
    monkeypatch.setenv(CALIBRATION_PROFILE_DIR_ENV, str(profile_root))

    assert calibration_data_root() == data_root.resolve()
    assert calibration_profile_root() == profile_root.resolve()
    assert resolve_calibration_roots() == (data_root.resolve(), profile_root.resolve())
    assert frozen_measurement_path("llm_sim", "fixture_gpu") == (
        data_root / "llm_sim" / "fixture_gpu.parquet"
    ).resolve()
    assert system_profile_path("llm_sim", "fixture_gpu") == (
        profile_root / "llm_sim" / "fixture_gpu.json"
    ).resolve()

    # Explicit CLI roots take precedence over environment overrides and are
    # resolved relative to the caller, without changing the process defaults.
    assert resolve_calibration_roots(
        data_dir="campaign/data", profile_dir="campaign/profiles", relative_to=tmp_path
    ) == (tmp_path / "campaign/data", tmp_path / "campaign/profiles")
    assert calibration_data_root() == data_root.resolve()
    assert calibration_profile_root() == profile_root.resolve()
