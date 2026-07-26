import pytest
import yaml
from importlib import metadata

from rag_stack.server.routers.rag_project import (
    _read_n_init,
    _scaffold_flashrag,
    _scaffold_static,
)


@pytest.mark.parametrize(
    ("optimizer_type", "expected_key"),
    [
        ("sobol", None),
        ("greedy", None),
        ("greedy_then_hv_anchor_agent_qnehvi", None),
        ("smac_mobo", "n_gt_init"),
        ("ax_qnehvi", "n_both_init"),
        ("ax_qnehvi_mixedgp", "n_both_init"),
        ("agent_qnehvi", "n_both_init"),
        ("agent_qnehvi_slo", "n_both_init"),
    ],
)
@pytest.mark.parametrize(
    ("builder", "expected_init"),
    [
        (lambda optimizer_type: _scaffold_static(optimizer_type, "cost_model"), 10),
        (_scaffold_flashrag, 5),
    ],
)
def test_scaffold_uses_optimizer_specific_init_key(
    optimizer_type,
    expected_key,
    builder,
    expected_init,
):
    params = builder(optimizer_type)["optimizer"]["params"]
    assert params["seed"] == 42
    assert "n_cheap_init" not in params
    assert set(params) == {"seed"} | ({expected_key} if expected_key else set())
    if expected_key:
        assert params[expected_key] == expected_init


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"n_both_init": 10}, 10),
        ({"n_gt_init": 7}, 7),
        ({"seed": 42}, 0),
    ],
)
def test_read_n_init_supports_joint_and_smac_keys(tmp_path, params, expected):
    (tmp_path / "rag_stack_config.yaml").write_text(
        yaml.safe_dump({"optimizer": {"params": params}}),
    )
    assert _read_n_init(tmp_path) == expected


@pytest.mark.parametrize(("backend", "performance_source"), [
    ("static_gt", "cost_model"),
    ("static_gt", "measured"),
    ("flashrag", "cost_model"),
])
def test_installed_backend_scaffolds_pass_canonical_validation_without_model_loading(
    monkeypatch, backend, performance_source,
):
    from rag_stack.config_validator import ConfigValidator
    from rag_stack.server.routers import rag_project

    # Exercise installed package discovery, not a mock descriptor or temporary
    # entry-point fixture: templates are selected through the actual API route.
    installed = {entry.name for entry in metadata.entry_points(group="rag_stack.evaluators")}
    if backend == "flashrag" and backend not in installed:
        pytest.skip("The optional FlashRAG evaluator provider is not installed")
    assert backend in installed, f"Install the editable {backend} evaluator provider"

    def forbidden_model_cache_probe(*args, **kwargs):
        raise AssertionError("Scaffold validation must not fetch or load model weights")

    monkeypatch.setattr(ConfigValidator, "_check_hf_model_cache", forbidden_model_cache_probe)
    generated = rag_project.scaffold_config(rag_project.ScaffoldRequest(
        eval_backend=backend, optimizer_type="agent_qnehvi",
        performance_source=performance_source,
    ))
    config = yaml.safe_load(generated["content"])
    assert config == generated["config"]
    rag_project._warm_static_rag_evaluator()
    errors = ConfigValidator(config, check_model_cache=False).validate()
    assert errors == [], "\n".join(errors)
