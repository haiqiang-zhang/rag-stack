"""Host-side config/cache contracts for opt-in historical quality generation."""

from tests import REPO_ROOT
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = REPO_ROOT
_spec = importlib.util.spec_from_file_location(
    "legacy_protocol_config_validator", ROOT / "rag_stack/config_validator.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("module", [
    {"component": "vllm"},
    {"component": "multi_query_expansion", "generator_backend": "vllm"},
])
def test_legacy_protocol_requires_explicit_native_static_quality_opt_in(module):
    validator = _module.ConfigValidator.__new__(_module.ConfigValidator)
    validator.errors = []
    validator.gt_backend = "static_gt"
    validator.performance_source = "cost_model"
    legacy = {**module, "generation_protocol": "legacy_completion"}
    validator._check_vllm_chat_contract([legacy], "pipeline")
    assert validator.errors == []
    validator._check_vllm_chat_contract([{**module, "use_chat_template": False}], "pipeline")
    assert len(validator.errors) == 1
    assert "raw-completion mode is disabled" in validator.errors[0]


@pytest.mark.parametrize("backend,performance,module", [
    ("static_gt", "measured", {"component": "vllm"}),
    ("flashrag", "cost_model", {"component": "vllm"}),
    ("static_gt", "cost_model", {"component": "vllm_api"}),
    ("static_gt", "cost_model", {"generator_backend": "vllm_api"}),
    ("static_gt", "cost_model", {"generator_backend": "openai_llm"}),
])
def test_legacy_protocol_rejected_outside_native_static_quality(backend, performance, module):
    validator = _module.ConfigValidator.__new__(_module.ConfigValidator)
    validator.errors = []
    validator.gt_backend = backend
    validator.performance_source = performance
    validator._check_vllm_chat_contract([
        {**module, "generation_protocol": "legacy_completion"},
    ], "pipeline")
    assert len(validator.errors) == 1
    assert "restricted to native" in validator.errors[0]


def test_quality_store_retains_protocol_across_transport_normalization():
    from rag_stack.quality_store import _strip_transport_only_params as normalize

    chat = {"pipeline": {"modules": [{"component": "vllm", "model": "qwen"}]}}
    legacy = deepcopy(chat)
    legacy["pipeline"]["modules"][0]["generation_protocol"] = "legacy_completion"
    api = deepcopy(chat)
    api["pipeline"]["modules"][0].update(component="vllm_api", uri="http://unused")
    assert normalize(chat) == normalize(api)
    assert normalize(legacy) != normalize(chat)
    assert normalize(legacy)["pipeline"]["modules"][0]["generation_protocol"] == "legacy_completion"
