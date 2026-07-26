"""Host configuration validation for the evaluator's local chat protocol.

Concrete prompt handling belongs to the evaluator's generation protocol tests.
"""

import pytest

from rag_stack.config_validator import ConfigValidator


@pytest.mark.parametrize(
    "module",
    [
        {"component": "vllm", "use_chat_template": False},
        {"component": "multi_query_expansion", "generator_backend": "vllm",
         "use_chat_template": False},
    ],
)
def test_config_validator_rejects_raw_local_vllm_for_every_role(module):
    validator = ConfigValidator.__new__(ConfigValidator)
    validator.errors = []
    validator._check_vllm_chat_contract([module], "pipeline.node.stage")
    assert len(validator.errors) == 1
    assert "Raw completions are unsupported" in validator.errors[0]


def test_config_validator_accepts_implicit_chat_default():
    validator = ConfigValidator.__new__(ConfigValidator)
    validator.errors = []
    validator._check_vllm_chat_contract(
        [
            {"component": "vllm"},
            {"component": "multi_query_expansion", "generator_backend": "vllm"},
        ],
        "pipeline.node.stage",
    )
    assert validator.errors == []
