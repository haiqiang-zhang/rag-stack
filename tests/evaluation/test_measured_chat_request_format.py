import asyncio
import copy
from types import SimpleNamespace

import pytest

from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import (
    MeasuredEvaluator,
    apply_measured_generation_defaults,
)
from rag_stack_evaluator.static_rag_evaluator.measured.provider import MeasuredProvider
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_pd_pair import VllmPdPair
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
    MEASURED_REQUEST_FORMAT_KEY,
    REQUEST_FORMAT_CHAT_COMPLETIONS,
    REQUEST_FORMAT_COMPLETIONS,
    VllmSubprocess,
)
from scripts.paper.replay_msmarco_agent_v2_old_pareto_measured import (
    MEASURED_SOURCE_MAX_TOKENS_KEY,
    apply_measured_overlay,
)


def test_replay_overlay_preserves_vllm_api_chat_caps():
    config = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "stage": "query_expansion",
                        "modules": [
                            {
                                "component": "hyde",
                                "generator_backend": "vllm_api",
                                "model": "aux",
                                "uri": "http://gateway",
                            }
                        ],
                    },
                    {
                        "stage": "generator",
                        "modules": [
                            {
                                "component": "vllm_api",
                                "model": "main",
                                "max_tokens": 512,
                                "uri": "http://gateway",
                            }
                        ],
                    },
                ]
            }
        ]
    }

    overlay = apply_measured_overlay(config)
    qe, generator = config["node_lines"][0]["nodes"]
    qe_module = qe["modules"][0]
    generator_module = generator["modules"][0]

    assert overlay == {
        "generator_vllm_overlays": 1,
        "query_expansion_vllm_overlays": 1,
        "chat_completions_overlays": 2,
    }
    assert qe_module[MEASURED_REQUEST_FORMAT_KEY] == REQUEST_FORMAT_CHAT_COMPLETIONS
    assert qe_module[MEASURED_SOURCE_MAX_TOKENS_KEY] == 4096
    assert generator_module[MEASURED_REQUEST_FORMAT_KEY] == REQUEST_FORMAT_CHAT_COMPLETIONS
    assert generator_module[MEASURED_SOURCE_MAX_TOKENS_KEY] == 512

    measured = apply_measured_generation_defaults(config)
    measured_qe, measured_generator = measured["node_lines"][0]["nodes"]
    assert measured_qe["modules"][0]["max_tokens"] == 4096
    assert measured_generator["modules"][0]["max_tokens"] == 512
    assert MEASURED_SOURCE_MAX_TOKENS_KEY not in measured_qe["modules"][0]
    assert MEASURED_SOURCE_MAX_TOKENS_KEY not in measured_generator["modules"][0]


def test_provider_does_not_reapply_generation_defaults():
    config = {
        "node_lines": [
            {
                "nodes": [
                    {
                        "stage": "query_expansion",
                        "modules": [
                            {
                                "component": "hyde",
                                "generator_backend": "vllm_api",
                                "model": "aux",
                            }
                        ],
                    },
                    {
                        "stage": "generator",
                        "modules": [
                            {
                                "component": "vllm_api",
                                "model": "main",
                                "max_tokens": 512,
                            }
                        ],
                    },
                ]
            }
        ]
    }
    apply_measured_overlay(config)
    observed = {}

    class FakeMeasured:
        def evaluate(self, **kwargs):
            observed.update(copy.deepcopy(kwargs))
            checks = {
                name: {"observed": None, "passed": True, "required": True}
                for name in (
                    "driver_errors_window_zero",
                    "measurement_rate_stability_waves_ge_2",
                    "population_saturated",
                    "qps_subwindow_cv_le_0_15",
                    "warmup_rate_stable_not_cap_forced",
                )
            }
            return {
                "quality": {},
                "performance": {
                    "qps": 1.0,
                    "measured_gt_admissibility": {
                        "admissible": True,
                        "checks": checks,
                        "reasons": [],
                    },
                },
                "performance_execution_trace": {},
            }

    class FakeCache:
        def evict_all_hf_models(self):
            pass

        def evict_vllm(self):
            pass

    provider = object.__new__(MeasuredProvider)
    provider._cache = FakeCache()
    provider._measured = FakeMeasured()
    provider._deploy = SimpleNamespace(prepare_trial=lambda **kwargs: None)
    provider._n_queries = None
    provider._selection = "max_throughput"
    provider._memo = {}
    provider._dirty = False

    result = provider.evaluate(config, {})

    qe, generator = observed["config"]["node_lines"][0]["nodes"]
    assert result.performance_score == 1.0
    assert observed["generation_defaults_applied"] is True
    assert qe["modules"][0]["max_tokens"] == 4096
    assert generator["modules"][0]["max_tokens"] == 512
    assert MEASURED_SOURCE_MAX_TOKENS_KEY not in qe["modules"][0]


def test_sampling_params_defaults_to_chat_and_rejects_raw_completion():
    params = MeasuredEvaluator._sampling_params(
        {
            "temperature": 0.7,
            "max_tokens": 512,
            MEASURED_REQUEST_FORMAT_KEY: REQUEST_FORMAT_CHAT_COMPLETIONS,
        }
    )
    assert params[MEASURED_REQUEST_FORMAT_KEY] == REQUEST_FORMAT_CHAT_COMPLETIONS
    defaults = MeasuredEvaluator._sampling_params({"max_tokens": 512})
    assert defaults[MEASURED_REQUEST_FORMAT_KEY] == REQUEST_FORMAT_CHAT_COMPLETIONS

    with pytest.raises(ValueError, match="raw completions are disabled"):
        MeasuredEvaluator._sampling_params({
            MEASURED_REQUEST_FORMAT_KEY: REQUEST_FORMAT_COMPLETIONS,
        })

    with pytest.raises(ValueError, match="unsupported measured_request_format"):
        MeasuredEvaluator._sampling_params(
            {MEASURED_REQUEST_FORMAT_KEY: "not-an-api"}
        )


def test_vllm_subprocess_nonstreaming_chat_request_contract():
    subprocess = object.__new__(VllmSubprocess)
    subprocess.key = SimpleNamespace(model="Qwen/Test")
    subprocess._served_max_model_len = 32768
    observed = {}

    async def fake_raw(payload, *, request_format):
        observed["payload"] = payload
        observed["request_format"] = request_format
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"completion_tokens": 7},
        }

    subprocess._raw_completion = fake_raw
    text, perf = asyncio.run(
        subprocess.generate_one(
            "question",
            {
                "temperature": 0.7,
                "max_tokens": 512,
                MEASURED_REQUEST_FORMAT_KEY: REQUEST_FORMAT_CHAT_COMPLETIONS,
            },
        )
    )

    assert text == "answer"
    assert perf["n_output_tokens"] == 7
    assert observed["request_format"] == REQUEST_FORMAT_CHAT_COMPLETIONS
    assert observed["payload"]["messages"] == [
        {"role": "user", "content": "question"}
    ]
    assert "prompt" not in observed["payload"]
    assert observed["payload"]["truncate_prompt_tokens"] == 32256
    assert observed["payload"]["truncation_side"] == "right"
    assert observed["payload"]["logprobs"] is True
    assert observed["payload"]["n"] == 1


def test_vllm_subprocess_chat_request_is_the_default():
    subprocess = object.__new__(VllmSubprocess)
    subprocess.key = SimpleNamespace(model="Qwen/Test")
    subprocess._served_max_model_len = 32768
    observed = {}

    async def fake_raw(payload, *, request_format):
        observed["payload"] = payload
        observed["request_format"] = request_format
        return {
            "choices": [{"message": {"content": "continuation"}}],
            "usage": {"completion_tokens": 3},
        }

    subprocess._raw_completion = fake_raw
    text, perf = asyncio.run(
        subprocess.generate_one("prompt", {"temperature": 0.7, "max_tokens": 512})
    )

    assert text == "continuation"
    assert perf["n_output_tokens"] == 3
    assert observed["request_format"] == REQUEST_FORMAT_CHAT_COMPLETIONS
    assert observed["payload"]["messages"] == [
        {"role": "user", "content": "prompt"}
    ]
    assert "prompt" not in observed["payload"]
    assert observed["payload"]["logprobs"] is True


def test_vllm_pd_pair_uses_chat_on_both_sides_of_handoff():
    pair = object.__new__(VllmPdPair)
    pair.key = SimpleNamespace(model="Qwen/Test")
    pair._served_max_model_len = 32768
    pair.prefill_port = 10001
    pair.decode_port = 10002
    observed = []

    async def fake_post(port, payload, request_id, *, request_format):
        observed.append((port, payload, request_id, request_format))
        if port == pair.prefill_port:
            return {"kv_transfer_params": {"remote_request_id": "handle"}}
        return {
            "choices": [{"message": {"content": "pd answer"}}],
            "usage": {"completion_tokens": 9},
        }

    pair._raw_post = fake_post
    text, perf = asyncio.run(
        pair.generate_one(
            "question",
            {
                "temperature": 0.7,
                "max_tokens": 512,
            },
        )
    )

    assert text == "pd answer"
    assert perf["n_output_tokens"] == 9
    assert len(observed) == 2
    prefill, decode = observed
    assert prefill[2] == decode[2]
    assert prefill[3] == decode[3] == REQUEST_FORMAT_CHAT_COMPLETIONS
    assert prefill[1]["messages"] == decode[1]["messages"] == [
        {"role": "user", "content": "question"}
    ]
    assert "prompt" not in prefill[1]
    assert "prompt" not in decode[1]
    assert decode[1]["kv_transfer_params"] == {
        "remote_request_id": "handle"
    }
