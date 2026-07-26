import os

import pytest

import anyio
from pydantic import BaseModel

from rag_stack.ai_clients import get_ai_client
from rag_stack.ai_clients.openai_client import OpenAIClient


class _VerdictSchema(BaseModel):
	verdicts: list[dict]


@pytest.fixture(autouse=True)
def isolated_judge_environment(monkeypatch):
	for name in tuple(os.environ):
		if name.startswith(("JUDGE_", "VLLM_JUDGE_")):
			monkeypatch.delenv(name)
	monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")


@pytest.mark.parametrize("api_key", [None, "test-judge-key"])
def test_vllm_provider_uses_judge_env(monkeypatch, api_key):
	monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.example:8000")
	monkeypatch.setenv("JUDGE_MODEL", "Qwen/Qwen3.6-27B")
	monkeypatch.setenv("JUDGE_MAX_CONCURRENCY", "96")
	monkeypatch.setenv("JUDGE_TIMEOUT", "240")
	if api_key is not None:
		monkeypatch.setenv("JUDGE_API_KEY", api_key)

	client = get_ai_client("vllm")

	assert client.provider == "vllm"
	assert client.model == "Qwen/Qwen3.6-27B"
	assert client._base_url == "http://judge.example:8000/v1"
	assert client._api_key == (api_key or "EMPTY")
	assert client._max_concurrency == 96
	assert client._kwargs["timeout"] == 240.0
	assert client._structured_output_mode == "auto"
	assert client._default_request_kwargs["max_tokens"] == 1024


def test_vllm_judge_env_overrides_yaml_kwargs(monkeypatch):
	monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.example:8000")
	monkeypatch.setenv("JUDGE_MODEL", "Qwen/Qwen3.6-27B")
	monkeypatch.setenv("JUDGE_MAX_CONCURRENCY", "32")
	monkeypatch.setenv("JUDGE_TIMEOUT", "1200")
	monkeypatch.setenv("JUDGE_STRUCTURED_OUTPUT_MODE", "prompt_json")
	monkeypatch.setenv("JUDGE_MAX_TOKENS", "512")

	client = get_ai_client(
		"vllm",
		max_concurrency=16,
		timeout=180,
		structured_output_mode="auto",
		max_tokens=2048,
	)

	assert client._max_concurrency == 32
	assert client._kwargs["timeout"] == 1200.0
	assert client._structured_output_mode == "prompt_json"
	assert client._default_request_kwargs["max_tokens"] == 512


def test_vllm_prefix_model_overrides_env_model(monkeypatch):
	monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.example:8000/v1")
	monkeypatch.setenv("JUDGE_MODEL", "Env/Model")

	client = get_ai_client("vllm/Served/Model")

	assert client.model == "Served/Model"
	assert client._base_url == "http://judge.example:8000/v1"


def test_vllm_request_extra_body_is_not_client_init_kwarg(monkeypatch):
	monkeypatch.setenv("JUDGE_BASE_URL", "http://judge.example:8000")
	monkeypatch.setenv("JUDGE_MODEL", "Qwen/Qwen3.6-27B")

	client = get_ai_client(
		"vllm",
		extra_body={"custom": {"enabled": True}},
		max_tokens=768,
	)
	request_kwargs = client._prepare_request_kwargs({})

	assert "extra_body" not in client._kwargs
	assert "max_tokens" not in client._kwargs
	assert request_kwargs["max_tokens"] == 768
	assert request_kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
	assert request_kwargs["extra_body"]["custom"]["enabled"] is True


def test_vllm_structured_length_finish_retries_once_with_expanded_cap(monkeypatch):
	client = OpenAIClient(
		model="judge-model",
		base_url="http://judge.example:8000/v1",
		structured_output_mode="auto",
		default_request_kwargs={"max_tokens": 1024},
	)
	parse_calls = []

	async def fake_parse(messages, response_format, **kwargs):
		parse_calls.append(dict(kwargs))
		if len(parse_calls) == 1:
			raise RuntimeError("response content as the length limit was reached")
		return _VerdictSchema(verdicts=[{"verdict": "yes", "reason": "ok"}])

	async def unexpected_fallback(*args, **kwargs):
		raise AssertionError("expanded schema retry should have succeeded")

	monkeypatch.setattr(client, "_parse_structured_output", fake_parse)
	monkeypatch.setattr(client, "_json_object_structured_output", unexpected_fallback)

	result = anyio.run(
		lambda: client._structured_output_locked(
			[{"role": "user", "content": "score"}], _VerdictSchema,
		)
	)

	assert result.verdicts[0]["verdict"] == "yes"
	assert parse_calls == [{}, {"max_tokens": 4096}]


def test_vllm_structured_length_fallback_keeps_expanded_cap(monkeypatch):
	client = OpenAIClient(
		model="judge-model",
		base_url="http://judge.example:8000/v1",
		structured_output_mode="auto",
		default_request_kwargs={"max_tokens": 1024},
	)
	parse_calls = []
	json_calls = []

	async def truncated_parse(messages, response_format, **kwargs):
		parse_calls.append(dict(kwargs))
		raise RuntimeError("response content as the length limit was reached")

	async def successful_fallback(messages, response_format, **kwargs):
		json_calls.append(dict(kwargs))
		return _VerdictSchema(verdicts=[])

	monkeypatch.setattr(client, "_parse_structured_output", truncated_parse)
	monkeypatch.setattr(client, "_json_object_structured_output", successful_fallback)

	result = anyio.run(
		lambda: client._structured_output_locked(
			[{"role": "user", "content": "score"}], _VerdictSchema,
		)
	)

	assert result.verdicts == []
	assert parse_calls == [{}, {"max_tokens": 4096}]
	assert json_calls == [{"use_json_object": True, "max_tokens": 4096}]


def test_nvidia_hub_provider_removed():
	with pytest.raises(ValueError):
		get_ai_client("nvidia-hub/nvidia/qwen/qwen3.6-27b")
