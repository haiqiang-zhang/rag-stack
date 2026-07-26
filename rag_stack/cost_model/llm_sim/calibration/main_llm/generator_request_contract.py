"""Canonical generator request semantics used by every calibration sensor.

Serving-stage timing and shape proof use the production OpenAI-compatible
``/v1/chat/completions`` boundary.  The component sensor remains deliberately
narrow: it feeds the *same rendered chat-template token IDs* to an offline
EngineCore and publishes only audited worker CUDA time.  Tokenisation, HTTP,
scheduler, sampling/logprob result handling, and response materialisation stay
outside that component boundary and therefore remain stage overhead.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


REQUEST_CONTRACT_SCHEMA = "rag_stack.generator_chat_request"
REQUEST_FORMAT = "chat_completions"
ENDPOINT = "/v1/chat/completions"
CHAT_TEMPLATE_SOURCE = "tokenizer.apply_chat_template"
TRUNCATE_PROMPT_TOKENS_POLICY = "served_max_model_len_minus_max_tokens"
TRUNCATION_SIDE = "right"
MAX_TOKENS_POLICY = "workload_output_tokens"
IGNORE_EOS_PURPOSE = (
    "calibration_exact_shape_control_not_production_stopping_policy"
)
COMPONENT_SERVICE_BOUNDARY = (
    "audited_worker_cuda_execution_only; chat_tokenization_http_scheduler_"
    "sampling_logprob_result_handling_excluded"
)
STAGE_OVERHEAD_BOUNDARY = (
    "production_chat_http_parent_interdeparture_minus_component_gpu_service; "
    "tokenization_http_scheduler_and_noncomponent_sampling_logprob_result_"
    "handling_remain_stage_overhead"
)
DIRECT_PREFILL_STAGE_OVERHEAD_BOUNDARY = (
    "direct_disaggregated_prefill_parent_resume_to_all_completion_window_"
    "divided_by_exact_completed_prefill_cohorts_minus_component_gpu_service;_"
    "runtime_application_once_per_logical_prefill_frontier"
)
DIRECT_DECODE_STAGE_OVERHEAD_BOUNDARY = (
    "direct_disaggregated_decode_scheduler_CLOCK_MONOTONIC_context_frontier_"
    "window_divided_by_exact_pure_decode_scheduler_cycles_minus_component_"
    "gpu_service"
)
CALIBRATION_TEMPERATURE = 0.7
CALIBRATION_SEED = 0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_FIELDS = {
    "schema",
    "endpoint",
    "request_format",
    "messages",
    "chat_template",
    "rendered_prompt_tokens",
    "rendered_prompt_token_ids_sha256",
    "temperature",
    "seed",
    "stream",
    "ignore_eos",
    "ignore_eos_purpose",
    "logprobs",
    "top_logprobs",
    "n",
    "max_tokens_policy",
    "truncate_prompt_tokens_policy",
    "truncation_side",
}
_CHAT_TEMPLATE_FIELDS = {
    "source",
    "tokenize",
    "add_generation_prompt",
    "chat_template_sha256",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"generator request contract {field} is invalid")
    return int(value)


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in value
        )
    ):
        raise ValueError("chat template did not return a token-id sequence")
    return [int(token) for token in value]


def validate_generator_request_contract(
    value: Any,
    *,
    expected_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """Validate and normalize one exact measured-compatible request contract."""

    if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
        raise ValueError("generator request contract fields are invalid")
    if (
        value.get("schema") != REQUEST_CONTRACT_SCHEMA
        or value.get("endpoint") != ENDPOINT
        or value.get("request_format") != REQUEST_FORMAT
    ):
        raise ValueError("unexpected generator request contract schema/endpoint")

    messages = value.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], Mapping)
        or set(messages[0]) != {"role", "content"}
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
        or not messages[0]["content"]
    ):
        raise ValueError("generator request contract messages are invalid")

    template = value.get("chat_template")
    if not isinstance(template, Mapping) or set(template) != _CHAT_TEMPLATE_FIELDS:
        raise ValueError("generator request contract chat_template is invalid")
    if (
        template.get("source") != CHAT_TEMPLATE_SOURCE
        or template.get("tokenize") is not True
        or template.get("add_generation_prompt") is not True
        or not isinstance(template.get("chat_template_sha256"), str)
        or _SHA256_RE.fullmatch(template["chat_template_sha256"]) is None
    ):
        raise ValueError("generator request contract chat_template is invalid")

    prompt_tokens = _positive_int(
        value.get("rendered_prompt_tokens"),
        field="rendered_prompt_tokens",
    )
    if expected_prompt_tokens is not None and prompt_tokens != int(
        expected_prompt_tokens
    ):
        raise ValueError(
            "generator request contract rendered prompt differs from workload"
        )
    token_digest = value.get("rendered_prompt_token_ids_sha256")
    if not isinstance(token_digest, str) or _SHA256_RE.fullmatch(token_digest) is None:
        raise ValueError(
            "generator request contract rendered token digest is invalid"
        )

    temperature = value.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) != CALIBRATION_TEMPERATURE
        or value.get("seed") != CALIBRATION_SEED
        or isinstance(value.get("seed"), bool)
        or value.get("stream") is not False
        or value.get("ignore_eos") is not True
        or value.get("ignore_eos_purpose") != IGNORE_EOS_PURPOSE
        or value.get("logprobs") is not True
        or value.get("n") != 1
        or isinstance(value.get("n"), bool)
        # One canonical field drives both the HTTP chat request and offline
        # SamplingParams.  Keeping a separate offline value would allow the
        # two calibration sensors to silently drift again.
        or value.get("top_logprobs") != 0
        or isinstance(value.get("top_logprobs"), bool)
        or value.get("max_tokens_policy") != MAX_TOKENS_POLICY
        or value.get("truncate_prompt_tokens_policy")
        != TRUNCATE_PROMPT_TOKENS_POLICY
        or value.get("truncation_side") != TRUNCATION_SIDE
    ):
        raise ValueError("generator request sampling contract is invalid")

    return json.loads(_canonical_json(value).decode("utf-8"))


def validate_generator_request_contracts(
    value: Any,
    *,
    expected_prompt_tokens: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate a prompt-token keyed set used by an offline component grid."""

    if not isinstance(value, Mapping) or not value:
        raise ValueError("generator request contracts must be a non-empty object")
    expected = (
        None
        if expected_prompt_tokens is None
        else sorted({int(item) for item in expected_prompt_tokens})
    )
    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, contract in value.items():
        if not isinstance(raw_key, str) or not raw_key.isdigit():
            raise ValueError("generator request contract prompt-token key is invalid")
        prompt_tokens = int(raw_key)
        if str(prompt_tokens) != raw_key or prompt_tokens < 1:
            raise ValueError("generator request contract prompt-token key is invalid")
        normalized[raw_key] = validate_generator_request_contract(
            contract,
            expected_prompt_tokens=prompt_tokens,
        )
    observed = sorted(int(key) for key in normalized)
    if expected is not None and observed != expected:
        raise ValueError(
            "generator request contracts do not cover exact prompt-token shapes"
        )
    return {key: normalized[key] for key in sorted(normalized, key=int)}


def render_generator_chat_prompt_token_ids(
    tokenizer: Any,
    contract: Mapping[str, Any],
) -> list[int]:
    """Render and verify the exact chat-template tokens used by component work."""

    normalized = validate_generator_request_contract(contract)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("generator tokenizer has no chat template")
    if hashlib.sha256(template.encode("utf-8")).hexdigest() != normalized[
        "chat_template"
    ]["chat_template_sha256"]:
        raise ValueError("generator tokenizer chat template differs from contract")
    token_ids = _token_ids(tokenizer.apply_chat_template(
        normalized["messages"],
        tokenize=True,
        add_generation_prompt=True,
    ))
    if len(token_ids) != normalized["rendered_prompt_tokens"]:
        raise ValueError("rendered chat prompt token count differs from contract")
    if _sha256(token_ids) != normalized["rendered_prompt_token_ids_sha256"]:
        raise ValueError("rendered chat prompt token IDs differ from contract")
    return token_ids


def build_generator_request_contracts(
    model: str,
    prompt_tokens: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Build deterministic one-user-message contracts for exact rendered lengths."""

    from transformers import AutoTokenizer

    targets = sorted({int(value) for value in prompt_tokens})
    if not targets or targets[0] < 1:
        raise ValueError("generator request prompt-token targets are invalid")
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError(f"generator model {model!r} has no chat template")
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()

    contracts: dict[str, dict[str, Any]] = {}
    for target in targets:
        selected_messages: list[dict[str, str]] | None = None
        selected_ids: list[int] | None = None
        # The canonical Qwen instruction templates add a fixed boundary and
        # tokenize each subsequent ``w`` as one token.  Search rather than
        # hard-code that overhead so tokenizer/template drift fails closed.
        for word_count in range(1, target + 1):
            messages = [{"role": "user", "content": " ".join(["w"] * word_count)}]
            token_ids = _token_ids(tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            ))
            if len(token_ids) == target:
                selected_messages = messages
                selected_ids = token_ids
                break
            if len(token_ids) > target:
                break
        if selected_messages is None or selected_ids is None:
            raise ValueError(
                f"cannot construct an exact {target}-token rendered chat prompt "
                f"for {model!r}"
            )
        contract = {
            "schema": REQUEST_CONTRACT_SCHEMA,
            "endpoint": ENDPOINT,
            "request_format": REQUEST_FORMAT,
            "messages": selected_messages,
            "chat_template": {
                "source": CHAT_TEMPLATE_SOURCE,
                "tokenize": True,
                "add_generation_prompt": True,
                "chat_template_sha256": template_sha,
            },
            "rendered_prompt_tokens": target,
            "rendered_prompt_token_ids_sha256": _sha256(selected_ids),
            "temperature": CALIBRATION_TEMPERATURE,
            "seed": CALIBRATION_SEED,
            "stream": False,
            "ignore_eos": True,
            "ignore_eos_purpose": IGNORE_EOS_PURPOSE,
            "logprobs": True,
            "top_logprobs": 0,
            "n": 1,
            "max_tokens_policy": MAX_TOKENS_POLICY,
            "truncate_prompt_tokens_policy": TRUNCATE_PROMPT_TOKENS_POLICY,
            "truncation_side": TRUNCATION_SIDE,
        }
        contracts[str(target)] = validate_generator_request_contract(
            contract,
            expected_prompt_tokens=target,
        )
    return contracts


def generator_chat_payload(
    contract: Mapping[str, Any],
    *,
    model: str,
    output_tokens: int,
    served_max_model_len: int,
) -> dict[str, Any]:
    """Materialize the exact non-streaming measured chat-completions body."""

    normalized = validate_generator_request_contract(contract)
    max_tokens = _positive_int(output_tokens, field="output_tokens")
    context = _positive_int(served_max_model_len, field="served_max_model_len")
    if normalized["rendered_prompt_tokens"] + max_tokens > context:
        raise ValueError("generator calibration request exceeds served context")
    return {
        "model": str(model),
        "messages": normalized["messages"],
        "temperature": float(normalized["temperature"]),
        "seed": normalized["seed"],
        "max_tokens": max_tokens,
        "stream": normalized["stream"],
        "ignore_eos": normalized["ignore_eos"],
        "logprobs": normalized["logprobs"],
        "top_logprobs": normalized["top_logprobs"],
        "n": normalized["n"],
        "truncate_prompt_tokens": max(1, context - max_tokens),
        "truncation_side": normalized["truncation_side"],
    }


def generator_request_contract_sha256(value: Mapping[str, Any]) -> str:
    """Return a stable digest only after full request-contract validation."""

    return _sha256(validate_generator_request_contract(value))


def generator_request_contracts_sha256(value: Mapping[str, Any]) -> str:
    """Bind a complete prompt-shape keyed request-contract generation."""

    return _sha256(validate_generator_request_contracts(value))
