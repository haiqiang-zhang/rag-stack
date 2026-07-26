"""Generator-transport projection shared by system-transfer entry points.

The transfer API publishes a target search space, while Phase 2a replays the
resolved pipeline committed with each source quality record.  Both surfaces
must apply the same ``vllm_api`` -> in-process ``vllm.LLM.chat`` conversion so
an explicitly local-chat target never retains a source service URI.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


LOCAL_VLLM_CHAT_DEFAULTS = {
    "gpu_memory_utilization": 0.4,
    "dtype": "bfloat16",
    "use_chat_template": True,
}
VLLM_API_ONLY_FIELDS = ("uri", "batch", "request_timeout")
VLLM_API_DEFAULT_MAX_TOKENS = 4096


def project_to_vllm_chat(value: Any) -> tuple[Any, dict]:
    """Deep-copy and project every vLLM API module onto local chat.

    ``value`` may be an unresolved search-space pipeline or a resolved runtime
    pipeline.  Only dictionaries that explicitly identify a vLLM transport
    are changed; model, temperature, and every other algorithm choice remain
    untouched.
    """

    projected = deepcopy(value)
    report = {
        "backend": "vllm_chat",
        "component_conversions": 0,
        "generator_backend_conversions": 0,
        "api_default_max_tokens_preserved": 0,
        "transport_fields_removed": 0,
        "local_vllm_modules": 0,
    }

    def _walk(item: Any) -> None:
        if isinstance(item, list):
            for nested in item:
                _walk(nested)
            return
        if not isinstance(item, dict):
            return

        converted = False
        if item.get("component") == "vllm_api":
            item["component"] = "vllm"
            report["component_conversions"] += 1
            converted = True
        if item.get("generator_backend") == "vllm_api":
            item["generator_backend"] = "vllm"
            report["generator_backend_conversions"] += 1
            converted = True

        uses_local_vllm = (
            item.get("component") == "vllm"
            or item.get("generator_backend") == "vllm"
        )
        if uses_local_vllm:
            if (
                "use_chat_template" in item
                and item["use_chat_template"] is not True
            ):
                raise ValueError(
                    "system transfer requires local vLLM chat; "
                    "use_chat_template must be true"
                )
            if converted and "max_tokens" not in item:
                item["max_tokens"] = VLLM_API_DEFAULT_MAX_TOKENS
                report["api_default_max_tokens_preserved"] += 1
            for field in VLLM_API_ONLY_FIELDS:
                if field in item:
                    item.pop(field)
                    report["transport_fields_removed"] += 1
            for key, default in LOCAL_VLLM_CHAT_DEFAULTS.items():
                item.setdefault(key, default)
            report["local_vllm_modules"] += 1

        for nested in item.values():
            _walk(nested)

    _walk(projected)
    return projected, report


def config_uses_only_local_vllm(config: dict) -> bool:
    """Whether the target algorithm pipeline explicitly selects local vLLM.

    A mixed local/API declaration is not guessed at.  The system-transfer API
    makes ``generator_backend: vllm`` unambiguous by projecting the published
    target first, so that target naturally satisfies this predicate.
    """

    pipeline = ((config.get("algo_search_space") or {}).get("pipeline"))
    modes: set[str] = set()

    def _walk(item: Any) -> None:
        if isinstance(item, list):
            for nested in item:
                _walk(nested)
            return
        if not isinstance(item, dict):
            return
        for key in ("component", "generator_backend"):
            backend = item.get(key)
            if backend in {"vllm", "vllm_api"}:
                modes.add(str(backend))
        for nested in item.values():
            _walk(nested)

    _walk(pipeline)
    return "vllm" in modes and "vllm_api" not in modes
