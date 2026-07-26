"""Unified AI client factory for rag-stack.

Usage:
    from rag_stack.ai_clients import get_ai_client

    client = get_ai_client("openai/gpt-4o-2024-08-06")
    client = get_ai_client("anthropic/claude-code/sonnet")
"""

from rag_stack.ai_clients.base import AIClient


def get_ai_client(model_name: str, **kwargs) -> AIClient:
	"""Create an AI client from a model_name string.

	model_name format:
	    vllm[/<model>]                  → OpenAIClient wired to a vLLM HTTP server
	    openai/<model>                  → OpenAIClient
	    openai/cscs/<model>             → OpenAIClient wired to CSCS Swiss AI
	    anthropic/claude-code/<model>   → ClaudeCodeClient
	    anthropic/minimax/<model>       → AnthropicClient (MiniMax backend)
	    anthropic/<model>               → AnthropicClient (real Anthropic)
	    openai/codex-cli/<model>        → CodexCliClient (wraps `codex exec`)

	Args:
	    model_name: Provider-prefixed model identifier.
	    **kwargs: Provider-specific options (api_key, base_url, cwd, etc.)

	Returns:
	    An AIClient instance.
	"""
	if model_name.startswith("openai/codex-cli/"):
		from rag_stack.ai_clients.codex_cli_client import CodexCliClient
		model = model_name[len("openai/codex-cli/"):]
		return CodexCliClient(model=model, **kwargs)
	elif model_name == "vllm" or model_name.startswith("vllm/"):
		from rag_stack.ai_clients.openai_client import build_vllm_client
		model = model_name[len("vllm/"):] if model_name.startswith("vllm/") else None
		return build_vllm_client(model=model, **kwargs)
	elif model_name.startswith("openai/cscs/"):
		# CSCS Swiss AI inference (OpenAI-compatible, requires CSCS_SERVING_API).
		# Must come BEFORE the generic "openai/" branch.
		from rag_stack.ai_clients.openai_client import build_cscs_client
		model = model_name[len("openai/cscs/"):]
		return build_cscs_client(model=model, **kwargs)
	elif model_name.startswith("openai/openrouter/"):
		# OpenRouter (OpenAI-compatible, requires OPENROUTER_API_KEY).
		# Must come BEFORE the generic "openai/" branch.
		from rag_stack.ai_clients.openai_client import build_openrouter_client
		model = model_name[len("openai/openrouter/"):]
		return build_openrouter_client(model=model, **kwargs)
	elif model_name.startswith("openai/deepseek/"):
		# DeepSeek (OpenAI-compatible, requires DEEPSEEK_API_KEY).
		# Must come BEFORE the generic "openai/" branch.
		from rag_stack.ai_clients.openai_client import build_deepseek_client
		model = model_name[len("openai/deepseek/"):]
		return build_deepseek_client(model=model, **kwargs)
	elif model_name.startswith("openai/"):
		from rag_stack.ai_clients.openai_client import OpenAIClient
		model = model_name[len("openai/"):]
		return OpenAIClient(model=model, **kwargs)
	elif model_name.startswith("anthropic/claude-code/"):
		from rag_stack.ai_clients.claude_code_client import ClaudeCodeClient
		model = model_name[len("anthropic/claude-code/"):]
		return ClaudeCodeClient(model=model, **kwargs)
	elif model_name.startswith("anthropic/minimax/"):
		from rag_stack.ai_clients.anthropic_client import build_minimax_client
		model = model_name[len("anthropic/minimax/"):]
		return build_minimax_client(model=model, **kwargs)
	elif model_name.startswith("anthropic/"):
		from rag_stack.ai_clients.anthropic_client import build_anthropic_client
		model = model_name[len("anthropic/"):]
		return build_anthropic_client(model=model, **kwargs)
	else:
		raise ValueError(
			f"Unknown model_name format: '{model_name}'. "
			f"Expected 'vllm[/<model>]', 'openai/<model>', "
			f"'openai/codex-cli/<model>', "
			f"'anthropic/claude-code/<model>', 'anthropic/minimax/<model>', "
			f"or 'anthropic/<model>'."
		)


__all__ = ["AIClient", "get_ai_client"]
