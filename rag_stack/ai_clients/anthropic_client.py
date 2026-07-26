"""Anthropic-protocol API client for rag-stack.

Works with any server that speaks the Anthropic Messages API:
- Real Anthropic (api.anthropic.com)
- MiniMax's Anthropic-compatible endpoint (https://api.minimaxi.com/anthropic)

The factory routes provider prefixes to preset base_url + env var:

    anthropic/<model>             → real Anthropic, ANTHROPIC_API_KEY
    anthropic/minimax/<model>     → MiniMax, MINIMAX_API_KEY
"""

import json
import logging
import os
import random
import re
import time
from typing import Any, List, Optional, Type

import anyio
from pydantic import BaseModel

from rag_stack.ai_clients.base import AIClient

logger = logging.getLogger("RAG-Stack")

MINIMAX_ANTHROPIC_BASE_URL = "https://api.minimaxi.com/anthropic"


class AnthropicClient(AIClient):
	"""Async Anthropic Messages API wrapper.

	Lazy-creates ``anthropic.AsyncAnthropic``. ``base_url`` + ``api_key``
	are injected by the factory so the same class backs both real Anthropic
	and OEM-compatible endpoints (MiniMax, etc.).
	"""

	provider = "anthropic"

	def __init__(
		self,
		model: str,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		max_concurrency: int = 10,
		max_retries: int = 10,
		max_tokens: int = 4096,
		**kwargs,
	):
		self.model = model
		self._api_key = api_key
		self._base_url = base_url
		self._max_concurrency = max_concurrency
		self._max_retries = max_retries
		self._default_max_tokens = max_tokens
		self._kwargs = kwargs
		self._client = None
		self._limiter: Optional[anyio.CapacityLimiter] = None
		self._rate_limit_until: float = 0.0

	@property
	def async_client(self):
		if self._client is None:
			from anthropic import AsyncAnthropic
			init_kwargs: dict[str, Any] = {}
			if self._api_key:
				init_kwargs["api_key"] = self._api_key
			if self._base_url:
				init_kwargs["base_url"] = self._base_url
			init_kwargs["max_retries"] = 2
			init_kwargs.update(self._kwargs)
			self._client = AsyncAnthropic(**init_kwargs)
		return self._client

	async def aclose(self) -> None:
		client, self._client = self._client, None
		if client is not None:
			await client.close()

	def _get_limiter(self) -> anyio.CapacityLimiter:
		if self._limiter is None:
			self._limiter = anyio.CapacityLimiter(self._max_concurrency)
		return self._limiter

	@staticmethod
	def _split_system(messages: List[dict]) -> tuple[str, List[dict]]:
		"""Anthropic takes ``system`` as a top-level kwarg, not a message."""
		system_parts: list[str] = []
		rest: List[dict] = []
		for m in messages:
			if m.get("role") == "system":
				c = m.get("content", "")
				if isinstance(c, str) and c:
					system_parts.append(c)
			else:
				rest.append(m)
		return "\n\n".join(system_parts), rest

	async def chat(self, messages: List[dict], **kwargs) -> str:
		from anthropic import RateLimitError

		async with self._get_limiter():
			client = self.async_client
			model = kwargs.pop("model", self.model)
			max_tokens = kwargs.pop("max_tokens", self._default_max_tokens)
			system, conv = self._split_system(messages)
			create_kwargs: dict[str, Any] = {
				"model": model,
				"messages": conv,
				"max_tokens": max_tokens,
			}
			if system:
				create_kwargs["system"] = system
			create_kwargs.update(kwargs)

			# The SDK refuses non-streaming requests whose estimated
			# wall-clock would exceed 10 min — observed empirically at
			# ``max_tokens ≳ 21000`` for OEM endpoints whose
			# tokens-per-second estimate is conservative.  Always stream
			# when over a safe-non-streaming threshold; the final text is
			# the same.
			use_stream = max_tokens > 16384

			for attempt in range(self._max_retries):
				now = time.monotonic()
				if self._rate_limit_until > now:
					wait = self._rate_limit_until - now
					logger.info(
						f"[Anthropic] Waiting {wait:.1f}s for global rate limit cooldown"
					)
					await anyio.sleep(wait)
				try:
					if use_stream:
						async with client.messages.stream(**create_kwargs) as stream:
							msg = await stream.get_final_message()
						return "".join(
							b.text for b in msg.content
							if getattr(b, "type", None) == "text"
						)
					resp = await client.messages.create(**create_kwargs)
					return "".join(
						b.text for b in resp.content
						if getattr(b, "type", None) == "text"
					)
				except RateLimitError:
					wait = min(10 * (2 ** attempt) + random.random(), 120)
					self._rate_limit_until = time.monotonic() + wait
					logger.warning(
						f"[Anthropic] 429 rate limited, pausing {wait:.1f}s "
						f"(attempt {attempt + 1}/{self._max_retries})"
					)
					await anyio.sleep(wait)
				except Exception:
					for msg in messages:
						content = msg.get("content", "")
						preview = content[:200] if isinstance(content, str) else str(content)[:200]
						logger.error(
							f"[Anthropic] {msg.get('role', '?')} "
							f"({len(content) if isinstance(content, str) else '?'} chars): {preview}"
						)
					raise
			raise RuntimeError(
				f"[Anthropic] Failed after {self._max_retries} rate limit retries"
			)

	async def structured_output(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		**kwargs,
	) -> BaseModel:
		"""Force JSON-schema output via a single forced tool call.

		Anthropic's Messages API has no native JSON-schema response mode,
		so the canonical pattern is to register one tool whose input_schema
		is the target schema and force the model to emit it via
		``tool_choice={"type": "tool", "name": ...}``. We parse the
		tool_use block back into the pydantic model. If the server happens
		to return plain JSON in a text block instead (some OEM endpoints
		drop tool_use on small models), we fall back to json-in-text.
		"""
		async with self._get_limiter():
			client = self.async_client
			model = kwargs.pop("model", self.model)
			max_tokens = kwargs.pop("max_tokens", self._default_max_tokens)
			system, conv = self._split_system(messages)

			tool_name = "extract"
			tools = [{
				"name": tool_name,
				"description": f"Return a {response_format.__name__} object.",
				"input_schema": response_format.model_json_schema(),
			}]
			create_kwargs: dict[str, Any] = {
				"model": model,
				"messages": conv,
				"max_tokens": max_tokens,
				"tools": tools,
				"tool_choice": {"type": "tool", "name": tool_name},
			}
			if system:
				create_kwargs["system"] = system
			create_kwargs.update(kwargs)

			resp = await client.messages.create(**create_kwargs)

		for block in resp.content:
			if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
				return response_format.model_validate(block.input)

		text = "".join(
			b.text for b in resp.content if getattr(b, "type", None) == "text"
		)
		if not text:
			raise ValueError(
				f"[Anthropic] No tool_use or text content in response for "
				f"{response_format.__name__}"
			)
		try:
			data = json.loads(text)
		except json.JSONDecodeError:
			m = re.search(r"\{.*\}", text, re.DOTALL)
			if not m:
				raise ValueError(
					f"[Anthropic] Could not parse JSON from response: {text[:200]}"
				)
			data = json.loads(m.group(0))
		return response_format.model_validate(data)

	def __repr__(self):
		return f"AnthropicClient(model='{self.model}', base_url='{self._base_url}')"


def build_minimax_client(model: str, **kwargs) -> AnthropicClient:
	"""Build an AnthropicClient wired to MiniMax's Anthropic-compatible endpoint.

	Reads ``MINIMAX_API_KEY`` from env unless ``api_key`` is given.
	``base_url`` defaults to MiniMax's Anthropic endpoint; override via kwarg
	to point at a staging host if needed.
	"""
	api_key = kwargs.pop("api_key", None) or os.environ.get("MINIMAX_API_KEY")
	base_url = kwargs.pop("base_url", None) or MINIMAX_ANTHROPIC_BASE_URL
	if not api_key:
		raise RuntimeError(
			"MINIMAX_API_KEY not set. Add it to .env or pass api_key=... "
			"to get_ai_client()."
		)
	return AnthropicClient(model=model, api_key=api_key, base_url=base_url, **kwargs)


def build_anthropic_client(model: str, **kwargs) -> AnthropicClient:
	"""Build an AnthropicClient targeting real Anthropic.

	Reads ``ANTHROPIC_API_KEY`` from env unless ``api_key`` is given.
	"""
	api_key = kwargs.pop("api_key", None) or os.environ.get("ANTHROPIC_API_KEY")
	if not api_key:
		raise RuntimeError(
			"ANTHROPIC_API_KEY not set. Add it to .env or pass api_key=... "
			"to get_ai_client()."
		)
	return AnthropicClient(model=model, api_key=api_key, **kwargs)
