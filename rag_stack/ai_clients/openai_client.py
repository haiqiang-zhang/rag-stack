"""OpenAI API client for rag-stack."""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Type

import anyio

from pydantic import BaseModel

from rag_stack.ai_clients.base import AIClient

logger = logging.getLogger("RAG-Stack")


def _classify_openai_error(e: BaseException) -> str:
	"""Route an OpenAI(-compatible) API error for retry/fallback decisions.

	Returns one of:

	* ``"format_unsupported"`` — the server REJECTED ``response_format`` /
	  json_object (an HTTP 400 naming it). Switching to prompt-only JSON is the
	  correct, one-time fallback here.
	* ``"fatal"`` — auth (401), forbidden/insufficient-credits (402/403), or any
	  other 4xx bad-request that a retry or a mode-switch cannot fix → raise fast.
	* ``"transient"`` — timeout / connection drop / 429 / 5xx. Retry the SAME
	  mode (bounded); do NOT switch modes. This is the key distinction: a timeout
	  is the judge being SLOW, not the format being unsupported — re-sending the
	  same slow request as prompt-only just compounds load (the cascade that
	  blew DeepEval's gather deadline). Unknown errors default here (a bounded
	  retry is safe; cascading is what we avoid).
	"""
	status = getattr(e, "status_code", None)
	if status is None:
		status = getattr(getattr(e, "response", None), "status_code", None)
	if status == 400:
		msg = str(e).lower()
		if any(k in msg for k in ("response_format", "json_object", "json mode", "json schema")):
			return "format_unsupported"
		return "fatal"
	if status in (401, 402, 403):
		return "fatal"
	if status == 429 or (isinstance(status, int) and 500 <= status < 600):
		return "transient"
	try:
		from openai import APITimeoutError, APIConnectionError, RateLimitError
		if isinstance(e, (APITimeoutError, APIConnectionError, RateLimitError)):
			return "transient"
		from openai import AuthenticationError, PermissionDeniedError
		if isinstance(e, (AuthenticationError, PermissionDeniedError)):
			return "fatal"
	except Exception:
		pass
	return "transient"


def _is_openai_length_finish_error(e: BaseException) -> bool:
	"""Return whether structured generation stopped at its output-token cap.

	The OpenAI SDK exposes this as ``LengthFinishReasonError``.  Keep the
	message fallback for compatible clients that surface the same finish reason
	without importing the SDK's concrete exception type.
	"""
	try:
		from openai import LengthFinishReasonError
		if isinstance(e, LengthFinishReasonError):
			return True
	except Exception:
		pass
	return "length limit was reached" in str(e).lower()


class OpenAIClient(AIClient):
	"""OpenAI API client wrapper.

	Lazy-creates openai.AsyncClient from env var OPENAI_API_KEY.
	Also exposes the raw async_client for backward compatibility
	with existing QA generation functions.
	"""

	provider = "openai"

	def __init__(
		self,
		model: str = "gpt-5-mini-2025-08-07",
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		max_concurrency: int = 10,
		# 429 / RateLimitError retries. Provider routing (OpenRouter
		# allow_fallbacks=true) handles most rate-limit cases by falling
		# over to another provider within a single request. 2 attempts
		# with a 10s cap → ≤ ~20s of our-layer backoff.
		max_retries: int = 3,   # 07-08: aligned with the structured_output path
		structured_output_mode: str = "parse",
		request_extra_body: Optional[dict] = None,
		default_request_kwargs: Optional[dict] = None,
		**kwargs,
	):
		self._reasoning = kwargs.pop("reasoning", False)
		# Treat legacy YAML/client kwargs named extra_body as per-request
		# defaults, not AsyncClient constructor kwargs.
		legacy_extra_body = kwargs.pop("extra_body", None)
		if request_extra_body is None and legacy_extra_body is not None:
			request_extra_body = legacy_extra_body
		self.model = model
		self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
		self._base_url = base_url
		self._max_concurrency = max_concurrency
		self._max_retries = max_retries
		self._structured_output_mode = structured_output_mode
		self._request_extra_body = request_extra_body or {}
		self._default_request_kwargs = dict(default_request_kwargs or {})
		default_extra_body = self._default_request_kwargs.pop("extra_body", None)
		if default_extra_body:
			self._request_extra_body = self._merge_dicts(
				self._request_extra_body,
				default_extra_body,
			)
		self._kwargs = kwargs
		self._client = None
		self._limiter: Optional[anyio.CapacityLimiter] = None
		self._rate_limit_until: float = 0  # global pause timestamp
		# OpenRouter: disable reasoning/thinking by default to prevent
		# token budget exhaustion on models like Qwen3 that enable it.
		# Override with reasoning=True in YAML config to enable.
		self._is_openrouter = base_url and "openrouter.ai" in base_url

	@property
	def async_client(self):
		"""Return the raw openai.AsyncClient (lazy-created).

		Use this for backward compatibility with existing functions
		that take client: AsyncClient as a parameter.
		"""
		if self._client is None:
			from openai import AsyncClient

			init_kwargs = {}
			if self._api_key:
				init_kwargs["api_key"] = self._api_key
			if self._base_url:
				init_kwargs["base_url"] = self._base_url
			init_kwargs["max_retries"] = 1  # ↓ from 2: avoid compounding with our own retry
			# Cap per-request HTTP timeout at 30s. Llama-3.3-70B normal
			# completion is ~1-5s; 30s is the give-up budget for one HTTP
			# attempt. The SDK does 1 retry → max 60s per "logical call",
			# then our OpenAIClient takes over.
			init_kwargs.setdefault("timeout", 30.0)
			init_kwargs.update(self._kwargs)
			self._client = AsyncClient(**init_kwargs)
		return self._client

	async def aclose(self) -> None:
		client, self._client = self._client, None
		if client is not None:
			await client.close()

	@staticmethod
	def _merge_dicts(base: Optional[dict], override: Optional[dict]) -> dict:
		"""Recursively merge request option dictionaries."""
		merged = dict(base or {})
		for key, value in (override or {}).items():
			if (
				isinstance(value, dict)
				and isinstance(merged.get(key), dict)
			):
				merged[key] = OpenAIClient._merge_dicts(merged[key], value)
			else:
				merged[key] = value
		return merged

	def _provider_extra_body(self) -> dict:
		if self._is_openrouter and not self._reasoning:
			# OpenRouter: disable thinking entirely. `exclude` only hides the
			# reasoning trace; `enabled: false` actually prevents the model
			# from using a thinking phase. `chat_template_kwargs.enable_thinking`
			# is the Qwen3-family-specific lever that flips the chat-template
			# `<think>` tokens off — providers vary on which lever they honor,
			# so we set both for belt-and-braces. Without this, Qwen3.6 35B-A3B
			# burns 60k+ output tokens rambling and trips LengthFinishReasonError.
			#
			# `provider.order` pins the cheapest providers that survived a
			# 30-concurrent stress test on Llama-3.3-70B (AkashML / Nebius /
			# Together / DeepInfra all 30/30 OK; Parasail / Groq / SambaNova /
			# Inceptron / Novita historically 429-prone under load). Allow
			# fallback so one provider hiccup doesn't kill the call.
			return {
				"reasoning": {"enabled": False},
				"chat_template_kwargs": {"enable_thinking": False},
				"provider": {
					"order": ["AkashML", "Nebius", "Together", "DeepInfra"],
					"allow_fallbacks": True,
				},
			}
		return {}

	def _prepare_request_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
		request_kwargs = dict(self._default_request_kwargs)
		request_kwargs.update(kwargs)

		extra_body = self._provider_extra_body()
		extra_body = self._merge_dicts(extra_body, self._request_extra_body)
		if "extra_body" in request_kwargs:
			extra_body = self._merge_dicts(extra_body, request_kwargs.pop("extra_body"))
		if extra_body:
			request_kwargs["extra_body"] = extra_body
		return request_kwargs

	async def structured_output(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		**kwargs,
	) -> BaseModel:
		async with self._get_limiter():
			return await self._structured_output_locked(
				messages, response_format, **kwargs,
			)

	async def _structured_output_locked(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		**kwargs,
	) -> BaseModel:
		mode = kwargs.pop("structured_output_mode", self._structured_output_mode)
		if self._base_url and "deepseek" in str(self._base_url) and mode in {"auto", "parse"}:
			# DeepSeek rejects the json_schema response_format used by
			# beta.chat.completions.parse — fall back to json_object mode.
			mode = "json_object"
		if mode == "json_object":
			return await self._json_object_structured_output(
				messages, response_format, use_json_object=True, **kwargs,
			)
		if mode == "prompt_json":
			return await self._json_object_structured_output(
				messages, response_format, use_json_object=False, **kwargs,
			)
		if mode not in {"auto", "parse"}:
			raise ValueError(
				"structured_output_mode must be one of: "
				"'auto', 'parse', 'json_object', 'prompt_json'"
			)
		try:
			return await self._parse_structured_output(messages, response_format, **kwargs)
		except Exception as parse_error:
			if mode != "auto":
				raise
			fallback_kwargs = dict(kwargs)
			if _is_openai_length_finish_error(parse_error):
				# Contextual-precision schemas contain one verdict+reason object per
				# retrieved passage.  The normal 1024-token judge cap is deliberately
				# conservative, but high-top-k cases can require a few thousand output
				# tokens.  Retrying/falling back at the *same* cap deterministically
				# produced another truncated JSON document, which DeepEval then
				# silently dropped from a partial metric mean.  Expand only calls that
				# actually hit the finish-length guard; normal judge calls retain the
				# low cap and latency profile.
				current_raw = fallback_kwargs.get(
					"max_tokens", self._default_request_kwargs.get("max_tokens", 1024),
				)
				try:
					current = max(1, int(current_raw))
				except (TypeError, ValueError):
					current = 1024
				expanded = max(current, min(8192, max(4096, current * 4)))
				fallback_kwargs["max_tokens"] = expanded
				logger.warning(
					"[OpenAI] structured output reached max_tokens=%s for model=%s; "
					"retrying schema parse once with max_tokens=%s",
					current,
					self.model,
					expanded,
				)
				try:
					return await self._parse_structured_output(
						messages, response_format, **fallback_kwargs,
					)
				except Exception as expanded_parse_error:
					parse_error = expanded_parse_error
			logger.warning(
				"[OpenAI] structured parse failed for model=%s base_url=%s; "
				"falling back to JSON-object prompting: %s",
				self.model,
				self._base_url,
				parse_error,
			)
			try:
				return await self._json_object_structured_output(
					messages, response_format, use_json_object=True, **fallback_kwargs,
				)
			except Exception as fallback_error:
				raise RuntimeError(
					"structured_output failed in parse mode and JSON fallback"
				) from fallback_error

	async def _parse_structured_output(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		**kwargs,
	) -> BaseModel:
		client = self.async_client
		request_kwargs = self._prepare_request_kwargs(kwargs)
		model = request_kwargs.pop("model", self.model)
		# Retry loop for empty/rate-limited responses. Some OpenRouter providers
		# (Novita Llama-3.3-70B) intermittently return BOTH .parsed=None and
		# .content=None under load — looks like a 200 OK with empty body. We
		# back off and retry up to N times before bubbling up. Kept low (2)
		# because provider routing already auto-falls-over; if 2 retries fail,
		# it's a deterministic provider/schema issue, no point burning more.
		empty_retries = 2
		for empty_attempt in range(empty_retries + 1):
			completion = await client.beta.chat.completions.parse(
				model=model,
				messages=messages,
				response_format=response_format,
				**request_kwargs,
			)
			parsed = completion.choices[0].message.parsed
			if parsed is not None:
				return parsed
			content = completion.choices[0].message.content
			if content is not None:
				# Provider returned text but didn't validate the schema —
				# coerce via Pydantic, fall back to raw string for the
				# caller's downstream parser (DeepEval's trimAndLoadJson).
				try:
					if hasattr(response_format, "model_validate_json"):
						return response_format.model_validate_json(content)
					if hasattr(response_format, "parse_raw"):
						return response_format.parse_raw(content)
				except Exception:
					pass
				return content
			# Both .parsed and .content are None → empty response. Back off
			# and retry. Doubles each time: 0.5s, 1s, 2s, 4s.
			if empty_attempt < empty_retries:
				import anyio as _anyio
				await _anyio.sleep(0.5 * (2 ** empty_attempt) + random.random() * 0.3)
		raise RuntimeError(
			"OpenAI structured-output returned both .parsed=None and "
			".content=None after retries (provider rate-limited or down)"
		)

	def _validate_structured_content(
		self,
		content: str,
		response_format: Type[BaseModel],
	) -> Any:
		try:
			if hasattr(response_format, "model_validate_json"):
				return response_format.model_validate_json(content)
			if hasattr(response_format, "parse_raw"):
				return response_format.parse_raw(content)
		except Exception:
			pass

		import re as _re
		match = _re.search(r"\{.*\}", content, _re.DOTALL)
		if match:
			try:
				if hasattr(response_format, "model_validate_json"):
					return response_format.model_validate_json(match.group(0))
				if hasattr(response_format, "parse_raw"):
					return response_format.parse_raw(match.group(0))
			except Exception:
				pass
		return content

	async def _json_object_structured_output(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		use_json_object: bool = True,
		**kwargs,
	) -> Any:
		"""Structured output fallback for OpenAI-compatible JSON-mode servers.

		Uses ``response_format={"type":"json_object"}`` when available with the
		target pydantic schema injected into the prompt, then validates the returned
		JSON into ``response_format``. Falls back to extracting the first ``{...}`` and,
		last resort, returns the raw content string for the caller's own parser.
		"""
		import json as _json
		client = self.async_client
		try:
			schema = _json.dumps(response_format.model_json_schema())
		except AttributeError:
			schema = _json.dumps(response_format.schema())
		except Exception:
			schema = ""
		instr = (
			"Respond with ONLY a single JSON object that conforms to this JSON schema "
			"(no markdown, no code fences, no prose):\n" + schema
		)
		msgs = list(messages) + [{"role": "system", "content": instr}]
		request_kwargs = self._prepare_request_kwargs(kwargs)
		model = request_kwargs.pop("model", self.model)
		last = None
		last_error = None
		for attempt in range(3):
			try:
				call_kwargs = dict(request_kwargs)
				if attempt and "timeout" not in call_kwargs:
					# Escalating per-request timeout. The client default (30s) is
					# tuned for short judge calls; metrics that ship the whole
					# retrieval context (e.g. contextual precision on agentic
					# configs) can legitimately need more, and a request that
					# timed out once will time out again at the SAME budget —
					# re-sending the identical doomed request is what burned GT
					# seats (07-08: 3-5/15 cases failing all attempts with zero
					# HTTP responses logged). Double the budget each retry.
					_base_to = float(self._kwargs.get("timeout", 30.0) or 30.0)
					call_kwargs["timeout"] = _base_to * (2 ** attempt)
				if use_json_object:
					call_kwargs["response_format"] = {"type": "json_object"}
				completion = await client.chat.completions.create(
					model=model, messages=msgs, **call_kwargs,
				)
			except Exception as e:
				last_error = e
				kind = _classify_openai_error(e)
				# Switch to prompt-only ONLY when the server genuinely rejects
				# json_object (HTTP 400 naming response_format). A timeout / 429 /
				# 5xx is the judge being SLOW, not a format problem — switching
				# modes there re-sends the same slow request and compounds load
				# (the deepseek timeout→prompt-only cascade that blew DeepEval's
				# gather deadline). Retry the SAME mode instead.
				if kind == "format_unsupported" and use_json_object:
					logger.warning(
						"[OpenAI] model=%s rejects json_object response_format; "
						"switching to prompt-only JSON validation: %s", model, e,
					)
					return await self._json_object_structured_output(
						messages, response_format, use_json_object=False, **kwargs,
					)
				if kind == "fatal":
					# auth / 402 insufficient-credits / bad params — a retry or
					# mode-switch cannot fix it; surface it so the judge-down
					# guard upstream sees it instead of burning the retry budget.
					raise
				# transient (timeout / connection / 429 / 5xx) → bounded retry of
				# the SAME mode (the SDK already did its own single retry).
				await anyio.sleep(0.4 * (2 ** attempt) + random.random() * 0.2)
				continue
			content = completion.choices[0].message.content
			if content:
				last = content
				validated = self._validate_structured_content(content, response_format)
				if not isinstance(validated, str):
					return validated
			await anyio.sleep(0.4 * (2 ** attempt) + random.random() * 0.2)
		if last is not None:
			return last
		if last_error is not None:
			raise RuntimeError("structured_output JSON fallback failed") from last_error
		raise RuntimeError("structured_output JSON fallback: empty response after retries")

	async def _deepseek_structured_output(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		**kwargs,
	) -> Any:
		return await self._json_object_structured_output(
			messages, response_format, use_json_object=True, **kwargs,
		)

	def _get_limiter(self) -> anyio.CapacityLimiter:
		if self._limiter is None:
			self._limiter = anyio.CapacityLimiter(self._max_concurrency)
		return self._limiter

	async def chat(self, messages: List[dict], **kwargs) -> str:
		async with self._get_limiter():
			from openai import RateLimitError
			client = self.async_client
			request_kwargs = self._prepare_request_kwargs(kwargs)
			model = request_kwargs.pop("model", self.model)
			for attempt in range(self._max_retries):
				# Global pause: if another task hit 429, wait for it
				now = time.monotonic()
				if self._rate_limit_until > now:
					wait = self._rate_limit_until - now
					logger.info(f"[OpenAI] Waiting {wait:.1f}s for global rate limit cooldown")
					await anyio.sleep(wait)
				try:
					completion = await client.chat.completions.create(
						model=model,
						messages=messages,
						**request_kwargs,
					)
					return completion.choices[0].message.content
				except RateLimitError:
					# Exponential backoff capped at 10s. With OpenRouter
					# fallbacks handling most 429s within a single request,
					# our outer retry is mostly a belt-and-braces; keep it
					# cheap.
					wait = min(2 * (2 ** attempt) + random.random(), 10)
					self._rate_limit_until = time.monotonic() + wait
					logger.warning(f"[OpenAI] 429 rate limited, all tasks pausing {wait:.1f}s (attempt {attempt + 1}/{self._max_retries})")
					await anyio.sleep(wait)
				except Exception as e:
					# 07-08: retry TRANSIENT failures (connection drops,
					# timeouts, 5xx) instead of raising on the first one —
					# chat() only caught 429 while the structured_output path
					# already classified and retried; a single DeepSeek
					# connection blip was costing the advisory LLM channel a
					# whole proposal round. Fatal errors (auth/4xx) and
					# exhausted budgets still dump the prompts and raise.
					if (_classify_openai_error(e) != "fatal"
							and attempt < self._max_retries - 1):
						wait = 0.5 * (2 ** attempt) + random.random() * 0.3
						logger.warning(
							f"[OpenAI] chat transient failure "
							f"({type(e).__name__}: {e}); retrying in {wait:.1f}s "
							f"(attempt {attempt + 1}/{self._max_retries})"
						)
						await anyio.sleep(wait)
						continue
					for msg in messages:
						content = msg.get("content", "")
						preview = content[:200] if isinstance(content, str) else str(content)[:200]
						logger.error(f"[OpenAI] {msg['role']} ({len(content)} chars): {preview}")
					raise
			raise RuntimeError(f"[OpenAI] Failed after {self._max_retries} rate limit retries")

	def __repr__(self):
		return f"OpenAIClient(model='{self.model}')"


CSCS_OPENAI_BASE_URL = "https://api.swissai.svc.cscs.ch/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _env_int(name: str, default: int) -> int:
	value = os.environ.get(name)
	if value is None or value == "":
		return default
	try:
		return int(value)
	except ValueError as e:
		raise ValueError(f"{name} must be an integer, got {value!r}") from e


def _env_float(name: str, default: float) -> float:
	value = os.environ.get(name)
	if value is None or value == "":
		return default
	try:
		return float(value)
	except ValueError as e:
		raise ValueError(f"{name} must be a number, got {value!r}") from e


def _env_int_any(names: List[str], default: int) -> int:
	for name in names:
		value = os.environ.get(name)
		if value is not None and value != "":
			return _env_int(name, default)
	return default


def _env_float_any(names: List[str], default: float) -> float:
	for name in names:
		value = os.environ.get(name)
		if value is not None and value != "":
			return _env_float(name, default)
	return default


def _env_str_any(names: List[str]) -> Optional[str]:
	for name in names:
		value = os.environ.get(name)
		if value is not None and value != "":
			return value
	return None


def _normalize_vllm_base_url(base_url: str) -> str:
	base_url = base_url.strip().rstrip("/")
	if not base_url:
		raise ValueError("vLLM judge base_url is empty")
	if not base_url.endswith("/v1"):
		base_url = f"{base_url}/v1"
	return base_url


def build_openrouter_client(model: str, **kwargs) -> "OpenAIClient":
	"""Build an OpenAIClient wired to OpenRouter (OpenAI-compatible).

	Reads ``OPENROUTER_API_KEY`` from env unless ``api_key`` is given.
	``base_url`` defaults to ``https://openrouter.ai/api/v1``.

	The ``OpenAIClient.__init__`` detects ``openrouter.ai`` in the base_url
	and auto-disables reasoning/thinking (``extra_body={"reasoning":
	{"enabled": False}}``) to prevent token-budget exhaustion on models that
	default-enable a thinking phase (Qwen3 series). Override with
	``reasoning=True`` to re-enable.
	"""
	api_key = kwargs.pop("api_key", None) or os.environ.get("OPENROUTER_API_KEY")
	base_url = kwargs.pop("base_url", None) or OPENROUTER_BASE_URL
	if not api_key:
		raise ValueError(
			"OPENROUTER_API_KEY not set. Add it to .env or pass api_key=... "
			"explicitly."
		)
	return OpenAIClient(
		model=model, api_key=api_key, base_url=base_url, **kwargs,
	)


def build_cscs_client(model: str, **kwargs) -> "OpenAIClient":
	"""Build an OpenAIClient wired to CSCS Swiss AI inference (OpenAI-compatible).

	Reads ``CSCS_SERVING_API`` from env unless ``api_key`` is given.
	``base_url`` defaults to ``https://api.swissai.svc.cscs.ch/v1``;
	override via kwarg if Swiss AI moves the endpoint.

	Example model_name (passed via :func:`rag_stack.ai_clients.get_ai_client`):
	``"openai/cscs/Qwen/Qwen3.5-397B-A17B-TaKf"`` →
	``OpenAIClient(model="Qwen/Qwen3.5-397B-A17B-TaKf", base_url=CSCS, api_key=$CSCS_SERVING_API)``.
	"""
	api_key = kwargs.pop("api_key", None) or os.environ.get("CSCS_SERVING_API")
	base_url = kwargs.pop("base_url", None) or CSCS_OPENAI_BASE_URL
	if not api_key:
		raise ValueError(
			"CSCS_SERVING_API not set. Add it to .env or pass api_key=... "
			"explicitly. The CSCS Swiss AI inference endpoint requires a "
			"bearer token (different from OPENAI_API_KEY / MINIMAX_API_KEY)."
		)
	return OpenAIClient(
		model=model, api_key=api_key, base_url=base_url, **kwargs,
	)


def build_deepseek_client(model: str, **kwargs) -> "OpenAIClient":
	"""Build an OpenAIClient wired to DeepSeek (OpenAI-compatible).

	Reads ``DEEPSEEK_API_KEY`` from env unless ``api_key`` is given.
	``base_url`` defaults to ``DEEPSEEK_BASE_URL`` env var or
	``https://api.deepseek.com``. Models: ``deepseek-reasoner`` (R1, strong
	reasoning) or ``deepseek-chat`` (V3, faster). Used by the agent-guided
	optimizer (``agent_qnehvi``) to read eval history + sub-metrics and
	propose configs / region preferences.
	"""
	api_key = kwargs.pop("api_key", None) or os.environ.get("DEEPSEEK_API_KEY")
	base_url = (
		kwargs.pop("base_url", None)
		or os.environ.get("DEEPSEEK_BASE_URL")
		or DEEPSEEK_BASE_URL
	)
	if not api_key:
		raise ValueError(
			"DEEPSEEK_API_KEY not set. Add it to .env or pass api_key=... "
			"explicitly."
		)
	return OpenAIClient(
		model=model, api_key=api_key, base_url=base_url, **kwargs,
	)


def build_vllm_client(model: Optional[str] = None, **kwargs) -> "OpenAIClient":
	"""Build an OpenAI-compatible client for a remote vLLM judge server.

	Runtime resolution keeps benchmark configs portable across judge hosts:

	* ``model`` comes from ``vllm/<model>`` or ``JUDGE_MODEL``.
	* ``base_url`` comes from explicit kwargs or ``JUDGE_BASE_URL``.
	* ``api_key`` defaults to ``EMPTY`` because vLLM normally has no auth.
	* ``max_concurrency`` defaults to ``JUDGE_MAX_CONCURRENCY`` or 64, matching
	  vLLM's batched serving model better than the generic OpenAI default.
	* ``max_tokens`` defaults to ``JUDGE_MAX_TOKENS`` or 1024 to keep judge
	  tail latency bounded.
	"""
	model = (
		model
		or kwargs.pop("served_model_name", None)
		or os.environ.get("JUDGE_MODEL")
		or os.environ.get("VLLM_JUDGE_MODEL")
	)
	if not model:
		raise ValueError(
			"vLLM judge model not set. Use model: vllm/<served-model-name> "
			"or set JUDGE_MODEL=<served-model-name>."
		)

	base_url = (
		kwargs.pop("base_url", None)
		or os.environ.get("JUDGE_BASE_URL")
		or os.environ.get("VLLM_JUDGE_BASE_URL")
	)
	if not base_url:
		raise ValueError(
			"vLLM judge base URL not set. Set JUDGE_BASE_URL=http://host:port/v1 "
			"or pass base_url=... explicitly."
		)
	base_url = _normalize_vllm_base_url(str(base_url))

	api_key = (
		kwargs.pop("api_key", None)
		or os.environ.get("JUDGE_API_KEY")
		or os.environ.get("VLLM_JUDGE_API_KEY")
		or "EMPTY"
	)
	max_concurrency_env = _env_str_any(
		["JUDGE_MAX_CONCURRENCY", "VLLM_JUDGE_MAX_CONCURRENCY"],
	)
	max_concurrency = kwargs.pop("max_concurrency", None)
	if max_concurrency_env is not None:
		max_concurrency = int(max_concurrency_env)
	elif max_concurrency is None:
		max_concurrency = 64
	else:
		max_concurrency = int(max_concurrency)
	max_retries_env = _env_str_any(["JUDGE_MAX_RETRIES", "VLLM_JUDGE_MAX_RETRIES"])
	max_retries = kwargs.pop("max_retries", None)
	if max_retries_env is not None:
		max_retries = int(max_retries_env)
	elif max_retries is None:
		max_retries = 2
	else:
		max_retries = int(max_retries)
	timeout_env = _env_str_any(["JUDGE_TIMEOUT", "VLLM_JUDGE_TIMEOUT"])
	if timeout_env is not None:
		kwargs["timeout"] = float(timeout_env)
	elif "timeout" not in kwargs:
		kwargs["timeout"] = _env_float_any(
			["JUDGE_TIMEOUT", "VLLM_JUDGE_TIMEOUT"], 180.0,
		)
	structured_output_mode_env = _env_str_any(
		["JUDGE_STRUCTURED_OUTPUT_MODE", "VLLM_JUDGE_STRUCTURED_OUTPUT_MODE"],
	)
	if structured_output_mode_env is not None:
		kwargs["structured_output_mode"] = structured_output_mode_env
	elif "structured_output_mode" not in kwargs:
		kwargs["structured_output_mode"] = "auto"

	default_request_kwargs = kwargs.pop("default_request_kwargs", None)
	default_request_kwargs = dict(default_request_kwargs or {})
	legacy_max_tokens = kwargs.pop("max_tokens", None)
	if legacy_max_tokens is not None:
		default_request_kwargs["max_tokens"] = int(legacy_max_tokens)
	max_tokens_env = _env_str_any(["JUDGE_MAX_TOKENS", "VLLM_JUDGE_MAX_TOKENS"])
	if max_tokens_env is not None:
		default_request_kwargs["max_tokens"] = int(max_tokens_env)
	elif "max_tokens" not in default_request_kwargs:
		default_request_kwargs["max_tokens"] = 1024

	request_extra_body = kwargs.pop("request_extra_body", None)
	legacy_extra_body = kwargs.pop("extra_body", None)
	default_extra_body = {}
	if not kwargs.get("reasoning", False):
		default_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
	request_extra_body = OpenAIClient._merge_dicts(default_extra_body, request_extra_body)
	request_extra_body = OpenAIClient._merge_dicts(request_extra_body, legacy_extra_body)

	client = OpenAIClient(
		model=model,
		api_key=api_key,
		base_url=base_url,
		max_concurrency=max_concurrency,
		max_retries=max_retries,
		request_extra_body=request_extra_body,
		default_request_kwargs=default_request_kwargs,
		**kwargs,
	)
	client.provider = "vllm"
	return client
