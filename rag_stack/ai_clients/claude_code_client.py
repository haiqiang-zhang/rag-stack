"""Claude Code SDK client for rag-stack.

Supports two modes:
- Isolated: each call is independent (no context sharing) — uses query() function
- Stateful: multi-turn conversation with shared context — uses ClaudeSDKClient

Isolated mode is default (for dataset generation).
Stateful mode is used by autonomous optimizer.
"""

import json
import logging
import re
from typing import Any, List, Optional, Type
from pydantic import BaseModel

from rag_stack.ai_clients.base import AIClient

logger = logging.getLogger("RAG-Stack")


class ClaudeCodeClient(AIClient):
	"""Claude Code SDK client wrapper.

	Two modes controlled by `stateful` flag:
	- stateful=False (default): each call uses query() — isolated context, no threading needed
	- stateful=True: uses ClaudeSDKClient — persistent conversation across calls
	"""

	provider = "claude_code"

	def __init__(
		self,
		model: str = "sonnet",
		system_prompt: str = "",
		permission_mode: str = "acceptEdits",
		allowed_tools: Optional[list] = None,
		max_turns: Optional[int] = 1,
		thinking: bool = False,
		effort: Optional[str] = "low",
		cwd: Optional[str] = None,
		stateful: bool = False,
		max_budget_usd: Optional[float] = None,
		output_format: Optional[dict] = None,
		max_concurrency: int = 10,
	):
		self.model = model
		self._system_prompt = system_prompt
		self._permission_mode = permission_mode
		self._allowed_tools = allowed_tools if allowed_tools is not None else []
		self._max_turns = max_turns
		self._thinking = {"type": "adaptive"} if thinking else {"type": "disabled"}
		self._effort = effort
		self._max_budget_usd = max_budget_usd
		self._output_format = output_format
		self._cwd = cwd
		self._stateful = stateful
		self._limiter: Optional[Any] = None
		self._max_concurrency = max_concurrency

		# Stateful mode only
		self._client = None
		self._client_ctx = None

	# ------------------------------------------------------------------
	# Isolated mode: query() function — no shared context
	# ------------------------------------------------------------------

	def _get_limiter(self):
		"""Lazy-init CapacityLimiter (must be created inside async context)."""
		if self._limiter is None:
			import anyio
			self._limiter = anyio.CapacityLimiter(self._max_concurrency)
		return self._limiter

	async def _isolated_query(self, prompt: str, system_prompt_override: Optional[str] = None) -> str:
		"""Send a single isolated query using the SDK query() function.

		Each call is independent — no conversation history.
		Concurrency is limited by CapacityLimiter to avoid overwhelming the SDK.
		"""
		from claude_agent_sdk import (
			AssistantMessage,
			ClaudeAgentOptions,
			TextBlock,
			query,
		)

		async with self._get_limiter():
			system_prompt = system_prompt_override if system_prompt_override is not None else self._system_prompt
			options_kwargs: dict[str, Any] = dict(
				model=self.model,
				system_prompt=system_prompt,
				permission_mode=self._permission_mode,
				allowed_tools=self._allowed_tools,
				max_turns=self._max_turns,
				thinking=self._thinking,
				effort=self._effort,
			)
			if self._max_budget_usd is not None:
				options_kwargs["max_budget_usd"] = self._max_budget_usd
			if self._output_format is not None:
				options_kwargs["output_format"] = self._output_format
			options = ClaudeAgentOptions(**options_kwargs)
			if self._cwd:
				options.cwd = self._cwd

			response_text = ""
			async for message in query(prompt=prompt, options=options):
				if isinstance(message, AssistantMessage):
					for block in message.content:
						if isinstance(block, TextBlock):
							response_text += block.text
			return response_text

	# ------------------------------------------------------------------
	# Stateful mode: ClaudeSDKClient — shared conversation context
	# ------------------------------------------------------------------

	async def _ensure_stateful_client(self, system_prompt: Optional[str] = None):
		"""Initialize the stateful ClaudeSDKClient (lazy, once)."""
		if self._client is not None:
			return self._client

		from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

		opts_kwargs: dict[str, Any] = dict(
			model=self.model,
			system_prompt=system_prompt or self._system_prompt,
			permission_mode=self._permission_mode,
			allowed_tools=self._allowed_tools,
			thinking=self._thinking,
			effort=self._effort,
		)
		if self._max_budget_usd is not None:
			opts_kwargs["max_budget_usd"] = self._max_budget_usd
		opts = ClaudeAgentOptions(**opts_kwargs)
		if self._cwd:
			opts.cwd = self._cwd

		self._client_ctx = ClaudeSDKClient(options=opts)
		self._client = await self._client_ctx.__aenter__()
		return self._client

	async def _stateful_query(self, text: str) -> str:
		"""Send a query on the persistent conversation. Context is shared across calls."""
		from claude_agent_sdk import AssistantMessage, TextBlock

		client = await self._ensure_stateful_client()
		await client.query(text)

		response_text = ""
		async for msg in client.receive_response():
			if isinstance(msg, AssistantMessage):
				for block in msg.content:
					if isinstance(block, TextBlock):
						response_text += block.text
		return response_text

	# ------------------------------------------------------------------
	# Public API (delegates to isolated or stateful based on mode)
	# ------------------------------------------------------------------

	async def query_text(self, prompt: str) -> str:
		"""Send a query and return text response.

		Isolated or stateful depending on how the client was created.
		"""
		if self._stateful:
			return await self._stateful_query(prompt)
		return await self._isolated_query(prompt)

	async def ensure_client(self, system_prompt: Optional[str] = None):
		"""Initialize stateful client. Only valid in stateful mode."""
		if not self._stateful:
			return None
		return await self._ensure_stateful_client(system_prompt)

	async def structured_output(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		**kwargs,
	) -> BaseModel:
		"""Send messages and parse JSON response into Pydantic model."""
		prompt_parts = []
		for msg in messages:
			role = msg.get("role", "user")
			content = msg.get("content", "")
			prompt_parts.append(f"{role}: {content}")
		prompt = "\n\n".join(prompt_parts)
		prompt += f"\n\nRespond ONLY with valid JSON matching this schema: {response_format.model_json_schema()}"

		response_text = await self.query_text(prompt)

		# Parse JSON from response
		m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response_text, re.DOTALL)
		json_str = m.group(1) if m else response_text.strip()

		try:
			data = json.loads(json_str)
		except json.JSONDecodeError:
			m2 = re.search(r"\{.*\}", response_text, re.DOTALL)
			if m2:
				data = json.loads(m2.group(0))
			else:
				raise ValueError(f"Could not parse JSON from Claude response: {response_text[:200]}")

		return response_format.model_validate(data)

	async def chat(self, messages: List[dict], **kwargs) -> str:
		"""Send messages by properly separating system prompt and user content.

		System prompt is passed via ClaudeAgentOptions.system_prompt,
		user messages are concatenated as the prompt string.

		kwargs:
			output_format: optional dict like {"type": "json_schema", "schema": {...}}
		"""
		system_prompt = ""
		user_parts = []
		for msg in messages:
			role = msg.get("role", "user")
			content = msg.get("content", "")
			if role == "system":
				system_prompt = content
			else:
				user_parts.append(content)
		prompt = "\n\n".join(user_parts)
		return await self._isolated_query(prompt, system_prompt_override=system_prompt)

	def __repr__(self):
		mode = "stateful" if self._stateful else "isolated"
		return f"ClaudeCodeClient(model='{self.model}', mode='{mode}')"
