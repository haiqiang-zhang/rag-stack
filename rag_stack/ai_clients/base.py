"""Base AI client interface for rag-stack."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type

from pydantic import BaseModel


class AIClient(ABC):
	"""Base AI client interface.

	All AI clients share this interface. Provider-specific capabilities
	are accessed via properties (e.g. async_client for OpenAI).
	"""

	provider: str = ""
	model: str = ""

	@abstractmethod
	async def structured_output(
		self,
		messages: List[dict],
		response_format: Type[BaseModel],
		**kwargs,
	) -> BaseModel:
		"""Call LLM with structured output (Pydantic model response)."""
		...

	@abstractmethod
	async def chat(self, messages: List[dict], **kwargs) -> str:
		"""Simple chat completion, return text."""
		...

	async def aclose(self) -> None:
		"""Release long-lived async resources (HTTP connection pools).

		Must be awaited on the event loop the client was used from — an HTTP
		pool closed at GC time lands on a loop that may already be closed and
		surfaces as an unretrieved "Event loop is closed" task exception.
		Default no-op for providers without persistent connections
		(CLI-subprocess clients); SDK-backed clients override.
		"""
		return None
