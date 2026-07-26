import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from rag_stack.ai_clients.base import AIClient

logger = logging.getLogger("RAG-Stack")


class AbstractDomainAdapter(ABC):
	"""Base class for domain-specific article generation adapters.

	Each adapter wraps RAGEval's s1 (config generation) and s2 (article generation)
	logic for a specific domain (medical, law, finance).
	"""

	domain: str
	lang: str

	def __init__(self, schema_dir: str, lang: str = "en"):
		self.schema_dir = schema_dir
		self.lang = lang
		self._schema_path = Path(schema_dir) / self.domain / lang

	def _load_json(self, filename: str) -> Any:
		path = self._schema_path / filename
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)

	@abstractmethod
	def load_schema_templates(self) -> Dict[str, Any]:
		"""Load the document schema templates (e.g., hospitalization record template)."""
		...

	@abstractmethod
	def load_topic_catalog(self) -> Dict[str, List[dict]]:
		"""Load available topics organized by type.

		Returns dict like {"Infectious Diseases": [{"Name": "Hepatitis A", ...}, ...]}
		"""
		...

	@abstractmethod
	async def generate_config(
		self,
		ai_client: AIClient,
		topic_type: str,
		topic_item: dict,
		schema_template: dict,
		random_seed: int = 42,
	) -> dict:
		"""Generate a filled config JSON for one topic (s1 logic)."""
		...

	@abstractmethod
	async def generate_article(
		self, ai_client: AIClient, config: dict, topic_type: str, topic_item: dict
	) -> str:
		"""Convert a config JSON to narrative article text (s2 logic)."""
		...

	def _load_jsonl(self, filename: str) -> List[dict]:
		"""Load a JSONL file from the prompts directory."""
		path = Path(self.schema_dir) / "prompts" / filename
		prompts = []
		with open(path, "r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if line:
					prompts.append(json.loads(line))
		return prompts

	def load_article_prompts(self) -> Dict[str, dict]:
		"""Load s1/s2 article generation prompts from JSONL file."""
		prompts = self._load_jsonl(f"{self.domain}_{self.lang}_article.jsonl")
		return {p["prompt_type"]: p for p in prompts}

	def load_qa_prompts(self) -> List[dict]:
		"""Load QA prompt templates from JSONL file."""
		return self._load_jsonl(f"{self.domain}_{self.lang}.jsonl")

	def load_keypoint_prompts(self) -> List[dict]:
		return self._load_jsonl("keypoint_prompt.jsonl")

	def load_reference_refinement_prompts(self) -> List[dict]:
		return self._load_jsonl("reference_refinement_prompt.jsonl")
