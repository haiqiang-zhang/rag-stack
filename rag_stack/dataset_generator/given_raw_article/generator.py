import importlib
import itertools
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel

from rag_stack.ai_clients import get_ai_client
from rag_stack.ai_clients.base import AIClient
from rag_stack.dataset_generator.base import AbstractDatasetGenerator, GeneratedDataset
from rag_stack.dataset_generator.schema import Corpus, QA, Raw
from rag_stack.dataset_generator.given_raw_article.qa.query.prompt import QUERY_GEN_PROMPT
from rag_stack.dataset_generator.given_raw_article.qa.generation_gt.prompt import GEN_GT_SYSTEM_PROMPT
from rag_stack.dataset_generator.given_raw_article.qa.generation_gt.base import add_gen_gt

logger = logging.getLogger("RAG-Stack")


# ---------------------------------------------------------------------------
# Provider-agnostic QA functions (work with any AIClient)
# ---------------------------------------------------------------------------

class QueryResponse(BaseModel):
	query: str


class AnswerResponse(BaseModel):
	answer: str


async def generate_query(row: dict, ai_client: AIClient, lang: str = "en") -> dict:
	"""Generate a factoid question from passage contents using any AIClient."""
	context = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	context_str = "Text:\n" + "\n".join([f"{i + 1}. {c}" for i, c in enumerate(context)])

	system_msg = QUERY_GEN_PROMPT["factoid_single_hop"][lang][0].content
	user_prompt = f"{context_str}\n\nGenerated Question from the Text:\n"

	messages = [
		{"role": "system", "content": system_msg},
		{"role": "user", "content": user_prompt},
	]

	result = await ai_client.structured_output(messages, QueryResponse)
	row["query"] = result.query
	return row


async def generate_answer(row: dict, ai_client: AIClient, lang: str = "en") -> dict:
	"""Generate a concise answer from passage + question using any AIClient."""
	retrieval_gt_contents = list(itertools.chain.from_iterable(row["retrieval_gt_contents"]))
	passage_str = "\n".join(retrieval_gt_contents)
	query = row["query"]

	messages = [
		{"role": "system", "content": GEN_GT_SYSTEM_PROMPT["concise"][lang]},
		{"role": "user", "content": f"Text:\n<|text_start|>\n{passage_str}\n<|text_end|>\n\nQuestion:\n{query}\n\nAnswer:"},
	]

	result = await ai_client.structured_output(messages, AnswerResponse)
	return add_gen_gt(row, result.answer)


# Built-in step name → function mapping
_BUILTIN_STEPS = {
	"generate_query": generate_query,
	"generate_answer": generate_answer,
}


def _resolve_fn(fn_ref: str) -> Callable:
	"""Resolve a function reference: built-in name or dotted import path."""
	if fn_ref in _BUILTIN_STEPS:
		return _BUILTIN_STEPS[fn_ref]
	module_path, func_name = fn_ref.rsplit(".", 1)
	mod = importlib.import_module(module_path)
	return getattr(mod, func_name)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GivenRawArticleGenerator(AbstractDatasetGenerator):
	"""Dataset generator that starts from existing parsed documents.

	Wraps the Raw -> chunk -> Corpus -> sample -> QA pipeline.
	All LLM calls use the global model_name via ai_client.
	"""

	strategy_name = "given_raw_article"

	def __init__(
		self,
		raw_df: pd.DataFrame,
		chunk_module_name: str,
		chunk_params: Optional[Dict[str, Any]] = None,
		sample_fn: Optional[Callable] = None,
		sample_params: Optional[Dict[str, Any]] = None,
		qa_pipeline: Optional[List[Dict[str, Any]]] = None,
		model_name: str = "anthropic/claude-code/sonnet",
	):
		self.raw = Raw(raw_df)
		self.chunk_module_name = chunk_module_name
		self.chunk_params = chunk_params or {}
		self.sample_fn = sample_fn
		self.sample_params = sample_params or {}
		self.qa_pipeline = qa_pipeline or []
		self.model_name = model_name

	def generate(self, **kwargs) -> GeneratedDataset:
		start = time.time()

		# Lazy-create AI client from global model_name
		_ai_client = None

		def _get_ai_client():
			nonlocal _ai_client
			if _ai_client is None:
				_ai_client = get_ai_client(self.model_name)
			return _ai_client

		# Step 1: Chunk (Raw -> Corpus)
		corpus = self.raw.chunk(
			module_name=self.chunk_module_name, **self.chunk_params
		)

		# Step 2: Sample (Corpus -> QA skeleton)
		if self.sample_fn is None:
			raise ValueError("sample_fn is required for given_raw_article strategy")
		qa = corpus.sample(fn=self.sample_fn, **self.sample_params)

		is_claude = self.model_name.startswith("anthropic/claude-code/")

		# Step 3: QA enrichment pipeline
		for step in self.qa_pipeline:
			fn = step["fn"]
			step_kwargs = step.get("kwargs", {})

			if isinstance(fn, str) and fn == "make_retrieval_gt_contents":
				qa = qa.make_retrieval_gt_contents()
			else:
				# Inject ai_client for all LLM steps
				if "ai_client" not in step_kwargs:
					step_kwargs["ai_client"] = _get_ai_client()

				if is_claude:
					# Claude Code SDK: process rows synchronously via its own event loop
					# (batch_apply creates a separate event loop that conflicts with the SDK)
					qa = self._apply_sync(qa, fn, step.get("is_filter", False), step_kwargs)
				elif step.get("is_filter", False):
					qa = qa.batch_filter(fn=fn, **step_kwargs)
				else:
					qa = qa.batch_apply(fn=fn, **step_kwargs)

		elapsed = time.time() - start

		return GeneratedDataset(
			raw=self.raw,
			corpus=corpus,
			qa=qa,
			strategy_name=self.strategy_name,
			params={
				"model_name": self.model_name,
				"chunk_module_name": self.chunk_module_name,
				"chunk_params": self.chunk_params,
				"sample_fn": self.sample_fn.__name__ if callable(self.sample_fn) else str(self.sample_fn),
				"sample_params": self.sample_params,
			},
			elapsed_seconds=elapsed,
		)

	@staticmethod
	def _apply_sync(qa: QA, fn: Callable, is_filter: bool, kwargs: dict) -> QA:
		"""Process QA rows sequentially for Claude Code SDK.

		Each row gets its own isolated query() call — no shared context.
		Uses asyncio.run() per row since the SDK query() function is async.
		"""
		import asyncio

		qa_dicts = qa.data.to_dict(orient="records")
		results = []
		for i, row in enumerate(qa_dicts):
			logger.info(f"Processing row {i+1}/{len(qa_dicts)} with Claude Code SDK...")
			result = asyncio.run(fn(row, **kwargs))
			results.append(result)

		if is_filter:
			mask = results
			return QA(qa.data[mask], qa.linked_corpus)
		return QA(pd.DataFrame(results), qa.linked_corpus)

	@classmethod
	def from_config(cls, config: Dict[str, Any]) -> "GivenRawArticleGenerator":
		raw_path = config["raw_data_path"]
		raw_df = pd.read_parquet(raw_path, engine="pyarrow")

		sample_fn_ref = config["sample_fn"]
		sample_fn = _resolve_fn(sample_fn_ref)

		qa_pipeline = []
		for step_cfg in config.get("qa_pipeline", []):
			fn_ref = step_cfg["fn"]
			if fn_ref == "make_retrieval_gt_contents":
				step = {"fn": fn_ref}
			else:
				step = {
					"fn": _resolve_fn(fn_ref),
					"kwargs": step_cfg.get("kwargs", {}),
					"is_filter": step_cfg.get("is_filter", False),
				}
			qa_pipeline.append(step)

		return cls(
			raw_df=raw_df,
			chunk_module_name=config["chunk_module_name"],
			chunk_params=config.get("chunk_params", {}),
			sample_fn=sample_fn,
			sample_params=config.get("sample_params", {}),
			qa_pipeline=qa_pipeline,
			model_name=config.get("model_name", "anthropic/claude-code/sonnet"),
		)
