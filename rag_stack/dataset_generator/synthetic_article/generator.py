import anyio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from rag_stack.ai_clients import get_ai_client
from rag_stack.ai_clients.base import AIClient
from rag_stack.dataset_generator.base import AbstractDatasetGenerator, GeneratedDataset
from rag_stack.dataset_generator.schema import Corpus, QA, Raw
from rag_stack.dataset_generator.synthetic_article.domain_adapters.base import AbstractDomainAdapter
from rag_stack.dataset_generator.synthetic_article.domain_adapters.medical import MedicalAdapter
from rag_stack.dataset_generator.synthetic_article.qa_generator import QAGenerator

logger = logging.getLogger("RAG-Stack")



_DOMAIN_ADAPTERS = {
	"medical": MedicalAdapter,
}


class SyntheticArticleGenerator(AbstractDatasetGenerator):
	"""Dataset generator using RAGEval-style synthetic article generation.

	Pipeline: schema → config → doc → QAR → reference refinement → keypoint → aggregate
	"""

	strategy_name = "synthetic_article"

	def __init__(
		self,
		schema_dir: str,
		output_dir: str,
		domain: str = "medical",
		lang: str = "en",
		json_idx: int = 0,
		num_multi_doc: int = 1,
		num_irrelevant: int = 10,
		model_name: str = "anthropic/claude-code/sonnet",
		chunk_module_name: str = "langchain_chunk",
		chunk_params: Optional[Dict[str, Any]] = None,
		random_seed: int = 42,
	):
		self.schema_dir = schema_dir
		self.output_dir = output_dir
		self.domain = domain
		self.lang = lang
		self.json_idx = json_idx
		self.num_multi_doc = num_multi_doc
		self.num_irrelevant = num_irrelevant
		self.model_name = model_name
		self.chunk_module_name = chunk_module_name
		self.chunk_params = chunk_params or {"chunk_method": "recursivecharacter", "chunk_size": 512, "chunk_overlap": 50}
		self.random_seed = random_seed

	def generate(self, **kwargs) -> GeneratedDataset:
		start = time.time()

		adapter = self._create_adapter()

		output_path = Path(self.output_dir)
		config_dir = output_path / "config"
		doc_dir = output_path / "doc"

		ai_client = get_ai_client(self.model_name)
		qa_gen = QAGenerator(domain_adapter=adapter, output_dir=self.output_dir)

		# [s1] Config generation
		logger.info("=== Stage s1: Config Generation ===")
		anyio.run(self._run_config_generation_async, ai_client, adapter, config_dir)

		# [s2] Article generation
		logger.info("=== Stage s2: Article Generation ===")
		anyio.run(self._run_article_generation_async, ai_client, adapter, config_dir, doc_dir)

		# [3] Single-Doc QRA
		logger.info("=== Stage 3: Single-Doc QRA ===")
		anyio.run(qa_gen.run_single_doc_qra, ai_client, str(config_dir), str(doc_dir), self.json_idx)

		# [4] Multi-Doc QRA
		logger.info("=== Stage 4: Multi-Doc QRA ===")
		anyio.run(qa_gen.run_multi_doc_qra, ai_client, self.json_idx, self.num_multi_doc)

		# [5] Irrelevant QRA
		logger.info("=== Stage 5: Irrelevant QRA ===")
		anyio.run(qa_gen.run_irrelevant_qra, ai_client, self.json_idx, self.num_irrelevant)

		# [6] Reference Refinement
		logger.info("=== Stage 6: Reference Refinement ===")
		anyio.run(qa_gen.run_reference_refinement, ai_client, self.json_idx)

		# [7] Keypoint Generation
		logger.info("=== Stage 7: Keypoint Generation ===")
		anyio.run(qa_gen.run_keypoint_generation, ai_client, self.json_idx)

		# [8] Aggregate into GeneratedDataset
		logger.info("=== Stage 8: Aggregation ===")
		raw = self._build_raw(doc_dir)
		corpus = raw.chunk(module_name=self.chunk_module_name, **self.chunk_params)
		qa = self._build_qa(output_path, corpus)

		elapsed = time.time() - start

		dataset = GeneratedDataset(
			raw=raw,
			corpus=corpus,
			qa=qa,
			strategy_name=self.strategy_name,
			params={
				"domain": self.domain,
				"lang": self.lang,
				"json_idx": self.json_idx,
				"model_name": self.model_name,
				"num_multi_doc": self.num_multi_doc,
				"num_irrelevant": self.num_irrelevant,
			},
			elapsed_seconds=elapsed,
		)

		# Save final parquets
		final_dir = output_path / "final"
		final_dir.mkdir(parents=True, exist_ok=True)
		raw.data.to_parquet(str(final_dir / "raw.parquet"))
		corpus.data.to_parquet(str(final_dir / "corpus.parquet"))
		qa.data.to_parquet(str(final_dir / "qa.parquet"))
		logger.info(f"Final outputs saved to {final_dir}")

		return dataset

	# ── Config Generation (s1) — parallel ────────────────────────────────

	async def _run_config_generation_async(
		self, ai_client: AIClient, adapter: AbstractDomainAdapter, config_dir: Path
	) -> None:
		catalog = adapter.load_topic_catalog()
		templates = adapter.load_schema_templates()
		seed = self.random_seed

		async def _gen_one(topic_type, item, seed):
			try:
				name = item.get("Name", item.get("名称", "unknown"))
				out_dir = config_dir / topic_type / str(self.json_idx)
				out_path = out_dir / f"{name}.json"
				if out_path.exists():
					logger.info(f"Skipping existing config: {out_path}")
					return
				logger.info(f"Generating config: {topic_type}/{name}")
				config = await adapter.generate_config(
					ai_client, topic_type, item, templates, random_seed=seed
				)
				out_dir.mkdir(parents=True, exist_ok=True)
				with open(out_path, "w", encoding="utf-8") as f:
					json.dump(config, f, ensure_ascii=False, indent=2)
				logger.info(f"Saved config: {out_path}")
			except Exception as e:
				logger.error(f"[s1] Failed for {topic_type}/{item.get('Name', '?')}: {e}")

		async with anyio.create_task_group() as tg:
			for topic_type, topic_items in catalog.items():
				for item in topic_items:
					tg.start_soon(_gen_one, topic_type, item, seed)
					seed += 1

	# ── Article Generation (s2) — parallel ───────────────────────────────

	async def _run_article_generation_async(
		self, ai_client: AIClient, adapter: AbstractDomainAdapter,
		config_dir: Path, doc_dir: Path,
	) -> None:
		catalog = adapter.load_topic_catalog()

		async def _gen_one(topic_type, item):
			try:
				name = item.get("Name", item.get("名称", "unknown"))
				config_path = config_dir / topic_type / str(self.json_idx) / f"{name}.json"
				out_dir = doc_dir / topic_type / str(self.json_idx)
				out_path = out_dir / f"{name}.txt"
				if out_path.exists():
					logger.info(f"Skipping existing article: {out_path}")
					return
				if not config_path.exists():
					logger.warning(f"Config not found: {config_path}")
					return
				with open(config_path, "r", encoding="utf-8") as f:
					config = json.load(f)
				logger.info(f"Generating article: {topic_type}/{name}")
				article = await adapter.generate_article(ai_client, config, topic_type, item)
				out_dir.mkdir(parents=True, exist_ok=True)
				with open(out_path, "w", encoding="utf-8") as f:
					f.write(article)
				logger.info(f"Saved article: {out_path}")
			except Exception as e:
				logger.error(f"[s2] Failed for {topic_type}/{item.get('Name', '?')}: {e}")

		async with anyio.create_task_group() as tg:
			for topic_type, topic_items in catalog.items():
				for item in topic_items:
					tg.start_soon(_gen_one, topic_type, item)

	# ── Aggregation (Stage 8) ────────────────────────────────────────────

	def _build_raw(self, doc_dir: Path) -> Raw:
		"""Collect all generated articles into a Raw DataFrame."""
		rows = []
		for txt_file in sorted(doc_dir.rglob("*.txt")):
			content = txt_file.read_text(encoding="utf-8")
			rows.append({
				"raw_id": str(uuid.uuid4()),
				"contents": content,
				"path": str(txt_file),
				"texts": content,  # Required by chunk module
			})
		df = pd.DataFrame(rows)
		logger.info(f"Built Raw with {len(df)} documents")
		return Raw(df)

	def _build_qa(self, output_path: Path, corpus: Corpus) -> QA:
		"""Aggregate all QA from intermediate results into a QA DataFrame."""
		qar_config_dir = output_path / "qar" / "config"
		multidoc_dir = output_path / "qar" / "multidoc" / str(self.json_idx)
		irrelevant_path = output_path / "qar" / "irrelevant.json"

		q_key = "question" if self.lang == "en" else "问题"
		a_key = "answer" if self.lang == "en" else "答案"
		qt_key = "question type" if self.lang == "en" else "问题类型"

		all_qa = []

		# Single-doc QA
		for config_file in sorted(qar_config_dir.rglob(f"*/{self.json_idx}/*.json")):
			config = self._load_json(config_file)
			doc_content = config.get("Generated Article", "")
			# Find matching corpus doc_ids by content overlap
			doc_ids = self._find_corpus_doc_ids(doc_content, corpus)

			for qa_key in ["qa_fact_based", "qa_multi_hop", "qa_summary"]:
				for item in config.get(qa_key, []):
					all_qa.append(self._format_qa_row(item, doc_ids, q_key, a_key, qt_key))

		# Multi-doc QA
		if multidoc_dir.exists():
			for md_file in sorted(multidoc_dir.glob("*.json")):
				md_data = self._load_json(md_file)
				for qa_key in ["qa_multi_document_information_integration", "qa_multi_document_compare"]:
					for item in md_data.get(qa_key, []):
						all_qa.append(self._format_qa_row(item, [], q_key, a_key, qt_key))

		# Irrelevant QA
		if irrelevant_path.exists():
			irr_data = self._load_json(irrelevant_path)
			for item in irr_data:
				all_qa.append(self._format_qa_row(item, [], q_key, a_key, qt_key))

		df = pd.DataFrame(all_qa)
		logger.info(f"Built QA with {len(df)} entries")
		return QA(df, corpus)

	def _format_qa_row(
		self, item: dict, doc_ids: List[str],
		q_key: str, a_key: str, qt_key: str,
	) -> dict:
		refs = item.get("ref", [])
		# Flatten refs if they are dicts (multi-doc format)
		ref_texts = []
		if isinstance(refs, list):
			for r in refs:
				if isinstance(r, dict):
					ref_texts.append(r.get("content", str(r)))
				else:
					ref_texts.append(str(r))

		return {
			"qid": str(uuid.uuid4()),
			"query": item.get(q_key, ""),
			"generation_gt": [item.get(a_key, "")],
			"retrieval_gt": [doc_ids] if doc_ids else [],
			"question_type": item.get(qt_key, ""),
			"references": ref_texts,
			"keypoints": item.get("keypoints", []),
			"domain": self.domain,
			"lang": self.lang,
		}

	@staticmethod
	def _find_corpus_doc_ids(doc_content: str, corpus: Corpus) -> List[str]:
		"""Find corpus chunk doc_ids that contain content from this document."""
		if not doc_content or corpus.data is None:
			return []
		matched = []
		for _, row in corpus.data.iterrows():
			chunk_text = row["contents"]
			if chunk_text in doc_content:
				matched.append(row["doc_id"])
		return matched

	# ── Utilities ─────────────────────────────────────────────────────────

	def _create_adapter(self) -> AbstractDomainAdapter:
		adapter_cls = _DOMAIN_ADAPTERS.get(self.domain)
		if adapter_cls is None:
			raise ValueError(f"Unsupported domain: {self.domain}. Available: {list(_DOMAIN_ADAPTERS.keys())}")
		return adapter_cls(schema_dir=self.schema_dir, lang=self.lang)

	@staticmethod
	def _load_json(path) -> Any:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)

	@classmethod
	def from_config(cls, config: Dict[str, Any]) -> "SyntheticArticleGenerator":
		return cls(
			schema_dir=config["schema_dir"],
			output_dir=config["output_dir"],
			domain=config.get("domain", "medical"),
			lang=config.get("lang", "en"),
			json_idx=config.get("json_idx", 0),
			num_multi_doc=config.get("num_multi_doc", 1),
			num_irrelevant=config.get("num_irrelevant", 10),
			model_name=config.get("model_name", "anthropic/claude-code/sonnet"),
			chunk_module_name=config.get("chunk_module_name", "langchain_chunk"),
			chunk_params=config.get("chunk_params"),
			random_seed=config.get("random_seed", 42),
		)
