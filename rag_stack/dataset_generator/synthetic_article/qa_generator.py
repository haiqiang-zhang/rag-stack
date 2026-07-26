import anyio
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

from rag_stack.ai_clients.base import AIClient
from rag_stack.dataset_generator.synthetic_article.domain_adapters.base import AbstractDomainAdapter
from rag_stack.dataset_generator.synthetic_article.postprocess import (
	postprocess_qa,
	postprocess_irrelevant_en,
	postprocess_irrelevant_zh,
	postprocess_reference_check_single_doc_en,
	postprocess_reference_check_single_doc_zh,
	postprocess_reference_check_multi_doc_en,
	postprocess_reference_check_multi_doc_zh,
)

logger = logging.getLogger("RAG-Stack")

class QAGenerator:
	"""Generates QA pairs following the RAGEval pipeline stages 3-7.

	All methods are async, using anyio task groups for parallel LLM calls.
	Concurrency is limited via CapacityLimiter to avoid overwhelming the SDK.
	"""

	def __init__(self, domain_adapter: AbstractDomainAdapter, output_dir: str):
		self.adapter = domain_adapter
		self.output_dir = Path(output_dir)
		self.lang = domain_adapter.lang
		self.prompts = {p["prompt_type"]: p for p in domain_adapter.load_qa_prompts()}

	# ── Stage 3: Single-Doc QRA ──────────────────────────────────────────

	async def run_single_doc_qra(self, ai_client: AIClient, config_dir: str, doc_dir: str, json_idx: int) -> None:
		qar_config_dir = self.output_dir / "qar" / "config"
		prompt_types_qa = self._single_doc_qa_types()
		ref_prompt = self._get_prompt("single document reference")
		config_base, doc_base = Path(config_dir), Path(doc_dir)

		async def _process_one(topic_dir, config_file):
			out_dir = qar_config_dir / topic_dir.name / str(json_idx)
			out_path = out_dir / config_file.name
			if out_path.exists():
				return
			doc_name = config_file.stem
			config = self._load_json(config_file)
			doc_path = (doc_base / topic_dir.name / str(json_idx) / config_file.stem).with_suffix(".txt")
			doc_content = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""

			# Step 1: Generate all 3 QA types in parallel
			qa_keys = list(prompt_types_qa.keys())
			qa_prompts = []
			for qa_key, prompt_type in prompt_types_qa.items():
				prompt = self._get_prompt(prompt_type)
				if prompt:
					qa_prompts.append((qa_key, prompt))
			qa_responses = [None] * len(qa_prompts)

			async def _gen_qa(idx, qa_key, prompt):
				logger.info(f"[QRA] {doc_name} → generating {prompt_types_qa[qa_key]}")
				user_prompt = prompt["user_prompt"].format(config=config)
				msgs = [{"role": "system", "content": prompt["system_prompt"]}, {"role": "user", "content": user_prompt}]
				response = await ai_client.chat(msgs)
				qa_responses[idx] = await postprocess_qa(response, prompt["system_prompt"], user_prompt, ai_client, self.lang)

			async with anyio.create_task_group() as tg:
				for idx, (qa_key, prompt) in enumerate(qa_prompts):
					tg.start_soon(_gen_qa, idx, qa_key, prompt)

			for idx, (qa_key, _) in enumerate(qa_prompts):
				if qa_responses[idx]:
					config[qa_key] = qa_responses[idx]

			# Step 2: Extract refs for all 3 QA types in parallel
			config["Generated Article"] = doc_content
			if doc_content and ref_prompt:
				ref_responses = [None] * len(qa_prompts)

				async def _gen_ref(idx, qa_key):
					if not config.get(qa_key):
						return
					logger.info(f"[QRA] {doc_name} → extracting refs for {qa_key}")
					user_prompt = ref_prompt["user_prompt"].format(doc=doc_content, qa_pairs=config[qa_key])
					msgs = [{"role": "system", "content": ref_prompt["system_prompt"]}, {"role": "user", "content": user_prompt}]
					response = await ai_client.chat(msgs)
					ref_responses[idx] = await postprocess_qa(response, ref_prompt["system_prompt"], user_prompt, ai_client, self.lang)

				async with anyio.create_task_group() as tg:
					for idx, (qa_key, _) in enumerate(qa_prompts):
						tg.start_soon(_gen_ref, idx, qa_key)

				unable = "Unable to answer" if self.lang == "en" else "无法回答"
				a_key = "answer" if self.lang == "en" else "答案"
				for idx, (qa_key, _) in enumerate(qa_prompts):
					if ref_responses[idx]:
						valid = [item for item in ref_responses[idx] if unable not in item.get(a_key, "")]
						if valid:
							config[qa_key] = ref_responses[idx]
						else:
							logger.warning(f"[QRA] {doc_name} → ref extraction returned all 'Unable to answer' for {qa_key}, keeping original QA")

			# Step 3: Save once
			out_dir.mkdir(parents=True, exist_ok=True)
			self._save_json(out_path, config)
			logger.info(f"[QRA] Single-doc QRA saved: {out_path}")

		tasks = []
		for topic_dir in sorted(config_base.iterdir()):
			if not topic_dir.is_dir():
				continue
			idx_dir = topic_dir / str(json_idx)
			if not idx_dir.exists():
				continue
			for config_file in sorted(idx_dir.glob("*.json")):
				tasks.append((_process_one, (topic_dir, config_file)))
		async with anyio.create_task_group() as tg:
			for fn, args in tasks:
				tg.start_soon(fn, *args)

	# ── Stage 4: Multi-Doc QRA ───────────────────────────────────────────

	async def run_multi_doc_qra(self, ai_client: AIClient, json_idx: int, number: int = 1) -> None:
		qar_config_dir = self.output_dir / "qar" / "config"
		multidoc_dir = self.output_dir / "qar" / "multidoc" / str(json_idx)


		all_configs = []
		for topic_dir in sorted(qar_config_dir.iterdir()):
			if not topic_dir.is_dir():
				continue
			idx_dir = topic_dir / str(json_idx)
			if not idx_dir.exists():
				continue
			for config_file in sorted(idx_dir.glob("*.json")):
				all_configs.append((topic_dir.name, config_file))
		if len(all_configs) < 2:
			logger.warning("Not enough configs for multi-doc QRA")
			return

		integration_prompt = self._get_prompt("Multi-document Information Integration Question")
		comparison_prompt = self._get_prompt("Multi-document Comparison Question")
		multi_ref_prompt = self._get_prompt("multi document reference")
		postprocess_multi_ref = postprocess_reference_check_multi_doc_en if self.lang == "en" else postprocess_reference_check_multi_doc_zh

		async def _process_pair(pair):
			(_, file1), (_, file2) = pair
			out_path = multidoc_dir / f"{file1.stem}_{file2.stem}.json"
			if out_path.exists():
				logger.info(f"Skipping existing: {out_path}")
				return
			config1_raw, config2_raw = self._load_json(file1), self._load_json(file2)
			doc1 = config1_raw.get("Generated Article", "")
			doc2 = config2_raw.get("Generated Article", "")
			# Slice configs to only Hospitalization Record + DiseaseType (same as RAGEval)
			config1 = {k: config1_raw[k] for k in ["Hospitalization Record", "DiseaseType"] if k in config1_raw}
			config2 = {k: config2_raw[k] for k in ["Hospitalization Record", "DiseaseType"] if k in config2_raw}
			result = {"config_1": file1.name, "config_2": file2.name}
			for qa_key, prompt in [("qa_multi_document_information_integration", integration_prompt), ("qa_multi_document_compare", comparison_prompt)]:
				if not prompt:
					continue
				user_prompt = prompt["user_prompt"].format(config_1=config1, config_2=config2)
				msgs = [{"role": "system", "content": prompt["system_prompt"]}, {"role": "user", "content": user_prompt}]
				response = await ai_client.chat(msgs)
				result[qa_key] = await postprocess_qa(response, prompt["system_prompt"], user_prompt, ai_client, self.lang)
			if multi_ref_prompt and doc1 and doc2:
				sentences1 = self._split_sentences(doc1)
				sentences2 = self._split_sentences(doc2)
				new_sentences_1 = [f"[{i+1}] {s}" for i, s in enumerate(sentences1)]
				new_sentences_2 = [f"[{i+1}] {s}" for i, s in enumerate(sentences2)]
				for qa_key in ["qa_multi_document_information_integration", "qa_multi_document_compare"]:
					if not result.get(qa_key):
						continue
					numbered_doc1 = "\n".join(new_sentences_1)
					numbered_doc2 = "\n".join(new_sentences_2)
					user_prompt = multi_ref_prompt["user_prompt"].format(doc_1=numbered_doc1, doc_2=numbered_doc2, qa_pairs=result[qa_key])
					msgs = [{"role": "system", "content": multi_ref_prompt["system_prompt"]}, {"role": "user", "content": user_prompt}]
					response = await ai_client.chat(msgs)
					refined = await postprocess_multi_ref(response, self.adapter.domain, new_sentences_1, new_sentences_2, multi_ref_prompt["system_prompt"], user_prompt, ai_client)
					if refined:
						result[qa_key] = refined
			multidoc_dir.mkdir(parents=True, exist_ok=True)
			self._save_json(out_path, result)
			logger.info(f"Multi-doc QRA saved: {out_path}")

		rng = random.Random(42)
		pairs = [p for p in (self._pick_pair(all_configs, rng) for _ in range(number)) if p]
		async with anyio.create_task_group() as tg:
			for pair in pairs:
				tg.start_soon(_process_pair, pair)

	# ── Stage 5: Irrelevant QRA ──────────────────────────────────────────

	async def run_irrelevant_qra(self, ai_client: AIClient, json_idx: int, number: int = 10) -> None:
		"""Same as RAGEval: serial loop, count QA items, stop when count >= number."""
		qar_config_dir = self.output_dir / "qar" / "config"
		out_path = self.output_dir / "qar" / "irrelevant.json"
		prompt = self._get_prompt("Irrelevant Unsolvable Question")
		if not prompt:
			logger.warning("No irrelevant prompt found")
			return
		postprocess_irr = postprocess_irrelevant_en if self.lang == "en" else postprocess_irrelevant_zh

		doc_files = []
		for topic_dir in sorted(qar_config_dir.iterdir()):
			if not topic_dir.is_dir():
				continue
			idx_dir = topic_dir / str(json_idx)
			if not idx_dir.exists():
				continue
			for f in sorted(idx_dir.glob("*.json")):
				doc_files.append(f)

		# Resume: load existing results
		output = []
		done_sources = set()
		if out_path.exists():
			output = self._load_json(out_path)
			if isinstance(output, list):
				done_sources = {item.get("_source") for item in output if "_source" in item}

		count = len(output)
		while count < number:
			for config_file in doc_files:
				source_key = config_file.stem
				if source_key in done_sources:
					continue
				try:
					config = self._load_json(config_file)
					doc_content = config.get("Generated Article", "")
					if not doc_content:
						continue
					# Extract hospital_patient_name (same as RAGEval)
					try:
						basic_info = config['Hospitalization Record']['Basic Information']
						hospital_patient_name = basic_info['Hospital Name'] + '_' + basic_info['Name']
					except KeyError:
						hospital_patient_name = ""
					user_prompt = prompt["user_prompt"].format(doc=doc_content)
					msgs = [{"role": "system", "content": prompt["system_prompt"]}, {"role": "user", "content": user_prompt}]
					response = await ai_client.chat(msgs)
					results = await postprocess_irr(
						response, prompt["system_prompt"], user_prompt, ai_client,
						domain=self.adapter.domain, name=hospital_patient_name,
					)
					for item in results:
						item["_source"] = source_key
					output.extend(results)
					done_sources.add(source_key)
					count += len(results)
					# Save after each config for resume
					out_path.parent.mkdir(parents=True, exist_ok=True)
					self._save_json(out_path, output)
					logger.info(f"[Irrelevant QRA] {source_key}: {len(results)} items (total: {count})")
					if count >= number:
						break
				except Exception as e:
					logger.error(f"[Irrelevant QRA] Failed for {config_file.name}: {e}")
					done_sources.add(source_key)
			else:
				# All files exhausted but count < number — avoid infinite loop
				logger.warning(f"[Irrelevant QRA] All files processed but only {count}/{number} items generated")
				break

		# Final save: strip _source tags, truncate to number
		final = [{k: v for k, v in item.items() if k != "_source"} for item in output]
		self._save_json(out_path, final[:number])
		logger.info(f"Irrelevant QRA saved: {out_path} ({min(len(final), number)} items)")

	# ── Stage 6: Reference Refinement ────────────────────────────────────

	async def run_reference_refinement(self, ai_client: AIClient, json_idx: int) -> None:
		refinement_prompts = self.adapter.load_reference_refinement_prompts()
		if not refinement_prompts:
			logger.warning("No reference refinement prompts found")
			return
		ref_prompt = refinement_prompts[0]
		qar_config_dir = self.output_dir / "qar" / "config"


		postprocess_ref = postprocess_reference_check_single_doc_en if self.lang == "en" else postprocess_reference_check_single_doc_zh

		q_type_key = "question type" if self.lang == "en" else "问题类型"
		q_key = "question" if self.lang == "en" else "问题"
		a_key = "answer" if self.lang == "en" else "答案"

		async def _refine_one(config_file):
			config = self._load_json(config_file)
			doc_content = config.get("Generated Article", "")
			if not doc_content:
				return
			sentences = self._split_sentences(doc_content)
			new_sentences = [f"[{i+1}] {s}" for i, s in enumerate(sentences)]

			# Prepare stripped QA pairs (remove ref, keep only question type/question/answer) — same as RAGEval
			qa_pairs_by_key = {}
			for qa_key in self._single_doc_qa_keys():
				if not config.get(qa_key):
					continue
				qa_pairs_by_key[qa_key] = [
					{q_type_key: item[q_type_key], q_key: item[q_key], a_key: item[a_key]}
					for item in config[qa_key]
				]

			# Generate all 3 ref refinements in parallel
			ref_responses = {}

			async def _refine_qa_key(qa_key, qa_pairs):
				user_prompt = ref_prompt["user_prompt"].format(doc=new_sentences, qa_pairs=qa_pairs)
				msgs = [{"role": "system", "content": ref_prompt["system_prompt"]}, {"role": "user", "content": user_prompt}]
				response = await ai_client.chat(msgs)
				ref_responses[qa_key] = await postprocess_ref(response, new_sentences, ref_prompt["system_prompt"], user_prompt, ai_client)

			async with anyio.create_task_group() as tg:
				for qa_key, qa_pairs in qa_pairs_by_key.items():
					tg.start_soon(_refine_qa_key, qa_key, qa_pairs)

			for qa_key, refined in ref_responses.items():
				if refined:
					config[qa_key] = refined
			self._save_json(config_file, config)
			logger.info(f"Reference refined: {config_file}")

		tasks = []
		for topic_dir in sorted(qar_config_dir.iterdir()):
			if not topic_dir.is_dir():
				continue
			idx_dir = topic_dir / str(json_idx)
			if not idx_dir.exists():
				continue
			for config_file in sorted(idx_dir.glob("*.json")):
				tasks.append((_refine_one, (config_file,)))
		async with anyio.create_task_group() as tg:
			for fn, args in tasks:
				tg.start_soon(fn, *args)

	# ── Stage 7: Keypoint Generation ─────────────────────────────────────

	async def run_keypoint_generation(self, ai_client: AIClient, json_idx: int) -> None:
		keypoint_prompts = self.adapter.load_keypoint_prompts()
		if not keypoint_prompts:
			logger.warning("No keypoint prompts found")
			return
		kp_prompt = keypoint_prompts[0]
		qar_config_dir = self.output_dir / "qar" / "config"
		q_key = "question" if self.lang == "en" else "问题"
		a_key = "answer" if self.lang == "en" else "答案"


		async def _keypoint_one(config_file):
			config = self._load_json(config_file)
			for qa_key in self._single_doc_qa_keys():
				if not config.get(qa_key):
					continue
				for item in config[qa_key]:
					if "keypoints" in item:
						continue
					answer = item.get(a_key, "")
					if not answer:
						continue
					user_prompt = kp_prompt["user_prompt"].format(question=item.get(q_key, ""), answer=answer)
					msgs = [{"role": "system", "content": kp_prompt["system_prompt"]}, {"role": "user", "content": user_prompt}]
					response = await ai_client.chat(msgs)
					item["keypoints"] = [kp.strip() for kp in response.strip().split("\n") if kp.strip()]
			self._save_json(config_file, config)
			logger.info(f"Keypoints generated: {config_file}")

		async with anyio.create_task_group() as tg:
			for topic_dir in sorted(qar_config_dir.iterdir()):
				if not topic_dir.is_dir():
					continue
				idx_dir = topic_dir / str(json_idx)
				if not idx_dir.exists():
					continue
				for config_file in sorted(idx_dir.glob("*.json")):
					tg.start_soon(_keypoint_one, config_file)

	# ── Helpers ───────────────────────────────────────────────────────────

	def _single_doc_qa_types(self) -> Dict[str, str]:
		if self.lang == "en":
			return {"qa_fact_based": "Factual Question", "qa_multi_hop": "Multi-hop Reasoning Question", "qa_summary": "Summarization Question"}
		return {"qa_fact_based": "事实性问题", "qa_multi_hop": "多跳推理问题", "qa_summary": "总结性问题"}

	def _single_doc_qa_keys(self) -> List[str]:
		return ["qa_fact_based", "qa_multi_hop", "qa_summary"]

	def _get_prompt(self, prompt_type: str) -> Optional[dict]:
		return self.prompts.get(prompt_type)

	def _split_sentences(self, text: str) -> List[str]:
		"""Split text into sentences. Copied from RAGEval's split_sentences / cut_sent."""
		if self.lang == "en":
			from pysbd import Segmenter
			return Segmenter(language="en").segment(text)
		else:
			# RAGEval's cut_sent for Chinese
			para = re.sub('([。！？\\?])([^\u201c\u2018])', r"\1\n\2", text)
			para = re.sub('(\\.{6})([^\u201c\u2018])', r"\1\n\2", para)
			para = re.sub('(\\…{2})([^\u201c\u2018])', r"\1\n\2", para)
			para = re.sub('([。！？\\?][\u201d\u2019])([^，。！？\\?])', r'\1\n\2', para)
			para = para.rstrip()
			return [s for s in para.split("\n") if s]

	@staticmethod
	def _pick_pair(configs: list, rng: random.Random):
		"""Pick two configs from different topics (same as RAGEval: random.sample(topic_list, 2))."""
		if len(configs) < 2:
			return None
		topics = {}
		for name, path in configs:
			topics.setdefault(name, []).append((name, path))
		keys = list(topics.keys())
		if len(keys) < 2:
			logger.warning("Not enough different topics for multi-doc pair selection")
			return None
		t1, t2 = rng.sample(keys, 2)
		return (rng.choice(topics[t1]), rng.choice(topics[t2]))

	@staticmethod
	def _load_json(path) -> dict:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)

	@staticmethod
	def _save_json(path, data) -> None:
		with open(path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)
