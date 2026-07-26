import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from rag_stack.dataset_generator.schema import Corpus, QA, Raw
from rag_stack.utils.preprocess import cast_qa_dataset, cast_corpus_dataset, validate_qa_from_corpus_dataset
from rag_stack.utils.config import load_yaml_config

logger = logging.getLogger("RAG-Stack")


@dataclass
class GeneratedDataset:
	"""Output of a dataset generation strategy.

	Contains Raw (full articles), Corpus (chunked passages), and QA (question-answer pairs).
	Can be directly consumed by StaticRAGEvaluatorQualityOnly and Controller.
	"""

	raw: Raw
	corpus: Corpus
	qa: QA
	strategy_name: str
	params: Dict[str, Any] = field(default_factory=dict)
	elapsed_seconds: float = 0.0
	artifacts: Dict[str, Any] = field(default_factory=dict)

	def validate(self) -> None:
		"""Check required columns exist in corpus and qa data."""
		assert self.corpus is not None and self.corpus.data is not None, "corpus.data is None"
		assert self.qa is not None and self.qa.data is not None, "qa.data is None"

		required_corpus_cols = {"doc_id", "contents", "metadata"}
		assert required_corpus_cols.issubset(self.corpus.data.columns), (
			f"corpus missing columns: {required_corpus_cols - set(self.corpus.data.columns)}"
		)
		required_qa_cols = {"qid", "query", "retrieval_gt", "generation_gt"}
		assert required_qa_cols.issubset(self.qa.data.columns), (
			f"qa missing columns: {required_qa_cols - set(self.qa.data.columns)}"
		)

	def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
		"""Return cast and validated (qa_df, corpus_df) tuple."""
		qa_df = cast_qa_dataset(self.qa.data.copy())
		corpus_df = cast_corpus_dataset(self.corpus.data.copy())
		validate_qa_from_corpus_dataset(qa_df, corpus_df)
		return qa_df, corpus_df

	def to_parquet(self, qa_path: str, corpus_path: str) -> None:
		"""Save QA and Corpus to parquet files for reuse.

		Saves all QA columns (including extra metadata like question_type,
		references, keypoints). The evaluator only reads the standard columns
		(qid, query, retrieval_gt, generation_gt) so extra columns are safe.
		"""
		qa_df = self.qa.data.reset_index(drop=True)
		qa_df.to_parquet(qa_path)
		corpus_df = self.corpus.data.reset_index(drop=True)
		corpus_df.to_parquet(corpus_path)


class AbstractDatasetGenerator(ABC):
	"""Strategy interface for producing GeneratedDataset.

	Concrete subclasses implement generate() which returns a GeneratedDataset.
	"""

	strategy_name: str = ""
	model_name: str = ""

	@abstractmethod
	def generate(self, **kwargs) -> GeneratedDataset:
		"""Execute the dataset generation strategy.

		Subclasses should record elapsed time internally via self._elapsed_seconds.
		"""
		...

	@classmethod
	@abstractmethod
	def from_config(cls, config: Dict[str, Any]) -> "AbstractDatasetGenerator":
		"""Construct from a config dict (typically loaded from YAML)."""
		...

	@classmethod
	def from_yaml(cls, yaml_path: str) -> "AbstractDatasetGenerator":
		"""Construct from a YAML config file.

		Uses load_yaml_config() for env var substitution and type conversion.
		"""
		config = load_yaml_config(yaml_path)
		return cls.from_config(config)
