from datetime import datetime
import runpy
import unicodedata

import emoji
import pandas as pd

from rag_stack.utils import preprocess


def _expected_text(text: str) -> str:
	return unicodedata.normalize("NFC", emoji.demojize(text))


def test_parallel_cast_preserves_corpus_contract(monkeypatch):
	monkeypatch.setenv("RAG_STACK_PREPROC_CORES", "1")
	existing_time = datetime(2026, 1, 2, 3, 4, 5)
	source = pd.DataFrame(
		{
			"doc_id": ["none", "space", "empty", "emoji", "nfd", "meta"],
			"contents": [None, "  \n", "", "A ©", "Cafe\u0301", "plain"],
			"metadata": [
				None,
				{},
				None,
				{"label": "©"},
				{},
				{"last_modified_datetime": existing_time, "prev_id": "©"},
			],
		}
	)

	actual = preprocess.cast_corpus_dataset(source)

	assert actual["doc_id"].tolist() == ["empty", "emoji", "nfd", "meta"]
	assert actual["contents"].tolist() == [
		_expected_text(""),
		_expected_text("A ©"),
		_expected_text("Cafe\u0301"),
		_expected_text("plain"),
	]
	for metadata in actual["metadata"]:
		assert metadata["last_modified_datetime"] is not None
		assert "prev_id" in metadata
		assert "next_id" in metadata
	assert actual.iloc[1]["metadata"]["label"] == _expected_text("©")
	assert actual.iloc[3]["metadata"]["last_modified_datetime"] == existing_time
	assert actual.iloc[3]["metadata"]["prev_id"] == _expected_text("©")


def test_cast_is_noop_for_same_in_memory_corpus(monkeypatch):
	monkeypatch.setenv("RAG_STACK_PREPROC_CORES", "1")
	frame = pd.DataFrame(
		{"doc_id": ["a"], "contents": ["text"], "metadata": [{}]}
	)
	first = preprocess.cast_corpus_dataset(frame)

	def fail_if_recast(_texts):
		raise AssertionError("already-cast corpus was normalized twice")

	monkeypatch.setattr(preprocess, "_parallel_normalize", fail_if_recast)
	assert preprocess.cast_corpus_dataset(first) is first


def test_parallel_normalize_spawn_path(monkeypatch):
	monkeypatch.setenv("RAG_STACK_PREPROC_CORES", "2")
	texts = ["A ©", "Cafe\u0301"] * 25_000
	actual = preprocess._parallel_normalize(texts)
	assert actual[:2] == [_expected_text("A ©"), _expected_text("Cafe\u0301")]
	assert len(actual) == len(texts)


def test_rag_stack_main_is_safe_to_import_in_spawn_worker(monkeypatch):
	import rag_stack.cli

	def fail_if_cli_restarts():
		raise AssertionError("spawn-worker import re-entered the rag-stack CLI")

	monkeypatch.setattr(rag_stack.cli, "main", fail_if_cli_restarts)
	runpy.run_module("rag_stack.__main__", run_name="spawn_probe")
