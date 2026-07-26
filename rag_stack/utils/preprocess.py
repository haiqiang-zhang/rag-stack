import os
from datetime import datetime

import numpy as np
import pandas as pd

# NOTE: ``preprocess_text`` (from static_rag_evaluator) is imported LAZILY inside
# the two functions that use it — importing it at module top makes this
# low-level ``rag_stack.utils`` module depend on the high-level
# ``static_rag_evaluator`` package at load time, which creates a circular import
# (utils.preprocess → SRE/__init__ → retrieval/base → utils.preprocess).


# ── QA dataset contract ──────────────────────────────────────────────────
# A QA row MUST carry: qid, query, generation_gt. Everything else is OPTIONAL.
#
#   references  (optional) — the *text* of the retrieval ground truth (evidence
#       sentences / passages). When present it is THE retrieval-GT text; when
#       absent we fall back to generation_gt (the answer) as the retrieval-GT
#       text. Either spelling ``references`` / ``reference`` is accepted.
#
#   retrieval_gt (chunk-UUID GT) is DEPRECATED and dropped. ``chunk_size`` is a
#       search-space dimension, so the corpus is re-chunked per eval and chunk
#       UUIDs are not stable — a UUID-anchored retrieval GT is ill-defined.
#       Retrieval quality is therefore anchored on TEXT (token overlap of
#       references / answer vs the retrieved chunk text — the AutoRAG
#       ``retrieval_token_*`` metrics, FlashRAG-style answer-containment),
#       never on chunk identity.
REQUIRED_QA_COLUMNS = ["qid", "query", "generation_gt"]
_REFERENCE_COLUMNS = ("references", "reference")


def validate_qa_dataset(df: pd.DataFrame):
	missing = [c for c in REQUIRED_QA_COLUMNS if c not in df.columns]
	assert not missing, (
		f"QA df must have columns {REQUIRED_QA_COLUMNS} (qid, query, generation_gt "
		f"required; references optional; retrieval_gt deprecated). "
		f"Missing {missing}; got {list(df.columns)}"
	)


def validate_corpus_dataset(df: pd.DataFrame):
	columns = ["doc_id", "contents", "metadata"]
	assert set(columns).issubset(df.columns), (
		f"df must have columns {columns}, but got {df.columns}"
	)


def validate_raw_corpus(df: pd.DataFrame, *, require_raw: bool = True):
	"""Constraint for the corpus the pipeline will CHUNK (chunk_size is a
	search-space dimension), so the input MUST be the RAW (un-chunked) corpus.

	Canonical raw-corpus schema is ``{doc_id, contents}`` (legacy ``texts`` is
	reverse-aliased to ``contents`` at chunk time, so it is accepted too).

	When ``require_raw`` (chunk sweeps active), a PRE-CHUNKED corpus is rejected:
	a pre-chunked corpus carries ``start_end_idx`` (chunk-offset bookkeeping) and
	short, uniform passages. Re-chunking already-chunked passages is meaningless
	and silently corrupts retrieval — so we fail loudly with a clear message.
	"""
	has_text = ("contents" in df.columns) or ("texts" in df.columns)
	assert ("doc_id" in df.columns) and has_text, (
		f"raw corpus must have columns {{doc_id, contents}} (got {list(df.columns)})"
	)
	if require_raw and "start_end_idx" in df.columns:
		raise ValueError(
			"Corpus looks PRE-CHUNKED (carries 'start_end_idx') but chunk_size/"
			"chunker is in the search space — pass the RAW (un-chunked) corpus "
			"instead. Re-chunking an already-chunked corpus is meaningless and "
			"breaks retrieval."
		)


def _cast_text_list(x) -> list:
	"""Coerce a cell (str / list / np.ndarray / None) to a flat list[str]."""
	if x is None:
		return []
	if isinstance(x, np.ndarray):
		x = x.tolist()
	if isinstance(x, str):
		return [x]
	if isinstance(x, (list, tuple)):
		out = []
		for el in x:
			out.extend(_cast_text_list(el))
		return out
	return [str(x)]


def cast_qa_dataset(df: pd.DataFrame):
	from rag_stack_evaluator.static_rag_evaluator.utils.util import preprocess_text  # lazy: avoid utils↔SRE import cycle

	def cast_generation_gt(gt):
		if isinstance(gt, str):
			return [gt]
		elif isinstance(gt, list):
			return gt
		elif isinstance(gt, np.ndarray):
			return cast_generation_gt(gt.tolist())
		else:
			raise ValueError(f"generation_gt must be str or list, but got {type(gt)}")

	df = df.reset_index(drop=True)
	validate_qa_dataset(df)
	assert df["qid"].apply(lambda x: isinstance(x, str)).sum() == len(df), (
		"qid must be string type."
	)
	assert df["query"].apply(lambda x: isinstance(x, str)).sum() == len(df), (
		"query must be string type."
	)
	df["generation_gt"] = df["generation_gt"].apply(cast_generation_gt)
	df["query"] = df["query"].apply(preprocess_text)
	df["generation_gt"] = df["generation_gt"].apply(
		lambda x: list(map(preprocess_text, x))
	)

	# Optional references = retrieval-GT *text* (evidence sentences). Normalize
	# to list[str]; dragonball-style datasets ship these, most ship none.
	ref_col = next((c for c in _REFERENCE_COLUMNS if c in df.columns), None)
	if ref_col is not None:
		df["references"] = df[ref_col].apply(
			lambda x: [preprocess_text(t) for t in _cast_text_list(x)]
		)

	# Retrieval-GT TEXT = references when present & non-empty, else generation_gt
	# (the answer). Stored as ``retrieval_gt_contents`` (List[List[str]]) — the
	# field the AutoRAG ``retrieval_token_*`` metrics consume — so token-overlap
	# retrieval recall works WITHOUT any chunk-UUID ground truth.
	def _resolve_gt_text(row) -> list:
		refs = list(row["references"]) if ref_col is not None else []
		if refs:
			return [refs]
		return [list(row["generation_gt"])]

	df["retrieval_gt_contents"] = df.apply(_resolve_gt_text, axis=1)

	# Drop the deprecated chunk-UUID retrieval GT (see contract note above).
	df = df.drop(columns=[c for c in ("retrieval_gt",) if c in df.columns])
	return df


def _normalize_text(text: str) -> str:
	"""emoji-demojize + NFC unicode normalize — the picklable worker for the
	parallel contents pass. Inlined (rather than importing
	``util.preprocess_text``) so ``spawn`` workers don't re-import the heavy
	rag_stack tree; kept byte-identical to ``preprocess_text``."""
	import emoji
	import unicodedata
	return unicodedata.normalize("NFC", emoji.demojize(text))


def _preproc_cores() -> int:
	"""Core budget for the corpus-preprocessing pool. Set by the caller (the
	msmarco launcher exports ``RAG_STACK_PREPROC_CORES`` = cores/proc); a
	standalone run (no launcher, env unset) defaults to ``max(1, cpu_count-2)``
	— same as the faiss build-thread default, so both parallelize the box."""
	v = os.environ.get("RAG_STACK_PREPROC_CORES")
	if v and v.strip():
		try:
			return max(1, int(v))
		except ValueError:
			pass
	return max(1, (os.cpu_count() or 3) - 2)


def _parallel_normalize(texts: list) -> list:
	"""Unicode/emoji-normalize a list of strings across cores. Serial below a
	threshold (pool spawn + re-import isn't worth it on small corpora); this is
	the single hot pass at MS MARCO scale (8.84M chunks)."""
	n = len(texts)
	cores = _preproc_cores()
	if cores <= 1 or n < 50_000:
		return [_normalize_text(t) for t in texts]
	import multiprocessing as mp
	chunksize = max(1, n // (cores * 8))
	with mp.get_context("spawn").Pool(cores) as pool:
		return pool.map(_normalize_text, texts, chunksize=chunksize)


def cast_corpus_dataset(df: pd.DataFrame):
	"""Normalize a corpus DataFrame: drop empty rows, unicode/emoji-normalize
	the contents, and guarantee the metadata keys the pipeline expects
	(``last_modified_datetime`` / ``prev_id`` / ``next_id``).

	At MS MARCO scale (8.84M rows) the old form ran ~9 single-threaded pandas
	``.apply`` passes (a per-row ``datetime.now()`` storm, five metadata dict
	rebuilds, four validation re-scans, and a serial demojize over ~2.8 GB of
	text). This form does ONE parallel contents pass + ONE vectorized metadata
	pass, with the empty-row filter vectorized and the redundant assertions
	dropped (the keys are guaranteed by construction).

	IDEMPOTENT: the cast'd frame is tagged, so re-casting the SAME corpus is a
	no-op. This matters because ``DatasetManager`` casts once at init and then
	``filter_exist_ids`` re-casts ``self.corpus_data`` on EVERY vectordb index
	build — at MS MARCO scale that redundant re-cast is minutes each. A freshly
	read / re-chunked corpus carries no tag, so it still casts normally."""
	if df.attrs.get("_rag_corpus_cast"):
		return df
	df = df.reset_index(drop=True)
	validate_corpus_dataset(df)

	# drop None / all-whitespace contents (vectorized; "" is kept, matching the
	# old ``x.isspace()`` semantics — the empty string is not whitespace)
	s = df["contents"].astype("string")
	drop = s.isna() | s.str.fullmatch(r"\s+").fillna(False)
	if drop.any():
		df = df[~drop].reset_index(drop=True)

	# contents: the one hot O(N) pass, parallelized across cores
	df["contents"] = _parallel_normalize(df["contents"].tolist())

	# metadata: ONE pass — one shared timestamp (not per-row now()); ensure the
	# required keys exist; unicode-normalize string values. Replaces 5 dict-
	# rebuild passes + 4 assertion scans.
	now = datetime.now()

	def _fix_meta(m):
		m = dict(m) if isinstance(m, dict) and m else {}
		if m.get("last_modified_datetime") is None:
			m["last_modified_datetime"] = now
		m.setdefault("prev_id", None)
		m.setdefault("next_id", None)
		for k, v in m.items():
			if isinstance(v, str):
				m[k] = _normalize_text(v)
		return m

	df["metadata"] = df["metadata"].map(_fix_meta)
	df.attrs["_rag_corpus_cast"] = True
	return df


def validate_qa_from_corpus_dataset(qa_df: pd.DataFrame, corpus_df: pd.DataFrame):
	# No-op (kept for call-site compatibility): the retrieval ground truth is now
	# TEXT (references / generation_gt), not corpus chunk UUIDs, so there is
	# nothing to cross-check against the per-eval re-chunked corpus. The old
	# UUID-existence assertion was itself a crash source once ``chunk_size``
	# became a search-space dimension (qa UUIDs never match a re-chunked corpus).
	return


def load_data_from_parquet(qa_data_path: str, corpus_data_path: str):
	"""Load, cast, and validate QA and corpus datasets from parquet files."""
	if not os.path.exists(qa_data_path):
		raise ValueError(f"QA data path {qa_data_path} does not exist.")
	if not os.path.exists(corpus_data_path):
		raise ValueError(f"Corpus data path {corpus_data_path} does not exist.")
	if not qa_data_path.endswith(".parquet"):
		raise ValueError(f"QA data path {qa_data_path} is not a parquet file.")
	if not corpus_data_path.endswith(".parquet"):
		raise ValueError(f"Corpus data path {corpus_data_path} is not a parquet file.")

	qa_data = cast_qa_dataset(pd.read_parquet(qa_data_path, engine="pyarrow"))
	corpus_data = cast_corpus_dataset(pd.read_parquet(corpus_data_path, engine="pyarrow"))
	validate_qa_from_corpus_dataset(qa_data, corpus_data)
	return qa_data, corpus_data
