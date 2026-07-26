"""Fetch the FULL MS MARCO corpus + QnA answers into the rag-stack schema.

Unlike ``load_msmarco_dataset.py`` (MarkrAI's small AutoRAG demo sample), this
builds the real thing:

  * corpus.parquet — Tevatron/msmarco-passage-corpus: 8.84M passages
    (docid, title, text) → (doc_id, contents, metadata). Passages are short
    (~60-100 words), so character chunking is ~1 chunk/passage and the
    vector count stays ≈ corpus size.
  * qa_test.parquet / qa_train.parquet — microsoft/ms_marco v2.1 dev split,
    answerable rows only (drops "No Answer Present."); 100-query test set
    (the paper protocol budget — mirrors dragonball qa_en_sample_100);
    generation_gt prefers wellFormedAnswers over answers. Two disjoint
    seed-fixed samples. ``references`` carries the GOLD passage texts
    (``passages.is_selected == 1``) — the chunker-invariant retrieval-GT text
    that cast_qa_dataset resolves into ``retrieval_gt_contents`` (rows without
    a selected passage get an empty list and fall back to generation_gt).

Outputs land NEXT TO THIS SCRIPT by default — data belongs to the repo
checkout the script runs from; never point --save_path at another checkout.

Written in streaming batches so peak RAM stays bounded.
"""
import os
import pathlib
from datetime import datetime

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def _write_corpus(save_path: str, batch_rows: int = 200_000) -> int:
	corpus = load_dataset(
		"Tevatron/msmarco-passage-corpus", split="train", streaming=True
	)
	out = os.path.join(save_path, "corpus.parquet")
	if os.path.exists(out):
		raise ValueError(f"{out} already exists")
	schema = pa.schema([
		("doc_id", pa.string()),
		("contents", pa.string()),
		("metadata", pa.struct([("last_modified_datetime", pa.timestamp("us"))])),
	])
	now = datetime.now()
	writer = pq.ParquetWriter(out, schema)
	doc_ids, contents, n = [], [], 0
	try:
		for row in corpus:
			title = (row.get("title") or "").strip()
			text = (row.get("text") or "").strip()
			doc_ids.append(f"mm_{row['docid']}")
			contents.append(f"{title}\n{text}" if title and title != "-" else text)
			if len(doc_ids) >= batch_rows:
				writer.write_batch(pa.record_batch([
					pa.array(doc_ids),
					pa.array(contents),
					pa.array([{"last_modified_datetime": now}] * len(doc_ids),
					         type=schema.field("metadata").type),
				], schema=schema))
				n += len(doc_ids)
				print(f"[corpus] {n:,} passages written", flush=True)
				doc_ids, contents = [], []
		if doc_ids:
			writer.write_batch(pa.record_batch([
				pa.array(doc_ids),
				pa.array(contents),
				pa.array([{"last_modified_datetime": now}] * len(doc_ids),
				         type=schema.field("metadata").type),
			], schema=schema))
			n += len(doc_ids)
	finally:
		writer.close()
	print(f"[corpus] DONE: {n:,} passages -> {out}", flush=True)
	return n


def _write_qa(save_path: str, n_test: int = 100, n_train: int = 200,
              seed: int = 43) -> None:
	qa = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
	qa = qa.select_columns(
		["query_id", "query", "answers", "wellFormedAnswers", "passages"])

	def _gt(row):
		wfa = [a for a in (row["wellFormedAnswers"] or []) if a and a.strip()]
		ans = [a for a in (row["answers"] or [])
		       if a and a.strip() and a.strip() != "No Answer Present."]
		return wfa or ans

	def _refs(row):
		p = row["passages"]
		return [t.strip() for s, t in zip(p["is_selected"], p["passage_text"])
		        if s == 1 and t and t.strip()]

	rows = []
	n_no_ref = 0
	for row in qa:
		gt = _gt(row)
		if gt:
			refs = _refs(row)
			n_no_ref += not refs
			rows.append((str(row["query_id"]), row["query"], gt, refs))
	print(f"[qa] answerable rows: {len(rows):,} "
	      f"({n_no_ref:,} without a selected gold passage)", flush=True)
	rng = np.random.RandomState(seed)
	idx = rng.permutation(len(rows))
	import pandas as pd
	def _df(sel):
		return pd.DataFrame(
			[(rows[i][0], rows[i][1], np.array(rows[i][2], dtype=object),
			  np.array(rows[i][3], dtype=object))
			 for i in sel],
			columns=["qid", "query", "generation_gt", "references"],
		)
	_df(idx[:n_test]).to_parquet(os.path.join(save_path, "qa_test.parquet"), index=False)
	_df(idx[n_test:n_test + n_train]).to_parquet(
		os.path.join(save_path, "qa_train.parquet"), index=False)
	print(f"[qa] DONE: {n_test} test + {n_train} train", flush=True)


@click.command()
@click.option("--save_path", type=str,
              default=str(pathlib.PurePath(__file__).parent),
              help="Directory for corpus.parquet / qa_*.parquet")
def main(save_path):
	_write_qa(save_path)          # small first — fail fast on schema/network
	_write_corpus(save_path)


if __name__ == "__main__":
	main()
