from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from rag_stack.cost_model.faiss_hnsw_sim import coefficients
from rag_stack.cost_model.faiss_hnsw_sim import train


_TARGET = ""
_COUNTER = None


def _fake_train(base_vecs, dataset, embedding, *, max_n=200_000):
	with _COUNTER.get_lock():
		_COUNTER.value += 1
	time.sleep(0.5)
	Path(_TARGET).write_bytes(b"valid")
	return _TARGET, {}


def _build_once_worker():
	train.get_or_build_predictor(
		"dataset", "embedding", np.zeros((2, 2), dtype=np.float32),
	)


def test_get_or_build_predictor_serializes_processes(monkeypatch, tmp_path):
	global _TARGET, _COUNTER
	_TARGET = str(tmp_path / "predictor.joblib")
	ctx = mp.get_context("fork")
	_COUNTER = ctx.Value("i", 0)
	monkeypatch.setattr(coefficients, "dataset_predictor_path", lambda key: _TARGET)
	monkeypatch.setattr(train, "train_from_vectors", _fake_train)
	monkeypatch.setattr(
		train, "_valid_predictor",
		lambda path: os.path.isfile(path) and Path(path).read_bytes() == b"valid",
	)
	workers = [ctx.Process(target=_build_once_worker) for _ in range(3)]
	for worker in workers:
		worker.start()
	for worker in workers:
		worker.join(10)
	assert [worker.exitcode for worker in workers] == [0, 0, 0]
	assert _COUNTER.value == 1


def test_atomic_predictor_publish_cleans_temp(tmp_path):
	class FakePredictor:
		def save(self, path):
			Path(path).write_bytes(b"complete")

	destination = tmp_path / "nested" / "predictor.joblib"
	destination.parent.mkdir()
	train._atomic_save_predictor(FakePredictor(), str(destination))
	assert destination.read_bytes() == b"complete"
	assert not list(destination.parent.glob("predictor.joblib.tmp.*"))
