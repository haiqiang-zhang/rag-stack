"""Bind the concrete profile files consumed by lazy component pricing.

The numerical models retain their existing loaders. A bound service rejects a
changed dependency before and after lazy pricing; it never silently switches
curves. Complete profile values, including absence, enter the JSON contract.
"""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import fields
from functools import wraps
import json
from pathlib import Path


def _stamp(path: Path):
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _document(path: Path):
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    if path.suffix == ".json":
        return {"format": "json", "value": json.loads(data)}
    # Existing fitted predictor files remain opaque data here. Restoration
    # never deserializes this value; the component's normal loader owns that.
    return {"format": "base64", "value": base64.b64encode(data).decode("ascii")}


class PricingBindings:
    def __init__(self, paths: dict[str, Path], *, profile_root: Path | None = None):
        self.paths = {name: Path(path) for name, path in paths.items()}
        self.profile_root = profile_root
        before = {name: _stamp(path) for name, path in self.paths.items()}
        self.documents = {name: _document(path) for name, path in self.paths.items()}
        self.stamps = {name: _stamp(path) for name, path in self.paths.items()}
        if before != self.stamps:
            self._changed("profile files changed during binding")

    @staticmethod
    def _changed(profile):
        from rag_stack.cost_model.compiled import CompilationBlocked

        raise CompilationBlocked(
            "consume_plan", "pricing_binding_changed",
            "A bound component pricing profile changed; compile a new preview",
            profile=profile,
        )

    def validate(self):
        if self.profile_root is not None:
            from rag_stack.calibration_paths import calibration_profile_root
            if calibration_profile_root() != self.profile_root:
                self._changed("calibration_profile_root")
        for name, path in self.paths.items():
            try:
                stamp = _stamp(path)
                if stamp == self.stamps[name]:
                    continue
                unchanged = _document(path) == self.documents[name]
            except (OSError, ValueError):
                unchanged = False
            if not unchanged:
                self._changed(name)
            self.stamps[name] = stamp

    def wrap(self, fn):
        @wraps(fn)
        def bound(*args, **kwargs):
            self.validate()
            result = fn(*args, **kwargs)
            self.validate()
            return result

        return bound

    def attach(self, latency):
        latency.pricing_binding["profiles"] = deepcopy(self.documents)
        if not self.paths:
            return
        for field in fields(latency):
            if field.name in {"pricing_binding", "validate_pricing_binding"}:
                continue
            value = getattr(latency, field.name)
            if callable(value):
                setattr(latency, field.name, self.wrap(value))
            elif isinstance(value, dict):
                setattr(latency, field.name, {
                    key: self.wrap(item) if callable(item) else item
                    for key, item in value.items()
                })
        latency.validate_pricing_binding = self.validate
        self.validate()


def capture_pricing_bindings(stage_models, config, hardware):
    """Select dependencies using the same concrete component model dispatch.

    Cache entries are refreshed at binding so their values agree with the
    captured files. Existing bound plans guard those files before further use.
    No dataset files or experiment results are inspected.
    """
    from rag_stack.calibration_paths import calibration_profile_root, system_profile_path
    from rag_stack.cost_model.faiss_ivf_sim.adapter import (
        FaissIVFCostModel, _find_retrieval_module, _find_vectordb_config,
    )
    from rag_stack.cost_model.faiss_hnsw_sim.adapter import FaissHNSWCostModel
    from rag_stack.cost_model.llm_sim.genz_adapter import GenZLLMSimBase
    from rag_stack.cost_model.llm_sim.compressor_llmlingua2_sim import (
        PassageCompressorLLMLingua2CostModel,
    )
    from rag_stack.cost_model.llm_sim.calibration import (
        _default_calibration_dir, detect_hardware_key, invalidate_cache,
    )
    from rag_stack.cost_model.rrf import RRFCostModel

    paths = {}
    cpu = str(getattr(hardware, "cpu_key", "") or getattr(hardware, "cpu_preset", ""))
    _, retrieval = _find_retrieval_module(config)
    vdb = _find_vectordb_config(config, retrieval.get("vectordb", ""))
    dataset = (config.get("dataset") or {}).get("dataset_name") or vdb.get("dataset_name") or ""
    embedding = vdb.get("embedding_model") or ""
    models = tuple(stage_models.values())
    if any(isinstance(model, GenZLLMSimBase) for model in models):
        gpu = detect_hardware_key(hardware)
        paths[f"llm_sim/{gpu}"] = _default_calibration_dir() / f"{gpu}.json"
        invalidate_cache()
    for model in models:
        if isinstance(model, FaissIVFCostModel):
            from rag_stack.cost_model.faiss_ivf_sim.imbalance import (
                imbalance_key, imbalance_profile_path,
            )
            paths[f"faiss_ivf/{cpu}"] = system_profile_path("faiss_ivf", cpu)
            # init() stores the active corpus on the class for ordinary
            # sweeps. A retained plan must not follow a later run's corpus.
            model._corpus_id = str(model._corpus_id)
            corpus = model._corpus_id or (config.get("dataset") or {}).get("dataset_name") or ""
            if corpus:
                key = imbalance_key(corpus, embedding)
                paths[f"faiss_ivf/imbalance/{key}"] = Path(imbalance_profile_path(key))
                type(model)._profile_cache.pop(key, None)
        if isinstance(model, FaissHNSWCostModel):
            from rag_stack.cost_model.faiss_hnsw_sim.coefficients import (
                dataset_predictor_key, dataset_predictor_path,
            )
            from rag_stack.cost_model.faiss_hnsw_sim.workload_predictor import _DEFAULT_MODEL_PATH

            paths[f"faiss_hnsw/{cpu}"] = system_profile_path("faiss_hnsw", cpu)
            paths["faiss_hnsw/default_predictor"] = _DEFAULT_MODEL_PATH
            candidates = [(dataset, embedding)]
            candidates.extend({("sift1m", "raw"): [("sift1M", "raw")],
                               ("gist1m", "raw"): [("gist1M", "raw")],
                               ("glove", "glove"): [("glove-100-angular", "glove")]
                               }.get((dataset, embedding), []))
            for source, embed in candidates:
                if source and embed:
                    key = dataset_predictor_key(source, embed)
                    paths[f"faiss_hnsw/predictor/{key}"] = Path(dataset_predictor_path(key))
            type(model)._predictor_cache.pop((dataset, embedding), None)
        if isinstance(model, PassageCompressorLLMLingua2CostModel):
            gpu = detect_hardware_key(hardware)
            paths[f"compressor_llmlingua2/{gpu}"] = system_profile_path("compressor_llmlingua2", gpu)
        if isinstance(model, RRFCostModel):
            paths[f"retrieval_worker/{cpu}"] = system_profile_path("retrieval_worker", cpu)
    return PricingBindings(paths, profile_root=calibration_profile_root())
