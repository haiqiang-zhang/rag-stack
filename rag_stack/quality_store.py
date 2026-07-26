"""Hardware-independent quality evaluation store.

A QualityStore is independent of any project or hardware configuration.  A
record is keyed by the *algorithm* configuration (the decoded configuration
with its top-level ``system`` block removed) and is valid only for one exact
corpus + QA pair.  Quality metrics and the frozen quality trace are stored
as one record so a cache hit can drive the same trace-aware CM path as a fresh
quality evaluation.
"""

from __future__ import annotations

import datetime
import ast
import fcntl
import hashlib
import itertools
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np

from rag_stack.types import EvalType
from rag_stack.evaluator_contract import normalize_backend_result, configured_backend_options
import torch
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter

from rag_stack.search_space import RagSearchSpace, SEP
from rag_stack.optimizer.utils import config_signature
from rag_stack.quality_memo import quality_memo_key
from rag_stack.quality_store_contract import (
    MANIFEST_OPTIONAL_FIELDS,
    QUALITY_STORE_KEY_KIND,
    QUALITY_STORE_KIND,
    validate_quality_store_manifest,
    validate_quality_store_record_marker,
)
from rag_stack.utils.config import load_yaml_config
from rag_stack.utils import (
    aggregate_quality_metrics,
)
from rag_stack_evaluator.static_rag_evaluator.gen_fingerprint import (
    TRANSPORT_ONLY_PARAMS,
)

logger = logging.getLogger("RAG-Stack")

from rag_stack.security import strip_secrets as _strip_secrets


def _strip_transport_only_params(value: Any) -> Any:
    """Recursively remove and normalize transport-only cache identity."""
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if key in TRANSPORT_ONLY_PARAMS:
                continue
            if (
                key == "generator_backend"
                and isinstance(item, str)
                and item in {"vllm", "vllm_api"}
            ):
                # Local vLLM is the default auxiliary-generation backend;
                # its API spelling selects transport, not generation semantics.
                continue
            normalized_item = _strip_transport_only_params(item)
            if key == "component" and normalized_item == "vllm_api":
                # Default components share the chat protocol. An explicit
                # historical generation_protocol remains in the identity,
                # so legacy native completions cannot reuse chat quality.
                normalized_item = "vllm"
            normalized[key] = normalized_item
        return normalized
    if isinstance(value, list):
        return [_strip_transport_only_params(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_transport_only_params(item) for item in value)
    return value


def _algo_config(decoded_config: dict | None) -> dict:
    """Return the hardware/transport-independent decoded configuration."""
    algo = deepcopy({
        key: value
        for key, value in (decoded_config or {}).items()
        if key != "system"
    })
    return _strip_secrets(_strip_transport_only_params(algo))


def _algo_record_key(decoded_config: dict | None) -> str:
    """Use the exact same algorithm key as the live controller memo."""
    return quality_memo_key(decoded_config or {})


def _parse_csv_cell(value: str) -> Any:
    """Restore values serialized by ``all_evaluations.csv``.

    Whole pipeline/corpus sections are written as Python-repr dictionaries.
    Leaving them as strings collapses distinct algorithm configurations onto
    incorrect cache keys, so parse only the safe container-shaped forms.
    """
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value[:1] in "[{":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def _atomic_json_dump(path: str, payload: Any) -> None:
    """Atomically replace one JSON artifact in the destination directory."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".quality-store-", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _store_lock(store_path: str):
    """Serialize manifest/record updates, including across NFS clients."""
    os.makedirs(store_path, exist_ok=True)
    lock_path = os.path.join(store_path, ".quality_store.lock")
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _persist_record_files(
    store_path: str,
    record_hash: str,
    *,
    config: dict,
    quality_metrics: dict,
    execution_dag: dict,
    pipeline_config: dict,
    record_meta: dict,
) -> dict:
    """Commit a multi-file record with an invalidated/checksummed marker."""
    paths = {
        "config": os.path.join(store_path, "configs", f"{record_hash}.json"),
        "quality": os.path.join(store_path, "quality", f"{record_hash}.json"),
        "trace": os.path.join(store_path, "traces", f"{record_hash}.json"),
        "pipeline_config": os.path.join(
            store_path, "pipeline_configs", f"{record_hash}.json",
        ),
    }
    marker_path = os.path.join(store_path, "records", f"{record_hash}.json")
    # An old marker must not bless a half-written replacement.
    try:
        os.unlink(marker_path)
    except FileNotFoundError:
        pass
    _atomic_json_dump(paths["config"], config)
    _atomic_json_dump(paths["quality"], quality_metrics)
    _atomic_json_dump(paths["trace"], execution_dag)
    _atomic_json_dump(paths["pipeline_config"], _strip_secrets(pipeline_config))
    committed = dict(record_meta)
    committed["artifact_sha256"] = {
        name: _sha256_file(path) for name, path in paths.items()
    }
    validate_quality_store_record_marker(
        committed,
        source=f"QualityStore record marker {record_hash}",
    )
    _atomic_json_dump(marker_path, committed)
    return committed


def _split_quality_payload(payload: dict) -> tuple[dict, Any]:
    """Separate persisted metrics from evaluator-private control payloads."""
    metrics = dict(payload or {})
    trace = metrics.pop("__execution_dag__", None)
    metrics.pop("__fp_hit__", None)
    # Future evaluator-private fields must never become objective metrics.
    metrics = {k: v for k, v in metrics.items() if not str(k).startswith("__")}
    return metrics, trace


def _validate_trace(
    trace: Any,
    *,
    source: str,
    expected_queries: int | None = None,
) -> dict:
    from rag_stack.rag_ir.trace import validate_quality_trace_envelope

    if trace is None:
        raise ValueError(f"{source}: missing execution_dag.json / quality trace")
    try:
        validate_quality_trace_envelope(trace)
        stored_workflow = (trace.get("provenance") or {}).get("workflow_schema")
        if stored_workflow is not None:
            from rag_stack.rag_ir.workflow_schema import WorkflowSchema, validate_workflow_trace
            validate_workflow_trace(WorkflowSchema.from_dict(stored_workflow), trace)
    except Exception as exc:
        raise ValueError(f"{source}: not a valid frozen quality trace: {exc}") from exc
    provenance = trace.get("provenance") or {}
    if provenance.get("lossy") is True:
        raise ValueError(f"{source}: lossy projected traces are not cacheable")
    if expected_queries is not None and len(trace.get("queries") or []) != expected_queries:
        raise ValueError(
            f"{source}: trace/QA cardinality mismatch "
            f"({len(trace.get('queries') or [])} != {expected_queries})"
        )
    return trace


def _configured_metric_names(config: dict | None) -> list[str]:
    return [
        str(metric.get("metric_name") or metric.get("metric"))
        for metric in ((config or {}).get("eval_backend_setting") or {}).get(
            "metrics", []
        )
        if isinstance(metric, dict)
        and (metric.get("metric_name") or metric.get("metric"))
    ]


def _quality_import_source_candidate(row: dict) -> bool:
    """Whether a project row can own reusable quality artifacts.

    Normal successful rows are eligible.  Historical cost-model runs can
    additionally contain an ``OOM`` row *after* the expensive quality run
    completed: those rows carry real metrics and a frozen trace, while only
    the source system's CM performance failed.  A later system transfer must
    reprice that algorithm instead of discarding its quality observation.

    The narrow ``CM:`` provenance guard prevents a GT/measured-runtime OOM
    from being reinterpreted as reusable quality.  Artifact existence,
    metric completeness and strict trace validity are checked separately
    before a candidate is committed.
    """

    eval_type = str(row.get("eval_type") or "").strip()
    if eval_type in {
        EvalType.GT.value,
        EvalType.GT_FP.value,
        EvalType.CM.value,
    }:
        return True
    return (
        eval_type == EvalType.OOM.value
        and str(row.get("eval_error") or "").lstrip().startswith("CM:")
    )


def _has_complete_numeric_quality(
    quality_metrics: dict,
    configured_metrics: list[str],
) -> bool:
    """Require every configured metric to be a finite JSON number."""

    for name in configured_metrics:
        value = quality_metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not np.isfinite(float(value)):
            return False
    return True


def _quality_spec(config: dict | None) -> dict:
    setting = deepcopy((config or {}).get("eval_backend_setting") or {})
    setting["_evaluator_backend"] = str(
        ((config or {}).get("global") or {}).get("eval_backend", "static_gt")
    ).strip().lower()
    # Keep the whole evaluator setting, not just metric names.  Fields such as
    # judge model/prompt kwargs, ``submetric_query_subset`` and
    # ``queries_per_trial`` change the produced observation even when the
    # public metric name remains identical.  Operational-only differences may
    # cause a conservative miss, which is preferable to a semantically wrong
    # hit that changes optimizer history.
    return _canonical_quality_spec(setting)


def _canonical_quality_spec(spec: dict | None) -> dict:
    """Canonical quality/judge semantics attached to every cached record.

    Metric declaration order does not change a score, so sort the declarations
    by their canonical JSON form.  All other list order is preserved: it may be
    meaningful to an aggregation/prompt and an unnecessary miss is safer than
    reusing a score produced by different judge semantics.
    """
    clean = _strip_secrets(deepcopy(spec or {}))
    # Historical records came from the default static evaluator. Explicitly
    # namespace new plugins so equal metric names cannot reuse another backend's
    # quality/trace pair. backend_options already carries scorer/model identity.
    clean.setdefault("_evaluator_backend", "static_gt")
    metrics = clean.get("metrics") or []
    if isinstance(metrics, list):
        clean["metrics"] = sorted(
            metrics,
            key=lambda value: json.dumps(
                value, sort_keys=True, separators=(",", ":"), default=str,
            ),
        )
    else:
        clean["metrics"] = metrics
    clean.setdefault("combined_quality", None)
    return clean


def _quality_specs_match(left: dict | None, right: dict | None) -> bool:
    """Exact, secret-free equality for metric implementation/judge settings."""
    return _canonical_quality_spec(left) == _canonical_quality_spec(right)


def _record_priority(record: dict | None) -> tuple[int, int, int]:
    """Prefer complete metric sets, then the newest whole metrics+trace pair."""
    record = record or {}
    return (
        1 if record.get("full") else 0,
        len(record.get("metric_names") or []),
        int(record.get("source_mtime_ns") or 0),
    )


def _norm_dataset_path(path: Any) -> str:
    if path is None:
        return ""
    value = str(path).strip()
    if not value:
        return ""
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        try:
            return os.path.relpath(os.path.abspath(expanded), os.getcwd())
        except ValueError:
            return os.path.abspath(expanded)
    return os.path.normpath(expanded)


def _file_fingerprint(path: Any) -> dict:
    value = str(path or "").strip()
    if not value:
        return {}
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(expanded)
    if not os.path.isfile(expanded):
        return {}
    digest = hashlib.sha256()
    with open(expanded, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": os.path.getsize(expanded)}


def _parquet_num_rows(path: Any) -> int | None:
    value = os.path.abspath(os.path.expanduser(str(path or "")))
    if not os.path.isfile(value):
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(value).metadata.num_rows)
    except Exception:
        return None


def _qa_semantic_fingerprint(path: Any) -> str | None:
    """Hash the quality-relevant QA rows independent of parquet metadata.

    Project materialization adds derived columns such as
    ``retrieval_gt_contents``.  Hash only the authoritative QA fields so the
    source parquet and its normalized project copy retain the same identity.
    """
    value = os.path.abspath(os.path.expanduser(str(path or "")))
    if not os.path.isfile(value):
        return None
    fields = (
        "qid", "query", "generation_gt", "references", "keypoints",
        "question_type", "domain", "lang",
    )
    try:
        import pyarrow.parquet as pq

        schema_names = set(pq.ParquetFile(value).schema_arrow.names)
        columns = [name for name in fields if name in schema_names]
        if "query" not in columns or "generation_gt" not in columns:
            return None
        rows = pq.read_table(value, columns=columns).to_pylist()
        payload = json.dumps(
            rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        return None


def _dataset_identity(
    config: dict | None,
    qa_data_path: str | None = None,
    corpus_data_path: str | None = None,
) -> dict:
    dataset_cfg = (config or {}).get("dataset") or {}
    qa = qa_data_path or dataset_cfg.get("qa") or ""
    corpus = corpus_data_path or dataset_cfg.get("corpus") or ""
    raw_corpus = (
        ((config or {}).get("algo_search_space") or {})
        .get("corpus", {})
        .get("raw_path", "")
    )
    return {
        "dataset_name": str(dataset_cfg.get("dataset_name") or ""),
        "qa": _norm_dataset_path(qa),
        "corpus": _norm_dataset_path(corpus),
        "raw_corpus": _norm_dataset_path(raw_corpus),
        "qa_fingerprint": _file_fingerprint(qa),
        "qa_semantic_sha256": _qa_semantic_fingerprint(qa),
        "corpus_fingerprint": _file_fingerprint(corpus),
        "raw_corpus_fingerprint": _file_fingerprint(raw_corpus),
        "qa_rows": _parquet_num_rows(qa),
    }


def _dataset_matches(left: dict | None, right: dict | None) -> bool:
    if not left or not right:
        return False
    left_qa = left.get("qa_fingerprint") or {}
    right_qa = right.get("qa_fingerprint") or {}
    left_corpus = (
        left.get("raw_corpus_fingerprint")
        or left.get("corpus_fingerprint")
        or {}
    )
    right_corpus = (
        right.get("raw_corpus_fingerprint")
        or right.get("corpus_fingerprint")
        or {}
    )
    left_qa_identity = left.get("qa_semantic_sha256") or left_qa.get("sha256")
    right_qa_identity = right.get("qa_semantic_sha256") or right_qa.get("sha256")
    if left_qa_identity and right_qa_identity and left_corpus.get("sha256") and right_corpus.get("sha256"):
        rows_match = (
            left.get("qa_rows") is None
            or right.get("qa_rows") is None
            or int(left.get("qa_rows")) == int(right.get("qa_rows"))
        )
        return (
            left_qa_identity == right_qa_identity
            and left_corpus.get("sha256") == right_corpus.get("sha256")
            and rows_match
        )
    keys = ("dataset_name", "qa", "corpus", "raw_corpus")
    return all(str(left.get(k, "")) == str(right.get(k, "")) for k in keys)


class QualityStore:
    """Hardware-independent quality evaluation store.

    Independent of project/hardware.  Can be reused across multiple hardware
    replays and grows incrementally as new configs are evaluated.

    Persistence layout::

        {store_path}/
            manifest.json
            configs/{hash}.json          # decoded ALGO config (no system)
            quality/{hash}.json          # quality_metrics from GT evaluator
            traces/{hash}.json           # frozen quality trace
            pipeline_configs/{hash}.json # full resolved pipeline config
            records/{hash}.json          # completeness/source metadata
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @staticmethod
    def quality_spec(config: dict | None) -> dict:
        """Return the cache-visible quality/judge semantics of a RAG config."""
        return _quality_spec(config)

    @staticmethod
    def quality_specs_match(left: dict | None, right: dict | None) -> bool:
        """Whether two records were scored under identical quality semantics."""
        return _quality_specs_match(left, right)

    def __init__(
        self,
        config_path: str,
        qa_data_path: str | None = None,
        corpus_data_path: str | None = None,
        store_path: str = "",
        dataset: Optional[Any] = None,
    ):
        """Load config, dataset, build search space + GT evaluator.

        Does NOT initialise hardware / StaticAssembly.
        """
        self.store_path = os.path.abspath(store_path)
        from rag_stack.project_config import CONFIG_FREEZE_FILENAME, ProjectConfigError
        if any(os.path.exists(os.path.join(self.store_path, name)) for name in (
            "summary.json", CONFIG_FREEZE_FILENAME,
        )):
            raise ProjectConfigError(
                "QualityStore destination is a project directory; use a separate store path."
            )

        # Set PROJECT_DIR before loading config so ${PROJECT_DIR} expands correctly
        os.environ["PROJECT_DIR"] = self.store_path

        self.config = load_yaml_config(config_path)
        # Pre-flight config validation
        from rag_stack.config_validator import ConfigValidator
        errors = ConfigValidator(self.config).validate()
        if errors:
            raise ValueError(
                f"Config validation failed ({len(errors)} error(s)):\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # Search space adapter (shared for all RAG-specific decode/grid ops);
        # ``self.search_space`` is the raw Ax SearchSpace consumed by paths
        # that don't need the adapter.
        self._ssops = RagSearchSpace.from_yaml(self.config)
        self.search_space = self._ssops.ss
        self._base_decoded_config = self._ssops.build_base_decoded_config(
            self.config,
        )

        # Data + chunk lifecycle — owned by a single DatasetManager that the
        # quality evaluator shares. This is what makes chunk sweeps actually
        # re-chunk: _resolve_config builds corpus_runtime.chunker, and the
        # evaluator resolves + activates it through this same manager (before
        # the dedup, QualityStore built the chunker block but never chunked, so
        # chunk sweeps silently no-op'd).
        from rag_stack.dataset_manager import DatasetManager
        self._data = DatasetManager(
            project_dir=self.store_path,
            config=self.config,
            qa_data_path=qa_data_path,
            corpus_data_path=corpus_data_path,
            dataset=dataset,
        )
        # Identity is tied to the authoritative INPUT files, not the project's
        # materialized parquet copies. DatasetManager may normalize/rewrite
        # parquet metadata, so byte-hashing those copies makes two projects
        # created from the same corpus/QA appear different.
        dataset_cfg = self.config.get("dataset") or {}
        qa_identity_path = qa_data_path or dataset_cfg.get("qa")
        corpus_identity_path = corpus_data_path or dataset_cfg.get("corpus")
        self.dataset_identity = _dataset_identity(
            self.config,
            qa_data_path=qa_identity_path,
            corpus_data_path=corpus_identity_path,
        )
        self.qa_data = self._data.qa_data
        self.corpus_data = self._data.corpus_data
        self.token_stats = self._data.token_stats

        # Auto-inject N into vectordb configs + stash token stats on the config.
        self._data.inject_corpus_size()
        self.config["_token_stats"] = self.token_stats.to_dict()

        # GT evaluator shares the manager (corpus + chunk cache are one instance).
        from rag_stack.evaluation.registry import create_quality_evaluator
        backend = str((self.config.get("global") or {}).get("eval_backend", "static_gt")).lower()
        self.rag_quality_evaluator = create_quality_evaluator(
            dataset_manager=self._data,
            project_dir=self.store_path,
            backend=backend,
            backend_options=configured_backend_options(self.config),
        )

        # Combined-quality aggregation spec (see ``aggregate_quality_metrics``
        # docstring); ``None`` = legacy mean over all non-None sub-metrics.
        self._combined_quality_spec = (
            (self.config.get("eval_backend_setting") or {}).get("combined_quality")
        )

        # ``load_full`` must be able to reopen a store on another machine.
        # Keep a self-contained copy of the source config beside the data.
        os.makedirs(self.store_path, exist_ok=True)
        store_config_path = os.path.join(self.store_path, "rag_stack_config.yaml")
        if (
            not os.path.isfile(store_config_path)
            and os.path.abspath(config_path) != os.path.abspath(store_config_path)
        ):
            shutil.copy2(config_path, store_config_path)

        # In-memory index of evaluated config hashes (loaded or built up)
        self._evaluated: Dict[str, dict] = {}  # hash → quality result dict
        self._manifest: dict = {}  # manifest data (populated by load() or _save_manifest)
        manifest_path = os.path.join(self.store_path, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f) or {}
            validate_quality_store_manifest(manifest, source=manifest_path)
            if not _dataset_matches(manifest.get("dataset"), self.dataset_identity):
                raise ValueError(
                    "Quality store corpus/QA do not match the requested sweep: "
                    f"store={manifest.get('dataset')}, "
                    f"requested={self.dataset_identity}"
                )
            self._manifest = manifest
            self._load_evaluated_records(manifest)

    @staticmethod
    def key_for_config(decoded_config: dict) -> str:
        """Public cache key: decoded algorithm config, never deployment."""
        return _algo_record_key(_algo_config(decoded_config))

    @staticmethod
    def algorithm_config(decoded_config: dict) -> dict:
        """Public canonicalizer used by importers and cache clients."""
        return _algo_config(decoded_config)

    def _canonical_algo_config(self, decoded_config: dict) -> dict:
        """Fill fixed algorithm defaults, then remove deployment dimensions."""
        enriched = self._ssops.enrich_configs(
            [decoded_config], self._base_decoded_config,
        )[0]
        return _algo_config(enriched)

    @staticmethod
    def make_dataset_identity(
        config: dict,
        *,
        qa_data_path: str | None = None,
        corpus_data_path: str | None = None,
    ) -> dict:
        return _dataset_identity(config, qa_data_path, corpus_data_path)

    def assert_dataset_compatible(
        self,
        config: dict,
        *,
        qa_data_path: str,
        corpus_data_path: str,
    ) -> None:
        """Fail closed unless this cache belongs to the exact corpus + QA."""
        candidate = _dataset_identity(config, qa_data_path, corpus_data_path)
        expected = getattr(self, "dataset_identity", None) or (
            getattr(self, "_manifest", {}) or {}
        ).get("dataset")
        if not _dataset_matches(expected, candidate):
            raise ValueError(
                "Quality store corpus/QA do not match the current project: "
                f"store={expected}, project={candidate}"
            )

    @property
    def requires_quality_trace(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Phase 1: sweep
    # ------------------------------------------------------------------

    def sweep(
        self,
        n_grid: int = 5,
        n_configs: int = 0,
        strategy: str = "exhaustive",
    ) -> dict:
        """Run quality sweep and persist results.

        Args:
            n_grid: Default number of grid points for continuous parameters.
                Individual parameters can override via ``n_grid`` in their
                YAML definition (e.g. ``{range: [0.2, 1.0], n_grid: 10}``).
                Only used by the ``"exhaustive"`` strategy.
            n_configs: Number of configs to sample.  Only used by the
                ``"sobol"`` strategy.
            strategy: Config generation strategy:
                - ``"exhaustive"``: grid discretisation + cartesian product
                - ``"sobol"``: Sobol quasi-random sequence (requires ``n_configs``)

        Returns:
            Summary dict with ``n_configs``, ``n_failed``, ``quality_summary``.
        """
        if strategy == "exhaustive":
            configs = self._enumerate_configs(n_grid)
        elif strategy == "sobol":
            if n_configs <= 0:
                raise ValueError("n_configs must be > 0 for sobol strategy")
            configs = self._sobol_configs(n_configs)
        else:
            raise NotImplementedError(f"Strategy '{strategy}' not yet supported")

        logger.info(f"QualityStore sweep ({strategy}): {len(configs)} configs to evaluate")
        return self._evaluate_and_persist(
            configs, n_grid=n_grid, n_configs_requested=n_configs, strategy=strategy,
        )

    # ------------------------------------------------------------------
    # Incremental evaluation (Phase 2b)
    # ------------------------------------------------------------------

    def evaluate_configs(self, configs: list[dict]) -> list[dict]:
        """Evaluate quality for specific configs, write results to store.

        Called by Controller during Phase 2b incremental optimisation.
        Skips configs already in the store (dedup by config_hash).

        Returns:
            List of result dicts for newly evaluated configs:
            ``[{config_hash, config, quality_metrics, quality_combined}, ...]``
        """
        new_configs: list[dict] = []
        for raw_cfg in configs:
            cfg = self._canonical_algo_config(raw_cfg)
            h = _algo_record_key(cfg)
            if h not in self._evaluated:
                new_configs.append(cfg)

        if not new_configs:
            return []

        results: list[dict] = []
        for cfg in new_configs:
            h = _algo_record_key(cfg)
            try:
                pipeline_config = self._resolve_config(cfg)
                run_dir = os.path.join(self.store_path, "runs", h)
                quality_payload = self.rag_quality_evaluator.evaluate(
                    pipeline_config, run_dir=run_dir,
                )
                quality_payload = normalize_backend_result(
                    quality_payload,
                    backend=(self.config.get("global") or {}).get("eval_backend", "static_gt"),
                    pipeline_config=pipeline_config,
                    question_ids=(self.qa_data["qid"].tolist()
                                  if getattr(self, "qa_data", None) is not None else None),
                )
                quality_metrics, execution_dag = _split_quality_payload(
                    quality_payload,
                )
                execution_dag = _validate_trace(
                    execution_dag,
                    source=f"QualityStore live eval {h}",
                    expected_queries=len(self.qa_data),
                )
                combined = aggregate_quality_metrics(quality_metrics, self._combined_quality_spec)

                _, changed = self.append_record(
                    cfg, pipeline_config, quality_metrics, execution_dag,
                    full=self._metrics_are_full(quality_metrics),
                    source={"kind": "quality_sweep"},
                )
                if changed:
                    results.append(self._evaluated[h])

                logger.info(
                    f"QualityStore: evaluated {h} quality={combined:.4f}"
                )
            except Exception as e:
                logger.warning(f"QualityStore: config {h} failed: {e}")
                self._save_failed(h, cfg, str(e))
                self._save_manifest()
        return results

    # ------------------------------------------------------------------
    # Load existing store
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, store_path: str) -> QualityStore:
        """Load an existing quality store from disk.

        This is a lightweight load that reads the manifest and quality
        results without re-initialising the GT evaluator.  The returned
        instance can be used for ``get_quality_results()`` but not for
        ``sweep()`` or ``evaluate_configs()`` (those need config + data).
        """
        store_path = os.path.abspath(store_path)
        manifest_path = os.path.join(store_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"No manifest.json found in {store_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)
        validate_quality_store_manifest(manifest, source=manifest_path)

        instance = object.__new__(cls)
        instance.store_path = store_path
        instance._evaluated = {}
        instance._combined_quality_spec = manifest.get("combined_quality")
        instance.dataset_identity = manifest.get("dataset", {})

        instance._manifest = manifest
        instance._load_evaluated_records(manifest)
        logger.info(
            f"QualityStore loaded: {len(instance._evaluated)} configs from {store_path}"
        )
        return instance

    @classmethod
    def load_full(cls, store_path: str) -> QualityStore:
        """Load an existing store and initialise its GT evaluator.

        Hardware replay Phase 2a only needs the lightweight :meth:`load`, but
        Phase 2b refinement must evaluate quality for newly suggested configs.
        Stores produced by :meth:`from_project` persist the source config but
        keep large datasets content-addressed by fingerprint; the config's
        authoritative corpus and QA paths must therefore remain reachable.
        """
        store_path = os.path.abspath(store_path)
        manifest_path = os.path.join(store_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"No manifest.json found in {store_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        config_path = manifest.get("config_path") or os.path.join(
            store_path, "rag_stack_config.yaml",
        )
        if not os.path.isabs(config_path):
            config_path = os.path.join(store_path, config_path)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Quality store cannot run refinement; config not found: {config_path}"
            )

        instance = cls(config_path=config_path, store_path=store_path)
        instance._manifest = manifest
        instance._combined_quality_spec = manifest.get(
            "combined_quality", instance._combined_quality_spec,
        )
        instance.dataset_identity = manifest.get(
            "dataset", getattr(instance, "dataset_identity", {}),
        )
        instance._load_evaluated_records(manifest)
        return instance

    def _load_evaluated_records(self, manifest: dict) -> None:
        """Load persisted configs + quality records into ``self._evaluated``."""
        validate_quality_store_manifest(
            manifest, source=os.path.join(self.store_path, "manifest.json"),
        )
        requires_trace = True
        for h in manifest.get("config_hashes", []):
            quality_path = os.path.join(self.store_path, "quality", f"{h}.json")
            config_path = os.path.join(self.store_path, "configs", f"{h}.json")
            pipeline_path = os.path.join(
                self.store_path, "pipeline_configs", f"{h}.json",
            )
            trace_path = os.path.join(self.store_path, "traces", f"{h}.json")
            record_path = os.path.join(self.store_path, "records", f"{h}.json")
            required = [quality_path, config_path, pipeline_path]
            if requires_trace:
                required.extend([trace_path, record_path])
            missing = [path for path in required if not os.path.isfile(path)]
            if missing:
                if requires_trace:
                    raise ValueError(
                        f"Quality store record {h} is incomplete; missing {missing}"
                    )
                continue
            with open(quality_path) as f:
                quality_metrics = json.load(f)
            with open(config_path) as f:
                config = json.load(f)
            record_meta: dict = {}
            if os.path.isfile(record_path):
                with open(record_path) as f:
                    record_meta = json.load(f) or {}
            canonical_config = _algo_config(config)
            if requires_trace and config != canonical_config:
                raise ValueError(
                    f"Quality store record {h} has a non-canonical algo config"
                )
            if requires_trace and _algo_record_key(canonical_config) != h:
                raise ValueError(
                    f"Quality store record {h} has a non-canonical algo config key"
                )
            if requires_trace:
                validate_quality_store_record_marker(
                    record_meta,
                    source=f"Quality store record {h}",
                )
                if record_meta.get("algo_hash") != h:
                    raise ValueError(
                        f"Quality store record {h} has mismatched commit marker key"
                    )
                artifact_paths = {
                    "config": config_path,
                    "quality": quality_path,
                    "trace": trace_path,
                    "pipeline_config": pipeline_path,
                }
                checksums = record_meta.get("artifact_sha256") or {}
                if set(checksums) != set(artifact_paths):
                    raise ValueError(
                        f"Quality store record {h} has no complete artifact checksums"
                    )
                for name, path in artifact_paths.items():
                    actual = _sha256_file(path)
                    if checksums.get(name) != actual:
                        raise ValueError(
                            f"Quality store record {h} {name} checksum mismatch"
                        )
                with open(trace_path) as f:
                    _validate_trace(
                        json.load(f), source=trace_path,
                        expected_queries=(manifest.get("dataset") or {}).get(
                            "qa_rows"
                        ),
                    )
            self._evaluated[h] = {
                "config_hash": h,
                "config": config,
                "quality_metrics": quality_metrics,
                "quality_combined": aggregate_quality_metrics(
                    quality_metrics,
                    (record_meta.get("quality_spec") or {}).get(
                        "combined_quality"
                    ) or self._combined_quality_spec,
                ),
                "full": bool(record_meta.get("full", True)),
                "execution_dag_path": trace_path if os.path.isfile(trace_path) else None,
                "record": record_meta,
            }

    # ------------------------------------------------------------------
    # Synthesize from a project's prior evaluations
    # ------------------------------------------------------------------

    @classmethod
    def from_project(cls, project_dir: str, store_path: str, merge: bool = False) -> str:
        """Import a project while serializing all store mutations."""
        store_path = os.path.abspath(store_path)
        with _store_lock(store_path):
            # A project directory is never a cache destination. Reject before
            # importing records or copying provenance over its frozen config.
            from rag_stack.project_config import CONFIG_FREEZE_FILENAME, ProjectConfigError
            if (
                os.path.exists(os.path.join(store_path, "summary.json"))
                or os.path.exists(os.path.join(store_path, CONFIG_FREEZE_FILENAME))
                or (
                    os.path.exists(os.path.join(store_path, "rag_stack_config.yaml"))
                    and not os.path.exists(os.path.join(store_path, "manifest.json"))
                )
            ):
                raise ProjectConfigError(
                    "QualityStore destination is a project directory; use a separate store path."
                )
            return cls._from_project_locked(project_dir, store_path, merge=merge)

    @classmethod
    def _from_project_locked(
        cls, project_dir: str, store_path: str, merge: bool = False,
    ) -> str:
        """Create or append an algo-keyed store from reusable quality records.

        Only records carrying a valid frozen ``execution_dag.json`` are
        eligible. Successful GT/GT_FP rows are normal inputs; historical
        source-CM OOM rows are also eligible when their quality run completed
        and left every configured metric plus the trace. ``merge=True`` is an
        append/update operation guarded by the exact authoritative corpus + QA
        fingerprints. Metrics and trace are always replaced together; a
        complete metric set outranks a partial one, otherwise the newest source
        record wins.
        """
        project_dir = os.path.abspath(project_dir)
        store_path = os.path.abspath(store_path)
        evals_dir = os.path.join(project_dir, "evaluations")
        csv_path = os.path.join(project_dir, "all_evaluations.csv")
        project_cfg_path = os.path.join(project_dir, "rag_stack_config.yaml")
        if not os.path.isdir(evals_dir):
            raise FileNotFoundError(f"No evaluations/ directory in {project_dir}")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"No all_evaluations.csv in {project_dir}")
        if not os.path.isfile(project_cfg_path):
            raise FileNotFoundError(f"No rag_stack_config.yaml in {project_dir}")

        import csv as _csv
        import yaml

        with open(project_cfg_path) as f:
            proj_cfg = yaml.safe_load(f) or {}
        dataset_cfg = proj_cfg.get("dataset") or {}
        source_qa = dataset_cfg.get("qa")
        source_corpus = dataset_cfg.get("corpus")
        # Prefer authoritative YAML inputs.  Fall back to the persisted raw
        # source copies only when the original path is no longer reachable.
        if not source_qa or not os.path.isfile(os.path.expanduser(str(source_qa))):
            source_qa = os.path.join(project_dir, "data", "qa.parquet")
        if not source_corpus or not os.path.isfile(os.path.expanduser(str(source_corpus))):
            raw_copy = os.path.join(project_dir, "data", "raw_corpus.parquet")
            source_corpus = raw_copy if os.path.isfile(raw_copy) else os.path.join(
                project_dir, "data", "original_corpus.parquet",
            )
        project_dataset = _dataset_identity(
            proj_cfg, qa_data_path=source_qa, corpus_data_path=source_corpus,
        )
        if not (
            (project_dataset.get("qa_fingerprint") or {}).get("sha256")
            and (project_dataset.get("corpus_fingerprint") or {}).get("sha256")
        ):
            raise ValueError(
                "Cannot create a reusable QualityStore without readable corpus "
                f"and QA inputs: {project_dataset}"
            )

        manifest_path = os.path.join(store_path, "manifest.json")
        existing_manifest: dict = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                existing_manifest = json.load(f) or {}
            if not merge:
                raise FileExistsError(
                    f"Quality store already exists: {store_path}; use merge=True"
                )
            validate_quality_store_manifest(existing_manifest, source=manifest_path)
            if not _dataset_matches(
                existing_manifest.get("dataset") or {}, project_dataset,
            ):
                raise ValueError(
                    "Quality store corpus/QA do not match project corpus/QA: "
                    f"store={existing_manifest.get('dataset')}, "
                    f"project={project_dataset}"
                )
        elif merge:
            raise FileNotFoundError(
                f"Cannot append to {store_path}: manifest.json is missing"
            )
        elif os.path.isdir(store_path) and any(
            name != ".quality_store.lock" for name in os.listdir(store_path)
        ):
            raise FileExistsError(
                f"Refusing to create a store in non-empty directory {store_path}"
            )

        with open(csv_path, newline="") as f:
            rows = list(_csv.DictReader(f))
        ssops = RagSearchSpace.from_yaml(proj_cfg)
        base_decoded = ssops.build_base_decoded_config(proj_cfg)
        skip_prefixes = (
            "performance", "quality", "eval_type", "eval_error", "eval_id",
        )
        decoded_by_id: Dict[int, tuple[dict, str]] = {}
        source_row_by_id: Dict[int, dict] = {}
        for row in rows:
            try:
                eid = int(row.get("eval_id", ""))
            except (TypeError, ValueError):
                continue
            if not _quality_import_source_candidate(row):
                continue
            flat: dict = {}
            for column, value in row.items():
                if not value or any(
                    column == prefix or column.startswith(prefix + "__")
                    for prefix in skip_prefixes
                ):
                    continue
                flat[column] = _parse_csv_cell(value)
            decoded = ssops.decode_to_nested(flat)
            enriched = ssops.enrich_configs([decoded], base_decoded)[0]
            algo = _algo_config(enriched)
            decoded_by_id[eid] = (algo, _algo_record_key(algo))
            source_row_by_id[eid] = {
                "eval_type": str(row.get("eval_type") or ""),
                "eval_error": str(row.get("eval_error") or ""),
            }

        for dirname in ("configs", "quality", "traces", "pipeline_configs", "records"):
            os.makedirs(os.path.join(store_path, dirname), exist_ok=True)

        # A record metadata file is the multi-file commit marker. Existing
        # stores are loaded strictly so a prior partial write cannot be used.
        evaluated: Dict[str, dict] = {}
        if merge:
            loaded = cls.load(store_path)
            evaluated = dict(loaded._evaluated)

        configured_metrics = _configured_metric_names(proj_cfg)
        imported_hashes: list[str] = []
        replaced_hashes: list[str] = []
        skipped_trace_records: list[dict] = []
        duplicate_hashes: list[str] = []
        n_quality_candidates = 0
        for dirname in sorted(os.listdir(evals_dir)):
            if not dirname.startswith("eval_"):
                continue
            eval_path = os.path.join(evals_dir, dirname)
            try:
                eid = int(dirname.removeprefix("eval_"))
            except ValueError:
                continue
            mapped = decoded_by_id.get(eid)
            if mapped is None:
                continue
            qpath = os.path.join(eval_path, "quality_metrics.json")
            ppath = os.path.join(eval_path, "pipeline_config.json")
            tpath = os.path.join(eval_path, "execution_dag.json")
            if not os.path.isfile(qpath) or not os.path.isfile(ppath):
                continue
            n_quality_candidates += 1
            if not os.path.isfile(tpath):
                skipped_trace_records.append({"eval_id": eid, "reason": "missing_trace"})
                continue
            try:
                with open(tpath) as f:
                    execution_dag = _validate_trace(
                        json.load(f), source=tpath,
                        expected_queries=project_dataset.get("qa_rows"),
                    )
            except Exception as exc:
                skipped_trace_records.append({"eval_id": eid, "reason": str(exc)})
                continue
            with open(qpath) as f:
                quality_metrics, embedded_trace = _split_quality_payload(json.load(f))
            if embedded_trace is not None:
                embedded_trace = _validate_trace(
                    embedded_trace,
                    source=qpath,
                    expected_queries=project_dataset.get("qa_rows"),
                )
                if embedded_trace != execution_dag:
                    skipped_trace_records.append({
                        "eval_id": eid,
                        "reason": "quality_metrics and execution_dag carry different traces",
                    })
                    continue
            if not quality_metrics:
                skipped_trace_records.append({"eval_id": eid, "reason": "empty_quality"})
                continue
            source_row = source_row_by_id[eid]
            if (
                source_row["eval_type"] == EvalType.OOM.value
                and not _has_complete_numeric_quality(
                    quality_metrics, configured_metrics,
                )
            ):
                skipped_trace_records.append({
                    "eval_id": eid,
                    "reason": "source_cm_oom_incomplete_quality",
                })
                continue
            with open(ppath) as f:
                pipeline_config = json.load(f)

            algo, record_hash = mapped
            marker_path = os.path.join(eval_path, "quality_memo.json")
            # Old marker hashes were based on sparse optimizer configs and are
            # intentionally not the current key. Keep them only as provenance.
            legacy_memo_hash = None
            if os.path.isfile(marker_path):
                try:
                    with open(marker_path) as f:
                        legacy_memo_hash = (json.load(f) or {}).get("algo_hash")
                except Exception:
                    pass
            source_mtime_ns = max(
                os.stat(path).st_mtime_ns for path in (qpath, ppath, tpath)
            )
            full = all(
                quality_metrics.get(name) is not None
                for name in configured_metrics
            )
            record_meta = {
                "algo_hash": record_hash,
                "full": full,
                "metric_names": sorted(quality_metrics),
                "quality_spec": _quality_spec(proj_cfg),
                "source": {
                    "project_dir": project_dir,
                    "eval_id": eid,
                    "eval_type": source_row["eval_type"],
                    "eval_error": source_row["eval_error"],
                },
                "source_eval_type": source_row["eval_type"],
                "source_eval_error": source_row["eval_error"],
                "source_mtime_ns": source_mtime_ns,
                "legacy_sparse_memo_hash": legacy_memo_hash,
                "updated_at": datetime.datetime.now().isoformat(),
            }

            previous = evaluated.get(record_hash)
            if previous is not None:
                if previous.get("config") != algo:
                    raise RuntimeError(
                        f"QualityStore hash collision for {record_hash}"
                    )
                previous_meta = previous.get("record") or {}
                if not _quality_specs_match(
                    previous_meta.get("quality_spec"),
                    record_meta.get("quality_spec"),
                ):
                    raise ValueError(
                        "Cannot replace the same algorithm config with scores "
                        "from different quality/judge semantics; use a separate "
                        f"QualityStore (algo_hash={record_hash})"
                    )
                if _record_priority(record_meta) <= _record_priority(previous_meta):
                    duplicate_hashes.append(record_hash)
                    continue
                replaced_hashes.append(record_hash)

            record_meta = _persist_record_files(
                store_path,
                record_hash,
                config=algo,
                quality_metrics=quality_metrics,
                execution_dag=execution_dag,
                pipeline_config=pipeline_config,
                record_meta=record_meta,
            )
            evaluated[record_hash] = {
                "config_hash": record_hash,
                "config": algo,
                "quality_metrics": quality_metrics,
                "quality_combined": aggregate_quality_metrics(
                    quality_metrics,
                    existing_manifest.get("combined_quality")
                    if merge else (
                        (proj_cfg.get("eval_backend_setting") or {}).get(
                            "combined_quality"
                        )
                    ),
                ),
                "full": full,
                "execution_dag_path": os.path.join(
                    store_path, "traces", f"{record_hash}.json",
                ),
                "record": record_meta,
            }
            imported_hashes.append(record_hash)

        if not evaluated:
            raise RuntimeError(
                f"No valid GT evaluations found in {project_dir}; "
                f"quality_candidates={n_quality_candidates}, "
                f"trace_rejections={len(skipped_trace_records)}"
            )

        # Keep the first source config for provenance/refinement. Do not copy
        # project data into this global cache: large corpora can be multi-GB,
        # and normal cache hits/hardware replay need only the content identity.
        # Refinement uses the current Controller's DatasetManager.
        store_cfg_path = os.path.join(store_path, "rag_stack_config.yaml")
        if not merge:
            shutil.copy2(project_cfg_path, store_cfg_path)

        source_project_dirs = list(existing_manifest.get("source_project_dirs") or [])
        legacy_source = existing_manifest.get("source_project_dir")
        if legacy_source and legacy_source not in source_project_dirs:
            source_project_dirs.append(legacy_source)
        if project_dir not in source_project_dirs:
            source_project_dirs.append(project_dir)
        combined_quality_spec = (
            existing_manifest.get("combined_quality")
            if merge else (
                (proj_cfg.get("eval_backend_setting") or {}).get("combined_quality")
            )
        )
        manifest = {
            "kind": QUALITY_STORE_KIND,
            "key_kind": QUALITY_STORE_KEY_KIND,
            "data_storage": "content_addressed_reference",
            "status": "completed",
            "pid": None,
            "created_at": existing_manifest.get(
                "created_at", datetime.datetime.now().isoformat(),
            ),
            "updated_at": datetime.datetime.now().isoformat(),
            "strategy": "from_project_merge" if merge else "from_project",
            "n_grid": existing_manifest.get("n_grid"),
            "n_configs_requested": existing_manifest.get("n_configs_requested", 0),
            "n_configs": len(evaluated),
            "token_stats": existing_manifest.get("token_stats") or proj_cfg.get(
                "_token_stats", {}
            ),
            "dataset": existing_manifest.get("dataset") if merge else project_dataset,
            "combined_quality": combined_quality_spec,
            "config_path": "rag_stack_config.yaml",
            "config_hashes": sorted(evaluated),
            "failed_hashes": list(existing_manifest.get("failed_hashes") or []),
            "source_project_dir": source_project_dirs[0],
            "source_project_dirs": source_project_dirs,
            "last_imported_project_dir": project_dir,
            "last_imported_hashes": sorted(set(imported_hashes)),
            "last_replaced_hashes": sorted(set(replaced_hashes)),
            "last_duplicate_hashes": sorted(set(duplicate_hashes)),
            "last_trace_rejections": skipped_trace_records,
        }
        _atomic_json_dump(manifest_path, manifest)
        # Read back through the strict loader before reporting success. This
        # verifies every commit marker, checksum, key and trace cardinality.
        cls.load(store_path)
        logger.info(
            "QualityStore import %s: added/updated=%d replaced=%d duplicates=%d "
            "trace_rejections=%d total=%d -> %s",
            project_dir, len(set(imported_hashes)), len(set(replaced_hashes)),
            len(duplicate_hashes), len(skipped_trace_records), len(evaluated),
            store_path,
        )
        return store_path

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_quality_results(self) -> list[dict]:
        """Return all evaluated quality results.

        Each entry: ``{config_hash, config, quality_metrics, quality_combined}``.
        """
        return list(self._evaluated.values())

    def get_record(self, cfg_hash: str) -> dict | None:
        """Return one indexed record without loading its trace payload."""
        return self._evaluated.get(cfg_hash)

    def get_token_stats(self) -> dict:
        """Return stored token stats (from manifest if loaded, else live)."""
        if hasattr(self, "_manifest"):
            return self._manifest.get("token_stats", {})
        return self.token_stats.to_dict()

    def get_pipeline_config(self, cfg_hash: str) -> dict:
        """Load a stored resolved pipeline config by hash."""
        path = os.path.join(
            self.store_path, "pipeline_configs", f"{cfg_hash}.json",
        )
        with open(path) as f:
            return json.load(f)

    def get_execution_dag(self, cfg_hash: str) -> dict | None:
        """Load and validate a record's frozen quality trace."""
        path = os.path.join(self.store_path, "traces", f"{cfg_hash}.json")
        if not os.path.isfile(path):
            if self.requires_quality_trace:
                raise FileNotFoundError(
                    f"Quality store record {cfg_hash} is missing its trace"
                )
            return None
        with open(path) as f:
            trace = json.load(f)
        expected = (
            (getattr(self, "_manifest", {}) or {}).get("dataset") or {}
        ).get("qa_rows")
        return _validate_trace(
            trace, source=path, expected_queries=expected,
        )

    def _metrics_are_full(self, quality_metrics: dict) -> bool:
        configured = [
            metric.get("metric_name") or metric.get("metric")
            for metric in (self.config.get("eval_backend_setting") or {}).get(
                "metrics", []
            )
            if isinstance(metric, dict)
            and (metric.get("metric_name") or metric.get("metric"))
        ]
        return all(quality_metrics.get(name) is not None for name in configured)

    # ------------------------------------------------------------------
    # Internal: config enumeration
    # ------------------------------------------------------------------

    def _enumerate_configs(self, n_grid: int) -> list[dict]:
        """Enumerate all configs via cartesian product.

        - ChoiceParameter → all values
        - RangeParameter  → np.linspace(lo, hi, K) where K is per-param
          ``n_grid`` (from YAML) or the global ``n_grid`` fallback.
          INT ranges use np.unique(np.round(...)).
        """
        params = self.search_space.parameters
        if not params:
            return [{}]

        # Build per-parameter n_grid lookup from original YAML config
        param_n_grid = self._extract_param_n_grid()

        param_names = list(params.keys())
        param_values: list[list] = []

        for name in param_names:
            p = params[name]
            if isinstance(p, ChoiceParameter):
                param_values.append(list(p.values))
            elif isinstance(p, RangeParameter):
                k = param_n_grid.get(name, n_grid)
                grid = np.linspace(p.lower, p.upper, k)
                if p.parameter_type == ParameterType.INT:
                    grid = np.unique(np.round(grid)).astype(int)
                param_values.append(grid.tolist())
            else:
                param_values.append([None])

        configs: list[dict] = []
        seen: set = set()
        for combo in itertools.product(*param_values):
            raw = dict(zip(param_names, combo))
            active = self._ssops.active_params(raw)
            cfg = self._ssops.decode_to_nested(
                {name: value for name, value in raw.items() if name in active}
            )
            cfg = self._canonical_algo_config(cfg)
            sig = config_signature(cfg)
            if sig not in seen:
                seen.add(sig)
                configs.append(cfg)

        return configs

    def _sobol_configs(self, n_configs: int) -> list[dict]:
        """Generate configs via Sobol quasi-random sequence.

        Draws ``n_configs`` points in [0,1]^D using Sobol, snaps discrete
        (ChoiceParameter) dims, applies hierarchical dependency masking,
        decodes to config dicts, and deduplicates.

        Typically produces slightly fewer than ``n_configs`` unique configs
        due to snapping and dedup.
        """
        from botorch.utils.sampling import draw_sobol_samples

        dim = len(self.search_space.parameters)
        if dim == 0:
            return [{}]

        # Oversample to compensate for dedup loss after snapping
        n_raw = min(n_configs * 3, max(n_configs + 100, n_configs * 2))
        bounds = torch.zeros(2, dim)
        bounds[1] = 1.0

        raw = draw_sobol_samples(bounds=bounds, n=n_raw, q=1).squeeze(1)
        raw = self._ssops.snap_to_grid(raw)

        # Decode and dedup
        configs: list[dict] = []
        seen: set = set()
        for row_idx in range(raw.shape[0]):
            cfg = self._canonical_algo_config(
                self._ssops.decode_to_config(raw[row_idx])
            )
            sig = config_signature(cfg)
            if sig in seen:
                continue
            seen.add(sig)
            configs.append(cfg)
            if len(configs) >= n_configs:
                break

        return configs

    def _extract_param_n_grid(self) -> dict[str, int]:
        """Extract per-parameter n_grid from original YAML config.

        Scans vectordb, node_lines, and the top-level ``corpus`` block for
        ``{range: [...], n_grid: K}`` defs. Returns ``{param_name: K}`` for
        params that have explicit n_grid (others fall back to the global
        n_grid passed to ``QualityStore.sweep``).
        """
        result: dict[str, int] = {}

        # Vectordb params (algo_search_space.vectordb — the input-schema home).
        from rag_stack.search_space import algo_vectordb_blocks

        for vdb in algo_vectordb_blocks(self.config):
            vdb_name = vdb.get("name", "")
            for k, v in vdb.items():
                if isinstance(v, dict) and "range" in v and "n_grid" in v:
                    param_name = f"vectordb{SEP}{vdb_name}{SEP}{k}"
                    result[param_name] = int(v["n_grid"])

        # Per-method-block node_lines params. Naming matches what
        # ``_extract_pipeline_mode_params`` emits: pipeline__<method>__<rest>.
        from rag_stack.search_space.rag_search_space import pipeline_method_blocks
        for block in pipeline_method_blocks(self.config):
            if not isinstance(block, dict):
                continue
            method = block.get("mode")
            if not isinstance(method, str):
                continue
            method_prefix = f"pipeline{SEP}{method}{SEP}"
            for node_line in block.get("node_lines") or []:
                for node in node_line.get("nodes", []):
                    stage = node["stage"]
                    for k, v in node.items():
                        if k in ("stage", "strategy", "modules", "optional"):
                            continue
                        if isinstance(v, dict) and "range" in v and "n_grid" in v:
                            result[f"{method_prefix}{stage}{SEP}{k}"] = int(v["n_grid"])
                    for mod in node.get("modules", []):
                        mod_type = mod.get("component", "")
                        for k, v in mod.items():
                            if k in ("component", "generator_backend"):
                                continue
                            if isinstance(v, dict) and "range" in v and "n_grid" in v:
                                result[f"{method_prefix}{stage}{SEP}{mod_type}{SEP}{k}"] = int(v["n_grid"])

        # Corpus / chunker params (mirrors search_space._extract_corpus_params).
        # Naming: corpus__chunker__<param>, matching what
        # RagSearchSpace.build_ax_searchspace emits.
        corpus = (self.config.get("algo_search_space") or {}).get("corpus") or {}
        if isinstance(corpus, dict):
            chunker = corpus.get("chunker")
            if isinstance(chunker, dict):
                for k, v in chunker.items():
                    if isinstance(v, dict) and "range" in v and "n_grid" in v:
                        result[f"corpus{SEP}chunker{SEP}{k}"] = int(v["n_grid"])

        return result

    # ------------------------------------------------------------------
    # Internal: config resolution (mirrors Controller._resolve_config)
    # ------------------------------------------------------------------

    def _resolve_config(self, decoded_config: dict) -> dict:
        """Apply decoded optimizer params to the base YAML config.

        Delegates to :class:`RagSearchSpace` for the pipeline / vectordb
        resolution so QualityStore and Controller share one resolver
        implementation. This avoids duplicating the list-of-method-blocks
        walking logic.
        """
        from rag_stack.search_space import RagSearchSpace

        config = deepcopy(self.config)
        config.pop("optimizer", None)

        # Corpus chunker resolution (delegated to the shared DatasetManager so
        # QualityStore and Controller use one implementation).
        chunker_runtime = self._data.resolve_corpus_chunker(decoded_config)
        if chunker_runtime:
            config["corpus_runtime"] = {"chunker": chunker_runtime}
        config.pop("corpus", None)

        # Pipeline tree resolution (mode + per-method node_lines).
        ssops = RagSearchSpace.from_yaml(self.config)
        pipeline_runtime, node_lines = ssops.resolve_pipeline_block(
            self.config, decoded_config,
        )
        config["pipeline_runtime"] = pipeline_runtime
        config["node_lines"] = node_lines
        config.pop("pipeline", None)

        # Vectordb overrides.
        ssops.resolve_vectordbs(config, decoded_config)

        # Store records always contain a canonical per-query trace for later
        # hardware replay, even when the consuming project selects static CM.
        # Request the same dynamic result contract from every backend.
        config.setdefault("_runtime", {})["rag_ir_mode"] = "dynamic"
        return config

    @staticmethod
    def _scalarify_value(v):
        if isinstance(v, list):
            if len(v) == 1:
                return v[0]
            raise ValueError(f"Unresolved sweep parameter: {v}")
        if isinstance(v, dict) and ("range" in v or "ordered" in v):
            raise ValueError(f"Unresolved range/ordered parameter: {v}")
        return v

    @staticmethod
    def _is_sweepable(v) -> bool:
        """Mirror of Controller._is_sweepable for the corpus-runtime branch."""
        if isinstance(v, list) and len(v) > 1:
            return True
        if isinstance(v, dict) and ("range" in v or "ordered" in v):
            return True
        return False

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _evaluate_and_persist(
        self,
        configs: list[dict],
        n_grid: int = 5,
        n_configs_requested: int = 0,
        strategy: str = "exhaustive",
    ) -> dict:
        """Evaluate configs and persist to store."""
        for dirname in ("configs", "quality", "traces", "pipeline_configs", "records"):
            os.makedirs(os.path.join(self.store_path, dirname), exist_ok=True)

        failed_hashes: list[str] = []
        quality_summary: dict[str, float] = {}

        for i, raw_cfg in enumerate(configs):
            cfg = self._canonical_algo_config(raw_cfg)
            h = _algo_record_key(cfg)

            # Skip already evaluated
            if h in self._evaluated:
                quality_summary[h] = self._evaluated[h]["quality_combined"]
                logger.info(f"  [{i+1}/{len(configs)}] {h} already evaluated, skipping")
                continue

            try:
                pipeline_config = self._resolve_config(cfg)
                run_dir = os.path.join(self.store_path, "runs", h)
                t0 = time.monotonic()
                quality_payload = self.rag_quality_evaluator.evaluate(
                    pipeline_config, run_dir=run_dir,
                )
                t1 = time.monotonic()
                quality_payload = normalize_backend_result(
                    quality_payload,
                    backend=(self.config.get("global") or {}).get("eval_backend", "static_gt"),
                    pipeline_config=pipeline_config,
                    question_ids=(self.qa_data["qid"].tolist()
                                  if getattr(self, "qa_data", None) is not None else None),
                )
                quality_metrics, execution_dag = _split_quality_payload(
                    quality_payload,
                )
                execution_dag = _validate_trace(
                    execution_dag,
                    source=f"QualityStore sweep {h}",
                    expected_queries=len(self.qa_data),
                )
                combined = aggregate_quality_metrics(quality_metrics, self._combined_quality_spec)
                self.append_record(
                    cfg,
                    pipeline_config,
                    quality_metrics,
                    execution_dag,
                    full=self._metrics_are_full(quality_metrics),
                    source={"kind": "quality_sweep"},
                )
                quality_summary[h] = combined
                self._save_manifest(n_grid=n_grid, n_configs_requested=n_configs_requested, strategy=strategy)

                logger.info(
                    f"  [{i+1}/{len(configs)}] {h} quality={combined:.4f} "
                    f"time={t1 - t0:.1f}s"
                )
            except Exception as e:
                logger.warning(
                    f"  [{i+1}/{len(configs)}] {h} FAILED: {e}"
                )
                failed_hashes.append(h)
                self._save_failed(h, cfg, str(e))
                self._save_manifest(n_grid=n_grid, n_configs_requested=n_configs_requested, strategy=strategy)

        self._save_manifest(n_grid=n_grid, n_configs_requested=n_configs_requested, strategy=strategy)

        n_ok = len(self._evaluated)
        logger.info(
            f"QualityStore sweep complete: {n_ok} evaluated, "
            f"{len(failed_hashes)} failed"
        )
        return {
            "n_configs": n_ok,
            "n_failed": len(failed_hashes),
            "quality_store_path": self.store_path,
            "quality_summary": quality_summary,
        }

    def append_record(
        self,
        decoded_config: dict,
        pipeline_config: dict,
        quality_metrics: dict,
        execution_dag: dict,
        *,
        full: bool | None = None,
        source: dict | None = None,
        quality_spec: dict | None = None,
    ) -> tuple[str, bool]:
        """Append one record under the store-wide writer lock."""
        with _store_lock(self.store_path):
            return self._append_record_locked(
                decoded_config,
                pipeline_config,
                quality_metrics,
                execution_dag,
                full=full,
                source=source,
                quality_spec=quality_spec,
            )

    def _append_record_locked(
        self,
        decoded_config: dict,
        pipeline_config: dict,
        quality_metrics: dict,
        execution_dag: dict,
        *,
        full: bool | None = None,
        source: dict | None = None,
        quality_spec: dict | None = None,
    ) -> tuple[str, bool]:
        """Atomically append/refresh one whole quality+trace cache record.

        Returns ``(algo_hash, changed)``.  A cache record is never assembled
        from metrics and a trace belonging to different evaluations.
        """
        if hasattr(self, "_ssops") and hasattr(self, "_base_decoded_config"):
            cfg = self._canonical_algo_config(decoded_config)
        else:
            cfg = _algo_config(decoded_config)
        h = _algo_record_key(cfg)
        manifest_path = os.path.join(self.store_path, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                disk_manifest = json.load(f) or {}
            known_manifest = getattr(self, "_manifest", {}) or {}
            if (
                disk_manifest.get("updated_at") != known_manifest.get("updated_at")
                or disk_manifest.get("config_hashes")
                != known_manifest.get("config_hashes")
            ):
                # Another process appended/replaced a record since this
                # instance was loaded. The lock makes this one refresh enough.
                current = QualityStore.load(self.store_path)
                self._evaluated = dict(current._evaluated)
                self._manifest = dict(current._manifest)
                self.dataset_identity = current.dataset_identity
        metrics, embedded_trace = _split_quality_payload(quality_metrics)
        if embedded_trace is not None:
            if execution_dag is not None and embedded_trace != execution_dag:
                raise ValueError(f"QualityStore record {h} carries two different traces")
            execution_dag = embedded_trace
        expected = (
            (getattr(self, "dataset_identity", {}) or {}).get("qa_rows")
            or (((getattr(self, "_manifest", {}) or {}).get("dataset") or {}).get("qa_rows"))
        )
        execution_dag = _validate_trace(
            execution_dag,
            source=f"QualityStore append {h}",
            expected_queries=expected,
        )
        if full is None:
            full = self._metrics_are_full(metrics) if hasattr(self, "config") else True
        now_ns = time.time_ns()
        record_meta = {
            "algo_hash": h,
            "full": bool(full),
            "metric_names": sorted(metrics),
            "quality_spec": _canonical_quality_spec(
                quality_spec
                if quality_spec is not None
                else _quality_spec(getattr(self, "config", None))
            ),
            "source": _strip_secrets(source or {}),
            "source_mtime_ns": int((source or {}).get("source_mtime_ns") or now_ns),
            "updated_at": datetime.datetime.now().isoformat(),
        }
        previous = self._evaluated.get(h)
        if previous is not None:
            if previous.get("config") != cfg:
                raise RuntimeError(f"QualityStore hash collision for {h}")
            previous_spec = (previous.get("record") or {}).get("quality_spec")
            if not _quality_specs_match(
                previous_spec, record_meta.get("quality_spec"),
            ):
                raise ValueError(
                    "Cannot replace the same algorithm config with scores from "
                    "different quality/judge semantics; use a separate "
                    f"QualityStore (algo_hash={h})"
                )
            if _record_priority(record_meta) <= _record_priority(previous.get("record")):
                return h, False

        committed_meta = self._save_single(
            h, cfg, pipeline_config, metrics, execution_dag,
            record_meta=record_meta,
        )
        combined_spec = getattr(self, "_combined_quality_spec", None)
        if combined_spec is None:
            combined_spec = (getattr(self, "_manifest", {}) or {}).get(
                "combined_quality"
            )
        combined_spec = (
            (committed_meta.get("quality_spec") or {}).get("combined_quality")
            or combined_spec
        )
        self._evaluated[h] = {
            "config_hash": h,
            "config": cfg,
            "quality_metrics": metrics,
            "quality_combined": aggregate_quality_metrics(metrics, combined_spec),
            "full": bool(full),
            "execution_dag_path": os.path.join(
                self.store_path, "traces", f"{h}.json",
            ),
            "record": committed_meta,
        }
        self._save_manifest()
        return h, True

    def _save_single(
        self,
        h: str,
        cfg: dict,
        pipeline_config: dict,
        quality_metrics: dict,
        execution_dag: dict,
        *,
        record_meta: dict,
    ):
        """Persist one record, writing its commit marker last."""
        return _persist_record_files(
            self.store_path,
            h,
            config=cfg,
            quality_metrics=quality_metrics,
            execution_dag=execution_dag,
            pipeline_config=pipeline_config,
            record_meta=record_meta,
        )

    def _save_failed(self, h: str, cfg: dict, reason: str):
        """Record a failed evaluation."""
        failed_dir = os.path.join(self.store_path, "failed")
        os.makedirs(failed_dir, exist_ok=True)
        _atomic_json_dump(
            os.path.join(failed_dir, f"{h}.json"),
            {"config": cfg, "reason": reason},
        )

    def update_status(
        self,
        status: str,
        *,
        pid: int | None = None,
        strategy: str | None = None,
        n_grid: int | None = None,
        n_configs_requested: int = 0,
    ) -> None:
        """Atomically update operational metadata without changing records."""
        with _store_lock(self.store_path):
            self._save_manifest(
                n_grid=n_grid,
                n_configs_requested=n_configs_requested,
                strategy=strategy,
            )
            path = os.path.join(self.store_path, "manifest.json")
            with open(path) as f:
                manifest = json.load(f)
            manifest["status"] = str(status)
            manifest["pid"] = pid
            manifest["updated_at"] = datetime.datetime.now().isoformat()
            _atomic_json_dump(path, manifest)
            self._manifest = manifest

    def _save_manifest(
        self,
        n_grid: int | None = None,
        n_configs_requested: int = 0,
        strategy: str | None = None,
    ):
        """Write or update manifest.json."""
        manifest_path = os.path.join(self.store_path, "manifest.json")

        # Preserve existing fields when updating
        existing: dict = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                existing = json.load(f)

        failed_dir = os.path.join(self.store_path, "failed")
        failed_hashes = []
        if os.path.isdir(failed_dir):
            failed_hashes = [
                f.replace(".json", "") for f in os.listdir(failed_dir)
                if f.endswith(".json")
            ]

        manifest = {
            **{
                key: value
                for key, value in existing.items()
                if key in MANIFEST_OPTIONAL_FIELDS
            },
            "kind": QUALITY_STORE_KIND,
            "key_kind": QUALITY_STORE_KEY_KIND,
            "status": existing.get("status", "running"),
            "pid": existing.get("pid"),
            "created_at": existing.get(
                "created_at", datetime.datetime.now().isoformat(),
            ),
            "updated_at": datetime.datetime.now().isoformat(),
            "strategy": strategy or existing.get("strategy", "unknown"),
            "n_grid": n_grid or existing.get("n_grid"),
            "n_configs_requested": n_configs_requested or existing.get("n_configs_requested"),
            "n_configs": len(self._evaluated),
            "token_stats": (
                self.token_stats.to_dict() if hasattr(self, "token_stats")
                else existing.get("token_stats", {})
            ),
            "dataset": getattr(self, "dataset_identity", None) or existing.get("dataset", {}),
            "combined_quality": getattr(self, "_combined_quality_spec", None) or existing.get("combined_quality"),
            "config_path": existing.get("config_path", "rag_stack_config.yaml"),
            "config_hashes": sorted(self._evaluated.keys()),
            "failed_hashes": failed_hashes,
        }
        _atomic_json_dump(manifest_path, manifest)
        self._manifest = manifest
