"""Executor-independent workflow structure paired with observed quality traces.

Nodes, producer edges, and execution order describe the pipeline. The legacy
stage summary, component flags, and corpus statistics remain available to CM
adapters. Constructing or validating this contract never requires a cost model;
each consumer validates its own execution or pricing capabilities separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

__all__ = [
    "CorpusStats",
    "WorkflowSchema",
    "corpus_stats_from_pipeline_config",
    "dataflow_type_of",
    "build_workflow_schema",
]


@dataclass
class CorpusStats:
    """Corpus-level statistics the schema carries INTO the cost model.

    Frozen-contract rule: the CM accepts exactly (workflow schema, quality
    trace) [+ configs in fixed mode]. Corpus statistics used to ride on
    ``pipeline_config["_token_stats"]`` and ``vectordb[i]["N"]`` as side
    channels the CM hard-required; they are now a first-class schema field.
    The field set mirrors ``TokenStats.to_dict()`` exactly (4 required base
    keys; the rest optional) plus the per-vectordb index sizes that the FAISS
    stage models price on.
    """

    # Required base keys — TokenStats.from_dict raises without these.
    avg_chunk_tokens: int
    prompt_template_tokens: int
    avg_output_tokens: int
    avg_query_tokens: int
    # Always serialized with a default by TokenStats.to_dict().
    bytes_per_token: float = 4.0
    # Optional refinements (present when the producer computed them).
    avg_llm_input_tokens: Optional[int] = None
    avg_query_expansion_input_tokens: Optional[int] = None
    avg_query_expansion_output_tokens: Optional[int] = None
    avg_passage_reranker_input_tokens: Optional[int] = None
    avg_compressor_elems_per_chunk: Optional[float] = None
    avg_compressor_elem_tokens: Optional[int] = None
    chunk_token_quantiles: Optional[list] = None
    # Per-vectordb corpus size: vectordb name -> N vectors. Replaces the
    # ``vectordb[i]["N"]`` side channel (whose absence silently priced FAISS
    # at 1,000,000 vectors).
    vectordb_index_sizes: Dict[str, int] = field(default_factory=dict)

    _REQUIRED_BASE_KEYS = (
        "avg_chunk_tokens",
        "prompt_template_tokens",
        "avg_output_tokens",
        "avg_query_tokens",
    )

    def token_stats_dict(self) -> dict:
        """The exact ``pipeline_config["_token_stats"]`` payload this carries.

        Byte-for-byte the ``TokenStats.to_dict()`` convention: base keys plus
        ``bytes_per_token`` always present, optional keys only when set — so
        materializing schema stats back into a config reproduces the original
        dict exactly (the bit-parity requirement of the contract freeze).
        """
        out = {
            "avg_chunk_tokens": self.avg_chunk_tokens,
            "prompt_template_tokens": self.prompt_template_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_query_tokens": self.avg_query_tokens,
            "bytes_per_token": self.bytes_per_token,
        }
        for key in (
            "avg_llm_input_tokens",
            "avg_query_expansion_input_tokens",
            "avg_query_expansion_output_tokens",
            "avg_passage_reranker_input_tokens",
            "avg_compressor_elems_per_chunk",
            "avg_compressor_elem_tokens",
            "chunk_token_quantiles",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusStats":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"CorpusStats has unsupported fields: {sorted(unknown)}"
            )
        missing = [k for k in cls._REQUIRED_BASE_KEYS if k not in d]
        if missing:
            raise ValueError(f"CorpusStats missing required keys: {missing}")
        return cls(**d)


def corpus_stats_from_pipeline_config(
    pipeline_config: dict,
) -> Optional[CorpusStats]:
    """The ONE sanctioned lift of corpus statistics into the schema.

    Reads the producer-stamped ``_token_stats`` dict plus every
    ``vectordb[i]["N"]``. Returns ``None`` when the config carries no
    ``_token_stats`` at all (schemas built from bare configs stay legal —
    the CM raises at materialization time if it actually needs the stats);
    raises when stats ARE present but corrupt (missing base keys / unknown
    keys), because a half-stamped config must fail loudly, not price wrong.
    """
    stats = pipeline_config.get("_token_stats")
    if stats is None:
        return None
    if not isinstance(stats, dict):
        raise TypeError("pipeline_config['_token_stats'] must be a dict")
    known = {f.name for f in fields(CorpusStats)} - {"vectordb_index_sizes"}
    unknown = set(stats) - known
    if unknown:
        raise ValueError(
            f"pipeline_config['_token_stats'] has unsupported keys: "
            f"{sorted(unknown)}"
        )
    missing = [
        k for k in CorpusStats._REQUIRED_BASE_KEYS if k not in stats
    ]
    if missing:
        raise ValueError(
            f"pipeline_config['_token_stats'] missing required keys: {missing}"
        )
    index_sizes: Dict[str, int] = {}
    for vdb in pipeline_config.get("vectordb") or []:
        if not isinstance(vdb, dict):
            continue
        n = vdb.get("N")
        if n is None:
            continue
        name = vdb.get("name") or vdb.get("vectordb")
        if not name:
            # Positional fallback keeps multi-vectordb configs unambiguous
            # even when entries carry no name field.
            name = f"vectordb_{(pipeline_config.get('vectordb') or []).index(vdb)}"
        index_sizes[str(name)] = int(n)
    return CorpusStats(
        **{k: stats[k] for k in stats},
        vectordb_index_sizes=index_sizes,
    )


@dataclass
class WorkflowSchema:
    """The structure rag_cm dispatches on (RAGO ``RAGSchema``-equivalent).

    Carries the ``dataflow_type`` (the dispatch key), which
    **PerformanceStages** are present (``stages`` holds their flat names), and
    the presence flags the latency walk needs. The per-question token DATA
    stays in the trace; the corpus-level DISTRIBUTIONS ride here
    (``corpus_stats``) so the CM needs no side channels. ``runtime_stats`` is
    the static (no-trace) path's aggregate payload — CM-init, quality-store
    refinement replays and ``rag_ir_mode=static`` price without a trace.
    """

    dataflow_type: str                       # "sequential" | "agentic"
    stages: List[str] = field(default_factory=list)   # PerformanceStage names
    has_query_expansion: bool = False
    has_reranker: bool = False
    has_encode: bool = False
    has_compressor: bool = False
    corpus_stats: Optional[CorpusStats] = None
    runtime_stats: Optional[dict] = None
    nodes: List[dict] = field(default_factory=list)
    execution_order: List[str] = field(default_factory=list)
    execution_policy: str = "sequential"

    def __post_init__(self) -> None:
        validate_workflow_nodes(self.nodes, self.execution_order, self.execution_policy)

    @property
    def is_agentic(self) -> bool:
        return self.dataflow_type == "agentic"

    def to_dict(self) -> dict:
        """JSON-safe form for the fixed mode's cross-process hand-off."""
        out: Dict[str, Any] = {
            "dataflow_type": self.dataflow_type,
            "stages": list(self.stages),
            "has_query_expansion": self.has_query_expansion,
            "has_reranker": self.has_reranker,
            "has_encode": self.has_encode,
            "has_compressor": self.has_compressor,
        }
        if self.corpus_stats is not None:
            out["corpus_stats"] = self.corpus_stats.to_dict()
        if self.runtime_stats is not None:
            out["runtime_stats"] = self.runtime_stats
        if self.nodes:
            from copy import deepcopy
            out["nodes"] = deepcopy(self.nodes)
            out["execution_order"] = list(self.execution_order)
            out["execution_policy"] = self.execution_policy
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "WorkflowSchema":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"WorkflowSchema has unsupported fields: {sorted(unknown)}"
            )
        payload = dict(d)
        stats = payload.get("corpus_stats")
        if isinstance(stats, dict):
            payload["corpus_stats"] = CorpusStats.from_dict(stats)
        return cls(**payload)


def dataflow_type_of(pipeline_config: dict) -> str:
    """Derive the dataflow type from the resolved pipeline's runtime.

    ``"agentic"`` iff the pipeline runs a variable-round reasoning loop — an
    agentic ``rag_dataflow`` (react) or an iterative quality method
    (``flare`` / ``ircot``); otherwise ``"sequential"``. NOTE: the co-mode
    resolver always carries the chosen ``rag_dataflow`` in pipeline_runtime
    — INCLUDING ``"sequential"`` — so the test is on the VALUE, not presence
    (the historical presence test misrouted every sequential co-mode trial
    to the agentic engine).
    """
    runtime = pipeline_config.get("pipeline_runtime") or {}
    mode = runtime.get("mode", "sequential")
    dataflow = str(runtime.get("rag_dataflow") or "").lower()
    agentic = (
        dataflow not in ("", "sequential")
        or mode in {"flare", "ircot", "react", "corag", "searchr1"}
    )
    return "agentic" if agentic else "sequential"


def build_workflow_schema(
    pipeline_config: dict,
    stages: Optional[List[str]] = None,
    *,
    trace=None,
) -> WorkflowSchema:
    """Build the schema from the resolved ``pipeline_config`` (+ optional
    PerformanceStage-name list from ``build_performance_stage_cost_models``).

    Automatically lifts the producer-stamped corpus statistics
    (``_token_stats`` + ``vectordb[i]["N"]``) into ``corpus_stats`` — every
    existing 1-arg/2-arg call site gets the frozen-contract schema without
    threading new parameters.

    ``trace`` optionally supplies the evaluator's captured calls. Observed local
    fusion is then bound as a component without changing the caller's config.
    Query expansion alone does not declare a fusion algorithm.
    """
    from rag_stack.rag_ir.plan import structural_nodes, structural_stages

    observed_calls = []
    if trace is not None:
        from rag_stack.rag_ir.trace import normalize_quality_trace
        from rag_stack.rag_ir.normalize import with_observed_trace_components

        normalized_traces = normalize_quality_trace(trace).traces
        observed_calls = [call for calls in normalized_traces for call in calls]
        pipeline_config = with_observed_trace_components(pipeline_config, normalized_traces)
    else:
        observed_calls = pipeline_config.get("_evaluator_fusion_calls") or []
    nodes = structural_nodes(pipeline_config)
    derived_stages = structural_stages(nodes, observed_calls)
    if stages is None:
        stages = derived_stages
    sset = set(stages)
    # Every observed operation must be declared by the workflow. Conditional
    # workflows may skip configured stages, so this is a
    # subset check, not an equality requirement on observed calls.
    for observed in pipeline_config.get("_evaluator_trace_stages", []):
        pair = {f"{observed}_prefill", f"{observed}_decode"}
        represented = pair <= sset if observed in {"generator", "query_expansion"} else (
            observed in sset or pair <= sset
        )
        if not represented:
            raise ValueError(
                f"Evaluator trace stage {observed!r} has no declared workflow operation"
            )
    derived = WorkflowSchema(
        dataflow_type=dataflow_type_of(pipeline_config),
        stages=list(stages),
        has_query_expansion="query_expansion_prefill" in sset,
        has_reranker="passage_reranker" in sset,
        has_encode="semantic_retrieval_encode" in sset,
        has_compressor=any(s.startswith("passage_compressor") for s in sset),
        corpus_stats=corpus_stats_from_pipeline_config(pipeline_config),
        nodes=nodes,
        execution_order=[node["node_id"] for node in nodes],
    )
    explicit = pipeline_config.get("_evaluator_workflow_schema")
    if explicit is None:
        if trace is not None:
            validate_workflow_trace(derived, trace)
        return derived
    if not isinstance(explicit, dict):
        raise TypeError("Evaluator workflow schema must be a dictionary")
    schema = WorkflowSchema.from_dict(explicit)
    if any(node["stage"] == "hybrid_retrieval" for node in nodes):
        identity_fields = ("node_id", "stage", "components", "inputs")
        declared = [{key: node[key] for key in identity_fields} for node in nodes]
        supplied = [{key: node[key] for key in identity_fields} for node in schema.nodes]
        if declared != supplied:
            raise ValueError("Hybrid evaluator workflow must preserve the configured node identities and inputs")
    if schema.dataflow_type not in {"sequential", "agentic"}:
        raise ValueError("Evaluator workflow must declare sequential or agentic dataflow")
    if len(schema.stages) != len(set(schema.stages)) or set(schema.stages) != sset:
        raise ValueError(
            "Evaluator workflow stages must match the declared pipeline operations; "
            f"workflow={schema.stages}, resolved={list(stages)}"
        )
    for flag in ("has_query_expansion", "has_reranker", "has_encode", "has_compressor"):
        if not isinstance(getattr(schema, flag), bool) or getattr(schema, flag) != getattr(derived, flag):
            raise ValueError(f"Evaluator workflow {flag} conflicts with resolved components")
    # The external runner owns orchestration. Host-owned corpus statistics remain
    # available when a runner supplies only the structural portion of the IR.
    if schema.corpus_stats is None:
        schema.corpus_stats = derived.corpus_stats
    if trace is not None:
        validate_workflow_trace(schema, trace)
    return schema


def validate_workflow_nodes(nodes: list, order: list, policy: str) -> None:
    """Validate the serial structural contract without importing any executor."""
    from rag_stack.rag_ir.call_identity import copy_input_refs, validate_identifier

    if policy != "sequential":
        raise ValueError("Only sequential execution_policy is currently supported")
    if not isinstance(nodes, list) or not isinstance(order, list):
        raise ValueError("workflow nodes and execution_order must be lists")
    seen = set()
    stages_by_node = {}
    for node in nodes:
        required = {"node_id", "stage", "components", "inputs", "operations"}
        if not isinstance(node, dict) or set(node) != required:
            raise ValueError(f"workflow node requires exactly {sorted(required)}")
        for key in ("node_id", "stage"):
            validate_identifier(node[key], where=f"workflow node {key}")
        if node["node_id"] in seen:
            raise ValueError("workflow node_id must be unique")
        for key in ("components", "operations"):
            values = node[key]
            if not isinstance(values, list):
                raise ValueError(f"workflow node {key} must be a list")
            for value in values:
                validate_identifier(value, where=f"workflow node {key}")
            if len(values) != len(set(values)):
                raise ValueError(f"workflow node {key} must be unique")
        for refs in copy_input_refs(node["inputs"], where="workflow node inputs").values():
            if not set(refs) <= seen:
                raise ValueError("workflow inputs must reference preceding nodes")
        if node["stage"] == "hybrid_retrieval":
            expected_ports = {"dense": "semantic_retrieval", "lexical": "lexical_retrieval"}
            if set(node["inputs"]) != set(expected_ports):
                raise ValueError("Hybrid workflow inputs require dense and lexical ports")
            for port, stage in expected_ports.items():
                producers = node["inputs"][port]
                if len(producers) != 1 or stages_by_node[producers[0]] != stage:
                    raise ValueError(f"Hybrid {port} input requires one {stage} producer")
        seen.add(node["node_id"])
        stages_by_node[node["node_id"]] = node["stage"]
    if order != [node["node_id"] for node in nodes]:
        raise ValueError("execution_order must list every workflow node in serial order")


def validate_workflow_trace(workflow: WorkflowSchema, trace: dict) -> None:
    """Check actual calls against their declared nodes and hybrid producers.

    Old traces without node identity remain readable. A declared hybrid
    workflow requires complete identities/profiles; it cannot use a legacy
    dense-only trace or reconstruct missing fusion observations.
    """
    from rag_stack.rag_ir.trace import validate_quality_trace_envelope

    validate_quality_trace_envelope(trace)
    nodes = {node["node_id"]: node for node in workflow.nodes}
    node_positions = {node_id: index for index, node_id in enumerate(workflow.execution_order)}
    hybrid_nodes = {key: node for key, node in nodes.items() if node["stage"] == "hybrid_retrieval"}
    required_retrieval = {}
    for node_id, node in hybrid_nodes.items():
        required_retrieval[node_id] = {"retrieval_fusion"}
        required_retrieval[node["inputs"]["dense"][0]] = {
            "semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
        }
        required_retrieval[node["inputs"]["lexical"][0]] = {"lexical_retrieval"}
    for query in trace["queries"]:
        prior = {}
        seen_hybrid = set()
        latest_by_node = {}
        last_node_position = -1
        observed_by_node = {}
        for call in query["calls"]:
            node_id = call.get("node_id")
            if node_id is not None and nodes:
                if node_id not in nodes:
                    raise ValueError(f"Trace node {node_id!r} is absent from workflow")
                if call["stage"] not in nodes[node_id]["operations"]:
                    raise ValueError(f"Trace operation {call['stage']!r} is absent from node {node_id!r}")
                observed_by_node.setdefault(node_id, set()).add(call["stage"])
                if workflow.dataflow_type == "sequential":
                    position = node_positions[node_id]
                    if position < last_node_position:
                        raise ValueError("Trace node calls must follow the declared sequential execution_order")
                    last_node_position = position
            if node_id in hybrid_nodes:
                if call["stage"] != "retrieval_fusion":
                    raise ValueError("Hybrid nodes emit retrieval_fusion operations only")
                expected = hybrid_nodes[node_id]["inputs"]
                refs = call.get("input_refs", {})
                if set(refs) != set(expected):
                    raise ValueError("Hybrid fusion requires explicit dense and lexical input references")
                for port, producers in expected.items():
                    actual = refs[port]
                    if len(actual) != len(producers) or any(
                        prior[ref].get("node_id") != producer or latest_by_node.get(producer) != ref
                        for ref, producer in zip(actual, producers)
                    ):
                        raise ValueError(f"Hybrid {port} must reference its producer's current result")
                if "input_profile" not in call or "output_profile" not in call:
                    raise ValueError("Hybrid fusion requires actual input and output profiles")
                sizes = call["input_profile"].get("list_sizes")
                producer_sizes = [
                    prior[ref].get("output_profile", {}).get("items")
                    for port in ("dense", "lexical") for ref in refs.get(port, [])
                ]
                if sizes is None or any(size is None for size in producer_sizes):
                    raise ValueError("Hybrid fusion requires candidate counts from both actual producers")
                if sizes != producer_sizes:
                    raise ValueError("Hybrid fusion input list sizes disagree with referenced producer outputs")
                seen_hybrid.add(node_id)
            if "call_id" in call:
                prior[call["call_id"]] = call
                if node_id is not None:
                    latest_by_node[node_id] = call["call_id"]
        if set(hybrid_nodes) != seen_hybrid:
            raise ValueError("Hybrid workflow requires observed fusion for every query; recollect the trace")
        for node_id, required_operations in required_retrieval.items():
            if not required_operations <= observed_by_node.get(node_id, set()):
                raise ValueError(
                    f"Hybrid workflow requires observed retrieval operations for {node_id!r}; recollect the trace"
                )
