"""Single-source reranker forward/padding policy.

Two sides of the project must agree on how an encoder reranker tiles and pads
its query-passage pairs, or the CM prices a forward the real runtime never
executes (r26 autopsy: colbert priced 4.8x under because the CM used the mean
pair length where every real forward runs at the padded call max):

  * the measured runtime (``static_rag_evaluator/measured``) sets the node's
    REAL forward microbatch (``module_param["batch"]``);
  * the cost model (``cost_model/engines.py``) prices per-forward latency at
    the forward shape those semantics imply.

Everything here is a STRUCTURAL fact read off the node implementations —
there are no fitted or tunable values in this module.
"""

from __future__ import annotations

from collections.abc import Mapping

# Measured-runtime forward microbatch for MonoT5 (a 3B encoder-decoder whose
# pair forward OOMs when tied to the request batch while colocated with a live
# vLLM engine). This is runtime POLICY shared verbatim by the CM so both sides
# tile the same number of pairs per forward.
MONOT5_FORWARD_BATCH = 8
MONOT5_FORWARD_EXECUTION_SCHEMA = "monot5_forward_execution"

# How each implementation pads its pairs (code-verified structural facts):
#
#   "call"     one ``tokenizer(padding=True)`` over EVERY pair in the worker
#              call, then sliced into forward-sized views — every forward runs
#              at the CALL-max sequence length
#              (colbert.py ``get_colbert_embedding_batch``: tokenize-then-slice).
#
#   "forward"  pairs are cut into forward microbatches FIRST, each microbatch
#              tokenized/padded separately to its own max
#              (monot5.py ``_score_batch`` via ``make_batch``; tart.py,
#              flag_embedding.py, sentence_transformer.py likewise).
_PADDING_SEMANTICS: dict[str, str] = {
    "colbert_reranker": "call",
}
_DEFAULT_PADDING_SEMANTICS = "forward"


def reranker_forward_batch(
    component: str | None,
    request_batch_size: int,
    override: int | None = None,
) -> int:
    """Pairs per model forward for a GPU reranker component.

    ``override`` is the explicit ``reranker_max_batch_size`` system knob when
    the caller has one (measured side); otherwise the per-component runtime
    policy applies: MonoT5 clamps to :data:`MONOT5_FORWARD_BATCH`, every other
    component forwards at the request batch.
    """
    rb = max(1, int(request_batch_size))
    if override is not None:
        return max(1, min(rb, int(override)))
    if str(component or "").lower() == "monot5":
        return max(1, min(rb, MONOT5_FORWARD_BATCH))
    return rb


def reranker_forward_batch_from_system(
    component: str | None,
    request_batch_size: int,
    system_config: Mapping[str, object] | None,
) -> int:
    """Resolve the reranker forward cap from one concrete system config.

    This is measured-runtime structure shared with the CM. Search-space
    templates and deployment metadata must already have been resolved before
    this boundary.
    """

    if system_config is None:
        resolved: Mapping[str, object] = {}
    elif isinstance(system_config, Mapping):
        resolved = system_config
    else:
        raise TypeError("system_config must be a concrete mapping")
    raw_cap = resolved.get("reranker_max_batch_size")
    if raw_cap is None:
        cap = None
    else:
        if isinstance(raw_cap, bool) or not isinstance(raw_cap, int) or raw_cap <= 0:
            raise ValueError(
                "system.reranker_max_batch_size must be a positive integer"
            )
        cap = int(raw_cap)
    return reranker_forward_batch(
        component,
        request_batch_size,
        override=cap,
    )


def reranker_padding_semantics(component: str | None) -> str:
    """``"call"`` (whole-call padding) or ``"forward"`` (per-microbatch)."""
    return _PADDING_SEMANTICS.get(
        str(component or "").lower(), _DEFAULT_PADDING_SEMANTICS
    )


def reranker_padding_group_pairs(
    component: str | None,
    request_batch_size: int,
    fanout: int,
    forward_batch: int | None = None,
) -> int:
    """Number of pairs padded TOGETHER (the group whose max sets every
    member's padded length): the whole worker call for "call" semantics,
    one forward microbatch for "forward" semantics."""
    rb = max(1, int(request_batch_size))
    if reranker_padding_semantics(component) == "call":
        return rb * max(1, int(fanout))
    if forward_batch is None:
        forward_batch = reranker_forward_batch(component, rb)
    return max(1, int(forward_batch))
