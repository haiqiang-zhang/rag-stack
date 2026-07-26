"""Inline ``{ordered: [...]}`` value form — semantic orderings at the knob.

String choices are UNORDERED by default (components, index families). A knob
whose YAML value order IS the semantic order (the Qwen model-size ladder:
1.5B < 3B < 7B < 14B — string sort scrambles it to 1.5B < 14B < 3B < 7B)
declares it INLINE, right where its values live (same family as the
``{range: [lo, hi]}`` form):

    model:
      ordered:
      - Qwen/Qwen2.5-1.5B-Instruct
      - ...

The factory builds ``is_ordered=True`` in declaration order, so EVERY
optimizer reads the same flag: ssops moves the dim from the Hamming set to
numeric-kernel treatment, SMAC builds an OrdinalHyperparameter, Ax
transforms encode ordinal levels, and greedy's grid walker probes trade-off
endpoints first. ``is_sweepable``/``scalarify_value``/the validator all
treat the form as a first-class sweep spec.

Run: python -m pytest tests/config/test_ordered_knobs.py -q
"""
from __future__ import annotations

from tests import REPO_ROOT

from pathlib import Path

import pytest
import yaml


from ax.core.parameter import ChoiceParameter

from rag_stack.search_space import SEP, RagSearchSpace

_CONFIG_PATH = (
    REPO_ROOT
    / "configs" / "rag_stack" / "paper_bench_greedy_forward_s43.yaml"
)

_LADDER = [f"Qwen/Qwen2.5-{s}-Instruct" for s in ("1.5B", "3B", "7B", "14B")]


def _cfg() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_declared_model_dims_promoted_in_declaration_order():
    ss = RagSearchSpace.from_yaml(_cfg())
    model_dims = [n for n in ss.param_names if n.endswith(f"{SEP}model")]
    assert len(model_dims) >= 4  # generator + 3 QE components
    for n in model_dims:
        p = ss.ss.parameters[n]
        assert isinstance(p, ChoiceParameter) and p.is_ordered, n
        assert list(p.values) == _LADDER, n  # YAML order, never string-sorted


def test_undeclared_string_choices_stay_categorical():
    """Every UNDECLARED string choice with ≥3 values stays unordered.
    (2-value choices are excluded: Ax itself forces ``is_ordered=True`` on
    binary parameters, where ordinal ≡ categorical anyway.)"""
    ss = RagSearchSpace.from_yaml(_cfg())
    checked = 0
    for n, p in ss.ss.parameters.items():
        if not isinstance(p, ChoiceParameter) or n.endswith(f"{SEP}model"):
            continue
        if len(p.values) < 3 or any(isinstance(v, (int, float)) for v in p.values):
            continue
        assert not p.is_ordered, n
        checked += 1
    assert checked >= 2  # embedding_model dims, components, …
    # suffix 'model' must NOT have matched 'embedding_model'
    emb = [n for n in ss.param_names if n.endswith(f"{SEP}embedding_model")]
    assert emb and all(not ss.ss.parameters[n].is_ordered for n in emb)


def _generator_module(cfg: dict) -> dict:
    for nl in cfg["algo_search_space"]["pipeline"]["node_lines"]:
        for node in nl["nodes"]:
            if node.get("stage") == "generator":
                return node["modules"][0]
    raise AssertionError("generator module not found")


def test_malformed_ordered_forms_raise():
    cfg = _cfg()
    _generator_module(cfg)["model"] = {"ordered": "Qwen/Qwen2.5-7B-Instruct"}
    with pytest.raises(ValueError, match="ordered"):
        RagSearchSpace.from_yaml(cfg)

    cfg2 = _cfg()
    _generator_module(cfg2)["model"] = {"ordered": ["Qwen/Qwen2.5-7B-Instruct"]}
    with pytest.raises(ValueError, match="ordered"):
        RagSearchSpace.from_yaml(cfg2)


def test_sweep_spec_contract_and_validator_flatten():
    from rag_stack.config_validator import _flatten_values
    from rag_stack.search_space.rag_search_space import (
        is_sweepable,
        scalarify_value,
    )

    spec = {"ordered": ["a", "b", "c"]}
    assert is_sweepable(spec)
    with pytest.raises(ValueError, match="range/ordered"):
        scalarify_value(spec)
    assert _flatten_values(spec) == ["a", "b", "c"]


def test_ssops_treats_declared_dim_as_numeric_kernel_with_index_spacing():
    ss = RagSearchSpace.from_yaml(_cfg())
    n = next(n for n in ss.param_names if n.endswith(f"{SEP}model"))
    i = ss.param_names.index(n)
    assert i not in ss.cat_dims_set          # ordinal → numeric-kernel dim
    assert ss.choice_grid(n) == [0.0, 1 / 3, 2 / 3, 1.0]  # index spacing


def test_smac_builds_ordinal_with_declaration_sequence():
    from ConfigSpace.hyperparameters import OrdinalHyperparameter

    from rag_stack.optimizer.smac.configspace import ax_to_configspace

    ss = RagSearchSpace.from_yaml(_cfg())
    cs = ax_to_configspace(ss.ss, seed=1)
    n = next(n for n in ss.param_names if n.endswith(f"{SEP}model"))
    hp = cs[n]
    assert isinstance(hp, OrdinalHyperparameter)
    assert list(hp.sequence) == _LADDER


def test_numeric_auto_ordered_untouched_and_declaration_append_only():
    """The contract: numeric choice grids are AUTO-ordered by the factory
    (sorted values) regardless of any declaration, and ordered_knobs is
    APPEND-only — declaring an already-ordered numeric knob is a harmless
    no-op; nothing can ever demote is_ordered True → False."""
    # numeric grids ordered with NO ordered-wrapping at all
    cfg = _cfg()
    ss0 = RagSearchSpace.from_yaml(cfg)
    numeric = [
        n for n, p in ss0.ss.parameters.items()
        if isinstance(p, ChoiceParameter)
        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in p.values)
        and len(p.values) >= 3
    ]
    assert numeric
    assert all(ss0.ss.parameters[n].is_ordered for n in numeric)

    # wrapping a NUMERIC knob in {ordered: [...]} is redundant-but-harmless:
    # Ax requires numeric ordered choices sorted (magnitude IS the semantic
    # order), so it builds exactly like the plain list — never demoted
    cfg2 = _cfg()
    _generator_module(cfg2)["temperature"] = {"ordered": [1.0, 0.7, 0.3, 0.0]}
    ss1 = RagSearchSpace.from_yaml(cfg2)
    t = next(
        n for n in ss1.param_names
        if n.endswith(f"generator{SEP}vllm{SEP}temperature")
    )
    p = ss1.ss.parameters[t]
    assert p.is_ordered and list(p.values) == [0.0, 0.3, 0.7, 1.0]


def test_raw_config_consumers_handle_ordered_form():
    """Consumers peeking at UNRESOLVED YAML values must be shape-agnostic —
    TokenStats fed a whole {ordered: [...]} dict into the tokenizer resolver
    and crashed a live resume on .lower(). sweep_values is the one helper."""
    from rag_stack.cost_model.token_stats import TokenStats
    from rag_stack.search_space.rag_search_space import sweep_values

    assert sweep_values("x") == ["x"]
    assert sweep_values(["a", "b"]) == ["a", "b"]
    assert sweep_values({"ordered": ["a", "b", "c"]}) == ["a", "b", "c"]
    assert sweep_values({"range": [1, 8]}) == [1, 8]

    cfg = _cfg()
    model = TokenStats._find_generator_model(cfg)
    assert model == "Qwen/Qwen2.5-1.5B-Instruct"  # first of the ladder
