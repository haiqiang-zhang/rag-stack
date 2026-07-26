"""Ax SearchSpace <-> SMAC ConfigSpace conversion helpers.

SMAC optimizes over a ConfigSpace ``ConfigurationSpace`` rather than an Ax
``SearchSpace``, so the single SMAC baseline (``SMACMOBOOptimizer``) needs this
translation layer. Hierarchical search spaces are preserved: Ax's
``ChoiceParameter.dependents`` become ConfigSpace ``EqualsCondition`` edges, so
the conditional/gated structure is expressed natively on both sides.

This module is conversion-only — it owns no optimizer. ``SMACMOBOOptimizer``
(``rag_stack/optimizer/baseline/smac_baseline.py``) imports ``ax_to_configspace``
and ``cs_config_to_dict`` from here.
"""

import logging
from typing import Optional

from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    Integer,
)
from ConfigSpace.hyperparameters import OrdinalHyperparameter


def _ordinal_values(param) -> "Optional[list]":
    """Native ascending sequence when EVERY choice value is numeric (bools
    excluded) — the ConfigSpace twin of the Ax side's ordered-choice encoding
    (ssops.choice_grid, 07-03 alignment): SMAC's RF then sees rank structure
    on max_iter/top_k/batch/... instead of one-hot categories. String choices
    DECLARED ordered (algo_search_space.ordered_knobs → ``is_ordered=True``,
    e.g. the Qwen model-size ladder) keep their declaration order. None →
    keep Categorical (undeclared string/mixed choices)."""
    vals = list(param.values)
    if not vals:
        return None
    if all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals
    ):
        return sorted(vals)
    # ≥3 guards against Ax's FORCED is_ordered=True on binary parameters
    # (rag_dataflow, enabled flags): with 2 levels ordinal ≡ categorical,
    # so keep those Categorical and leave the gating conditions untouched.
    if param.is_ordered and len(vals) >= 3:
        return vals  # declaration order — never string-sorted
    return None


def _cs_value(param, val):
    """Cast a raw value to the ConfigSpace representation of ``param``:
    the native numeric element for Ordinal choices, ``str`` for Categorical
    choices (matched through ``param.values`` so YAML strings normalize)."""
    str_values = [str(v) for v in param.values]
    native = param.values[str_values.index(str(val))]
    return native if _ordinal_values(param) is not None else str(native)


logger = logging.getLogger("RAG-Stack")


# ---------------------------------------------------------------------------
# Ax SearchSpace <-> ConfigSpace conversion
# ---------------------------------------------------------------------------

def ax_to_configspace(search_space: SearchSpace, seed: int = 42) -> ConfigurationSpace:
    """Convert an Ax SearchSpace (possibly hierarchical) to a ConfigSpace."""
    cs = ConfigurationSpace(seed=seed)

    hp_map = {}
    for name, param in search_space.parameters.items():
        if isinstance(param, RangeParameter):
            if param.parameter_type == ParameterType.INT:
                hp = Integer(name, bounds=(int(param.lower), int(param.upper)),
                             log=param.log_scale)
            else:
                hp = Float(name, bounds=(float(param.lower), float(param.upper)),
                           log=param.log_scale)
        elif isinstance(param, ChoiceParameter):
            ordinal_seq = _ordinal_values(param)
            if ordinal_seq is not None:
                hp = OrdinalHyperparameter(name, sequence=ordinal_seq)
            else:
                hp = Categorical(name, items=[str(v) for v in param.values])
        else:
            raise ValueError(f"Unsupported parameter type: {type(param)}")
        cs.add(hp)
        hp_map[name] = hp

    for choice_name, choice_param in search_space.parameters.items():
        if not (
            isinstance(choice_param, ChoiceParameter)
            and choice_param.is_hierarchical
        ):
            continue
        for choice_value, dep_names in choice_param.dependents.items():
            # Gate value must match the parent's CS representation (native
            # for Ordinal parents, str for Categorical parents).
            gate_value = _cs_value(choice_param, choice_value)
            for dep_name in dep_names:
                cs.add(EqualsCondition(
                    hp_map[dep_name], hp_map[choice_name], gate_value
                ))

    return cs


# ---------------------------------------------------------------------------
# ConfigSpace Configuration -> config dict conversion
# ---------------------------------------------------------------------------

def cs_config_to_dict(
    config,
    search_space: SearchSpace,
    ssops: Optional[AxSearchSpaceOps] = None,
) -> dict:
    """Convert a ConfigSpace Configuration to a hierarchical config dict.

    Delegates the SEP naming-convention rules to the supplied ``ssops``
    (caller is responsible for using the SAME ssops the consumer
    downstream expects — e.g. controller passes ``RagSearchSpace`` so the
    RAG-namespace lift is applied). Falls back to a fresh generic
    ``AxSearchSpaceOps`` when called without ssops (benchmark / synthetic
    callers that produce 2-segment naming, no namespace lift needed).
    """
    flat: dict = {}
    for name in search_space.parameters:
        value = config.get(name)
        if value is None:
            continue

        param = search_space.parameters[name]
        if isinstance(param, ChoiceParameter):
            str_values = [str(v) for v in param.values]
            idx = str_values.index(str(value))
            value = param.values[idx]
        elif isinstance(param, RangeParameter):
            if param.parameter_type == ParameterType.INT:
                value = int(round(float(value)))
            else:
                value = float(value)
        flat[name] = value

    if ssops is None:
        ssops = AxSearchSpaceOps(search_space)
    result = ssops.decode_to_nested(flat)

    if "nprobe" in result and "nlist" in result:
        result["nprobe"] = min(result["nprobe"], result["nlist"])

    return result


def arm_to_cs_config(
    arm_parameters: dict,
    cs: ConfigurationSpace,
    search_space: SearchSpace,
) -> "Configuration":
    """Convert an Ax ``arm.parameters`` (flat dict, hierarchical-cast) to a
    SMAC ``Configuration``.

    Ax's ``get_sobol(...).gen(n).arms`` produces flat dicts containing only
    ACTIVE params (inactive params absent due to hierarchical conditions).
    We cast each value to the matching ConfigSpace type (Categorical → str,
    int → int, float → float) and pass to ``Configuration(cs, values=...)``,
    which validates against the CS conditions and represents inactive params
    natively.
    """
    from ConfigSpace import Configuration

    values: dict = {}
    for name, val in arm_parameters.items():
        if name not in cs:
            continue
        param = search_space.parameters.get(name)
        if param is None:
            continue
        if isinstance(param, ChoiceParameter):
            values[name] = _cs_value(param, val)
        elif isinstance(param, RangeParameter):
            if param.parameter_type == ParameterType.INT:
                values[name] = int(round(float(val)))
            else:
                values[name] = float(val)
        else:
            values[name] = val
    return Configuration(cs, values=values)


def dict_to_cs_config(
    config_dict: dict,
    cs: ConfigurationSpace,
    search_space: SearchSpace,
    ssops: Optional[AxSearchSpaceOps] = None,
) -> "Configuration":
    """Convert a hierarchical config dict back to a ConfigSpace Configuration.

    Reverses ``cs_config_to_dict`` through the ssops encode→decode→active
    round-trip — the SAME inverse the Ax optimizers' warm_start uses
    (``_active_ax_parameters``). A naive nested-path walk cannot invert the
    decode's namespace conventions (e.g. the component lift, where
    ``query_expansion.component = multi_query_expansion`` replaces a nesting
    level), which silently dropped gated params and made the Configuration
    fail with ActiveHyperparameterNotSetError.

    Raises ``ValueError`` when the input is missing a parameter that is
    ACTIVE for its own gating choices: the encode step would silently fill
    the encoding default, fabricating a configuration that was never
    evaluated (a corrupted checkpoint row must be skipped by the caller,
    not laundered into the surrogate's history).
    """
    from ConfigSpace import Configuration

    from rag_stack.search_space.ax_search_space_ops import SEP

    if ssops is None:
        ssops = AxSearchSpaceOps(search_space)

    # Which flat names does the nested dict ACTUALLY provide? Mirrors
    # encode_flat_config's walk (leaf → name_index) plus the component-
    # qualified names the overlay below re-derives.
    if not hasattr(ssops, "_name_index"):
        ssops._name_index = ssops.build_name_index()
    provided: set = set()

    def _collect(prefix: tuple, val) -> None:
        if isinstance(val, dict):
            comp = val.get("component")
            for k, v in val.items():
                _collect(prefix + (k,), v)
                if comp is not None and k != "component" and not isinstance(v, dict):
                    qualified = SEP.join(list(prefix) + [str(comp), k])
                    if qualified in search_space.parameters:
                        provided.add(qualified)
        else:
            name = ssops._name_index.get(prefix)
            if name is not None:
                provided.add(name)

    for k, v in config_dict.items():
        _collect((k,), v)

    x = ssops.encode_flat_config(config_dict).reshape(-1)
    flat = ssops.decode_flat(x)
    active = ssops.active_params(flat)
    named = {name: flat.get(name) for name in active}

    missing = [
        name for name in active
        if name not in provided
        # A gating choice value the dict DID provide can differ from the
        # roundtrip only for collided leaves the overlay fixes below; the
        # gater names themselves are always leaf-provided when present.
    ]
    if missing:
        raise ValueError(
            f"checkpoint config is missing active parameter(s) {sorted(missing)}; "
            "refusing to fabricate encoding defaults for an evaluated row"
        )

    # Component-aware overlay: ``build_name_index`` maps colliding leaf paths
    # (two components sharing a leaf name, e.g. hyde__temperature vs
    # multi_query_expansion__temperature) to whichever param comes first, so
    # the ssops round-trip silently writes the lifted value into the WRONG
    # component's column and the active component's column keeps its encoding
    # default. The nested dict still knows the truth — its ``component`` field
    # qualifies every sibling leaf — so re-derive those names exactly.
    def _overlay(path_parts: list, d: dict) -> None:
        comp = d.get("component")
        for k, v in d.items():
            if isinstance(v, dict):
                _overlay(path_parts + [k], v)
            elif comp is not None and k != "component":
                name = SEP.join(path_parts + [str(comp), k])
                if name in search_space.parameters:
                    named[name] = v

    _overlay([], config_dict)

    # Cast values to ConfigSpace types (Ordinal native, Categorical str)
    values: dict = {}
    for name, val in named.items():
        if name not in cs or val is None:
            continue
        if hasattr(val, "item"):
            val = val.item()
        param = search_space.parameters.get(name)
        if param is None:
            continue
        if isinstance(param, ChoiceParameter):
            values[name] = _cs_value(param, val)
        elif isinstance(param, RangeParameter):
            if param.parameter_type == ParameterType.INT:
                values[name] = int(round(float(val)))
            else:
                values[name] = float(val)
        else:
            values[name] = val

    return Configuration(cs, values=values)
