"""Query-local call identities and data dependencies shared by IR producers.

References describe data provenance. They do not authorize parallel execution:
the current contract requires every producer to precede its consumer.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping


def validate_identifier(value: object, *, where: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")


def copy_input_refs(value: object, *, where: str = "input_refs") -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must map named input ports to call-ID lists")
    for port, refs in value.items():
        validate_identifier(port, where=f"{where} port")
        if not isinstance(refs, list) or not refs:
            raise ValueError(f"{where}.{port} must be a non-empty call-ID list")
        for ref in refs:
            validate_identifier(ref, where=f"{where}.{port} reference")
        if len(set(refs)) != len(refs):
            raise ValueError(f"{where}.{port} contains duplicate references")
    return deepcopy(value)


def validate_call_identity(call: Mapping, prior: Mapping[str, Mapping], *, where: str) -> None:
    """Validate an optional identity extension against this query's prior calls."""
    if "node_id" in call or "input_refs" in call:
        if "call_id" not in call:
            raise ValueError(f"{where} node/input identity requires call_id")
    if "call_id" in call:
        validate_identifier(call["call_id"], where=f"{where}.call_id")
        if call["call_id"] in prior:
            raise ValueError(f"{where} has duplicate call_id {call['call_id']!r}")
    if "node_id" in call:
        validate_identifier(call["node_id"], where=f"{where}.node_id")
    if "input_refs" in call:
        refs = copy_input_refs(call["input_refs"], where=f"{where}.input_refs")
        for values in refs.values():
            for ref in values:
                if ref not in prior:
                    raise ValueError(
                        f"{where} input reference {ref!r} must identify an earlier call in the same query"
                    )
