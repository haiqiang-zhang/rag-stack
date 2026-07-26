"""Structural CPU facts shared by cost-model scheduling."""

from __future__ import annotations

from typing import Any

def logical_cores(hardware: Any) -> int:
    """Logical CPU cores of the deployment box.

    Resolution order: the system config's CPU preset (a declared hardware
    fact, e.g. epyc_9124 -> 32), then an explicitly declared positive
    ``num_cores``. Missing/unknown target identity fails closed; the CM never
    substitutes the controller host's ``os.cpu_count()``.
    """
    preset = getattr(hardware, "cpu_preset", "") or ""
    if preset:
        try:
            from rag_stack.cost_model.cpu_specs import get_spec
            n = int(get_spec(preset).num_cores)
            if n > 0:
                return n
        except Exception as exc:
            raise ValueError(
                f"CPU preset {preset!r} is not resolvable to an exact core count"
            ) from exc
    n = int(getattr(hardware, "num_cores", 0) or 0)
    if n > 0:
        return n
    raise ValueError(
        "system.cpu must resolve to a preset or declare a positive num_cores; "
        "the local controller host is not a pricing fallback"
    )
