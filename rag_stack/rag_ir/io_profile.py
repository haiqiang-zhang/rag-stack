"""Hardware-independent summaries of structured component inputs and outputs.

Profiles contain counts only. Summarizing identifiers never stores them in the
trace, and missing byte volumes remain unknown rather than being inferred.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

__all__ = [
    "validate_io_profile", "copy_io_profile", "summarize_items", "summarize_lists",
]

_FIELDS = frozenset({"items", "list_sizes", "list_unique_items", "unique_items"})


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def validate_io_profile(profile: Any, *, where: str = "I/O profile") -> None:
    """Validate a JSON count summary without changing it.

    ``items`` counts all input occurrences. ``unique_items`` counts their union;
    optional list fields preserve each input list's size and distinct count.
    Unknown values are omitted, never filled with zero.
    """
    if not isinstance(profile, dict):
        raise TypeError(f"{where} must be an object")
    unknown = set(profile) - _FIELDS
    if unknown:
        raise ValueError(f"{where} has unsupported fields: {sorted(unknown)}")
    if "items" not in profile:
        raise ValueError(f"{where} requires items")
    items = _count(profile["items"], f"{where}.items")
    sizes = profile.get("list_sizes")
    if "list_sizes" in profile:
        if not isinstance(sizes, list):
            raise TypeError(f"{where}.list_sizes must be a list")
        for index, value in enumerate(sizes):
            _count(value, f"{where}.list_sizes[{index}]")
        if sum(sizes) != items:
            raise ValueError(f"{where}.list_sizes must sum to items")
    distinct = profile.get("list_unique_items")
    if "list_unique_items" in profile:
        if sizes is None:
            raise ValueError(f"{where}.list_unique_items requires list_sizes")
        if not isinstance(distinct, list) or len(distinct) != len(sizes):
            raise ValueError(f"{where}.list_unique_items must match list_sizes")
        for index, (value, size) in enumerate(zip(distinct, sizes)):
            _count(value, f"{where}.list_unique_items[{index}]")
            if value > size or (size > 0 and value == 0):
                raise ValueError(f"{where}.list_unique_items is inconsistent with list_sizes")
    if "unique_items" in profile:
        unique = _count(profile["unique_items"], f"{where}.unique_items")
        if unique > items or (items > 0 and unique == 0):
            raise ValueError(f"{where}.unique_items is inconsistent with items")
        if distinct is not None and not max(distinct, default=0) <= unique <= sum(distinct):
            raise ValueError(f"{where}.unique_items is outside the input-list union bounds")


def copy_io_profile(profile: Any) -> dict:
    """Return a validated, independent JSON summary."""
    validate_io_profile(profile)
    return deepcopy(profile)


def _identifiers(values: Iterable[Any]) -> Iterable[Any]:
    if isinstance(values, (str, bytes)):
        raise TypeError("identifiers must be an iterable of item IDs, not text")
    return values


def summarize_items(ids: Iterable[Any]) -> dict:
    """Count occurrences and distinct hashable IDs without retaining IDs."""
    unique = set()
    count = 0
    for item in _identifiers(ids):
        unique.add(item)
        count += 1
    return {"items": count, "unique_items": len(unique)}


def summarize_lists(list_of_ids: Iterable[Iterable[Any]]) -> dict:
    """Summarize actual ranked lists, including duplicates and their union."""
    sizes, distinct, union = [], [], set()
    for values in _identifiers(list_of_ids):
        unique = set()
        count = 0
        for item in _identifiers(values):
            unique.add(item)
            count += 1
        sizes.append(count)
        distinct.append(len(unique))
        union.update(unique)
    return {
        "items": sum(sizes), "list_sizes": sizes,
        "list_unique_items": distinct, "unique_items": len(union),
    }
