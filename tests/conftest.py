"""Shared test options; model execution belongs in explicit diagnostics."""

import pytest


@pytest.fixture
def anyio_backend():
    """The host runtime and its async tests use asyncio."""
    return "asyncio"
