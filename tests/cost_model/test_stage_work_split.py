from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_stack.cost_model.stage_work_split import logical_cores


def test_logical_cores_uses_declared_target_not_controller_host(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 999)

    assert logical_cores(SimpleNamespace(cpu_preset="", num_cores=7)) == 7
    with pytest.raises(ValueError, match="local controller host"):
        logical_cores(SimpleNamespace(cpu_preset="", num_cores=0))


def test_logical_cores_unknown_preset_fails_closed():
    with pytest.raises(ValueError, match="not resolvable"):
        logical_cores(SimpleNamespace(cpu_preset="unknown_cpu", num_cores=99))
