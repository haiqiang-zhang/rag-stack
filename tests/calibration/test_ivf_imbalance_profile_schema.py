"""CPU-only tests for the CM IVF imbalance serialization boundary."""

from __future__ import annotations

from tests import REPO_ROOT

import json
from pathlib import Path

import pytest

from rag_stack.calibration_inventory import _ivf_workload_validation_error
from rag_stack.cost_model.faiss_ivf_sim.imbalance import (
    load_imbalance_profile,
    save_imbalance_profile,
)
from rag_stack.ivf_imbalance_profile_schema import (
    unwrap_ivf_imbalance_profile,
)
PROFILE_ROOT = REPO_ROOT / "calibration_profiles" / "faiss_ivf" / "imbalance"
PAYLOAD = {
    "d": 4,
    "N": 100,
    "nlist": {"8": {"1": 1.25}},
}


def _document() -> dict:
    return {"profile": PAYLOAD}


@pytest.mark.parametrize(
    "invalid",
    [
        PAYLOAD,
        {},
        {"profile": []},
        {**_document(), "unexpected": "field"},
    ],
    ids=[
        "bare-payload",
        "empty",
        "non-object-profile",
        "extra-field",
    ],
)
def test_loader_rejects_every_non_exact_envelope(tmp_path, invalid):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError):
        load_imbalance_profile("fixture", directory=str(tmp_path))


def test_writer_emits_exact_envelope_and_loader_returns_payload(tmp_path):
    path = Path(save_imbalance_profile("fixture", PAYLOAD, directory=str(tmp_path)))
    document = json.loads(path.read_text(encoding="utf-8"))

    assert set(document) == {"profile"}
    assert document["profile"] == PAYLOAD
    assert load_imbalance_profile("fixture", directory=str(tmp_path)) == PAYLOAD


def test_writer_rejects_a_non_object_payload(tmp_path):
    with pytest.raises(ValueError, match="payload must be a JSON object"):
        save_imbalance_profile("fixture", [], directory=str(tmp_path))


def test_committed_imbalance_profiles_use_the_exact_envelope():
    paths = sorted(PROFILE_ROOT.glob("*.json"))
    assert paths

    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        profile = unwrap_ivf_imbalance_profile(document, source=str(path))
        assert set(profile) == {"d", "N", "nlist"}
        assert load_imbalance_profile(
            path.stem,
            directory=str(PROFILE_ROOT),
        ) == profile
        assert _ivf_workload_validation_error(document) is None
