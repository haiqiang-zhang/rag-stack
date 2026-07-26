from rag_stack.cost_model.faiss_ivf_sim.adapter import FaissIVFCostModel
from rag_stack.cost_model.faiss_ivf_sim.imbalance import (
    imbalance_anchor_nlist,
    profile_cell_imbalance,
)
from rag_stack.cost_model.workload_profile import _ivf_imbalance_anchor_nlists


def test_imbalance_anchor_uses_default_factor_four():
    assert imbalance_anchor_nlist(2715) == 208
    assert imbalance_anchor_nlist(30_484) == 698
    assert imbalance_anchor_nlist(273_656) == 2092
    assert _ivf_imbalance_anchor_nlists(2715) == [208]


def test_single_curve_coverage_matched_lookup(monkeypatch):
    """THE corpus×embedding curve serves every nlist via coverage matching:
    a request at a different nlist maps to the point with the same probed
    fraction (nprobe/nlist), never to a second curve."""
    from rag_stack.cost_model.faiss_ivf_sim import imbalance

    profile = {"nlist": {"208": {"10": 1.25, "40": 1.10}}}
    monkeypatch.setattr(imbalance, "load_imbalance_profile", lambda key, directory=None: profile)

    # exact hit wins verbatim
    assert profile_cell_imbalance("d", "e", nlist=208, nprobe=10) == 1.25
    # nlist=417, nprobe=20 → same coverage as (208, ~10) → 1.25
    assert profile_cell_imbalance("d", "e", nlist=417, nprobe=20) == 1.25
    # nlist=104, nprobe=20 → same coverage as (208, 40) → 1.10
    assert profile_cell_imbalance("d", "e", nlist=104, nprobe=20) == 1.10


def test_get_or_build_never_grows_a_nonempty_profile(monkeypatch):
    from rag_stack.cost_model.faiss_ivf_sim import imbalance

    profile = {"nlist": {"208": {"10": 1.25, "40": 1.10}}}
    saved = []
    monkeypatch.setattr(imbalance, "load_imbalance_profile", lambda key, directory=None: profile)
    monkeypatch.setattr(imbalance, "save_imbalance_profile",
                        lambda key, prof, directory=None: saved.append(prof))

    def exploding_build():
        raise AssertionError("single-curve policy must never rebuild an index")

    out = imbalance.get_or_build_imbalance(
        "k", 417, [20], build_index=exploding_build, queries=None,
    )
    assert out == {20: 1.25}   # coverage-matched from THE 208 curve
    assert not saved            # and nothing written


def test_profile_cell_imbalance_uses_nearest_nprobe(monkeypatch):
    from rag_stack.cost_model.faiss_ivf_sim import imbalance

    profile = {"nlist": {"128": {"1": 1.5, "16": 1.2}}}
    monkeypatch.setattr(imbalance, "load_imbalance_profile", lambda key, directory=None: profile)

    assert profile_cell_imbalance(
        "eli5",
        "huggingface_all_mpnet_base_v2",
        nlist=128,
        nprobe=12,
    ) == 1.2
