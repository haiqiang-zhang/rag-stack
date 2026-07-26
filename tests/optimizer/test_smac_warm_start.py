"""SMAC checkpoint resume: warm_start must inject history and never re-burn.

SMAC was the only optimizer without ``warm_start`` (base raises
NotImplementedError), which made every measured-baseline restart begin from
eval_0000 — on 2026-07-06 nine restarts of smac_s43 discarded ~18 completed GT
trials (~2.3 h of GPU). These tests pin the resume contract on the REAL
baseline search space:

  1. ``dict_to_cs_config`` (previously unexercised legacy API) round-trips the
     production decode on sampled configs, including hierarchical/gated ones.
  2. ``warm_start`` injects checkpoint rows into the runhistory, and a
     subsequent ``suggest()`` never returns an already-evaluated config —
     the same-seed initial design is answered from history, not re-run.
  3. Nothing injectable → loud RuntimeError (controller's "starting fresh").
"""
import yaml
import pytest

from rag_stack.optimizer.smac.configspace import (
    ax_to_configspace,
    cs_config_to_dict,
    dict_to_cs_config,
)
from rag_stack.optimizer.smac.optimizer import SMACMOBOOptimizer
from rag_stack.optimizer.utils import config_signature
from rag_stack.search_space.rag_search_space import RagSearchSpace

CFG = "configs/baselines/sgs_h100x4_dragonball_smac_measured_s43.yaml"

OBJECTIVES = [
    {"objective_name": "quality", "cost": 1.0, "ref_point": 0.0},
    {"objective_name": "performance", "cost": 1.0, "ref_point": 0.0},
]


@pytest.fixture(scope="module")
def space():
    return RagSearchSpace.from_yaml(yaml.safe_load(open(CFG)))


@pytest.fixture(scope="module")
def cs(space):
    return ax_to_configspace(space.ss, seed=43)


def _make_optimizer(space, seed: int = 43) -> SMACMOBOOptimizer:
    return SMACMOBOOptimizer(
        config_space=space.ss,
        objectives=OBJECTIVES,
        seed=seed,
        n_gt_init=5,
        ssops=space,
    )


def _objectives_like_controller(q: float, p: float) -> dict:
    """Shape of ``valid_evals[i]['objectives']`` built by _load_checkpoint."""
    return {
        "quality": {"combined": q, "answer_correctness": q},
        "performance": p,
    }


# ---------------------------------------------------------------------------
# 1. reverse encoding round-trip on the real space
# ---------------------------------------------------------------------------

def test_dict_to_cs_config_roundtrips_production_decode(space, cs):
    for cfg in cs.sample_configuration(60):
        decoded = cs_config_to_dict(cfg, space.ss, ssops=space)
        recoded = dict_to_cs_config(decoded, cs, space.ss, ssops=space)
        redecoded = cs_config_to_dict(recoded, space.ss, ssops=space)
        assert config_signature(redecoded) == config_signature(decoded)


# ---------------------------------------------------------------------------
# 2. warm_start end-to-end: inject → model has data → no re-burn
# ---------------------------------------------------------------------------

def test_warm_start_injects_and_never_reburns(space):
    # A fresh optimizer produces the seed's first K configs (its initial
    # design prefix) — exactly what a checkpoint of an interrupted run holds.
    donor = _make_optimizer(space)
    checkpoint = []
    for i in range(3):
        configs, _ = donor.suggest()
        donor.add_objectives(
            configs,
            {
                "quality": [{"combined": 0.3 + 0.1 * i}],
                "performance": [10.0 + i],
            },
            eval_ids=[i],
        )
        checkpoint.append(
            {
                "eval_id": i,
                "eval_type": "GT",
                "config": configs[0],
                "objectives": _objectives_like_controller(0.3 + 0.1 * i, 10.0 + i),
            }
        )

    resumed = _make_optimizer(space)
    resumed.warm_start(checkpoint)

    assert len(resumed._smac.runhistory) == 3
    # get_pareto reflects the warm data (controller dashboards after resume)
    pareto_configs, pareto_objs, pareto_ids = resumed.get_pareto()
    assert pareto_configs and set(pareto_ids) <= {0, 1, 2}

    # The resume contract: subsequent suggestions never repeat an injected
    # config (the same-seed initial design must be answered from history).
    seen = {config_signature(ev["config"]) for ev in checkpoint}
    for step in range(4):
        configs, _ = resumed.suggest()
        sig = config_signature(configs[0])
        assert sig not in seen, (
            f"resume re-proposed an already-evaluated config at step {step}"
        )
        resumed.add_objectives(
            configs,
            {"quality": [{"combined": 0.5}], "performance": [20.0]},
            eval_ids=[100 + step],
        )
        seen.add(sig)


def test_warm_start_skips_cm_only_rows(space):
    donor = _make_optimizer(space)
    configs, _ = donor.suggest()
    checkpoint = [
        {
            "eval_id": 0,
            "eval_type": "CM",
            "config": configs[0],
            # decoupled-run CM row: performance only — SMAC (non-decoupled)
            # cannot use it
            "objectives": {"performance": 12.0},
        },
        {
            "eval_id": 1,
            "eval_type": "GT",
            "config": configs[0],
            "objectives": _objectives_like_controller(0.4, 15.0),
        },
    ]
    resumed = _make_optimizer(space)
    resumed.warm_start(checkpoint)
    assert len(resumed._smac.runhistory) == 1


def test_warm_start_raises_when_nothing_injectable(space):
    resumed = _make_optimizer(space)
    with pytest.raises(RuntimeError, match="none of the 1"):
        resumed.warm_start(
            [{"eval_id": 0, "eval_type": "CM", "config": {}, "objectives": {}}]
        )
