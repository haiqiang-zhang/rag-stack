"""Process-stable Greedy proposal and checkpoint-replay regressions.

The Greedy seed must fully determine proposal identity.  In particular, a
checkpoint resumed by a fresh Python process cannot inherit entropy from that
process's randomized hash seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.greedy import GreedyParetoOptimizer
from rag_stack.search_space import AxSearchSpaceOps


_OBJECTIVES = [
    {"objective_name": "performance"},
    {"objective_name": "quality"},
]
_VALUES = {
    "alpha__x": ["a0", "a1", "a2"],
    "beta__x": [f"b{i}" for i in range(6)],
    "gamma__x": [f"g{i}" for i in range(6)],
    "delta__x": [f"d{i}" for i in range(6)],
    "epsilon__x": [f"e{i}" for i in range(6)],
    "zeta__x": [f"z{i}" for i in range(6)],
}


def _space() -> AxSearchSpaceOps:
    return AxSearchSpaceOps(SearchSpace(parameters=[
        ChoiceParameter(
            name,
            ParameterType.STRING,
            values=values,
            is_ordered=False,
            sort_values=False,
        )
        for name, values in _VALUES.items()
    ]))


def _objectives(config: dict) -> dict[str, list[float]]:
    # Every coordinate moves along a deterministic Pareto trade-off, keeping
    # multiple breeding bases alive and making their order observable.
    score = 0
    for weight, (stage, values) in enumerate(_VALUES.items(), start=1):
        score += weight * values.index(config[stage.split("__", 1)[0]]["x"])
    return {
        "performance": [float(10_000 - score)],
        "quality": [float(score)],
    }


def _optimizer(n_iterations: int = 44) -> GreedyParetoOptimizer:
    ssops = _space()
    return GreedyParetoOptimizer(
        ssops.ss,
        _OBJECTIVES,
        ssops=ssops,
        search_mode="forward",
        n_iterations=n_iterations,
        seed=43,
        beam_width=3,
        stage_weight="log",
    )


def _drive(optimizer, count: int, *, start_eval_id: int = 0):
    history = []
    identities = []
    for offset in range(count):
        configs, _ = optimizer.suggest()
        if configs is None:
            break
        config = configs[0]
        objectives = _objectives(config)
        eval_id = start_eval_id + offset
        optimizer.add_objectives(configs, objectives, eval_ids=[eval_id])
        identities.append(json.dumps(config, sort_keys=True, separators=(",", ":")))
        history.append({
            "config": config,
            "objectives": {
                "performance": objectives["performance"][0],
                "quality": objectives["quality"][0],
            },
            "eval_id": eval_id,
        })
    return history, identities


_CHILD_SCRIPT = r"""
import json
import logging

logging.disable(logging.CRITICAL)

from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace
from rag_stack.optimizer.greedy import GreedyParetoOptimizer
from rag_stack.search_space import AxSearchSpaceOps

values = {
    "alpha__x": ["a0", "a1", "a2"],
    "beta__x": [f"b{i}" for i in range(6)],
    "gamma__x": [f"g{i}" for i in range(6)],
    "delta__x": [f"d{i}" for i in range(6)],
    "epsilon__x": [f"e{i}" for i in range(6)],
    "zeta__x": [f"z{i}" for i in range(6)],
}
ssops = AxSearchSpaceOps(SearchSpace(parameters=[
    ChoiceParameter(
        name, ParameterType.STRING, values=choices,
        is_ordered=False, sort_values=False,
    )
    for name, choices in values.items()
]))
optimizer = GreedyParetoOptimizer(
    ssops.ss,
    [{"objective_name": "performance"}, {"objective_name": "quality"}],
    ssops=ssops,
    search_mode="forward",
    n_iterations=44,
    seed=43,
    beam_width=3,
    stage_weight="log",
)
identities = []
for eval_id in range(16):
    configs, _ = optimizer.suggest()
    if configs is None:
        break
    config = configs[0]
    score = 0
    for weight, (stage, choices) in enumerate(values.items(), start=1):
        score += weight * choices.index(config[stage.split("__", 1)[0]]["x"])
    optimizer.add_objectives(
        configs,
        {"performance": [float(10_000 - score)], "quality": [float(score)]},
        eval_ids=[eval_id],
    )
    identities.append(json.dumps(config, sort_keys=True, separators=(",", ":")))
print(json.dumps(identities))
"""


def _fresh_process_identities(python_hash_seed: str) -> list[str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = python_hash_seed
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_proposals_are_identical_across_python_hash_seeds():
    seed_one = _fresh_process_identities("1")
    seed_two = _fresh_process_identities("2")

    assert len(seed_one) >= 12
    assert seed_two == seed_one


def test_checkpoint_replay_consumes_all_38_outcomes_without_flush():
    uninterrupted = _optimizer()
    history, full_identities = _drive(uninterrupted, 44)
    assert len(history) == 44

    resumed = _optimizer()
    resumed.warm_start(history[:38])
    suffix_history, suffix_identities = _drive(
        resumed, 6, start_eval_id=38,
    )

    assert resumed._replay_fed == 38
    assert not resumed._replay_queue
    assert resumed._replay_miss_streak == 0
    assert len(suffix_history) == 6
    assert suffix_identities == full_identities[38:]
