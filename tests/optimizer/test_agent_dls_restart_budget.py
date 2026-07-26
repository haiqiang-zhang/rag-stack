"""Self-contained contracts for grouped DLS restart budgeting."""

import inspect

import torch

from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI
from rag_stack.optimizer.agent.dls import DLSMixin
from rag_stack.optimizer.agent.llm_agent import AgentAdvice


class _IdentitySearchSpaceOps:
    def clamp(self, rows):
        return rows


class _RestartHarness(DLSMixin):
    def __init__(self, restart_budget: int):
        self.tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
        self.dls_num_restarts = restart_budget
        self.seed = 17
        self._t = 4
        self.ssops = _IdentitySearchSpaceOps()
        self._x_rows = [
            torch.tensor([0.0, 0.0], **self.tkwargs),
            torch.tensor([0.2, 0.8], **self.tkwargs),
            torch.tensor([0.8, 0.2], **self.tkwargs),
            torch.tensor([1.0, 1.0], **self.tkwargs),
        ]
        self._perf = [1.0, 3.0, 2.0, 4.0]
        self._corr = [4.0, 2.0, 3.0, 1.0]
        self.requested_random_counts = []

    def _hier_pool_sobol(self, n: int, seed_offset: int = 100_000):
        self.requested_random_counts.append((n, seed_offset))
        values = torch.arange(n * 2, **self.tkwargs).reshape(n, 2)
        return values / max(1, n * 2 - 1)


def test_agent_optimizer_default_search_budget_is_bounded():
    parameters = inspect.signature(AgentGuidedQNEHVI.__init__).parameters

    assert parameters["n_candidates"].default == 256
    assert parameters["dls_num_restarts"].default == 8


def test_grouped_random_budget_is_not_reduced_by_structured_starts():
    harness = _RestartHarness(restart_budget=7)

    many_llm_starts = AgentAdvice(
        centers=torch.linspace(0.0, 1.0, 10, **harness.tkwargs).reshape(5, 2)
    )
    with_advice = harness._restart_seed_groups(many_llm_starts)
    without_advice = harness._restart_seed_groups(None)

    assert with_advice["llm"].shape[0] == 5
    assert with_advice["incumbent"].shape[0] > 0
    assert with_advice["random"].shape[0] == 7
    assert without_advice["random"].shape[0] == 7
    assert harness.requested_random_counts == [
        (7, 300_000),
        (7, 300_000),
    ]


def test_grouped_random_budget_keeps_existing_two_start_floor():
    harness = _RestartHarness(restart_budget=1)

    groups = harness._restart_seed_groups(None)

    assert groups["random"].shape[0] == 2
    assert harness.requested_random_counts == [(2, 300_000)]
