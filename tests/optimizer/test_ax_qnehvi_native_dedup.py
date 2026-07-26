from collections.abc import Iterator

import pytest
from ax.core import Arm, ChoiceParameter, GeneratorRun, ParameterType, SearchSpace

from rag_stack.optimizer.botorch.ax_qnehvi import AxQNEHVI
from rag_stack.optimizer.botorch.ax_qnehvi_mixedgp import AxQNEHVIMixedGP


def _optimizer(optimizer_class: type[AxQNEHVI]) -> AxQNEHVI:
    search_space = SearchSpace(
        parameters=[
            ChoiceParameter(
                name="x",
                parameter_type=ParameterType.INT,
                values=[0, 1],
                is_ordered=False,
            )
        ]
    )
    return optimizer_class(
        config_space=search_space,
        objectives=[
            {"objective_name": "quality", "ref_point": 0.0},
            {"objective_name": "performance", "ref_point": 0.0},
        ],
        seed=43,
        n_both_init=2,
        use_gpu=False,
    )


@pytest.mark.parametrize("optimizer_class", [AxQNEHVI, AxQNEHVIMixedGP])
def test_all_ax_qnehvi_generation_steps_use_native_deduplication(
    optimizer_class: type[AxQNEHVI],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = _optimizer(optimizer_class)
    strategy = optimizer._client.generation_strategy

    # Both optimizer families use the same builder and must deduplicate in
    # both phases: the initial Sobol design and the model-based BO search.
    assert len(strategy._nodes) == 2
    assert [node.should_deduplicate for node in strategy._nodes] == [True, True]

    # Seed the experiment with x=0, then make each underlying generator emit
    # that duplicate before a feasible unseen arm. Ax's native GenerationNode
    # loop must discard x=0 and return x=1 for both the Sobol and BO nodes.
    optimizer._client.attach_trial(parameters={"x": 0})
    for node in strategy._nodes:
        generated: Iterator[GeneratorRun] = iter(
            [
                GeneratorRun(arms=[Arm(parameters={"x": 0})]),
                GeneratorRun(arms=[Arm(parameters={"x": 1})]),
            ]
        )
        calls = 0

        def fake_generate(**_: object) -> GeneratorRun:
            nonlocal calls
            calls += 1
            return next(generated)

        monkeypatch.setattr(node, "_gen", fake_generate)
        result = node._gen_maybe_deduplicate(
            experiment=optimizer._client.experiment,
            pending_observations=None,
            data=None,
        )

        assert calls == 2
        assert [arm.parameters for arm in result.arms] == [{"x": 1}]
