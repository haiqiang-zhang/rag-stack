"""Dataset-generation output and registry contracts without LLM calls."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.dataset_generator import GeneratedDataset, create_dataset_generator_from_yaml
from rag_stack.dataset_generator import registry
from rag_stack.dataset_generator.schema import Corpus, QA, Raw
from rag_stack.utils.preprocess import load_data_from_parquet


@pytest.fixture
def dataset():
    corpus = pd.DataFrame({
        "doc_id": ["d0", "d1"],
        "contents": ["Passage zero.", "Passage one."],
        "metadata": [{"last_modified_datetime": "2025-01-01"}] * 2,
    })
    qa = pd.DataFrame({
        "qid": ["q0", "q1"],
        "query": ["What is zero?", "What is one?"],
        "retrieval_gt": [[["d0"]], [["d1"]]],
        "generation_gt": [["zero"], ["one"]],
        "references": [["Passage zero."], ["Passage one."]],
        "question_type": ["fact", "fact"],
    })
    return GeneratedDataset(
        raw=Raw(), corpus=Corpus(corpus), qa=QA(qa), strategy_name="fixture",
    )


@pytest.mark.parametrize("part,column", [("corpus", "doc_id"), ("qa", "generation_gt")])
def test_generated_dataset_rejects_missing_required_columns(dataset, part, column):
    dataset.validate()
    frame = getattr(dataset, part)
    frame.data = frame.data.drop(columns=[column])

    with pytest.raises(AssertionError, match=f"{part} missing columns"):
        dataset.validate()


def test_generated_dataset_normalizes_for_evaluation_without_mutating_source(dataset):
    original_qa = deepcopy(dataset.qa.data)
    original_corpus = deepcopy(dataset.corpus.data)

    qa, corpus = dataset.to_dataframes()

    assert qa["qid"].tolist() == ["q0", "q1"]
    assert corpus["doc_id"].tolist() == ["d0", "d1"]
    assert qa["retrieval_gt_contents"].tolist() == [[["Passage zero."]], [["Passage one."]]]
    assert "retrieval_gt" not in qa
    pd.testing.assert_frame_equal(dataset.qa.data, original_qa)
    pd.testing.assert_frame_equal(dataset.corpus.data, original_corpus)


def test_generated_dataset_parquet_preserves_metadata_and_evaluator_input(dataset, tmp_path):
    qa_path, corpus_path = tmp_path / "qa.parquet", tmp_path / "corpus.parquet"

    dataset.to_parquet(str(qa_path), str(corpus_path))
    qa, corpus = load_data_from_parquet(str(qa_path), str(corpus_path))

    assert qa["question_type"].tolist() == ["fact", "fact"]
    assert qa["references"].tolist() == [["Passage zero."], ["Passage one."]]
    assert qa["generation_gt"].tolist() == [["zero"], ["one"]]
    assert corpus["contents"].tolist() == ["Passage zero.", "Passage one."]


def test_unknown_generator_strategy_is_rejected():
    with pytest.raises(KeyError, match="Unknown dataset generation strategy"):
        registry.get_dataset_generator_class("missing-strategy")


def test_registered_strategy_is_resolved_and_receives_yaml_parameters(monkeypatch, tmp_path):
    class Generator:
        @classmethod
        def from_config(cls, config):
            return cls(), config

    # Restore the registry after this test so no plugin registration leaks.
    monkeypatch.setattr(registry, "_STRATEGY_REGISTRY", dict(registry._STRATEGY_REGISTRY))
    original_import = registry.importlib.import_module

    def import_module(name):
        return SimpleNamespace(Generator=Generator) if name == "fixture.module" else original_import(name)

    monkeypatch.setattr(registry.importlib, "import_module", import_module)
    registry.register_strategy("fixture", "fixture.module", "Generator")
    config_path = tmp_path / "generator.yaml"
    config_path.write_text("strategy: fixture\nrandom_seed: 42\ndomain: medical\n")

    generator, config = create_dataset_generator_from_yaml(str(config_path))

    assert isinstance(generator, Generator)
    assert config == {"random_seed": 42, "domain": "medical"}
