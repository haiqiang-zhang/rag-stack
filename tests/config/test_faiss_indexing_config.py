"""Host validation of the evaluator indexing-thread option."""

from copy import deepcopy
from pathlib import Path

import pytest
import rag_stack
import yaml

from rag_stack.config_validator import ConfigValidator


@pytest.mark.parametrize("threads, valid", [(24, True), (0, False)])
def test_faiss_indexing_thread_config(threads, valid):
    config_path = (Path(rag_stack.__file__).resolve().parent.parent / "configs" / "rag_stack" / "paper_bench_greedy_forward_s43.yaml")
    config = deepcopy(yaml.safe_load(config_path.read_text()))
    config.setdefault("system", {}).setdefault("retrieval", {})["faiss_indexing_thread"] = threads
    errors = ConfigValidator(config, check_model_cache=False).validate()
    assert (not any("faiss_indexing_thread" in error for error in errors)) is valid
