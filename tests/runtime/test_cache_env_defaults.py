"""Host-tool cache env defaults: repo .cache/, setdefault-only.

vLLM compile artifacts (GBs per model/shape) in the quota'd NFS $HOME blew
the home directory twice; rag_stack's import must point every rebuildable
tool cache at <repo>/.cache instead — without ever overriding an explicit
user environment."""

import os
import subprocess
import sys

_VARS = (
    "VLLM_CACHE_ROOT",
    "TRITON_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "LLAMA_INDEX_CACHE_DIR",
    "HF_HOME",
)


def _probe(extra_env):
    env = {k: v for k, v in os.environ.items() if k not in _VARS}
    env.update(extra_env)
    code = (
        "import os, rag_stack, json;"
        f"print(json.dumps({{k: os.environ.get(k) for k in {_VARS!r}}}))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env,
        capture_output=True, text=True, check=True,
    )
    import json
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_defaults_land_in_repo_cache():
    got = _probe({})
    import rag_stack

    repo = os.path.dirname(os.path.dirname(os.path.abspath(rag_stack.__file__)))
    assert got["VLLM_CACHE_ROOT"] == os.path.join(repo, ".cache", "vllm")
    assert got["TRITON_CACHE_DIR"] == os.path.join(repo, ".cache", "triton")
    assert got["FLASHINFER_WORKSPACE_BASE"] == repo
    assert got["LLAMA_INDEX_CACHE_DIR"] == os.path.join(repo, ".cache", "llama_index")
    assert got["HF_HOME"] == os.path.join(repo, ".cache", "huggingface")
    for v in got.values():
        assert os.path.expanduser("~") not in v


def test_user_environment_always_wins():
    got = _probe({"VLLM_CACHE_ROOT": "/data/custom/vllm",
                  "HF_HOME": "/data/shared/hf"})
    assert got["VLLM_CACHE_ROOT"] == "/data/custom/vllm"
    assert got["HF_HOME"] == "/data/shared/hf"
