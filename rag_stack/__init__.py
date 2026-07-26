import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

# Load .env using the path from .rag_stack_settings.json (env_file_path),
# falling back to project root .env.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SETTINGS_FILE = Path(os.environ["RAG_STACK_SETTINGS_PATH"]) if os.environ.get("RAG_STACK_SETTINGS_PATH") else _PROJECT_ROOT / ".rag_stack_settings.json"

def _resolve_env_file() -> Path:
    if _SETTINGS_FILE.is_file():
        try:
            custom = json.loads(_SETTINGS_FILE.read_text()).get("env_file_path", "").strip()
            if custom:
                p = Path(custom)
                return p if p.is_absolute() else _PROJECT_ROOT / p
        except (json.JSONDecodeError, OSError):
            pass
    return _PROJECT_ROOT / ".env"

# override=True: the .env file is the SINGLE SOURCE OF TRUTH for credentials.
# Without it, a stale export inherited from the launching shell's profile
# silently wins over a freshly updated .env (dotenv's default) — a rotated
# judge key in .env kept 402-ing because the old key rode in via os.environ.
# Runtime routing and concurrency are different: the multi-server launcher
# must be able to select the current judge host without rewriting a shared
# repository .env. Preserve those explicit process-level values while keeping
# credential precedence unchanged.
_RUNTIME_ENV_KEYS = (
    "RAG_LLM_BASE_URL",
    "JUDGE_BASE_URL",
    "JUDGE_MODEL",
    "JUDGE_MAX_CONCURRENCY",
    "JUDGE_DEEPEVAL_MAX_CONCURRENT",
    # Exact-computation scheduling controls selected per system-transfer run.
    # The API stamps audited caps into the child environment; a shared .env
    # must not silently replace them while importing rag_stack in that child.
    "CM_SWEEP_WORKERS",
    "CM_DES_WORKERS",
)
_runtime_env = {
    key: os.environ[key] for key in _RUNTIME_ENV_KEYS if key in os.environ
}
load_dotenv(_resolve_env_file(), override=True)
os.environ.update(_runtime_env)

# Host-tool caches default into the repository's .cache/ instead of $HOME.
# vLLM's torch.compile artifacts alone run to GBs per (model, shape) and
# have blown the quota'd NFS home twice; every one of these is a
# rebuildable cache that belongs on the big data disk with the repo.
# setdefault only — a shell export or .env line always wins (dotenv loads
# above, so a .env override lands before these defaults are consulted).
# HF_HOME holds model WEIGHTS: on a box with several worktrees export it
# to one shared path instead (see docs/user_guide.md "Cache directories")
# so each worktree does not duplicate hundreds of GB of checkpoints.
_CACHE_ROOT = _PROJECT_ROOT / ".cache"
for _var, _path in (
    # Preserve rag_stack_evaluator's historical <main-repo>/.cache/rag_stack
    # default now that the package lives in the submodule.
    ("RAG_STACK_CACHE_DIR", _CACHE_ROOT / "rag_stack"),
    ("VLLM_CACHE_ROOT", _CACHE_ROOT / "vllm"),        # torch.compile + engine caches (~/.cache/vllm)
    ("TRITON_CACHE_DIR", _CACHE_ROOT / "triton"),     # triton JIT kernels (~/.triton)
    ("FLASHINFER_WORKSPACE_BASE", _PROJECT_ROOT),     # flashinfer appends .cache/flashinfer itself
    ("LLAMA_INDEX_CACHE_DIR", _CACHE_ROOT / "llama_index"),
    ("HF_HOME", _CACHE_ROOT / "huggingface"),         # hub weights + tokenizers
):
    os.environ.setdefault(_var, str(_path))

# Force wider console when stderr isn't a TTY (nohup / redirect to file).
# rich.Console defaults to 80 columns without a terminal, which makes
# detached benchmark logs nearly unreadable. 240 cols matches a typical
# wide editor and won't wrap interactively (Console only widens, never
# squeezes below the actual terminal in TTY mode).
_log_console = Console(width=240) if not sys.stderr.isatty() else None
logging.basicConfig(
    level=logging.INFO,
    format="[%(filename)s:%(lineno)s] >> %(message)s",
    handlers=[RichHandler(console=_log_console, rich_tracebacks=True)],
)

# Explicitly set RAG-Stack logger level so SMAC can't swallow it
# by overriding root logger's level.
logging.getLogger("RAG-Stack").setLevel(logging.INFO)

# Force DeepEval's PostHog telemetry OFF before any deepeval import. Its
# telemetry client intermittently raises "Counters can only be incremented by
# non-negative amounts" from INSIDE metric.a_measure(), which crashes EVERY
# judge case at once — the all-cases-errored guard then correctly marks the
# whole trial INVALID, spuriously wasting ~1 measured trial per 50 on a
# library bug (not a real judge outage). Opt out unconditionally (respect an
# explicit user override if already set). deepeval reads this at its import
# time, which always follows this module's import.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
