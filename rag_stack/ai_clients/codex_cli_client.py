"""OpenAI Codex CLI client for rag-stack.

Wraps the `codex exec` non-interactive subcommand:
    https://developers.openai.com/codex/cli/reference#codex-exec

Each call spawns a subprocess — fully isolated, no shared conversation state
(mirrors ClaudeCodeClient's isolated mode).

Key CLI flags used:
    -m MODEL                 model selection
    -C DIR                   working directory
    -s SANDBOX               sandbox mode (read-only | workspace-write | danger-full-access)
    --full-auto              convenience for workspace-write, no prompts
    --skip-git-repo-check    allow running outside a git repo
    --ephemeral              don't persist session files
    --output-schema FILE     constrain final response to a JSON Schema
    -o FILE                  write final message to this file
    -c KEY=VALUE             override config (e.g. model_reasoning_effort=low)
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from typing import Any, List, Optional, Type

from pydantic import BaseModel

from rag_stack.ai_clients.base import AIClient

logger = logging.getLogger("RAG-Stack")


def _to_strict_schema(schema: dict) -> dict:
    """Make a pydantic-generated JSON schema OpenAI-strict compatible.

    Codex's ``--output-schema`` requires every object node to declare
    ``additionalProperties: false`` and to list all its keys under
    ``required``. Pydantic's ``model_json_schema()`` sets neither by
    default. Walk the schema tree and patch in place.
    """
    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if "properties" in node:
                    node["required"] = list(node["properties"].keys())
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    import copy
    out = copy.deepcopy(schema)
    _walk(out)
    # $defs subtrees also need the same treatment (walked above already).
    return out


class CodexCliClient(AIClient):
    """OpenAI Codex CLI (`codex exec`) wrapper.

    Design mirrors ClaudeCodeClient's isolated mode — each call is an
    independent subprocess invocation. Codex CLI has no persistent session
    analogue to ClaudeSDKClient for stateful use from a single long-lived
    Python object, so we only expose isolated calls here.

    Notes:
    - Codex CLI does not have a native per-call system-prompt flag. We
      prepend the system prompt into the user prompt (matching how
      ClaudeCodeClient.chat separates them, but here concatenated since
      the CLI accepts a single prompt string).
    - Structured output uses `--output-schema` + `-o <file>` to capture a
      JSON response constrained to the pydantic model's schema.
    - Concurrency is capped with an anyio CapacityLimiter to avoid forking
      too many subprocesses concurrently.
    """

    provider = "codex_cli"

    def __init__(
        self,
        model: str = "gpt-5.4",
        system_prompt: str = "",
        cwd: Optional[str] = None,
        sandbox: str = "workspace-write",
        full_auto: bool = True,
        skip_git_repo_check: bool = True,
        ephemeral: bool = True,
        effort: Optional[str] = "low",
        extra_config: Optional[dict[str, str]] = None,
        codex_bin: Optional[str] = None,
        max_concurrency: int = 20,
        timeout_s: Optional[float] = None,
    ):
        self.model = model
        self._system_prompt = system_prompt
        self._cwd = cwd
        self._sandbox = sandbox
        self._full_auto = full_auto
        self._skip_git_repo_check = skip_git_repo_check
        self._ephemeral = ephemeral
        self._effort = effort
        self._extra_config = dict(extra_config or {})
        self._codex_bin = codex_bin or shutil.which("codex") or "codex"
        self._max_concurrency = max_concurrency
        self._timeout_s = timeout_s
        self._limiter: Optional[Any] = None

    # ------------------------------------------------------------------
    # Subprocess invocation
    # ------------------------------------------------------------------

    def _get_limiter(self):
        """Lazy-init CapacityLimiter (must be created inside async context)."""
        if self._limiter is None:
            import anyio
            self._limiter = anyio.CapacityLimiter(self._max_concurrency)
        return self._limiter

    def _build_argv(
        self,
        output_schema_path: Optional[str],
        output_last_message_path: Optional[str],
    ) -> list[str]:
        """Assemble the `codex exec` command line (prompt is passed on stdin)."""
        argv: list[str] = [self._codex_bin, "exec"]

        if self.model:
            argv += ["-m", self.model]
        if self._cwd:
            argv += ["-C", self._cwd]
        if self._sandbox and not self._full_auto:
            argv += ["-s", self._sandbox]
        if self._full_auto:
            argv += ["--full-auto"]
        if self._skip_git_repo_check:
            argv += ["--skip-git-repo-check"]
        if self._ephemeral:
            argv += ["--ephemeral"]
        if output_schema_path:
            argv += ["--output-schema", output_schema_path]
        if output_last_message_path:
            argv += ["-o", output_last_message_path]

        # Config overrides via -c key=value
        cfg: dict[str, str] = {}
        if self._effort:
            cfg["model_reasoning_effort"] = f'"{self._effort}"'
        cfg.update(self._extra_config)
        for k, v in cfg.items():
            argv += ["-c", f"{k}={v}"]

        # Prompt is read from stdin when we pass "-"
        argv += ["-"]
        return argv

    async def _exec(
        self,
        prompt: str,
        *,
        system_prompt_override: Optional[str] = None,
        output_schema: Optional[dict] = None,
    ) -> str:
        """Run one `codex exec` invocation. Returns the final agent message."""
        sys_prompt = (
            system_prompt_override
            if system_prompt_override is not None
            else self._system_prompt
        )
        full_prompt = (
            f"{sys_prompt}\n\n{prompt}" if sys_prompt else prompt
        )

        # Prepare temp files for schema + last-message output
        tmp_dir = tempfile.mkdtemp(prefix="codex_cli_")
        schema_path: Optional[str] = None
        last_msg_path = os.path.join(tmp_dir, "last_message.txt")
        try:
            if output_schema is not None:
                schema_path = os.path.join(tmp_dir, "schema.json")
                with open(schema_path, "w") as f:
                    json.dump(output_schema, f)

            argv = self._build_argv(schema_path, last_msg_path)

            async with self._get_limiter():
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout_b, stderr_b = await asyncio.wait_for(
                        proc.communicate(input=full_prompt.encode("utf-8")),
                        timeout=self._timeout_s,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.wait()
                    raise RuntimeError(
                        f"codex exec timed out after {self._timeout_s}s"
                    )

                if proc.returncode != 0:
                    stderr_s = stderr_b.decode("utf-8", errors="replace")
                    stdout_s = stdout_b.decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"codex exec failed (exit {proc.returncode})\n"
                        f"stderr:\n{stderr_s}\n"
                        f"stdout:\n{stdout_s[:2000]}"
                    )

            # Prefer the -o file (clean last message), fall back to stdout
            if os.path.exists(last_msg_path):
                with open(last_msg_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    return text
            return stdout_b.decode("utf-8", errors="replace")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # AIClient interface
    # ------------------------------------------------------------------

    async def chat(self, messages: List[dict], **kwargs) -> str:
        """Flatten messages into a single prompt; system → system_prompt_override."""
        system_prompt = ""
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                user_parts.append(content)
        prompt = "\n\n".join(user_parts)
        return await self._exec(prompt, system_prompt_override=system_prompt)

    async def structured_output(
        self,
        messages: List[dict],
        response_format: Type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """Constrain Codex's final response to the pydantic model's JSON schema."""
        system_prompt = ""
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                user_parts.append(content)
        prompt = "\n\n".join(user_parts)

        schema = _to_strict_schema(response_format.model_json_schema())
        response_text = await self._exec(
            prompt,
            system_prompt_override=system_prompt,
            output_schema=schema,
        )

        # Codex may still return the JSON wrapped in prose / fences; try to
        # parse the whole text first, then fall back to extracting a fenced
        # block or a raw {...} substring.
        try:
            data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", response_text, re.DOTALL)
            if m:
                data = json.loads(m.group(1))
            else:
                m2 = re.search(r"\{.*\}", response_text, re.DOTALL)
                if not m2:
                    raise ValueError(
                        f"Could not parse JSON from Codex response: "
                        f"{response_text[:200]}"
                    )
                data = json.loads(m2.group(0))

        return response_format.model_validate(data)

    def __repr__(self):
        return f"CodexCliClient(model='{self.model}', sandbox='{self._sandbox}')"
