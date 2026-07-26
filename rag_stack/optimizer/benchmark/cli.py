"""Optimizer benchmark CLI.

Two supported YAML shapes, auto-detected by ``load_config``:

1. **Legacy single-run** — one ``optimizer`` + one ``synthetic_function`` block::

    optimizer: {type: ..., params: {...}}
    synthetic_function: {type: ..., params: {...}}
    gt_budget: 50
    seed: 42

2. **Matrix sweep** — cartesian product ``optimizers × synthetic_functions × seeds``::

    defaults:
      gt_budget: 50
      seeds: [42]
      auto_analyze: true
    optimizers:
      - type: smac_decoupled_parego
        params: {...}
      - type: ax_qnehvi
        gt_budget: 30   # per-optimizer override
    synthetic_functions:
      - type: dtlz2_metrics
    skip:
      - optimizer: ...
        synthetic_function: ...

Invocation::

    python -m rag_stack benchmark --config benchmark_config.yaml
    python -m rag_stack.optimizer.benchmark --config benchmark_config.yaml

Both forms work — the first goes through the top-level ``rag_stack/cli.py``
dispatcher; the second through ``benchmark/__main__.py``.
"""

import argparse
import itertools
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from rag_stack.optimizer import OPTIMIZER_REGISTRY
from rag_stack.optimizer.benchmark.build_pareto_cache import DEFAULT_OUTPUT_DIR
from rag_stack.optimizer.benchmark.runner import BenchmarkRunner
from rag_stack.optimizer.benchmark.synthetic_functions import SYNTHETIC_FUNCTION_REGISTRY


# ---------------------------------------------------------------------------
# Mode detection + validation
# ---------------------------------------------------------------------------

# Keys recognized at optimizer-entry level for per-optimizer overrides of the
# shared ``defaults`` block. Everything else on an optimizer entry is treated
# as an error to avoid silent typos.
_MATRIX_OPT_RESERVED = {"type", "name", "params"}
_MATRIX_OVERRIDABLE = {"gt_budget", "checkpoint_interval", "output_dir"}

# Allowed characters in a benchmark `name` (drives subfolder layout, so we
# disallow anything that breaks filesystems / globs / shell pipes).
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Single-file manifest inside each per-benchmark subfolder. Holds:
#   synthetic_function: the benchmark config (type + name + params)
#   optimizers: list of every optimizer experiment that's been run
# against this benchmark, identified by `name` (also unique within the
# subfolder = experiment ID).
_EXPERIMENTS_FILENAME = "experiments.yaml"


def _validate_benchmark_name(name: str, ctx: str) -> None:
    """Reject anything that isn't ``[a-zA-Z0-9_-]+``. The benchmark `name`
    becomes a literal subfolder name, so spaces / slashes / dots / glob
    metacharacters are flat-out unsafe."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            f"{ctx}: invalid benchmark name {name!r}. "
            f"Only [a-zA-Z0-9_-]+ characters allowed."
        )


def _validate_optimizer_name(name: str, ctx: str) -> None:
    """Same character set as benchmark names — optimizer ``name`` is a
    literal filename component (``<name>.yaml`` snapshot, plus
    ``<name>__<benchmark>.csv``)."""
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            f"{ctx}: invalid optimizer name {name!r}. "
            f"Only [a-zA-Z0-9_-]+ characters allowed."
        )


def benchmark_subfolder(output_dir: str, name: str) -> Path:
    """Resolve the per-benchmark subfolder under the shared output_dir.
    Each benchmark variant (= unique ``synthetic_functions[].name``) gets
    its own folder so its CSVs, eval logs, Pareto cache, and config
    snapshot live together and never cross-contaminate other variants.
    """
    return Path(output_dir) / name


def experiments_path(output_dir: str, name: str) -> Path:
    """Resolve the per-benchmark ``experiments.yaml`` location."""
    return benchmark_subfolder(output_dir, name) / _EXPERIMENTS_FILENAME


def ensure_experiments_yaml(
    output_dir: str,
    fn_entry: Dict[str, Any],
    opt_entry: Dict[str, Any],
) -> Path:
    """Idempotently write/extend the per-benchmark ``experiments.yaml``.

    The file is the single source of truth for *what was run* in a
    subfolder. It holds:

      ``synthetic_function``: the benchmark config (constant per
      subfolder; deep-equal verified on every call).

      ``optimizers``: list of every optimizer experiment, looked up by
      ``name`` (= experiment ID within this benchmark). New optimizer
      entries get appended; existing ones are deep-equal verified.

    Mismatch on either block raises ``SystemExit`` — that's the
    failure mode this layout is designed to surface (two YAMLs claiming
    the same ``name`` for different configs would silently overwrite
    each other's results without it).

    Returns the subfolder path so callers can drop CSVs into it.
    """
    name = fn_entry["name"]
    sub = benchmark_subfolder(output_dir, name)
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / _EXPERIMENTS_FILENAME

    if path.is_file():
        with open(path) as f:
            doc = yaml.safe_load(f) or {}
        # 1. Benchmark config must match.
        stored_fn = doc.get("synthetic_function") or {}
        if stored_fn != fn_entry:
            raise SystemExit(
                f"Benchmark name {name!r} already has a stored config in\n"
                f"  {path}\n"
                f"that differs from the current run's synthetic_function entry.\n"
                f"  stored: {stored_fn!r}\n"
                f"  new:    {fn_entry!r}\n"
                f"Pick a different `name` for this benchmark, or align the configs."
            )
        # 2. Optimizer config: append-or-verify by `name`.
        opts = doc.get("optimizers") or []
        if not isinstance(opts, list):
            raise SystemExit(
                f"{path}: 'optimizers' is malformed (expected a list, got "
                f"{type(opts).__name__})"
            )
        existing = next(
            (o for o in opts if isinstance(o, dict) and o.get("name") == opt_entry["name"]),
            None,
        )
        if existing is not None:
            if existing != opt_entry:
                # Param drift on the same `name` happens in practice when
                # extending an existing sweep with new seeds + tweaked
                # acqf settings (e.g. lowering qNIPV restarts to fit
                # memory). The CSV still appends cleanly because seed +
                # iteration are the row keys. Emit a warning instead of
                # blocking — the per-run snapshot in the CSV is already
                # the authoritative record of what each row was run with.
                import warnings
                warnings.warn(
                    f"Optimizer name {opt_entry['name']!r} stored config in "
                    f"{path} differs from this run's entry.\n"
                    f"  stored: {existing!r}\n"
                    f"  new:    {opt_entry!r}\n"
                    f"Continuing — different seed batches may have used "
                    f"different params. Use distinct `name` if you need "
                    f"strict per-CSV consistency."
                )
                # Update the manifest to the latest config so the file
                # always reflects what the most recent batch was run with.
                idx = next(
                    i for i, o in enumerate(opts)
                    if isinstance(o, dict) and o.get("name") == opt_entry["name"]
                )
                opts[idx] = opt_entry
                doc["optimizers"] = opts
                with open(path, "w") as f:
                    yaml.safe_dump(doc, f, sort_keys=False)
            # else: already present and matching — no write needed.
        else:
            opts.append(opt_entry)
            doc["optimizers"] = opts
            with open(path, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False)
    else:
        # First write — initialize the manifest.
        doc = {"synthetic_function": fn_entry, "optimizers": [opt_entry]}
        with open(path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)

    return sub


def _is_matrix_config(config: dict) -> bool:
    """True if YAML uses the matrix-sweep schema (plural keys)."""
    return isinstance(config.get("optimizers"), list)


def load_config(path: str) -> dict:
    """Load and validate a YAML benchmark config (either schema)."""
    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    if _is_matrix_config(config):
        _validate_matrix_config(config, path)
    else:
        _validate_legacy_config(config, path)

    return config


def _validate_legacy_config(config: dict, path: str) -> None:
    for key in ("optimizer", "synthetic_function"):
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in {path}")

    opt = config["optimizer"]
    if not isinstance(opt, dict):
        raise ValueError(f"{path}: 'optimizer' must be a mapping")
    if "type" not in opt:
        raise ValueError(f"{path}: 'optimizer' missing required 'type'")
    if "name" not in opt:
        raise ValueError(
            f"{path}: 'optimizer' missing required 'name' "
            f"(used as the per-experiment YAML/CSV identifier inside the "
            f"benchmark subfolder)"
        )
    _validate_optimizer_name(opt["name"], f"{path}: optimizer.name")
    if opt["type"] not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{opt['type']}'. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    sf_config = config["synthetic_function"]
    if not isinstance(sf_config, dict):
        raise ValueError(
            f"{path}: 'synthetic_function' must be a mapping with 'type' and 'name' keys"
        )
    if "type" not in sf_config:
        raise ValueError(f"{path}: 'synthetic_function' missing required 'type'")
    if "name" not in sf_config:
        raise ValueError(
            f"{path}: 'synthetic_function' missing required 'name' "
            f"(used as the per-benchmark subfolder under output_dir)"
        )
    _validate_benchmark_name(sf_config["name"], f"{path}: synthetic_function.name")
    if sf_config["type"] not in SYNTHETIC_FUNCTION_REGISTRY:
        raise ValueError(
            f"{path}: unknown synthetic_function.type {sf_config['type']!r}. "
            f"Available: {list(SYNTHETIC_FUNCTION_REGISTRY.keys())}"
        )


def _validate_matrix_config(config: dict, path: str) -> None:
    optimizers = config.get("optimizers")
    functions = config.get("synthetic_functions")
    if not optimizers:
        raise ValueError(f"{path}: 'optimizers' must be a non-empty list")
    if not isinstance(functions, list) or not functions:
        raise ValueError(f"{path}: 'synthetic_functions' must be a non-empty list")

    defaults = config.get("defaults", {}) or {}
    seeds = defaults.get("seeds", [42])
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path}: 'defaults.seeds' must be a non-empty list")

    seen_names = set()
    for i, entry in enumerate(optimizers):
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(f"{path}: optimizers[{i}] needs a 'type' field")
        if "name" not in entry:
            raise ValueError(
                f"{path}: optimizers[{i}] missing required 'name' "
                f"(per-experiment ID inside the benchmark subfolder; used as "
                f"the optimizer-config YAML filename and the CSV filename prefix)"
            )
        _validate_optimizer_name(entry["name"], f"{path}: optimizers[{i}].name")
        if entry["type"] not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"{path}: optimizers[{i}] unknown type '{entry['type']}'. "
                f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
            )
        stray = set(entry.keys()) - _MATRIX_OPT_RESERVED - _MATRIX_OVERRIDABLE
        if stray:
            raise ValueError(
                f"{path}: optimizers[{i}] has unrecognized keys {sorted(stray)}. "
                f"Allowed: type, name, params, or defaults overrides "
                f"{sorted(_MATRIX_OVERRIDABLE)}"
            )
        name = entry["name"]
        if name in seen_names:
            raise ValueError(
                f"{path}: duplicate optimizer name '{name}'. Use the 'name' field "
                f"to disambiguate entries that share the same 'type'."
            )
        seen_names.add(name)

    seen_fn_names: set[str] = set()
    for i, entry in enumerate(functions):
        if not isinstance(entry, dict) or "type" not in entry:
            raise ValueError(f"{path}: synthetic_functions[{i}] needs a 'type' field")
        if "name" not in entry:
            raise ValueError(
                f"{path}: synthetic_functions[{i}] missing required 'name' "
                f"(used as per-benchmark subfolder name; must be unique)"
            )
        _validate_benchmark_name(
            entry["name"], f"{path}: synthetic_functions[{i}].name"
        )
        if entry["name"] in seen_fn_names:
            raise ValueError(
                f"{path}: duplicate synthetic_functions name {entry['name']!r}. "
                f"Within one config every benchmark name must be unique."
            )
        seen_fn_names.add(entry["name"])
        if entry["type"] not in SYNTHETIC_FUNCTION_REGISTRY:
            raise ValueError(
                f"{path}: synthetic_functions[{i}] unknown type {entry['type']!r}. "
                f"Available: {list(SYNTHETIC_FUNCTION_REGISTRY.keys())}"
            )

    for i, entry in enumerate(config.get("skip", []) or []):
        if not isinstance(entry, dict) or "optimizer" not in entry or "synthetic_function" not in entry:
            raise ValueError(
                f"{path}: skip[{i}] must be {{optimizer: ..., synthetic_function: ...}}"
            )


# ---------------------------------------------------------------------------
# Optimizer instantiation
# ---------------------------------------------------------------------------

def create_optimizer(config: dict, synthetic_function):
    """Instantiate an optimizer from YAML config."""
    opt_type = config["optimizer"]["type"]
    opt_params = dict(config["optimizer"].get("params", {}))  # copy
    seed = config.get("seed", 42)
    cls = OPTIMIZER_REGISTRY[opt_type]

    # Hand the optimizer the synthetic function's OWN ssops when it exposes one
    # (e.g. rag_stack_historical_surrogate's RagSearchSpace) AND the optimizer's
    # constructor accepts an `ssops` kwarg. Otherwise the optimizer builds a fresh
    # AxSearchSpaceOps whose decode_to_config produces configs the function's
    # evaluate() can't resolve (sweep params like top_k=[1,2,4,8,16] stay
    # unresolved). Skipped for optimizers/functions that don't expose/accept it.
    import inspect
    if hasattr(synthetic_function, "ssops") and "ssops" not in opt_params:
        try:
            if "ssops" in inspect.signature(cls.__init__).parameters:
                opt_params["ssops"] = synthetic_function.ssops
        except (ValueError, TypeError):
            pass

    # Mirror the runner's budget only for constructors that explicitly consume
    # it. Silent **kwargs sinks must not make dead controls appear effective.
    try:
        accepts_budget = (
            "n_iterations" in inspect.signature(cls.__init__).parameters
        )
    except (ValueError, TypeError):
        accepts_budget = False
    if accepts_budget:
        opt_params.setdefault("n_iterations", int(config.get("gt_budget", 80)))

    opt = cls(
        config_space=synthetic_function.search_space,
        objectives=synthetic_function.objectives,
        seed=seed,
        **opt_params,
    )
    _warn_unconsumed_params(cls, opt, set(config["optimizer"].get("params", {})))
    return opt


def _known_ctor_params(*classes) -> set:
    """Every parameter name an optimizer family can consume: explicit
    ``__init__`` signature names across each class's MRO PLUS names read
    dynamically via ``kwargs.get/pop("name")`` in those ``__init__`` bodies
    (the dominant pattern in the agent optimizers — signature inspection
    alone cannot see them)."""
    import inspect
    known: set = set()
    seen: set = set()
    kw_re = re.compile(r"""kwargs\.(?:get|pop)\(\s*['"](\w+)['"]""")
    for cls in classes:
        if cls is None:
            continue
        for klass in getattr(cls, "__mro__", [cls]):
            if klass in seen or klass is object:
                continue
            seen.add(klass)
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            try:
                known.update(inspect.signature(init).parameters)
            except (TypeError, ValueError):
                pass
            try:
                known.update(kw_re.findall(inspect.getsource(init)))
            except (OSError, TypeError):
                pass
    known.discard("self")
    return known


def _warn_unconsumed_params(cls, opt, requested: set) -> None:
    """Warn when no optimizer-family constructor consumes a requested param.

    A stale or misspelled key swallowed by ``**kwargs`` makes an A/B compare
    identical mechanisms. Keep this non-fatal because some dynamic consumers
    cannot be recovered through signature/source inspection.
    """
    sub_classes = [
        type(getattr(opt, attr)) for attr in ("greedy", "agent")
        if getattr(opt, attr, None) is not None
    ]
    nested_agent = getattr(getattr(opt, "agent", None), "agent", None)
    if nested_agent is not None:
        sub_classes.append(type(nested_agent))
    known = _known_ctor_params(cls, *sub_classes)
    unknown = sorted(requested - known)
    if unknown:
        import logging
        logging.getLogger("RAG-Stack").warning(
            "benchmark: optimizer params %s matched NO constructor parameter "
            "or kwargs.get() consumer across %s — if these are meant to be "
            "mechanism flags, verify the mechanism exists in this checkout "
            "(a stale/typo'd flag silently A/Bs two identical optimizers).",
            unknown, [c.__name__ for c in {cls, *sub_classes}],
        )
        print(f"[benchmark] WARNING: unrecognized optimizer params {unknown} "
              "— possible silent no-op flag", flush=True)


def _append_run_provenance(subfolder, opt_name: str, seed) -> None:
    """One line per launched run in <subfolder>/runs.log: timestamp, run id,
    and the code commit — so 'the flag predates the mechanism' timeline
    errors are detectable after the fact. Sidecar file on purpose:
    experiments.yaml entries are deep-equal-verified across re-runs and must
    stay commit-independent."""
    import subprocess
    import time as _time
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(subfolder), timeout=5,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    try:
        with open(Path(subfolder) / "runs.log", "a") as fh:
            fh.write(f"{_time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"optimizer={opt_name} seed={seed} commit={commit}\n")
    except OSError:
        pass


def run_single(
    config: dict,
    fn_entry: Dict[str, Any],
    display_name: Optional[str] = None,
) -> None:
    """Run one (optimizer, synthetic_function) benchmark.

    Each optimizer owns its phase scheduling internally — the runner just
    iterates ``suggest()`` / ``add_objectives()`` until ``cumulative_gt_calls``
    hits ``gt_budget``.

    ``fn_entry`` is the YAML's ``synthetic_function`` (legacy) or
    ``synthetic_functions[i]`` (matrix) block — contains ``type`` (registry
    key), ``name`` (benchmark identity / subfolder), and optional ``params``.
    """
    opt_entry = config["optimizer"]
    # display_name is passed by the matrix-mode caller; in the legacy
    # single-run path we just read from the entry's `name` field
    # (validated as required upstream).
    opt_name = display_name or opt_entry["name"]
    seed = config.get("seed", 42)
    gt_budget = config.get("gt_budget", 80)
    checkpoint_interval = config.get("checkpoint_interval", 10)
    output_dir = config.get("output_dir", str(DEFAULT_OUTPUT_DIR))

    fn_type = fn_entry["type"]
    fn_name = fn_entry["name"]
    func_params = dict(fn_entry.get("params") or {})
    func = SYNTHETIC_FUNCTION_REGISTRY[fn_type](**func_params)
    optimizer = create_optimizer(config, func)

    print(f"\n{'=' * 60}")
    print(f"  {opt_name} on {fn_name} | seed={seed} gt_budget={gt_budget}")
    print(f"{'=' * 60}")

    # Single-file experiments manifest: writes/extends
    # `<subfolder>/experiments.yaml` to record the benchmark config AND
    # this optimizer's config. Two consistency checks fire:
    #   - the synthetic_function block must match what's stored (or this
    #     is the first run for this `name`)
    #   - any pre-existing optimizer entry with the same `name` must
    #     deep-equal the new one — `name` is the per-experiment ID, so
    #     two distinct configs sharing it gets caught immediately.
    subfolder = ensure_experiments_yaml(output_dir, fn_entry, opt_entry)
    _append_run_provenance(subfolder, opt_name, seed)

    runner = BenchmarkRunner(
        function=func,
        optimizer=optimizer,
        gt_budget=gt_budget,
        checkpoint_interval=checkpoint_interval,
        cache_dir=subfolder,
    )

    snapshots = runner.run()

    # CSV layout: <output_dir>/<benchmark_name>/<optimizer_name>__<benchmark_name>.csv
    # benchmark_name is duplicated in filename so a single CSV (e.g. shared
    # in a paper or attached to an issue) is self-describing without folder
    # context.
    csv_path = subfolder / f"{opt_name}__{fn_name}.csv"
    runner.save_csv(snapshots, str(csv_path), seed=seed)
    runner.save_eval_history(str(csv_path), seed=seed)


# ---------------------------------------------------------------------------
# Matrix expansion + execution
# ---------------------------------------------------------------------------

def _expand_matrix(config: dict) -> List[Dict[str, Any]]:
    """Expand a matrix-mode config into a list of legacy-shape per-run dicts.

    Each per-run dict carries the full ``synthetic_function`` block
    (type + name + params) so ``run_single`` has everything it needs to
    write the per-benchmark snapshot and resolve the subfolder path.
    The ``skip`` list still matches on ``(optimizer_display_name,
    synthetic_function_name)`` — name is the canonical benchmark
    identity, not type.
    """
    defaults = config.get("defaults", {}) or {}
    seeds = defaults.get("seeds", [42])
    shared = {k: v for k, v in defaults.items() if k != "seeds" and k != "auto_analyze"}

    skip_set = {
        (entry["optimizer"], entry["synthetic_function"])
        for entry in (config.get("skip") or [])
    }

    runs: List[Dict[str, Any]] = []
    for opt_entry, fn_entry, seed in itertools.product(
        config["optimizers"], config["synthetic_functions"], seeds
    ):
        display_name = opt_entry.get("name", opt_entry["type"])
        fn_name = fn_entry["name"]

        if (display_name, fn_name) in skip_set:
            continue

        # Merge: defaults → optimizer-level overrides (stray keys already
        # rejected by _validate_matrix_config).
        merged = dict(shared)
        for k in _MATRIX_OVERRIDABLE:
            if k in opt_entry:
                merged[k] = opt_entry[k]

        # Carry the FULL optimizer entry through (incl. name + any
        # per-optimizer overrides) so run_single can write a faithful
        # experiment-config snapshot that captures everything affecting
        # this run.
        opt_snapshot = {
            "type": opt_entry["type"],
            "name": display_name,
            "params": dict(opt_entry.get("params") or {}),
        }
        for k in _MATRIX_OVERRIDABLE:
            if k in opt_entry:
                opt_snapshot[k] = opt_entry[k]
        merged["optimizer"] = opt_snapshot
        # Carry the FULL fn entry through (incl. name) so run_single can
        # write the snapshot + use the subfolder layout.
        merged["synthetic_function"] = {
            "type": fn_entry["type"],
            "name": fn_name,
            "params": dict(fn_entry.get("params") or {}),
        }
        merged["seed"] = seed
        merged["_display_name"] = display_name
        runs.append(merged)

    return runs


def _run_matrix(config: dict) -> None:
    """Execute a matrix-sweep config: iterate all runs, analyze at the end."""
    runs = _expand_matrix(config)
    n = len(runs)
    if n == 0:
        print("Matrix expansion produced 0 runs (everything skipped?). Nothing to do.")
        return

    print(f"\nMatrix sweep: {n} run(s) total.")

    succeeded: List[str] = []
    failed: List[tuple[str, str]] = []

    for i, per_run in enumerate(runs, start=1):
        display_name = per_run.pop("_display_name")
        fn_entry = per_run["synthetic_function"]
        label = f"{display_name} on {fn_entry['name']} | seed={per_run['seed']}"

        print(f"\n{'=' * 64}")
        print(f"  [run {i}/{n}] {label}")
        print(f"{'=' * 64}", flush=True)

        try:
            run_single(per_run, fn_entry, display_name=display_name)
            succeeded.append(label)
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[run {i}/{n}] FAILED: {exc!r}\n{tb}", flush=True)
            failed.append((label, repr(exc)))
        finally:
            # Cross-run GPU cleanup: each run's optimizer holds a GP /
            # MTGP / qLogNEHVI graph that pins ~10-25 GB of CUDA memory
            # (autograd buffers, kernel state, MC sample tensors). Without
            # explicit cleanup the next run starts with a half-full GPU
            # and seed-2/3/4 hit OOM cascades. gc.collect() drops the
            # python-side refs; empty_cache() returns the freed CUDA
            # blocks to the driver.
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass

    print(f"\n{'=' * 64}")
    print(f"  Sweep complete: {len(succeeded)}/{n} succeeded, {len(failed)} failed.")
    print(f"{'=' * 64}")
    for label, err in failed:
        print(f"  FAILED: {label}  →  {err}")

    defaults = config.get("defaults", {}) or {}
    if defaults.get("auto_analyze", True):
        output_dir = defaults.get("output_dir", "tmp_outputs/benchmark_results")
        from rag_stack.optimizer.benchmark.analyze import analyze_directory
        print(f"\nAnalyzing results in {output_dir} ...")
        analyze_directory(output_dir)


# ---------------------------------------------------------------------------
# Argparse hooks (used by both top-level rag_stack CLI and __main__.py)
# ---------------------------------------------------------------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    """Attach benchmark-specific args to a parser. Used by both this module's
    ``main()`` and the top-level ``rag_stack/cli.py`` subcommand registration.
    """
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--plot", action="store_true", help="Auto-generate plots after run")


def cmd_benchmark(args) -> None:
    """Dispatcher: parse the YAML, route to legacy-single-run or matrix-sweep."""
    config = load_config(args.config)

    if _is_matrix_config(config):
        _run_matrix(config)
        return

    # --- Legacy single-run path ---
    # synthetic_function is now a dict with required `type` and `name`
    # (validated in _validate_legacy_config). The historical "all" string
    # shorthand is gone — it didn't fit the per-benchmark `name` model.
    fn_entry = config["synthetic_function"]
    run_single(config, fn_entry)

    if args.plot:
        output_dir = config.get("output_dir", str(DEFAULT_OUTPUT_DIR))
        from rag_stack.optimizer.benchmark.analyze import analyze_directory
        analyze_directory(output_dir)


def main() -> None:
    """Standalone entry point — used by ``__main__.py``."""
    parser = argparse.ArgumentParser(description="Benchmark optimizer performance")
    add_args(parser)
    args = parser.parse_args()
    cmd_benchmark(args)


if __name__ == "__main__":
    main()
