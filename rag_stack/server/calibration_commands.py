"""Apps command descriptions for the public calibration dispatcher.

These describe launcher fields, not calibration readiness or measured support.
The command heads are checked against the CLI's own lightweight declarations.
"""

from rag_stack.calibration_cli import COMPONENTS, META_COMMANDS, PASSTHROUGH, VERBS


def command_options(args: list[str]) -> dict:
    head = args[0]
    if head not in {*COMPONENTS, *META_COMMANDS, *PASSTHROUGH, *VERBS}:
        raise ValueError(f"Unsupported calibration command: {head}")
    is_fit = head == "fit" or (
        head in {"generator-tp", "generator-disagg"} and args[1:2] == ["fit"]
    )
    return {
        "needs_config": head != "hnsw-workload",
        "supports_force": not is_fit and head in {
            "ivf", "hnsw", "measure", "all", "all-faiss", "generator-tp", "generator-disagg",
        },
        "supports_maxiter": head in {"ivf", "hnsw", "fit", "all", "all-faiss"},
        "supports_data_paths": head == "workload",
        "requires_project_dir": head == "workload",
        "supports_data_dir": True,
        "supports_profile_dir": True,
        "read_only": head in META_COMMANDS,
        "required_extra_args": ["--cost-model", "--module"] if head == "llm" else [],
        "extra_placeholder": (
            "--cost-model passage_reranker_sim --module tart" if head == "llm"
            else "--scope pipeline" if head in META_COMMANDS
            else ""
        ),
    }


def calibration_commands() -> list[dict]:
    commands = [
        ("status", "Inspect calibration readiness"),
        ("plan", "Plan missing calibration work"),
        *[(component, f"{component.upper()} · measure if needed + fit") for component in COMPONENTS],
        *[(f"{verb} {component}", f"{component.upper()} · {verb} only")
          for component in COMPONENTS for verb in ("measure", "fit")],
        ("all-faiss", "FAISS · measure and fit IVF + HNSW"),
        ("workload", "Workload · corpus × embedding profiles"),
        ("llm", "Encoder / reranker · component calibration"),
        ("generator", "Generator · component calibration"),
        ("generator-tp measure", "Generator TP/PP · measure"),
        ("generator-tp fit", "Generator TP/PP · fit"),
        ("generator-disagg measure", "Generator P/D · measure"),
        ("generator-disagg fit", "Generator P/D · fit"),
        ("retrieval-worker", "Retrieval worker · RRF calibration"),
        ("llmlingua2", "LLMLingua-2 · compressor calibration"),
        ("hnsw-workload", "HNSW workload · train predictor"),
    ]
    return [
        {"value": command, "label": label, **command_options(command.split())}
        for command, label in commands
    ]
