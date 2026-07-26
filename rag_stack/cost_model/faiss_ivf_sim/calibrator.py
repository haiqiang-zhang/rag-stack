import re
import warnings

warnings.filterwarnings("ignore", message="DataFrameGroupBy.apply operated on the grouping columns")


def parse_index_config(index_name: str) -> dict:
    """
    Parse FAISS index factory string to extract configuration.

    Examples:
        "IVF1024,PQ32x4fs"  -> {nlist: 1024, M: 32, nbits: 4, fast_scan: True}
        "IVF1024,PQ32x4fsr" -> {nlist: 1024, M: 32, nbits: 4, fast_scan: True, by_residual: True}
    """
    config = {
        "nlist": 1024,
        "M": 32,
        "nbits": 8,
        "fast_scan": False,
        "by_residual": False,
        "index_type": "pq",
    }

    assert "IVF" in index_name, (
        f"Expected an IVF index factory string, got: {index_name!r}"
    )

    ivf_part, pq_part = index_name.split(",", 1)
    config["nlist"] = int(ivf_part.replace("IVF", ""))

    # IVF-Flat: "IVF{nlist},Flat" — full vectors, no PQ. M/nbits keep their
    # placeholder defaults (the Flat cost-model path does not use them).
    if "PQ" not in pq_part:
        assert "Flat" in pq_part, (
            f"Unsupported IVF index (expected '...,PQ...' or '...,Flat'): {index_name!r}"
        )
        config["index_type"] = "flat"
        config["by_residual"] = False
        assert config["nlist"] > 0, f"nlist must be positive, got {config['nlist']}"
        return config

    # Handles both formats:
    #   PQ32x4fs  → M=32, nbits=4, fast_scan=True,  by_residual=False
    #   PQ32x4fsr → M=32, nbits=4, fast_scan=True,  by_residual=True
    #   PQ16      → M=16, nbits=8, fast_scan=False,  by_residual=True
    #   PQ32x8    → M=32, nbits=8, fast_scan=False,  by_residual=True
    match = re.search(r"PQ(\d+)(?:x(\d+)(fs)?(r)?)?", pq_part)
    assert match, f"Failed to parse PQ part: {pq_part!r}"

    config["M"] = int(match.group(1))
    if match.group(2) is not None:
        config["nbits"] = int(match.group(2))
    config["fast_scan"] = match.group(3) is not None

    # FastScan: by_residual only with explicit 'r' suffix
    # Standard PQ (nbits=8): by_residual=True (FAISS default for IVF-PQ)
    if config["fast_scan"]:
        config["by_residual"] = match.group(4) is not None
    else:
        config["by_residual"] = True

    assert config["nlist"] > 0, f"nlist must be positive, got {config['nlist']}"
    assert config["M"] > 0, f"M must be positive, got {config['M']}"
    assert config["nbits"] in (4, 8), (
        f"nbits must be 4 (FastScan) or 8 (standard PQ), got {config['nbits']}"
    )
    assert not config["fast_scan"] or config["nbits"] == 4, (
        f"FastScan requires nbits=4, got nbits={config['nbits']}"
    )

    return config


# ── Hardware from YAML ────────────────────────────────────────────────────────


def register_hardware_from_yaml(yaml_path: str) -> str:
    """Parse a rag_stack YAML config and register its CPU specs as a hardware preset.

    Returns the CPU hardware key (``system.cpu`` preset string, or inline
    ``system.cpu.hardware_key``).
    """
    from rag_stack.types import SystemConfig
    from rag_stack.cost_model.calibration_config import load_calibration_config
    from rag_stack.cost_model.faiss_system_registry import register_configured_faiss_preset

    config = load_calibration_config(yaml_path, require_cpu=True)
    system = config["system"]
    sys_config = SystemConfig.from_dict(system)
    cpu = system.get("cpu")
    if isinstance(cpu, str):
        preset_name = cpu
    else:
        preset_name = (cpu or {}).get("hardware_key")
    if not preset_name:
        raise ValueError(
            f"{yaml_path}: system.cpu must be a preset string or a dict with hardware_key"
        )
    register_configured_faiss_preset("ivf", preset_name, sys_config)
    return preset_name
