
from tests import REPO_ROOT
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from rag_stack.cost_model.faiss_ivf_sim.coefficients import (
    IVFProfileError,
    build_hardware_from_coefficients,
    load_coefficients,
)
from rag_stack.cost_model.faiss_ivf_sim.cost_model import IVFPQCostModel
from rag_stack.cost_model.faiss_ivf_sim.ivfpq_config import IVFPQConfig
from rag_stack.cost_model.faiss_ivf_sim.system_presets import cpu_amd_epyc_9124
from rag_stack.cost_model.faiss_ivf_sim.hardware_calibrator import (
    fitted_coefficient_fields,
)
from rag_stack.ivf_profile_schema import (
    IVF_PROFILE_FIELDS,
    IVF_RUNTIME_COEFFICIENT_FIELDS,
    ivf_profile_validation_error,
)




def _profile_path(cpu_key: str) -> Path:
    return (
        REPO_ROOT
        / "calibration_profiles"
        / "faiss_ivf"
        / f"{cpu_key}.json"
    )


@pytest.mark.parametrize(
    ("cpu_key", "expected"),
    [
        (
            "epyc_9124",
            {
                "blas_efficiency": 0.4269965593984215,
                "simd_efficiency": 0.43525897100583166,
                "mape_latency": 9.752286671218931,
                "max_ape_latency": 64.59624150379439,
                "num_trials": 144,
            },
        ),
        (
            "epyc_7742_2s",
            {
                "blas_efficiency": 0.3782732415525549,
                "simd_efficiency": 1.0,
                "mape_latency": 17.07300372951293,
                "max_ape_latency": 65.07261345337314,
                "num_trials": 144,
            },
        ),
    ],
)
def test_migrated_ivf_profiles_preserve_coefficients_fit_and_grid(
    cpu_key, expected,
):
    path = _profile_path(cpu_key)
    document = json.loads(path.read_text())
    coefficients = load_coefficients(str(path))

    assert set(document) == IVF_PROFILE_FIELDS
    assert asdict(coefficients) == document
    assert ivf_profile_validation_error(document, cpu_key) is None
    assert {field: document[field] for field in expected} == expected
    assert document["index_configs"] == ["IVF,Flat", "IVF,PQ", "IVF,PQx4fs"]
    assert document["batch_sizes"] == [1, 4, 16, 64]
    assert document["fitted_fields"] == list(IVF_RUNTIME_COEFFICIENT_FIELDS)


@pytest.mark.parametrize(
    "legacy_field",
    [
        "provenance",
        "evidence_scope",
        "source_path",
        "source_sha256",
        "raw_hardware_source",
        "raw_hardware_source_sha256",
        "timestamp",
        "fit_method",
    ],
)
def test_ivf_loader_rejects_removed_profile_provenance_fields(
    tmp_path, legacy_field,
):
    document = json.loads(_profile_path("epyc_9124").read_text())
    document[legacy_field] = (
        {"legacy": True} if legacy_field == "provenance" else "legacy"
    )
    path = tmp_path / "epyc_9124.json"
    path.write_text(json.dumps(document))

    with pytest.raises(IVFProfileError, match=rf"extra=.*{legacy_field}"):
        load_coefficients(str(path))


def test_standard_pq_scan_does_not_recharge_precomputed_table():
    cfg = IVFPQConfig(
        index_type="pq",
        d=128,
        N=200_000,
        nlist=512,
        nprobe=128,
        M=32,
        nbits=8,
        k=10,
        batch_size=64,
        by_residual=True,
        use_precomputed_table=True,
        parallel_mode=0,
        cell_imbalance=1.0,
    )
    hw = cpu_amd_epyc_9124(active_threads=32)

    breakdown = IVFPQCostModel(cfg, hw, warm_cache=True).estimate()
    scan = next(op for op in breakdown.operators if op.name == "3_pq_scan_adc")

    codes_bytes = cfg.batch_size * cfg.vectors_scanned * cfg.pq_code_bytes
    lut_read = cfg.batch_size * cfg.nprobe * cfg.lut_size_bytes
    precomputed_table_visit = cfg.batch_size * cfg.nprobe * cfg.M * cfg.ksub * 4

    assert precomputed_table_visit > 0
    assert scan.total_bytes == pytest.approx(codes_bytes + lut_read)
    assert scan.unique_bytes == pytest.approx(codes_bytes)


def test_retired_compute_scalability_coefficient_is_rejected(tmp_path):
    source = "calibration_profiles/faiss_ivf/epyc_9124.json"
    document = json.loads(open(source).read())
    assert "compute_scalability_alpha" not in document

    document["compute_scalability_alpha"] = 0.03
    injected = tmp_path / "ivf.json"
    injected.write_text(json.dumps(document))
    with pytest.raises(IVFProfileError, match="extra=.*compute_scalability_alpha"):
        load_coefficients(str(injected))


def test_builtin_ivf_compute_scaling_remains_linear():
    coefficients = load_coefficients(
        "calibration_profiles/faiss_ivf/epyc_9124.json"
    )
    hardware = build_hardware_from_coefficients(
        coefficients, active_threads=32,
    )
    assert not hasattr(hardware, "compute_scalability_alpha")
    assert hardware._compute_peak_ops(32, "simd") == pytest.approx(
        32 * hardware._compute_peak_ops(1, "simd")
    )


def test_runtime_coefficients_have_exact_standalone_fit_coverage():
    assert fitted_coefficient_fields() == frozenset(
        IVF_RUNTIME_COEFFICIENT_FIELDS
    )
    document = json.loads(open(
        "calibration_profiles/faiss_ivf/epyc_9124.json"
    ).read())
    assert document["fitted_fields"] == list(IVF_RUNTIME_COEFFICIENT_FIELDS)
    for retired in (
        "compute_efficiency",
        "fixed_overhead_us",
        "pq_precomp_per_probe_overhead_us",
        "mode1_query_sync_us",
        "pq_rap_thread_cache_exponent",
        "pq_rap_fastscan_c1",
        "pq_rap_fastscan_c2",
        "pq_rap_fastscan_c3",
        "dram_random_access_latency_ns",
        "numa_remote_bw_ceiling",
    ):
        assert retired not in document


def test_uncalibrated_ivf_shapes_fail_closed():
    with pytest.raises(ValueError, match="parallel_mode=1 is unsupported"):
        IVFPQConfig(parallel_mode=1)
    with pytest.raises(ValueError, match="residual FastScan is unsupported"):
        IVFPQConfig(nbits=4, M=32, by_residual=True)
