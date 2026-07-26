"""hardware_key is the single calibration locator: a known PRESET key (gpu in
GenZ system_configs / cpu in CPU_SPECS) may NOT also carry explicit specs; a NEW
key may. interconnect is exempt (preset⊕override). See config_validator
._check_hw_preset_exclusive."""
from __future__ import annotations

from rag_stack.config_validator import ConfigValidator


def _errors(system: dict) -> list[str]:
    v = ConfigValidator.__new__(ConfigValidator)
    v.errors = []
    v._check_hw_preset_exclusive(system)
    return v.errors


def test_bare_preset_strings_ok():
    assert _errors({"gpu": "H100_GPU", "cpu": "epyc_9124"}) == []


def test_preset_key_with_extra_specs_rejected():
    errs = _errors({"gpu": {"hardware_key": "H100_GPU", "flops_tflops": 989}})
    assert errs and "H100_GPU" in errs[0] and "flops_tflops" in errs[0]


def test_preset_dict_bare_key_ok():
    assert _errors({"gpu": {"hardware_key": "H100_GPU"}}) == []


def test_new_key_with_specs_ok():
    assert _errors({
        "gpu": {"hardware_key": "my_h200", "flops_tflops": 1979, "hbm_bw_gbps": 4800},
        "cpu": {"hardware_key": "my_epyc", "num_cores": 64, "mem_bandwidth_gbps": 460},
    }) == []


def test_inline_hardware_without_identity_is_rejected():
    gpu_errors = _errors({"gpu": {"flops_tflops": 1979}})
    cpu_errors = _errors({"cpu": {"num_cores": 64}})
    assert gpu_errors and "hardware_key" in gpu_errors[0]
    assert cpu_errors and "hardware_key" in cpu_errors[0]


def test_cpu_preset_via_cpu_preset_field_rejected():
    errs = _errors({"cpu": {"cpu_preset": "epyc_9124", "num_cores": 99}})
    assert errs and "epyc_9124" in errs[0]


def test_interconnect_preset_plus_override_is_exempt():
    # interconnect keeps preset⊕override — a link type with a measured bandwidth is fine.
    assert _errors({
        "gpu": "H100_GPU",
        "interconnect": {"gpu_gpu": {"type": "nvlink4", "bandwidth_gbps": 300}},
    }) == []
