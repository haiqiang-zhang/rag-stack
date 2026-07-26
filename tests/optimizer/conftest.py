"""Optimizer tests use CPU objectives and explicit agent transports."""

import pytest


@pytest.fixture(scope="module", autouse=True)
def cpu_optimizer_with_fake_clients():
    import torch
    import rag_stack.ai_clients as clients

    def live_client(*args, **kwargs):
        pytest.fail("Optimizer tests must inject a fake AI client")

    def initialize_cuda(*args, **kwargs):
        pytest.fail("Optimizer tests must use CPU tensors")

    # Module scope also covers shared fitted-optimizer fixtures. Ax snapshots
    # visible device RNG state even for CPU Sobol; expose only CPU to this
    # suite while retaining guards against accidental device initialization.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(clients, "get_ai_client", live_client)
        patch.setattr(torch.cuda, "is_available", lambda: False)
        patch.setattr(torch.cuda, "device_count", lambda: 0)
        patch.setattr(torch.cuda, "init", initialize_cuda)
        patch.setattr(torch.cuda, "_lazy_init", initialize_cuda)
        if hasattr(torch._C, "_cuda_init"):
            patch.setattr(torch._C, "_cuda_init", initialize_cuda)
        yield
