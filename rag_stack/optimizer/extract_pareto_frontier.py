import torch
from typing import Optional, Tuple

from botorch.models.model_list_gp_regression import ModelListGP
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize
from torch import Tensor

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps


def extract_pareto_frontier(
    model: ModelListGP,
    bounds: Tensor,
    search_space,
    tkwargs: dict,
    n_test: int = 10000,
    fidelity_dim: Optional[int] = None,
    target_fidelity: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Extract Pareto frontier from posterior mean.

    Supports both regular and multi-fidelity cases:
    - Regular: samples entire parameter space
    - Multi-fidelity: samples parameter space only, fixes fidelity to target_fidelity

    Args:
        model: Trained ModelListGP
        bounds: Parameter bounds (2, DIM) tensor
        search_space: Ax SearchSpace defining parameters
        tkwargs: torch kwargs (dtype, device)
        n_test: Number of test points to sample
        fidelity_dim: Optional index of fidelity dimension (if multi-fidelity)
        target_fidelity: Optional target fidelity value (required if fidelity_dim is set)

    Returns:
        pareto_x: Pareto-optimal design points (original space)
        pareto_y: Pareto-optimal objective values
    """
    ssops = AxSearchSpaceOps(search_space, tkwargs)

    # Determine if multi-fidelity
    is_multifidelity = (fidelity_dim is not None)

    if is_multifidelity:
        if target_fidelity is None:
            raise ValueError("target_fidelity must be provided when fidelity_dim is set")

        # Multi-fidelity: sample only parameter dimensions
        dim_params = fidelity_dim  # Assuming fidelity is the last dimension
        param_bounds = bounds[:, :dim_params]
        test_x_params = draw_sobol_samples(bounds=param_bounds, n=n_test, q=1).squeeze(1)
        test_x_params = ssops.snap_to_grid(test_x_params)

        # Add target fidelity dimension
        fidelity_col = torch.full((n_test, 1), target_fidelity, **tkwargs)
        test_x = torch.cat([test_x_params, fidelity_col], dim=-1)
    else:
        # Regular: sample entire parameter space
        test_x = draw_sobol_samples(bounds=bounds, n=n_test, q=1).squeeze(1)
        test_x = ssops.snap_to_grid(test_x)

    # Normalize and predict
    test_x_norm = normalize(test_x, bounds)

    with torch.no_grad():
        # ModelListGP requires unsqueeze(-2) for batch handling
        pred = model.posterior(test_x_norm.unsqueeze(-2)).mean.squeeze(-2)

    # Find Pareto-optimal
    pareto_mask = is_non_dominated(pred)
    pareto_x = test_x[pareto_mask]
    pareto_y = pred[pareto_mask]

    # Sort by first objective (typically P = performance/latency)
    sort_idx = pareto_y[:, 0].argsort()
    pareto_x = pareto_x[sort_idx]
    pareto_y = pareto_y[sort_idx]

    return pareto_x, pareto_y