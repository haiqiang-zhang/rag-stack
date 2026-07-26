"""SMAC optimizer + Ax→ConfigSpace conversion.

SMAC optimizes over a ConfigSpace ``ConfigurationSpace`` — a different
search-space object than Ax's ``SearchSpace`` — so this package owns both:

  - ``SMACMOBOOptimizer``  — the single SMAC baseline (SMAC3 facade + ParEGO)
  - ``ax_to_configspace`` & friends — translate the Ax ``SearchSpace`` (built by
    ``RagSearchSpace.build_ax_searchspace``) into a SMAC ``ConfigurationSpace``,
    preserving hierarchy via ConfigSpace ``EqualsCondition``.
"""

from . import _sklearn_compat  # noqa: F401 — restore sklearn.tree._tree.DTYPE for SMAC, must precede SMAC import
from .optimizer import SMACMOBOOptimizer
from .configspace import (
    ax_to_configspace,
    cs_config_to_dict,
    arm_to_cs_config,
    dict_to_cs_config,
)

__all__ = [
    "SMACMOBOOptimizer",
    "ax_to_configspace",
    "cs_config_to_dict",
    "arm_to_cs_config",
    "dict_to_cs_config",
]
