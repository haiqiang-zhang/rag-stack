"""Compatibility shim for SMAC against newer scikit-learn.

SMAC 2.4.0 (the latest release) does ``from sklearn.tree._tree import DTYPE``
in ``smac/model/random_forest/random_forest.py`` and uses it only to cast the
feature matrix: ``np.asarray(X, dtype=DTYPE)``. ``DTYPE`` has always been the
private alias ``numpy.float32`` for the tree feature dtype. scikit-learn >= 1.9
stopped exporting that name from the compiled ``sklearn.tree._tree`` extension
(it kept ``NODE_DTYPE``), so the import raises ``ImportError`` and takes the
whole optimizer registry down with it.

Restoring the symbol to its historical value (``np.float32``) is exact, not a
guess, and keeps the rest of the environment — which depends on a recent
sklearn (Ax / BoTorch / gpytorch) — untouched. Import this module before
importing anything from ``smac``.
"""

from __future__ import annotations


def _ensure_sklearn_tree_dtype() -> None:
    try:
        import sklearn.tree._tree as _tree
    except Exception:
        # If sklearn itself is unavailable, let SMAC's own import raise the
        # informative error rather than masking it here.
        return
    if not hasattr(_tree, "DTYPE"):
        import numpy as np

        # Historical sklearn definition: DTYPE = np.float32 (tree feature dtype).
        _tree.DTYPE = np.float32  # type: ignore[attr-defined]


_ensure_sklearn_tree_dtype()
