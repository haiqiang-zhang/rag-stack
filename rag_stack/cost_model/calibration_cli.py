"""Cost-model calibration entry point.

The package entry point and the top-level calibration dispatcher execute the
same commands and use the same profile-root contract.
"""

from rag_stack.calibration_cli import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
