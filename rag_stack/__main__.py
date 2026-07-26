"""Allow running as ``python -m rag_stack``."""
from rag_stack.cli import main

# Guard is REQUIRED: multiprocessing ``spawn`` re-imports the main module
# (as ``__mp_main__``) in every worker. Without the guard, each worker would
# re-run the whole CLI recursively — corpus preprocessing uses a spawn Pool.
if __name__ == "__main__":
    main()
