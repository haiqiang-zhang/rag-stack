"""Entry point for ``python -m rag_stack.optimizer.benchmark``.

Equivalent to ``python -m rag_stack benchmark``, which routes through
``rag_stack/cli.py``'s subcommand dispatcher to the same ``cmd_benchmark``.
"""
from rag_stack.optimizer.benchmark.cli import main

if __name__ == "__main__":
    main()
