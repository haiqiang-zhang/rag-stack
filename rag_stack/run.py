"""Backward-compatible entry point — delegates to ``rag_stack.cli optimize``.

Use ``python -m rag_stack optimize ...`` instead.
"""

import sys
from rag_stack.cli import cmd_optimize, _add_optimize_args
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run RAG-Stack optimization")
    _add_optimize_args(parser)
    args = parser.parse_args()
    cmd_optimize(args)


if __name__ == "__main__":
    main()
