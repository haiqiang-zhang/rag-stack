"""
Unified CLI for FAISS IVF-PQ cost simulator.

Usage:
    python -m rag_stack.cost_model.faiss_ivf_sim benchmark [options]

(The Ax Bayesian-optimization `calibrate` command was removed — coefficients are
now hand/direct-fit; see the per-operator calibrate tools under benchmarks/.)
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m rag_stack.cost_model.faiss_ivf_sim <command> [options]")
        print()
        print("Commands:")
        print("  benchmark   Compare cost model predictions vs actual FAISS results")
        sys.exit(0)

    command = sys.argv[1]
    # Remove the subcommand from argv so argparse in each module works correctly
    remaining_args = sys.argv[2:]

    if command == "benchmark":
        from .simulator_benchmark import benchmark_main
        benchmark_main(remaining_args)
    else:
        print(f"Unknown command: {command}")
        print("Use 'benchmark'")
        sys.exit(1)


if __name__ == "__main__":
    main()
