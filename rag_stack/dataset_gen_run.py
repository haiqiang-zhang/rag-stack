"""CLI entry point for running dataset generation.

Usage:
    python -m rag_stack.dataset_gen_run \
        --config /path/to/config.yaml \
        --output-dir /path/to/datasets_dir/my_dataset/

The config YAML must contain a 'strategy' field to select the generator class.
Output parquet files (qa.parquet, corpus.parquet) are written to output-dir.
"""

import argparse
import logging
import os
import shutil
import sys
import time
import traceback

from rag_stack.dataset_gen_summary import (
    DATASET_GEN_CONFIG_FILENAME,
    DatasetGenStatus,
    load_dataset_gen_summary,
    save_dataset_gen_summary,
    create_dataset_gen_summary,
)

logger = logging.getLogger("RAG-Stack")


def main():
    parser = argparse.ArgumentParser(description="Run RAG-Stack dataset generation")
    parser.add_argument("--config", required=True, help="Path to generation config YAML")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated parquet files")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isfile(config_path):
        print(f"Error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Read strategy from config
    from rag_stack.utils.config import load_yaml_config
    raw_config = load_yaml_config(config_path)
    strategy = raw_config.get("strategy", "unknown")

    logger.info(f"Starting dataset generation in {output_dir}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Strategy: {strategy}")

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy config into output dir
    dest_config = os.path.join(output_dir, DATASET_GEN_CONFIG_FILENAME)
    if os.path.abspath(config_path) != os.path.abspath(dest_config):
        shutil.copy2(config_path, dest_config)

    # Write PID + status=running to summary
    my_pid = os.getpid()
    summary = load_dataset_gen_summary(output_dir)
    if summary is None:
        summary = create_dataset_gen_summary(strategy=strategy, config_filename=os.path.basename(config_path))
    summary.status = DatasetGenStatus.RUNNING
    summary.pid = my_pid
    summary.strategy = strategy
    summary.timestamp = __import__("datetime").datetime.now().isoformat()
    save_dataset_gen_summary(output_dir, summary)
    logger.info(f"PID {my_pid} registered in summary.json")

    start_time = time.time()
    try:
        from rag_stack.dataset_generator import create_dataset_generator_from_yaml

        generator = create_dataset_generator_from_yaml(config_path)
        dataset = generator.generate()
        dataset.validate()

        qa_path = os.path.join(output_dir, "qa.parquet")
        corpus_path = os.path.join(output_dir, "corpus.parquet")
        dataset.to_parquet(qa_path, corpus_path)

        elapsed = time.time() - start_time

        # Update summary with results
        summary = load_dataset_gen_summary(output_dir) or summary
        summary.status = DatasetGenStatus.COMPLETED
        summary.elapsed_seconds = round(elapsed, 2)
        summary.qa_rows = len(dataset.qa.data) if dataset.qa and dataset.qa.data is not None else 0
        summary.corpus_rows = len(dataset.corpus.data) if dataset.corpus and dataset.corpus.data is not None else 0
        save_dataset_gen_summary(output_dir, summary)

        logger.info(
            f"Dataset generation complete: "
            f"{summary.qa_rows} QA rows, "
            f"{summary.corpus_rows} corpus rows, "
            f"{elapsed:.1f}s elapsed"
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Dataset generation failed: {e}")
        traceback.print_exc()
        summary = load_dataset_gen_summary(output_dir) or summary
        summary.status = DatasetGenStatus.FAILED
        summary.elapsed_seconds = round(elapsed, 2)
        summary.error = str(e)
        save_dataset_gen_summary(output_dir, summary)
        sys.exit(1)


if __name__ == "__main__":
    main()
