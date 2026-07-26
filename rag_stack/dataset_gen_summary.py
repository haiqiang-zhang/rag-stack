"""Dataset generation job summary — tracks status, PID, and output metadata.

Mirrors project_summary.py pattern. Each generation output folder contains
a summary.json with job_type="dataset_generation" to distinguish from plain
dataset folders and RAG optimization projects.
"""

from __future__ import annotations

import datetime
import json
import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class DatasetGenStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


DATASET_GEN_CONFIG_FILENAME = "dataset_gen_config.yaml"


class DatasetGenSummary(BaseModel):
    job_type: str = "dataset_generation"
    status: DatasetGenStatus
    strategy: str = ""
    config_filename: str = ""
    pid: Optional[int] = None
    timestamp: str = ""
    elapsed_seconds: float = 0
    qa_rows: int = 0
    corpus_rows: int = 0
    error: Optional[str] = None


def load_dataset_gen_summary(output_dir: str) -> Optional[DatasetGenSummary]:
    """Load summary.json from an output directory. Returns None if missing or not a dataset gen job."""
    path = os.path.join(output_dir, "summary.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("job_type") != "dataset_generation":
        return None
    return DatasetGenSummary(**data)


def save_dataset_gen_summary(output_dir: str, summary: DatasetGenSummary) -> None:
    """Write summary.json to an output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary.model_dump(), f, indent=2)


def create_dataset_gen_summary(
    status: DatasetGenStatus = DatasetGenStatus.CREATED,
    strategy: str = "",
    config_filename: str = "",
) -> DatasetGenSummary:
    """Create a new DatasetGenSummary with current timestamp."""
    return DatasetGenSummary(
        status=status,
        strategy=strategy,
        config_filename=config_filename,
        timestamp=datetime.datetime.now().isoformat(),
    )
