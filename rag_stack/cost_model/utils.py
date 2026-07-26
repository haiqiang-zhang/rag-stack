# ---------------------------------------------------------------------------
# Utilities (from rago/llm_sim/genz_scripts/llm_perf.py)
# ---------------------------------------------------------------------------



import numpy as np
import pandas as pd


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _get_parallelism(num_devices: int) -> list[tuple[int, int]]:
    """All valid (TP, PP) combos for num_devices."""
    combos = []
    tp = 1
    while tp <= num_devices:
        pp = num_devices // tp
        if _is_power_of_two(pp):
            combos.append((tp, pp))
        tp *= 2
    return combos


def _get_powers_of_two_up_to(n: int) -> list[int]:
    result = []
    p = 1
    while p <= n:
        result.append(p)
        p *= 2
    return result


def _get_pareto_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where doubling batch_size doesn't improve throughput."""
    rows_to_drop = []
    group_columns = [
        "model.name", "model.stage", "model.dec_steps",
        "model.seq_len_inference_prefill", "hardware.name", "hardware.num_chips",
    ]
    existing_cols = [c for c in group_columns if c in df.columns]
    if not existing_cols:
        return df
    for _, group_df in df.groupby(existing_cols):
        batch_sizes = sorted(group_df["batch_size"].unique())
        for batch_size in batch_sizes:
            if batch_size == batch_sizes[-1]:
                break
            n_row = group_df[group_df["batch_size"] == batch_size]
            two_n_row = group_df[group_df["batch_size"] == batch_size * 2]
            if len(two_n_row) == 0:
                continue
            latency_n = n_row["time_ms"].values[0]
            latency_2n = two_n_row["time_ms"].values[0]
            if latency_2n >= 2 * latency_n or np.allclose(latency_2n, 2 * latency_n, rtol=1e-2):
                rows_to_drop.extend(
                    group_df[group_df["batch_size"] >= batch_size * 2].index
                )
                break
    return df.drop(rows_to_drop)