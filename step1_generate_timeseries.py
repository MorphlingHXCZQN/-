"""Step 1: Construct the minimal time-series dataset for downstream analysis.

Inputs:
- data/modelready_timeseries.csv

Outputs:
- data/ts_step1_basic.csv

Notes:
- No modeling
- No scoring
- No feature selection
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/modelready_timeseries.csv")
DEFAULT_OUTPUT = Path("data/ts_step1_basic.csv")


@dataclass(frozen=True)
class Step1Config:
    input_path: Path
    output_path: Path
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    timestamp_candidates: tuple[str, ...] = (
        "charttime",
        "window_start",
        "timestamp",
        "event_time",
    )
    event_time_col: str = "event_time"


def _resolve_timestamp_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Missing timestamp column. Expected one of: "
        + ", ".join(candidates)
    )


def _validate_required_columns(df: pd.DataFrame, timestamp_col: str, event_time_col: str) -> None:
    required = {"stay_id", timestamp_col, event_time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def _to_datetime(series: pd.Series, col_name: str) -> pd.Series:
    converted = pd.to_datetime(series, errors="coerce")
    if converted.isna().all():
        raise ValueError(f"Column {col_name} could not be parsed as datetime.")
    return converted


def _assign_day_index(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    if "day_index" in df.columns:
        return df
    times = _to_datetime(df[timestamp_col], timestamp_col)
    min_time = times.groupby(df["stay_id"]).transform("min")
    df = df.copy()
    df["day_index"] = ((times - min_time).dt.total_seconds() // 86400).astype(int)
    return df


def _aggregate_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    group_cols = ["stay_id", "day_index"]
    if "window_start" not in df.columns:
        df = df.copy()
        df["window_start"] = _to_datetime(df[timestamp_col], timestamp_col)
    df["window_start"] = _to_datetime(df["window_start"], "window_start")
    df["window_end"] = df["window_start"] + pd.Timedelta(hours=24)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in {"day_index"}]
    non_numeric_cols = [c for c in df.columns if c not in set(numeric_cols + group_cols)]

    agg_map: dict[str, str] = {col: "mean" for col in numeric_cols}
    for col in non_numeric_cols:
        agg_map[col] = "first"

    aggregated = df.groupby(group_cols, as_index=False).agg(agg_map)
    return aggregated


def _derive_labels(df: pd.DataFrame, event_time_col: str) -> pd.DataFrame:
    df = df.copy()
    df[event_time_col] = _to_datetime(df[event_time_col], event_time_col)
    event_min = df.groupby("stay_id")[event_time_col].transform("min")
    window_end = _to_datetime(df["window_end"], "window_end")
    label = (event_min > window_end) & (event_min <= window_end + pd.Timedelta(hours=24))
    df["y_future_24h"] = label.astype(int)
    return df


def _assign_split(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stays = df["stay_id"].drop_duplicates().to_numpy()
    rng.shuffle(stays)
    n_total = len(stays)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_ids = set(stays[:n_train])
    val_ids = set(stays[n_train : n_train + n_val])
    test_ids = set(stays[n_train + n_val :])

    def _split(stay_id: int) -> str:
        if stay_id in train_ids:
            return "train"
        if stay_id in val_ids:
            return "val"
        return "test"

    df = df.copy()
    df["dataset_split"] = df["stay_id"].map(_split)
    return df


def run_step1(config: Step1Config) -> Path:
    df = pd.read_csv(config.input_path, low_memory=False)
    timestamp_col = _resolve_timestamp_column(df, config.timestamp_candidates)
    _validate_required_columns(df, timestamp_col, config.event_time_col)

    df = _assign_day_index(df, timestamp_col)
    df = _aggregate_features(df, timestamp_col)
    df = _derive_labels(df, config.event_time_col)
    df = _assign_split(df, config.seed, config.train_ratio, config.val_ratio)

    df = df.sort_values(["stay_id", "day_index"]).reset_index(drop=True)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_path, index=False)
    return config.output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Step 1 time-series dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-time-col", type=str, default="event_time")
    args = parser.parse_args()

    config = Step1Config(
        input_path=args.input,
        output_path=args.output,
        event_time_col=args.event_time_col,
    )
    run_step1(config)


if __name__ == "__main__":
    main()
