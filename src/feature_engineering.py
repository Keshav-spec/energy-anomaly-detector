"""
Feature engineering and preprocessing pipeline for hourly energy anomaly detection.
Part of the AI-Based Energy Anomaly Detector project.
"""

from pathlib import Path
from typing import Tuple, Union
import pandas as pd


def build_hourly_features(
    df: pd.DataFrame,
    target_col: str = "Global_active_power",
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transforms minute-level power consumption series into hourly engineered features.

    Features created:
    - usage: Resampled hourly average power consumption (kW)
    - hour_of_day: 0-23 diurnal cycle indicator
    - day_of_week: 0-6 weekly cycle indicator (Monday=0, Sunday=6)
    - rolling_mean_24h: 24-hour backward rolling average to baseline recent load

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (full_hourly_df, train_df, test_df)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    print(f"Resampling '{target_col}' to 1-hour intervals...")
    hourly_usage = df[target_col].resample("1h").mean()

    # Interpolate gaps linearly
    missing_before = hourly_usage.isna().sum()
    if missing_before > 0:
        print(f"Interpolating {missing_before} missing hourly intervals...")
        hourly_usage = hourly_usage.interpolate(method="linear")
        # In case edge values are still NaN (e.g. leading)
        hourly_usage = hourly_usage.bfill().ffill()

    # Construct features DataFrame
    features = pd.DataFrame(index=hourly_usage.index)
    features["usage"] = hourly_usage.values
    features["hour_of_day"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["rolling_mean_24h"] = (
        features["usage"].rolling(window=24, min_periods=1).mean()
    )

    # Chronological train/test split
    split_idx = int(len(features) * train_ratio)
    train_df = features.iloc[:split_idx].copy()
    test_df = features.iloc[split_idx:].copy()

    features["split"] = "train"
    features.iloc[split_idx:, features.columns.get_loc("split")] = "test"

    print(f"Total hourly samples: {len(features):,}")
    print(f"Train split: {len(train_df):,} rows ({train_df.index.min()} to {train_df.index.max()})")
    print(f"Test split: {len(test_df):,} rows ({test_df.index.min()} to {test_df.index.max()})")

    return features, train_df, test_df


def save_processed_datasets(
    features: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Union[str, Path] = "data/processed",
) -> None:
    """Saves engineered feature tables to disk."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features.to_csv(out_path / "hourly_features.csv", index=True)
    train_df.to_csv(out_path / "train_features.csv", index=True)
    test_df.to_csv(out_path / "test_features.csv", index=True)
    print(f"Processed feature tables saved to {out_path.resolve()}")


if __name__ == "__main__":
    from data_loader import load_raw_data

    df = load_raw_data()
    features, train_df, test_df = build_hourly_features(df)
    save_processed_datasets(features, train_df, test_df)
