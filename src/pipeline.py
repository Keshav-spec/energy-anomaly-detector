"""
End-to-end execution pipeline for Phase 1 and Phase 2 of Energy Anomaly Detector.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_raw_data
from feature_engineering import build_hourly_features, save_processed_datasets


def run_phase_1_exploration(df: pd.DataFrame, output_dir: Path) -> Path:
    """Executes Phase 1 data validation and plots a sample 1-week normal usage curve."""
    print("\n--- [Phase 1] Data Exploration & Validation ---")
    print(f"Total records: {len(df):,}")
    print(f"Time span: {df.index.min()} to {df.index.max()}")
    print("Checking for null/missing values per column:")
    print(df.isna().sum())

    # Visual confirmation: 1 week of normal usage (Jan 8 - Jan 14, 2007)
    sample_week = df.loc["2007-01-08":"2007-01-14", "Global_active_power"]
    plt.figure(figsize=(15, 6), dpi=120)
    plt.plot(sample_week.index, sample_week.values, color="#0284c7", linewidth=1.0)
    plt.title(
        "Household Power Consumption: 1-Week Sample (Jan 8 – Jan 14, 2007)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Global Active Power (kW)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "sample_week_usage.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Sample 1-week normal usage plot saved: {plot_path}")
    return plot_path


def run_phase_2_preprocessing(df: pd.DataFrame, processed_dir: Path, output_dir: Path):
    """Executes Phase 2 feature engineering, train/test chronological split, and data persistence."""
    print("\n--- [Phase 2] Preprocessing & Feature Engineering ---")
    features, train_df, test_df = build_hourly_features(df, target_col="Global_active_power", train_ratio=0.80)

    # Validate feature table
    assert features.isna().sum().sum() == 0, "Error: Missing values detected in engineered feature table!"
    assert "usage" in features.columns
    assert "hour_of_day" in features.columns
    assert "day_of_week" in features.columns
    assert "rolling_mean_24h" in features.columns
    assert len(train_df) + len(test_df) == len(features)

    # Comparison plot: Minute noise vs Hourly resampled vs 24h rolling baseline
    sample_range = slice("2007-03-01", "2007-03-03")
    plt.figure(figsize=(15, 6), dpi=120)
    plt.plot(
        df.loc[sample_range].index,
        df.loc[sample_range, "Global_active_power"],
        color="#94a3b8",
        alpha=0.6,
        label="Raw minute-level power",
    )
    plt.plot(
        features.loc[sample_range].index,
        features.loc[sample_range, "usage"],
        color="#2563eb",
        linewidth=2.0,
        label="Hourly average (usage)",
    )
    plt.plot(
        features.loc[sample_range].index,
        features.loc[sample_range, "rolling_mean_24h"],
        color="#ea580c",
        linewidth=2.0,
        linestyle="--",
        label="24h Rolling baseline",
    )
    plt.title(
        "Raw Minute Noise vs. Hourly Mean & 24h Rolling Baseline (March 1–3, 2007)",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Timestamp", fontsize=11)
    plt.ylabel("Active Power (kW)", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    comp_plot_path = output_dir / "plots" / "minute_vs_hourly_comparison.png"
    plt.savefig(comp_plot_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved: {comp_plot_path}")

    # Persist datasets
    save_processed_datasets(features, train_df, test_df, output_dir=processed_dir)
    return features, train_df, test_df


def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_path = base_dir / "data" / "raw" / "household_power_consumption.txt"
    processed_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "outputs"

    df_raw = load_raw_data(raw_path)
    run_phase_1_exploration(df_raw, output_dir)
    run_phase_2_preprocessing(df_raw, processed_dir, output_dir)
    print("\nPhase 1 and Phase 2 successfully completed!")


if __name__ == "__main__":
    main()
