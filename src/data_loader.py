"""
Data loading and ingestion module for Household Electric Power Consumption dataset.
Part of the AI-Based Energy Anomaly Detector project.
"""

from pathlib import Path
from typing import Union
import pandas as pd


def load_raw_data(
    filepath: Union[str, Path] = "data/raw/household_power_consumption.txt",
    nrows: int = None,
) -> pd.DataFrame:
    """
    Loads raw household electricity consumption data.

    Parameters:
    -----------
    filepath : str or Path
        Path to the semicolon-separated raw dataset file.
    nrows : int, optional
        Number of rows to read for quick testing or exploration.

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with DatetimeIndex and float-cast numeric columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path.resolve()}")

    print(f"Loading raw dataset from {path}...")
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        low_memory=False,
        nrows=nrows,
    )

    print(f"Raw shape loaded: {df.shape}")
    print("Parsing datetime index...")
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
    )
    df.drop(columns=["Date", "Time"], inplace=True)
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Successfully loaded and formatted {len(df):,} records.")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print("Dataset Summary:")
    print(df.info())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nStatistical Description:")
    print(df.describe())
