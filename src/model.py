"""
Isolation Forest model training and inference module.
Part of the AI-Based Energy Anomaly Detector project.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ["usage", "hour_of_day", "day_of_week", "rolling_mean_24h"]


def train_isolation_forest(
    train_df: pd.DataFrame,
    features: list = None,
    n_estimators: int = 200,
    contamination: float = 0.02,
    random_state: int = 42,
    model_output_path: Union[str, Path] = "src/model.pkl",
) -> IsolationForest:
    """
    Trains an Isolation Forest model on historical feature data and saves it.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training feature set.
    features : list
        Feature column names to train on.
    n_estimators : int
        Number of decision trees (default 200).
    contamination : float
        Expected proportion of anomalies (default 0.02 = 2%).
    random_state : int
        Random seed for reproducibility.
    model_output_path : str or Path
        Destination path for serializing the trained model.

    Returns:
    --------
    IsolationForest
        Trained model instance.
    """
    if features is None:
        features = FEATURE_COLS

    missing_cols = [col for col in features if col not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Features missing from training DataFrame: {missing_cols}")

    X_train = train_df[features]
    print(f"Training Isolation Forest on {len(X_train):,} samples with features: {features}...")
    print(f"Hyperparameters: n_estimators={n_estimators}, contamination={contamination}, random_state={random_state}")

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)

    out_path = Path(model_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Trained model saved to: {out_path.resolve()}")
    return model


def evaluate_anomalies(
    model: IsolationForest,
    eval_df: pd.DataFrame,
    features: list = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Runs anomaly detection inference and computes decision scores.

    -1 = Anomaly
     1 = Normal
    """
    if features is None:
        features = FEATURE_COLS

    X = eval_df[features]
    results_df = eval_df.copy()

    # Predict: -1 for anomaly, 1 for normal
    results_df["anomaly_label"] = model.predict(X)
    # Decision function: lower score = more anomalous
    results_df["anomaly_score"] = model.decision_function(X)
    results_df["is_anomaly"] = results_df["anomaly_label"] == -1

    total = len(results_df)
    anomalies = int(results_df["is_anomaly"].sum())
    normal = total - anomalies
    anomaly_rate = (anomalies / total) * 100 if total > 0 else 0.0

    metrics = {
        "total_samples": total,
        "normal_count": normal,
        "anomaly_count": anomalies,
        "anomaly_percentage": anomaly_rate,
    }

    print("\n--- Anomaly Evaluation Summary ---")
    print(f"Total samples tested: {total:,}")
    print(f"Normal hours: {normal:,} ({(normal/total)*100:.2f}%)")
    print(f"Flagged anomalies: {anomalies:,} ({anomaly_rate:.2f}%)")
    print(f"Decision scores range: [{results_df['anomaly_score'].min():.4f}, {results_df['anomaly_score'].max():.4f}]")

    return results_df, metrics


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "data" / "processed" / "train_features.csv"
    test_path = base_dir / "data" / "processed" / "test_features.csv"

    if not train_path.exists() or not test_path.exists():
        print("Processed feature datasets not found! Run pipeline.py first.")
    else:
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)

        model = train_isolation_forest(train_df, model_output_path=base_dir / "src" / "model.pkl")
        results_df, metrics = evaluate_anomalies(model, test_df)

        # Save test predictions
        out_preds = base_dir / "outputs" / "test_anomalies.csv"
        results_df.to_csv(out_preds)
        print(f"Test predictions and anomaly scores saved to: {out_preds}")
