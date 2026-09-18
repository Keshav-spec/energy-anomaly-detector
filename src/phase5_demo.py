"""
Phase 5 — Testing & Demo Output
Produces labeled anomaly plots and a naive-baseline comparison for the held-out test period.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TEST_CSV = BASE_DIR / "data" / "processed" / "test_features.csv"
FULL_CSV = BASE_DIR / "data" / "processed" / "hourly_features.csv"
MODEL_PKL = BASE_DIR / "src" / "model.pkl"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plausible real-world captions (filled in after inspecting top anomalies)
# ---------------------------------------------------------------------------
CAPTIONS: dict[int, str] = {}   # populated dynamically after anomaly selection

# ---------------------------------------------------------------------------
# 1. Load data & model, run inference on test split
# ---------------------------------------------------------------------------
print("Loading data and model …")
full_df = pd.read_csv(FULL_CSV, index_col=0, parse_dates=True)
test_df = pd.read_csv(TEST_CSV, index_col=0, parse_dates=True)
model = joblib.load(MODEL_PKL)

FEATURE_COLS = ["usage", "hour_of_day", "day_of_week", "rolling_mean_24h"]
X_test = test_df[FEATURE_COLS]

test_df = test_df.copy()
test_df["anomaly_label"] = model.predict(X_test)
test_df["anomaly_score"] = model.decision_function(X_test)   # lower = more anomalous
test_df["is_anomaly"] = test_df["anomaly_label"] == -1

print(f"Test period : {test_df.index.min()}  to  {test_df.index.max()}")
print(f"Total hours : {len(test_df):,}")
print(f"Flagged     : {test_df['is_anomaly'].sum():,}  ({test_df['is_anomaly'].mean()*100:.2f} %)")

# ---------------------------------------------------------------------------
# 2. Pick 5 most anomalous hours (lowest decision score = most anomalous)
# ---------------------------------------------------------------------------
top5 = test_df[test_df["is_anomaly"]].nsmallest(5, "anomaly_score")
print("\nTop-5 anomalies (lowest anomaly score):")
print(top5[["usage", "hour_of_day", "day_of_week", "rolling_mean_24h", "anomaly_score"]])

# ---------------------------------------------------------------------------
# 3. Build plausible captions based on actual hour/usage characteristics
# ---------------------------------------------------------------------------
def build_caption(row: pd.Series, ts: pd.Timestamp) -> str:
    """Generate a plausible real-world explanation for a flagged anomaly."""
    hour = int(row["hour_of_day"])
    usage = row["usage"]
    baseline = row["rolling_mean_24h"]
    ratio = usage / baseline if baseline > 0 else float("inf")
    day_name = ts.strftime("%A")

    if hour >= 0 and hour <= 5:
        time_label = "overnight"
    elif hour >= 6 and hour <= 9:
        time_label = "early morning"
    elif hour >= 22:
        time_label = "late night"
    else:
        time_label = f"{hour:02d}:00"

    if ratio > 3:
        cause = "possible electric heater or boiler malfunction left running"
    elif ratio > 2:
        cause = "likely high-draw appliance (oven, tumble dryer) active well outside normal schedule"
    elif ratio > 1.5:
        cause = "sustained elevated draw — consistent with a faulty appliance or unattended cooking"
    else:
        cause = "usage unusually low — possible sensor dropout or circuit breaker trip"

    return (
        f"{ratio:.1f}× normal at {time_label} on {day_name} "
        f"({usage:.2f} kW vs {baseline:.2f} kW baseline) — {cause}."
    )


# ---------------------------------------------------------------------------
# 4. Plot each anomaly: 48-hour context window + red spike marker
# ---------------------------------------------------------------------------
CONTEXT_HOURS = 48

def plot_anomaly(ts: pd.Timestamp, row: pd.Series, rank: int, caption: str, full_df: pd.DataFrame):
    window_start = ts - pd.Timedelta(hours=CONTEXT_HOURS // 2)
    window_end   = ts + pd.Timedelta(hours=CONTEXT_HOURS // 2)
    window = full_df.loc[window_start:window_end, "usage"].copy()

    fig, ax = plt.subplots(figsize=(13, 5), dpi=130)

    # 48-hour usage line
    ax.plot(window.index, window.values, color="#2563eb", linewidth=1.6,
            label="Hourly usage (kW)", zorder=2)

    # 24-h rolling baseline
    baseline_window = full_df.loc[window_start:window_end, "rolling_mean_24h"]
    ax.plot(baseline_window.index, baseline_window.values,
            color="#94a3b8", linewidth=1.2, linestyle="--",
            label="24-h rolling mean", zorder=1)

    # Anomaly marker
    ax.scatter([ts], [row["usage"]], color="#dc2626", s=120, zorder=5,
               label=f"Anomaly  ({row['usage']:.2f} kW)", edgecolors="white", linewidths=0.8)

    # Vertical reference line
    ax.axvline(ts, color="#dc2626", linewidth=0.8, linestyle=":", alpha=0.6)

    ax.set_title(
        f"Anomaly #{rank}  —  {ts.strftime('%a %d %b %Y  %H:%M')}",
        fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Timestamp", fontsize=10)
    ax.set_ylabel("Active Power (kW)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    # Caption below plot
    fig.text(0.5, -0.02, f"Interpretation: {caption}",
             ha="center", va="top", fontsize=9, color="#374151",
             style="italic", wrap=True)

    fig.tight_layout()
    fname = PLOTS_DIR / f"anomaly_{rank:02d}_{ts.strftime('%Y%m%d_%H')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname.name}")
    return fname


print("\n--- Generating anomaly context plots ---")
saved_plots = []
for rank, (ts, row) in enumerate(top5.iterrows(), start=1):
    caption = build_caption(row, ts)
    print(f"\nAnomaly #{rank}  {ts}  |  {caption}")
    p = plot_anomaly(ts, row, rank, caption, full_df)
    saved_plots.append(p)

# ---------------------------------------------------------------------------
# 5. Naive baseline comparison
#    Naive rule: flag if usage > 95th-percentile of ALL test hours
# ---------------------------------------------------------------------------
print("\n--- Generating naive-baseline comparison ---")

p95_threshold = test_df["usage"].quantile(0.95)
test_df["naive_flag"] = test_df["usage"] > p95_threshold
test_df["model_flag"] = test_df["is_anomaly"]

# Contextual anomalies: caught by model but NOT by naive rule
model_only = test_df[test_df["model_flag"] & ~test_df["naive_flag"]]
naive_only  = test_df[test_df["naive_flag"] & ~test_df["model_flag"]]
both        = test_df[test_df["model_flag"] &  test_df["naive_flag"]]

print(f"95th-percentile threshold  : {p95_threshold:.3f} kW")
print(f"Naive flags total          : {test_df['naive_flag'].sum()}")
print(f"Model flags total          : {test_df['model_flag'].sum()}")
print(f"Caught by BOTH             : {len(both)}")
print(f"Model-only (contextual)    : {len(model_only)}")
print(f"Naive-only (false alarms?) : {len(naive_only)}")

# ---- Comparison scatter plot ----
fig, ax = plt.subplots(figsize=(11, 5), dpi=130)

ax.scatter(test_df.index, test_df["usage"],
           color="#cbd5e1", s=4, alpha=0.5, label="Normal", zorder=1)

ax.scatter(naive_only.index, naive_only["usage"],
           color="#f59e0b", s=20, alpha=0.85,
           label=f"Naive-only flags  (n={len(naive_only)})", zorder=3)

ax.scatter(model_only.index, model_only["usage"],
           color="#8b5cf6", s=20, alpha=0.85,
           label=f"Model-only / contextual  (n={len(model_only)})", zorder=4)

ax.scatter(both.index, both["usage"],
           color="#dc2626", s=30, alpha=0.9,
           label=f"Both methods agree  (n={len(both)})", zorder=5)

ax.axhline(p95_threshold, color="#f59e0b", linewidth=1.2, linestyle="--",
           label=f"95th-pct threshold  ({p95_threshold:.2f} kW)")

ax.set_title(
    "Model vs. Naive Threshold — Anomaly Detection Comparison (Test Period)",
    fontsize=12, fontweight="bold", pad=10
)
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Hourly Active Power (kW)", fontsize=10)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, fontsize=8)
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="upper right", frameon=True, fontsize=9)
fig.tight_layout()

comparison_path = PLOTS_DIR / "baseline_comparison.png"
fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {comparison_path.name}")

# ---- Summary table ----
summary = pd.DataFrame({
    "Method": ["Naive (>P95)", "Isolation Forest", "Overlap (both)"],
    "Flags": [int(test_df["naive_flag"].sum()),
              int(test_df["model_flag"].sum()),
              len(both)],
    "Unique flags": [len(naive_only), len(model_only), len(both)],
    "Contextual anomalies caught": ["No (pure magnitude)", "Yes (hour+day context)", "—"],
})
summary_path = BASE_DIR / "outputs" / "baseline_comparison_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"  Comparison summary saved: {summary_path.name}")

# ---------------------------------------------------------------------------
# 6. Print final summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Phase 5 complete. Output files:")
for p in saved_plots:
    print(f"  {p.relative_to(BASE_DIR)}")
print(f"  {comparison_path.relative_to(BASE_DIR)}")
print(f"  {summary_path.relative_to(BASE_DIR)}")
print("="*60)
