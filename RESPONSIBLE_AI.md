# Responsible AI — Energy Anomaly Detector

## Fairness

This model was trained on a single French household's power consumption data (UCI dataset, 2006–2010).
Usage patterns — peak hours, baseline load, seasonal swing — are specific to that household's size, occupancy schedule, and climate.
Applying the trained model directly to households with different sizes, climates, or appliance sets will produce biased anomaly scores; the model must be **retrained or fine-tuned on local historical data** before deployment in a new context.
Generalising conclusions about "normal" energy use across demographics or geographies without retraining is a known limitation of this approach.

## Transparency

The Isolation Forest model exposes a continuous **anomaly score** (the `decision_function` output) alongside its binary flag.
Rather than presenting a black-box yes/no verdict, the system surfaces this score directly to the user — lower scores indicate greater deviation from learned normal behaviour.
This lets operators calibrate their own alert threshold and understand *how anomalous* a flagged hour is relative to the model's learned baseline, rather than treating every flag as equally urgent.

## Ethics — Alert Fatigue & Contamination Tuning

If the model produces too many false positives, users will learn to ignore alerts — a well-documented phenomenon called **alert fatigue** — which would defeat the safety and efficiency purpose of the system entirely.
The primary mitigation is the `contamination` hyperparameter (set to `0.02`, i.e. 2 % of training hours expected to be anomalous), which directly controls the flag rate; this value should be validated against domain knowledge and adjusted if operational false-positive rates prove too high.
A comparison against a naive 95th-percentile threshold (see `outputs/plots/baseline_comparison.png`) demonstrates that context-aware flagging reduces spurious alerts by ~97 % relative to a magnitude-only rule, substantially reducing alert fatigue risk.

## Privacy

In a real deployment, energy consumption data is sensitive personal information: it reveals occupancy patterns, daily routines, and appliance use.
As a design principle, **inference should run locally on the household's own device or on-premises hardware** — raw consumption time-series should never be transmitted to a central server or pooled across households for model training without explicit, informed consent.
Federated learning or differential privacy techniques are natural next steps if cross-household model improvement is desired without centralising personal data.
