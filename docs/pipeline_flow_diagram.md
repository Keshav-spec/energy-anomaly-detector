# Prototype Flow Diagram — AI-Based Energy Anomaly Detector

```mermaid
flowchart TD
    A([Raw Data\nhousehold_power_consumption.txt\n2.07M minute-level records]) --> B

    B["Resample & Feature Engineering\n─────────────────────────\n• Resample to 1-hour intervals\n• Interpolate missing gaps\n• Extract hour_of_day, day_of_week\n• Compute 24-h rolling mean baseline"] --> C

    C{"Chronological Split\n80 / 20"}
    C -->|"Train set\n27,671 hours"| D
    C -->|"Test set\n6,918 hours"| E

    D["Isolation Forest Training\n─────────────────────────\n• n_estimators = 200\n• contamination = 0.02\n• Features: usage, hour_of_day,\n  day_of_week, rolling_mean_24h"] --> E

    E["Inference — Anomaly Scoring\n─────────────────────────\n• Predict label  -1 = anomaly  /  1 = normal\n• Compute decision_function score\n  (lower = more anomalous)"] --> F

    F{"Anomaly\nflagged?"}
    F -->|"Yes  (label = -1)"| G
    F -->|"No  (label = 1)"| H([Normal hour\nno action])

    G["Flagged Anomaly\n─────────────────────────\nRecord: timestamp, usage kW,\nanomaly_score, hour, day"] --> I
    G --> J

    I["Context Plot\n─────────────────────────\n• 48-h usage window  blue line\n• Anomaly marker  red dot\n• 24-h rolling baseline  dashed\n• Plausible-cause caption\n→ outputs/plots/anomaly_NN.png"]

    J["Baseline Comparison\n─────────────────────────\n• Naive rule: usage > P95\n• Model: contextual score\n• ~97 % fewer false alerts\n→ outputs/plots/baseline_comparison.png"]
```

> **Export instructions:** paste the fenced block above into [mermaid.live](https://mermaid.live) and use *Download PNG* or *Download SVG* to get the image for your slide deck / report.
