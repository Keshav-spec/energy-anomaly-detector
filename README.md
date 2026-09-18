# AI-Based Energy Anomaly Detector

**Project Focus**: Flag wasteful/abnormal electricity usage in homes or buildings near real-time  
**UN Sustainable Development Goals (SDG)**:
- **SDG 7**: Affordable & Clean Energy
- **SDG 12**: Responsible Consumption & Production  
**Internship Program**: 1M1B AI for Sustainability (IBM SkillsBuild & AICTE)

---

## Project Structure

```
energy-anomaly-detector/
├── data/              # Raw + processed dataset
├── notebooks/         # Exploratory data analysis & experiments
├── src/               # Core Python modules & anomaly detection scripts
├── outputs/           # Visualization plots & flagged anomaly reports/logs
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

---

## Setup & Getting Started

### 1. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Responsible AI

| Pillar | Position |
|---|---|
| **Fairness** | Trained on a single French household (UCI, 2006–2010). Usage patterns — peak hours, baseline load, seasonal swing — are household-specific. The model must be retrained on local historical data before deployment in a new household, climate, or occupancy context; applying it unchanged across demographics is a known limitation. |
| **Transparency** | The Isolation Forest exposes a continuous **anomaly score** (scikit-learn `decision_function`) alongside its binary flag. This score is surfaced directly to the user rather than hidden behind a black-box verdict, allowing operators to understand *how* anomalous a flagged hour is and to calibrate their own alert threshold. |
| **Ethics — Alert Fatigue** | Excessive false positives train users to ignore alerts, defeating the system's purpose. The `contamination` hyperparameter (default `0.02`) is the primary lever for controlling flag rate and must be validated against operational feedback. Context-aware flagging already reduces spurious alerts by ~97 % vs. a naive 95th-percentile threshold (see `outputs/plots/baseline_comparison.png`). |
| **Privacy** | Energy consumption data reveals occupancy patterns and daily routines. As a design principle, inference should run **locally on the household's own device** — raw time-series must never be transmitted to a central server or pooled without explicit informed consent. Federated learning is the recommended path if cross-household improvement is ever needed. |

> Full write-up: [`RESPONSIBLE_AI.md`](RESPONSIBLE_AI.md)

---

## Roadmap / Phases
- [x] **Phase 0**: Environment Setup & Project Directory Scaffolding
- [x] **Phase 1**: Data Acquisition & Exploration (2.07M records, datetime indexed, 1-week normal usage plotted)
- [x] **Phase 2**: Data Preprocessing & Feature Engineering (Hourly aggregation, linear gap interpolation, 24h rolling baseline, 80/20 chronological split)
- [x] **Phase 3**: Anomaly Detection Modeling (Isolation Forest trained on 27,671 hours, serialized to `src/model.pkl`, anomalies detected on 6,918 test hours)
- [x] **Phase 5**: Testing & Demo Output (5 labeled anomaly plots, naive-baseline comparison, `outputs/plots/`)
- [x] **Phase 6**: Responsible AI Write-Up
- [x] **Phase 7**: Deliverable Assembly (`deliverable/Energy_Anomaly_Detector.pptx`, 9 slides)
- [x] **Feature**: Push Alert Integration (email via `smtplib`; opt-in `send_alerts=True` in `evaluate_anomalies`)
- [ ] **Phase 4**: Evaluation, Threshold Tuning & Alerting System
