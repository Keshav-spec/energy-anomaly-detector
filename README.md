<div align="center">

# ⚡ Energy Anomaly Detector

**AI-powered detection of wasteful and abnormal electricity usage — with real-time email alerts.**

*Built for the 1M1B AI for Sustainability Virtual Internship (IBM SkillsBuild & AICTE) · Aligned with UN SDG 7 & SDG 12*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SDG 7](https://img.shields.io/badge/SDG-7%20Clean%20Energy-FCC30B?style=for-the-badge)](https://sdgs.un.org/goals/goal7)
[![SDG 12](https://img.shields.io/badge/SDG-12%20Responsible%20Consumption-BF8B2E?style=for-the-badge)](https://sdgs.un.org/goals/goal12)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<!-- 📌 ADD IMAGE HERE: banner.png — a wide banner/hero image (1200x400px approx).
     Could be a simple graphic combining a house/building icon + a power/lightning bolt + a graph line.
     Save as: assets/banner.png and reference below -->
<img src="assets/banner.png" alt="Energy Anomaly Detector banner" width="100%">

</div>

---

## 📖 Overview

Household and building electricity usage often contains silent inefficiencies — appliances left on, faulty equipment drawing excess power, HVAC systems misbehaving — that go unnoticed for weeks and quietly inflate both bills and carbon footprint.

**Energy Anomaly Detector** uses an unsupervised **Isolation Forest** model on hourly usage patterns (time-of-day and day-of-week aware) to flag abnormal consumption in near real-time, and pushes an **email alert** the moment an anomaly is detected — so the user can act on it immediately instead of discovering it on next month's bill.

> Built as part of the **1M1B AI for Sustainability Virtual Internship**, in collaboration with **IBM SkillsBuild** and **AICTE**.

---

## ✨ Features

- 🔍 **Context-aware anomaly detection** — compares usage against the expected pattern for that specific hour and day, not a flat average
- 📊 **Feature-engineered pipeline** — hourly resampling, rolling-average baselines, time-based features
- 📧 **Real-time email alerts** — automatic notification the moment an anomaly is flagged, with usage vs. expected values in the message
- 🧩 **Modular alert system** — built to support additional channels (SMS, Home Assistant) via a single config flag
- 🖼️ **Visual anomaly reports** — before/after usage plots for every flagged event
- 🛡️ **Responsible AI by design** — transparent scoring, rate-limited alerts, no data leaves the local environment

---

<!-- 📌 ADD IMAGE HERE: demo.gif or screenshot.png — a screen recording (as GIF) or screenshot
     showing the pipeline running in terminal AND the resulting email alert side by side.
     Save as: assets/demo.gif -->
## 🎬 Demo

<div align="center">
<img src="assets/demo.gif" alt="Pipeline running and email alert demo" width="85%">
</div>

---

## 🏗️ Architecture

<!-- 📌 ADD IMAGE HERE: architecture-diagram.png — the Mermaid flow diagram generated
     via IBM Bob in Phase 4 of the build (Raw data → Preprocessing → Isolation Forest →
     Flagged anomalies → Email alert). Export it as PNG/SVG.
     Save as: assets/architecture-diagram.png -->
<div align="center">
<img src="assets/architecture-diagram.png" alt="Pipeline architecture diagram" width="90%">
</div>

```
Raw smart-meter data
        │
        ▼
Hourly resampling + feature engineering
   (hour_of_day, day_of_week, rolling_mean_24h)
        │
        ▼
   Isolation Forest model
        │
        ▼
   Anomaly flagged? ──No──▶ Log normally
        │
       Yes
        │
        ▼
   Email alert dispatched
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data handling | pandas, numpy |
| Model | scikit-learn (Isolation Forest) |
| Visualization | matplotlib, seaborn |
| Alerts | smtplib (email) |
| Dev workflow | IBM Bob (planning, code generation, diagrams) |
| Dataset | UCI Individual Household Electric Power Consumption |

---

## 📂 Project Structure

```
energy-anomaly-detector/
├── assets/                  # images for this README
│   ├── banner.png
│   ├── demo.gif
│   ├── architecture-diagram.png
│   └── anomaly-plots/
├── data/
│   ├── raw/                 # original dataset
│   └── processed/           # hourly_features.csv
├── notebooks/
│   └── 01_load_explore.ipynb
├── outputs/                 # generated anomaly plots + logs
├── src/
│   ├── pipeline.py          # end-to-end run script
│   ├── model.pkl            # trained Isolation Forest
│   └── alerts.py            # email alert logic
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/Keshav-spec/energy-anomaly-detector.git
cd energy-anomaly-detector

# Set up virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure email alerts

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

```env
ALERT_EMAIL=your_email@gmail.com
ALERT_EMAIL_PASSWORD=your_gmail_app_password
```

> Use a [Gmail App Password](https://myaccount.google.com/apppasswords) — not your real account password.

---

## ▶️ Usage

```bash
# Run the full pipeline: load data → detect anomalies → send alerts
python src/pipeline.py
```

Flagged anomalies are:
- Logged to `outputs/flagged_anomalies.csv`
- Plotted to `outputs/anomaly-plots/`
- Emailed immediately to the configured address

---

## 📈 Results

<!-- 📌 ADD IMAGE HERE: 3–5 anomaly plots from Phase 5 of the build (actual vs expected usage,
     with the anomaly highlighted). Save each as: assets/anomaly-plots/anomaly_1.png, anomaly_2.png, etc. -->
<div align="center">
<img src="assets/anomaly-plots/anomaly_1.png" alt="Example detected anomaly" width="80%">
</div>

| Metric | Value |
|---|---|
| Detection method | Isolation Forest vs. naive 95th-percentile threshold |
| Anomalies flagged (test period) | *fill in your number* |
| False-positive mitigation | Alert rate-limited to 1/hour |

---

## 🛡️ Responsible AI Considerations

- **Transparency** — every alert includes the model's anomaly score, not just a binary flag
- **Fairness** — trained on a single household's data; usage patterns may not generalize across household sizes/climates without retraining
- **Privacy** — all processing runs locally; no usage data is transmitted to a third party
- **Alert fatigue mitigation** — notifications are rate-limited to avoid desensitizing the user during sustained anomalies

---

## 🎯 Impact

Early detection of wasteful electricity usage helps households and buildings reduce both cost and carbon footprint, directly supporting **SDG 7 (Affordable & Clean Energy)** and **SDG 12 (Responsible Consumption & Production)**.

---

## 🗺️ Roadmap

- [x] Isolation Forest anomaly detection
- [x] Email alert integration
- [ ] SMS alerts (Twilio)
- [ ] Home Assistant notification integration
- [ ] Multi-household / federated deployment

---

## 👤 Author

**Keshav**
B.Tech Computer Science (Data Science), VIT Chennai
[GitHub](https://github.com/Keshav-spec)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

<div align="center">

*Built with 💡 for the 1M1B AI for Sustainability Virtual Internship*

</div>