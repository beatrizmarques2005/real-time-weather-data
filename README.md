# 🏃 Weather Windows for Outdoor Races — Angra do Heroísmo vs Braga

**Data Stream Mining · Practical Assignment I (PA48) · NOVA IMS**

> An outdoor running event organiser needs reliable weather windows for recurring races. We turn hourly Open-Meteo weather streams for **Angra do Heroísmo** and **Braga** into a leakage-free **Adverse Outdoor Event** target (rain or strong wind). We then evaluate five incremental classifiers under a prequential (**PREDICT → EVALUATE → LEARN**) protocol and use the evidence to support a **business recommendation** and a **technical recommendation**.

## 📋 Assignment Card

| Field | Value |
|---|---|
| Assignment code | `PA48` |
| Group | `GroupYY` |
| City A | Angra do Heroísmo (Terceira, Azores) — 38.6539, -27.2184 |
| City B | Braga (mainland, North) — 41.5514, -8.4231 |
| Business context | Outdoor Running Event Organiser |
| Official target | **T6 — Adverse Outdoor Event** |
| Business question | *Which city offers more reliable weather windows for recurring outdoor races?* |
| Historical period | 2024-01-01 → 2024-12-31, hourly, UTC (see `config.yaml`) |
| Warm-up | first 720 hours (30 days) of each stream |

## 🎯 Objective

The study supports two separate decisions:

- **Business recommendation**: which city offers more reliable weather windows for recurring outdoor races, judged on how often, how persistently and when adverse rain-or-wind conditions occur.
- **Technical recommendation**: which streaming approach is most appropriate for continuously predicting or monitoring adverse outdoor conditions, considering event detection, class imbalance, predictive metrics, and prediction/update cost.

The weather analysis covers only the weather-related operational dimension. It does not account for costs, participant demand, accessibility, travel logistics or other factors relevant to a real race-location decision.

## 🧪 Methodology

### Data
Hourly historical data from the [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api): `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`. Both streams are retrieved in **UTC**. This keeps the two cities aligned in time (the Azores run one hour behind mainland Portugal) and avoids daylight-saving duplicates or gaps. Raw snapshots are stored in `data/raw/`.

### Target — T6 Adverse Outdoor Event

$y_t = 1 \iff P_t \ge 0.1\ \text{mm} \;\lor\; W_t \ge \tau_W, \qquad \tau_W = Q_{0.80}(W)\ \text{from pooled warm-up}$

- **Rain component (T1)**: fixed threshold of 0.1 mm.
- **Strong-wind component (T5)**: τ<sub>W</sub> is the 80th percentile of wind speed over the **pooled warm-up observations of both cities**. It is estimated **once** and **frozen** before evaluation.
- Because `precipitation(t)` and `wind_speed(t)` define the target, neither is used as a predictor at time *t*.

### Feature representations
| Representation | Content |
|---|---|
| **A — Previous state** | `temperature(t-1)`, `relative_humidity(t-1)`, `precipitation(t-1)`, `wind_speed(t-1)` |
| **B — Stream-aware context** | Representation A + deltas (`x(t-1) − x(t-2)`), rolling mean and std (6 h / 24 h), rainy hours and past adverse-event rate in recent windows |

Every feature uses only observations up to `t-1`.

### Models (`river`)
| # | Model | Role |
|---|---|---|
| 1 | Gaussian Naive Bayes | Probabilistic baseline |
| 2 | Online Logistic Regression (with online scaling) | Incremental linear learner |
| 3 | Hoeffding Tree | Incremental tree |
| 4 | Online Bagging (Hoeffding Trees) | Online-resampling ensemble |
| 5 | Adaptive Random Forest | Adaptive ensemble of stream trees |

**Reference baselines (optional, Guidelines 10.1 — they do not replace the five mandatory models):**

| Baseline | Learning regime | Role |
|---|---|---|
| Majority (`dummy.PriorClassifier`) | Online: updates class counts | "Better than what?" floor |
| Persistence (`dummy.NoChangeClassifier`) | Online: predicts the last revealed label y(t-1) | Naive temporal floor for a persistent weather event |
| FrozenLogReg | Offline: learns on warm-up only, then predict-only | Same model family as OnlineLogReg, different learning regime |

### Protocol
- Chronological replay only, with no random train/test split
- **PREDICT → EVALUATE → LEARN** for every event
- Warm-up events are learned but not scored
- All models and both representations of a city share the **same evaluation history**
- Fresh model instances for every run, with a fixed random seed

### Metrics
Accuracy, Macro-F1, Precision and Recall (adverse-event class), confusion matrix, and **prediction time / update time per event** and throughput. Every metric is tracked **cumulatively** and over a **sliding window of the last 168 evaluated predictions (1 week)**, using an incremental confusion matrix. At the end of each run the cumulative metrics are checked against a batch scikit-learn computation. See `docs/course_alignment.md` for the mapping to Week 4 and the Guidelines.

## 📁 Repository Structure

```
real-time-weather-data/
├── README.md
├── config.yaml              # PA48 card, cities, period, warm-up, seed
├── requirements.txt
├── docs/course_alignment.md  # Week 4 + Guidelines → code mapping
├── data/
│   └── raw/                 # Open-Meteo snapshots (cached CSVs)
├── notebooks/               # city-level EDA and figures
├── results/
│   ├── metrics.csv
│   ├── configurations.csv
│   ├── history.csv.gz       # cumulative + window metrics per evaluated event
│   └── figures/
└── src/
    ├── practical_assignment.py   # main experiment
    ├── data_loader.py            # retrieval + data-quality checks
    ├── targets.py                # T1–T6 + frozen warm-up thresholds
    ├── features.py               # Representations A and B (leakage-free)
    ├── models.py                 # five mandatory learners
    └── prequential.py            # predict → evaluate → learn loop
    ├── plots.py                  # performance, confusion-matrix and cost figures
```

## ⚙️ Setup & Reproduction

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/practical_assignment.py --config config.yaml
```

The script downloads (or reuses cached) data for both cities and prints the data-quality reports and the frozen wind threshold. It then writes `results/metrics.csv`, `results/configurations.csv`, `results/history.csv.gz` and the figures in `results/figures/`.

## 🔁 Reproducibility

| Item | Value |
|---|---|
| Python | *fill in (e.g. 3.11)* |
| Package versions | recorded automatically in `results/configurations.csv` |
| Random seed | `42` |
| Historical period | 2024-01-01 → 2024-12-31 (UTC) |

## ✅ Progress

- [ ] Register pair and PA48 in Moodle
- [x] Fill in `config.yaml` with the assignment card
- [ ] Data-quality inspection — Angra do Heroísmo / Braga
- [ ] Stream characterisation (statistics, temporal evolution, windows, extremes)
- [ ] T6 target construction and class distribution
- [ ] Prequential experiment: 2 cities × 2 representations × 5 models
- [ ] ≥ 2 original analyses / visualisations
- [ ] Report (6–8 pages) + AI Assistance Statement
- [ ] `PA48_GroupYY_Report.pdf` + `PA48_GroupYY_Audit.zip`

## 👥 Team

| Name | Student number | Primary city |
|---|---|---|
| Beatriz Marques | 20231605 | Angra do Heroísmo (City A) |
| Duarte Fernandes | 20231619 | Braga (City B) |
