# 🏃 Weather Windows for Outdoor Races — Angra do Heroísmo vs Braga

**Data Stream Mining · Practical Assignment I (PA48) · NOVA IMS**

> An outdoor running event organiser needs reliable weather windows for recurring races. We turn hourly Open-Meteo weather streams for **Angra do Heroísmo** and **Braga** into a leakage-free **Adverse Outdoor Event** target (rain or strong wind). We then evaluate five incremental classifiers under a prequential (**PREDICT → EVALUATE → LEARN**) protocol and use the evidence to support a **business recommendation** and a **technical recommendation**.

## 📋 Assignment Card

| Field | Value |
|---|---|
| Assignment code | `PA48` |
| Group | `GroupYY` — **to fill in** (`config.yaml → assignment.group`) |
| City A | Angra do Heroísmo (Terceira, Azores) — 38.6539, -27.2184 |
| City B | Braga (mainland, North) — 41.5514, -8.4231 |
| Business context | Outdoor Running Event Organiser |
| Official target | **T6 — Adverse Outdoor Event** |
| Business question | *Which city offers more reliable weather windows for recurring outdoor races?* |
| Historical period | 2021-01-01 → 2026-09-30, hourly, UTC (see `config.yaml`) |
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

### Stream characterisation and original analyses
`src/stream_analysis.py` applies **the same definitions to both cities** and writes `results/stream_summary.csv` plus three figures. Extremes are counted and dated, never removed.

| ID | Original analysis (Guidelines 3.2) | Output |
|---|---|---|
| O1 | **Race-window reliability**: share of days whose local 08:00–12:00 window (`config.yaml → analysis.race_window`) has no adverse hour, overall and by month | `city_comparison.png` (a), `stream_summary.csv` |
| O2 | **Adverse / clear spell persistence**: run-length distributions, survival curves and $P(y_t=1\mid y_{t-1}=1)$ | `city_comparison.png` (b), `city_*_stream_profile.png` (d) |
| O3 | **When and why**: event rate by local hour and by month, split by source (rain only / wind only / both) | `city_*_stream_profile.png` (b, c) |
| O4 | **Representation effect vs cost**: ΔMacro-F1 and Δrecall (Rep B − Rep A) next to the cost multiplier | `model_evidence.png` |

Local hours use `Atlantic/Azores` (Angra) and `Europe/Lisbon` (Braga) for description only; the stream and all features stay in UTC.

### Metrics
Accuracy, Macro-F1, Precision and Recall (adverse-event class), confusion matrix, and **prediction time / update time per event** and throughput. Every metric is tracked **cumulatively** and over a **sliding window of the last 168 evaluated predictions (1 week)**, using an incremental confusion matrix. At the end of each run the cumulative metrics are checked against a batch scikit-learn computation. Each module's docstring states which Week 4–6 concept and Guidelines rule it implements.

## 📁 Repository Structure

```
real-time-weather-data/
├── README.md
├── config.yaml              # PA48 card, cities, period, warm-up, seed
├── requirements.txt
├── data/
│   └── raw/                 # Open-Meteo snapshots (cached CSVs)
├── notebooks/               # 01 data & quality · 02 target & features · 03 prequential · 04 results & original analyses
├── results/
│   ├── metrics.csv
│   ├── configurations.csv
│   ├── history.csv.gz       # cumulative + window metrics per evaluated event
│   ├── stream_summary.csv   # descriptive / event / persistence statistics per city
│   └── figures/
└── src/
    ├── practical_assignment.py   # main experiment
    ├── data_loader.py            # retrieval + data-quality checks
    ├── targets.py                # T1–T6 read from config.yaml + frozen warm-up thresholds
    ├── features.py               # Representations A and B (leakage-free)
    ├── models.py                 # five mandatory learners + baselines + complexity
    ├── prequential.py            # predict → evaluate → learn loop
    ├── plots.py                  # performance, confusion-matrix, cost and trade-off figures
    └── stream_analysis.py        # city profiles, stream_summary.csv, original analyses O1–O4
```

## ⚙️ Setup & Reproduction

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/practical_assignment.py --config config.yaml
```

The script downloads (or reuses cached) data for both cities and prints the data-quality reports and the frozen wind threshold. It then writes `results/stream_summary.csv`, `results/metrics.csv`, `results/configurations.csv`, `results/history.csv.gz` and the eleven figures in `results/figures/`. Paths resolve from the repository root, so the command also works from inside `src/`. A full run takes roughly 10–15 minutes (ARF and Online Bagging are the slowest learners).

Notebooks `01`–`04` reproduce the same steps one at a time; run them in order from the `notebooks/` folder.

## 📦 Audit package

The audit ZIP follows Guidelines 15.1: `README.md`, `config.yaml`, `requirements.txt`, `src/`, `notebooks/`, `data/raw/` (cached Open-Meteo snapshots) and `results/` (`metrics.csv`, `configurations.csv`, `stream_summary.csv`, `history.csv.gz`, `figures/`). Do not include virtual environments or `guidelines.pdf`.

## 🔁 Reproducibility

| Item | Value |
|---|---|
| Python | 3.13 (recorded in `results/configurations.csv`) |
| Key packages | `river 0.26.1`, `scikit-learn`, `pandas`, `numpy`, `matplotlib` — exact versions recorded in `results/configurations.csv` |
| Frozen threshold | τ<sub>W</sub> = 26.9 km/h (pooled warm-up Q0.80 of wind speed) |
| Evaluated events | 49,656 per city (identical for every model and representation) |
| Random seed | `42` |
| Historical period | 2021-01-01 → 2026-09-30 (UTC) |

## ✅ Progress

- [x] Register pair and PA48 in Moodle
- [x] Fill in `config.yaml` with the assignment card
- [x] Data-quality inspection — Angra do Heroísmo / Braga
- [x] Stream characterisation (statistics, temporal evolution, windows, extremes)
- [x] T6 target construction and class distribution
- [x] Prequential experiment: 2 cities × 2 representations × 5 models (+ 3 reference baselines)
- [x] ≥ 2 original analyses / visualisations (O1–O4)
- [ ] Interpretation: notebook *Insights* cells and report discussion — **students only** (Guidelines Ch. 16)
- [ ] Report (6–8 pages) + AI Assistance Statement
- [ ] Group number in `config.yaml`, then `PA48_GroupYY_Report.pdf` + `PA48_GroupYY_Audit.zip`

## 👥 Team

| Name | Student number | Primary city |
|---|---|---|
| Beatriz Marques | 20231605 | Angra do Heroísmo (City A) |
| Duarte Fernandes | 20231619 | Braga (City B) |
