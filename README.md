# 🌦️ Real-Time Weather Data — Business-Oriented Data Stream Mining

**Data Stream Mining · Practical Assignment I · NOVA IMS**

> From real-time weather data to operational decisions. We turn an Open-Meteo weather stream for two Portuguese cities into a leakage-free binary operational event, evaluate five incremental classifiers under a prequential (**PREDICT → EVALUATE → LEARN**) protocol, and use the evidence to support a **business recommendation** and a **technical recommendation**.

## 📋 Assignment Card

| Field | Value |
|---|---|
| Assignment code | `PAxx` *(to confirm in Moodle)* |
| Group | `GroupYY` |
| City A | *TBD* |
| City B | *TBD* |
| Business context | *TBD* |
| Official target | *T1–T6 — TBD* |
| Business question | *TBD* |
| Historical period | *YYYY-MM-DD → YYYY-MM-DD* (hourly) |
| Warm-up | first *N* hours of each stream (see `config.yaml`) |

## 🎯 Objective

The study supports two separate decisions:

- **Business recommendation** — which city appears more suitable from the weather-related operational perspective defined in our assignment card.
- **Technical recommendation** — which streaming approach is most appropriate for continuous event prediction or monitoring, considering event detection, class imbalance, predictive metrics, and prediction/update cost.

The weather analysis covers only the operational dimension stated on the card; it does not account for rent, wages, infrastructure or other location factors.

## 🧪 Methodology

### Data
Hourly historical data from the [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api): `temperature_2m`, `relative_humidity_2m`, `precipitation`, `wind_speed_10m`. Data is retrieved in **UTC** so the chronology has no daylight-saving duplicates or gaps. Raw snapshots are stored in `data/raw/`.

### Target
The target follows the official definition on our card (T1–T6). Data-derived thresholds (T2–T5, and the wind part of T6) are computed **once** from the **pooled warm-up observations of both cities**, then **frozen** before evaluation.

### Feature representations
| Representation | Content |
|---|---|
| **A — Previous state** | `temperature(t-1)`, `relative_humidity(t-1)`, `precipitation(t-1)`, `wind_speed(t-1)` |
| **B — Stream-aware context** | Representation A + deltas (`x(t-1) − x(t-2)`), rolling mean and std (6 h / 24 h), rainy hours and past event rate in recent windows |

Every feature uses only observations up to `t-1`. No current target-defining value is ever used as a predictor.

### Models (`river`)
| # | Model | Role |
|---|---|---|
| 1 | Gaussian Naive Bayes | Probabilistic baseline |
| 2 | Online Logistic Regression (with online scaling) | Incremental linear learner |
| 3 | Hoeffding Tree | Incremental tree |
| 4 | Online Bagging (Hoeffding Trees) | Online-resampling ensemble |
| 5 | Adaptive Random Forest | Adaptive ensemble of stream trees |

### Protocol
- Chronological replay only, with no random train/test split
- **PREDICT → EVALUATE → LEARN** for every event
- Warm-up events are learned but not scored
- All models and both representations of a city share the **same evaluation history**
- Fresh model instances for every run, with a fixed random seed

### Metrics
Accuracy, Macro-F1, Precision and Recall (event class), confusion matrix, and **prediction time / update time per event**.

## 📁 Repository Structure

```
real-time-weather-data/
├── README.md
├── config.yaml              # assignment card, cities, period, warm-up, seed
├── requirements.txt
├── data/
│   └── raw/                 # Open-Meteo snapshots (cached CSVs)
├── notebooks/               # city-level EDA and figures
├── results/
│   ├── metrics.csv
│   ├── configurations.csv
│   └── figures/
└── src/
    ├── practical_assignment.py   # main experiment
    ├── data_loader.py            # retrieval + data-quality checks
    ├── targets.py                # T1–T6 + frozen warm-up thresholds
    ├── features.py               # Representations A and B (leakage-free)
    ├── models.py                 # five mandatory learners
    └── prequential.py            # predict → evaluate → learn loop
```

## ⚙️ Setup & Reproduction

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. fill in config.yaml with the assignment card (cities, coordinates, target, period)
# 2. run the full experiment
python src/practical_assignment.py --config config.yaml
```

The script downloads (or reuses cached) data, prints data-quality reports and the frozen thresholds, then writes `results/metrics.csv` and `results/configurations.csv`.

## 🔁 Reproducibility

| Item | Value |
|---|---|
| Python | *fill in (e.g. 3.11)* |
| Package versions | recorded automatically in `results/configurations.csv` |
| Random seed | `42` |
| Historical period | see `config.yaml` |

## ✅ Progress

- [ ] Register pair and PA code in Moodle
- [ ] Fill in `config.yaml` with the assignment card
- [ ] Data-quality inspection — City A / City B
- [ ] Stream characterisation (statistics, temporal evolution, windows, extremes)
- [ ] Target construction and class distribution
- [ ] Prequential experiment: 2 cities × 2 representations × 5 models
- [ ] ≥ 2 original analyses / visualisations
- [ ] Report (6–8 pages) + AI Assistance Statement
- [ ] `PAxx_GroupYY_Report.pdf` + `PAxx_GroupYY_Audit.zip`

## 👥 Team

| Name | Student number | Primary city |
|---|---|---|
| Beatriz Marques | *TBD* | City A |
| *Partner name* | *TBD* | City B |
