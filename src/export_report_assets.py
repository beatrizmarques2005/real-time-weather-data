"""Export the report's tables and figures from the saved audit outputs (no interpretation).

Run AFTER `python src/practical_assignment.py`:

    python src/export_report_assets.py

It writes LaTeX table snippets to `report/tables/` and copies the report figures to
`report/figures/`, so every number and figure in the report comes from the submitted run.
Upload (or re-upload) both folders to Overleaf after each clean run.
"""
import ast
import shutil

import pandas as pd

from data_loader import ROOT, load_config

MODELS = ["GaussianNB", "OnlineLogReg", "HoeffdingTree", "OnlineBagging", "ARF"]
LABELS = {"GaussianNB": "Gaussian NB", "OnlineLogReg": "Online Logistic Reg.",
          "HoeffdingTree": "Hoeffding Tree", "OnlineBagging": "Online Bagging",
          "ARF": "Adaptive Random Forest", "Majority": "\\textit{Majority (ref.)}",
          "Persistence": "\\textit{Persistence (ref.)}", "FrozenLogReg": "\\textit{Frozen LogReg (ref.)}"}
FIGURES = {"city_a_stream_profile.png": "cityA_stream_profile.png",
           "city_b_stream_profile.png": "cityB_stream_profile.png",
           "city_comparison.png": "city_comparison.png",
           "model_evidence.png": "model_evidence.png",
           "time_performance_tradeoff.png": "time_performance_tradeoff.png",
           "confusion_matrices.png": "confusion_matrices.png",
           "perf_window_macro_f1.png": "perf_window_macro_f1.png"}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}\\%"


def num(x, d=1):
    return f"{float(x):,.{d}f}"


def write(path, text):
    text = text.replace("Angra do Heroismo", "Angra do Hero\\'ismo")   # LaTeX accent
    path.write_text(text.strip() + "\n", encoding="utf-8")


def results_table(metrics, city):
    rows = []
    for rep in ("A", "B"):
        part = metrics[(metrics.city == city) & (metrics.representation == rep)].set_index("model")
        names = MODELS + [m for m in ("Majority", "Persistence") if m in part.index]
        for i, model in enumerate(names):
            r = part.loc[model]
            sep = "\\addlinespace[2pt]\n" if model == "Majority" else ""
            rows.append(f"{sep}{rep if i == 0 else ''} & {LABELS[model]} & {r.accuracy:.3f} & "
                        f"{r.macro_f1:.3f} & {r.precision:.3f} & {r.recall:.3f} \\\\")
        if rep == "A":
            rows.append("\\midrule")
    return "\n".join(rows)


def cost_table(metrics, cities):
    rows = []
    for rep in ("A", "B"):
        for i, model in enumerate(MODELS):
            cells = []
            for city in cities:
                r = metrics[(metrics.city == city) & (metrics.representation == rep) &
                            (metrics.model == model)].iloc[0]
                cells += [f"{r.pred_ms_per_event:.3f}", f"{r.update_ms_per_event:.3f}"]
            rows.append(f"{rep if i == 0 else ''} & {LABELS[model]} & " + " & ".join(cells) + " \\\\")
        if rep == "A":
            rows.append("\\midrule")
    return "\n".join(rows)


def confusion_table(metrics, cities):
    rows = []
    for city in cities:
        for rep in ("A", "B"):
            for i, model in enumerate(MODELS):
                r = metrics[(metrics.city == city) & (metrics.representation == rep) &
                            (metrics.model == model)].iloc[0]
                lead = f"{city} & {rep}" if i == 0 else " & "
                rows.append(f"{lead} & {LABELS[model]} & {int(r.tn):,} & {int(r.fp):,} & "
                            f"{int(r.fn):,} & {int(r.tp):,} \\\\")
            rows.append("\\addlinespace[2pt]")
        rows.append("\\midrule")
    return "\n".join(rows[:-1])


def complexity_table(metrics, cities):
    rows = []
    for city in cities:
        for rep in ("A", "B"):
            for model in ("HoeffdingTree", "OnlineBagging", "ARF"):
                r = metrics[(metrics.city == city) & (metrics.representation == rep) &
                            (metrics.model == model)].iloc[0]
                drifts = "--" if pd.isna(r.drifts_detected) else f"{int(r.drifts_detected)}"
                warns = "--" if pd.isna(r.warnings_detected) else f"{int(r.warnings_detected)}"
                rows.append(f"{city} & {rep} & {LABELS[model]} & {int(r.total_tree_nodes):,} & "
                            f"{r.mean_tree_height:.1f} & {warns} & {drifts} \\\\")
    return "\n".join(rows)


def city_profile_table(s):
    lines = [
        ("Hourly observations (evaluated after warm-up)", f"{int(s.n_obs):,} ({{\\EvalEvents}})",
         "Stream length; identical evaluation history for every model"),
        ("Missing / duplicated / irregular / invalid", f"{int(s.missing_values)} / {int(s.duplicated_timestamps)} / "
         f"{int(s.irregular_steps)} / {int(s.invalid_values)}", "Data quality; no observation removed"),
        ("T6 positive-event rate (all hours)", pct(s.event_rate), "Share of hours exposed to rain or strong wind"),
        ("Rain-hour rate / strong-wind-hour rate", f"{pct(s.rain_hour_rate)} / {pct(s.strong_wind_hour_rate)}",
         "The two components of T6"),
        ("Events: rain only / wind only / both", f"{pct(s.share_rain_only, 0)} / {pct(s.share_wind_only, 0)} / "
         f"{pct(s.share_both, 0)}", "Source of adverse hours"),
        ("Wind speed mean $\\pm$ sd (km/h); P95", f"{num(s.wind_mean)} $\\pm$ {num(s.wind_std)}; {num(s.wind_p95)}",
         "Wind regime relative to $\\tau_W$"),
        ("Persistence $P(y_t{=}1 \\mid y_{t-1}{=}1)$", f"{s.persistence_p11:.3f}", "Probability an adverse hour is followed by another"),
        ("Adverse spell: mean / median / max (h)", f"{num(s.adverse_spell_mean_h)} / {num(s.adverse_spell_median_h, 0)} / "
         f"{int(s.adverse_spell_max_h)}", "How long adverse periods last"),
        ("Clear spell: mean / median / max (h)", f"{num(s.clear_spell_mean_h)} / {num(s.clear_spell_median_h, 0)} / "
         f"{int(s.clear_spell_max_h)}", "How long usable periods last"),
        ("Extremes: hours wind $\\ge$ 40 km/h / rain $\\ge$ 5 mm", f"{int(s.hours_wind_ge_extreme)} / {int(s.hours_precip_ge_extreme)}",
         "Valid extremes, retained"),
    ]
    return "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in lines)


def comparison_table(summary, cities):
    a, b = (summary[c] for c in cities)
    def mon(s, which):
        return f"{MONTHS[int(s[f'race_window_{which}_month']) - 1]} ({pct(s[f'race_window_{which}_value'], 0)})"
    rows = [
        ("T6 positive-event rate", pct(a.event_rate), pct(b.event_rate)),
        ("Event rate in warm-up (first 720 h)", pct(a.warmup_event_rate), pct(b.warmup_event_rate)),
        ("Yearly event rate (min--max)", f"{pct(a.yearly_event_rate_min)}--{pct(a.yearly_event_rate_max)}",
         f"{pct(b.yearly_event_rate_min)}--{pct(b.yearly_event_rate_max)}"),
        ("Days with no adverse hour", pct(a.event_free_day_share), pct(b.event_free_day_share)),
        ("Reliable race windows (08--12 h local)$^{\\dagger}$", pct(a.race_window_reliability), pct(b.race_window_reliability)),
        ("Best / worst month for race windows$^{\\dagger}$", f"{mon(a, 'best')} / {mon(a, 'worst')}",
         f"{mon(b, 'best')} / {mon(b, 'worst')}"),
        ("24-h windows fully clear / fully adverse", f"{pct(a.share_24h_windows_all_clear)} / {pct(a.share_24h_windows_all_adverse)}",
         f"{pct(b.share_24h_windows_all_clear)} / {pct(b.share_24h_windows_all_adverse)}"),
        ("Adverse spells $\\ge$ 24 h (count)$^{\\dagger}$", f"{int(a.adverse_spells_ge_24h)}", f"{int(b.adverse_spells_ge_24h)}"),
        ("Mean clear spell (h)$^{\\dagger}$", num(a.clear_spell_mean_h), num(b.clear_spell_mean_h)),
        ("Precipitation (mm per year)", num(a.precip_total_mm_per_year, 0), num(b.precip_total_mm_per_year, 0)),
    ]
    return "\n".join(f"{r[0]} & {r[1]} & {r[2]} \\\\" for r in rows)


def main():
    cfg = load_config(ROOT / "config.yaml")
    res = ROOT / cfg["paths"]["results"]
    rep = ROOT / "report"
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(res / "metrics.csv")
    summary = pd.read_csv(res / "stream_summary.csv", index_col=0)
    conf = pd.read_csv(res / "configurations.csv", index_col=0)["value"]
    cities = [cfg["cities"]["city_a"]["name"], cfg["cities"]["city_b"]["name"]]
    # stream_summary.csv mixes numbers and timestamps; convert the numeric rows only.
    summary = summary.apply(lambda row: pd.to_numeric(row, errors="coerce")
                            if pd.to_numeric(row, errors="coerce").notna().all() else row, axis=1)
    thresholds = ast.literal_eval(conf["thresholds"])
    n_eval = int(metrics.n_eval.iloc[0])

    write(rep / "tables" / "run_values.tex", "\n".join([
        f"\\newcommand{{\\TauW}}{{{thresholds['T5']:.1f}}}",
        f"\\newcommand{{\\EvalEvents}}{{{n_eval:,}}}",
        f"\\newcommand{{\\PythonVersion}}{{{conf['python']}}}",
        f"\\newcommand{{\\RiverVersion}}{{{conf['river']}}}",
        f"\\newcommand{{\\SklearnVersion}}{{{conf['scikit-learn']}}}",
        f"\\newcommand{{\\PandasVersion}}{{{conf['pandas']}}}",
        f"\\newcommand{{\\NumpyVersion}}{{{conf['numpy']}}}",
        f"\\newcommand{{\\RunPlatform}}{{{conf.get('platform', 'not recorded').replace('_', '-')}}}",
    ]))
    for key, city in zip(("city_a", "city_b"), cities):
        write(rep / "tables" / f"{key}_profile.tex", city_profile_table(summary[city]))
        write(rep / "tables" / f"{key}_results.tex", results_table(metrics, city))
    write(rep / "tables" / "comparison.tex", comparison_table(summary, cities))
    write(rep / "tables" / "cost.tex", cost_table(metrics, cities))
    write(rep / "tables" / "confusion.tex", confusion_table(metrics, cities))
    write(rep / "tables" / "complexity.tex", complexity_table(metrics, cities))
    for src, dst in FIGURES.items():
        shutil.copy(res / "figures" / src, rep / "figures" / dst)
    print(f"Report assets written to {rep}/tables and {rep}/figures")


if __name__ == "__main__":
    main()
