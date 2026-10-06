"""Descriptive performance and cost figures (no interpretation)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("accuracy", "macro_f1", "recall", "precision")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _date_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.set_ylim(-0.04, 1.04)
    ax.grid(alpha=0.25)


def _lines(ax, subset, column, names, colors):
    for name in names:
        part = subset[subset.model == name].iloc[::24]  # plot every 24th hour; full history is saved
        if part.empty:
            continue
        ax.plot(part.timestamp, part[column], label=name, color=colors[name],
                linestyle="--" if part.role.iloc[0] == "baseline" else "-", lw=1.2)
    _date_axis(ax)


def generate_plots(history, metrics, output_dir):
    """Write the six assignment figures to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    cities = list(metrics.city.drop_duplicates())
    reps = list(metrics.representation.drop_duplicates())
    names = list(metrics.model.drop_duplicates())
    mandatory = list(metrics.loc[metrics.role == "mandatory", "model"].drop_duplicates())
    colors = dict(zip(names, plt.cm.tab10(np.linspace(0, 0.9, len(names)))))

    for column, filename, title in (
        ("cumulative_macro_f1", "perf_cumulative_macro_f1.png", "Cumulative Macro-F1"),
        ("window_macro_f1", "perf_window_macro_f1.png", "Window Macro-F1"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, layout="constrained")
        for i, city in enumerate(cities):
            for j, rep in enumerate(reps):
                ax = axes[i, j]
                _lines(ax, history[(history.city == city) & (history.representation == rep)],
                       column, names, colors)
                ax.set_title(f"{city} · Rep {rep}")
                ax.set_ylabel("Macro-F1")
                ax.set_xlabel("Date (UTC)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=9)
        fig.suptitle(title)
        _save(fig, output_dir / filename)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, layout="constrained")
    for i, city in enumerate(cities):
        subset = history[(history.city == city) & (history.representation == "B")]
        for j, (column, label) in enumerate((("cumulative_recall", "Cumulative"),
                                              ("window_recall", "Window"))):
            ax = axes[i, j]
            _lines(ax, subset, column, mandatory, colors)
            ax.set_title(f"{city} · Rep B · {label}")
            ax.set_ylabel("Event-class recall")
            ax.set_xlabel("Date (UTC)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=9)
    fig.suptitle("Cumulative vs window event-class recall")
    _save(fig, output_dir / "perf_cumulative_vs_window_recall.png")

    fig, axes = plt.subplots(len(cities), len(mandatory), figsize=(16, 6),
                             layout="constrained")
    for i, city in enumerate(cities):
        for j, name in enumerate(mandatory):
            row = metrics[(metrics.city == city) & (metrics.representation == "B") &
                          (metrics.model == name)].iloc[0]
            matrix = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
            ax = axes[i, j]
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(metrics.n_eval))
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{matrix[a, b]:,}", ha="center", va="center", fontsize=9,
                            color="white" if matrix[a, b] > max(metrics.n_eval) / 2 else "black")
            ax.set_title(f"{city}\n{name}", fontsize=10)
            ax.set_xticks([0, 1], ["0", "1"])
            ax.set_yticks([0, 1], ["0", "1"])
            ax.set_xlabel("Predicted")
            if j == 0:
                ax.set_ylabel("True")
    fig.suptitle("Final confusion matrices · Rep B · mandatory models")
    _save(fig, output_dir / "confusion_matrices.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, layout="constrained")
    for i, city in enumerate(cities):
        for j, rep in enumerate(reps):
            ax = axes[i, j]
            part = metrics[(metrics.city == city) & (metrics.representation == rep)]
            pos = np.arange(len(part))
            ax.barh(pos - 0.19, part.macro_f1, 0.38, label="Macro-F1")
            ax.barh(pos + 0.19, part.recall, 0.38, label="Event recall")
            ax.set_yticks(pos, part.model, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.02)
            ax.set_xlabel("Score")
            ax.set_title(f"{city} · Rep {rep}")
            ax.grid(axis="x", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Final model comparison")
    _save(fig, output_dir / "model_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for i, city in enumerate(cities):
        for j, rep in enumerate(reps):
            ax = axes[i, j]
            part = metrics[(metrics.city == city) & (metrics.representation == rep)]
            pos = np.arange(len(part))
            ax.barh(pos - 0.19, part.pred_ms_per_event, 0.38, label="Predict")
            ax.barh(pos + 0.19, part.update_ms_per_event, 0.38, label="Update")
            ax.set_yticks(pos, part.model, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Milliseconds per evaluated event")
            ax.set_title(f"{city} · Rep {rep}")
            ax.grid(axis="x", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle("Processing cost (frozen update = 0)")
    _save(fig, output_dir / "processing_cost.png")
