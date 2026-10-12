"""Test-then-train evaluation with exact cumulative and recent confusion matrices.

Course alignment (W04, Guidelines Ch. 11):
- Prequential protocol: PREDICT -> EVALUATE -> LEARN, one event at a time, in chronological order.
- Warm-up events are learned but not scored; frozen models stop learning after warm-up.
- Metrics are state: an incremental confusion matrix (update/remove) instead of batch recomputation.
- Cumulative vs recent performance: exact last-N window backed by a deque.
- "Accuracy is not the whole story": Macro-F1 (mean of both class F1s), event-class precision/recall
  and confusion counts are all reported.
- Computational cost: prediction and update time per evaluated event, throughput, optional
  tracemalloc peak memory (timings are environment-dependent).
- W04 Challenge 03 checkpoints 5 and 7: confusion counts sum to n_eval, and final metrics are
  asserted equal to batch scikit-learn values (tolerance 1e-9).
"""
from collections import deque
import time
import tracemalloc

import pandas as pd
from sklearn import metrics as sk_metrics


class ConfusionMatrix:
    """Binary confusion matrix; rows are true labels, columns predictions."""

    def __init__(self):
        self.counts = [[0, 0], [0, 0]]

    def update(self, y_true, y_pred):
        assert y_true in (0, 1) and y_pred in (0, 1)
        self.counts[y_true][y_pred] += 1

    def remove(self, y_true, y_pred):
        assert self.counts[y_true][y_pred] > 0
        self.counts[y_true][y_pred] -= 1

    @property
    def n(self):
        return sum(map(sum, self.counts))

    @property
    def tp(self):
        return self.counts[1][1]

    @property
    def fp(self):
        return self.counts[0][1]

    @property
    def tn(self):
        return self.counts[0][0]

    @property
    def fn(self):
        return self.counts[1][0]

    @staticmethod
    def _ratio(num, den):
        return num / den if den else 0.0

    def accuracy(self):
        return self._ratio(self.tp + self.tn, self.n)

    def precision(self):
        return self._ratio(self.tp, self.tp + self.fp)

    def recall(self):
        return self._ratio(self.tp, self.tp + self.fn)

    def f1(self, cls):
        tp = self.counts[cls][cls]
        fp = self.counts[1 - cls][cls]
        fn = self.counts[cls][1 - cls]
        return self._ratio(2 * tp, 2 * tp + fp + fn)

    def macro_f1(self):
        return (self.f1(0) + self.f1(1)) / 2


class StreamingEvaluator:
    """Incremental cumulative and exact last-window performance state."""

    def __init__(self, window_size):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.cumulative = ConfusionMatrix()
        self.window = ConfusionMatrix()
        self.pairs = deque()
        self.window_size = window_size

    def update(self, y_true, y_pred):
        self.cumulative.update(y_true, y_pred)
        if len(self.pairs) == self.window_size:
            old_true, old_pred = self.pairs.popleft()
            self.window.remove(old_true, old_pred)
        self.pairs.append((y_true, y_pred))
        self.window.update(y_true, y_pred)


def run_prequential(model, X, y, is_warmup, online=True, window_size=168,
                    timestamps=None, track_memory=False):
    """Score only after warm-up; predict, score, then optionally learn each event."""
    if not (len(X) == len(y) == len(is_warmup)):
        raise ValueError("X, y and warm-up masks must have identical lengths")
    if timestamps is not None and len(timestamps) != len(y):
        raise ValueError("timestamps and y must have identical lengths")
    evaluator = StreamingEvaluator(window_size)
    history, truths, predictions = [], [], []
    pred_time = update_time = total_seconds = 0.0
    cold_starts = 0
    tracking = False
    peak_memory_mb = float("nan")
    records = X.to_dict("records")
    times = [None] * len(y) if timestamps is None else timestamps
    try:
        for x, y_true, warm, timestamp in zip(records, y, is_warmup, times):
            y_true = int(y_true)
            if not warm:
                if track_memory and not tracking:
                    tracemalloc.start()
                    tracking = True
                event_start = time.perf_counter()
                t0 = time.perf_counter()
                y_pred = model.predict_one(x)  # 1. PREDICT
                pred_time += time.perf_counter() - t0
                if y_pred is None:
                    cold_starts += 1
                    y_pred = 0
                y_pred = int(y_pred)
                evaluator.update(y_true, y_pred)  # 2. EVALUATE
                truths.append(y_true)
                predictions.append(y_pred)
                cum, win = evaluator.cumulative, evaluator.window
                row = {"step": cum.n,
                       "cumulative_accuracy": cum.accuracy(),
                       "cumulative_macro_f1": cum.macro_f1(),
                       "cumulative_recall": cum.recall(),
                       "cumulative_precision": cum.precision(),
                       "window_accuracy": win.accuracy(),
                       "window_macro_f1": win.macro_f1(),
                       "window_recall": win.recall(),
                       "window_precision": win.precision()}
                if timestamps is not None:
                    row["timestamp"] = timestamp
                history.append(row)

            if warm or online:  # 3. LEARN: frozen models learn only on warm-up
                t0 = time.perf_counter()
                model.learn_one(x, y_true)
                if not warm:
                    update_time += time.perf_counter() - t0
            if not warm:
                total_seconds += time.perf_counter() - event_start
    finally:
        if tracking:
            _, peak = tracemalloc.get_traced_memory()
            peak_memory_mb = peak / (1024 ** 2)
            tracemalloc.stop()

    cm = evaluator.cumulative
    n_eval = cm.n
    # Challenge 03 checkpoint 5: counts add up; checkpoint 7: match batch scikit-learn.
    assert cm.tp + cm.fp + cm.tn + cm.fn == n_eval
    if n_eval:
        expected = {
            "accuracy": sk_metrics.accuracy_score(truths, predictions),
            "macro_f1": sk_metrics.f1_score(truths, predictions, average="macro",
                                              labels=[0, 1], zero_division=0),
            "precision": sk_metrics.precision_score(truths, predictions, pos_label=1,
                                                     labels=[0, 1], zero_division=0),
            "recall": sk_metrics.recall_score(truths, predictions, pos_label=1,
                                               labels=[0, 1], zero_division=0),
        }
        for metric, batch in expected.items():
            assert abs(getattr(cm, metric)() - batch) < 1e-9, metric

    summary = {
        "n_eval": n_eval, "event_rate": (cm.tp + cm.fn) / n_eval if n_eval else float("nan"),
        "accuracy": cm.accuracy(), "macro_f1": cm.macro_f1(),
        "precision": cm.precision(), "recall": cm.recall(),
        "tp": cm.tp, "fp": cm.fp, "tn": cm.tn, "fn": cm.fn,
        "window_accuracy_final": evaluator.window.accuracy(),
        "window_macro_f1_final": evaluator.window.macro_f1(),
        "cold_starts": cold_starts,
        "pred_ms_per_event": 1000 * pred_time / n_eval if n_eval else 0.0,
        "update_ms_per_event": 1000 * update_time / n_eval if n_eval else 0.0,
        "total_seconds": total_seconds,
        "events_per_second": n_eval / total_seconds if total_seconds else 0.0,
        "peak_memory_mb": peak_memory_mb,
    }
    return summary, pd.DataFrame(history)
