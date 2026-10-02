"""Prequential evaluation: PREDICT -> EVALUATE -> LEARN."""
import time

from river import metrics


def run_prequential(model, X, y, is_warmup):
    """Replay the stream in order. Warm-up events are learned but not scored.
    Timing is averaged over evaluated events only."""
    acc, mf1 = metrics.Accuracy(), metrics.MacroF1()
    prec, rec = metrics.Precision(pos_val=1), metrics.Recall(pos_val=1)
    tp = fp = tn = fn = 0
    pred_time = update_time = 0.0
    n_eval = 0

    for x, y_true, warm in zip(X.to_dict("records"), y.tolist(), is_warmup.tolist()):
        if not warm:
            t0 = time.perf_counter()
            y_pred = model.predict_one(x)           # 1. PREDICT
            pred_time += time.perf_counter() - t0
            y_pred = 0 if y_pred is None else int(y_pred)
            for m in (acc, mf1, prec, rec):         # 2. EVALUATE
                m.update(y_true, y_pred)
            tp += int(y_true == 1 and y_pred == 1)
            fp += int(y_true == 0 and y_pred == 1)
            tn += int(y_true == 0 and y_pred == 0)
            fn += int(y_true == 1 and y_pred == 0)
            n_eval += 1

        t0 = time.perf_counter()
        model.learn_one(x, y_true)                  # 3. LEARN
        if not warm:
            update_time += time.perf_counter() - t0

    return {
        "n_eval": n_eval,
        "event_rate": (tp + fn) / n_eval if n_eval else float("nan"),
        "accuracy": acc.get(), "macro_f1": mf1.get(),
        "precision": prec.get(), "recall": rec.get(),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "pred_ms_per_event": 1000 * pred_time / max(n_eval, 1),
        "update_ms_per_event": 1000 * update_time / max(n_eval, 1),
    }
