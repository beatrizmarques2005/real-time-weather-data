# Week 4 and Assignment 1 alignment

| Concept / requirement | Implementation |
|---|---|
| W04 prequential test-then-train | `src/prequential.py`: predict → evaluate → learn in chronological order. |
| W04 shared warm-up | `src/practical_assignment.py`: same pooled first 720 hours for frozen threshold; identical filtered warm-up mask for all learners; warm-up learned, not scored. |
| W04 metrics as state / incremental confusion matrix | `ConfusionMatrix` in `src/prequential.py`: `update`, `remove`, binary counts and zero-safe metrics. |
| W04 cumulative vs windowed performance | `StreamingEvaluator`: cumulative counts and exact deque-backed last-168-evaluated-predictions counts; `results/history.csv.gz` and three time-series figures. |
| W04 accuracy is not the whole story / Macro-F1 | Both class F1 scores averaged; positive-class recall and precision, accuracy and confusion counts in `results/metrics.csv`. |
| W04 “Better than what?” | `src/models.py`: online `PriorClassifier` majority baseline. |
| W04 prediction ≠ online learning | `FrozenLogReg` and `OnlineLogReg` have the same pipeline; frozen learns during warm-up only, while both keep predicting. Persistence uses the last revealed label. |
| W04 computational cost and throughput | `src/prequential.py`: evaluated-event prediction/update time, total elapsed evaluation-event time, throughput and optional `tracemalloc` peak; `processing_cost.png`. |
| W04 pitfalls: shuffling, leakage, different histories | No shuffle; `src/features.py` shifts all predictors to t−1; pooled thresholds fixed from warm-up; `src/practical_assignment.py` shares the valid-row mask and each run receives a fresh model. |
| W04 Challenge 03 checkpoints 5 and 7 | End-of-run confusion-count assertion and batch scikit-learn checks for accuracy, Macro-F1, precision and recall within 1e−9 in `src/prequential.py`. |
| Guidelines Ch. 8 representations | `src/features.py`: A lag-1 previous state; B adds past-only deltas, rolling summaries and revealed past event rates. |
| Guidelines Ch. 10 models | `src/models.py`: five mandatory incremental learners retained. Baselines are optional reference learners under Guidelines 10.1, not replacements for the five. |
| Guidelines Ch. 11 metrics | `src/prequential.py`, `results/metrics.csv`, `results/history.csv.gz`, `src/plots.py`: confusion counts, accuracy, Macro-F1, positive precision/recall, costs and trajectories. |
| Guidelines Ch. 12 experimental rules 1–10 | `config.yaml` specifies target/cities/period/seed; `src/data_loader.py` records quality, `src/targets.py` freezes pooled warm-up threshold, `src/features.py` prevents future features, `src/practical_assignment.py` enforces chronological shared evaluation, fresh instances and audit output (`results/configurations.csv`). Consult the original Guidelines for the full rule wording. |

The ≥2 original analyses (Guidelines 3.2) and all interpretation remain the students’ work. Figures contain no interpretation.
