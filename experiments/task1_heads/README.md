# `task1_heads`

Holds the embedding fixed and varies only the probe head, to find out whether
`task1_v2`'s numbers were set by the backbone or by the head.

```bash
sbatch experiments/task1_heads/launch.sh     # mean pooling, no volume normalization
sbatch experiments/task1_heads/launch_1.sh   # walnut, ensemble pooling, norm both
uv run python experiments/task1_heads/collect.py
uv run python experiments/task1_heads/plot_scores.py   # -> figures/scores.png
```

## Results

| ckpt | embedding | head | scaler | AUROC | 95% CI | selected C | ‖w‖ | p spread | time |
|---|---|---|---|---|---|---|---|---|---|
| pt-full | mean / no norm | logistic_cv (roc_auc) | True | **0.990** | 0.944 – 1.000 | 0.0001 | 0.010 | 0.057 | 132s |
| walnut-vitl | mean / no norm | logistic_cv (roc_auc) | True | **0.885** | 0.711 – 1.000 | 0.0001 | 0.009 | 0.060 | 130s |
| walnut-vitl | ensemble / norm both | logistic_cv (roc_auc) | True | **0.990** | 0.942 – 1.000 | 0.0001 / 0.0001 | 0.010 / 0.001 | 0.657 | 424s |
| pt-full | mean / no norm | logistic_cv (roc_auc) | False | **0.856** | 0.615 – 1.000 | 0.0001 | 0.000 | 0.575 | 148s |
| walnut-vitl | mean / no norm | logistic_cv (roc_auc) | False | **0.817** | 0.609 – 0.969 | 2.783 | 3.527 | 0.251 | 146s |
| walnut-vitl | ensemble / norm both | logistic_cv (roc_auc) | False | **0.933** | 0.788 – 1.000 | 21.54 / 21.54 | 10.330 / 7.208 | 0.714 | 588s |
| pt-full | mean / no norm | logistic_cv (neg_log_loss) | True | **0.942** | 0.817 – 1.000 | 0.3594 | 0.533 | 0.996 | 54s |
| walnut-vitl | mean / no norm | logistic_cv (neg_log_loss) | True | **0.808** | 0.593 – 1.000 | 0.04642 | 0.364 | 0.997 | 55s |
| walnut-vitl | ensemble / norm both | logistic_cv (neg_log_loss) | True | **1.000** | 1.000 – 1.000 | 0.04642 / 2.783 | 0.358 / 2.833 | 0.591 | 427s |
| pt-full | mean / no norm | logistic C=0.01 | True | **0.981** | 0.917 – 1.000 | — | 0.216 | 0.873 | 37s |
| walnut-vitl | mean / no norm | logistic C=0.01 | True | **0.885** | 0.714 – 1.000 | — | 0.220 | 0.883 | 35s |
| walnut-vitl | ensemble / norm both | logistic C=0.01 | True | **0.990** | 0.944 – 1.000 | — / — | 0.219 / 0.129 | 0.644 | 211s |
| pt-full | mean / no norm | logistic C=1 | True | **0.962** | 0.867 – 1.000 | — | 0.632 | 0.999 | 24s |
| walnut-vitl | mean / no norm | logistic C=1 | True | **0.856** | 0.673 – 1.000 | — | 0.699 | 0.999 | 26s |
| walnut-vitl | ensemble / norm both | logistic C=1 | True | **1.000** | 1.000 – 1.000 | — / — | 0.678 / 1.915 | 0.606 | 212s |
| pt-full | mean / no norm | logistic C=1 | False | **0.952** | 0.836 – 1.000 | — | 2.165 | 0.592 | 25s |
| walnut-vitl | mean / no norm | logistic C=1 | False | **0.837** | 0.633 – 0.981 | — | 1.712 | 0.349 | 23s |
| walnut-vitl | ensemble / norm both | logistic C=1 | False | **0.904** | 0.744 – 1.000 | — / — | 1.601 / 1.742 | 0.713 | 221s |
| pt-full | mean / no norm | logistic C=100 | True | **0.990** | 0.944 – 1.000 | — | 1.220 | 1.000 | 31s |
| walnut-vitl | mean / no norm | logistic C=100 | True | **0.904** | 0.745 – 1.000 | — | 1.421 | 1.000 | 32s |
| walnut-vitl | ensemble / norm both | logistic C=100 | True | **1.000** | 1.000 – 1.000 | — / — | 1.346 / 7.427 | 0.599 | 205s |
| pt-full | mean / no norm | lda | True | **0.942** | 0.816 – 1.000 | — | 34.563 | 1.000 | 419s |
| walnut-vitl | mean / no norm | lda | True | **0.769** | 0.558 – 0.950 | — | 28.721 | 1.000 | 421s |
| walnut-vitl | ensemble / norm both | lda | True | **1.000** | 1.000 – 1.000 | — / — | 31.067 / 5.518 | 0.630 | 553s |
| pt-full | mean / no norm | lda | False | **0.942** | 0.816 – 1.000 | — | 1757.192 | 1.000 | 417s |
| walnut-vitl | mean / no norm | lda | False | **0.769** | 0.558 – 0.950 | — | 3025.046 | 1.000 | 414s |
| walnut-vitl | ensemble / norm both | lda | False | **1.000** | 1.000 – 1.000 | — / — | 3329.697 / 194.395 | 0.630 | 307s |

`selected C` and `‖w‖` are the head fit on all 21 subjects; the fold heads are not saved.
Ensemble rows show global / local, the two probes it fits from the same config.
