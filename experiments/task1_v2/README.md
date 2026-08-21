# `task1_v2`

## Results

| ckpt | masking | pooling | norm vol | norm test vol | AUROC | 95% CI | selected C | ‖w‖ | time |
|---|---|---|---|---|---|---|---|---|---|
| pt-full | mean | mean | False | False | **0.990** | 0.944 – 1.000 | 0.0001 | 0.010 | 108s |
| pt-full | zero | mean | False | False | **0.990** | 0.944 – 1.000 | 0.0001 | 0.010 | 112s |
| pt-full | zero | mean | False | True | **0.962** | 0.870 – 1.000 | 0.0001 | 0.010 | 112s |
| pt-full | zero | mean | True | False | **0.971** | 0.894 – 1.000 | 0.0001 | 0.010 | 111s |
| pt-full | zero | mean | True | True | **0.971** | 0.894 – 1.000 | 0.0001 | 0.010 | 64s |
| pt-full | zero | local | False | False | **0.817** | 0.587 – 1.000 | 0.3594 | 1.127 | 102s |
| pt-full | zero | local | True | True | **0.769** | 0.531 – 0.969 | 0.3594 | 1.174 | 102s |
| pt-full | zero | ensemble | False | False | **0.981** | 0.911 – 1.000 | 0.0001 / 0.3594 | 0.010 / 1.127 | 112s |
| pt-full | zero | ensemble | True | True | **0.971** | 0.889 – 1.000 | 0.0001 / 0.3594 | 0.010 / 1.174 | 31s |
| walnut-vitl | mean | mean | False | False | **0.894** | 0.731 – 1.000 | 0.0001 | 0.010 | 97s |
| walnut-vitl | zero | mean | False | False | **0.885** | 0.711 – 1.000 | 0.0001 | 0.009 | 99s |
| walnut-vitl | zero | mean | False | True | **0.894** | 0.712 – 1.000 | 0.0001 | 0.009 | 101s |
| walnut-vitl | zero | mean | True | False | **0.894** | 0.694 – 1.000 | 0.0001 | 0.010 | 115s |
| walnut-vitl | zero | mean | True | True | **0.894** | 0.694 – 1.000 | 0.0001 | 0.010 | 53s |
| walnut-vitl | zero | local | False | False | **0.769** | 0.538 – 0.942 | 21.54 | 7.569 | 687s |
| walnut-vitl | zero | local | True | True | **0.952** | 0.816 – 1.000 | 0.0001 | 0.001 | 682s |
| walnut-vitl | zero | ensemble | False | False | **0.904** | 0.750 – 1.000 | 0.0001 / 21.54 | 0.009 / 7.569 | 688s |
| walnut-vitl | zero | ensemble | True | True | **0.990** | 0.942 – 1.000 | 0.0001 / 0.0001 | 0.010 / 0.001 | 453s |

`selected C` and `‖w‖` are the head fit on all 21 subjects; the fold heads are not saved.
The `Cs=10` grid is `logspace(-4, 4, 10)`, so **1e-4 is its floor** — every global head
pins there rather than selecting from inside the grid. Ensemble rows show global / local.

## Figures

```bash
uv run python plot_scores.py    # -> figures/scores.png, one row per run
uv run python plot_compare.py   # -> figures/compare.png
```
