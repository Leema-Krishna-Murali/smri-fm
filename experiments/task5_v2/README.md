# `task5_v2`

Task 5's 0.984 was the field of view, not the anatomy (`experiments/explore_fomo_task5`). This
strips with SynthSeg and crops to a common 133mm AP slab, over three arms and both checkpoints.

```bash
sbatch experiments/task5_v2/launch.sh                                   # 6 runs, ~80s each
uv run python experiments/task5_v2/verify_crop.py \
    | tee experiments/task5_v2/output/verify_crop.log                   # the gate, before scoring
uv run python experiments/task5_v2/collect.py                           # the table below
srun --jobid=<JOBID> --overlap uv run --no-sync python \
    experiments/task5_v2/dump_embeddings.py                             # -> output/embeddings/
uv run python experiments/task5_v2/compare_embeddings.py \
    | tee experiments/task5_v2/output/compare_embeddings.log
```

## Results

`verify_crop.py` gates this: the coverage scalars fall from LOO AUROC **0.976 to 0.450**, so a
drop below is the confound leaving. The floor is **0.833**, six scalars and no backbone.

| ckpt | crop train | crop test | AUROC | 95% CI | rho(p, edge) in y | AUROC \| anatomy | selected C | ‖w‖ | time |
|---|---|---|---|---|---|---|---|---|---|
| pt-full | False | False | **0.981** | 0.941 – 1.000 | +0.23 / +0.17 | 0.849 | 0.005995 | 0.247 | 87s |
| walnut-vitl | False | False | **0.946** | 0.875 – 0.993 | -0.37 / +0.00 | 0.793 | 0.04642 | 0.484 | 87s |
| pt-full | False | True | **0.905** | 0.799 – 0.986 | +0.51 / +0.08 | 0.750 | 0.005995 | 0.247 | 83s |
| walnut-vitl | False | True | **0.884** | 0.773 – 0.969 | -0.30 / +0.18 | 0.733 | 0.04642 | 0.484 | 82s |
| pt-full | True | True | **0.898** | 0.790 – 0.979 | -0.01 / +0.32 | 0.741 | 0.005995 | 0.243 | 81s |
| walnut-vitl | True | True | **0.882** | 0.774 – 0.972 | -0.06 / +0.12 | 0.734 | 0.04642 | 0.501 | 78s |

Cropping the test subjects costs ~0.08; cropping the training subjects too recovers nothing,
against the 0.036 the fold seed alone moves at n=48. Stripping alone did nothing (0.984 -> 0.981).

## Why there was nothing to recover

| | cosine(subject uncropped, same subject cropped) | cosine between different subjects |
|---|---|---|
| pt-full | 0.9996 [0.9980, 1.0000] | 0.9897 |
| walnut-vitl | 0.9991 [0.9965, 1.0000] | 0.9803 |

Removing a quarter of a case's field of view moves its pooled embedding 10-20x less than the
distance to another subject — mean-pooling ~4000 tokens is nearly invariant to the crop.

## The two checkpoints are nearly the same feature space

Top-10 subspace canonical correlations `1.00 1.00 1.00 1.00 0.99 0.99 0.95 0.93 0.83 0.72`, and
the components pair off (PC1 +0.98, PC2 +0.96, PC3 +0.88, PC4 -0.87). Both discriminate on PC4,
pt-full's better at 0.917 against 0.840; the C difference is a tie-break on a flat curve.
