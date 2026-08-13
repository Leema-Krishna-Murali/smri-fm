# fomo_tune_walnut_v0_1

Tasks 1, 2, 3 and 5 on the walnut-v0.1 ViT-L checkpoint, everything else identical to
`experiments/fomo_tune_baseline`. Same architecture as the default (`mae_vit_large`, patch 8,
208x240x208), so it is a `ckpt_path` override and nothing else.

`hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth` — 52,643 subjects of
FOMO300K, subject-scaling run at a fixed 99k steps, global batch 256. Siblings on the hub are
`vitb/sub-52k` and `vitl/sub-52k/ddep8` (8-layer decoder).

## Results

Baseline column is `pretrain_full_90_10_h100`, from `experiments/fomo_tune_baseline`. Point
estimate with its 95% bootstrap CI as [low, high]. This run is `ead1264`.

| Task | Metric | baseline | walnut-v0.1 vitl/sub-52k |
|---|---|---|---|
| 1 infarct, n=21 | AUROC | 0.990 [0.944, 1.000] | 0.894 [0.731, 1.000] |
| 2 meningioma, n=23 | Dice | 0.195 [0.098, 0.303] | 0.195 [0.092, 0.306] |
| 3 brain age, n=494 | Pearson r | 0.963 [0.957, 0.969] | 0.968 [0.963, 0.972] |
| 3 brain age, n=494 | MAE (y) | 3.69 [3.45, 3.95] | 3.50 [3.29, 3.74] |
| 5 polymicrogyria, n=48 | AUROC | 0.984 [0.953, 1.000] | 0.995 [0.979, 1.000] |

Task 3 is the only task with the n to resolve a difference, and it moves the right way: r +0.005,
MAE -0.19y, with the new CI sitting almost entirely above the baseline's on r. Task 5 moves up and
task 1 down, both inside CIs that are far wider than the gap — task 1's whole drop is 10 of 104
subject pairs, one or two subjects reordered. Task 2's Dice is unchanged while its oracle falls
0.271 -> 0.234, so the ceiling on those features is lower even though thresholding lands the same.

The CIs above are unpaired, so overlap does not settle a comparison. `preds.json` holds the
per-subject out-of-fold predictions if a paired test is worth running.
