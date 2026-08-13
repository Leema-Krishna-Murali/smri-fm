# fomo_tune_walnut_v0_1

Tasks 1, 2, 3 and 5 on the walnut-v0.1 ViT-L checkpoint, everything else identical to
`experiments/fomo_tune_baseline`. Same architecture as the default (`mae_vit_large`, patch 8,
208x240x208), so it is a `ckpt_path` override and nothing else.

`hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth` — 52,643 subjects of
FOMO300K, subject-scaling run at a fixed 99k steps, global batch 256. Siblings on the hub are
`vitb/sub-52k` and `vitl/sub-52k/ddep8` (8-layer decoder).

## Results

Baseline column is `pretrain_full_90_10_h100`, from `experiments/fomo_tune_baseline`.

| Task | Metric | baseline | walnut-v0.1 vitl/sub-52k |
|---|---|---|---|
| 1 infarct | AUROC | 0.990 | |
| 2 meningioma | Dice | 0.195 | |
| 3 brain age | Pearson r | 0.963 | |
| 3 brain age | MAE (y) | 3.69 | |
| 5 polymicrogyria | AUROC | 0.984 | |
