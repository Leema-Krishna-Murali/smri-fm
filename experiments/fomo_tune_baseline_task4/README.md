# fomo_tune_baseline_task4

The first scored runs of `main_task4.py`. Everything verified so far is geometry and plumbing —
nothing yet about whether the features carry the signal.

Two sweeps, run as two independent jobs.

## Sweep A — geometry (`launch_geometry.sh`, 10 runs)

`scale` x `subcell` at the default features (final block, post-norm). A sub-cell spans
`8 / (scale * subcell)` mm, and that alone sets the Dice ceiling, so the two knobs are swept
together and the comparison is the **equal-cell-size pairs**: at 1mm cells, is it better to buy the
resolution with input resolution or with sub-patch decoding?

| | subcell 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| **scale 1** (1mm) | 8mm / 0.074 | 4mm / 0.217 | 2mm / 0.459 | 1mm / 0.714 |
| **scale 2** (0.5mm, native) | 4mm / 0.217 | 2mm / 0.459 | 1mm / 0.714 | 0.5mm / 1.000 |
| **scale 3** (0.33mm) | — | 1.33mm / ? | 0.67mm / ? | — |

Cells are `cell_mm / geometric ceiling`, the ceiling measured exactly over all 40 subjects.
`scale=3` is a two-point probe rather than a row: its cell sizes are not multiples of the 0.5mm
native grid, so the ceiling table does not cover them. Order is `s2_c4` (the current default)
first, so the headline number lands in ten minutes, then the rest of scale 2, then 1, then 3.

Two things when reading it. The input is **0.5mm native**, so `scale=2` is native, `scale=1`
downsamples to the pretraining resolution, and `scale=3` only magnifies. And the ceilings are
geometric: through the real predict path `scale=2, subcell=4` reaches 0.665–0.770 where
`scale=1, subcell=8` reaches 0.564–0.679, since nearest-resampling a 2mm structure onto a 1mm grid
loses some of it.

## Sweep B — block (`launch_block.sh`, 6 runs)

Blocks 3, 7, 11, 15, 19, 23 of 24, at the default `scale=2, subcell=4`. `block=null` is the final
post-norm output, already covered by sweep A's `s2_c4`.

The sub-patch decoder assumes a token still carries within-patch layout after 24 blocks. If the last
block has discarded it the head degenerates to the 4mm ceiling of 0.217 whatever `subcell` says, and
nothing else in the design tests that. Run in parallel with A at the default geometry; if A moves the
best cell size a long way, B is an hour to redo there.

## Not swept

`target_sigma_mm`, a smoothing knob on top of whichever geometry wins, so it is cheaper once that is
known. `alphas` lands at 1e4, interior to the grid.

## Output

Per run under `output/<name>/`: `metrics.json`, `log.txt`, `config.yaml`, `curves.npz` (per-subject
x per-label x per-threshold Dice, predicted and true voxel counts), and `folds/<held-out subject>/`
— a loadable model dir plus `prediction.npz`, the voxels that fold claims and their labels.

A fold is saved at its own subject's **oracle** cut, chosen on that subject's labels: for inspection,
never a number to quote.

## Cost

A fold is ~8s and the one-time cache fill ~155s, so a run is ~9 min at `scale=2`.

## Results

Not yet run. `collect.py` builds the table from `output/*/`.
