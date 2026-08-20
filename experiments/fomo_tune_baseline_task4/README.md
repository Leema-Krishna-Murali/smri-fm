# fomo_tune_baseline_task4

The first scored runs of `main_task4.py`. Everything verified so far is geometry and plumbing,
nothing yet about whether the features carry the signal.

Two sweeps, run as two independent jobs.

## Sweep A: geometry (`launch_geometry.sh`, 10 runs)

`scale` x `subcell` at the default features (final block, post-norm). A sub-cell spans
`8 / (scale * subcell)` mm, and that alone sets the Dice ceiling, so the two knobs are swept
together, and the comparison is the **equal-cell-size pairs**: at 1mm cells, is it better to buy the
resolution with input scale or with sub-patch decoding?

| | subcell 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| **scale 1** (1mm) | 8mm / 0.074 | 4mm / 0.217 | 2mm / 0.459 | 1mm / 0.714 |
| **scale 2** (0.5mm, native) | 4mm / 0.217 | 2mm / 0.459 | 1mm / 0.714 | 0.5mm / 1.000 |
| **scale 3** (0.33mm) | | 1.33mm / ? | 0.67mm / ? | |

Cells are `cell_mm / geometric ceiling`, the ceiling measured exactly over all 40 subjects.
`scale=3` is a two-point probe rather than a row: its cell sizes are not multiples of the 0.5mm
native grid, so the ceiling table does not cover them. Order is `s2_c4` (the current default)
first, so the headline number lands in ten minutes, then the rest of scale 2, then 1, then 3.

Two things when reading it. The input is **0.5mm native**, so `scale=2` is native, `scale=1`
downsamples to the pretraining resolution, and `scale=3` magnifies. And the ceilings are geometric:
through the real predict path `scale=2, subcell=4` reaches 0.665-0.770 where `scale=1, subcell=8`
reaches 0.564-0.679, since nearest-resampling a 2mm structure onto a 1mm grid loses some of it.

## Sweep B: block (`launch_block.sh`, 6 runs)

Blocks 3, 7, 11, 15, 19, 23 of 24, at the default `scale=2, subcell=4`. `block=null` is the final
post-norm output, already covered by sweep A's `s2_c4`.

The sub-patch decoder assumes a token still carries within-patch layout after 24 blocks. If the last
block has discarded it, the head degenerates to the 4mm ceiling of 0.217 whatever `subcell` says,
and nothing else in the design tests that. Run in parallel with A at the default geometry. If A
moves the best cell size a long way, B is an hour to redo there.

## Not swept

`target_sigma_mm`, a smoothing knob on top of whichever geometry wins, so it is cheaper once that is
known. `alphas` lands at 1e4, interior to the grid.

## Output

Per run under `output/<name>/`: `metrics.json`, `log.txt`, `config.yaml`, `curves.npz` (per-subject
x per-label x per-threshold Dice, predicted and true voxel counts), and `folds/<held-out subject>/`,
a loadable model dir plus `prediction.npz`, the voxels that fold claims and their labels.

A fold is saved at its own subject's **oracle** cut, chosen on that subject's labels: for
inspection, never a number to quote.

## Cost

A fold is ~8s and the one-time cache fill ~155s, so a run is ~9 min at `scale=2`.

## Results

All 16 runs at `69c2d36`; `collect.py` rebuilds this from `output/*/`. The `min` column is wall
clock on a shared node, not a timing.

**Best is 0.082 against a leaderboard leader at 0.40 and a geometric ceiling of 0.714 at 1mm cells.
At 11% of ceiling nothing here is geometry-limited**, which makes the grid's equal-cell-size
question moot: cell size moves the score by ~0.01 while the ceiling moves by 0.5, and the pairs
disagree with each other (2mm favours scale 1, 1mm favours scale 2, both inside the CIs).

| run | scale | subcell | cell mm | block | ceiling | dice | nerve | vessel | oracle | thr | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1_c1 | 1 | 1 | 8.00 | final | 0.074 | **0.017** [0.015, 0.020] | 0.000 | 0.035 | 0.027 | 3.8e-03 | 8 |
| s1_c2 | 1 | 2 | 4.00 | final | 0.217 | **0.032** [0.024, 0.041] | 0.010 | 0.054 | 0.049 | 4.7e-03 | 9 |
| s1_c4 | 1 | 4 | 2.00 | final | 0.459 | **0.045** [0.035, 0.057] | 0.044 | 0.046 | 0.055 | 3.8e-03 | 8 |
| s1_c8 | 1 | 8 | 1.00 | final | 0.714 | **0.036** [0.029, 0.045] | 0.032 | 0.041 | 0.042 | 4.7e-03 | 27 |
| s2_c1 | 2 | 1 | 4.00 | final | 0.217 | **0.035** [0.028, 0.041] | 0.000 | 0.070 | 0.045 | 9.1e-03 | 57 |
| s2_c2 | 2 | 2 | 2.00 | final | 0.459 | **0.038** [0.031, 0.045] | 0.000 | 0.077 | 0.048 | 9.1e-03 | 13 |
| s2_c4 | 2 | 4 | 1.00 | final | 0.714 | **0.045** [0.036, 0.054] | 0.001 | 0.089 | 0.055 | 1.1e-02 | 11 |
| s2_c8 | 2 | 8 | 0.50 | final | 1.000 | **0.050** [0.042, 0.058] | 0.026 | 0.074 | 0.058 | 9.1e-03 | 38 |
| s3_c2 | 3 | 2 | 1.33 | final | - | **0.072** [0.058, 0.087] | 0.048 | 0.096 | 0.080 | 1.4e-02 | 15 |
| s3_c4 | 3 | 4 | 0.67 | final | - | **0.082** [0.068, 0.097] | 0.064 | 0.100 | 0.093 | 1.4e-02 | 33 |
| blk03 | 2 | 4 | 1.00 | 3 | 0.714 | **0.061** [0.050, 0.073] | 0.004 | 0.119 | 0.072 | 1.1e-02 | 17 |
| blk07 | 2 | 4 | 1.00 | 7 | 0.714 | **0.056** [0.044, 0.067] | 0.001 | 0.111 | 0.067 | 1.4e-02 | 11 |
| blk11 | 2 | 4 | 1.00 | 11 | 0.714 | **0.056** [0.046, 0.066] | 0.001 | 0.111 | 0.065 | 1.4e-02 | 20 |
| blk15 | 2 | 4 | 1.00 | 15 | 0.714 | **0.053** [0.044, 0.063] | 0.002 | 0.105 | 0.065 | 1.4e-02 | 33 |
| blk19 | 2 | 4 | 1.00 | 19 | 0.714 | **0.050** [0.041, 0.059] | 0.004 | 0.097 | 0.060 | 1.4e-02 | 33 |
| blk23 | 2 | 4 | 1.00 | 23 | 0.714 | **0.049** [0.040, 0.057] | 0.011 | 0.086 | 0.060 | 1.1e-02 | 22 |

### The nerve is barely claimed, and a shared threshold is part of why

Nerve truth is 347 voxels a subject. At the selected threshold `s1_c1`, `s2_c1` and `s2_c2` claim
**zero nerve voxels in all 40 subjects**, and blk07-blk19 claim 1-2 voxels in 1-4 subjects. The
vessel meanwhile over-claims its 759 true voxels by 2-30x.

`binarize` takes an argmax over labels and then one shared cut, and the two labels want different
cuts: the nerve's own best is 5.9e-3-7.3e-3 where the vessel's is 1.1e-2-1.8e-2. The shared value is
dragged to the vessel's, because the vessel dominates the mean being maximized. Reading each label
at its own best threshold, off the same `curves.npz`:

| run | nerve @ shared | nerve @ own | vessel @ own | mean |
|---|---|---|---|---|
| s2_c4 | 0.001 | 0.021 | 0.089 | 0.045 -> 0.055 |
| blk15 | 0.002 | 0.031 | 0.105 | 0.053 -> 0.068 |
| blk03 | 0.004 | 0.020 | 0.119 | 0.061 -> 0.070 |
| s3_c4 | 0.064 | 0.069 | 0.122 | 0.082 -> 0.096 |

So a per-label threshold is worth ~+0.01-0.015 mean on every config, with no refit, read straight
off the measured curves. The nerve tops out near 0.03 either way, so this recovers only a small part
of the gap. These figures also still include the argmax suppression, which a per-label threshold
applied after the argmax cannot undo.

### The mean over labels rewards abandoning a label

`blk03` has the best mean (0.061) and finds the nerve in **3/40** subjects. `s1_c4` scores worse
(0.045) but finds it in **22/40** at roughly the right volume (375 claimed against 347 true).
Whether the challenge aggregates macro over labels is still not something we can read anywhere, so
picking a config on the mean is picking on an assumption.

### Depth is the strongest lever, and it is vessel-only

Vessel Dice at its own best threshold, by block: **0.119** (3), 0.111 (7), 0.111 (11), 0.105 (15),
0.097 (19), 0.091 (23), 0.089 (final). Monotone over seven points, and still climbing at the
earliest block tested. The nerve is flat at 0.02-0.03 across the whole depth range, with blk23
marginally the best of them.

### The two best levers were never combined

`scale=3` wins the geometry outright and is the only setting where the nerve does anything (0.069
at its own threshold), while `block=3` wins the depth sweep. They were run on separate axes and
never together.

## Sweep C: depth (`launch_depth.sh`)

Appended 2026-08-18, after changes to `main_task4.py`:

- Separate threshold per label rather than one shared threshold.
- `depth` replaces `block` and is a forward pre-hook. `depth=k+1` = `block=k`, `depth=0` = post patch/pos embed, `depth=None` = full post-norm model.
- `alphas` now runs to `1e8`, since a `depth=0` test run chose the old top value `1e6`.

## Results

| run | scale | subcell | cell mm | depth | ceiling | dice | nerve | vessel | oracle | nerve cut | vessel cut | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_d00 | 2 | 4 | 1.00 | 0 | 0.714 | **0.068** [0.052, 0.085] | 0.061 | 0.076 | 0.083 | 1.5e-03 | 2.4e-03 | 13 |
| s2_d01 | 2 | 4 | 1.00 | 1 | 0.714 | **0.126** [0.099, 0.156] | 0.136 | 0.117 | 0.147 | 7.3e-03 | 1.1e-02 | 13 |
| s2_d02 | 2 | 4 | 1.00 | 2 | 0.714 | **0.135** [0.107, 0.164] | 0.146 | 0.123 | 0.162 | 9.1e-03 | 1.4e-02 | 13 |
| s2_d04 | 2 | 4 | 1.00 | 4 | 0.714 | **0.130** [0.106, 0.156] | 0.142 | 0.118 | 0.148 | 7.3e-03 | 1.1e-02 | 13 |
| s2_d06 | 2 | 4 | 1.00 | 6 | 0.714 | **0.128** [0.105, 0.153] | 0.145 | 0.111 | 0.144 | 7.3e-03 | 1.1e-02 | 11 |
| s2_d08 | 2 | 4 | 1.00 | 8 | 0.714 | **0.120** [0.095, 0.147] | 0.128 | 0.111 | 0.140 | 9.1e-03 | 1.4e-02 | 11 |
| s2_d10 | 2 | 4 | 1.00 | 10 | 0.714 | **0.121** [0.097, 0.146] | 0.132 | 0.110 | 0.138 | 9.1e-03 | 1.4e-02 | 11 |
| s2_d12 | 2 | 4 | 1.00 | 12 | 0.714 | **0.126** [0.103, 0.150] | 0.142 | 0.109 | 0.138 | 9.1e-03 | 1.4e-02 | 11 |
| s3_d00 | 3 | 4 | 0.67 | 0 | - | **0.115** [0.093, 0.136] | 0.118 | 0.112 | 0.127 | 3.0e-03 | 4.7e-03 | 13 |
| s3_d01 | 3 | 4 | 0.67 | 1 | - | **0.180** [0.145, 0.213] | 0.223 | 0.137 | 0.198 | 1.1e-02 | 1.8e-02 | 22 |
| s3_d02 | 3 | 4 | 0.67 | 2 | - | **0.194** [0.161, 0.227] | 0.237 | 0.151 | 0.217 | 1.1e-02 | 1.8e-02 | 22 |
| s3_d04 | 3 | 4 | 0.67 | 4 | - | **0.199** [0.162, 0.234] | 0.249 | 0.149 | 0.219 | 1.4e-02 | 2.2e-02 | 22 |
| s3_d06 | 3 | 4 | 0.67 | 6 | - | **0.188** [0.155, 0.221] | 0.241 | 0.136 | 0.216 | 1.4e-02 | 2.2e-02 | 14 |
| s3_d08 | 3 | 4 | 0.67 | 8 | - | **0.195** [0.159, 0.227] | 0.258 | 0.132 | 0.214 | 1.4e-02 | 2.2e-02 | 14 |
| s3_d10 | 3 | 4 | 0.67 | 10 | - | **0.198** [0.164, 0.229] | 0.247 | 0.149 | 0.212 | 1.4e-02 | 2.2e-02 | 14 |
| s3_d12 | 3 | 4 | 0.67 | 12 | - | **0.186** [0.154, 0.217] | 0.230 | 0.142 | 0.205 | 1.4e-02 | 2.2e-02 | 14 |
| s3_dfinal | 3 | 4 | 0.67 | 24 | - | **0.134** [0.111, 0.159] | 0.145 | 0.123 | 0.160 | 1.4e-02 | 1.8e-02 | 13 |

*(Observations from CL)*

Large improvement overall 0.082 → 0.199.

### Big improvement from separate thresholds

`s3_dfinal` >> `s3_c4` and `s2_d04` >> `blk03`, where the only differences are the thresholding and new label argmax rule.

### Some ViT blocks help

Depth 0 (post patch embed, before any ViT blocks) is not the best overall, and much worse than depth 1. Best seems to be depth ~4.

### Upsampling helps

Scale 3 consistently outperforms scale 2, despite the artificial upsampling beyond the native resolution. Although nb we are at subcell=4, so scale 2 prediction is coarser.

## Sweep D: scale (`launch_scale.sh`)

Another follow-up sweep, investigating input scale and sub-cell prediction. Looking at `scale=4` (0.25mm input, ~50mm^3 FOV).

## Results

| run | scale | subcell | cell mm | depth | ceiling | dice | nerve | vessel | oracle | nerve cut | vessel cut | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s2_c2_d04 | 2 | 2 | 2.00 | 4 | 0.459 | **0.114** [0.090, 0.140] | 0.123 | 0.105 | 0.144 | 7.3e-03 | 1.1e-02 | 11 |
| s2_c4_d04 | 2 | 4 | 1.00 | 4 | 0.714 | **0.130** [0.106, 0.156] | 0.142 | 0.118 | 0.148 | 7.3e-03 | 1.1e-02 | 11 |
| s2_c8_d04 | 2 | 8 | 0.50 | 4 | 1.000 | **0.128** [0.106, 0.152] | 0.141 | 0.116 | 0.144 | 7.3e-03 | 1.1e-02 | 12 |
| s3_c2_d04 | 3 | 2 | 1.33 | 4 | - | **0.185** [0.152, 0.218] | 0.234 | 0.136 | 0.205 | 1.1e-02 | 1.8e-02 | 13 |
| s3_c4_d04 | 3 | 4 | 0.67 | 4 | - | **0.199** [0.162, 0.234] | 0.249 | 0.149 | 0.219 | 1.4e-02 | 2.2e-02 | 13 |
| s3_c8_d04 | 3 | 8 | 0.33 | 4 | - | **0.196** [0.164, 0.228] | 0.238 | 0.154 | 0.221 | 1.1e-02 | 1.8e-02 | 13 |
| s4_c2_d04 | 4 | 2 | 1.00 | 4 | 0.714 | **0.234** [0.194, 0.272] | 0.275 | 0.193 | 0.257 | 2.2e-02 | 2.8e-02 | 20 |
| s4_c4_d04 | 4 | 4 | 0.50 | 4 | 1.000 | **0.252** [0.212, 0.293] | 0.297 | 0.207 | 0.275 | 2.2e-02 | 2.8e-02 | 20 |
| s4_c8_d04 | 4 | 8 | 0.25 | 4 | - | **0.251** [0.211, 0.292] | 0.294 | 0.208 | 0.280 | 2.2e-02 | 2.8e-02 | 21 |

*(Observations from CL)*

Increasing the input resolution (now 4x the pretraining resolution) continues to give better performance. Even after controlling for sub-cell prediction resolution.

Sub-cell prediction resolution has a smaller effect, but consistent across input scales (`subcell=4` > `subcell=2`).
