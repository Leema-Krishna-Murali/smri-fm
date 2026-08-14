# explore_fomo_task4

Task 4 has no script yet. What are the two structures, and where does a model have to be pointed to
see them? Measured on the native 0.5mm grid — they are ~2mm thick, so the 1mm transform is not a
frame you can look at them in.

```bash
uv run python explore.py     # -> explore.tsv, figures/*.png, npz crops in output/cache
```

## The structures

All 40 subjects are `(360, 512, 512)` at `0.500 x 0.488 x 0.488`mm, RAS, seg affine equal to the
image affine, a few degrees oblique but rigid. **Label 1 has exactly two components in every
subject** — one nerve per side. Label 2 has 2 to 5, one vessel per side entering and leaving the
labelled slab. That matches the challenge's `1=nerve, 2=vessel`.

Over 80 sides:

| | median | range |
|---|---|---|
| nerve volume | 17.9 mm³ | 4.2 – 73.0 |
| vessel volume | 42.7 mm³ | 2.5 – 125.5 |
| nerve length along its principal axis | 9.9 mm | 3.1 – 19.4 |
| nerve mean cross extent | 2.07 mm | 1.03 – 3.71 |
| nerve-to-vessel minimum distance | 0.92 mm | 0.49 – 3.41 |

**The nerve runs anterior-posterior, not superior-inferior**: |A| = 0.98 against |R| = 0.09 and
|S| = 0.18. The vessel has no such preference (|A| = 0.75, |R| = 0.34, |S| = 0.32) and sprawls
7.5 x 11.7 x 8.1mm as it wraps the brainstem, which is why it fragments.

**Nerve and vessel touch** — 30 of 80 sides in adjacent voxels, never more than 3.4mm apart. That
is the point of the task.

**Intensity barely separates the classes.** On a scale where 0 is the local median and 1 the local
95th percentile, nerve sits at 0.14 and vessel at 0.09, and the vessel is the darker of the two on
48 of 80 sides — a coin flip's worth of margin over the 40 that chance gives. Shape and course tell
them apart, not brightness.

The left nerve is consistently the larger (median 21.6 mm³ against 15.1) — laterality in the
annotation or in the cohort, unexplained.

## Where they are

Spread of the nerve centroid across subjects, within side, in mm:

| placement rule | sd (R, A, S) | range (R, A, S) |
|---|---|---|
| fixed voxel index | 1.9, 7.8, 10.6 | 10.1, 42.9, 50.0 |
| centred on the mask centroid | 1.5, 3.7, 4.5 | 7.3, 23.8, 22.2 |
| centred on the mask bounding box | 1.9, 4.9, 3.8 | 9.3, 27.2, 18.0 |

As the crop a model would be handed — the smallest box holding all 40 subjects' labels:

| placement rule | box | volume |
|---|---|---|
| fixed voxel index | 114 x 111 x 124 | 187 mL |
| centred on the mask centroid | 114 x 70 x 72 | 69 mL |
| centred on the mask bounding box | 117 x 76 x 63 | 67 mL |

The mask is the same `data > data.mean()` the transform uses, so the anchor is free and label-free.
**Worth taking: 2.7x less volume for one mean and one centroid.** R does not shrink because it is
already tight — heads are centred left-right and the gap between sides is anatomy.

It does not make position informative on its own. ±10mm of residual against a 2mm structure means
no fixed mask or atlas prior in this frame is a baseline worth beating.

## The box to hand a model

**128 x 96 x 96 voxels at 0.5mm (64 x 47 x 47 mm), centred (0, +4, −16) voxels from the subject's
own mask centroid.** A 99%-coverage, 95%-confidence normal tolerance box over the label extremes
is 117 x 84 x 89; this clears that on every axis, stays a multiple of 16, and holds all 40
subjects with 3.2 / 6.3 / 4.9 mm to spare. Widen R to 144 for 7.2mm there, at 158 mL.

One box over both sides rather than one per side. Per-side would be 64 x 96 x 96 at the same A and
S, so the same total voxels and the same class imbalance, trading a half-size tensor and n=80 for
a left/right convention to get wrong.

The labels are **121 mm³ inside 141 mL, about 1 voxel in 1200**, whichever way it is cut.

```bash
uv run python model_box.py     # -> figures/model_box.png
```

## The figures

- `subjects.png` — axial, 8 slices over the whole crop box, plain over annotated. The only plane
  holding both sides, the pons and the cistern in one panel.
- `zoom.png` — sagittal, 8 consecutive slices per side, windowed per panel. The nerve drifts out of
  sagittal by 0.94mm over its length, against 1.54mm axial and 9.28mm coronal, so it stays a
  continuous band; vessel-above-or-below-nerve is also a sagittal relation.
- `planes.png` — three planes through each nerve, per side. Read this one to see what the
  structures are: the nerve a mid-grey band crossing bright CSF, the vessel a dark flow void.
- `model_box.png` — six subjects spanning the hard cases, three rows each: the head with the box
  outlined, the box, the box with labels. In-plane the panels are exactly the model's input; the
  slices are picked around each side's nerve, which inference cannot do, so read the through-plane
  sampling as optimistic.
- `geometry.png` — the scalars above.

Sides are the two nerve components ordered by mean R, with each vessel component assigned to the
nearer nerve, so the table is per side and not per component. Intensity is quoted against the
side's label bounding box grown by 10 voxels, since the volumes have no common scale.
