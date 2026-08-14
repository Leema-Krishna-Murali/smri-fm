# explore_fomo_task4

Task 4 has no script yet. Before writing one: what are the two structures, and where does a model
have to be pointed to see them? Everything is measured on the native 0.5mm grid, because the
structures are ~2mm thick and the 1mm transform is not a frame you can look at them in.

```bash
uv run python explore.py     # -> explore.tsv, figures/*.png, npz crops in output/cache
```

## 1. The cohort is one protocol, and the labels are strictly bilateral

All 40 subjects are `(360, 512, 512)` at `0.500 x 0.488 x 0.488`mm, RAS, seg affine equal to the
image affine. Volumes are a few degrees oblique (up to 4°) but rigid, so index distances scaled by
the zooms are true mm.

**Label 1 has exactly two components in every subject** — one nerve per side, under 26-connectivity.
Label 2 has 2 to 5, which is one vessel per side entering and leaving the labelled slab. `1=nerve,
2=vessel` is stated by the challenge (`third_party/fomo_submission.md`), and the data agrees with
it: the two-per-subject structure is the nerve's, and the fragmenting one wraps the brainstem.

## 2. The structures are ~2mm thick and always in contact

Over 80 sides, median [min, max]:

| | median | range |
|---|---|---|
| nerve volume | 17.9 mm³ | 4.2 – 73.0 |
| vessel volume | 42.7 mm³ | 2.5 – 125.5 |
| nerve length along its principal axis | 9.9 mm | 3.1 – 19.4 |
| nerve mean cross extent | 2.07 mm | 1.03 – 3.71 |
| nerve-to-vessel minimum distance | 0.92 mm | 0.49 – 3.41 |

**The nerve runs anterior-posterior, not superior-inferior.** The principal axis has |A| = 0.98
median against |R| = 0.09 and |S| = 0.18. It is the cisternal segment of CN V leaving the lateral
pons, about 10mm long and 4 voxels across.

**Nerve and vessel touch.** 30 of 80 sides have them in adjacent voxels and none is more than
3.4mm apart, which is the point of the task — the labels are drawn where there is neurovascular
contact.

Two things worth noting before they surprise us later. The left nerve is consistently the larger
one (median 21.6 mm³ against 15.1), which is either laterality in the annotation or in the cohort.
And **intensity does not separate the two classes**: on a scale where 0 is the local median and 1
the local 95th percentile, the nerve sits at 0.08 and the vessel at 0.06, and the vessel is the
darker of the pair on only 33 of 80 sides. The pictures agree — what tells them apart is shape and
course, not brightness. Both are far below CSF, which is what the cistern around them is.

## 3. A free anchor cuts the crop 2.7x, and still leaves ±10mm

The label centroid moves a long way across subjects. Within side, in mm:

| placement rule | sd (R, A, S) | range (R, A, S) |
|---|---|---|
| fixed voxel index | 1.9, 7.8, 10.6 | 10.1, 42.9, 50.0 |
| centred on the mask centroid | 1.5, 3.7, 4.5 | 7.3, 23.8, 22.2 |
| centred on the mask bounding box | 1.9, 4.9, 3.8 | 9.3, 27.2, 18.0 |

The mask is the same `data > data.mean()` the transform uses, computed on the native volume, so it
is free and label-free — available at inference on one image.

Turned into the crop that a model would actually be handed, the smallest box holding all 40
subjects' labels:

| placement rule | box | volume |
|---|---|---|
| fixed voxel index | 114 x 111 x 124 | 187 mL |
| centred on the mask centroid | 114 x 70 x 72 | 69 mL |
| centred on the mask bounding box | 117 x 76 x 63 | 67 mL |

**So the anchor is worth taking: 2.7x less volume to process, for one mean and one centroid.**
The R axis does not shrink, because it is already tight — heads are well centred left-right and the
gap between the two sides is anatomy, not positioning.

What the anchor does *not* do is make position informative on its own. ±10mm of residual against a
2mm-thick structure means a fixed mask, or a probabilistic atlas in this frame, cannot be a
baseline worth beating. Anything that works will have to find the structures in the image.

## The figures

- `subjects.png` — 40 subjects, two rows each, plain over annotated, 8 axial slices spanning the
  labelled slab, over the full fixed crop box. Context: pons, cistern, temporal bone.
- `zoom.png` — same layout cropped to the labels with 20 voxels of margin, windowed per panel.
  The vessel curving around the brainstem across many slices is clearest here, and it is why label
  2 fragments.
- `planes.png` — axial, coronal and sagittal through each nerve centroid, per side, all 40. This
  is the one to read to see what the structures are: the nerve is a mid-grey band crossing bright
  CSF, the vessel a dark flow void alongside it.
- `geometry.png` — the scalars above.

## How the numbers are computed

Sides are the two nerve components, ordered by mean R. Each vessel component is assigned to the
nearer nerve by centroid distance, so `vessel_voxels` is per side and not per component.

`length` and `thickness` are extents along the eigenvectors of the component's voxel coordinates in
mm, longest first; thickness averages the two short axes. `contact_mm` is a `cKDTree` nearest
neighbour from nerve voxels to vessel voxels, so 0.49 means adjacent voxels.

Intensity is quoted against a local box, the side's label bounding box grown by 10 voxels, since the
volumes have no common scale. The few strongly negative values are sides where that box is mostly
bone and air and its median sits above the tissue the labels are in.

The crop cache is built from a fixed box that does not consult the labels, and `cache_subject`
asserts that every labelled voxel falls inside it.
