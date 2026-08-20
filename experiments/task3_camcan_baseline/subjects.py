"""One mid-axial slice per CamCAN subject, sorted by age. The whole holdout on one page.

The counterpart of `explore_fomo_task3/figures/subjects.png`, for the cohort the head fails on.
CamCAN arrives as a full head on a 192x256x256 grid, so unlike the task 3 images there is no
skull-stripping and `data > 0` is not a brain mask -- the slice is placed off the same
mean-threshold mask the backbone transform uses.

CPU only, ~2 minutes on 32 workers.

    uv run python experiments/task3_camcan_baseline/subjects.py
"""

import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from huggingface_hub import snapshot_download
from scipy import ndimage

OUT_DIR = Path(__file__).parent
RUN_DIR = OUT_DIR / "output/task3_camcan"
CROP_MM = (180, 216)
STRIDE = 4  # every 4th subject by age, enough to read the sweep without a 7M png
Z_FRACTION = 0.55
COLUMNS = 26


def tile(path: Path) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    brain = data > data.mean()
    box = [(idx.min(), idx.max() + 1) for idx in np.nonzero(brain)]
    centroid = ndimage.center_of_mass(brain)

    z = int(box[2][0] + Z_FRACTION * (box[2][1] - box[2][0]))
    plane = np.zeros(CROP_MM, dtype=np.float32)
    starts = [int(centroid[axis]) - CROP_MM[axis] // 2 for axis in (0, 1)]
    source = [
        slice(max(0, s), min(dim, s + width))
        for s, width, dim in zip(starts, CROP_MM, data.shape[:2])
    ]
    target = [slice(src.start - s, src.stop - s) for src, s in zip(source, starts)]
    plane[target[0], target[1]] = data[source[0], source[1], z] / np.percentile(data[brain], 99)
    return (255 * np.rot90(plane)[::2, ::2].clip(0, 1)).astype(np.uint8)


def main() -> None:
    rows = [json.loads(line) for line in (RUN_DIR / "camcan_preds.json").read_text().splitlines()]
    rows.sort(key=lambda row: row["age"])
    rows = rows[::STRIDE]

    root = Path(snapshot_download("medarc/CamCAN-T3", repo_type="dataset"))
    paths = [root / row["subject"] / "anat" / f"{row['subject']}_T1w.nii.gz" for row in rows]
    with Pool(32) as pool:
        tiles = pool.map(tile, paths)

    height, width = tiles[0].shape
    grid = int(np.ceil(len(rows) / COLUMNS))
    mosaic = np.zeros((grid * height, COLUMNS * width), dtype=np.uint8)
    for k, one in enumerate(tiles):
        r, c = divmod(k, COLUMNS)
        mosaic[r * height : (r + 1) * height, c * width : (c + 1) * width] = one

    fig, ax = plt.subplots(figsize=(COLUMNS * 0.95, grid * 1.16))
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for k, row in enumerate(rows):
        r, c = divmod(k, COLUMNS)
        ax.text(
            c * width + 2,
            r * height + 9,
            f"{row['age']:.0f}y {row['subject'].removeprefix('sub-CC')}",
            color="yellow",
            fontsize=3.6,
        )
        ax.text(
            c * width + 2,
            r * height + height - 3,
            f"{row['pred']:.0f}y  {row['pred'] - row['age']:+.0f}y",
            color="#7fd4ff",
            fontsize=3.6,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"CamCAN transfer, every {STRIDE}th of 481 subjects sorted by age ({len(rows)} shown). "
        "one axial slice at 55% of the "
        "mean-threshold box.\nyellow: true age, subject id.  blue: predicted age, error",
        fontsize=11,
    )
    fig.tight_layout()
    (OUT_DIR / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / "figures/subjects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
