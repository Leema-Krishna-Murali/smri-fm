"""Pooled embeddings for all 48 task-5 subjects, both checkpoints, cropped and not.

The runs save only preds and the head, so anything about the feature space itself needs a
re-forward. ~2 min on one H100.

    srun --jobid=<JOBID> --overlap uv run --no-sync python \
        experiments/task5_v2/dump_embeddings.py
"""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from fomo_tune.datasets import load_fomo_task5
from fomo_tune.main_task5 import Config, Task5Method
from fomo_tune.synthseg import synthseg_strip_dataset

CKPTS = {
    "ptfull": "hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth",
    "walnut": "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth",
}
OUT = Path(__file__).parent / "output/embeddings"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(synthseg_strip_dataset(load_fomo_task5(), source="t1w"))
    subjects = np.array([row["subject"] for row in rows])
    y = np.array([row["label"] for row in rows])

    for tag, ckpt_path in CKPTS.items():
        cfg = OmegaConf.merge(OmegaConf.structured(Config), {"ckpt_path": ckpt_path})
        method = Task5Method(cfg)
        for crop in (False, True):
            method.cfg.crop_ap = crop
            X = np.stack([method.features(row) for row in rows])
            path = OUT / f"{tag}_crop-{str(crop).lower()}.npz"
            np.savez(path, X=X, subjects=subjects, y=y)
            print(f"{path.name}  {X.shape}  {X.dtype}", flush=True)
        del method
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
