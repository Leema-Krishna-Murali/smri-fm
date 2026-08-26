"""FOMO task 3: brain age regression, scored by Pearson r and MAE as the challenge scores it.

`Task3Method` is the part we tune -- features, head, hyperparameters. The protocol below it is
fixed so scores stay comparable across iterations: 20-fold over the 494 subjects, pool the
out-of-fold predictions, bootstrap subjects for the CI.

The method fits on K2 augmented views of every subject and averages the same views at test time.
Views are deterministic in the subject and seed, so `precompute` embeds each one exactly once and
the folds only ever read the cache. Set `train_aug=false test_aug=false` for the plain baseline.

`train` runs that protocol then fits and saves a head; `predict` is the challenge contract, one t1
path in and one age out. Both go through `Task3Method.predict`, so every fold exercises the path
the submission will run.
"""

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import fomo_tune.synthseg as synthseg
from fomo_tune import task3_augmentation
from fomo_tune.backbone import load_backbone
from fomo_tune.task3_augmentation import K2_DRAWS, augment_draw, k2_views
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]

# seeds the test-time views when the caller has no subject to key on, as in the container
TTA_SUBJECT = "t1"

AGE_RANGE = (20.0, 80.0)
AGE_BINS = 6

# searched only when `alpha` is null
ALPHAS = np.logspace(-3, 6, 19)


@dataclass
class Config:
    task: str = "task3"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/pretrain_full_90_10_h100/checkpoint-last.pth"
    output_root: str = "output/fomo_tune"
    name: str = "task3"
    evals: tuple[str, ...] = ()
    depth: int | None = 16
    alpha: float | None = None
    train_aug: bool = True
    test_aug: bool = True
    balance_age: bool = False
    workers: int = 8
    feature_cache: str | None = "cache/features"
    device: str = "cuda"
    seed: int = 4466


# ---- method: the part we tune -----------------------------------------------------------


def age_weights(ages: np.ndarray) -> np.ndarray:
    """One over the density of the training ages."""
    clipped = np.clip(ages, *AGE_RANGE)
    edges = np.linspace(*AGE_RANGE, AGE_BINS + 1)
    counts, _ = np.histogram(clipped, bins=edges)
    index = np.clip(np.digitize(clipped, edges) - 1, 0, len(counts) - 1)
    weights = 1.0 / counts[index]

    effective = weights.sum() ** 2 / (weights**2).sum()
    assert effective > 0.5 * len(ages), f"age weights concentrate: effective n {effective:.0f}"
    return weights


class ViewDataset(Dataset):
    """One item is one subject's views for one draw, already through the backbone transform.

    Splitting on the draw rather than the subject halves the volumes in flight per worker.
    """

    def __init__(self, rows: list[dict], transform, seed: int):
        self.rows = rows
        self.transform = transform
        self.seed = seed
        self.specs = [(index, draw) for index in range(len(rows)) for draw in K2_DRAWS]

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> list[tuple[str, dict, dict]]:
        row_index, draw = self.specs[index]
        row = self.rows[row_index]

        views = [
            view
            for view in augment_draw(row, self.seed, draw)
            if draw == K2_DRAWS[0] or view["variant"] != "clean"
        ]

        return [
            (
                row["subject"],
                {
                    "key": view["subject"],
                    "variant": view["variant"],
                    "age": float(view["age"]),
                    "fit_weight": float(view["fit_weight"]),
                },
                self.transform(view["t1w"]),
            )
            for view in views
        ]


class Task3Method:
    """Frozen sMRI MAE, mean-pooled tokens over the t1w, ridge head over K2 augmented views."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, self.transform = load_backbone(cfg.ckpt_path)
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.cache: dict[str, np.ndarray] = {}
        self.views: dict[str, list[dict]] = {}
        self.head = None

    @torch.inference_mode()
    def embed(self, sample: dict[str, torch.Tensor]) -> np.ndarray:
        """(D,) from one prepared sample, mean-pooled over its post-norm tokens `depth` blocks in."""
        batch = {key: value[None].to(self.device) for key, value in sample.items()}

        encoder = self.backbone.encoder
        captured = []
        handle = None
        if self.cfg.depth is not None:
            handle = encoder.blocks[self.cfg.depth].register_forward_pre_hook(
                lambda module, args: captured.append(args[0])
            )
        try:
            with torch.autocast("cuda", torch.bfloat16, enabled=self.device.type == "cuda"):
                out = self.backbone(batch)
        finally:
            if handle is not None:
                handle.remove()

        # batch size 1 leaves no padded token slots, which the depth hook's flat slice relies on
        assert out["token_mask"].all(), "the token sequence is padded"
        if self.cfg.depth is None:
            tokens = out["patch_embeds"][0]
        else:
            tokens = encoder.norm(captured[0])[encoder.num_prefix_tokens :]
        return tokens.mean(dim=0).float().cpu().numpy()

    def features(self, images: Images) -> np.ndarray:
        """(D,) per subject. A pure function of the images, so training and inference agree."""
        return self.embed(self.transform(images["t1w"]))

    def cache_path(self, tag: str) -> Path | None:
        """Where `tag`'s embedded views live. The augmentation source is in the fingerprint, so
        editing it invalidates the cache rather than silently reusing the old views."""
        if not self.cfg.feature_cache:
            return None
        fingerprint = "|".join(
            [
                self.cfg.ckpt_path,
                str(self.cfg.depth),
                str(self.cfg.seed),
                Path(task3_augmentation.__file__).read_text(),
            ]
        )
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
        return Path(self.cfg.feature_cache) / f"{self.cfg.task}-{tag}-{digest}.joblib"

    def precompute(self, rows: list[dict], tag: str) -> None:
        """Embed every view of every row once, so `fit` and `predict` only read the cache."""
        path = self.cache_path(tag)
        if path is not None and path.exists():
            state = joblib.load(path)
            self.cache.update(state["cache"])
            self.views.update(state["views"])
            logger.info(f"{tag}: loaded {len(state['views'])} subjects from {path}")

        pending = [row for row in rows if row["subject"] not in self.views]
        if not pending:
            return

        dataset = ViewDataset(pending, self.transform, self.cfg.seed)
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.cfg.workers,
            prefetch_factor=2 if self.cfg.workers else None,
        )

        start = time.perf_counter()
        for index, items in enumerate(loader):
            for subject, record, sample in items:
                self.cache[record["key"]] = self.embed(sample)
                self.views.setdefault(subject, []).append(record)
            if (index + 1) % 20 == 0:
                logger.info(
                    f"{tag}: embedded {index + 1}/{len(dataset)} draws "
                    f"({time.perf_counter() - start:.0f}s)"
                )
        logger.info(
            f"{tag}: {len(pending)} subjects, {sum(len(self.views[r['subject']]) for r in pending)} "
            f"views ({time.perf_counter() - start:.0f}s)"
        )

        if path is not None:
            views = {row["subject"]: self.views[row["subject"]] for row in rows}
            cache = {
                record["key"]: self.cache[record["key"]]
                for records in views.values()
                for record in records
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".tmp-{os.getpid()}-{path.name}"
            joblib.dump({"cache": cache, "views": views}, tmp)
            tmp.rename(path)

    def fit(self, rows: list[dict]) -> None:
        records = [
            record
            for row in rows
            for record in self.views[row["subject"]]
            if self.cfg.train_aug or record["variant"] == "clean"
        ]
        X = np.stack([self.cache[record["key"]] for record in records])
        y = np.array([record["age"] for record in records], dtype=float)
        weights = np.array([record["fit_weight"] for record in records], dtype=float)

        if self.cfg.balance_age:
            weights = weights * age_weights(y)

        # one subject's views carry a total weight of one, so alpha means the same in every config
        weights = weights * len(rows) / weights.sum()

        alpha = self.cfg.alpha
        if alpha is None:
            # over the clean views alone, where one row is one subject. Rescaled to mean one
            # because Ridge trades sample_weight against alpha.
            clean = np.array([record["variant"] == "clean" for record in records], dtype=bool)
            clean_weights = weights[clean] * clean.sum() / weights[clean].sum()
            selector = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            selector.fit(
                X[clean],
                y[clean],
                standardscaler__sample_weight=clean_weights,
                ridgecv__sample_weight=clean_weights,
            )
            alpha = float(selector[-1].alpha_)

        self.head = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.head.fit(
            X,
            y,
            standardscaler__sample_weight=weights,
            ridge__sample_weight=weights,
        )

    def predict(self, images: Images, key: str | None = None) -> float:
        """Age in years, a weighted average over the test-time views.

        `key` names a subject `precompute` has already embedded, which is how cross-validation
        reuses a held-out subject's views. Without one, as in the container, the views are built
        here and seeded by a fixed pseudo-subject.
        """
        if key is not None:
            records = self.views[key]
            if not self.cfg.test_aug:
                records = [record for record in records if record["variant"] == "clean"]
            X = np.stack([self.cache[record["key"]] for record in records])
            weights = np.array([record["fit_weight"] for record in records], dtype=float)
        elif self.cfg.test_aug:
            row = {"subject": TTA_SUBJECT, "age": 0.0, "t1w": images["t1w"]}
            views = list(k2_views(row, self.cfg.seed))
            X = np.stack([self.features(view) for view in views])
            weights = np.array([view["fit_weight"] for view in views], dtype=float)
        else:
            X = self.features(images)[None]
            weights = np.ones(1)
        return float(self.head.predict(X) @ weights / weights.sum())

    def save(self, model_dir: Path) -> None:
        """Everything `load` needs but the backbone weights, which stay wherever `ckpt_path`
        points -- a few hundred KB, so a run saves one without copying a 3.7G checkpoint."""
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        joblib.dump(self.head, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task3Method":
        """Rebuild a fitted method from `save`. Overrides are Config fields, for what differs
        between here and the container -- the backbone path, the device."""
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        method.head = joblib.load(model_dir / "head.joblib")
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

# Every image the task ships. The method picks which of them it wants, as at inference, where the
# challenge hands over the modalities whether or not a model uses them.
IMAGE_COLS = ("t1w",)


def cross_validate(
    rows: list[dict], method: Task3Method, seed: int = 0, n_folds: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold age for every subject, each predicted by a head fit on the other folds."""
    y = np.array([row["age"] for row in rows], dtype=float)
    oof = np.zeros(len(rows), dtype=float)
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    start = time.perf_counter()
    for fold, (train, test) in enumerate(folds.split(rows)):
        method.fit([rows[i] for i in train])
        for i in test:
            images = {col: rows[i][col] for col in IMAGE_COLS}
            oof[i] = method.predict(images, key=rows[i]["subject"])
        logger.info(
            f"fold {fold + 1}/{n_folds} n={len(test)} mae={np.abs(y[test] - oof[test]).mean():.2f} "
            f"({time.perf_counter() - start:.0f}s)"
        )
    return y, oof


def evaluate(rows: list[dict], method: Task3Method) -> tuple[np.ndarray, np.ndarray]:
    """Age for every subject from an already-fitted method, through `predict`."""
    y = np.array([row["age"] for row in rows], dtype=float)
    pred = np.array(
        [method.predict({col: row[col] for col in IMAGE_COLS}, key=row["subject"]) for row in rows],
        dtype=float,
    )
    return y, pred


def metrics(y: np.ndarray, oof: np.ndarray) -> dict:
    return {
        "pearson_r": float(np.corrcoef(y, oof)[0, 1]),
        "mae": float(np.abs(y - oof).mean()),
    }


def score(
    y: np.ndarray, oof: np.ndarray, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05
) -> dict:
    """Both challenge metrics, each with a percentile CI resampling subjects with replacement."""
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(y), size=(n_boot, len(y)))

    summary = {}
    for name, point in metrics(y, oof).items():
        samples = [metrics(y[rows], oof[rows])[name] for rows in resamples]
        low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        summary[name] = point
        summary[f"{name}_ci_low"] = float(low)
        summary[f"{name}_ci_high"] = float(high)
    return summary


# ---- entrypoints ------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    # imported here, not at the top, so the container needs no dataset stack to run `predict`
    from fomo_tune.datasets import load_camcan, load_fomo_task3

    eval_loaders = {
        "camcan": lambda: synthseg.synthseg_strip_dataset(load_camcan(), source="t1w"),
    }

    cfg = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.from_dotlist(args.overrides))
    run_dir = Path(cfg.output_root) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)
    set_seed(cfg.seed)
    logger.info(f"run {cfg.name} (git {git_sha()})")
    logger.info(f"config:\n{OmegaConf.to_yaml(cfg).rstrip()}")
    OmegaConf.save(cfg, run_dir / "config.yaml")

    rows = list(load_fomo_task3())
    ages = np.array([row["age"] for row in rows])
    logger.info(
        f"dataset: {len(rows)} subjects, age {ages.min()}-{ages.max()} mean {ages.mean():.1f}"
    )

    method = Task3Method(cfg)
    method.precompute(rows, cfg.task)

    start = time.perf_counter()
    y, oof = cross_validate(rows, method)
    run_time = time.perf_counter() - start
    summary = score(y, oof)

    # the shipped head sees all n subjects, so it is not any of the models scored above
    method.fit(rows)
    method.save(run_dir / "model")

    preds = [
        {"subject": row["subject"], "age": float(age), "pred": float(pred)}
        for row, age, pred in zip(rows, y, oof)
    ]
    (run_dir / "preds.json").write_text("".join(json.dumps(pred) + "\n" for pred in preds))

    record = {"name": cfg.name, **summary, "run_time": round(run_time, 1)}
    scores = "  ".join(f"{k}={v:.4f}" for k, v in summary.items())
    logger.info(f"result: {scores}  ({run_time:.0f}s)")

    record["evals"] = {}
    for eval_name in cfg.evals:
        holdout = list(eval_loaders[eval_name]())
        holdout_ages = np.array([row["age"] for row in holdout])
        logger.info(
            f"{eval_name}: {len(holdout)} subjects, age {holdout_ages.min():.1f}-"
            f"{holdout_ages.max():.1f} mean {holdout_ages.mean():.1f}"
        )
        method.precompute(holdout, eval_name)
        y_eval, pred_eval = evaluate(holdout, method)
        eval_summary = score(y_eval, pred_eval)
        eval_preds = [
            {"subject": row["subject"], "age": float(age), "pred": float(p)}
            for row, age, p in zip(holdout, y_eval, pred_eval)
        ]
        (run_dir / f"{eval_name}_preds.json").write_text(
            "".join(json.dumps(row) + "\n" for row in eval_preds)
        )
        record["evals"][eval_name] = eval_summary
        eval_scores = "  ".join(f"{k}={v:.4f}" for k, v in eval_summary.items())
        logger.info(f"{eval_name}: {eval_scores}")

    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")


def predict(args: argparse.Namespace) -> None:
    """The challenge contract: a t1 path in, one age written to `--output`.

    `/app/predict.py` in the container is a shim over this, so what scores the submission is the
    code cross-validation already ran, not something generated at build time.
    """
    overrides = {"device": args.device}
    if args.ckpt_path:
        overrides["ckpt_path"] = args.ckpt_path
    method = Task3Method.load(args.model_dir, **overrides)

    img = nib.load(args.t1)
    seg = synthseg.synthseg(img)
    img = synthseg.applymask(img, seg)
    age = method.predict({"t1w": img})

    args.output.write_text(f"{age:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(required=True)

    train_parser = modes.add_parser("train", help="cross-validate over the task, then fit and save")
    train_parser.add_argument("overrides", nargs="*", help="config overrides, e.g. device=cpu")
    train_parser.set_defaults(run=train)

    predict_parser = modes.add_parser("predict", help="one subject, one age in years")
    predict_parser.add_argument("--t1", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--model-dir", type=Path, default=Path("/app/model"))
    predict_parser.add_argument("--ckpt-path", help="overrides the trained config's backbone path")
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
