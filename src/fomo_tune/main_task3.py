"""FOMO task 3: brain age regression."""

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import fomo_tune.synthseg as synthseg
from fomo_tune.backbone import SmriMaeTransform, load_backbone
from fomo_tune.perturb import PERTURBATIONS
from fomo_tune.utils import git_sha, set_seed, setup_logging

logger = logging.getLogger("fomo_tune")

Images = dict[str, nib.Nifti1Image]

# searched only when `alpha` is null
ALPHAS = np.logspace(-3, 6, 19)


@dataclass
class Config:
    task: str = "task3"
    ckpt_path: str = "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
    output_root: str = "output/fomo_tune"
    name: str = "task3"
    evals: tuple[str, ...] = ()
    train_views: tuple[str, ...] = ()
    depth: int | None = None
    alpha: float | None = None
    workers: int = 8
    feature_cache: str | None = "cache/features"
    device: str = "cuda"
    seed: int = 4466


# ---- method: the part we tune -----------------------------------------------------------


class SubjectDataset(Dataset):
    """One item is one subject's views, perturbed and through the backbone transform.

    The clean view is always first, which is the one `predict` reads back.
    """

    def __init__(self, rows: list[dict], transform, views: tuple[str, ...]):
        self.rows = rows
        self.transform = transform
        self.views = views

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, list[dict]]:
        subject = self.rows[index]["subject"]
        img = self.rows[index]["t1w"]
        views = [
            self.transform(img if view == "clean" else PERTURBATIONS[view](img))
            for view in self.views
        ]
        return subject, views


class Task3Method:
    """Frozen sMRI MAE, mean-pooled tokens over the t1w, ridge head."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbone, transform = load_backbone(cfg.ckpt_path)
        self.transform = SmriMaeTransform(
            img_size=transform.img_size, spacing=transform.spacing, masking="zero"
        )
        self.device = torch.device(cfg.device)
        self.backbone.to(self.device).eval().requires_grad_(False)
        self.cache: dict[str, np.ndarray] = {}
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

    def cache_path(self, tag: str, views: tuple[str, ...]) -> Path | None:
        """Where `tag`'s embeddings live, keyed by everything that changes what they are."""
        if not self.cfg.feature_cache:
            return None
        fingerprint = "|".join(
            [self.cfg.ckpt_path, str(self.cfg.depth), str(self.cfg.seed), *views]
        )
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:12]
        return Path(self.cfg.feature_cache) / f"{self.cfg.task}-{tag}-{digest}.joblib"

    def precompute(self, rows: list[dict], tag: str, views: tuple[str, ...] = ("clean",)) -> None:
        """Embed every row's views once, so `fit` and `predict` only read the cache."""
        path = self.cache_path(tag, views)
        self.cache = joblib.load(path) if path is not None and path.exists() else {}
        if self.cache:
            logger.info(f"{tag}: loaded {len(self.cache)} subjects from {path}")

        pending = [row for row in rows if row["subject"] not in self.cache]
        if not pending:
            return

        dataset = SubjectDataset(pending, self.transform, views)
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.cfg.workers,
            prefetch_factor=2 if self.cfg.workers else None,
        )

        start = time.perf_counter()
        for index, (subject, samples) in enumerate(loader):
            self.cache[subject] = np.stack([self.embed(sample) for sample in samples])
            if (index + 1) % 50 == 0:
                logger.info(
                    f"{tag}: embedded {index + 1}/{len(dataset)} subjects "
                    f"({time.perf_counter() - start:.0f}s)"
                )
        logger.info(f"{tag}: {len(pending)} subjects ({time.perf_counter() - start:.0f}s)")

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".tmp-{os.getpid()}-{path.name}"
            joblib.dump({row["subject"]: self.cache[row["subject"]] for row in rows}, tmp)
            tmp.rename(path)

    def fit(self, rows: list[dict]) -> None:
        views = np.stack([self.cache[row["subject"]] for row in rows])
        ages = np.array([row["age"] for row in rows], dtype=float)
        n_subjects, n_views, _ = views.shape

        X = views.reshape(n_subjects * n_views, -1)
        y = np.repeat(ages, n_views)

        alpha = self.cfg.alpha
        if alpha is None:
            # over the clean views alone: RidgeCV's leave-one-out would otherwise hold out one
            # view of a subject while its siblings stayed in, and under-regularize
            clean = np.arange(n_subjects) * n_views
            selector = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
            selector.fit(X[clean], y[clean])
            alpha = float(selector[-1].alpha_)

        self.head = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        # one subject's views carry a total weight of one, so alpha means the same at any view count
        weight = np.full(len(X), 1.0 / n_views)
        self.head.fit(X, y, ridge__sample_weight=weight)

    def predict(self, images: Images, key: str | None = None) -> float:
        """Age in years. Pass key to load from cache."""
        X = self.cache[key][:1] if key is not None else self.features(images)[None]
        return float(self.head.predict(X)[0])

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(self.cfg, model_dir / "config.yaml")
        joblib.dump(self.head, model_dir / "head.joblib")

    @classmethod
    def load(cls, model_dir: Path, **overrides) -> "Task3Method":
        cfg = OmegaConf.merge(
            OmegaConf.structured(Config), OmegaConf.load(model_dir / "config.yaml"), overrides
        )
        method = cls(cfg)
        method.head = joblib.load(model_dir / "head.joblib")
        return method


# ---- protocol: the part we hold fixed ---------------------------------------------------

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
    train_views = ("clean", *cfg.train_views)
    method.precompute(rows, cfg.task, train_views)

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
        # "camcan" is the cohort as it comes; "camcan-thick_slice_5mm" corrupts it on the way in
        cohort, _, perturbation = eval_name.partition("-")
        holdout = list(eval_loaders[cohort]())
        holdout_ages = np.array([row["age"] for row in holdout])
        logger.info(
            f"{eval_name}: {len(holdout)} subjects, age {holdout_ages.min():.1f}-"
            f"{holdout_ages.max():.1f} mean {holdout_ages.mean():.1f}"
        )
        method.precompute(holdout, eval_name, (perturbation or "clean",))
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
