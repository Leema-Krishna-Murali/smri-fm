"""Why the two checkpoints behave differently on task 5, at the level of the feature space.

`dump_embeddings.py` has to have run. Four sets of 48x1024: two checkpoints, cropped and not.

Reads the covariates from `experiments/explore_fomo_task5/explore.tsv` and the post-crop brain
volume from `output/verify_crop_133.tsv`, so a component can be named by what it tracks rather
than by its index.

    uv run python experiments/task5_v2/compare_embeddings.py \
        | tee experiments/task5_v2/output/compare_embeddings.log
"""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

EXP_DIR = Path(__file__).parent
EMB_DIR = EXP_DIR / "output/embeddings"
EXPLORE = EXP_DIR.parent / "explore_fomo_task5/explore.tsv"
CROPPED = EXP_DIR / "output/verify_crop_133.tsv"

TAGS = ("ptfull", "walnut")
N_PC = 6


def unit(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load(tag: str, crop: bool) -> dict:
    return dict(np.load(EMB_DIR / f"{tag}_crop-{str(crop).lower()}.npz", allow_pickle=False))


def covariates(subjects: np.ndarray) -> dict[str, np.ndarray]:
    explore = {r["subject"]: r for r in csv.DictReader(EXPLORE.open(), delimiter="\t")}
    cropped = {r["subject"]: r for r in csv.DictReader(CROPPED.open(), delimiter="\t")}

    def pick(src: dict, key: str) -> np.ndarray:
        return np.array([float(src[s][key]) for s in subjects])

    return {
        "edge": pick(explore, "brain_at_ap_edge"),
        "ventricle": pick(explore, "ventricle_ml"),
        "brain (cropped)": pick(cropped, "brain_ml"),
        "skyra": (pick(explore, "matrix") == 260).astype(float),
    }


def signed_auroc(y: np.ndarray, score: np.ndarray) -> str:
    """A component has no inherent sign, so report the separation and mark a flipped one."""
    auroc = roc_auc_score(y, score)
    return f"{max(auroc, 1 - auroc):.3f}{' v' if auroc < 0.5 else '  '}"


def participation_ratio(variances: np.ndarray) -> float:
    """How many components carry the variance, counting a flat spectrum as its full length."""
    return float(variances.sum() ** 2 / (variances**2).sum())


def geometry(X: np.ndarray) -> tuple[PCA, str]:
    cosines = unit(X) @ unit(X).T
    off_diagonal = cosines[~np.eye(len(X), dtype=bool)]
    pca = PCA().fit(X)
    ratio = pca.explained_variance_ratio_
    summary = (
        f"cosine {off_diagonal.mean():.4f}  ‖x‖ {np.linalg.norm(X, axis=1).mean():.1f}  "
        f"PC1 {ratio[0]:.2f}  PC1-3 {ratio[:3].sum():.2f}  participation {participation_ratio(ratio):.1f}"
    )
    return pca, summary


def canonical_correlations(A: np.ndarray, B: np.ndarray, k: int = 10) -> np.ndarray:
    """Correlations between the k-dimensional principal subspaces of two embeddings."""
    qa = np.linalg.qr(PCA(k).fit_transform(A))[0]
    qb = np.linalg.qr(PCA(k).fit_transform(B))[0]
    return np.linalg.svd(qa.T @ qb, compute_uv=False)


def main() -> None:
    data = {(tag, crop): load(tag, crop) for tag in TAGS for crop in (False, True)}
    subjects = data[("ptfull", False)]["subjects"]
    y = data[("ptfull", False)]["y"]
    covs = covariates(subjects)

    print("=== geometry of the pooled embedding, 48 subjects x 1024\n")
    for (tag, crop), d in data.items():
        _, summary = geometry(d["X"])
        print(f"{tag:<7} crop={str(crop):<5} {summary}")

    print("\n\n=== what each principal component tracks (spearman; AUROC for the label)\n")
    for (tag, crop), d in data.items():
        pca = PCA(N_PC).fit(d["X"])
        scores = pca.transform(d["X"])
        print(f"{tag} crop={crop}")
        header = f"  {'PC':<4} {'var':>6} {'label AUROC':>12}" + "".join(
            f"{name:>18}" for name in covs
        )
        print(header)
        for i in range(N_PC):
            row = f"  {i + 1:<4} {pca.explained_variance_ratio_[i]:>6.2f} "
            row += f"{signed_auroc(y, scores[:, i]):>12}"
            for value in covs.values():
                row += f"{spearmanr(scores[:, i], value).statistic:>+18.2f}"
            print(row)
        print()

    print("\n=== how much the crop moves a subject\n")
    for tag in TAGS:
        A, B = data[(tag, False)]["X"], data[(tag, True)]["X"]
        same = (unit(A) * unit(B)).sum(axis=1)
        others = unit(A) @ unit(A).T
        others = others[~np.eye(len(A), dtype=bool)]
        print(
            f"{tag:<7} cosine(subject uncropped, same subject cropped) {same.mean():.4f} "
            f"[{same.min():.4f}, {same.max():.4f}]   "
            f"vs cosine between different subjects {others.mean():.4f}"
        )

    print("\n\n=== how much the two checkpoints share, principal subspaces of dimension 10\n")
    for crop in (False, True):
        correlations = canonical_correlations(
            data[("ptfull", crop)]["X"], data[("walnut", crop)]["X"]
        )
        print(
            f"crop={str(crop):<6} canonical correlations "
            + " ".join(f"{c:.2f}" for c in correlations)
        )

    print("\n\n=== do the two see the same thing? spearman between their PC scores over subjects")
    print("    (the 1024 channels are not comparable across models, the 48 subjects are)\n")
    for crop in (True,):
        a = PCA(N_PC).fit_transform(data[("ptfull", crop)]["X"])
        b = PCA(N_PC).fit_transform(data[("walnut", crop)]["X"])
        print(f"  crop={crop}" + "".join(f"{'walnut PC' + str(j + 1):>13}" for j in range(N_PC)))
        for i in range(N_PC):
            row = "".join(f"{spearmanr(a[:, i], b[:, j]).statistic:>+13.2f}" for j in range(N_PC))
            print(f"  ptfull PC{i + 1}" + row)


if __name__ == "__main__":
    main()
