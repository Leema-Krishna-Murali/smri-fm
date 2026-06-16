import numpy as np
import datasets as hfds
from datasets import Dataset, DatasetDict, load_dataset

from evaluation.tasks.brain_age_gap import BrainAgeGapTask
from evaluation.tasks.column import ColumnTask
from evaluation.tasks.registry import register_task

REPO_ID = "medarc/adni_eval"


def load_adni_eval_merged():
    dataset_dict = load_dataset(REPO_ID)

    # drop repeated scans per subject
    dataset_dict = {
        split: drop_duplicates(ds, key="participant_id") for split, ds in dataset_dict.items()
    }

    # merge splits into one dataset
    dataset, split_indices = concatenate_splits(dataset_dict)

    # fixed train/test splits
    train_idx = np.concatenate([split_indices["train"], split_indices["validation"]])
    splits = [(train_idx, split_indices["test"])]
    return dataset, splits


def concatenate_splits(dataset_dict: DatasetDict):
    offset = 0
    split_indices = {}
    for split, ds in dataset_dict.items():
        split_indices[split] = np.arange(offset, offset + len(ds))
        offset += len(ds)
    dataset = hfds.concatenate_datasets(list(dataset_dict.values()))
    return dataset, split_indices


def drop_duplicates(dataset: Dataset, key: str):
    values = np.asarray(dataset[key])
    uniq, indices = np.unique(values, return_index=True)
    if len(uniq) < len(values):
        dataset = dataset.select(indices)
    return dataset


@register_task
def adni_age() -> ColumnTask:
    dataset, splits = load_adni_eval_merged()
    return ColumnTask(
        name="adni_age",
        kind="regression",
        data=dataset,
        splitter=splits,
        target_column="age",
    )


@register_task
def adni_sex() -> ColumnTask:
    dataset, splits = load_adni_eval_merged()
    return ColumnTask(
        name="adni_sex",
        kind="classification",
        data=dataset,
        splitter=splits,
        target_column="sex",
    )


@register_task
def adni_ad_cn() -> ColumnTask:
    dataset, splits = load_adni_eval_merged()
    return ColumnTask(
        name="adni_ad_cn",
        kind="classification",
        data=dataset,
        splitter=splits,
        target_column="diagnosis",
    )


@register_task
def adni_synthseg_volumes() -> ColumnTask:
    dataset, splits = load_adni_eval_merged()
    return ColumnTask(
        name="adni_synthseg_volumes",
        kind="regression",
        data=dataset,
        splitter=splits,
        target_column="synthseg_volumes",
    )


@register_task
def adni_ad_cn_bag() -> ColumnTask:
    dataset, _ = load_adni_eval_merged()
    return BrainAgeGapTask(
        name="adni_ad_cn_bag",
        data=dataset,
        age_column="age",
        dx_column="diagnosis",
        control_label=0,
        case_label=1,
    )
