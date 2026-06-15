from sklearn.model_selection import KFold, StratifiedKFold

from evaluation.tasks.column import ColumnTask
from evaluation.tasks.dlbs.dataset import create_dlbs_t1w
from evaluation.tasks.registry import register_task


@register_task
def dlbs_age(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="dlbs_age",
        kind="regression",
        data=create_dlbs_t1w(),
        splitter=KFold(n_splits=n_splits, shuffle=True, random_state=seed),
        target_column="AgeMRI_W1",
    )


@register_task
def dlbs_sex(n_splits: int = 5, seed: int = 0) -> ColumnTask:
    return ColumnTask(
        name="dlbs_sex",
        kind="classification",
        data=create_dlbs_t1w(),
        splitter=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        target_column="Sex",
    )
