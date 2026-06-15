# Evaluation

Internal evaluation suite for the sMRI foundation model. The primary path is a
**frozen-feature sklearn probe**: extract embeddings once with a frozen model,
then fit a linear estimator per task split.

## Architecture

Four layers, with the probe (`main_probe.py`) as the only orchestration:

```
dataset builder  ->  task  ->  model (Model + Transform)  ->  probe
                                                              |
   reproducible      thin wrapper:    frozen embeddings    fit sklearn
   HF Dataset        kind, dataset,   nifti -> [B, D]      per split,
   of nifti+meta     split, scoring                        score, report
```

- **Model** ([models/base.py](models/base.py)) is a `(Model, Transform)` pair.
  `Transform` maps a `nib.Nifti1Image` to a collatable sample dict; `Model`
  maps a batch to an embedding `[B, D]`. Pooling lives inside the model.
- **Task** ([tasks/base.py](tasks/base.py)) is a thin, declarative wrapper around a dataset.
  It owns `dataset()` (canonical `{image, target, id}` samples), `split()`
  (`(train_idx, test_idx)` pairs — one for a fixed split, many for outer CV),
  `metrics(y_true, y_pred, test_idx)`, and `kind`. The task fully owns its
  splitting strategy so its score is unambiguous.
- **Probe** extracts features once over the dataset, then fits the estimator
  for `task.kind` (`RidgeCV` for regression, `LogisticRegressionCV` for
  classification — inner CV selects hyperparameters) on each split and reports
  per-fold mean ± std.

`metrics` receives `test_idx` so a task can reach into its own metadata for
auxiliary variables. This is what lets brain-age-gap style tasks
([tasks/brain_age_gap.py](tasks/brain_age_gap.py)) train on age but score the
gap against a clinical variable, with no special probe logic.

## Run

```bash
uv run python -m evaluation.main_probe --config <config.yaml> [key=value ...]
```

Outputs land in `<output_dir>/<name>/`: `metrics.json` (summary + per-fold),
`predictions.csv`, and `features.npz` (cached embeddings).

### Config

```yaml
name: dlbs_age_probe
output_dir: data/runs/evaluation
seed: 0
device: cuda
batch_size: 4
num_workers: 4

task:
  name: dlbs_age      # any extra keys are passed to the task builder

model:
  name: smri_mae      # remaining keys are passed to the model builder
  ckpt_path: /path/to/checkpoint.pth
  global_pool: patch
```

## Adding things

Tasks and models share the same registry pattern: a builder decorated with
`@register_task` / `@register_model`, discovered automatically and constructed
by name from config.

- **Task**: implement the `Task` protocol ([tasks/base.py](tasks/base.py)) and
  decorate a builder with `@register_task`. For column prediction over an HF
  dataset, use `ColumnTask` ([tasks/column.py](tasks/column.py)) with a sklearn
  splitter (see [tasks/dlbs/](tasks/dlbs/)).
- **Model**: write a `(Model, Transform)` pair and decorate the builder with
  `@register_model` (see [models/smri_mae.py](models/smri_mae.py)).
- **Dataset**: add a reproducible builder returning an HF `Dataset` of niftis +
  metadata next to its task (see [tasks/dlbs/dataset.py](tasks/dlbs/dataset.py)).
