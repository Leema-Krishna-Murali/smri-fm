# Evaluation

Internal evaluation suite. Currently only supporting frozen-feature sklearn probe.

## Run

```bash
uv run python -m evaluation.main_probe <model> <task> [--config cfg.yaml] [--overrides key=value ...]
# e.g.
uv run python -m evaluation.main_probe smri_mae dlbs_age --overrides model_kwargs.ckpt_path=/path/to/ckpt.pth
```

`model` and `task` are registered names (the CLI `--help` lists them). Run-level
settings come from [config/default_probe.yaml](config/default_probe.yaml),
overridden by an optional `--config` and then dot-list `--overrides`.

Outputs save in `<output_root>/<name>/` (default name `<model>__<task>`):

- `summary.csv`: one row of `model, task, tput, <metric>, <metric>_std`
- `metrics.json`: the summary plus per-fold scores
- `config.yaml`: the fully resolved config
- `log.txt`: run log

## Architecture

- [main_probe.py](main_probe.py) is the main entrypoint
- [models/](models/) contains model wrappers, e.g. [models/smri_mae.py](models/smri_mae.py). Each model defines a transform (`nib.Nifti1Image -> sample dict`) as well as the model itself (`batch dict -> embeddings`).
- [tasks/](tasks/) contains defined tasks, e.g. [tasks/fomo.py](tasks/fomo.py). Each task consists of a dataset as well as defined targets, splits, and scoring metrics.

## Adding things

Tasks and models share a registry: a builder decorated with `@register_task` /
`@register_model`, discovered automatically and constructed by name.

- **Task**: implement the `Task` protocol and decorate a builder with
  `@register_task`. For predicting a column of an HF dataset, use `ColumnTask`
  ([tasks/column.py](tasks/column.py)) with a sklearn splitter. See
  [tasks/dlbs.py](tasks/dlbs.py) and [tasks/fomo.py](tasks/fomo.py).
- **Model**: write a `(Model, Transform)` pair and decorate the builder with
  `@register_model`. See [models/smri_mae.py](models/smri_mae.py).
- **Dataset**: add a reproducible builder returning an HF `Dataset` of niftis +
  metadata next to its task (see `load_dlbs_t1w` / `load_fomo_task3`).
