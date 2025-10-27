# ml-classification-template

Minimal ML template with a simple data pipeline and model registry for classification poblems.

## Install

```bash
uv sync --group dev
```

## Layout

```
data/<dataset_name>/
  ├── raw/
  │   ├── full.parquet
  │   ├── train.parquet
  │   └── test.parquet
  └── preprocessed/
      ├── full.parquet
      ├── train.parquet
      └── test.parquet

outputs/
  ├── gridsearch/              # cv_summary.json
  ├── models/                  # index.json, current_best.json, *.skops
  └── metrics/{model_id}/      # predictions.parquet, metrics.json, plots/
```

## Configuration

Edit `configs/default.toml`:

```toml
[data]
dataset_name = "example_data"
target_name = "target"                     # column name of the target variable

[eval]
random_state = 123
test_size = 0.20
cv_splits = 5
# Choose ONE scorer you intend to use:
scoring = "neg_root_mean_squared_error"   # regression example
# scoring = "roc_auc"                     # classification example
n_jobs = -1

[search]
model_keys = ["<modelname>"]

[train]                         # optional explicit spec; if unset, train reuses registry "best"
## example for support vector machine
# model_key = "svm_rbf"         
# [train.params]
# model__C = 3.0
# model__gamma = "scale"

[predict]
model_id = "best"     # "best" or a concrete model id from the registry
```

Grid search uses the single metric specified in `[eval].scoring`. The registry records `cv_score_type` (scorer name) and `cv_score` (numeric value).

## Checklist Before Running

1. Add models to `src/package/models.py`
2. Add grid to `src/package/grid.py` as `GRID_SPACES[<model_key>]`
3. Include desired model keys in `[search].model_keys`
4. Set the scoring method for cv in `configs/*.toml`
5. Define the metrics used to evaluate predictions in `src/package/eval/metrics.py`
6. Define the plots used to evaluate predictions in `src/package/eval/plots.py`

## Example Workflow

```bash
# Ingest data into raw/full
uv run package register-data --in <path-of-data>

# Copy raw to preprocessed (add transformations here)
uv run package preprocess

# Split preprocessed full into train/test
uv run package split --stage pre

# Grid search on preprocessed train
uv run package search

# Predict on preprocessed test with best model
uv run package predict --plots

# View registry (sorted by cv_score desc, then recency)
uv run package models --top 5
```

## Commands

All commands accept `--config` or `-c` to specify a config file (default: `configs/default.toml`).

| Command | Description |
|---------|-------------|
| `uv run package register-data --in <file>` | Read CSV/Parquet and write to `raw/full.parquet` |
| `uv run package preprocess` | Copy `raw/full.parquet` → `preprocessed/full.parquet` |
| `uv run package split --stage {raw\|pre}` | Stratified split of `<stage>/full.parquet` into train/test |
| `uv run package search` | GridSearchCV on `preprocessed/train`; registers best estimators trained on full training data|
| `uv run package train` | Fit the best or specified model on `preprocessed/train`; registers artifact |
| `uv run package predict [--model-id <id>] [--plots] [--plots-out <dir>]` | Predict on `preprocessed/test`; saves metrics and plots |
| `uv run package models [--top K]` | Show registry: id, model, cv_score_type, cv_score, created_at, params |

## Model Selection

The registry stores each model with its evaluation scorer. When using `--model-id best`, the best model among entries matching the current `[eval].scoring` is selected. If no model exists for that scorer, run `package search` with that scorer first.