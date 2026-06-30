# Soft Sensor AutoResearch

Local offline AutoResearch for soft-sensor fitability validation with FDE TabPFN3.

This repository is both:

- a Codex skill, triggered explicitly as `$soft-sensor-autoresearch`
- a Python package and CLI for running the search locally

It is designed for soft sensing tasks where process features are used to predict a target value at the same time or at explicitly requested future horizons. The default horizon is `h=0`, which uses features ending at time `t` to predict the target at time `t`.

## What It Does

Soft Sensor AutoResearch runs a finite, rule-driven search over low-risk feature candidates and evaluates each candidate with local FDE models.

The workflow:

1. Load a `.csv` or `.parquet` dataset.
2. Infer the time column and numeric feature columns.
3. Build robust target holdout windows.
4. Rank raw, context, trend, window, coverage, and optional frequency features.
5. Evaluate candidates with TabPFN3 or TPT_tab.
6. Write an interactive Plotly `report.html` with run parameters and a resource log.

Candidate ranking uses direct mean R-squared across completed holdouts. The report keeps per-holdout R-squared, RMSE, MAE, target standard deviation, and failure reasons visible for diagnosis.

## Requirements

- Python `>=3.11`
- Local FDE or benchmark checkout
- Local FDE model weights
- Python dependencies from `pyproject.toml`

The runner discovers FDE in this order:

1. `--fde-root`
2. `FDE_SOURCE_PATH`
3. current working directory and parent directories
4. sibling `FDE` or `benchmark` directories

For TabPFN3, `FDE_TPT_WEIGHTS_DIR` should point to the parent directory containing `tabpfn3/`, for example:

```bash
export FDE_TPT_WEIGHTS_DIR=/path/to/FDE/packages/kernels/kernels/weights
```

The tool fails fast if the requested FDE model or weights are unavailable. It does not silently fall back to XGBoost as the prediction model; XGBoost is used only for feature screening and Top-N ranking.

## Install

From a local checkout:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[dev]'
```

For optional frequency candidates:

```bash
python -m pip install -e '.[frequency]'
```

## Use as a Codex Skill

Clone or sync this repository under a Codex skill directory, or symlink it:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
ln -s "$(pwd)" "${CODEX_HOME:-$HOME/.codex}/skills/soft-sensor-autoresearch"
```

Then invoke it explicitly:

```text
Use $soft-sensor-autoresearch on /path/to/data.csv with target column <target_column>.
```

The skill requires:

- dataset file path
- target column name

## CLI Usage

Run through the bundled script:

```bash
python scripts/run_soft_sensor_autoresearch.py /path/to/data.csv '<target_column>'
```

Or, after editable install:

```bash
soft-sensor-autoresearch /path/to/data.csv '<target_column>'
```

Common options:

```bash
soft-sensor-autoresearch /path/to/data.csv '<target_column>' \
  --time-budget-minutes 15 \
  --num-train-samples 400 \
  --top-features-n 32 \
  --validation-fraction 0.30 \
  --forecast-horizons 0 \
  --model-type tabpfn3 \
  --tabpfn-device auto \
  --fde-root /path/to/FDE \
  --output-dir /path/to/output
```

Useful flags:

- `--time-budget-minutes 0`: run the full finite candidate list.
- `--num-train-samples <n>`: control ICL context size.
- `--top-features-n <n>`: control ranked feature count entering the model.
- `--window-minutes <n>`: override inferred sampling window.
- `--forecast-horizons <steps>`: evaluate horizons such as `0`, `0:10`, or `0,1,3,6,10`.
- `--model-type tabpfn3|tpt`: choose FDE model path.
- `--tabpfn-device cpu|auto|mps|cuda`: select TabPFN device.
- `--include-frequency-candidate`: enable tsfresh/frequency features.
- `--no-resource-log`: disable resource logging.
- `--open`: open the generated HTML report after the run.

## Input Data

Supported formats:

- `.csv`
- `.parquet`

Column inference:

- The target column must be provided explicitly.
- The first parseable datetime column is treated as the time column.
- If the first column is not parseable as time, columns containing `time`, `timestamp`, or `date` are checked.
- Numeric non-target columns become feature columns.

The runner fails when it cannot infer a time column, cannot find the target, or cannot infer numeric feature columns.

## Outputs

Each run creates a timestamped directory:

```text
autoresearch_YYYYMMDD_HHMMSS/
  report.html
  resource_usage.csv
```

`report.html` contains candidate rankings, fit plots, per-holdout metrics, and failure diagnostics.

The report starts with a `Run Parameters` section. Check it before comparing scores; it records the target, data file, model type, window size, ICL training sample count, top feature count, validation fraction, forecast horizons, frequency-candidate setting, FDE root, and model runtime parameters.

`resource_usage.csv` records process-tree CPU/RSS usage. MPS runs may also include PyTorch MPS memory events. These rows are memory observations, not Apple GPU utilization percentages.

## Forecast Horizons

The default forecast horizon is `0`.

- `h=0` uses the feature window ending at `t` to predict the target at `t`.
- `h>0` uses the feature window ending at `t-h` to predict the target at `t`.
- A horizon step is one dataset sampling step; for example, `h=10` on 10-minute data is 100 minutes ahead.

Future-horizon evaluation keeps holdout target time ranges fixed across horizons. Compare mean R-squared, worst holdout R-squared, RMSE, and MAE by horizon, then validate any apparent lead-time signal against target autocorrelation and process causality.

## Apple Silicon and MPS

Prefer TabPFN3 on MPS for Apple Silicon runs.

In sandboxed environments, PyTorch may report `torch.backends.mps.is_available() == False` even on a supported Mac. When that happens, verify with an escalated MPS smoke test before deciding the machine lacks MPS support. Use `--tabpfn-device cpu` only when CPU fallback is intentional.

## Negative R-Squared Triage

Treat strongly negative R-squared across low-risk candidates as a data/preprocessing diagnostic signal, not as a reason to add synthetic formula features immediately.

Check first:

- leakage columns and synchronized duplicate tags
- missing-value handling
- natural sampling level
- raw versus downsampled aggregations
- holdout target distribution shifts
- RMSE relative to target standard deviation

Only expand feature families after the basic data checks are clear. Do not use SISSO-style synthetic feature candidates for this skill.

## Local Sync

Remote:

```bash
git remote -v
```

Expected origin:

```text
https://github.com/axin1212/soft-sensor-autoresearch.git
```

Fetch the latest GitHub state:

```bash
git fetch origin
git status --short --branch
```

If working directly on `main` and the tree is clean, update with:

```bash
git pull --ff-only origin main
```

If working on a feature branch, inspect divergence before merging or rebasing:

```bash
git log --oneline --left-right --cherry-pick HEAD...origin/main
git diff --stat HEAD..origin/main
```

## Test

```bash
python -m pytest
```
