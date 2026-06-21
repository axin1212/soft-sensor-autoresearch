---
name: soft-sensor-autoresearch
description: Run local offline soft-sensor AutoResearch with FDE TabPFN3, SISSO-style derived features, robust holdout scoring, and an interactive Plotly fitability report. Use only when the user explicitly names $soft-sensor-autoresearch or asks to run the soft-sensor AutoResearch skill.
---

# Soft Sensor AutoResearch

Use this skill only after explicit invocation.

Required user inputs:
- Dataset file path (`.csv` or `.parquet`)
- Target column name

Before running, check local FDE/model availability. Do not silently fall back to XGBoost if the requested FDE model is unavailable.

Read `references/fde-integration.md` when FDE discovery, TabPFN weights, or environment checks are relevant.
Read `references/search-policy.md` when explaining or modifying the search strategy.

Run:

```bash
python scripts/run_soft_sensor_autoresearch.py <data-file> <target-column>
```

Useful options:
- `--time-budget-minutes <minutes>` controls the search budget; default is 15.
- `--num-train-samples <n>` controls the ICL context size; reduce this on memory-limited laptops.
- `--top-features-n <n>` controls how many ranked features enter the model; default is 32.
- `--validation-fraction <fraction>` controls the total target-label fraction held out across robust windows; default is `0.30`.
- `--window-minutes <minutes>` overrides the context window. When omitted, infer it from the dataset sampling interval.
- `--model-type <tabpfn3|tpt>` selects the FDE model path. Default is `tabpfn3`.
- `--tabpfn-device <cpu|auto|mps|cuda>` controls TabPFN device; default is `cpu` for laptop stability.
- `--tabpfn-fit-mode <mode>` controls TabPFN preprocessing/cache mode; default is `low_memory`.
- `--tpt-device <cpu|auto|mps|cuda>` controls TPT_tab device; default is `mps`.
- `--tpt-fit-mode <mode>` controls TPT_tab preprocessing/cache mode; default is `fit_preprocessors`.
- `--tpt-n-estimators <n>` controls TPT_tab ensemble size; default is `1` for fast laptop validation.
- `--fde-root <path>` points to a local FDE or benchmark checkout.
- `--output-dir <path>` overrides the output directory; default is next to the dataset.

Model weights:
- `--model-type tabpfn3` uses FDE foundation TabPFN3 regressor weights under `weights/tabpfn3/*regressor*.ckpt`.
- `--model-type tpt` uses FDE `TPTTabRegressor` with `$FDE_TPT_WEIGHTS_DIR/TPT_tab/model.ckpt`.

TPT_tab runs in an isolated child process. This avoids Metal/TPT runtime crashes caused by fitting TPT in the same Python process that just performed FDE feature extraction.

Return the final `report.html` path and summarize the best robust score.
