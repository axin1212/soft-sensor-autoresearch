---
name: soft-sensor-autoresearch
description: Run local offline soft-sensor AutoResearch with FDE TabPFN3, SISSO-style derived features, mean R-squared candidate ranking, and an interactive Plotly fitability report. Use only when the user explicitly names $soft-sensor-autoresearch or asks to run the soft-sensor AutoResearch skill.
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
- `--time-budget-minutes <minutes>` controls the search budget; default is 15. Use `0` to remove the time cap and run the full finite candidate list.
- `--num-train-samples <n>` controls the ICL context size; reduce this on memory-limited laptops.
- The search also probes larger SISSO context sizes derived from `--num-train-samples`; with the default 400 it tries 700 and 900 before slow frequency candidates.
- `--top-features-n <n>` controls how many ranked features enter the model; default is 32.
- `--validation-fraction <fraction>` controls the total target-label fraction held out across robust windows; default is `0.30`.
- `--window-minutes <minutes>` overrides the context window. When omitted, infer it from the dataset sampling interval.
- `--model-type <tabpfn3|tpt>` selects the FDE model path. Default is `tabpfn3`.
- `--tabpfn-device <cpu|auto|mps|cuda>` controls TabPFN device; default is `auto`, preferring MPS when PyTorch reports it is available. On Apple Silicon, `auto` fails fast when PyTorch is built with MPS but the runtime cannot see Metal devices; pass `cpu` explicitly only when CPU fallback is intended.
- `--tabpfn-fit-mode <mode>` controls TabPFN preprocessing/cache mode; default is `fit_preprocessors` for stable local MPS validation.
- `--tabpfn-n-estimators <n>` controls TabPFN ensemble size; default is `1` for stable local MPS validation. The upstream TabPFN default is larger and may crash on this local FDE/MPS stack.
- `--tpt-device <cpu|auto|mps|cuda>` controls TPT_tab device; default is `mps`.
- `--tpt-fit-mode <mode>` controls TPT_tab preprocessing/cache mode; default is `fit_preprocessors`.
- `--tpt-n-estimators <n>` controls TPT_tab ensemble size; default is `1` for fast laptop validation.
- `--fde-root <path>` points to a local FDE or benchmark checkout.
- `--output-dir <path>` overrides the output directory; default is next to the dataset.
- Resource usage logging is enabled by default and writes `resource_usage.csv` next to `report.html`.
- `--resource-log-interval-seconds <seconds>` controls process-tree CPU/RSS sampling; default is `2.0`.
- `--no-resource-log` disables the default resource log.

Model weights:
- `--model-type tabpfn3` uses FDE foundation TabPFN3 regressor weights under `weights/tabpfn3/*regressor*.ckpt`.
- `--model-type tpt` uses FDE `TPTTabRegressor` with `$FDE_TPT_WEIGHTS_DIR/TPT_tab/model.ckpt`.

MPS execution:
- Prefer TabPFN3 on MPS for Apple Silicon runs.
- In Codex, Metal devices may be hidden inside the normal sandbox even when the Mac has a supported Apple GPU. If `--tabpfn-device auto` or `mps` reports MPS unavailable on Apple Silicon, verify with an escalated MPS smoke test and rerun the AutoResearch command with `sandbox_permissions="require_escalated"`.
- Do not silently fall back to CPU for Apple Silicon MPS runs. Use `--tabpfn-device cpu` only when the user explicitly wants CPU.

TPT_tab runs in an isolated child process. This avoids Metal/TPT runtime crashes caused by fitting TPT in the same Python process that just performed FDE feature extraction.

Resource logging records process-tree CPU percent and RSS memory. MPS runs may also append `mps_event` rows with PyTorch MPS allocated/driver memory at fit/predict stages. Apple GPU utilization and power require sudo `powermetrics`, so do not describe MPS memory rows as true GPU utilization percent.

Return the final `report.html` and `resource_usage.csv` paths and summarize the best mean R-squared score.
