---
name: soft-sensor-autoresearch
description: Run local offline soft-sensor AutoResearch with FDE TabPFN3, SISSO-style derived features, robust holdout scoring, and an interactive Plotly fitability report. Use only when the user explicitly names $soft-sensor-autoresearch or asks to run the soft-sensor AutoResearch skill.
---

# Soft Sensor AutoResearch

Use this skill only after explicit invocation.

Required user inputs:
- Dataset file path (`.csv` or `.parquet`)
- Target column name

Before running, check local FDE/TabPFN availability. Do not silently fall back to XGBoost if TabPFN3 is unavailable.

Read `references/fde-integration.md` when FDE discovery, TabPFN weights, or environment checks are relevant.
Read `references/search-policy.md` when explaining or modifying the search strategy.

Run:

```bash
python scripts/run_soft_sensor_autoresearch.py <data-file> <target-column>
```

Useful options:
- `--time-budget-minutes <minutes>` controls the search budget; default is 15.
- `--fde-root <path>` points to a local FDE or benchmark checkout.
- `--output-dir <path>` overrides the output directory; default is next to the dataset.

Return the final `report.html` path and summarize the best robust score.
