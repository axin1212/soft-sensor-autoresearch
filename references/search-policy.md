# Search Policy

The search is rule-driven AutoResearch for quick local validation.

Core rules:

- Use three target-label-based holdout intervals when enough labels exist.
- Degrade to two or one holdout when label count is limited; fail below eight non-null target labels.
- Keep model input fairness by selecting exactly Top-32 features for every candidate when enough columns exist.
- Include raw identity, FDE trend/window, coverage-aware context sampling, and optional frequency features in the same Top-32 competition.
- Treat ICL context sample count as fixed by `--num-train-samples` for this skill; the sampler caps to available labels when fewer are available.
- Expand candidates across requested forecast horizons. For h>0, model features are anchored h sampling steps before the target label while the holdout target interval remains fixed. Exclude training pairs whose feature anchor time or target time falls inside the holdout interval.
- Run baseline on all holdouts, then use the worst baseline holdout as quick-screen.
- Do not early stop. Run until the requested time budget is essentially exhausted.
- If two rounds do not improve clearly, backtrack to a prior high-value node and explore another path.

Score candidates by direct mean R-squared across completed holdout runs:

```text
mean(completed_holdout_r2_values)
```

Do not subtract stability, floor, or missing-window penalties from the ranking score. Keep per-holdout R-squared values visible in the report so robustness remains inspectable without making the score opaque.

HTML report:

- Sort by mean R-squared score.
- Show actual values on x-axis and predictions on y-axis.
- Draw a 45-degree reference line in every subplot.
- Display each holdout R-squared value in the subplot title.
- Include failures with reason text in the candidate index.
- Include the forecast horizon step for every candidate row.
