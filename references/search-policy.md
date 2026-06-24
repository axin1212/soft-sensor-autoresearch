# Search Policy

The search is rule-driven AutoResearch for quick local validation.

Core rules:

- Use three target-label-based holdout intervals when enough labels exist.
- Degrade to two or one holdout when label count is limited; fail below eight non-null target labels.
- Keep model input fairness by selecting exactly Top-32 features for every candidate when enough columns exist.
- Include raw identity, recent/coverage context, FDE trend/window, and optional frequency features in the same Top-32 competition.
- Treat ICL context sampling as a first-class search dimension. After the uniform identity baseline, probe identity features with recent and coverage context before adding trend/window features.
- Keep ICL context sample count fixed by `--num-train-samples` for the whole run; the sampler caps to available labels when fewer are available.
- Run baseline on all holdouts, then use the worst baseline holdout as quick-screen for low-risk candidates.
- Do not early stop. Run until the requested time budget is essentially exhausted.
- If two rounds do not improve clearly, backtrack to a prior high-value node and explore another path.
- If all low-risk candidates have strongly negative worst-holdout R-squared, audit preprocessing first: leakage columns, synchronized duplicate tags, missing-value handling, natural sampling level, downsampling/aggregation, and holdout target distribution shift.

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
