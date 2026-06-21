# Search Policy

The search is rule-driven AutoResearch for quick local validation.

Core rules:

- Use three target-label-based holdout intervals when enough labels exist.
- Degrade to two or one holdout when label count is limited; fail below eight non-null target labels.
- Keep model input fairness by selecting exactly Top-32 features for every candidate when enough columns exist.
- Include SISSO-style derived candidates and FDE trend/frequency features in the same Top-32 competition.
- Run baseline on all holdouts, then use the worst baseline holdout as quick-screen.
- Do not early stop. Run until the requested time budget is essentially exhausted.
- If two rounds do not improve clearly, backtrack to a prior high-value node and explore another path.

Score candidates by robust score:

```text
mean_r2 - 0.5 * std_r2 - 0.25 * max(0, 0.0 - min_r2)
```

Partial candidates receive an additional penalty of `0.1 * missing_holdout_count`.

HTML report:

- Sort by robust score.
- Show actual values on x-axis and predictions on y-axis.
- Draw a 45-degree reference line in every subplot.
- Display each holdout R-squared value in the subplot title.
- Include failures with reason text in the candidate index.
