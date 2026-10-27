# Lessons

## What I would keep
- **Measurement-only runs with twins** were the most valuable idea of the campaign: one judge run answered "where does the time go"
  for a whole kernel, and twice found a win nobody had predicted (attention tiles, pooler/router co-residency).
- **Numbers before builds, kill lines before runs, a promote gate.** No regression ever reached the board; dead ends died after one
  or two runs instead of days.
- **Untrusted until checked in code.** Secondary-model claims were verified against the source before any build; one of the five
  "quick wins" was refuted that way before costing a judge run.
- **Small, isolated, test-gated changes** from builder agents (generated files, exact-match substitutions, bitwise tests vs the
  board kernel) kept a fast-moving codebase correct.

## What I would change
- **Get the hardware first.** Without a local H200 every Hopper idea was debugged blind through a 16 KB log tail; teams iterating
  locally with profilers run hundreds of experiments per day where we ran ~20. This, not effort, set the ceiling.
- **Distrust first-principles performance models on new hardware.** Model-written predictions for Hopper-specific designs were off by
  1.5–2× (warp-specialised GEMM, tensor-core GEMV); predictions anchored in measured medians were accurate to ~1 %.
- **Decide earlier which race you are in.** Incremental tuning gave ~0.5–1 % per day while the field moved faster; the leaders' 1.4–1.7×
  margin implies a different structure, not better configs. A stop/continue checkpoint after day 2 would have saved effort.
- **Budget the orchestration.** Running many agent lanes in parallel hit usage limits twice; two builders plus cheap lanes was the
  sustainable rate.

## Conclusion
With our setup (no local Hopper GPU, an AI-agent team, three to four days of judge access) we reached the middle of a 1,000+ team field
on every problem (attention 1.77×, decode layer 1.43×, MoE 1.49×, BF16 projections 1.17× vs SGLang) and documented every step. We
are not in a position to compete for the top of the board, and archived the work here instead of spending more on diminishing returns.
