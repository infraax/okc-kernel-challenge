# C2 tools/local_check.py — local correctness harness (lane: Sonnet sub-agent, dev box)
Goal: run a candidate submission against the organisers' own reference/checks locally, so judge checks are spent
only on candidates that already pass.
- CLI: `.venv/bin/python tools/local_check.py <problem> <file> [--device cpu|cuda] [--cases tests|scored|small|all] [--seed N]`
- Cases: parse the "must pass these N tests" block and the scored-shapes table from `problems/<p>/TASK.md`
  (strings like `m 7, layer pooler` / `batch 16, ctx 80000, heads tp8, layer full, ...`, `dist outliers`) into the
  kwargs of the problem's `generate_input` (and weights generator where the problem has weights). `small` = same
  cases with ctx / m / batch scaled down so CPU finishes in < 2 min per problem.
- Run: import the candidate as a module (fresh per problem), call `prepare(weights)` when the problem has weights,
  then `kernel_fn`, then the reference's `check_implementation` (or the equivalent check entry point) exactly as the
  judge would. Print one line per case `PASS|FAIL <case> <message>` and a summary `local_check: <p> <k>/<n> passed`.
  Exit 1 on any failure.
- Also `--graph` (cuda only): capture kernel_fn in a CUDA graph once to catch host syncs / value-dependent allocation.
- Never modify `problems/**`. Import the reference only from the harness, never from a submission.
Done when (on the dev box):
  `.venv/bin/python tools/local_check.py muse_bf16_proj problems/muse_bf16_proj/submission.py --device cpu --cases small`
  → `local_check: muse_bf16_proj n/n passed`, and the same for the starter `submission.py` of all 5 problems
  (`--cases small`, cpu), plus `pytest -q tests/test_local_check.py` green (a deliberately wrong kernel must FAIL).
