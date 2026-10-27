# C3 tools/score_sim.py — scoring simulator (lane: Jules)
Goal: turn per-shape timings into the judge's board speedup, so we can predict a candidate's score before a check.
- Input: a profile file (format: `experiments/profiles/*.txt`, "scored shapes" table: case, sglang, yours, speedup, sol)
  or a JSON map {case: our_us}; plus the problem's TASK.md (loads, weights per shape).
- Score per TASK.md "Submitting/Shapes": load time = Σ shape time × weight; problem speedup = geometric mean over loads
  of sglang_load / ours_load; the judge subtracts ~4 µs graph replay cost from both sides per shape (state the
  assumption in the code). Also `--what-if case=us,...` to replace selected shapes and print the new speedup, and
  `--floor` to print the speedup if every shape hit its `sol`.
- Validate: on `experiments/profiles/muse_bf16_proj.728c9344dc3b4559.txt` the recomputed speedup must be within 0.01 of
  the file's `speedup` line (1.0022).
Done when: `python tools/score_sim.py experiments/profiles/muse_bf16_proj.728c9344dc3b4559.txt` → `speedup 1.00x`
(±0.01 of 1.0022) and `pytest -q tests/test_score_sim.py` green. Stdlib only (no numpy needed).
