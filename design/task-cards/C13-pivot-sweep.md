# C13 tools/pivot_sweep.py — pivot-ratio measurement library for prepare() sweeps (lane: Jules; source: fable-07 D4-2)
Engram's "remove the shared component" lesson applied to our judge measurements. A pure-python+torch helper that
measurement-only files import (copy-paste into the file — submissions are one file, so keep it ≤ 150 lines, no deps).
API: `sweep(variants: dict[name, callable], make_input, pivot: name, rounds=7, reps=5, flush_bytes=128<<20) -> results`
- Each callable is captured once as a CUDA graph; each round runs pivot and all variants in round-robin order with a
  read-only L2 flush before every replay (sum over a preallocated buffer), timing with CUDA events.
- Score per variant = median over rounds of (pivot_time / variant_time) — the judge's paired-ratio form; also report
  per-round pivot deviation (shared drift) and a bootstrap 95% CI over rounds; `rank()` refuses to order variants whose
  CIs overlap. `to_log(results) -> str` gives a compact, ≤ 80-char-per-variant table.
- CPU fallback for tests: when CUDA is unavailable use time.perf_counter and no flush (logic only).
Done when: `pytest -q tests/test_pivot_sweep.py` green (CPU: synthetic callables with known relative costs and injected
per-round drift → ranking correct, drift removed, CI overlap refusal works). Write scope: tools/pivot_sweep.py,
tests/test_pivot_sweep.py only.
