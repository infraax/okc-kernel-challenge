# C7 tools/diff_rules.py — serving patch rule-checker (lane: Jules)
Goal: refuse a serving patch that the organisers would refuse, before we spend one of 3 daily checks.
Rules (design/rules-2026-10-22.1.md "Serving" + the API rules below): unified git diff against SGLang a55ab7d5; only
paths under `python/sglang/`; never under `python/sglang/srt/model_loader/`; suffixes only .py .cu .cuh .h .hpp .c .cc
.cpp .json; no renames, mode changes, binary patches; ≤ 737280 bytes; no added line > 400 chars; none of the forbidden
words: kc-secret KC_RESULT KC_SERVING KC_GATE /kc-job result_secret serviceaccount SIMULATE.
Also warn (not fail) on: changes to DFlash weight loading, the expert count (8), the sliding window (2048), any norm
removal (lines removing `rmsnorm`/`RMSNorm`), and on any global/cross-request cache added in speculative/ngram code.
CLI: `python tools/diff_rules.py patch.diff` → `diff_rules: OK (<bytes> bytes, <files> files)` or one `FAIL <rule>: <detail>`
per violation, exit 1. Stdlib only. Tests: `tests/test_diff_rules.py` with small synthetic diffs per rule.
Done when: `pytest -q tests/test_diff_rules.py` green.
