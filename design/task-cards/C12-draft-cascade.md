# C12 tools/draft_cascade.py — engram-style 4-stage draft cascade simulator (lane: Jules; cross-check: Grok Build)
Context: serving track (SGLang + DFlash drafter, stock ≈ 4.2 accepted tokens/step on multi-turn agentic coding). Extra
drafting may only use the request's own tokens (no cross-request state). Idea source: Dex's engram 4-stage recall
(design/engram-v2-analysis.md §5). Existing baseline tool: tools/ngram_sim.py (plain prompt lookup).
Build tools/draft_cascade.py that simulates, per assistant turn, drafting from the conversation so far + the turn's own
generated prefix, with a cascade:
  S1 recall: all prompt-lookup matches for the last n tokens (n in {2..6}), candidate continuations (k ≤ 8 tokens).
  S2 margin: if one continuation dominates (longest suffix match, most recent, or count margin ≥ τ) → draft k tokens;
     if ambiguous → shorter draft (k_amb) or none.
  S3 confusion: within the same request, a continuation the verifier just rejected at the same context is demoted.
  S4 fallback: no confident candidate → "DFlash" (model it as a fixed acceptance distribution with mean 4.2 tokens,
     configurable) — report also the n-gram-only and DFlash-only baselines.
  Calibration: fit a table P(accept | match length bin, margin bin, last accepted run) on a TRAIN split, freeze it,
  choose draft length by expected accepted tokens on the TEST split (engram methodology: out-of-sample only).
Data: write a loader for a public multi-turn agentic-coding trajectory dataset from Hugging Face (verify the dataset
exists and its license; e.g. SWE-agent/OpenHands-style trajectories), converting to the JSONL {role, content} format of
tools/ngram_sim.py; tokenizer: a public Llama-class tokenizer if downloadable, else the fallback (flag it).
Metrics per policy: mean accepted tokens per verify step, fraction of output tokens drafted, restricted to the first 220
output tokens per turn too; report by turn index and by copy-heavy vs not.
Done when: `pytest -q tests/test_draft_cascade.py` green on a synthetic fixture with hand-computed expectations, and the
file's docstring has the exact command + a results table from a run on ≥ 200 public conversations (train/test split),
with the dataset name and license. Write scope: tools/draft_cascade.py, tests/test_draft_cascade.py,
tests/fixtures/cascade_*.jsonl, experiments/draft_cascade_results.md. No other files; deps: datasets, transformers (opt).
