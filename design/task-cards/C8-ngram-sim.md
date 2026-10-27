# C8 tools/ngram_sim.py — request-local n-gram draft acceptance simulator (lane: Jules)
Question it answers: how many output tokens of agentic-coding turns can be drafted from the *same conversation's* earlier
tokens (prompt lookup), on top of a drafter with ~4.2 accepted tokens per step? Decides whether serving lever L4 is worth it.
- Input: multi-turn agentic coding transcripts (JSONL: list of {role, content}); write a loader for at least one public
  dataset available on Hugging Face (name it in the README section of the file; e.g. SWE-agent / OpenHands trajectories)
  plus a tiny synthetic fixture under tests/fixtures/ for tests.
- Tokenize with a public Llama-class tokenizer if available offline, else a whitespace+punct fallback (flag it).
- For each assistant turn: simulate prompt-lookup drafting with n ∈ {2,3,4,5}, draft length k ∈ {4,8}, using only the
  conversation so far + the turn's own generated prefix (never other conversations). Report: fraction of output tokens
  covered by accepted drafts, mean accepted length per step, and the same restricted to the first 220 output tokens.
CLI: `python tools/ngram_sim.py <jsonl> [--n 3] [--k 8] [--limit N]` → a table + a one-line summary.
Done when: `pytest -q tests/test_ngram_sim.py` green on the fixture (hand-computed expected acceptance), and the file's
docstring explains how to run it on the public dataset. Deps allowed: `transformers` (tokenizer) optional, else stdlib.
