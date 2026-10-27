# Method: how the campaign was run

## Constraints that shaped everything
- **No local H200.** The only Hopper GPU was the organisers' judge: 10 scored submissions + 10–20 checks per problem per day, results
  as a per-shape table, a kernel profile and a 16 KB tail of our own stdout.
- **Dev card: RTX 2060 (sm_75).** Correctness tests ran locally; Hopper-only features (wgmma, TMA, mbarrier, PDL) were compiled for
  sm_90 and audited in PTX/ptxas output, but could only be *timed* on the judge.

## The loop
1. **Breakdown first.** Per problem and shape: our µs, the memory floor, per-kernel time and the weighted loss vs the floor
   (`tools/breakdown.py`, `design/breakdown/`). Work goes where weight × ln(ours/floor) is largest.
2. **Hypothesis with a number.** Every judge run carries `--hyp` and `--predict` (and a range); `tools/okc_run.py` refuses runs without
   them and appends hypothesis, prediction, verdict and job id to `data/judge-runs/log.jsonl`.
3. **Measurement-only runs.** The untimed `prepare()` (≤ 300 s) runs a study: the board kernel plus twins/variants, 3 seeds, a
   read-only 128 MB L2 flush, CUDA-graph replay, replay overhead subtracted, interleaved rounds. Output is compact `S`/`B`/`D` lines that
   fit the 16 KB log tail; the `D` line applies a pre-written decision table. The wrapper refuses to *submit* such files.
4. **Decision tables and kill lines written before the run** (e.g. "kill if GEMM1 bs256 > 205 µs").
5. **Promote gate** (`tools/promote.py`): submit only if the check beats the board *and* its CI lower bound beats the board's.
6. **Red team for attention:** two independent reviewers had to ACK a rules check (no caching, no work outside the timed call,
   precision, exact routing) before submission (`design/redteam/`).
7. **Two strikes:** an idea that fails twice is parked unless a new reading changes the picture.
8. **Read the generated code:** `tools/ptx_audit.py` compiles with the judge's real launch attributes. An earlier probe without them
   reported wrong facts (no pipelining, small smem) — retracted once audited.

## The agent team
| lane | role |
|---|---|
| human (Dex) | goals, priorities, every outward-facing decision |
| director agent | plans, writes task cards, reviews, merges (only with green tests), runs the judge, keeps the handover log |
| builder agents (2 at a time) | implement one card each in an isolated git worktree; never push, merge or touch the judge |
| design reviewer (expensive model, theory only) | one curated brief per major idea, shown to the human first |
| async coding agent | pure-Python tools from GitHub (e.g. `tools/chain.py`, trace report) |
| secondary models | red-team ACKs, idea lists, fusion maps — treated as *untrusted* until checked in code (`design/untrusted-drafts/`) |
Work survived usage-limit stops because every round was anchored in a handover log (prediction, observation, next move) and builders
committed WIP after each green step.

## Idea sources
Profiles and twins first; public papers and libraries (FlashAttention-3/4, CUTLASS, Marlin, Machete); a study of George Hotz's
tinygrad working style (`design/research/hotz-method/`: ceiling first, bisect by deleting, read the generated code); and a
construction-trade analogy worked out with a builder (`design/research/construction-analogy-*.md`) — "how much one person carries
per trip" led to the attention tile win; "the crew that leaves goes first" predicted the prefetch congestion we then measured.

## Tooling (in `tools/`)
`okc_run.py` judge wrapper + log · `promote.py` gate · `breakdown.py` loss per shape · `make_trace.py`/`parse_trace.py`/`trace_report.py`
CUPTI timelines · `ptx_audit.py` PTX/ptxas audit with real launch attributes · `okcdbg.py` + `ratchet.py` regression ratchets ·
`chain.py` kernel-chain analyzer · `gen_measure_*.py` generators that build measurement files from the board kernel's own source by
exact-match substitution (so twins cannot drift from the code they measure).
