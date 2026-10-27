# Open Kernel Challenge 2026 (Muse Spark Open): an AI-agent-driven GPU kernel campaign, archived

Four days (Oct 24–27, 2026), 243 commits, 86 judge runs, one human, a fleet of AI coding agents, no GPU of our own beyond an RTX 2060, against a leaderboard of 1,000+ teams
writing CUDA/Triton kernels for an H200. This repository is the honest record: what we tried, how we measured it, what worked,
what did not, and why we stopped.

> **Status (2026-10-27): archived.** After several days of iteration we concluded that, with our setup, we are not in a position to
> compete for the top of the board against teams with their own H200 nodes and deep kernel expertise. The work, data and method
> are published here as a portfolio record.

## The challenge
[Open Kernel Challenge](https://openkernelchallenge.dev) (kernel track, Oct 22 – Nov 4, 2026): make layers of the Meta *Muse Spark*
MoE model faster than SGLang's kernels on one NVIDIA H200. Each problem is one Python file (`kernel_fn`, optional untimed `prepare()`),
timed by the organisers' judge with CUDA-graph replay and a cold L2; the score is a weighted geometric-mean speedup vs SGLang.
Rules: no lower precision than SGLang, no caching across calls, no work outside the timed call. Limits: 10 scored submissions and
10–20 checks per problem per day. The problem files themselves (`okc init`) belong to the organisers and are not copied here.

## Results (final board entries)
| problem | what it is | our speedup vs SGLang | our rank | #10 | #1 |
|---|---|---|---|---|---|
| muse_attention_decode | FP8 paged attention, decode + 8-token verify, 80K context | **1.770×** | 313 | 2.62× | 3.03× |
| muse_decode_layer | the whole decoder layer (attention + router + NVFP4 experts + glue) | **1.433×** | 427 | 2.23× | 2.38× |
| muse_moe_w4a16 | routed experts, NVFP4 weights × BF16 activations | **1.486×** | 525 | 2.45× | 2.62× |
| muse_bf16_proj | the dense BF16 projections (skinny GEMV/GEMM) | **1.170×** | 799 | 1.56× | 1.64× |
| overall | | | **437** | | |
Ranks as of 2026-10-27 07:30 UTC. Full progression with judge job ids: [RESULTS.md](RESULTS.md).

## What worked
- **Measure before building.** Measurement-only submissions that run 10–40 variants inside the untimed `prepare()` turned one judge
  run into a full experiment (twins that delete one cost at a time). Best example: attention — the expected bottleneck (exp2/MUFU)
  turned out to be 7 %; the same run found a 1.13–1.16× tile-size win → board 1.689 → 1.770 the same morning.
- **Predict, then kill.** Every judge run carried a predicted number and a kill line; a promote script submitted only when the check
  beat the board including its confidence interval. Nothing regressed on the board.
- **Porting wins across problems** (attention kernel into the full layer: verify shapes −4…−8 %).
- **Tooling:** a judge wrapper that logs every run with hypothesis + prediction, a promote gate, CUPTI trace tools, a PTX audit that
  compiles with the judge's real launch attributes, a kernel-chain analyzer.

## What did not work (and the reading that killed it)
- **Warp-specialised MoE GEMM** (producer bulk-copy warp + 12 mma consumer warps, mbarrier ring): three builds, three judge readings —
  copy-issue bound, then consumer bound, then a ring-protocol floor ≈ the old kernel's rate. Parked.
- **exp2 polynomial split in attention** (FA4-style): −12…−21 %; the exp was never the bottleneck.
- **Early-launch (PDL) "tools before wait" prefetch** in the layer: +2 µs light, −3.9 µs full (congestion) — a nice confirmation of a
  construction-site analogy, not a win.
- **A WSTREAM-style tensor-core GEMV** for the BF16 projections: 0.2–2.5 TB/s vs 2.6–3.7 TB/s for the existing kernels.
- **Kernel fusions in the layer glue:** measured ≈ 0.
Full list with numbers: [HYPOTHESES.md](HYPOTHESES.md).

## Why we stopped
The leaders are 1.4–1.7× faster than us on every problem, and the gap kept widening while our own numbers rose (overall rank
374 → 437 in a day). The limiting factors were structural, not effort: no local H200 (every Hopper experiment was a remote judge run
with a 16 KB log tail), a dev card without Hopper features (wgmma/TMA/PDL untestable locally), and model-generated performance
predictions that were often off by 1.5–2×. See [LESSONS.md](LESSONS.md).

## How the work was done
One human (direction and decisions) + AI agents in defined lanes: a director agent (planning, review, merging, judge runs), builder
agents in isolated git worktrees, a design reviewer for theory-only briefs, an asynchronous coding agent for pure-Python tools, and
two independent reviewers (red team) that had to ACK attention submissions. Every round was anchored in a handover log so work
survived usage-limit stops. Details: [METHOD.md](METHOD.md).

## Repository map
| path | content |
|---|---|
| `kernels/<problem>/` | every submitted version (`sonnet_v*.py`, `director_v*.py`), measurement-only files (`measure_*`), trace files |
| `tools/` | judge wrapper (`okc_run.py`), promote gate, breakdown/trace/PTX-audit/chain analyzers, generators of measurement files |
| `tests/` | the local test suites (run on an RTX 2060; Hopper paths are audited via sm_90 compiles) |
| `data/judge-runs/` | `log.jsonl` (all 86 judge runs: hypothesis, prediction, verdict) + per-run judge profiles |
| `data/measurement-logs/` | raw log tails of the measurement runs (S/B/D lines) |
| `design/breakdown/` | per-problem time breakdowns and the result write-up of every experiment |
| `design/ai-design-reviews/` | theory-only design documents produced by an AI reviewer (with later errata) |
| `design/research/` | method research (George Hotz / tinygrad working style) and a construction-trade analogy used to generate hypotheses |
| `design/redteam/`, `design/task-cards/` | review records and the task cards given to builder agents |
| `design/untrusted-drafts/` | drafts from secondary AI lanes, kept as produced (unverified) |

License: Apache-2.0 (the challenge licenses submissions under Apache-2.0).
