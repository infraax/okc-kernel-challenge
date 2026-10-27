# Trace format v1 — exact per-kernel timeline on the judge H200 (Dex 2026-10-24: "full log breakdown, every step indexed, counted, timed")

Two levels, one line format. All lines are emitted by a MEASUREMENT-ONLY file's prepare() (untimed, ≤ 300 s) to stdout;
the judge returns the last 16 KB of the log, so lines are compact and names are interned.

## Level 0 — free (tools/breakdown.py, no judge run)
From the judge's own profile JSON: per shape wall / sol / sglang, run spread, top-12 kernels, hidden tail, idle = wall − Σkernels.
Limits: the judge profiles in a separate pass (idle can go negative), the tail beyond 12 kernels is hidden, no ordering, no gaps.

## Level 1 — timeline (tools/make_trace.py → kernels/<p>/trace_<stem>.py, one `okc check` per problem)
For each judged case (same case dicts + seeds as the judge JSON): inputs from the problem's generate_input recipe, kernel_fn
captured in a CUDA graph exactly like the judge, read-only L2 flush, then R profiled replays (torch.profiler / CUPTI kernel
activity). Per replay, every kernel's start/end (ns, relative to the first kernel's start), stream id, grid/block if known.
The emitted numbers are medians over the R replays per kernel position.

Lines (space separated, µs with 2 decimals):
```
TV 1 <problem> <file> torch=<ver> cuda=<ver> gpu=<name> cupti=<ok|fail:reason> replays=<R>
TN <name_id> <kernel name, truncated to 90 chars>
TS <shape_idx> w=<weight> wall=<median graph replay µs, cuda events> span=<first start → last end µs> n=<kernels> label=<k=v,k=v>
TK <shape_idx>.<pos> <name_id> s=<start µs> d=<dur µs> g=<gap before µs, vs previous end on ANY stream> st=<stream> o=<overlap µs with other streams>
TG <shape_idx> gaps=<Σ gap µs> head=<graph launch → first kernel µs> tail=<last end → replay end µs> busy=<union of kernel intervals µs>
TE <shape_idx> <error text>            (case failed; the run continues with the next case)
TZ done <seconds used>
```
Notes (as implemented by tools/make_trace.py + tools/trace_runtime.py):
- `st=` is omitted when 0 and `o=` when 0.00 (parsers default them to 0). `TS` also carries `rep=<ok>/<R>` (replays that shared the modal
  kernel sequence). `label` has no spaces. `TV` cupti is `graph`, `eager` (CUPTI saw no kernels inside graph replays: eager kernel_fn calls are
  profiled instead, head/tail then mean launch latency, not graph launch), or `fail:<why>` (TS lines carry wall only, n=0).
- `wall` = median of the cuda-event times of the same profiled replays the timeline comes from (graph mode). Positions: kernels, memsets and
  memcpys, in start order; `st` = stream rank by first appearance (0 = the first kernel's stream).
- Size: the whole block is built in memory; above `_TR_MAX_BYTES` (14000) the lowest-weight shapes are dropped (later index first on ties)
  with `TE <idx> dropped:size`; a lone shape still too long keeps its first kernels (`TE <idx> truncated:size n=<total>`). Cases that did not
  fit the time budget get `TE <idx> dropped:budget`; failed cases `TE <idx> <Exception>:<text>`.
- `decode_layer`: `local_start` is not in the judge JSON case; the trace takes the 32-expert range that the kernel's own top-8 ids of one eager
  call send the most pairs to (the judge uses the reference's routing for the same input).

Invariants checked by the parser: span ≈ busy + gaps (± overlap), wall ≥ span, Σd ≥ busy, positions are 0..n-1 in start order.

## Level 2 — inside our own kernels (only for the top bottleneck kernels Level 1 names)
`%globaltimer`/`clock64` stamps at phase boundaries (load, dequant, mma, reduce, store) written to a debug buffer by CTA 0 and
one CTA per SM-quarter; lines `TP <shape>.<pos> <phase> <µs from kernel start> <cta>`. SGLang/cuBLAS kernels cannot be
instrumented → microbench variants instead (Level 1 on isolated calls).

## Roofline column (bytes model, static)
Per kernel family per shape: bytes that must move (weights, KV, activations) and FLOPs → floor µs at 4.0 TB/s (H200 HBM3e
measured-achievable, not 4.8 peak) and 989 TFLOP/s BF16 dense. efficiency = floor / measured d.
