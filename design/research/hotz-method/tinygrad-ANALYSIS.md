# tinygrad study: ANALYSIS (delivered inline)

NOTE FOR CALLER: I could not create `<workspace>`. The Write tool rejected it ("Subagents should return findings as text, not write report files"), and I did not work around that. The full analysis is below, so please save it as ANALYSIS.md if you want the file. What does exist on disk, all under `<workspace>`:
- `repo/` is the clone. It is at commit `d16046f` (2026-10-24, v0.14.0), and I deepened the history to 2024-06 (10879 commits).
- `.venv/` holds tinygrad installed with numpy and tabulate.
- `exp/*.py` are my experiment scripts.
- `exp/out/*.txt` holds the raw outputs quoted below (DEBUG 1-7, rangeify trace, VIZ/CLI output, a CUDA C render for sm_90 via the NULL device, and a static per-kernel report).
- `exp/nullcuda.py` renders CUDA source for sm_90 without a GPU or NVRTC, by stubbing the compiler on `DEV=NULL`.

All runs were CPU only (`DEV=CPU:LLVM`, `CPU:X86`, `NULL`, `PYTHON`). I never touched the GPU, `~/.config` or `<secrets>`. Paths below are relative to `repo/`; line numbers are for `d16046f`.

**Version caveat.** Most blog posts and tutorials, including mesozoic-egg's notes and the old docs, describe a 2024/early-2025 tinygrad. That tinygrad had `LazyBuffer`, `ShapeTracker`, `Kernel`, `linearizer.py` and `ScheduleItem`. None of these exist now. The current design is "rangeify": indices are RANGE UOps, kernels are `CALL(SINK)` inside a `LINEAR`, and even the JIT command queue is compiled UOps. `docs/env_vars.md` is stale too (see 2.1). Treat the code as the source of truth.

## 0. Summary

1. tinygrad is a 25.5k-line ML compiler where everything is one IR (UOp) and every transformation is a `graph_rewrite` with a `PatternMatcher`.
   - The pipeline is Tensor, lazy UOp graph, `schedule/` (rangeify decides kernel boundaries), `codegen/` (about 35 named rewrite passes, with BEAM or heuristics inside), `renderer/` (C-like text, LLVM IR, PTX, AMD asm, x86), then `runtime/`.
   - The runtime talks to the device through its own command queues. The JIT is "compile the launch sequence into a kernel".
2. Observability is good because every pass and every kernel is a printable, nameable UOp graph, and every kernel carries a static `Estimates(ops, lds, mem)`.
   - The `DEBUG=2` line gives, per kernel, time, GFLOPS, unique-bytes GB/s and load/store-bytes GB/s.
   - Kernel names encode launch and loop shape, e.g. `r_16_128_4_32`.
3. Speed tooling is a chain of five things:
   - heuristics (`hand_coded_optimizations`);
   - optional BEAM, which times real compiled variants on the device (L2 flush, min of 3, early stop, disk cache);
   - TinyJit replay;
   - tests that assert exact kernel counts and memory (ratchets);
   - process replay, which diffs generated source between master and a PR.
4. Recompute-vs-store ("computed twice") is a coded policy: `remove_bufferize`, `schedule/rangeify.py:48-95`.
   - A producer is recomputed in each consumer if it reads at most 3 distinct buffers and no reduce inside it reads a buffer. Otherwise it is stored.
   - I printed the result: in a softmax-like block `exp2` appears in two kernels.
5. For okc, adopt:
   - the cumulative level ladder with a smoke run at max level;
   - per-kernel declared bytes/flops giving `%` of roofline and a stored-vs-all-traffic ratio;
   - shape-encoded kernel names;
   - kernel-count and bytes ratchet tests;
   - PTX/SASS-hash diffing as the analogue of process replay;
   - the BEAM harness (not the search space) inside `prepare()`.

   Do not adopt the lazy graph, rewrite-rule fusion or the own-driver machinery.

---

## 1. Architecture as it is today

### 1.1 Pipeline with anchors

```
Tensor (tensor.py, 549 lines; op methods live in mixin/*.py, ~6.5k lines, shared by Tensor and UOp)
  builds UOp graph lazily (uop/ops.py: class UOp :229, hash-consed via UOpMetaClass :191)
  Tensor.realize() -> Tensor.linear_with_vars
schedule/__init__.py
  transform_to_call :257        buffers become PARAMs, so graph structure is the schedule-cache key
  create_linear_with_vars :268
  lower_sink_to_linear :121     schedule cache (SCACHE=1 memory, 2 disk); prints "scheduled N kernels ... cache hit / CACHE MISS"
    prepare_rangeify (prepare.py:258): multi_pm, inline calls, "earliest rewrites" (prepare.py:188, includes split_reduceop :93)
    get_kernel_graph (rangeify.py:342)
      run_rangeify (indexing.py:181)      assign RANGEs backwards from consumers, decide realize points
      symbolic + reduce_collapse + pm_remove_bufferize (rangeify.py:48) = recompute-vs-store cost function
      pm_limit_bufs (:176), fix_store_hazard (:190)
      pm_add_buffers (bufferize_to_store :205): STAGE -> ALLOC+STORE
      split_kernels (split_store :324): CALL(SINK(KernelInfo()), *buffers)
    create_schedule :31         topological order with RAW and WAR dependencies -> Ops.LINEAR(CALLs)
  memory_plan_rewrite (schedule/memory.py:19)   TLSF arena for intermediates (JIT path)
engine/realize.py
  run_linear :282 = compile_linear :273 + link + per-call exec; track_stats :61 prints the DEBUG>=2 lines
  lower_and_compile :236        dedup by key, parallel worker pool (engine/worker.py)
codegen/__init__.py
  to_program :513 / do_to_program :473 / full_rewrite_to_sink :267
  apply_opts (codegen/opt/postrange.py:239) = NOOPT | explicit opts | BEAM | hand_coded_optimizations
  pm_to_program: linearize -> do_estimates (:442) -> render/assemble -> compile_cached (sqlite)
renderer/   cstyle.py (CUDA/HIP/Metal/OpenCL/C), ptx.py, llvmir.py, nir.py, wgsl.py, isa/x86.py, amd/ (own assembler)
runtime/    ops_cuda.py (126 lines, driver API), ops_nv.py, ops_amd.py (direct to hardware), ops_null.py; support/hcq2.py (command-queue UOps)
```

### 1.2 Everything is a UOp

- There are 75 `Ops` in `uop/__init__.py`. `sz.py` prints "ops: 75" and "flags: 53".
- A Tensor wraps one UOp. Because op methods are shared mixins, `a+b` builds either a tensor graph or a kernel body. That is how `Tensor.custom_kernel` and the `extra/` kernels are written.
- UOps are interned (hash-consed), so equal subgraphs are the same object. Rewrite dedup, the schedule-cache key (`UOp.key`) and the compile-cache key are hash lookups. This is global value numbering for free.
- A kernel is `CALL(SINK(arg=KernelInfo(name, opts_to_apply, beam, estimates)), *bufs)`. A schedule is `LINEAR(*CALLs)`. After codegen it becomes `PROGRAM(SINK, LINEAR(uops), SOURCE, BINARY)` with `ProgramInfo` (ops.py:1331).
- `graph_rewrite` (ops.py:1873) has two engines:
  - `unified_rewrite` (:1801) runs to a fixpoint, bottom-up or top-down, with a waitlist.
  - `walk_rewrite` (:1775) is a single MLIR-style walk.
- Patterns are `UPat` trees compiled to Python source on first use (`UPAT_COMPILE`), indexed by root op, with an early-reject set of required child ops (ops.py:1569-1595).
- Passes are named, e.g. `graph_rewrite(..., name="devectorize2")`. The `@rewrite_group` decorator (ops.py:1643) records each pass when `TRACK_MATCH_STATS>=2` (`VIZ` sets it). That naming underlies all the tooling.

### 1.3 The scheduler: what is fused into one kernel and why

Fusion is whatever remains after deciding where to materialize (`schedule/indexing.py: run_rangeify`, :181-300).

1. **Always realized.**
   - `pm_generate_realize_map` (:43) always realizes STOREs.
   - It also realizes the srcs of MSELECT/MSTACK and the inputs of custom-kernel CALLs.
   - `ALWAYS_CONTIGUOUS` (:24) is buffers, AFTER, CONST, LOAD and CALL.
2. **Range assignment, outputs to inputs.**
   - A realized node gets new ranges, which starts a new kernel loop nest.
   - A node with one consumer inherits the consumer's ranges, so it is fused into that kernel.
   - A node with 2 or more consumers shares ranges only if all consumers use identical index expressions. Otherwise it gets new ranges and is (partially) realized.
   - Movement ops are eliminated by rewriting ranges through them (`apply_movement_op` :163).
   - Reshape uses `_apply_reshape` plus the symbolic simplifier, which the code comment says is "doing a lot of heavy lifting". It replaces ShapeTracker view merging.
   - REDUCE creates ranges for its reduced axes. A node after a reduce that needs the reduce's ranges "ended" forces a realize ("fusion decision: REDUCE before the broadcast"). That is why `x - x.mean()` is 2 kernels (`test_reduce_broadcast_not_recomputed`, `test/null/test_schedule.py:352`).
3. **Cost function** `remove_bufferize` (`schedule/rangeify.py:48-95`). Each inserted STAGE is deleted, meaning the producer is recomputed in each consumer, unless one of these holds:
   - it is user `.contiguous()` (`removable=False`);
   - the producer reads more than 3 distinct buffers;
   - a REDUCE inside the producer reads a buffer.

   History: commit `360980f1a` (PR #12360, 2025-09-30) introduced the buffer-count rule with threshold 2. The current threshold is 3.
4. `pm_limit_bufs` (:176) forces a STAGE when a kernel would exceed `MAX_KERNEL_BUFFERS` (WEBGPU: 8). `fix_store_hazard` (:190) stages a source that loads from the buffer being stored at a different index.
5. `split_store` (:324) cuts kernels at STORE/END and renumbers ranges "so kernel deduping works".
6. `split_reduceop` (`prepare.py:93`) splits a reduce with few outputs and at least 32768 inputs into two reduces with a `.contiguous()` between them. This is the only place the scheduler adds a kernel for parallelism. `DEBUG>=3` prints `split N: shape -> shape -> shape`.

Verified results (2.6):
- `(x@w).relu().sum()` is 1 kernel.
- A softmax-like block is 4 kernels, with `exp` recomputed in 2 of them and the matmul result stored (3 consumers).
- `test/null/test_schedule.py` has 288 tests that pin these counts, and many carry `# TODO: should be 1?`.
- `docs/developer/speed.md` admits there is "no systematic way" to decide recompute vs store.

### 1.4 Codegen: `full_rewrite_to_sink` (codegen/__init__.py:267-395)

The named passes, in order:
- multi_pm
- early movement ops
- load collapse
- split ranges
- initial symbolic
- simplify ranges
- **apply_opts**
- postopt symbolic
- expander (UPCAST/UNROLL to vector shapes, WMMA expand)
- remove reduces (REDUCE to a REG accumulator; `fix_group_for_reduce` makes a LOCAL-memory two-stage reduce)
- add local buffers
- group gpudims (RANGE to blockIdx/threadIdx, `codegen/gpudims.py`)
- expand broadcast / add loads
- devectorize2
- early symbolic
- memory_coalescing (`late/coalesce.py`, merges scalar loads into vector loads)
- extra symbolic
- lower index dtypes
- final symbolic
- early/late decompositions (renderer-dependent, includes transcendentals and fast idiv)
- decomp dtypes
- move gates from index
- final rewrite (renderer `extra_matcher`)
- add implicit barriers (LOCAL memory RAW/WAR becomes `__syncthreads`)
- range to special
- add control flow (`late/linearizer.py`)
- number variables

After these, `pm_to_program` runs linearize, estimates, render and compile.

The Opt layer (`codegen/opt/postrange.py`, class `Scheduler` :14, distinct from the tensor scheduler) rewrites RANGE axes of one kernel.
- Axis types are DEVICE, GLOBAL, LOCAL, WARP, WEAK/LOOP (serial) and UPCAST.
- `OptOps` are SPLIT(axis, (amt, target_type[, top])), TC (WMMA), PADTO and SWAP. One SPLIT op covers upcast, local and group-for-reduce.
- Limits: upcast at most 16 (reduce unroll at most 32), shared memory at most `ren.shared_max`, and tensor-core opts must come first.
- `hand_coded_optimizations` (`opt/heuristic.py:8`) goes in this order:
  - tensor cores;
  - the matvec special case (4 block x 8 threads per row x 4 rows per thread);
  - "grouping" (SPLIT 16 LOCAL top=True) when the output is small;
  - upcast small masked axes;
  - upcast by a stride heuristic up to 32;
  - unroll a small last reduce;
  - local sizes preferring expand axes, at most 128 threads.

### 1.5 Renderers and runtime

- `cstyle.py` emits `extern "C" __global__ void __launch_bounds__(N) name(float* data0_1, ...)` for CUDA. `ptx.py` emits PTX directly. `isa/x86.py` and `renderer/amd/` do their own instruction selection and assembly.
- `ops_cuda.py` is a thin driver-API wrapper. `ops_nv.py` and `ops_amd.py` talk to the GPU directly. All go through `runtime/support/hcq2.py`, where a launch sequence is UOps.
- `hcq_compile` (hcq2.py:540) batches consecutive same-device calls into an `hcq_submit` CALL whose body is itself compiled (logs show `compiling hcq_submit`).
- `hcq_link` (:554) resolves allocations and addresses to constants and pre-writes the command-buffer words into pinned host memory.
- The docs claim this is "often even faster than CUDA Graph". I could not benchmark that without a GPU.

### 1.6 Comparison and rationale

| | PyTorch / inductor | JAX/XLA | Triton | tinygrad |
|---|---|---|---|---|
| Op set | ~2000 ATen ops | ~100 HLO | tile DSL | 75 UOps, composed in mixins |
| Fusion | inductor heuristics; eager has none | XLA fusion passes | manual | none unless a rule forces a boundary; recompute rule is about 30 lines |
| Kernel generator | inductor emits Triton | XLA emitters | is the compiler | own full stack to C/PTX/LLVM/asm |
| Tuning | autotune over Triton configs | GEMM/conv autotuner | `@triton.autotune` | BEAM over rewrites of the kernel AST |
| Launch | CUDA runtime, optional graphs | XLA runtime | CUDA | own queues; JIT = prebuilt command buffers |
| Debug | profiler, `TORCH_LOGS`, compile dumps | HLO dumps | `MLIR_ENABLE_DUMP` | one `DEBUG=1..7` ladder, VIZ, named passes, per-rule stats |
| Size | 100s of k lines | millions | LLVM/MLIR dependency | 25.5k lines, zero dependencies |

Why it differs (my reading):
- One IR and one rewrite engine mean there is no separate fuser, layout pass or autotuner IR. Every feature competes with the line budget, so it is re-expressed as a pattern.
- Generated kernels have statically known names, shapes, flops and bytes, so observability is cheap.
- Owning renderer, assembler and driver gives the hooks: disassembly at DEBUG=7, SQTT/PMC under VIZ=2 on AMD, and the JIT trick.
- The costs: Python-speed compile, heuristics instead of proofs, and scheduling decisions the docs call "not obvious".

### 1.7 Line-count discipline and CI

- `sz.py` counts lines and tokens per line (Python tokens only, docstrings removed, `runtime/autogen` and `viz/assets` excluded). My run printed `core lines: 9345`, `total lines: 25512`, `ops: 75`, `flags: 53`.
- `.github/workflows/test.yml:236-237` runs `MAX_LINE_COUNT=26500 python sz.py`.
- `.github/workflows/szdiff.yml` diffs base against PR on every PR and posts a sticky comment (`total lines changes: +N`). It skips stale branches.
- Other gates in `test.yml`:
  - ruff, mypy with a line-precision report, a pylint 2-space-indent check, and a `TYPED=1` run;
  - a `SPEC=2` run, which type-verifies the UOp graph between passes;
  - `CHECK_OOB=1` as a global env, where z3 proves every INDEX in bounds (`uop/validate.py`);
  - fuzzers;
  - a "Test DEBUG" step with **`DEBUG=100`** (line 60) so no debug print path can crash;
  - `DEBUG=7 python -m tinygrad.viz.cli --json | jq empty`, so the tooling output stays parseable.

---

## 2. Observability

### 2.1 Flags and what they print

Flags live in `ContextVar`s (`helpers.py:180-293`, plus `uop/ops.py:1597`) and ad-hoc `getenv` calls. Set them per scope with `with Context(DEBUG=4):`.

`docs/env_vars.md` is stale: it says 5 is UOps, 6 is linearized UOps and 7 is assembly. Real behaviour, taken from `grep "DEBUG >="` and verified by running:

| DEBUG | Added | Anchor |
|---|---|---|
| 1 | `opened device ... pid`; `scheduled N kernels in X ms \| CACHE MISS/cache hit <hash8> \| M uops in cache \| file:line` (at level 1 only if N>1); `compiling <name>` bar; `JIT captured N linears with M inputs`; `jit execs N calls` (10 or more); `memory reduced from A MB -> B MB, n -> m bufs`; `pruned from a -> b kernels` | schedule/__init__.py:138, device.py:42, jit.py:75,144,152, memory.py:58 |
| 2 | **one line per executed kernel or copy**; BEAM progress; allreduce strategy; dtype emulation notes. **Forces `wait=True`, so every kernel is synchronised and timed (this perturbs pipelining)** | realize.py:61-107, search.py:121-168 |
| 3 | `name opts: (Opt(...),...)`; `TC(axis)` candidates; `MATVEC:`; `split N:`; library loading and `LLVM init for 'znver2' with ...`; the `scheduled` line always | codegen/__init__.py:426, postrange.py:171, heuristic.py:71, prepare.py:113 |
| 4 | **generated source** (CUDA C / LLVM IR / asm); upcast heuristic reasoning; BEAM compile tracebacks | codegen/__init__.py:439,449,458, search.py:72 |
| 5 | **the kernel's optimized UOp AST** (`render_uir`, numbered SSA listing); CUDA PTX with line numbers | codegen/__init__.py:268, ops_cuda.py:116 |
| 6 | same as 5 in the core path | usb.py:24 |
| 7 | **disassembly** (`objdump -d` on CPU, ISA via `compiler.disassemble`); full LLVM IR module; `buffer: allocate/deallocate N bytes on DEV` | codegen/__init__.py:460, compiler_llvm.py:56, device.py:166,193 |
| 8 | sqlite cache queries traced | helpers.py:408 |

**The DEBUG=2 line** (`track_stats`, realize.py:61-107). Real sample from `exp/out/t3_debug2.txt`, colours stripped:
```
*** CPU        3 r_64_8_4_32_4                                  arg  3 mem   0.00 GB tm     43.37us/     1.25ms (     12 GFLOPS    1|30     GB/s)
*** NPY        1 copy   32.00 KB,     CPU <- NPY                arg  2 mem   0.00 GB tm    979.33us/     0.98ms (      0 GFLOPS    0|0      GB/s)
```
- `*** DEVICE` is followed by the running kernel counter.
- Header colour: green means this kernel's first run in the process, plain means a repeat, **magenta means executing inside a TinyJit replay**.
- `arg N` is the number of buffers. `mem X GB` is total allocated (`GlobalCounters.mem_used`), not per kernel.
- `tm A/B` is this kernel's time over cumulative kernel time.
- `(F GFLOPS  m|l GB/s)`:
  - F is static estimated ops divided by measured time;
  - m is estimated unique bytes (`Estimates.mem`) divided by time;
  - l is estimated all load/store bytes (`Estimates.lds`) divided by time.

  `l/m` is the average re-read factor.
- The time source is device timestamps from HCQ signals, otherwise `perf_counter` around `synchronize()`.
- `Estimates.from_uops` (`renderer/__init__.py:16-60`) is static. It multiplies by loop trip counts and caps `mem` at buffer size per buffer. It counts 2 ops for MULACC and `2*prod(dims)/..` for WMMA, with index arithmetic excluded.
- Hand-written kernels declare it: `KernelInfo(name=..., estimates=Estimates(ops=2*M*N*K, mem=...))`, e.g. `extra/gemm/moe_gemm.py`, name `hk_grouped_mxfp8_gemm_{E}_{M}_{N}_{K}`.

Other flags:

| Flag | Effect |
|---|---|
| `VIZ=1` | Rewrite-pass UI plus profile timeline. Implies `PROFILE=1` and `TRACK_MATCH_STATS=2`. With no tty it only saves `/tmp/rewrites.pkl.$USER` and `/tmp/profile.pkl.$USER`. `VIZ=2` adds AMD SQTT instruction trace and PMC counters. |
| `PROFILE=1` | Device and CPU `ProfileEvent`s, including `profile_marker`s, without the rewrite tracer. |
| `TRACK_MATCH_STATS=1/2/3`, `PRINT_MATCH_STATS=1` | Per-UPat match counts and time at exit. 2 saves a pickle. 3 prints every rule fire with its microseconds. |
| `DEBUG_RANGEIFY=1` | Prints the scheduler's range and realize decisions. |
| `NOOPT=1` | Skips `hand_coded_optimizations`. |
| `BEAM=n`, `JITBEAM`, `BEAM_DEBUG=1/2`, `BEAM_*`, `IGNORE_BEAM_CACHE`, `CACHELEVEL`, `CACHEDB` | BEAM (3.1). |
| `TC`, `TC_OPT`, `TC_SELECT`, `MV=0` | Tensor-core and matvec heuristics. |
| `SPLIT_REDUCEOP=0`, `MAX_KERNEL_BUFFERS`, `NO_MEMORY_PLANNER` | Scheduling and memory knobs. |
| `JIT=0`, `CAPTURING=0` | JIT control. |
| `SPEC=0..3` | UOp type verification between passes. |
| `CHECK_OOB=1` | z3 proof that every kernel INDEX is in bounds. |
| `VALIDATE_WITH_CPU=1` | Every kernel is also compiled for CPU and run on shadow buffers, outputs compared (rtol/atol 1e-3). This bisects numerics per kernel (realize.py:117-131, `_validate` :150). |
| `CAPTURE_PROCESS_REPLAY=1` | See 2.4. |
| `CCACHE=0`, `SCACHE=0/2`, `CACHELEVEL=0` | Cache switches. |
| `ASSERT_COMPILE=1` | Fails if anything would be compiled, i.e. asserts a full cache hit. |
| `PARALLEL=n` | Compile and BEAM worker processes. |
| `DEV=CPU:LLVM`, `NV:CUDA:sm_90`, `NULL:CUDA:sm_90` | Device, renderer and arch selection. |
| `DEV=NULL` + `NULL_ALLOW_COPYOUT=1` | A fake device that runs the whole compile, schedule and HCQ path with 1 us fake timestamps. CI schedules BERT, Llama-3 and gpt-oss (MXFP8, 4 experts) training on it with no GPU. |
| `BENCHMARK_LOG=name` | `extra/bench_log.py` ships step and kernel times to a metrics DB from CI. |

### 2.2 VIZ and the CLI (works headless)

```
VIZ=1 DEV=CPU:LLVM python exp/t3.py
python -m tinygrad.viz.cli --ls                # ALL USER TINY CPU NPY:COPY "CPU Memory"
python -m tinygrad.viz.cli -s CPU              # kernel timeline with markers
python -m tinygrad.viz.cli -s CPU -t           # top kernels, aggregated
```
Real output:
```
--- MARKER eager/   128.32ms
*** CPU     r_64_8_4_32_4     tm     33.00us/   640.24ms (16 GFLOPS       2 GB/s mem      40 GB/s lds   )
--- MARKER jit @ 0/   650.79ms
*** CPU     r_64_8_4_32_4     tm     26.00us/   659.02ms (20 GFLOPS       2 GB/s mem      51 GB/s lds   )
-t:
r_64_8_4_32_4                             131.00us       5  46.29%    20 GFLOPS       2 GB/s mem      50 GB/s lds
r_64_32                                    90.00us      10  31.80%     1 GFLOPS       1 GB/s mem       1 GB/s lds
```
- `-t` without `-s` also lists compiler-side TINY events. In my run the top rows were `do_to_program for r_64_32 198 ms`, `devectorize2 56 ms` and `schedule to linear 54 ms`. So you get a compile-time profile per pass.
- `--json` emits one object per event, e.g. `{"device":"CPU","name":"r_64_32","dur_ms":0.017,"fmt":{"FLOPS":..,"B/s mem":..,"B/s lds":..}}`.
- `--interval START END` restricts to `profile_marker` ranges.
- With `DEBUG` set, the CLI adds more (`viz/cli.py:215-221`):
  - `DEBUG=3` prints each kernel's base AST.
  - `DEBUG=4` prints source.
  - `DEBUG=5` lists rewrite passes with match counts.
  - `DEBUG=6` prints each pass's input graph.
  - `DEBUG=7` prints every rewrite as a diff with the rule's file:line.
- Real pass list for the softmax example: `prepare_rangeify / earliest rewrites - 2 / get_kernel_graph / run_rangeify / apply rangeify - 23 / symbolic+reduce_collapse+debuf - 2 / limit buffers / stage to store - 7 / split kernels - 4 / remove views from the kernel graph`.
- The `"CPU Memory"` source prints alloc/free events with the kernels that read or wrote each buffer.
- AMD only: `-s "gemm PMC" | rg SQC_LDS_BANK_CONFLICT`, and SQTT packets.

### 2.3 Kernel names encode the shape

`Scheduler.get_optimized_ast` (postrange.py:43-52) builds the name as follows.
- The prefix is `r` if the kernel has a REDUCE, else `E`.
- Then come the launch dims (SPECIAL), then every remaining RANGE size, each ANSI-coloured by axis type: blue global, cyan local, red serial loop/reduce, yellow upcast.

Measured examples:
- CPU `(x@w).relu().sum()` gives `r_128_64_32`.
- The same graph rendered for CUDA (NULL:CUDA:sm_90) gives `r_16_128_4_32`. The source has `__launch_bounds__(16)` and `lidx0 /* 16 */`, so the group-for-reduce choice is visible in the name.
- `BEAM=2` gives `r_64_16_32_2_4`. The last numbers are the chosen UPCAST splits, and `DEBUG=3` shows `opts: (Opt(SPLIT axis=0 (2, UPCAST)), ...)`.
- CUDA buffer arguments are named `data<slot>_<numel>` (`data0_1, data1_8192, data2_4096`), so the element count is in the argument name.
- `NO_COLOR=1` loses the axis types. Use the `opts:` line or the source in that case.

### 2.4 Process replay

Files: `test/external/process_replay/{README.md,process_replay.py,local.sh}`.
- With `CAPTURE_PROCESS_REPLAY=1`, every `do_to_program` pickles (function, args, kwargs, ContextVars, return value) into the sqlite cache.
- CI (`.github/actions/process-replay`) then checks out `origin/master`, replays the PR's inputs through the old code, and diffs the rendered source with `difflib`.
- A PR title containing `[pr]` means "refactor, kernels must not change", and any diff fails CI.
- It early-stops when more than 20% of kernels changed (PR #5480). `[skip_process_replay]` skips it.
- Net effect: refactors are mechanically proven kernel-identical, and perf PRs show exactly which kernels changed.

### 2.5 Using the tools

| Question | Tool |
|---|---|
| Computed twice | Read kernel bodies (`DEBUG=4`, or `DEBUG=3 viz.cli`) and look for the same `exp2` or `reduce` in two kernels. `DEBUG_RANGEIFY=1` shows what was realized vs recomputed. Compare static `Estimates.ops` summed over kernels with the ideal. |
| Unfused | Count `***` lines per step, the `scheduled N kernels` line, and `check_schedule(t, N)` tests. `.contiguous()` forces a boundary. |
| Bad memory access | Compare `m\|l GB/s` with the device peak. Read the index expressions in `DEBUG=4`. Check for `make_float4` from `memory_coalescing`. On AMD, bank-conflict counters under `VIZ=2`. |
| Wrong shape/stride | Kernel names, the `data<slot>_<numel>` argument names, the `DEBUG=5` AST (`param 8192` plus index math), `SPEC=2`, `CHECK_OOB=1`, `VALIDATE_WITH_CPU=1`. |
| Launch overhead | Cumulative `tm` vs wall time, magenta replay lines, `hcq_submit` batches, gaps in the VIZ timeline. |

### 2.6 Real examples (all from my runs)

**(a) One kernel for `(x@w).relu().sum()`.** DEBUG=5 prints the AST:
```
%0 = param 1 (out)  %2 = param 8192 (x)  %8 = param 4096 (w)
%3 = range 64 : WEAK 1   %5 = range 128 : WEAK 0   %10 = range 32 : WEAK 2
%13 = mul x[%3*128+%5], w[%5*32+%10]
%14 = reduce %13, %5 : op=add          # matmul k-reduce
%15 = max 0.0, %14                     # relu
%16 = reduce %15, %3, %10 : op=add     # outer sum
%17 = store %0, %16
```
CUDA rendering (via `exp/nullcuda.py`, `DEV=NULL:CUDA:sm_90`):
```
extern "C" __global__ void __launch_bounds__(16) r_16_128_4_32(float* data0_1, float* data1_8192, float* data2_4096) {
  __shared__ __align__(16) float buf3[16]; int lidx0 = threadIdx.x; /* 16 */
  for (Lidx1<4) for (Lidx2<32) { acc=0; for (Lidx0<128) acc += data1_8192[(lidx0<<9)+(Lidx1<<7)+Lidx0] * data2_4096[(Lidx0<<5)+Lidx2];
     buf1 += (0.0f<acc)?acc:0.0f; }
  buf3[lidx0] = buf1; __syncthreads(); ... if (lidx0==0) data0_1[0] = sum(buf3);
```
The NULL device's `DEBUG=2` printed `531 GFLOPS 49|2098 GB/s`. The 1 us time is fake, but the 43x ratio of lds to mem is real static information about re-reads.

**(b) A softmax-like block: 4 kernels, `exp` recomputed in two of them** (`exp/t4.py`).
```
*** CPU 3 r_64_8_4_32_4 arg 3   h = x@w  (stored: 3 consumers)
*** CPU 4 r_64_32       arg 2   m = h.max(-1)
*** CPU 5 r_64_32       arg 3   s = sum(exp2((h-m)*log2e))     <- exp NOT stored
*** CPU 6 r_32_16_4     arg 4   out = sum(exp2(..)/s * h)      <- exp recomputed
```
`DEBUG_RANGEIFY=1` explains it. `*** 2 Ops.EXP2 (64, 32)` was first marked realize (2 consumers). Then `remove_bufferize` deleted that STAGE, because the producer reads at most 3 buffers and no reduce in it reads a buffer. The matmul REDUCE (3 consumers, reduce reads buffers) stays materialized.

A static prototype of the detector (`exp/t5.py`, using `compile_linear` + `estimate_uop`) prints:
```
r_64_8_4_32_4  flops=524288 lds=1318912 mem=57344 lds/mem=23.0
r_64_32        flops=  3968 lds=   8448 mem= 8448 lds/mem= 1.0
r_64_32        flops=  8192 lds=   8704 mem= 8704 lds/mem= 1.0  {'EXP2': 32}
r_32_16_4      flops= 14336 lds=  24704 mem= 8832 lds/mem= 2.8  {'EXP2': 4}
kernels containing EXP2: 2
```

**(c) BEAM** (`BEAM=2 DEBUG=2`, CPU; progress is `\r`-overwritten on a tty):
```
   0.00s:                from   1 ->   1 actions  128   64   32
   2.98s:        41.07us from  11 ->   9 actions   64   16   32    2    4
   4.65s:        41.07us from  12 ->  10 actions   64   16   32    2    4     <- converged
BEAM_SEARCH: final tm=41.07us, applied_opts=[Opt(SPLIT axis=0 (2, UPCAST)), Opt(SPLIT axis=1 (4, UPCAST))]    (BEAM_DEBUG=1)
```
`BEAM_DEBUG=2` prints every candidate: index, uop count, compile time, run time. A second run hits the disk cache.

**(d) Rule profile** (`TRACK_MATCH_STATS=1 PRINT_MATCH_STATS=1`): `3993 / 25629 -- 152.81 / 431.66 ms -- TOTAL; 325 rules, 108 matched once`, with the slowest rules listed by file:line.

---

## 3. Speed tooling

### 3.1 BEAM (`codegen/opt/search.py`)

**What it searches.** Sequences of `Opt` actions on one kernel AST (search.py:14-22). The `actions` list has:
- SPLIT to UPCAST with amt in {0,2,3,4,5,7};
- SPLIT to LOCAL with amt in {0,2,3,4,8,13,16,29};
- SPLIT to LOCAL top=True with amt in {13,16,28,29,32,49,64,256};
- optional PADTO 32 (`BEAM_PADTO`);
- TC variants;
- SWAP pairs.

**Algorithm** (`beam_search` :105).
1. Expand every beam member with all legal actions. Illegal ones raise `KernelOptError`. The upcast product is capped at `BEAM_UPCAST_MAX=256` and the local product at `BEAM_LOCAL_MAX=1024`.
2. Compile all candidates in parallel worker processes. Each has a 10 s SIGALRM timeout (`BEAM_TIMEOUT_SEC`), and candidates above `BEAM_UOPS_MAX=3000` uops are rejected.
3. Dedup by binary (`seen_libs`).
4. Drop candidates whose static op count is more than 1000x the minimum.
5. Time the rest on the real device and keep the best `BEAM` (the width) as the new beam.
6. Stop when no candidate improves by `BEAM_MIN_PROGRESS` (0.01 us).

**Caching.** The winner's `applied_opts` goes to a diskcache table `beam_search`, keyed on `{ast.key, amt, allow_test_size, device, renderer suffix}`.

**Timing** (`_time_program` :34, `time_call` realize.py:290):
- the kernel is compiled once, run 3 times, and the **min** is taken;
- `early_stop` is 3x the current best;
- there is a device-side timeout (`BEAM_DEV_TIMEOUT`);
- times are device timestamps, not wall time;
- **L2 is cleared** via `dev.invalidate_caches()` on NV, otherwise by running a 1024x1024 `ones().contiguous()` kernel;
- `allow_test_size` shrinks the grid to at most 65536 workgroups and multiplies the time back, assuming linear scaling;
- buffers are uninitialised, sized from the AST params, and symbolic variables are set to their range midpoint.

**Cost.** Compile time multiplies. `JITBEAM` applies BEAM only to JIT capture. CI runs Llama-3 8B on 4 GPUs with `JITBEAM=2 IGNORE_BEAM_CACHE=1`.

### 3.2 There is no cost model for search

Search is pure measurement. What exists is the static `Estimates(ops, lds, mem)` (renderer/__init__.py:16), used for display, for BEAM's 1000x-compute pruning, and in tests.
- `test/null/test_uops_stats.py` asserts byte counts exactly:
  - `a+b` on uint8 1024x1024 gives `1024*1024*3`;
  - `a+a` counts the read once;
  - matmul gives `ops == 2*N^3`.
- `test/external/speed_v_theoretical.py` turns `2MNK` ops and A+B+C bytes plus timed kernels into TFLOPS and GB/s, and asserts thresholds after BEAM. On NV, gemm 4096 must exceed 109 TFLOPS ("theoretical 165") and gemv 16384x4096 must exceed 840 GB/s ("theoretical 1008").
- The only heuristic models are `hand_coded_optimizations` and the `remove_bufferize` rule.

### 3.3 TinyJit (`engine/jit.py`)

- Call 0 runs normally as a warm-up. Call 1 captures: `capturing.append(self)` makes the scheduler hand each LINEAR to `add_linear` instead of executing. Call 2 and later replay.
- At the end of capture, the LINEARs are concatenated and optionally pruned. `jit_lower` (:25) then does the following:
  - replaces input buffers by `PARAM(slot)`;
  - unbinds variables;
  - runs `memory_plan_rewrite`, a lifetime-based TLSF arena (`DEBUG=1` prints `memory reduced from A MB -> B MB`);
  - runs `compile_linear` (BEAM via `JITBEAM`), which batches into hcq submits.
- Replay (`CapturedJit.__call__` :60) checks arg names, shapes and dtypes, raising `JitError("args mismatch in JIT")` otherwise. It copies aliased inputs.
- Pitfalls (also in https://mesozoic-egg.github.io/tinygrad-notes/20240102_jit.html):
  - Python side effects vanish after capture.
  - No nested JIT.
  - **Returned tensors alias the arena, so the next call overwrites them.**
  - "didn't JIT anything!" if no kernel ran.
- Measured here on CPU (Python overhead only): scheduling 4 kernels took 42 ms (CACHE MISS), then 2.3 ms (cache hit), 1.3 ms (hit), then `JIT captured 1 linears with 2 inputs`.

### 3.4 Benchmarks and regression catching

| Mechanism | Detail |
|---|---|
| Kernel-count and memory ratchets | `check_schedule(t, N)` asserts exactly N kernels. `helper_test` in `test/null/test_real_world.py` (GPT-2, llama, UNet block, BERT...) asserts a max memory and a max kernel count. **It also fails if the actual is more than 20% below the limit ("too far from")**, which forces tightening when a PR improves things. (`test/helpers.py:48-100`, `test_real_world.py:14-40`) |
| Exact torch-backend counts | PR #15091 (commit `71f228f80`) changed `assertLessEqual` plus a warning ("Expectation can be lowered") to `assertEqual`. Counts dropped: conv+relu 8 to 6, bn 16 to 10, resnet block 17 to 12. |
| Static flops/bytes tests | `test/null/test_uops_stats.py`. |
| `DEV=NULL` full-model runs | BERT, Llama-3 and gpt-oss training scheduled and compiled in CI with no GPU (`test.yml:186-205,388`). |
| Process replay | See 2.4. |
| Speed vs theoretical and vs torch | `test/external/speed_v_theoretical.py`, `test/speed/external_test_speed_v_torch.py` (`TORCHCUDA=1`), run on real NV and AMD runners. |
| MLPerf and model benchmarks | `benchmark.yml` (603 lines) on self-hosted GPU runners: ResNet50 10 steps, BERT, Stable Diffusion, SDXL, Llama-3 8B 4-GPU with BEAM, hlb-cifar10 (target 93%), beautiful_mnist (96%). Times are logged through `BENCHMARK_LOG`. |
| Staging branch | Pushes to `update_benchmark_staging` run all benchmarks and record. |
| Fuzzing | Symbolic, fast idiv and shape-ops fuzzers; `SPEC=2` run. |
| Compiler microbenchmarks | `test/null/test_microbenchmarks.py`. |

---

## 4. Performance bugs and fixes (verified in git; `git show <hash>` in the clone)

1. **Softmax backward did redundant work through `max`.** PR #7009 (chenyu, 2024-10-12), https://github.com/tinygrad/tinygrad/pull/7009. The max's gradient cancels exactly but was still computed. Fix: `.detach()` on the max. The PR says "two less kernels per softmax backward ... BERT 3% faster, 209 -> 213 tflops".
2. **Same bug in `logsumexp` and `logcumsumexp`.** PR #17088 (merged 2026-07-20, commit `ef77963cf`), https://github.com/tinygrad/tinygrad/pull/17088. Fwd+bwd kernels went 5 to 3. New tests `test_logsumexp_backward` and `test_logcumsumexp_backward` assert `check_schedule(x.grad, 3)` and fail on master. The fix is one line, `m = self.max(...).detach()` (`mixin/op.py:661`).
3. **The recompute-vs-store cost function itself.** PR #12360 (geohot, 2025-09-30, `360980f1a`), https://github.com/tinygrad/tinygrad/pull/12360. It replaced "no REDUCE/COPY/ASSIGN in the producer" with "keep the buffer if the producer reads more than 2 distinct buffers, or a reduce inside it reads a buffer". The threshold is now 3 (`rangeify.py:76`).
4. **Too much recomputation in BERT, fixed by adding `contiguous()`.** Commit `041e9a41c` (chenyu, 2025-12-15, PR #13713): `return gelu(x).contiguous()` in `BertIntermediate`. The message is "faster step with a lot less recomputation". This is the opposite of fusion: force a materialization because the default policy recomputed an expensive chain in several consumers.
5. **Fusion lost to a materializing constant.** Commit `8ac62b28e` (PR #16439, chenyu, 2026-05-29, "fix AffineGrid fusion"). In ONNX `AffineGrid`, `Tensor.ones_like(grids[0], device=...)` became `grids[0].const_like(1)`. A CONST folds into the index math. (Inferred from title and diff; no kernel delta is stated.)
6. **Extra copy/realize kernels in front of custom kernels.**
   - Commit `37236a0a4` (PR #18391, 2026-09-24) tightened budgets: reshape 2 to 1, permute 3 to 2, double-permute 2 to 1, 2D-shrink 3 to 2, expand 3 to 2.
   - Commit `470c032a5` (PR #17423, 2026-08-05) fixed the bug class: computed inputs above a custom kernel get no ranges and must be realized, but view-only movement ops over a buffer must not add a kernel.
7. **Extra copy in multi-device allreduce with cast.** Commit `e3b3eea1e` (PR #16740, 2026-07-28). Test `test_allreduce_cast_half` pins the kernel count (8; 10 with assign). Exact bug is inferred from the title.
8. **A hand-written FlashAttention kernel gave a wrong LSE for sliding-window attention, so the host recomputed it with full tensor ops.** Commit `d80254c1d` (PR #17635, 2026-08-20, "fa: remove swa recompute"). It deleted `_windowed_lse` and `_windowed_delta` (47 lines, a second attention pass in backward) and fixed the kernel to apply the scale after the MMA under `#if WINDOW`. This is the "fused kernel plus a host fix-up that redoes the math" pattern.
9. **Slice-assign (the KV-cache write pattern) got no ranges.** Commit `3b75d8a7a` (2026-03-20, "fix double after bug in rangeify"). A new rule wraps the STORE in an inner AFTER so it gets ranges from the view shape. Test: `test_single_after_store`.
10. **Process-replay cost.** PR #5480 (2024-07-14) caps replay diffing at about 20% of kernels. It was motivated by #5478, which changed every kernel.
11. **Documented known-unfused list.** `# TODO: should be 1?` comments in `test/null/test_schedule.py`: `test_scaled_dot_product_attention_multireduce_fusion` accepts 4 kernels (ideal 1), and `test_ugly_reduceop_pairing` accepts 2 (ideal 1).
12. **Direction of travel.** Dozens of `gptoss: fused swiglu / router topk / qkv / combine / quantize` and `llama: fused ce / norm-mul-quantize-mxfp8` commits (e.g. `c04482910`, `e061b25f8`, `dfea9e799`, 2026-04 to 10). When the generic scheduler cannot fuse, they write custom UOp kernels (`extra/gptoss_kernels/*`, `extra/llama_kernels/*`) with `Estimates` and kernel-count tests. This is okc's situation.

Search recipe: `git log -i --grep='twice\|redundant\|recomput\|kernel count\|extra copy\|fused'`, and `git log -S'check_schedule(' -p test/null/test_schedule.py` (every change of a count is a fix or regression).

---

## 5. Transfer to okc: an "okcdbg" DEBUG tool

### 5.0 Constraints that shape it
- A remote judge shows a ~16 KB log tail. Judge runs are scarce (`okc_run.py` needs a hypothesis and a prediction). Batch measurement in untimed `prepare()` works.
- There is no local GPU, so offline work is static analysis and CPU emulation.
- H200 public figures: HBM3e 4.8 TB/s, ~989 TFLOPS BF16 dense, ~1979 TFLOPS FP8 dense. NVFP4 has no native Hopper tensor-core path, so W4A16 dequantizes in-kernel and MoE decode is bandwidth and launch bound.
- Calibrate achievable bandwidth in the first run (a copy kernel) rather than trusting the datasheet.
- tinygrad lesson: bytes and flops are free for a compiler. For hand kernels we must declare them once, next to the kernel.

### 5.1 Design

`okcdbg.py`, pure Python with optional torch/CUPTI, controlled by `OKC_DEBUG=0..7`, `OKC_DEBUG_BUDGET=14000` bytes and `OKC_DEBUG_ONLY=regex`. Levels are cumulative.

Rules for the 16 KB tail:
1. Print verbose material first (it gets cut) and a bounded summary block last.
2. Never truncate mid-line. Rank rows and drop whole ones with `...+N rows omitted`.
3. Every line has a tag (`#K`, `#S`, `#W`, `#B`, `#D`) so a script can grep it.
4. Fixed-width numbers, units only in the header.
5. Collapse CUDA-graph replays to one line per unique kernel with `n=`.
6. Optionally embed the previous run's per-kernel `us` (about 1 KB JSON) in the submission, so the tool prints `d_us` and `d%`. This matches our same-method-delta prediction practice.

| L | Name | Content |
|---|---|---|
| 1 | run | One line per problem/variant: switches, #graph nodes, #kernels, GPU-busy us (sum of kernel durations), wall us (graph replay), `busy%`, gaps, floor us (bytes model), `x_floor` (wall/floor), dup-work count, warnings (non-contiguous input, misaligned pointer). |
| 2 | kern | The tinygrad DEBUG=2 line per unique kernel: idx, shape-encoded name, grid, block, smem, regs, n, us, %tot, GB/s, %achievable-BW, TFLOPS, %peak, floor us, x_floor. Sorted by time, with an OTHER row. |
| 3 | bytes | Per-tensor declared read/write bytes, unique vs traffic (reuse factor), argument table (shape, stride, dtype, contiguous, ptr%16), the `prepare()` search table. |
| 4 | graph | Dataflow and fusion map with buffer ids and per-edge bytes, copies/memsets, inter-kernel gap timeline. |
| 5 | dups | Duplicate-work detectors D1-D4 (5.3). |
| 6 | trace | Raw CUPTI/profiler records as compact CSV with a name table, sampled to budget. |
| 7 | code | Per-kernel regs/spills/smem, PTX/SASS instruction histogram (ld.global, mma, cvt counts, `ex2` counts), source/IR hash. |

Add a smoke test that runs all levels on a mock trace on CPU (tinygrad's `DEBUG=100` trick) and checks the summary block parses.

Illustrative output (numbers invented, to show the format only):
```
#OKC v1 problem=moe_nvfp4_w4a16 variant=smallM T=32 E=128 topk=8 dev=H200 bw_ach=4210GB/s peak=4800
#K  i name                                  grid       blk  smem reg  n    us  %tot  GB/s %ach   TF  floor x_fl
#K  1 moe_gemm1_w4a16_T32_N4096_K7168_e8    (512,8,1)  256 49152 168  1  88.4  59.6  3310   79   71   61.0 1.45
#K  3 moe_gemm2_w4a16_T32_N7168_K2048_e8    (448,8,1)  256 49152 160  1  44.0  29.7  2900   69   34   30.5 1.44
#S  kernels=5 gpu_us=148.2 wall_us=156.0 busy=95.0% gaps_us=7.8 floor_us=100.1 x_floor=1.56
#W  waste_us: gemm1 +27.4  gemm2 +13.5  gaps +7.8 ... recoverable=48.1us -> bound 1.45x
```
`%ach` is relative to bandwidth measured in the same run. `floor` is declared minimal bytes divided by achievable bandwidth (or flops over peak, whichever is larger). `x_fl` is measured over floor, the same idea as `speed_v_theoretical`, per kernel. The sum is the waste ledger and should feed the existing `tools/waste.py`.

Declaring the bytes model (replaces `Estimates.from_uops`):
```python
@okcdbg.kernel(name="moe_gemm1_w4a16_T{T}_N{N}_K{K}_e{topk}",
    bytes=lambda a: dict(w=a.n_active_experts*a.N*a.K//2 + scales_bytes(a), x=a.T*a.K*2, out=a.T*topk*a.N*2),
    traffic=lambda a: a.grid_ctas*a.tile_bytes,           # all loads incl. re-reads (like Estimates.lds)
    flops=lambda a: 2*a.T*topk*a.N*a.K,
    reads_expected=dict(w=1.0, x=1.0))
```
- The decorator evaluates the lambdas on real shapes at launch or capture time.
- Take `n_active_experts` from the real routing tensor, not a formula. The project lesson is that synthetic routing misled MoE twice.
- Names: put T/N/K in the Triton kernel `__name__` and in the `load_inline` function name, so CUPTI rows are self-describing.

### 5.2 Getting the data
1. **Launch shim.** All launches go through `okcdbg.launch(fn, grid, block, args, meta)`. It records, per tensor arg, pointer, bytes, shape, stride, dtype, contiguity and ptr%16, plus scalars and the declared bytes/flops. It does this at warm-up and at graph capture, as tinygrad's captured LINEAR carries `Estimates` while only timings are re-measured.
2. **Timing.** CUDA events around the whole graph give wall time. `torch.profiler` (CUPTI kernel activity: name, start, end, grid, block, regs, shared) gives per-kernel device time inside replays. Join by (name, occurrence). Do this once in `prepare()`, outside the timed region.
3. **Calibration** in `prepare()`: a copy kernel, a BF16 GEMM and an FP8 GEMM give `bw_ach` and `tf_ach`. Print them at L1.
4. **L2 flush** between timed trials (tinygrad `clear_l2`). It matters for any decode kernel whose working set is below the ~50 MB L2. Do it in `prepare()` trials only, never in the judge-timed path.
5. **Offline dry-run** (tinygrad's `DEV=NULL`): `OKC_DEBUG_DRYRUN=1` makes `launch()` record without executing. The whole decoder layer's launch list, shapes, bytes and duplicates then print on the CPU box in seconds and can be diffed in CI.

### 5.3 Detecting "computed twice" and redundant reads

**D1, exact duplicate launches.** Define `sig = (kernel_id, grid, block, scalars, tuple((ptr, nbytes, version) for input args))`. The same `sig` twice in one step is a recompute. A per-buffer version counter, incremented whenever a kernel writes it, avoids false positives on in-place updates such as KV-cache appends.

**D2, value numbering over the launch DAG.** This catches the same value computed into different buffers, such as two identical RMSNorms or dequants.
- Each leaf (input, weight, constant) gets an id.
- Each kernel output gets `vn = hash(kernel_id, scalars, vn of inputs)`. Equal `vn` on two outputs means duplicate compute. tinygrad gets this free from interned UOps.
- It needs input/output arg roles, declared in the decorator or derived from Triton IR (`tl.load` vs `tl.store`) or `const T* __restrict__`.
- Report format: `#D2 vn=ab12 K3(out=b9) == K11(out=b21) wasted_us=7.2 bytes=14.7MB`.

**D3, bytes model vs expectation.** For each tensor class, accumulate `Σ read_bytes / unique_bytes` ("passes") and compare with `reads_expected`. Examples:
- MoE: the same expert-weight pointer appearing in two kernels, or token tiles of one expert split across CTAs so traffic/unique is above 1.
- FP8 paged-attention decode: the KV page read once per query head instead of once per KV head, so traffic/unique = heads per KV group. This needs only the grid decomposition.
- BF16 projections: split-K or N-tile CTAs re-reading the activation. Flag it when traffic/unique > 2 and the tensor is over 1 MB.

The static ratio is tinygrad's `lds/mem`. Cross-check it dynamically with **time-implied passes** `passes_t = us * bw_ach / unique_bytes`:
- `~1` means the kernel is at the floor.
- `~traffic/unique` means re-reads miss L2, which is a real cost.
- `<1` means the declared model is too big (cache hits or a wrong declaration).

This validates the bytes model itself.

**D4, trace heuristics.**
- Unexpected copies: kernel names matching `copy_|direct_copy|elementwise_kernel|CatArrayBatchedCopy|contiguous|cast`, plus Memcpy/Memset nodes over 64 KB.
- Dead writes: a buffer written but never read and not a declared result.
- Stride and alignment: an input that is non-contiguous or has `ptr%16 != 0` where the kernel declares vector loads.
- Gaps: `gap_i = start_{i+1} - end_i` per replay, reported as sum, max and the count above 3 us (calibrate the threshold from the median). Also `busy% = Σ dur / wall`. This is launch overhead for CUDA graphs.
- In-kernel recompute (tinygrad's `exp` twice) is invisible to a trace. For our own sources, use a static op histogram from Triton IR or PTX (`ex2`, `rcp`, `cvt` counts) against an expected count.

Each finding carries `wasted_us_est`: kernel us x (1 - 1/passes) for D3, the later kernel's us for D1/D2, the gap sum for D4. The summary ranks by it and feeds the waste ledger.

### 5.4 BEAM-style search in `prepare()`

What transfers is the harness, not tinygrad's search space (we have no AST to rewrite):
- A small discrete action space per kernel (BLOCK_M/N/K, `num_warps`, `num_stages`, split-K, swizzle, small-M path), filtered for shared memory (≤ 227 KB on H200), registers and divisibility. This mirrors `KernelOptError` and the BEAM caps.
- Compile all candidates up front, with a timeout and a size cap, and dedup identical binaries.
- Time each as it will really run: inside a CUDA graph, L2-flushed, min of 3, early-stop at 3x best, device timestamps.
- Choose per shape key (T bucket, active-expert bucket). With about 50 candidates, a sweep or coordinate descent is enough; full beam only matters when compositions explode.
- Persistence is the problem. The judge has no sqlite cache. Either rerun the search in each `prepare()` if budget allows, or run it as a judge probe, print the winner table, and bake it into code with an `OKC_VERIFY_TUNE=1` re-time path.
- Format copied from tinygrad: `#B moe_gemm1 T=32: cand=14 ok=11 best=88.4us cfg=(BM16,BN128,BK128,w4,s4) 2nd=91.0 worst=210.2 t_search=6.1s`.
- Never search with synthetic routing or inputs. Use `reference.py generate_input`.

### 5.5 What maps and what does not

**Adopt:**
1. A single cumulative level ladder, with a smoke run at max level (`DEBUG=100`).
2. Declared Estimates giving a `GFLOPS | stored-GB/s | all-traffic-GB/s` line. The pair is the cheapest redundant-read detector.
3. Kernel names that encode shape and chosen config, so a 16 KB tail stays readable.
4. A first-run/replay tag, and cumulative kernel time vs wall.
5. Ratchet tests: exact kernel count and memory per decoder layer, and fail when actual is more than 20% below the limit so limits get tightened. For us, a dry-run test with a golden launch list of `(name, grid, declared bytes)`.
6. PTX/SASS (or Triton IR) hash per kernel, as process replay. A refactor must reproduce the hashes, and otherwise the diff explains a timing change. Combined with our step switches, this makes attribution mechanical.
7. Per-stage numeric bisect, like `VALIDATE_WITH_CPU`. A debug run compares each stage with the torch reference on a small shape and prints the first failing kernel. For the 20-kernel layer this localises a regression in one judge round.
8. A `CHECK_OOB` mindset: `tl.device_assert` on index ranges in debug builds, and `TRITON_INTERPRET=1` CPU tests for index math.
9. The recompute rule of thumb from `remove_bufferize`: recompute a producer in each consumer if it is elementwise, touches at most 3 buffers, and has no buffer-reading reduce. Materialize otherwise.
   - RMSNorm output feeding Q/K/V has a reduce, so fuse one prologue or materialize.
   - RoPE + KV write is elementwise, so fuse it into the epilogue.
   - Weight dequant is elementwise on streamed tiles, so always recompute it in the GEMM.
   - Router logits to topk to weights is a reduction/sort, so keep it a separate kernel. The S2 fused-router experiment regressed -10% at small T, so measure rather than assume.
10. JIT discipline: warm-up, capture, replay; static input buffers; replay outputs alias arena memory. Log graph node count and arena size.
11. Timing hygiene: device timestamps, min-of-N, L2 flush, early stop. The shrunk-grid trick does not apply to tiny decode kernels.
12. `profile_marker` phases such as route, gemm1, combine, and `OKC_DEBUG_ONLY=regex` as the `--interval` analogue.
13. `Estimates` declared in `KernelInfo` for hand-written kernels (`moe_gemm.py`). That is exactly our pattern.

**Partial:**
- BEAM: the harness yes, the search space no, and persistence differs because the judge is stateless.
- Rewrite-rule fusion: the decision criteria (item 9) yes, the machinery no. We hand-write fused kernels, and a kernel generator would cost more than the remaining headroom is worth before the Nov 4 kernel-track deadline.
- Own driver and prebuilt command buffers: skip. PDL measured about 0 for us, and we cannot change the judge's driver stack.

**Do not adopt:**
- The line-count CI (at most a size cap on `okcdbg` itself, under about 400 lines, so it can ship inside a submission).
- The symbolic simplifier, UOp interning and rangeify.
- AMD SQTT-style tracing.
- Static `Estimates.from_uops` inference, which needs a compiler IR. As a stretch goal, TTIR could be used to cross-check declared bytes.

### 5.6 Implementation order
1. About 1 h: launch shim, `@kernel` registry, calibration, and the L1/L2 summary with budgeted printing and baseline diff, run inside `prepare()`.
2. About 1 h: torch.profiler/CUPTI join for per-kernel us under graph replay, plus gap and busy stats.
3. About 1 h: D1, D3 and D4. Wire `wasted_us_est` into `tools/waste.py`.
4. About 2 h: dry-run mode, the golden launch-list ratchet test, and an `OKC_DEBUG=100` smoke test.
5. About 2 h: the `prepare()` tuning harness for the top two kernels by waste.
6. Later: D2 value numbering, per-stage numeric bisect, PTX/SASS hash diff.

---

## 6. Reproduce and sources

Commands (venv: `uv venv .venv && uv pip install -e repo numpy tabulate`; plain `CPU` needs clang, which is not installed, so use `CPU:LLVM` or `CPU:X86`):
```
DEV=CPU:LLVM DEBUG=2..7 python exp/t2.py                          # exp/out/t2_llvm_debug*.txt
DEV=CPU:LLVM DEBUG_RANGEIFY=1 DEBUG=2 python exp/t4.py            # exp/out/t4_rangeify.txt
VIZ=1 DEV=CPU:LLVM python exp/t3.py; python -m tinygrad.viz.cli -s CPU [-t] [--json]; DEBUG=3 python -m tinygrad.viz.cli -s CPU
BEAM=2 BEAM_DEBUG=1 DEBUG=2 CACHEDB=/tmp/x.db DEV=CPU:LLVM python exp/t2.py
TRACK_MATCH_STATS=1 PRINT_MATCH_STATS=1 DEV=CPU:LLVM python exp/t2.py
NULL_ALLOW_COPYOUT=1 DEBUG=4 python exp/nullcuda.py exp/t2.py     # CUDA C for sm_90 without a GPU
DEV=CPU:LLVM python exp/t5.py                                     # static per-kernel Estimates + op histogram
python sz.py                                                      # line counts
```
Sources:
- Code: https://github.com/tinygrad/tinygrad (files cited inline), `docs/developer/{developer,speed,layout,uop}.md`, `tinygrad/viz/README.md`, `test/external/process_replay/README.md`, `AGENTS.md`.
- Docs: https://docs.tinygrad.org/developer/speed , https://docs.tinygrad.org/developer/developer , https://docs.tinygrad.org/env_vars
- mesozoic-egg notes (historical, LazyBuffer era): https://mesozoic-egg.github.io/tinygrad-notes/ (fusion: `20250117_fusion.html`; beam: `20241203_beam.html`; viz: `20241129_viz.html`; jit: `20240102_jit.html`; `scheduleitem.html`). Also https://anuraagw.me/blog/tinygrad-notes
- PRs: #7009, #17088, #12360, #5480 at https://github.com/tinygrad/tinygrad/pull/<n>. The rest are cited by commit hash.

Uncertainties:
- I measured no GPU numbers, so the JIT-vs-CUDA-graph claim and BEAM speedups on NVIDIA are the project's own claims.
- `NULL:CUDA` timings are fake (1 us); only the static estimates and rendered source are meaningful there.
- All numbers in section 5 are illustrative formatting, not measurements.
- The H200 peak figures are public specs as I recall them. Calibrate in-run.
- The effects of #16439 and #16740 are inferred from titles and diffs, because those commit messages are empty.
- The CUPTI graph-node fields I assume for the join are not verified. Check the CUPTI version on the judge.