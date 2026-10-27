# How George Hotz actually works when he optimises live — evidence from his streams

Prepared 2026-10-25 for the H200 kernel challenge (MoE W4A16 GEMMs, attention decode, dense projections, one decoder layer).
Files: `index.md` (63 streams), `raw/` (32 timestamped transcripts), `notes.md` (my working notes with timestamps), `list.tsv` (the 502-video channel list), `sub_captions_json3.tar.gz` (original caption JSON), tools in `../tools/`.

## 0. Provenance, honesty and how to read this

**What I had.** YouTube auto-captions (`yt-dlp --skip-download --write-auto-subs`, text only) from the channel `@geohotarchive` (the old handle `@georgehotzarchive` returns 404). 32 transcripts were fetched (about 3.5 MB of caption text; the raw caption JSON is kept compressed as `sub_captions_json3.tar.gz`, 6.7 MB). I did not get a transcript for one stream I wanted (`chnv4GhR174`, "auto optimizing a resnet 50"): YouTube answered HTTP 429 on the subtitle endpoint three times over several minutes. I backed off and did not try to get around it. No other source was needed for the main claims.

**What I read.** I did not read any stream end to end. The streams are 3–8 hours each and 60–80 % of the words are chat, politics, tea, and cryptocurrency jokes. I used (1) a density scan (counting performance words per 10–15 minutes) to find the technical stretches, (2) keyword scans (`slow`, `twice`, `wasting`, `flops`, `bandwidth`, `theoretical`, ...), and (3) continuous reading of the stretches that mattered. The MI350X GEMM stream (2025-11-01) I read from start to about 03:40:00 continuously and the rest by windows. Tiers in `index.md`: A = main evidence source (the technical stretches read continuously, the chatter skipped), B = secondary source (targeted windows only), C = keyword-scanned or skimmed, I = listed only (not fetched). Anything I did not read I do not claim.

**Quotes vs. summary.** A line in a block quote with a link is a **direct quote from the auto-caption**, lightly edited (see the next sentences). Caption oddities that I left alone: "Giga flops" often means TFLOPS, "tar flops", "hlb C far" for hlb_cifar. The link is `https://www.youtube.com/watch?v=<id>&t=<seconds>s`, so it opens at that point. Everything after the words "Summary" or outside quote blocks is my paraphrase. Quotes are lightly edited: I removed filler ("uh", "um", false starts), added punctuation, fixed the caption's spelling of names and terms ("tiny grad"/"tiny red" -> tinygrad, "lvm" -> LLVM, "Colonel" -> kernel, "Cindy group" -> simd group), marked my insertions with [brackets] and my cuts with "..."; I did not reorder words. Some quotes are still shaky because the captions themselves are (e.g. "tar flops"). I checked every quote against the transcript text near its timestamp with a script (`../tools/verify.py` in the work dir): of 310 quote fragments, 280 match at >= 80 % and the remaining ones differ only by caption spelling or by dropped/added filler words, except where I flagged otherwise. Numbers inside quotes are what the caption says; where the caption is ambiguous I say so.

**Dates.** YouTube upload date, which can be a day or more after the live stream.

**Caveats about the man himself.** He is loud, opinionated and often wrong in the moment (he says "Apple cheats", later finds it is a 4x4 grid of simdgroup matrices; he suspects the compiler, then his odds of a compiler bug go down;). I keep the evidence of what he *did* and *measured*; I drop the commentary on politics and companies. Two things matter for you: (1) almost all of his tooling is tinygrad's (VIZ, DEBUG levels, BEAM, UOps); we do not have it for CUDA/Hopper, so Part F translates each practice to what we can do with Nsight Compute, `cuobjdump`, `ncu`, `nsys`, CUPTI, and plain Python timers; (2) he has never (in the streams I read) worked on Hopper-specific features (wgmma, TMA, clusters). The Hopper content below is thin and I mark it.

Stream aliases used in links: MI350 = 4zwFDrsLm64 (2025-11-01, hand-written GEMM in UOps), BRRR = Pbbi3k5XTcs (2025-03-03, "GPU go brrrr"), MOE = upZf-BWF1i0 (2025-03-09), GPUS = OUzm06YaUsI (2025-03-30), R9070 = KXjEQHmWPsQ (2025-03-16), CPU1 = pKE3LV2Rdyw (2025-02-09), CPU2 = c1UHi7dncvo (2025-02-10), MCTS = R8uxmXmtOrk (2024-07-22), FSDP = h8PzTlTcfDg (2025-05-18), TORCH = zDwRrhQ9PDc (2025-02-23), TOUR = nyDGXlLp578 (2025-05-05), CLOUD = Gm0t-kAsHuY (2024-11-12), RESNET = AV5JXG8y4fA (2023-09-30), MIXTRAL = H40QRJFzThQ (2023-12-10), M1 = MksxUpucnKA (2023-02-18), MATMUL = VgSQ1GOC86s (2022-06-26), FAST22 = 9wQDMh3Lbjc (2022-11-08), AMX = _CP8d4nCnlw (2022-11-09), OPILOT = Ce528y1fI-0 (2022-06-19), IMNET = JjOLG5AjLqg (2023-01-31), MICRO = VdTxoCLfnak (2023-02-17), TENS = Ol485J8biQM (2025-05-25), FUZZ = BCnTXwhzzxA (2024-03-28), RDNA = NPinFkavsrk (2023-05-08).

## 1. The ten-line version

1. He **never optimises without a number**: first a reference (torch/BLAS/hipBLAS/cuBLAS/an open-source asm kernel), then the theoretical peak, then his own number; "where am I between the two" decides what to do.
2. He **computes the ceiling before touching code**: bandwidth ÷ bytes for decode (`42.6 TB/s ÷ 37 GB`), FLOPs from the kernel ÷ time for GEMM (`0.26 GFLOP in 0.15 ms = 1.7 TFLOPS`), MFMA cycles × units × clock for tensor peak (2.2 PFLOPS).
3. He **reads the generated code/graph** (DEBUG=4, disassembly, VIZ) to find wasted work: uncoalesced loads, gated loads plus an add of zeros, extra copies, double permutes, accumulators bouncing between registers.
4. He **bisects by deleting**: comment out the enqueue; disable global memory ("fake" kernel); do only the A load, then only the B load; put 2s in the registers; remove the local loads; each cut tells him which part owns the time.
5. He **counts kernels and bytes**: "61 versus 140 kernels" and "216 dispatches" are speed bugs by themselves.
6. He **separates the kinds of slowness**: Python/launch/compile vs scheduler (what gets fused, what gets recomputed) vs kernel (memory hierarchy, indexing, tensor cores). Each has a different fix and a different bound.
7. He **profiles the boring part** with an ordinary profiler (cProfile-like) when the "GPU" program is slow and finds Python (790 calls to `rand`, enqueue of scalars, logging).
8. He **checks correctness in tiny pieces** (all zeros, a single 1, ones, then random; one MFMA before a block of MFMAs; per-kernel CPU vs GPU compare) before looking at speed.
9. When stuck he **shrinks the problem to a minimal repro at the lowest API level** and asks the hardware a yes/no question; or **steals the answer** from the vendor kernel (decompile `metallib`, read the library's tile shape).
10. He **says out loud what he expects** before running ("I predict it will no longer get any faster at 64"; "expect 34.1, we get 30"), so that a surprise is visible.


---

# A. The loop, step by step (as observed)

He has no written method. The loop below is what repeats across 2022–2025 streams. Each step names the stream where I saw it.

## A0. Before touching anything: a reference and a ceiling

He opens almost every performance session by putting three numbers side by side: **baseline (what we have), reference (what a good library does), ceiling (what the hardware can do)**.

> "Tiny is currently getting 450 teraflops ... the torch gem gets uh 969 teraflop. So torch is about twice as fast as [tinygrad] and mma peak which we can get theoretically is about twice as fast as torch. So let's write a [matmul] in uops and see where we land." — [MI350 00:07:38](https://www.youtube.com/watch?v=4zwFDrsLm64&t=458s)

The "MMA peak" is a micro-benchmark that only issues tensor-core instructions (2.2 PFLOPS fp16/bf16 on MI350X) [MI350 00:06:53](https://www.youtube.com/watch?v=4zwFDrsLm64&t=413s). He then derives the same number by hand from the manual, to prove he understands it: 256 CUs, 1024 matrix cores, 16-cycle MFMA, 2.2 GHz, 16×16×32×2 flops per MFMA [MI350 01:41:12](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6072s).

Other instances: "this is the theoretical memory bandwidth, so it's 42.6 terabytes per second, that's terabytes divided by 37 gigabytes" for DeepSeek tokens/s [MOE 01:01:59](https://www.youtube.com/watch?v=upZf-BWF1i0&t=3719s); "let's try to figure out what's the theoretical gigaflops of this Ryzen" [MATMUL 00:21:48](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=1308s); "I should have told you beforehand what we'd expect so you guys wouldn't just think I'm pulling shit out of my ass" and then, for the 3080 Ti, "we're expect to see 34.1 and we get 30" [MICRO 01:13:12](https://www.youtube.com/watch?v=VdTxoCLfnak&t=4392s); "the stated bandwidth is 400 gigabytes per second" [MICRO 00:43:26](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2606s).

## A1. Run the smallest thing that can be wrong, and check it against a trusted implementation

Always before speed. On the MI350X he wrote "the stupidest [matmul] we can think of" first [MI350 00:20:02](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1202s), then compared to the normal tinygrad matmul with `allclose` and an absolute tolerance because half precision differs: "Wow. Halfs are awful." [MI350 00:26:10](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1570s). For a single tensor-core instruction he used a unit-test ladder that is worth copying literally: all zeros, all ones, a single element set to 1, move the 1 around until the first mismatch [MI350 01:10:47](https://www.youtube.com/watch?v=4zwFDrsLm64&t=4247s)–[MI350 01:14:47](https://www.youtube.com/watch?v=4zwFDrsLm64&t=4487s), then random.

> "let's focus on first principles and let's not just try to copy the other one." — [MI350 01:04:48](https://www.youtube.com/watch?v=4zwFDrsLm64&t=3888s)

He also runs the same program on another backend as an oracle (`CPU=1` runs the same UOp program on CPU [MI350 00:30:53](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1853s)); on the 9070XT he built a per-kernel `VALIDATE_WITH_CPU` mode (episode E16).

## A2. Look at structure before time: how many kernels, what are the copies

First look is the kernel list (`DEBUG=2`), then a graph view.

> "we can go here and run with debug equals 2. You can see there's four kernels and one of them is a copy." — [TOUR 00:03:19](https://www.youtube.com/watch?v=nyDGXlLp578&t=199s)

In 2022 the view was `GRAPH=1` with colours: "greens are free, greens don't actually do anything ... greens are movement ops. The gray ones ... it's not that the gray ones are slow, they're fast, the red ones are the slowest, but the problem is there's just so many gray ones and they're making copies and they're using memory" [OPILOT 01:23:10](https://www.youtube.com/watch?v=Ce528y1fI-0&t=4990s). In 2025 it is VIZ: tensor graph, kernel graph, memory graph, profiler [TOUR 00:07:13](https://www.youtube.com/watch?v=nyDGXlLp578&t=433s). What he looks for: **number of kernels** ("216 kernel dispatches" [OPILOT 05:15:42](https://www.youtube.com/watch?v=Ce528y1fI-0&t=18942s)), **copies**, **kernels that should have been fused**, **memory over time** [FSDP 00:40:55](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2455s).

## A3. Per-kernel numbers

`DEBUG=1`/`2` prints GFLOPS (and time) for every kernel: "if you do debug one it just tells you the G flops, that's how many G flops we're getting on stuff" [FAST22 02:57:40](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=10660s). When the framework does not tell him, he computes it: FLOPs of the op ÷ time ("that's 0.26 gigaflops and somehow it's doing it in point 15 ms ... 1.64 teraflops" [FAST22 00:20:36](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=1236s)), and on the MI350X he cross-checks the framework's own FLOP estimate against `GlobalCounters` and warns that the estimate comes from looking at the UOp program, so "there's a chance that it can be wrong" [MI350 01:31:44](https://www.youtube.com/watch?v=4zwFDrsLm64&t=5504s).

## A4. Read the generated code (and the assembly)

`DEBUG=4` prints source; higher debug prints the assembly; `NOOPT=1` shows the unoptimised kernel (only dark-blue globals and red reduces in VIZ) [TOUR 00:45:48](https://www.youtube.com/watch?v=nyDGXlLp578&t=2748s); `DEVECTORIZE=0` makes the code readable [BRRR 02:38:43](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=9523s). What he reads for:

- **coalescing**: "Look at that uncoalesced global memory load there. That is disgusting, boys." [MI350 00:17:17](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1037s); "not coalescing those writes, it should be a store four; none of the reads are grouped either" [RESNET 01:37:04](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5824s).
- **FMA fusion**: "you'll see that these are fused multiply adds" [FAST22 00:10:15](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=615s).
- **permutes/moves** that should not exist: "What is up with all the perms? The perms are the problem." [MI350 02:41:13](https://www.youtube.com/watch?v=4zwFDrsLm64&t=9673s)
- **accumulator register traffic**: "ACC VGPR read. This just stores them directly. What? Why does this one need to read from the ACC VGPRs?" [MI350 01:00:30](https://www.youtube.com/watch?v=4zwFDrsLm64&t=3630s); "LVM decides to be stupid and decides, 'Oh, let me just copy them back and forth.'" [MI350 03:07:34](https://www.youtube.com/watch?v=4zwFDrsLm64&t=11254s).
- **gated loads + adds of zeros** (cat kernel, E7).

## A5. Profile the host side with a normal profiler

When the "GPU" workload is slow but the kernels are short, he switches to a Python-level profile: `with Profiling():` [MOE 01:39:41](https://www.youtube.com/watch?v=upZf-BWF1i0&t=5981s), a `@profile` decorator [MCTS 01:29:32](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5372s), `PROFILE=1` + VIZ for GPU timelines [FSDP 00:46:00](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2760s). In 2022 it was plain `cProfile` output and later a hand-written profiler that prints milliseconds [OPILOT 04:40:53](https://www.youtube.com/watch?v=Ce528y1fI-0&t=16853s).

> "oh my god those are being copied to the gpu. no wonder it's so slow ... that's why you use a profiler" — [OPILOT 01:40:13](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6013s)

## A6. Bisect by deleting or faking parts

This is his most reusable move.

- "all the time is being spent here in this call [enqueue]. If I comment that out it's fast." [OPILOT 01:37:35](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5855s)
- Real conv vs "fake conv": "only real conv is slow" [OPILOT 01:39:48](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5988s).
- On the MI350X: a "fake" kernel with global memory disabled (2s put in registers), then only the A load, then only the B load, then no LDS loads, then LDS loads only [MI350 04:10:24](https://www.youtube.com/watch?v=4zwFDrsLm64&t=15024s), [MI350 03:10:30](https://www.youtube.com/watch?v=4zwFDrsLm64&t=11430s).

> "If I disable this chunk, will the compiler be smart and remove everything for me or can it actually run?" — [MI350 03:10:07](https://www.youtube.com/watch?v=4zwFDrsLm64&t=11407s)
> "if you take out the local loads, it gets that many T flops, which is pretty good, I guess." — [MI350 03:10:51](https://www.youtube.com/watch?v=4zwFDrsLm64&t=11451s)

He knows the trap: a compiler may delete the thing you removed work from ("these stupid compilers are smart and do like analysis and shit to know when like you didn't do it" [MICRO 00:55:55](https://www.youtube.com/watch?v=VdTxoCLfnak&t=3355s)).

## A7. Compare with a known-fast implementation and take its tricks one by one

On 2025-03-03 he put a hand-written asm SGEMM from a blog into tinygrad's runtime (41–43 TFLOPS fp32 on what I take to be an RDNA3 7900 XTX in his tinybox; he says tinygrad's best generated kernel at that time was about half of it [BRRR 02:11:50](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=7910s)) and then made tinygrad's optimiser reproduce each trick: "let's first get the optimizations to match" [BRRR 02:23:04](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=8584s). On the M1 he extracted Apple's own kernel (E10). For CPU matmul he kept numpy/OpenBLAS as the yardstick: "numpy matches the micro-architecture perfectly" [MATMUL 01:53:43](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=6823s).

## A8. Change one knob, run it again, keep or revert

On a single extracted kernel (RESNET): set `upcast` 2→8, run, "hey it's faster, look at that, let's go to 16 or something ... that's worse" [RESNET 01:31:43](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5503s); "not doing the upcast is actually faster, that's a good lesson" [RESNET 01:51:44](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=6704s). Then he hands the same knobs to search (BEAM, MCTS) and compares: "MCTS is winning most" [MCTS 01:50:26](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=6626s). He insists on a clean measurement of each knob: "run it 10 times" [RESNET 01:35:56](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5756s).

## A9. Write down what the real cost centre was

He closes sessions with a list ("lessons"), commits, and (2025) relies on a public CI performance page for tinygrad ("stats.tinygrad.win ... all the performance stats from the hosted CI" [FSDP 00:20:04](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=1204s); each point linked to a commit [CLOUD 04:05:55](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14755s)), so a regression is noticed. This is the ratchet.

## A10. The flag/tool table

| tool / flag | what he uses it for | where seen |
|---|---|---|
| `DEBUG=1/2` | kernel list, per-kernel time and GFLOPS, copies, BEAM progress | [TOUR 00:03:19](https://www.youtube.com/watch?v=nyDGXlLp578&t=199s), [FAST22 02:57:40](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=10660s) |
| `DEBUG=3` | candidate optimisations (BEAM), buffer args of a kernel | [MCTS 00:24:25](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=1465s), [FSDP 04:27:02](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=16022s) |
| `DEBUG=4` | generated source | [MI350 00:14:40](https://www.youtube.com/watch?v=4zwFDrsLm64&t=880s), [BRRR 01:22:22](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=4942s) |
| `DEBUG=5+` | assembly / disassembly | [RESNET 01:34:11](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5651s), [BRRR 01:22:22](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=4942s) |
| `NOOPT=1` | unoptimised kernel for a clean read | [TOUR 00:04:22](https://www.youtube.com/watch?v=nyDGXlLp578&t=262s), [TOUR 00:45:48](https://www.youtube.com/watch?v=nyDGXlLp578&t=2748s) |
| `DEVECTORIZE=0` | readable code | [BRRR 02:38:43](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=9523s) |
| `BEAM=n`, MCTS, JITBEAM | search over opts; "searching for faster kernels" | [MCTS 00:22:57](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=1377s), [FSDP 04:23:05](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=15785s) |
| `VIZ=1` | tensor graph, kernel graph, UOp rewrite steps, disassembly, memory graph, profiler | [MI350 00:29:02](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1742s), [TOUR 00:07:13](https://www.youtube.com/watch?v=nyDGXlLp578&t=433s) |
| `PROFILE=1` | GPU timeline for VIZ | [FSDP 00:46:00](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2760s) |
| `GRAPH=1` (2022) | coloured kernel graph | [OPILOT 01:20:19](https://www.youtube.com/watch?v=Ce528y1fI-0&t=4819s) |
| `with Profiling():` / `@profile` | Python-side profile | [MOE 01:39:41](https://www.youtube.com/watch?v=upZf-BWF1i0&t=5981s), [MCTS 01:29:32](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5372s) |
| `GlobalCounters` (ops, time sum) | FLOPS from the executed program | [MI350 01:30:56](https://www.youtube.com/watch?v=4zwFDrsLm64&t=5456s) |
| `VALIDATE_WITH_CPU` (his own, built live) | diff every kernel against CPU | [R9070 01:56:49](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=7009s) |
| `CPU=1` | run the same kernel on CPU as an oracle | [MI350 00:30:53](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1853s) |
| `TRACEMETA=0` | remove wrapper noise from the profile | [MOE 01:49:18](https://www.youtube.com/watch?v=upZf-BWF1i0&t=6558s) |
| `valgrind` / cachegrind (2022 CPU) | cache misses | [MATMUL 02:58:24](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=10704s) |
| NVIDIA's perf tools (2023) | "I used nvidia's performance benchmarking tools" | [MICRO 00:06:33](https://www.youtube.com/watch?v=VdTxoCLfnak&t=393s) |
| vendor ISA manual | MFMA cycles, register layouts, LDS size, instruction names | [MI350 00:09:16](https://www.youtube.com/watch?v=4zwFDrsLm64&t=556s), [MI350 01:46:42](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6402s) |

Notice what is **not** in his daily loop: Nsight/rocprof-style hardware counters. On AMD/Apple he says he has none ("I don't have any of this on metal ... a bunch of crappy performance counters which actually turn out to be completely useless" [MICRO 00:13:02](https://www.youtube.com/watch?v=VdTxoCLfnak&t=782s); "I love not having a profiler" [MI350 03:07:13](https://www.youtube.com/watch?v=4zwFDrsLm64&t=11233s); "We need SQTT to work" [MI350 03:59:54](https://www.youtube.com/watch?v=4zwFDrsLm64&t=14394s)). He asked for a bank-conflict profiler repeatedly [MI350 06:49:06](https://www.youtube.com/watch?v=4zwFDrsLm64&t=24546s). On NVIDIA in 2023 he did use NVIDIA's tools.

---

# B. Concrete episodes

Each episode: *symptom → how spotted → fix → how verified → numbers*. "Summary" = mine, quotes are caption text.

## E1. Scalar kernel arguments were each copied to the GPU as 4-byte buffers (openpilot model, 2022-06-19, [OPILOT 01:32:15](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5535s))

**Symptom.** He had just built "merge element-wise ops into the kernel" and expected a speed-up; the model did not get faster. Summary of the state: ~570 ms with the merge on (and "wrong"), ~800 ms before the work started.

> "okay so i don't understand why that's not faster. move to profile it. what's taking time, is the time being spent in the convolutions" — [OPILOT 01:33:00](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5580s)

**How spotted.** He wrapped the enqueue and profiled the Python side. The profile pointed to one call.

> "all the time is being spent here in this [call] right here. if i comment that out it's fast. okay that's fast." — [OPILOT 01:37:35](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5855s)
> "this is what you get for being 17 levels of abstraction away from the hardware" — [OPILOT 01:38:54](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5934s)

He then built a "fake conv" to separate the conv from the machinery: "doesn't seem slow, only real conv is slow" [OPILOT 01:39:48](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5988s). He suspected numpy int conversion, then saw the real reason:

> "oh my god those are being copied to the gpu. no wonder it's so slow ... that's going to be a big win for tinygrad too, and that's why you use a profiler." — [OPILOT 01:40:13](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6013s)
> "each one of those is constructing a four byte buffer and then copying that four byte buffer to the gpu and no wonder it's slow" — [OPILOT 01:42:16](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6136s)

**Fix.** Pass integer kernel args as scalars (`set_scalar_arg_types`) instead of making device buffers. A second stupidity found in the same profile: the enqueue loop sorted kernels by name: "why would you want to sort by name, what that's so dumb" [OPILOT 01:51:48](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6708s).

**Verified** by re-running the same model timing: "ah look it's fast now, look how fast that is now" [OPILOT 01:49:05](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6545s). Numbers in the captions after the fix: 48 ms (fake-conv config), 69 ms, 85 ms and 107 ms in the next ten minutes as he switched configurations ([OPILOT 01:50:41](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6641s), [OPILOT 01:51:06](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6666s), [OPILOT 01:51:28](https://www.youtube.com/watch?v=Ce528y1fI-0&t=6688s), [OPILOT 01:57:50](https://www.youtube.com/watch?v=Ce528y1fI-0&t=7070s)); the earlier ~570–800 ms is at [OPILOT 00:31:06](https://www.youtube.com/watch?v=Ce528y1fI-0&t=1866s), [OPILOT 01:32:15](https://www.youtube.com/watch?v=Ce528y1fI-0&t=5535s). The captions do not make it clean which config each number belongs to; the order of magnitude (about 5x) is clear.

**Why it matters for us.** The slowdown was *not in any kernel*. The check "is the time inside kernels or around them" is the first fork in his tree (see C1).

## E2. "Python is a lower bound": 24 ms of host time, then caching, then logs (same stream)

> "it's taking 24 milliseconds to run the python so this right now is a lower bound; even if our kernels were totally free they would be that slow" — [OPILOT 01:58:37](https://www.youtube.com/watch?v=Ce528y1fI-0&t=7117s)

Sequence of host-side fixes, each measured: LRU-cache the shape/graph work (36 → 25 ms [OPILOT 02:46:39](https://www.youtube.com/watch?v=Ce528y1fI-0&t=9999s), [OPILOT 03:09:49](https://www.youtube.com/watch?v=Ce528y1fI-0&t=11389s), then "okay 12 milliseconds in python, boom, oh that's fast now ... 13 milliseconds in python, 62 milliseconds total" [OPILOT 03:13:54](https://www.youtube.com/watch?v=Ce528y1fI-0&t=11634s)); then

> "49 milliseconds. we got rid of the logs. the logs, wow, wow, look at the time in python now. it was all just the logs." — [OPILOT 04:06:16](https://www.youtube.com/watch?v=Ce528y1fI-0&t=14776s)
> "don't log while timing" — [OPILOT 04:06:37](https://www.youtube.com/watch?v=Ce528y1fI-0&t=14797s)

and then "he's taking six milliseconds in python, it's taking 42 milliseconds in the kernels" [OPILOT 04:13:56](https://www.youtube.com/watch?v=Ce528y1fI-0&t=15236s), "it's taking 10 milliseconds to dispatch ... that's hard to do better, then cuda's just slow" [OPILOT 04:42:37](https://www.youtube.com/watch?v=Ce528y1fI-0&t=16957s), "we have a somewhat theoretical lower bound of 10 milliseconds here and we're at 26" [OPILOT 04:46:41](https://www.youtube.com/watch?v=Ce528y1fI-0&t=17201s), and finally "4 milliseconds in python" [OPILOT 05:02:51](https://www.youtube.com/watch?v=Ce528y1fI-0&t=18171s). External yardsticks he names: torch ≈35 ms, onnxruntime-CUDA with cache ≈3 ms, "three is the thing to compete with" [OPILOT 04:32:41](https://www.youtube.com/watch?v=Ce528y1fI-0&t=16361s).

**Lesson (his):** measure host time and device time as two numbers, and know which one bounds you. He computes the *floor* from the host side ("even if our kernels were free").

## E3. Kernel count is a metric: 410 → 227 → 216, and the double permute (same stream)

> "my optimizer is removing a lot of kernels, it goes down from 410 to 227 but this is 216 and there's like another 16 to lose" — [OPILOT 05:16:04](https://www.youtube.com/watch?v=Ce528y1fI-0&t=18964s)
> "this is where shape tracker needs to be improved so it's doing a double permute ... a permute and a reshape so it's just an extra copy on all of those" — [OPILOT 04:14:38](https://www.youtube.com/watch?v=Ce528y1fI-0&t=15278s)

Read from the graph: movement ops are free *if* the view system can absorb them; when it cannot, you pay a copy. He keeps counting kernels and dispatches as he optimises. The end state he wants: "pretty much only convolutions" (red), because "you have to convolve into a buffer" and everything else should be folded into them [OPILOT 05:19:21](https://www.youtube.com/watch?v=Ce528y1fI-0&t=19161s).

## E4. "We just stopped running the function twice" (BEAM/MCTS search speed, 2024-07-22)

Context: speeding the *search* that compiles and times candidate kernels.

First he measured where search time goes: "about half the time on compilation and half the time on running" [MCTS 01:03:01](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=3781s). Later:

> "oh I run that twice, wow. I run that always. that's kind of annoying. all right well that was a big waste of speed ... a big speed regression" — [MCTS 01:25:26](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5126s)
> "that was completely free. we made things both simpler and faster ... 33% faster" — [MCTS 01:26:09](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5169s)
> "free lunch in search ... we just stopped running the function twice" — [MCTS 01:27:33](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5253s)

Then he decorates the search with the profile helper and reads it: "ton of time spent in C-style render ... half the time is being spent doing stupid UOp graph ... symbolic floor div" [MCTS 01:30:20](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5420s)–[MCTS 01:31:02](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5462s); "look how nice tinygrad infrastructure is, that profiling just worked" [MCTS 01:31:25](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5485s).

He applies an Amdahl bound before choosing what to optimise:

> "if we're at 30 per second and we're spending 38% of our time on runtime, yeah, the best we could possibly do, even we got the other to zero, was 78 and that's not fast enough still" — [MCTS 01:42:08](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=6128s)

and a rule: "I don't really want to parallelize this, I just want to make it faster ... parallelization is always going to be a trick that's available to you whereas if you can straight up make improvements there are just wins out there" [MCTS 01:31:56](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5516s), [MCTS 01:37:44](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=5864s).

## E5. `__launch_bounds__` makes the CIFAR training "twice as fast" (same stream, 00:41:57)

> "you see this kernel modifier ... it makes hlb C far [hlb_cifar] twice as fast by basically specifying the max parameters it's going to launch with ... register pressure ... how did we discover it? it's not insider knowledge, it's open source, you just read the code, read the driver" — [MCTS 00:41:57](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=2517s), [MCTS 00:42:39](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=2559s)

Summary: the caption does not name the attribute. In the tinygrad source tree on this machine the HIP renderer emits `__attribute__((amdgpu_flat_work_group_size(1, N)))` and the CUDA renderer `__launch_bounds__(N)` (`tinygrad/renderer/cstyle.py`), where N is the product of the local sizes. My reading: telling the compiler the real maximum block size lets it use more registers per thread instead of assuming the worst case, which is what he calls register pressure. He found it by reading the driver docs and the generated code, not by guessing. CUDA analogue: `__launch_bounds__(maxThreads, minBlocks)`, `-maxrregcount`, and `-Xptxas -v` to see spills.

## E6. MoE bring-up: 6 s of "initialisation" was 790 `rand` calls (2025-03-09)

Before any kernel work he did the ceiling arithmetic (C2), then ran the model and found it slow to start.

> "why is this slow, let's try to figure it out. tinygrad has a useful helper called profiling and you can just do `with profiling`" — [MOE 01:39:41](https://www.youtube.com/watch?v=upZf-BWF1i0&t=5981s)
> "wow it's spending six seconds in Rand, that's annoying" — [MOE 01:41:07](https://www.youtube.com/watch?v=upZf-BWF1i0&t=6067s)
> "we're making 790 calls to rand even though it's only four layers ... four times 64 [experts] times some overhead" — [MOE 01:52:00](https://www.youtube.com/watch?v=upZf-BWF1i0&t=6720s)

**Fix:** one stacked weight tensor `[num_experts, ...]` instead of one tensor per expert: "we were actually creating separate tensors for the experts and it's probably not the right way to do it. You probably want to stack them" [MOE 01:58:24](https://www.youtube.com/watch?v=upZf-BWF1i0&t=7104s); result "162 milliseconds. everyone can wait that much" [MOE 01:52:46](https://www.youtube.com/watch?v=upZf-BWF1i0&t=6766s). Later "I put the experts in the matrix itself" (gather by expert index) and a to-do: it must not go to the CPU for top-k, and must be JIT-able [MOE 08:14:45](https://www.youtube.com/watch?v=upZf-BWF1i0&t=29685s), [MOE 08:13:23](https://www.youtube.com/watch?v=upZf-BWF1i0&t=29603s).

He also found his *correctness* bug not in tinygrad but in his weight loading: Hugging Face stores rotary dims in a different order. His comment is a rule of thumb: "whenever you see permute ... permutes where the shapes match are horrifying" [MOE 08:10:04](https://www.youtube.com/watch?v=upZf-BWF1i0&t=29404s); "tinygrad was correct and I was wrong ... I went through it methodically. Lessons were learned" [MOE 08:11:31](https://www.youtube.com/watch?v=upZf-BWF1i0&t=29491s).

## E7. Reading the code of a slow `cat` (2025-02-10)

**Symptom.** `cat` was 2x slower than torch's [CPU2 00:05:54](https://www.youtube.com/watch?v=c1UHi7dncvo&t=354s).

**How spotted.** `DEBUG=4`, then reading the C. Summary of his reading: tinygrad expresses concatenation as "a big empty tensor, first tensor plus zeros, zeros plus second tensor, added together" with a boolean "gate" deciding which half you are in; the loop reads both sources under masks and adds.

> "this is a very slow way to do this. what should really happen is that this loop should be broken into two loops ... you don't actually have to do the add" — [CPU2 00:08:37](https://www.youtube.com/watch?v=c1UHi7dncvo&t=517s)
> "torch is just going to basically do two memcopies" — [CPU2 00:12:20](https://www.youtube.com/watch?v=c1UHi7dncvo&t=740s)

Verification: none yet; he explains it as a missing rewrite rule (`x + 0 = x` style simplification of gated loads) and files it as a bounty. The point for us is the *method*: the symptom (2x) → read the code → name the wasted work (gated loads and add of zero) → state what the right code is (two memcpys) before fixing.

## E8. 150 kernels vs 61: a missing `eval()` (2025-02-23)

> "it's a lot of kernels compared to the tinygrad implementation ... 61 versus 140, I don't really see why that's doing that, looks like it's running everything twice for some reason ... it's scheduling 150 kernels" — [TORCH 02:10:07](https://www.youtube.com/watch?v=zDwRrhQ9PDc&t=7807s), [TORCH 02:10:49](https://www.youtube.com/watch?v=zDwRrhQ9PDc&t=7849s), [TORCH 02:11:12](https://www.youtube.com/watch?v=zDwRrhQ9PDc&t=7872s)

He first doubted tinygrad's scheduler; later, "we forgot eval on our model. you know how many bajillions of man years have been wasted ... look at how few kernels it is" [TORCH 02:55:00](https://www.youtube.com/watch?v=zDwRrhQ9PDc&t=10500s) (train-mode BatchNorm kept computing batch statistics). Method: compare kernel counts with a reference implementation of the same net; the diff is the bug. After the fix BEAM "just works" on torch code through the backend and ResNet-18 takes "12 milliseconds" [TORCH 02:58:00](https://www.youtube.com/watch?v=zDwRrhQ9PDc&t=10680s).

## E9. Reading a multi-GPU graph: "this view should move before the copy" (FSDP, 2025-05-18)

> "even though there's 10,000 you'll see that it's copied to both devices and then we chop it based on device number down to 5,000. Now in reality this view should be moving before the copy and right now we're wasting compute on that." — [FSDP 00:15:06](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=906s)

He reads reduction axes and strides off the graph ("you see the reduction across six and seven because it's a 5x5 convolution" [FSDP 00:16:12](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=972s); "only the channels dimension has one stride. See, the batch dimension and the width and height dimensions have zero strides" [FSDP 00:19:38](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=1178s)) and spots a ReLU written as compare-and-select instead of `max`. Then the memory graph:

> "look at all that wasted memory ... we just waste memory like it grows on trees" — [FSDP 00:40:35](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2435s)
> "We might not be ready for FSDP. There might be so much stupider stuff we're doing" — [FSDP 00:41:18](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2478s)
> "there's so much lower hanging fruit than FSDP ... this stupid disk copy which is happening every time" — [FSDP 00:43:05](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2585s)

Prioritisation by looking at the profile before building the feature. Later he suspects "duplicate buffers" in a BEAM failure and uses `DEBUG=3` to confirm ("why does it say there's only seven args there but I see 15" [FSDP 04:27:32](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=16052s)).

## E10. 3.7 → 8.1 TFLOPS on Apple M1 Max by reading Apple's own kernel (2023-02-18)

Setup: his 2048² matmul 3.7 TFLOPS, PyTorch/MPS ~7–8. Spent hours. Two instructive things.

(1) **A measurement bug that looked like a miracle:** "metal gets you bajillions of gigaflops ... the bad news is it gives you the wrong answer" because "these times are measured in seconds and we are looking for things measured in nanoseconds" [M1 04:36:05](https://www.youtube.com/watch?v=MksxUpucnKA&t=16565s)–[M1 04:36:47](https://www.youtube.com/watch?v=MksxUpucnKA&t=16607s); later "it ran so fast that we're dividing by zero again" [M1 04:49:04](https://www.youtube.com/watch?v=MksxUpucnKA&t=17344s). Rule: an absurdly high number is a bug in the clock or the work, not a result.

(2) **How he found the trick.** He unpacked Apple's `metallib`, disassembled to LLVM IR and read which kernel is loaded and what it does:

> "I used LLVM to find out which things it was actually loading, so these are all the kernels inside of [MPS]" — [M1 06:46:16](https://www.youtube.com/watch?v=MksxUpucnKA&t=24376s)
> "this is LLVM IR ... you see that there's this thing called async copy 2D. Now of course this is a secret, this is a secret API, so you don't get to use async copy 2D" — [M1 06:47:19](https://www.youtube.com/watch?v=MksxUpucnKA&t=24439s)
> "I saw async copy 2D and I was devastated, but then I saw this line and I'm like wait, what is this, why is the ... why are the things 4X? ... so I realized what it was doing was load in multiple simd group matrices ... so I needed to make a grid of 4x4 of the 8 by eights" — [M1 06:48:24](https://www.youtube.com/watch?v=MksxUpucnKA&t=24504s)
> "so I got 4.3 ... then 6.5 ... then it turns out if I increase the local work group size I can get it up to 6.7 ... then I stuck this thread group barrier in there just random chance and I got my 8 teraflops" — [M1 06:49:30](https://www.youtube.com/watch?v=MksxUpucnKA&t=24570s)

and then 8.1 with manual unrolling [M1 06:51:03](https://www.youtube.com/watch?v=MksxUpucnKA&t=24663s); he notes it only works for the right tile count ("it's worse if anything other than four there" [M1 06:57:48](https://www.youtube.com/watch?v=MksxUpucnKA&t=25068s)) and the right size ("see what happens if I make the matrix tinier, wow, it gets way slower" [M1 06:56:07](https://www.youtube.com/watch?v=MksxUpucnKA&t=24967s)). He is candid that he cannot explain the barrier: "I still can't get over that thread group barrier ... in reality it's probably doing the same thing as async copy" [M1 06:55:09](https://www.youtube.com/watch?v=MksxUpucnKA&t=24909s).

Also an earlier wrong lesson that (2) disproved: after a first simdgroup attempt at 1.7 TFLOPS he said "let this be a lesson to everybody who's like I'm gonna use [simd] groups, don't, they're really slow, right, and the truth is that lesson applies to nobody" [M1 06:40:22](https://www.youtube.com/watch?v=MksxUpucnKA&t=24022s).

## E11. Dropping a hand-written SGEMM into the compiler to learn what it does (2025-03-03, [BRRR 01:19:35](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=4775s))

> "y'all want fast [matmul]. Let's do fast [matmul]" — [BRRR 01:19:35](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=4775s)

He ported an open-source hand-tuned RDNA3 SGEMM (a blog's "fastest kernel") by pasting its assembly source into tinygrad's program slot and fixing the kernel-argument ABI (loading `N`, `alpha`, `beta`, pointers into the right scalar registers). While doing so the GPU hung and the driver reset itself: "look at that, Mech is active, issue a full reset ... everything works again" [BRRR 01:49:13](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=6553s). After fixing args: "41 teraflops ... that's a pretty good number of [teraflops]" [BRRR 02:02:48](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=7368s); later in the stream the same file reports "43 teraflops" with `FAST=1` [BRRR 06:40:41](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=24041s). Per matmul: 3.3 ms [BRRR 02:03:11](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=7391s).

> "this is the tinygrad optimization string that does double the tinygrad speed ... we should think about which of these tricks exist in tinygrad and which of these tricks don't" — [BRRR 02:11:50](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=7910s)
> "let's first get the optimizations to match" — [BRRR 02:23:04](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=8584s)

Procedure: compare global/local sizes of the fast kernel vs tinygrad's [BRRR 02:24:12](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=8652s); read the strides of the shape trackers to place local axes [BRRR 02:36:53](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=9413s); implement the missing trick (LDS tiling) as an optimisation pass live: "LDS tiling is the thing that tinygrad doesn't do, everything else we do" [BRRR 02:46:10](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=9970s); find that a barrier inside a loop needs a matching barrier at the end [BRRR 04:13:49](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=15229s); "all barriers must come after stores" [BRRR 04:20:50](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=15650s); place barriers "as late as possible but not after global" [BRRR 06:24:32](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=23072s); then register tiling by `upcast` [BRRR 04:38:09](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=16689s). He is explicit that the goal is a *search space* containing every trick:

> "the whole idea with tinygrad is going to be to take each one of these tricks and figure out how to apply them and search across all of them ... just going to look at all the CUDA kernels, see what tricks they're using, put the tricks into tinygrad and then be faster than torch" — [BRRR 04:57:40](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=17860s), [BRRR 04:58:01](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=17881s)

Caption caveat: the first ~75 minutes and many gaps are not technical.

## E12. The MI350X tensor-core GEMM in UOps (2025-11-01): a ladder with a lot of dead ends

This is the closest thing in the archive to our job (hand-writing a GEMM, memory hierarchy, MFMA). The numbers, in order of appearance (TFLOPS; MI350X; fp16 inputs; peak 2200; torch 969–978; tinygrad auto 314–320, with BEAM 450):

| time | what | TFLOPS |
|---|---|---|
| 00:07 | baselines | torch 969, tinygrad 450 (BEAM), peak 2200 [MI350 00:07:38](https://www.youtube.com/watch?v=4zwFDrsLm64&t=458s) |
| 01:30:35 | first correct large GEMM: one wave per block, no LDS, accumulators in registers, uncoalesced loads | **78** [MI350 01:30:35](https://www.youtube.com/watch?v=4zwFDrsLm64&t=5435s) |
| 02:25:49 | LDS tiles of 128x128, block-K, bank-conflict-avoiding layout | 61 ("not a lot of T flops") [MI350 02:25:49](https://www.youtube.com/watch?v=4zwFDrsLm64&t=8749s) |
| 02:32 | bigger blocks | 109, then 121 [MI350 02:32:18](https://www.youtube.com/watch?v=4zwFDrsLm64&t=9138s)–[MI350 02:32:39](https://www.youtube.com/watch?v=4zwFDrsLm64&t=9159s) |
| 02:38 | `#pragma unroll` on all loops | "60 down to 50" [MI350 02:38:50](https://www.youtube.com/watch?v=4zwFDrsLm64&t=9530s); code full of `v_perm` [MI350 02:41:13](https://www.youtube.com/watch?v=4zwFDrsLm64&t=9673s) |
| 02:49 | fix operand layout so K pairs are two halves in one register -> perms vanish | "up to 60" [MI350 02:49:11](https://www.youtube.com/watch?v=4zwFDrsLm64&t=10151s) |
| 03:12 | **ablation:** only LDS->register loads + MFMAs, no global traffic, one warp | **537** [MI350 03:13:18](https://www.youtube.com/watch?v=4zwFDrsLm64&t=11598s) |
| 03:20–03:36 | (still no global traffic) add warps via a special dim, then a loop | 100 → 233 → (a dip to 16 [MI350 03:32:35](https://www.youtube.com/watch?v=4zwFDrsLm64&t=12755s)) → 414 [MI350 03:20:08](https://www.youtube.com/watch?v=4zwFDrsLm64&t=12008s), [MI350 03:27:56](https://www.youtube.com/watch?v=4zwFDrsLm64&t=12476s), [MI350 03:36:05](https://www.youtube.com/watch?v=4zwFDrsLm64&t=12965s) |
| 03:54 | correct again after restructuring (caption does not say whether global memory is on) | 61 [MI350 03:54:46](https://www.youtube.com/watch?v=4zwFDrsLm64&t=14086s) |
| 04:27 | hand-written coalesced global->LDS copy | 101 [MI350 04:27:48](https://www.youtube.com/watch?v=4zwFDrsLm64&t=16068s) |
| 05:30 | vectorised/coalesced via the devectorizer logic | **192** [MI350 05:30:33](https://www.youtube.com/watch?v=4zwFDrsLm64&t=19833s) |
| 06:19 | double-buffer attempt | "we're actually doing two per. So we got to divide by two" (caption unclear) [MI350 06:19:17](https://www.youtube.com/watch?v=4zwFDrsLm64&t=22757s) |
| end | next steps: global->LDS direct loads (`buffer_load ... lds`), kill bank conflicts, "we need a profiler" | – [MI350 06:56:50](https://www.youtube.com/watch?v=4zwFDrsLm64&t=25010s), [MI350 07:01:57](https://www.youtube.com/watch?v=4zwFDrsLm64&t=25317s) |

He did **not** reach torch's 969 in this stream (best figure printed in the captions for a kernel that I believe includes the global loads: 192, about 20% of torch and 9% of peak; I could not confirm from the captions alone that global memory was on in that run). What the stream demonstrates is the method:

- **Ablate to find the ceiling of each stage.** 537 with global stores removed told him the MFMA+LDS inner loop was fine and the global path/occupancy was the problem. "You can kind of optimize these things in pieces" [MI350 03:12:36](https://www.youtube.com/watch?v=4zwFDrsLm64&t=11556s).
- **When a change makes it worse, believe it and diff the assembly.** "How did it get so much worse?" [MI350 03:20:08](https://www.youtube.com/watch?v=4zwFDrsLm64&t=12008s) → "if I change that back to an actual loop it gives me 500" [MI350 03:21:54](https://www.youtube.com/watch?v=4zwFDrsLm64&t=12114s); "Every time I do something it gets slower" [MI350 03:56:25](https://www.youtube.com/watch?v=4zwFDrsLm64&t=14185s).
- **Fix layout at the source of the instruction operands.** The `v_perm` flood came from packing two halves per register; fixing the *layout* removed it, not a compiler flag [MI350 02:42:20](https://www.youtube.com/watch?v=4zwFDrsLm64&t=9740s).
- **Know when the compiler is the problem.** LLVM copied accumulator registers around; "This compiler is stupid. Keep it in the same registers." [MI350 02:50:20](https://www.youtube.com/watch?v=4zwFDrsLm64&t=10220s); "we need some profiler to figure out why this is slow. And we need an RDNA renderer. This ends with us writing a compiler replacement." [MI350 02:51:25](https://www.youtube.com/watch?v=4zwFDrsLm64&t=10285s). Later: "If we're brute forcing, we've lost." [MI350 04:12:13](https://www.youtube.com/watch?v=4zwFDrsLm64&t=15133s); "I'm just like randomly putting in numbers now. We need a profiler. Need a profiler. We need an assembler." [MI350 05:01:54](https://www.youtube.com/watch?v=4zwFDrsLm64&t=18114s). That is his signal to change the *kind* of tool, not to grind harder.
- **Ablate the two input streams separately:** "What if we just do one of them fake? What if we just do the A's ... how about just the B's? ... they're both pretty terrible" [MI350 04:10:24](https://www.youtube.com/watch?v=4zwFDrsLm64&t=15024s).
- **Use an LLM as a checklist, not an author.** He asked ChatGPT for suggestions (`global_load_dwordx4`, double buffering, padding) and read them critically [MI350 05:32:28](https://www.youtube.com/watch?v=4zwFDrsLm64&t=19948s), [MI350 06:04:55](https://www.youtube.com/watch?v=4zwFDrsLm64&t=21895s); he also says "I almost never use it to write code and then copy and paste" [CPU2 02:16:56](https://www.youtube.com/watch?v=c1UHi7dncvo&t=8216s).

Hardware arithmetic he did on stream: LDS 160 KB per CU, so a 256x256x2-byte tile "doesn't fit", 128x128 does [MI350 01:46:42](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6402s)–[MI350 01:47:25](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6445s); 256 VGPR accumulators per thread "that's kind of what it's designed for" [MI350 01:45:12](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6312s); MFMA 16x16x32 takes the same 16 cycles as 16x16x16 on CDNA4: "So this is completely free. You get double the performance completely free." [MI350 00:15:49](https://www.youtube.com/watch?v=4zwFDrsLm64&t=949s).

## E13. FMA check, "2x off the peak", and "why is torch 25x faster" -> the AMX (2022-11-08/09)

> "that's 0.26 gigaflops and somehow it's doing it in point 15 ms ... so that's getting 1.7 teraflops here, I think that's right ... 1.64 teraflops" — [FAST22 00:20:36](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=1236s)–[FAST22 00:21:01](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=1261s)
> "we're pretty close to the theoretical max ... this one is like 2x off the theoretical max on the CPU ... how is it possible that torch is doing it 25x faster? ... because it's using the AMX coprocessor" — [FAST22 00:22:52](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=1372s)

So a 25x gap was explained by a *hardware unit*, not a software trick. Next day he programs the AMX: "on master it's 1189 times slower than torch, on [the new branch] it's 1.84 times slower" [AMX 04:25:09](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=15909s), with intermediate rungs "16x slower" [AMX 01:57:47](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=7067s), "8x" [AMX 03:15:43](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=11743s), "3x" [AMX 03:34:13](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=12853s), "2.74x ... we're reaching kind of where tinygrad's overhead is" [AMX 03:35:22](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=12922s). The last 2x he attributes (without proof) to "torch paid for the secret good version of AMX" [AMX 04:28:52](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=16132s).

Two smaller items: when the compiler would not emit FMAs for `max`-reduce, he looked for compiler flags (`ffp-contract=fast`) [FAST22 00:17:26](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=1046s); "we wasted all that time to figure out that you needed to put two in instead of three in order to get the fmas" [AMX 00:19:36](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=1176s), i.e. he records the cost of tiny details.

And a fusion data point (about the `mulsum` benchmark): "in PyTorch it has to construct a C which is A times B and then takes the sum of C, and you're wasting tons of memory bandwidth doing that, but in tinygrad it fuses those together" [FAST22 03:04:54](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=11094s).

## E14. CPU GEMM from 0.7 GFLOPS to 1.4 TFLOPS (2022-06-26)

Ladder (his words are in the captions, numbers are theirs): numpy 1 TFLOPS on his Ryzen [MATMUL 00:15:25](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=925s); naive C "a pathetic 0.7 gigaflops" [MATMUL 00:34:03](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=2043s); 3.3 [MATMUL 00:43:52](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=2632s); 16 after moving the K loop outside [MATMUL 02:01:01](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=7261s); 36 once FMAs appear [MATMUL 02:06:33](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=7593s); 50 with unrolling [MATMUL 02:10:32](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=7832s); 91 with register blocking [MATMUL 02:39:52](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=9592s) against numpy's 129 single-thread [MATMUL 03:46:40](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=13600s); finally multi-threaded "1.3 teraflops consistently" vs numpy "a pathetic 650" [MATMUL 06:11:43](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=22303s)–[MATMUL 06:12:04](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=22324s), theoretical 2.5 [MATMUL 05:51:26](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=21086s).

What he did at each rung: counted registers ("16 ymm registers, oh well, that's why four is the fastest" [MATMUL 02:32:05](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=9125s)), counted cache ("the l1 cache is only 32 kilobytes" [MATMUL 03:39:43](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=13183s); "let's just make it 768 or something just so we're under our L1 cache size" [MATMUL 03:41:10](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=13270s)), ran valgrind ("that loop looks fast, the problem is these things are cache misses" [MATMUL 02:58:24](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=10704s)), and isolated reads from compute: "if we're only doing the stores we're getting a good number of gflops ... it's the reads that kill you" [MATMUL 00:43:11](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=2591s). He refuses to jump to assembly early: "that's called premature optimization, don't do it" [MATMUL 01:55:25](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=6925s).

## E15. Micro-benchmarks and their sanity checks (2023-02-17)

His list: "the speed of global memory ... speed of local memory ... and the speed of FMAs" [MICRO 00:14:07](https://www.youtube.com/watch?v=VdTxoCLfnak&t=847s) plus empty-kernel launch (~7.5 µs on the M1 [MICRO 00:20:42](https://www.youtube.com/watch?v=VdTxoCLfnak&t=1242s)). The sanity checks are the interesting part:

> "how are the empty kernels the same speed as the global memory? that doesn't make any sense ... I feel like it's just lying to me and running the wrong program or just not running anything" — [MICRO 00:38:56](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2336s), [MICRO 00:39:26](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2366s)
> "maybe it's all just overhead [so] let's make this a little bigger" — [MICRO 00:40:41](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2441s)
> "we should multiply this by two because there's read and write" — [MICRO 00:43:26](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2606s)

Compared to published specs: Snapdragon 845 "memory bandwidth 29, and we get 24" [MICRO 00:47:22](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2842s); 3080 Ti "expect 34.1, we get 30" [MICRO 01:13:12](https://www.youtube.com/watch?v=VdTxoCLfnak&t=4392s); and the 4-loads-per-thread kernel got slower — "I bet you that's due to memory coalescing" [MICRO 00:51:28](https://www.youtube.com/watch?v=VdTxoCLfnak&t=3088s). Cross-check: same GEMM kernel CUDA 20 TFLOPS vs OpenCL 9 vs cuBLAS 21 vs untuned Triton 17 [MICRO 00:04:49](https://www.youtube.com/watch?v=VdTxoCLfnak&t=289s), [MICRO 00:11:15](https://www.youtube.com/watch?v=VdTxoCLfnak&t=675s).

## E16. New GPU gives wrong answers silently: per-kernel CPU diff, then a minimal repro (9070XT, 2025-03-16)

> "if instead of silently being wrong they had an assert in LLVM that said hey, we don't support that yet" — [R9070 01:28:09](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5289s)
> "I wish I had a simple way to just check every kernel with the CPU ... tinygrad kind of is the regression test for everybody" — [R9070 01:36:44](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5804s)

He disabled the JIT ("let's disable the JIT and let's find which kernel's wrong and let's do this at the tinygrad layer" [R9070 01:38:31](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5911s)), wrote a context variable `VALIDATE_WITH_CPU` that runs each scheduled item on the GPU and on a CPU copy of the buffers, then compared [R9070 01:56:49](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=7009s). Findings, with his probability estimate moving as evidence came: "now at 50% chance it's a bug in the compiler" [R9070 01:39:35](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5975s); "my odds that this is a compiler bug has gone down" [R9070 02:00:58](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=7258s); "half the elements don't match in this kernel ... it's one of the weight loading kernels" [R9070 02:20:07](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=8407s), [R9070 02:21:58](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=8518s); hypothesis "this global size is too large ... I really doubt that kernel is broken, I really doubt that's a compiler bug, we're nowhere near integer limits" [R9070 02:34:09](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=9249s). Then the minimal repro at the lowest layer (compile a program, allocate a buffer, launch with the big global size, fill a buffer):

> "we have now reduced it to the minimum" — [R9070 02:48:19](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=10099s)
> "it did exactly one megabyte of ones" — [R9070 02:49:45](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=10185s)

i.e. only the first 1 MB was written: a silent launch-size limit; HIP showed the same failure, so it is hardware/driver, not tinygrad. He kept a posture: "if you want to criticize somebody you have to show that you're better than them" [R9070 01:33:56](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5636s), and "this is why you never want to be an early adopter ... fine on a stream but I shouldn't be doing this at work" [R9070 01:39:59](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5999s).

## E17. Hand-optimising one ResNet kernel at a time (2023-09-30)

Workflow: schedule the model, pick a kernel, apply `upcast`/`local`/`unroll` by hand, run it 10–20 times, print GFLOPS, look at DEBUG=5. Results are small and local: "ah 108 G flops, look it's faster" by putting a dimension first [RESNET 01:41:40](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=6100s), then 86–90 in the next variants [RESNET 01:42:01](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=6121s), [RESNET 01:42:29](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=6149s); "not doing the upcast is actually faster" [RESNET 01:51:44](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=6704s). Reading the code of the best variant he finds a codegen bug ("not coalescing those writes ... none of the reads are grouped either ... see, look, we have bugs, we're finding bugs" [RESNET 01:37:04](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5824s)–[RESNET 01:39:39](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5979s)). At the end he totals the effect, and it is humbling:

> "we saved 02 [0.2] milliseconds. Why'd we focus on that kernel? That kernel wasn't the problem. This kernel is the problem." — [RESNET 02:03:31](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=7411s)
> "we should have focused on this kernel, this kernel could have saved us real time, but no, we focused on this kernel, but we did make it about 10% faster ... what's important is that we all learned" — [RESNET 02:07:06](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=7626s) (the caption of the millisecond figure just before this is garbled: "we saved 0. one of a millet")

Amdahl, after the fact. (He had not sorted kernels by time share first.)

## E18. "20 ms per step is 58 TFLOPS" (ImageNet/CIFAR bring-up, 2023-01-31)

> "no I can't be right, no no, never mind, it takes 20 milliseconds to do that. okay so the real their step time is 20 milliseconds, 20 milliseconds per step, if the step is this many ops, what's this divided by 20 milliseconds ... 58 teraflops" — [IMNET 02:18:11](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=8291s)–[IMNET 02:18:32](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=8312s)
> "that means some of the kernels have to be getting more than 58. 58 is the average teraflops" — [IMNET 02:24:32](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=8672s)

From the reference's step time and the op count he infers the reference *must* use tensor cores: "so it has to be using the tens[or] cores" [IMNET 02:26:08](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=8768s) (fp16 did not help his code), and counts his own: "it's doing 133 kernel dispatches now" [IMNET 02:15:55](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=8155s), "why are those not fused?" [IMNET 02:14:51](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=8091s); "so we are off by a factor of 41. great. with the optimizer we're only off by a factor of 20" [IMNET 02:07:42](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=7662s). Inferring that the reference's number implies a hardware feature is a recurring move of his.

## E19. Decode is memory-bound; "within a couple percent of the bandwidth or something is wrong" (2025)

> "batch size one LLMs are entirely bound by your memory bandwidth and accessing the weights, and if you aren't within a couple percent of the memory bandwidth there's something wrong with your runner." — [CPU1 03:11:48](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=11508s)
> (in response to "does flash attention improve batch-size-one LLM inference?" — [CPU1 03:11:03](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=11463s))

Same logic with numbers on other hardware: Tenstorrent "960 GB/s ... 8B llama ... 60 is the theoretical max ... we're getting like 47 out of a theoretical max of 60 ... unquantized" [TENS 02:37:00](https://www.youtube.com/watch?v=Ol485J8biQM&t=9420s)–[TENS 02:38:09](https://www.youtube.com/watch?v=Ol485J8biQM&t=9489s); DeepSeek "154 tokens per second in theory ... I would be extremely happy with 100" [MOE 01:19:01](https://www.youtube.com/watch?v=upZf-BWF1i0&t=4741s); CPU box thought-experiment "you'd only get one token per second [on a terabyte-per-second, terabyte-memory box] if you were running a non-mixture-of-experts LLM, but if you're running a mixture of experts one and your ratio is something like you're only accessing 10% of the weights then you're going to get 10 tokens per second" [CPU1 01:14:44](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=4484s). An LLM on CLOUD: "25 ms per token, we're getting kind of close to the bandwidth. I think this can actually go way faster if anyone even looked at it" [CLOUD 03:58:42](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14322s), and a waste called out: a Q6_K GGUF: "I think it's unpacking it to float 32, which is just a total waste [of] your RAM" [CLOUD 03:59:04](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14344s).

CPU analogue for a pure bandwidth loop: tinygrad's reduce got ~11 GB/s while torch got 71; he said "we'll write a minimum C program to try to spin through memory as fast as possible using ARM NEON extensions" [CPU2 00:40:49](https://www.youtube.com/watch?v=c1UHi7dncvo&t=2449s), got 71–82 GB/s [CPU2 01:15:31](https://www.youtube.com/watch?v=c1UHi7dncvo&t=4531s)–[CPU2 02:06:22](https://www.youtube.com/watch?v=c1UHi7dncvo&t=7582s), tried prefetch (no gain [CPU2 01:55:30](https://www.youtube.com/watch?v=c1UHi7dncvo&t=6930s)), and concluded the mismatch was the *two-stage split reduce* tuned for GPUs [CPU2 00:46:50](https://www.youtube.com/watch?v=c1UHi7dncvo&t=2810s), [CPU2 01:50:07](https://www.youtube.com/watch?v=c1UHi7dncvo&t=6607s). Also: "this isn't flop bound on the CPU" [CPU2 00:46:28](https://www.youtube.com/watch?v=c1UHi7dncvo&t=2788s).

## E20. Latency hiding by batching requests (CLOUD=1, 2024-11-12)

> "you can see the latency and it's about 200 milliseconds round trip ... every one of these things is unfortunately a round trip, but instead let's just start queuing them all up, because the only one that actually matters is copy out" — [CLOUD 00:28:54](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=1734s)–[CLOUD 00:29:36](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=1776s)

On Python overhead: "does Python affect perf?" ... "this is a graph API. We support graph APIs, like CUDA graph, but we also support Metal graph ... indirect command buffers. We also have this whole HCQ framework" ... "if you use OpenCL a ton of time is spent basically in malloc and free recreating the same structures over and over again" [CLOUD 04:08:20](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14900s)–[CLOUD 04:09:03](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14943s). And in the docs he wrote: "tinygrad has the best execution speed of any framework because it usually bypasses the GPU driver and prebuilds the command queue ... often even faster than CUDA graph ... very little to improve here as this is almost never the bottleneck" [CPU1 00:30:45](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=1845s).

## E21. Flaky hardware and drivers: reproduce first, then replace (2023–2025)

2023-05 RDNA: "okay now we have a reproduction ... we're ready to write a minimum reproduction for this crash" [RDNA 02:32:20](https://www.youtube.com/watch?v=NPinFkavsrk&t=9140s)–[RDNA 02:34:14](https://www.youtube.com/watch?v=NPinFkavsrk&t=9254s), picks a machine "that is fast to reboot" [RDNA 02:32:40](https://www.youtube.com/watch?v=NPinFkavsrk&t=9160s), reads the kernel driver source for a missing lock "you want to bet it's not locked" [RDNA 02:37:38](https://www.youtube.com/watch?v=NPinFkavsrk&t=9458s), tries the staging/upstream driver [RDNA 02:39:17](https://www.youtube.com/watch?v=NPinFkavsrk&t=9557s), two motherboards and four kernel versions [RDNA 05:28:13](https://www.youtube.com/watch?v=NPinFkavsrk&t=19693s). 2024-03 fuzzer: "this isn't actually a kernel crash, this is a GPU crash" [FUZZ 00:47:06](https://www.youtube.com/watch?v=BCnTXwhzzxA&t=2826s); finds the register map is already open source in the kernel tree and a user-space register debugger exists (UMR) [FUZZ 01:47:34](https://www.youtube.com/watch?v=BCnTXwhzzxA&t=6454s), [FUZZ 01:55:15](https://www.youtube.com/watch?v=BCnTXwhzzxA&t=6915s). By 2025 the decision is made: userspace map the GPU over PCIe, no AMD code; "the bugs are all in the kernel driver ... it doesn't crash" [BRRR 00:37:43](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=2263s), and a wedged GPU resets itself [BRRR 01:49:13](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=6553s).

---

# C. Thinking patterns and rules of thumb (each backed by an episode)

**C1. First fork: is the time inside the kernels or around them?** E1 (enqueue + scalar args), E2 (24 ms Python floor, logs), E4 (search time split 50/50 compile/run), E6 (6 s in `rand`), E20 (200 ms RTT). Each time his first move was a profiler on the host and a number for "device time vs wall time". Rule: *a slow GPU program is often not slow on the GPU.*

**C2. Compute the ceiling before you touch code.** E12/A0 (MMA peak, 2.2 PF), E19 (tokens/s = bandwidth ÷ bytes), E13 (1.7 TFLOPS from FLOPs ÷ time), MATMUL (cores × AVX lanes × clock), MICRO (spec sheet vs measured). He does the arithmetic out loud and writes it down; the gap tells him whether to look at software or hardware.

**C3. A big unexplained gap has a *mechanism*; find it before you chase it.** 25x vs torch = AMX unit [FAST22 00:22:52](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=1372s); the reference's 20 ms per step works out to 58 TFLOPS, so "it has to be using the tens[or] cores" [IMNET 02:26:08](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=8768s); Apple's 8 TFLOPS = 4x4 grid of 8x8 simdgroup matrices (E10). He does not try to hit a number he cannot explain.

**C4. Count things.** Kernels (410/227/216; 61 vs 140; 133), dispatches, registers (16 ymm; 256 VGPR accumulators), cache sizes (L1 32 KB, LDS 160 KB), bytes ("multiply by two because there's read and write"), flops per MFMA (16384). Counting makes waste visible without a profiler.

**C5. Isolate, then re-add.** "Fake" kernels, "only A", "only B", "no LDS loads", "only the stores" [MATMUL 00:43:11](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=2591s), "commenting out the call". Always followed by a step that puts the part back and measures the delta. He knows the failure mode (compiler deletes the removed work).

**C6. Compare with a faster implementation and copy its tricks one at a time.** BRRR (hand SGEMM), M1 (Apple's kernel), MATMUL (numpy/OpenBLAS), RESNET (hand-coded opts), IMNET (reference step time). Also as a life rule:

> "Stop thinking, start copying. A lot of people try to think and a lot of people try to be clever. In reality, the other people who you're copying from ... have thought about way more than you." — [FSDP 00:51:44](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=3104s)

**C7. Prioritise by share of total time, and say what the best case is.** [MCTS 01:42:08](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=6128s) (even at zero the other part gives 78/s); [RESNET 02:03:31](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=7411s) (wrong kernel, 0.2 ms); [FSDP 00:43:05](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2585s) (lower-hanging fruit than FSDP). In the 2022 openpilot stream he states a target in ms before starting ("looking for times about 10 milliseconds ... it's got to run in like 20 milliseconds on the device" [OPILOT 01:22:30](https://www.youtube.com/watch?v=Ce528y1fI-0&t=4950s)).

**C8. Distrust numbers that are too good, and numbers that are too bad.** Division by zero / "bajillions of gigaflops" (E10); "how are the empty kernels the same speed as the global memory? ... I feel like it's just lying to me" (E15, [MICRO 00:38:56](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2336s)); "there must be bugs because it's too fast" [MATMUL 01:16:34](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=4594s); "this is way too fast, this is insane" [IMNET 02:05:28](https://www.youtube.com/watch?v=JjOLG5AjLqg&t=7528s). Conversely "that's the worst teraflops I've ever seen" [MI350 03:31:30](https://www.youtube.com/watch?v=4zwFDrsLm64&t=12690s) triggers a look at the generated code rather than acceptance.

**C9. Make it work stupidly, then check, then make it fast.** The first MI350X kernel is "the stupidest [matmul] we can think of" [MI350 00:20:02](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1202s); MoE: "first make it slow and then we'll do it" [MOE 02:48:26](https://www.youtube.com/watch?v=upZf-BWF1i0&t=10106s); "premature optimization, don't do it" before assembly [MATMUL 01:55:25](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=6925s). And when his own complex derivation turned out wrong, the simple version won:

> "wow, it's so much simpler than like this one. I think this one's like just confused for no reason ... Why was I multiplying by two? It's so simple. ... You don't need any of that." — [MI350 01:15:32](https://www.youtube.com/watch?v=4zwFDrsLm64&t=4532s), [MI350 01:15:54](https://www.youtube.com/watch?v=4zwFDrsLm64&t=4554s)

**C10. Delete code; keep the surface small.** "Rule zero: surface all complexity" [CPU2 00:19:08](https://www.youtube.com/watch?v=c1UHi7dncvo&t=1148s); line budgets ("the library is 4,416 lines ... new goal is 5,000" [RESNET 02:08:33](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=7713s); "a debt ceiling" of 10K lines [CLOUD 04:08:20](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14900s)); cleanups at the end of a hack ("let's delete all these prints" [MOE 06:42:19](https://www.youtube.com/watch?v=upZf-BWF1i0&t=24139s)). He also says he codes slower off-stream and warns about performative speed: "you just code nice and slow, make sure it's correct, don't spam code" [MCTS 00:43:21](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=2601s).

**C11. Predict, then measure.** "At 64, I predict it will no longer get any faster" [GPUS 01:03:16](https://www.youtube.com/watch?v=OUzm06YaUsI&t=3796s) (then it didn't, he explains why); "expect 34.1, we get 30" [MICRO 01:13:12](https://www.youtube.com/watch?v=VdTxoCLfnak&t=4392s); "I should have told you beforehand what we'd expect". It converts the measurement into a test.

**C12. Primary sources over folklore.** ISA manual for cycles and layouts [MI350 00:09:16](https://www.youtube.com/watch?v=4zwFDrsLm64&t=556s); kernel driver source for the missing lock [RDNA 02:37:17](https://www.youtube.com/watch?v=NPinFkavsrk&t=9437s); register map "was open source the whole time" [FUZZ 02:17:35](https://www.youtube.com/watch?v=BCnTXwhzzxA&t=8255s); "reading docs, how else you want to do it" [MCTS 00:43:00](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=2580s). And he complains when the vendor docs are worse than the competitor's: "Nvidia has pictures, guys" [MI350 00:43:25](https://www.youtube.com/watch?v=4zwFDrsLm64&t=2605s).

**C13. Build the missing instrument immediately, even mid-hunt.** Micro-benchmarks (E15), a per-kernel validator (E16), a profiler that prints milliseconds [OPILOT 04:38:29](https://www.youtube.com/watch?v=Ce528y1fI-0&t=16709s), VIZ itself ("the best development environment for GPU stuff, hands down. We're putting all these profiler tools in here now" [MI350 00:29:46](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1786s)), an SQTT-based profiler [FSDP 00:46:00](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2760s), a stats page [FSDP 00:20:04](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=1204s).

**C14. Search for knobs, humans for structure.** BEAM/MCTS pick unroll/upcast/local (E4, A8) but trick *types* (LDS tiling, double buffering) must exist in the search space first (E11). So the loop: hand-write the best kernel you can → see which trick the search space lacks → add the trick as a searchable transform.

**C15. Say what "faster" means.** "What do you mean by faster, on what platform, on what model? Saying faster in general doesn't really mean anything" [MCTS 00:32:20](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=1940s). And his goal statement for the GEMM work that is relevant to our contest-style job:

> "Our goal isn't to be faster than PyTorch on things like gems, on things like Flash Attention ... Our goal is to be pretty much the same speed as PyTorch for the common paths, but then way faster on all the uncommon paths." — [MI350 06:45:19](https://www.youtube.com/watch?v=4zwFDrsLm64&t=24319s)

(Our case is the "uncommon paths": MoE W4A16, decode attention, whole layer.)

**C16. The unit of understanding is "what is the data movement."** The `speed` doc: "the main bottleneck in most kernels is accessing memory" [CPU1 00:46:13](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=2773s); "even cache is often much slower than accessing the register file" [CPU1 00:49:44](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=2984s); "tensor cores ... don't increase the total flops of the GPU but they allow the flops to be fully utilized with less data movement" [CPU1 00:55:01](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=3301s).

---

# D. Being stuck, hardware/driver weirdness, measurement noise

## D1. Being stuck

- **Shrink and re-derive.** Single MFMA, then zeros/ones/one-hot; "found a mismatch ... print both matrices" [MI350 01:12:46](https://www.youtube.com/watch?v=4zwFDrsLm64&t=4366s)–[MI350 01:13:13](https://www.youtube.com/watch?v=4zwFDrsLm64&t=4393s). When the manual is unreadable he copies the layout from a working implementation: "We can also potentially just cheat and look at how tiny does it because this is implemented correctly in tiny" [MI350 00:44:12](https://www.youtube.com/watch?v=4zwFDrsLm64&t=2652s), then insists on understanding: "let's focus on first principles" [MI350 01:04:48](https://www.youtube.com/watch?v=4zwFDrsLm64&t=3888s).
- **Change the tool, not the effort.** See E12: "If we're brute forcing, we've lost." [MI350 04:12:13](https://www.youtube.com/watch?v=4zwFDrsLm64&t=15133s); he moves down a level: assembly [CPU2 01:21:49](https://www.youtube.com/watch?v=c1UHi7dncvo&t=4909s), an assembler [MI350 05:01:54](https://www.youtube.com/watch?v=4zwFDrsLm64&t=18114s), his own driver [BRRR 00:37:43](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=2263s), a different compiler.
- **Make sure the thing is even running.** "I feel like it's just lying to me and running the wrong program" [MICRO 00:39:26](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2366s).
- **Hypotheses with probabilities.** "now at 50% chance it's a bug in the compiler, 50% chance something else" [R9070 01:39:35](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5975s), updated as tests come in [R9070 02:00:58](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=7258s).
- **Switch machine/version.** Two motherboards, four kernel versions, upstream driver [RDNA 05:28:13](https://www.youtube.com/watch?v=NPinFkavsrk&t=19693s); run on CPU as oracle [R9070 01:56:49](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=7009s).
- **Give the sub-problem away with a price.** $100 for the 8 TFLOPS recipe [M1 06:37:51](https://www.youtube.com/watch?v=MksxUpucnKA&t=23871s); "$500 if you can fix this, it's not that hard to fix, you just need to express these loads and stores and get them to cancel" [CPU2 00:27:31](https://www.youtube.com/watch?v=c1UHi7dncvo&t=1651s); "$300 ... find out how to modify the opt ops" [CPU2 01:50:07](https://www.youtube.com/watch?v=c1UHi7dncvo&t=6607s). Bounties are precise: a number or a test that must pass.
- **Take the break.** "do we have the energy for this right now, guys, or do we want to take a nap" [M1 03:05:13](https://www.youtube.com/watch?v=MksxUpucnKA&t=11113s); "some days I wake up ... and ask myself would an idiot do that and if the answer is yes I don't do that thing" [M1 06:42:04](https://www.youtube.com/watch?v=MksxUpucnKA&t=24124s). And he refuses to stop on a failure: "I don't like to end my streams as failures" [M1 07:10:08](https://www.youtube.com/watch?v=MksxUpucnKA&t=25808s) (he then continued for hours to get 8.1).
- **Persist on correctness:** "Never give up. Never surrender." [MI350 00:57:57](https://www.youtube.com/watch?v=4zwFDrsLm64&t=3477s) (said while the matrix was not close).

## D2. Hardware and driver weirdness

- **Silent wrongness is the worst class**, so make it loud: per-kernel validators (E16), asserts for hardware limits ("if you're building a thing you want developers to use you should put an assert" [R9070 02:37:11](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=9431s)), counts of what landed ("exactly one megabyte of ones" [R9070 02:49:45](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=10185s)).
- **A hang should be recoverable:** with his own user-space driver the GPU resets itself after a bad kernel and keeps going [BRRR 01:49:13](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=6553s); this is also what makes hand-written assembly experiments cheap.
- **Reproduce on the fast-reboot machine; read the driver source for a missing lock** [RDNA 02:32:40](https://www.youtube.com/watch?v=NPinFkavsrk&t=9160s), [RDNA 02:37:38](https://www.youtube.com/watch?v=NPinFkavsrk&t=9458s).
- **Find the register map and a register debugger before asking the vendor** (UMR) [FUZZ 01:47:34](https://www.youtube.com/watch?v=BCnTXwhzzxA&t=6454s), [FUZZ 01:55:15](https://www.youtube.com/watch?v=BCnTXwhzzxA&t=6915s).
- **Hardware limits that surprise:** grid dimension limit on the 9070XT (E16); local work-group size limit in OpenCL vs Metal (E10); the Z/accumulator register path on Apple's AMX (E13); register-bank structure (`ACC VGPR`) on CDNA (E12); LLVM silently mis-compiling an unsupported combination [R9070 01:28:09](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5289s).
- **Thermal and clocks:** the Snapdragon 845 "down clocked for temps" so measured bandwidth is 24 of 29 [MICRO 00:47:22](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2842s)–[MICRO 00:48:25](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2905s); Ryzen "throttled to 3400" [MATMUL 05:50:10](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=21010s).

## D3. Measurement noise (what he actually does)

- **Run it many times.** "I run it 10 times" [RESNET 01:35:56](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5756s); "I run the test eight times so trust me" [AMX 04:25:30](https://www.youtube.com/watch?v=_CP8d4nCnlw&t=15930s); "let's run it a few more times than three, let's run it five times" [M1 04:43:15](https://www.youtube.com/watch?v=MksxUpucnKA&t=16995s).
- **Make the problem bigger.** "There's a lot of variance there, we got to go bigger on the matrices" [MATMUL 05:23:17](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=19397s); "this is probably just all overhead [so] make it bigger" [MICRO 00:40:41](https://www.youtube.com/watch?v=VdTxoCLfnak&t=2441s).
- **Control the machine.** Pin the process to a core ("does everyone understand first off why it has to be pinned to a core? If anyone doesn't understand that, think about how caches work" [MATMUL 05:50:45](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=21045s)); set the CPU power governor to performance ("consistently getting 1.3 teraflops using the power of sleep" [MATMUL 06:11:43](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=22303s)); plug in the laptop [MICRO 00:27:55](https://www.youtube.com/watch?v=VdTxoCLfnak&t=1675s).
- **Do not log while timing** [OPILOT 04:06:37](https://www.youtube.com/watch?v=Ce528y1fI-0&t=14797s); the profiler itself costs: "see how much slower the profiler makes [viz]" [FSDP 00:46:00](https://www.youtube.com/watch?v=h8PzTlTcfDg&t=2760s).
- **Check the clock's units** (E10), and **synchronise** before timing ("is there a synchronize? this is synchronized, this is correct now" [FAST22 03:14:57](https://www.youtube.com/watch?v=9wQDMh3Lbjc&t=11697s)).
- **Stochastic searches can be unlucky:** "you could just get unlucky and get a slow one and then it exits" [MCTS 01:48:54](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=6534s); "I need that framework to do proper evaluation of these things" [MCTS 01:48:54](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=6534s). He accepts search noise as a reason for better evaluation harnesses, not for ignoring results.
- **Intermittent slowness is a clue:** "this only seems to be slow sometimes, I wonder why ... let's profile it" [MCTS 01:46:21](https://www.youtube.com/watch?v=R8uxmXmtOrk&t=6381s); he found it was dominated by fixed host costs for a 4 ms kernel.

---

# E. What he says about the topics you care about

Source quality note: these are scattered remarks inside long streams; where there is no real content I say "nothing".

## E.1 GEMM (dense, tensor cores, tiling)

- Mental model: GEMM as a cube. "instead of using dot, which is very confusing to me, you can imagine it's just like here's a square that's one matrix and the square is N by K, then here's another matrix K by N ... you can expand them out into a cube by broadcasting across that virtual dimension and then you can sum this way" [MICRO 00:09:06](https://www.youtube.com/watch?v=VdTxoCLfnak&t=546s). This is also how tinygrad expresses it (multiply then reduce), which is why fusion of the multiply into the reduce saves the intermediate (E13).
- Hierarchy he uses on MI350X: block -> 4 warps (256 threads) -> MFMA 16x16x32; LDS tile (128x128, 2-byte), registers for accumulators (256 VGPR/thread available), coalesced global->LDS copies, bank-conflict-free layout, then compute from LDS, double buffer [MI350 01:45:12](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6312s)–[MI350 02:08:07](https://www.youtube.com/watch?v=4zwFDrsLm64&t=7687s).
- Costs he found: operand layout (perms), accumulator register copying by the compiler, barrier placement, uncoalesced loads, bank conflicts ("bank conflicts like crazy" [MI350 06:49:06](https://www.youtube.com/watch?v=4zwFDrsLm64&t=24546s)).
- Tensor-core peak arithmetic, and the observation that CDNA4's 16x16x32 MFMA takes the same 16 cycles as the 16x16x16 one, i.e. double the flops for free [MI350 00:15:49](https://www.youtube.com/watch?v=4zwFDrsLm64&t=949s).
- Consumer-card caveat: "peak tensor flops with fp16 accumulate it's 330 but with fp32 accumulate it's only 165 ... product segmentation" [BRRR 00:43:17](https://www.youtube.com/watch?v=Pbbi3k5XTcs&t=2597s). (Relevant only as a reminder to check accumulate precision when quoting peaks.)
- Numerics: "tensor cores have different numerics ... different numeric stability properties" so compare with a tolerance, not equality [MI350 00:26:10](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1570s).

## E.2 Attention / flash attention

Very little concrete. What there is:

- Scheduler framing: flash attention "is basically all in this category" = deciding what gets grouped into one kernel and what is stored vs recomputed [CPU1 00:43:11](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=2591s).
- "I spent a lot of time trying to write automatic flash attention and then gave up. We'll get there." [MI350 01:34:18](https://www.youtube.com/watch?v=4zwFDrsLm64&t=5658s) (contract with AMD for a Llama 405B trainer, "we need flash attention").
- Decode: "does flash attention improve batch size one LLM inference? ... batch size one LLMs are entirely bound by your memory bandwidth and accessing the weights" [CPU1 03:11:25](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=11485s)–[CPU1 03:11:48](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=11508s). His implied answer: it is irrelevant to the decode ceiling, which is weight (and KV) bytes ÷ bandwidth. (He did not discuss KV-cache bytes in this answer.)
- KV cache is "the hardest thing to jit" [MOE 08:14:04](https://www.youtube.com/watch?v=upZf-BWF1i0&t=29644s); "we slow down quite a bit because our KV cache" [TENS 02:38:40](https://www.youtube.com/watch?v=Ol485J8biQM&t=9520s).
- He says the attention kernels are "the ones that are actually likely to be broken" on a new GPU [R9070 01:54:37](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=6877s).

## E.3 MoE

- Why it matters: weights are the bandwidth cost; "memory bandwidth and memory capacity basically the same thing" for dense models [MOE 00:47:48](https://www.youtube.com/watch?v=upZf-BWF1i0&t=2868s); "you should not be accessing every weight for every query" [MOE 00:48:29](https://www.youtube.com/watch?v=upZf-BWF1i0&t=2909s).
- Hard parts: "you're accessing different weights based on what your previous layer said" [MOE 00:50:00](https://www.youtube.com/watch?v=upZf-BWF1i0&t=3000s); naive references run every expert and mask: "why is it still running all the experts? ... it's going through every expert and running it and then multiplying it by something" [MIXTRAL 01:25:43](https://www.youtube.com/watch?v=H40QRJFzThQ&t=5143s)–[MIXTRAL 01:26:26](https://www.youtube.com/watch?v=H40QRJFzThQ&t=5186s); fix: "the gate outputs eight values, you can take the top two and softmax them" [MIXTRAL 01:30:50](https://www.youtube.com/watch?v=H40QRJFzThQ&t=5450s).
- Implementation: stack experts in one tensor, gather by index inside the graph, avoid CPU top-k, make it JIT-able [MOE 01:58:24](https://www.youtube.com/watch?v=upZf-BWF1i0&t=7104s), [MOE 08:13:23](https://www.youtube.com/watch?v=upZf-BWF1i0&t=29603s), [MOE 08:14:45](https://www.youtube.com/watch?v=upZf-BWF1i0&t=29685s).
- Ceiling: tokens/s = bandwidth ÷ activated bytes (E19); "shared expert" always runs [MOE 01:31:33](https://www.youtube.com/watch?v=upZf-BWF1i0&t=5493s); "64 experts crazy high number" [MOE 01:22:33](https://www.youtube.com/watch?v=upZf-BWF1i0&t=4953s).
- Nothing on grouped-GEMM scheduling, token sorting, or expert-parallel layouts.

## E.4 Quantised kernels

Nothing on W4A16 kernel construction. Remarks: "you can go from 16 to 8 basically for free ... I think four bits is too small" [GPUS 02:02:35](https://www.youtube.com/watch?v=OUzm06YaUsI&t=7355s); "if the model's actually trained with the quantization ... post quantization is kind of iffy" [MOE 01:17:03](https://www.youtube.com/watch?v=upZf-BWF1i0&t=4623s); "do I think quantization makes performance worse? yes, it definitely does, but how much worse? probably you're still better off" [MIXTRAL 02:08:09](https://www.youtube.com/watch?v=H40QRJFzThQ&t=7689s); a GGUF Q6_K unpacked to float32 "is just a total waste" [CLOUD 03:59:04](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14344s); "unquantized loading" as the denominator of the theoretical max [TENS 02:37:21](https://www.youtube.com/watch?v=Ol485J8biQM&t=9441s). Practical lesson: always compute the ceiling on *packed* bytes and make sure the code path never re-expands weights in memory.

## E.5 Memory bandwidth, fusion, launch overhead, graphs

- Bandwidth: see E19, E15. "If your interconnect bandwidth exceeds your memory bandwidth you can basically treat it like one GPU" [GPUS 02:19:37](https://www.youtube.com/watch?v=OUzm06YaUsI&t=8377s); on 4090s modded with more RAM: "they don't increase the RAM bandwidth ... if you only have a RAM bandwidth of 400 gigabytes a second, well, you can only read all that RAM [a limited number of times per second]" [GPUS 02:03:21](https://www.youtube.com/watch?v=OUzm06YaUsI&t=7401s).
- Arithmetic intensity, L1 bandwidth, reuse factor: [CPU1 00:50:07](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=3007s)–[CPU1 00:53:53](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=3233s).
- Indexing cost: "GPUs often have less integer math resources than floating point math, so this can sometimes be the [bottleneck]" [CPU1 01:00:34](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=3634s).
- Fusion: decide by bytes (E13); "when is it worth recomputing an arithmetic operation instead of storing and loading from memory" [CPU1 00:36:36](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=2196s); "cuDNN can't fuse the convolution kernel with every activation function so you're going to end up doing the convolution kernel and then the activation function and doing an extra [...]" [RESNET 02:13:39](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=8019s); he says the scheduler's decisions were "the main reason our BERT time was bad on the last MLPerf" [CPU1 00:42:28](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=2548s).
- Launch overhead: ~7.5 µs empty kernel (M1 OpenCL) [MICRO 00:20:42](https://www.youtube.com/watch?v=VdTxoCLfnak&t=1242s); "about 11 micros to launch a kernel and it is going through a bit of Python" [CPU2 00:05:32](https://www.youtube.com/watch?v=c1UHi7dncvo&t=332s); "1x1 add ... limited entirely by kernel dispatch speed" [CPU2 00:04:49](https://www.youtube.com/watch?v=c1UHi7dncvo&t=289s); dispatching 216 kernels in CUDA took "10 ms ... CUDA's just slow" [OPILOT 04:42:37](https://www.youtube.com/watch?v=Ce528y1fI-0&t=16957s).
- Graphs: "TinyJit ... bypasses the GPU driver and prebuilds the command queue ... often even faster than CUDA graph" [CPU1 00:31:05](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=1865s); graph APIs (CUDA graph, Metal ICB, HCQ) [CLOUD 04:08:43](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14923s).

## E.6 Hopper, tensor cores, new-generation features

Thin; do not treat as authority.

- "everything's going to move to like a producer-consumer model ... for the B200 you have to do a persistent kernel ... if you read the ThunderKittens mega kernel stuff, I think that's where everything is going, but this is AMD and we don't have to do any of that." [MI350 01:35:21](https://www.youtube.com/watch?v=4zwFDrsLm64&t=5721s), [MI350 01:37:55](https://www.youtube.com/watch?v=4zwFDrsLm64&t=5875s). "We can do warp specialization stuff if we have to ... I don't think we're going to do any of that." [MI350 01:44:50](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6290s)
- "newer Nvidia GPUs have a TMA, [that] is tensor memory accelerator, right, accelerator to assist with fast indexing; however this is not supported enti[rely] yet" [CPU1 01:04:26](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=3866s) (he is reading an LLM answer aloud; "yet" = in tinygrad).
- "async MMA: I don't know what you think async MMA really does" [MI350 00:11:28](https://www.youtube.com/watch?v=4zwFDrsLm64&t=688s); async global->LDS copy can be recognised as a pattern "up to the barrier" [MI350 02:06:32](https://www.youtube.com/watch?v=4zwFDrsLm64&t=7592s)–[MI350 02:07:14](https://www.youtube.com/watch?v=4zwFDrsLm64&t=7634s).
- Inspiration list for the future UOp "language": Tile-lang fragments, ThunderKittens [MI350 01:40:26](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6026s), [MI350 07:27:14](https://www.youtube.com/watch?v=4zwFDrsLm64&t=26834s).
- On beating libraries: asked about the timeline for being faster than PyTorch on B200s: "Never. Probably never. I'd say never for that one." [MI350 07:30:07](https://www.youtube.com/watch?v=4zwFDrsLm64&t=27007s); he says "I'm not 100% sure about the B200. I don't know enough about the B200's programming model" [MI350 01:35:00](https://www.youtube.com/watch?v=4zwFDrsLm64&t=5700s).
- H100: nothing of substance; he says "I have never run anything on an H100 in my life" [CLOUD 01:06:36](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=3996s) (2024-11) and prices a cloud H100 against 4090s ("2.5 4090s is one H100") [CLOUD 04:34:49](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=16489s). I found no Hopper-specific kernel work (wgmma, TMA, clusters) in the streams I read, and the newest streams in the archive (late 2025) are about agentic coding, not kernels.

---

# F. Practices we can adopt (each linked to the episode that shows it)

Format: **P#** practice — *evidence* — **H200 translation** (the translation is mine; the practice is his). H200 numbers are from public NVIDIA specs as I remember them (HBM3e ~4.8 TB/s; BF16 dense ~989 TFLOPS, FP8 ~1979; 132 SMs; ~227 KB usable shared memory per block; 64 K 32-bit registers per SM); verify on our box before relying on them.

## Set-up and ceilings

**P1. Put three numbers at the top of every kernel task: baseline, reference, ceiling.** *A0; E12 (450 / 969 / 2200); E19.* **H200:** baseline = current kernel; reference = cuBLAS/cuBLASLt/FlashInfer/vLLM-Marlin/CUTLASS or torch; ceiling = measured micro-benchmark (not spec sheet), see P3.

**P2. Derive the ceiling from first principles, on paper, before writing code.** *A0; E13 ("0.26 GFLOP in 0.15 ms = 1.7 TFLOPS"); E19.* **H200:** MoE decode step ≥ (active expert bytes + attention weight bytes + KV bytes) ÷ HBM bandwidth; W4A16 means packed bytes + scales/zeros; dense projection at large M ≥ 2·M·N·K ÷ tensor peak; write both and take the max.

**P3. Measure your own ceilings with micro-benchmarks and check them against the spec.** *E15 (global BW, local BW, FMA, empty-kernel launch; "expect 34.1, got 30"); E12 (MMA-peak program).* **H200:** a `bw_peak` copy kernel (read-only and read+write), an `mma_peak` kernel that only issues WGMMA from registers/smem, an empty-launch loop. Keep the numbers in the repo; report every kernel as % of the relevant ceiling.

**P4. Use his acceptance test for memory-bound kernels: "if you aren't within a couple percent of the memory bandwidth there's something wrong with your runner".** *E19 ([CPU1 03:11:48](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=11508s)); TENS 47/60.* **H200:** apply to decode attention (KV streaming) and to small-batch MoE experts; anything below ~85–90 % of measured peak needs an explanation written next to the number.

**P5. Predict the result before every run and write it in the log.** *C11; [GPUS 01:03:16](https://www.youtube.com/watch?v=OUzm06YaUsI&t=3796s); [MICRO 01:13:12](https://www.youtube.com/watch?v=VdTxoCLfnak&t=4392s).* **H200:** one line per experiment, of the form expected-number-and-why (my template); surprises become the todo list.

## Where is the time

**P6. First fork: time in kernels vs time around kernels.** *E1, E2, E4, E6, E20.* **H200:** `nsys` timeline of one decoder layer: sum of kernel durations vs wall time; look at launch gaps, `cudaMemcpy`, sync, Python. Use CUDA graphs or a persistent kernel if the gap is launch-bound; compute the host floor ("even if the kernels were free it would be this slow").

**P7. Count kernels, launches, copies, and bytes per layer; compare to a reference implementation of the same layer.** *E3 (410→227→216), E8 (61 vs 140), E18 (133).* **H200:** diff the kernel list of our layer vs vLLM/torch for the same input; every extra kernel/copy is a suspect; every missing fusion is a byte count.

**P8. Rank kernels by share of total time before hand-tuning; write the best case before starting.** *E17 (tuned the wrong kernel: 0.2 ms), E4 (Amdahl: 38 % runtime, best possible 78/s), E9.* **H200:** `nsys stats`/`ncu` sort by total time; log "if this kernel became free the layer would go from A to B".

**P9. Do not log or profile inside the timed region; warm up; repeat; report min and median; control clocks.** *D3 ([OPILOT 04:06:37](https://www.youtube.com/watch?v=Ce528y1fI-0&t=14797s), [RESNET 01:35:56](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5756s), [MATMUL 06:11:43](https://www.youtube.com/watch?v=VgSQ1GOC86s&t=22303s)).* **H200:** lock clocks (`nvidia-smi -lgc`), warm-up iterations, ≥20 repetitions, bigger problem when variance is high, keep GPU temperature/power in the log.

**P10. Treat an implausibly good number as a bug in the clock or the work; treat an implausibly bad one as a prompt to read the code.** *E10, E15, C8.* **H200:** check cudaEvent units, check output checksum against reference for every timed run, check the kernel actually launched (`ncu` kernel list), vary size.

## Reading the machine

**P11. Read the generated code every time a number surprises you.** *A4, E7, E12, E17.* **H200:** `cuobjdump -sass` / `nvdisasm` on the cubin; look for scalar `LDG.32` where `LDG.128` is expected (uncoalesced/unvectorised), `LDL/STL` (spills), `PRMT`/`MOV` storms around `HGMMA` accumulators, extra `BAR.SYNC`, and `-Xptxas -v` register/spill lines.

**P12. Check coalescing and vector width first on every global access.** *E12 ([MI350 00:17:17](https://www.youtube.com/watch?v=4zwFDrsLm64&t=1037s)), E17 ([RESNET 01:37:04](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=5824s)).* **H200:** `ncu` `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` per request ≈ ideal; use 16-byte loads (`cp.async` 16 B or TMA).

**P13. Count bank conflicts and fix by layout, not by hope.** *E12 ([MI350 06:49:06](https://www.youtube.com/watch?v=4zwFDrsLm64&t=24546s) "We have bank conflicts like crazy. We need the profiler"; [MI350 01:53:27](https://www.youtube.com/watch?v=4zwFDrsLm64&t=6807s) "Let's avoid bank conflicts").* **H200:** `ncu` `l1tex__data_bank_conflicts_pipe_lsu_mem_shared*`; use `ldmatrix`-compatible swizzles (the 128B swizzle for TMA/WGMMA).

**P14. If the compiler moves accumulators or inserts permutes, fix the *layout* that feeds the MMA.** *E12 (`v_perm` flood solved by packing K pairs).* **H200:** look for `PRMT` before `HGMMA`/`HMMA`; for W4A16 dequant, produce fragments in the register order the MMA wants (offline reorder of the packed weights) so that no shuffles remain.

**P15. Do the on-chip budget arithmetic before choosing a tile.** *E12 (a 256x256 tile of 2-byte values does not fit in the 160 KB LDS, 128x128 does; 256 VGPRs of accumulators per thread).* **H200:** smem for stages × (A tile + B tile + scales) ≤ ~227 KB; accumulators ≤ registers/thread × threads; write the table in the kernel's header comment.

**P16. Tell the compiler the launch shape.** *E5 (`launch_bounds` → "twice as fast").* **H200:** `__launch_bounds__(threads, minBlocks)`, `__maxnreg__`, check spills; try 128 vs 256 vs 384 threads (consumer/producer warp counts) deliberately.

**P17. Read the vendor's fast kernel to learn tile shapes and tricks.** *E10 (metallib → LLVM IR → 4x4 grid of 8x8), E11, C6.* **H200:** `ncu --print-kernel-base mangled` / kernel names of cuBLAS/cuBLASLt (they encode tile shape, stages), `cuobjdump -sass` of the library kernel, CUTLASS profiler results; compare tile shape, pipeline depth, cluster shape, epilogue to ours.

## Isolating and verifying

**P18. Build an ablation ladder for every kernel.** *A6, E12 (537 TFLOPS when global traffic removed; "only A", "only B"), E14 ("only stores"), E1.* **H200:** compile-time switches that (a) drop global loads (use constants), (b) drop one operand stream, (c) drop smem loads, (d) drop MMA, (e) drop epilogue. Tabulate TFLOPS or GB/s per variant: the biggest jump identifies the owner of the time. Guard against dead-code elimination (consume results via a cheap reduction or an asm volatile).

**P19. Correctness ladder for fragment layouts and dequant.** *A1, E12 (zeros, ones, one-hot, random; one MMA before a tile).* **H200:** for WGMMA descriptors/layouts and for the 4-bit unpack: test one tile with one-hot inputs; for W4A16 test one group with known scales/zeros, then a row, then the matrix; compare against fp32 reference with an explicit tolerance for fp16/bf16.

**P20. Always keep an oracle you can run per stage.** *E16 (`VALIDATE_WITH_CPU`), A1 (`CPU=1`).* **H200:** a harness that runs each kernel of the layer against torch fp32 on saved inputs and prints the first diverging stage and element index; run it on every commit.

**P21. Make silent failures loud.** *E16 (launch-size limit, "it did exactly one megabyte of ones" [R9070 02:49:45](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=10185s)); [R9070 01:28:09](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=5289s).* **H200:** assert grid dims, alignment, divisibility, `cudaGetLastError` after every launch in debug, `compute-sanitizer` nightly, a poison-fill of outputs to check coverage.

**P22. When stuck, reduce to a minimal repro at the lowest API and ask a yes/no question of the hardware.** *E16, E21.* **H200:** driver-API launch with a tiny cubin, buffer of ones, count what landed; vary one parameter at a time.

## Changing things

**P23. Port the tricks of a faster implementation one at a time and keep a ledger of "trick → delta".** *E11, A7.* **H200:** keep `tricks.md`: tile shape, stage count, swizzle, cluster, producer/consumer split, epilogue fusion, persistent scheduling, with the measured effect of each.

**P24. Change one knob, measure repeatedly, keep or revert; do not trust "it should help".** *A8, E17 ("not doing the upcast is actually faster").* **H200:** sweep tile/stage/warp counts in a script with fixed seeds; commit only a measured gain; record reverts too.

**P25. If a change makes things worse, diff the assembly of before/after instead of continuing.** *E12 ("How did it get so much worse?" → actual loop gives 500).* **H200:** `nvdisasm` diffs; check register count, spills, barrier count.

**P26. Use autotuning for the knob space; keep the structural choices manual; make the tuner itself fast and de-noised.** *C14, E4.* **H200:** autotune tiles/stages/warps per shape bucket; make sure the tuner never runs a candidate twice and separates compile time from run time; use multiple timing samples per candidate to avoid "unlucky" picks.

**P27. Decide fusions by bytes saved (and recompute vs store), not by launch count.** *E13 ("mulsum"), [CPU1 00:36:36](https://www.youtube.com/watch?v=pKE3LV2Rdyw&t=2196s), [RESNET 02:13:39](https://www.youtube.com/watch?v=AV5JXG8y4fA&t=8019s).* **H200:** for each boundary compute bytes written + re-read; fuse dequant into the GEMM prologue, activation/scale into the epilogue; for attention keep S/P on chip.

**P28. Never re-expand quantised weights in memory.** *E19 (GGUF Q6_K→fp32 "total waste").* **H200:** weights stay packed in HBM and are dequantised in registers; check bytes read ≈ packed size in `ncu`.

**P29. For MoE store experts stacked and index inside the graph/kernel; no CPU in the loop; no all-experts-then-mask.** *E6, MoE section ([MIXTRAL 01:25:43](https://www.youtube.com/watch?v=H40QRJFzThQ&t=5143s)).* **H200:** one `[E, N, K/2]` packed tensor, a router that emits token→expert lists on device, grouped GEMM with sorted tokens, graph-capturable end to end; confirm in `nsys` that no `cudaMemcpyDeviceToHost` sits in the layer.

**P30. Treat host overhead as its own budget: cache, batch, graph.** *E2, E20, E.5.* **H200:** CUDA graphs for the whole layer or the whole decode step; avoid per-step allocations ("a ton of time is spent in malloc and free recreating the same structures over and over again" [CLOUD 04:09:03](https://www.youtube.com/watch?v=Gm0t-kAsHuY&t=14943s)).

**P31. State in words what the right code would be before fixing.** *E7 ("should be two memcpys").* **H200:** write "what an ideal kernel does here" (bytes, instructions, barriers) in the PR; compare to the SASS.

## Process

**P32. Change the kind of tool when brute force stops working.** *E12 ("if we're brute forcing, we've lost"), D1.* **H200:** after 3 failed sweeps, stop sweeping: get an `ncu` section (warp-state stall reasons), look at SASS, or write the inner loop in PTX.

**P33. Build the missing instrument right away and keep it in the repo.** *C13.* **H200:** the micro-benchmarks (P3), an ablation switchboard (P18), a per-stage oracle (P20), a perf-dashboard row per commit (P34).

**P34. Keep a regression ratchet: perf number per commit in a table.** *A9 (stats page linked to commits).* **H200:** append `commit, shape, ms, %peak` to a CSV in CI; fail on >3 % regression.

**P35. Be explicit about what "faster" means: platform, model, shape, batch.** *C15.* **H200:** tables by (M, N, K, batch, expert count), not single numbers; report decode (M≤16) and prefill separately.

**P36. Record unverified explanations as hypotheses.** *E10 (barrier "probably doing the same thing as async copy"), E16 (compiler vs hardware probabilities).* **H200:** a `hypotheses.md` with expected evidence and the test that will confirm or kill it.

**P37. Use an LLM as a checklist generator, not an author; verify every suggestion with a number.** *E12 ([MI350 05:32:28](https://www.youtube.com/watch?v=4zwFDrsLm64&t=19948s)), [CPU2 02:16:56](https://www.youtube.com/watch?v=c1UHi7dncvo&t=8216s).* **H200:** ask for a list of known tricks (cp.async.bulk, 128B swizzle, split-K, stream-K, persistent CTAs, wgmma.mma_async scheduling), then test each.

**P38. Put a price on well-posed sub-problems and delegate them.** *D1 (bounties of $100–$500 with exact acceptance).* **H200:** hand a sub-agent a task with a number and a test (e.g. "reach ≥ X TB/s on the KV-stream kernel for these shapes").

---

# G. Limits of this evidence, and where not to copy him

1. **Most of his GPU-kernel hunts end unfinished or ambiguous.** The MI350X stream (7.6 h) ends below torch (best figure in the captions 192 of 969 TFLOPS) and with a to-do list; the M1 hunt took ~8 h and ended with an unexplained barrier. The value is the method, not the destination.
2. **His wins are overwhelmingly host-side, graph-side or "wrong kernel" fixes** (E1, E2, E4, E6, E8). The GPU-kernel-level wins that I could verify are: launch bounds (E5, "twice as fast" per his claim), the Apple 4x4 simdgroup grid (E10, 3.7→8.1), AMX (E13, 1189x→1.84x), and CPU GEMM (E14). I did not see him beat a vendor library on a Hopper-class GPU.
3. **No Hopper content.** I found no wgmma/TMA/cluster work, no W4A16 kernels, no decode-attention kernel (only the bandwidth argument), no grouped-GEMM for MoE. Everything in "H200 translation" is my extrapolation.
4. **Captions.** Auto-generated; numbers like "Giga flops" are often TFLOPS; some sequences of numbers cannot be assigned to a specific configuration (E1, E12 rows marked unclear). Quotes are literal captions; check the video at the linked second before citing a number publicly.
5. **Style traps.** He changes direction fast, tolerates long periods of printing, trusts his own diagnosis early (blames compiler; later retracts: [R9070 02:00:58](https://www.youtube.com/watch?v=KXjEQHmWPsQ&t=7258s)), and makes strong claims about companies and about speed ("Apple cheats", faster than PyTorch on B200: "never" [MI350 07:30:07](https://www.youtube.com/watch?v=4zwFDrsLm64&t=27007s)). I kept only the engineering content.
6. **Not covered:** the 2018 twitchslam "profiling" streams (checked: very little profiling content), the Edge-TPU/ANE reverse-engineering streams, the CUDA-userspace sniffing streams, the 2025 agentic-coding streams. The "auto optimizing a resnet 50" stream transcript could not be fetched (HTTP 429 on the caption endpoint; I stopped after three attempts).

# H. A one-page loop for our kernel work (derived)

1. Write baseline / reference / ceiling (P1–P3). Predict (P5).
2. Fork: kernel time vs wall time (P6). If wall ≫ kernels → host/launch/graph work first (P30).
3. Count kernels/copies/bytes vs reference (P7); rank by time share (P8).
4. For the top kernel: SASS + `ncu` (P11–P14); coalescing, spills, conflicts, permutes.
5. Ablation ladder (P18) → which stage owns the gap?
6. Port a trick from the fastest reference (P17, P23) → one change, repeated timing (P9, P24).
7. If worse: diff SASS (P25). If stuck: change tool (P32) or minimal repro (P22).
8. Verify against oracle on small shapes (P19–P20); log number, ratchet (P34), update ledger (P23).
