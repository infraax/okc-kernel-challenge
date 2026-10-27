# George Hotz (geohot) engineering philosophy, for GPU kernel speed work

The write tool refused to create `philosophy/ANALYSIS.md` ("subagents should return findings as text"). I did not work around it, so the full analysis is in this message. The raw source text is saved under `<workspace>`: 40 blog posts, the Lex #387 and Latent Space transcripts, 111 HN comments plus 6 HN threads, 34 PR threads, the tinygrad README/docs/CI files, release notes, and the Phoronix article.

How to read it:
- `QUOTE:` is verbatim from the cited source. Where I corrected machine-transcription errors, the fix is in [brackets].
- `SUMMARY:` is my paraphrase.
- `INFERENCE:` and `H200 TRANSLATION:` are my own reading, not Hotz.

Gaps:
- X/Twitter was not accessible. I only use tweets as quoted by third parties (Phoronix).
- His Twitch/YouTube streams are not covered; the sibling `streams` task has them.
- The bounty spreadsheet and `tinygrad.org/pitch.pdf` were not read (no PDF or sheet tooling). Bounty rules are reconstructed from the README and his PR comments.
- HN: the account `georgehotz` has 111 comments (I read all of them). The account `geohot` has 1. Reddit was not searched.
- Several stories are other people's PRs where geohot is the reviewer. I label whose words are whose.

---

## 0. Summary in six lines
1. He starts from the physical limit (FLOPs, bytes/s) and asks "what should the number be in theory?" before looking at code.
2. He thinks most wasted time is structural: code that serves other code, and data that is moved or computed but not needed. The cure is deleting layers, not tuning them.
3. He wants the whole stack visible and measurable. One env var (`DEBUG=2`) prints every kernel with FLOPs, bytes and GB/s.
4. He treats tests and CI as the signal for "safe to merge". A speedup claim without a benchmark is noise.
5. For kernels he bets on search over a simple, complete formulation (BEAM). By 2026 the repo also ships hand-written per-quant AMD LLM kernels, so he is pragmatic, not dogmatic.
6. He is often right about direction and often wrong about schedule and claims. Critics have fair points (section 5).

---

## 1. Core principles

### P1. "The best part is no part": delete requirements, not just code
Source: https://geohot.github.io/blog/jekyll/update/2025/12/29/five-years-of-tinygrad.html [2025-12-29]

QUOTE: "I think a lot of how software is thought about is wrong. All codebases have workarounds for issues in other parts of the codebase. Sometimes you are lucky and these workarounds are clear, but many times they are so deep and structural that you'll never see them reading the code line by line. I think this is so bad that 98% of lines of software are basically this in some way shape or form. tinygrad is following the Elon process for software."

QUOTE: "Make the requirements less dumb. The best part is no part."

QUOTE: "It's because each piece of code in that stack isn't focused on the goal. It's focused on the other pieces of code."

Practice (SUMMARY): before optimizing a step, ask whether the step must exist. For an LLM server the real requirement is "an OpenAI compatible API for quickly running LLMs on GPUs". For a kernel pipeline, layout transposes, dtype conversions, "make contiguous" copies and dequantize-everything steps are candidates for deletion, not tuning.

### P2. Simplicity is measured and enforced, and traded against speed deliberately
Sources: README https://github.com/tinygrad/tinygrad/blob/master/README.md (read 2026-10-25); Latent Space https://www.latent.space/p/geohot [2023-06]; PRs #1584, #1053, #9982.

QUOTE (README, Contributing): "No code golf! While low line count is a guiding light of this project, anything that remotely looks like code golf will be closed. The true goal is reducing complexity and increasing readability, and deleting `\n`s does nothing to help with that."

QUOTE (README): "Anything you claim is a "speedup" must be benchmarked. In general, the goal is simplicity, so even if your PR makes things marginally faster, you have to consider the tradeoff with maintainability and readability."

QUOTE (Latent Space, 00:10:10): "For a long time, TinyGrad had a hard limit at a thousand lines of code. And what this would force you to do is really make sure you were not wasting lines. I got rid of the restriction because it became a little code golfy at the end."

QUOTE (blog "Can tinygrad win?", 2025-07-06): "Line count is not a perfect proxy for complexity, but when you have differences of multiple orders of magnitude, it might mean something."

Evidence the rule is real (SUMMARY, from the repo):
- A CI bot comments a lines/tokens-per-line diff on every PR (`szdiff.yml`, `sz.py`). Tokens per line is tracked to expose golfing.
- CI has a hard cap: `MAX_LINE_COUNT=26500` in `test.yml`.
- Historical caps were 1000 lines (early 2023), 5000 (v0.8.0, 2024-01) and 8000 (v0.9.0, 2024-05).
- v0.13.0 had 22,853 lines. v0.14.0 (2026-08-24) had 26,040.
- His own PRs: "not worth the speed" (#1584, -53 lines), "Back off minor speed stuff for simplicity" (#1053), "simpler order = more speed" (#9982, -16 lines).

### P3. "Struggling to change many things = conceptual bug": refactor first
Source: PR #1141 https://github.com/tinygrad/tinygrad/pull/1141 [2023-07], geohot reviewing a "dedup kernel args" patch.

QUOTE: "Again, I look at this and I want to just click merge, but I can't. This should be 5 lines."

QUOTE: "I see how bufmap is written and it's clear to me this wasn't thought through, that does an expensive recomputation every time."

QUOTE: "So I spent about a half hour on this. It needs refactoring first, I don't see a clean way to write this in the current codebase. ... I think there's a -20 line refactor to improve this first, then deduping will be easy. ... The philosophy of tinygrad is when you are struggling and needing to change lots of things, there's a conceptual issue."

Lex #387 (02:13:41), QUOTE: "This is the main philosophy of tinygrad. You have never refactored enough. Your code can get smaller, your code can get simpler, your ideas can be more elegant."

Practice: if a perf fix touches six files, the structure is the bug. Do the enabling refactor as its own PR. The README asks for this: "prerequisite refactors before adding new functionality".

### P4. Make it visible: instrumentation is part of the product
Sources: Lex #387 (00:50:35); Latent Space (00:16:00-00:18:00); `docs/env_vars.md`; `tinygrad/viz/README.md`.

QUOTE (Lex): "If you run PyTorch on a GPU, you do some operation, and you don't know what kernels ran, you don't know how many kernels ran. You don't know how many flops were used. You don't know how much memory accesses were used. Tinygrad type debug equals two, and it will show you in this beautiful style, every kernel that's run, how many flops, and how many [bytes]."

QUOTE (Latent Space): "...it will tell you the exact number of flops and the exact number of memory accesses in each kernel. ... Just type debug equals two in any TinyGrad model."

SUMMARY:
- `DEBUG` is a ladder: devices, per-kernel timing and GB/s, applied opts, generated source, UOp IR, assembly.
- `VIZ=1` shows rewrites and runtime profiles. The v0.13.0 release added a `viz cli` "for both human and agent use".
- Hand-written kernels can declare `Estimates(ops=2*M*N*K, mem=...)` so the profiler shows their GFLOPS and GB/s (`extra/gemm/moe_gemm.py`, repo commit d16046f, 2026-10-24).

### P5. Theoretical limit first, then "how close are we?"
Sources: `docs/developer/speed.md`; PRs #9625, #812, #3000.

QUOTE (speed.md): "4090s have 1 TB/s of ram bandwidth and ~160 TFLOPS of compute, so you need to use each loaded value ~100 times. The L1 cache has around 40 TB/s of bandwidth, so in order to get full compute utilization you need to use each value ~4 times."

QUOTE (PR #9625, geohot, 2025-03-31): "I'm getting 53.3 tok/s on M3 Max with 400 GB/s. The model has 1.3B active params, for 2.6GB of weights, theoretical max is 400/2.6 = 153.8 tok/s. So this is close, can you get 76.9+ tok/s for the bounty?"
(76.9 is exactly 50% of the bandwidth bound.)

QUOTE (PR #812, 2023): "So there's two parts to this, the Python part and the GPU part. If you run with DEBUG=1 it'll separate them, I'm seeing 48.5 ms on my 7900XTX GPU. That theoretical max is the GPU part, the theoretical max of the Python part is probably like 0.1 ms."

QUOTE (PR #3000, 2024-01-10 and 2024-01-15): "Your script isn't printing GB/s. What GB/s do you expect?" and "Gb/s or GB/s? What should it be in theory?"

Practice: every benchmark number gets a unit, a bound and a fraction of the bound. His bounties are written as fractions of the bound.

### P6. A number that beats physics is a bug in the measurement
Source: PR #13379 "[BOUNTY]: llama 1B faster than torch on CPU in CI" https://github.com/tinygrad/tinygrad/pull/13379 [2025-11-21]. The PR body reported "tinygrad ... avg: 458.336 tok/s ... torch ... 3.885 tok/s".

QUOTE (geohot): "lol, where'd you get the 458 GB/s of ram bandwidth. it clearly isn't running the model"

INFERENCE: a ~1.2B-parameter model needs about 1.2 GB (int8) to 2.5 GB (fp16) of weight reads per token. So 458 tok/s would be roughly 0.5-1.1 TB/s of CPU RAM traffic, which is implausible. Hence "clearly isn't running the model". He caught it in one sentence because he did the arithmetic. The arithmetic is mine; the quote is exact.

### P7. Fusion first; most gains are "don't touch memory again"
Sources: HN 2022-11-04 https://news.ycombinator.com/item?id=33468485; Latent Space (00:12:00); speed.md.

QUOTE (HN, georgehotz): "Agreed. For anything at all common, most of the gains will be from fusion, the rest is just free."

QUOTE (Latent Space): "If you write that in NaivePyTorch, what it's going to do on the GPU is read A, read B in a kernel, and then store A times B in memory, and then launch another kernel to do A times B plus C. Okay, got to do those loads from memory. It's a whole extra round trip to memory that I just didn't have to do."

QUOTE (speed.md): "when is it worth recomputing an arithmetic operation instead of storing and loading from memory? ... with recompute (6 reads + 2 writes), no recompute (6 reads + 3 writes), so we should probably recompute. However, once you add movement ops and casts this is even harder to figure out. tinygrad doesn't yet have a systematic way to do it."

### P8. The beaten-path trap: odd shapes are where wasted work hides
Sources: HN 2022-11-04 https://news.ycombinator.com/item?id=33468051 and ...33471649.

QUOTE: "While this is true for most common GEMM looking ops, if you tread off the beaten path things get slow (odd channel sizes, batch sizes, etc...). Right now in PyTorch, GroupNorm is 2x slower than BatchNorm. There's no fundamental reason, just that the kernels loop over axes in a less than ideal order."

QUOTE: "Whole net performance at comma, when we switch from BatchNorm to GroupNorm it adds 70ms to the training step time, and it's -70ms for no norm. We also wrote a custom AllNorm that's like 10% slower than BatchNorm (and I put several hours into trying to optimize it)."

Practice: "no fundamental reason" is his phrase for "the number is below the bound for no physical reason, so there is a bug or implementation artifact to find". Test the shapes you really run, not only square GEMMs.

### P9. Neural nets are static; exploit it
Sources: Lex #387 (00:42:45-00:45:19); Latent Space (00:05:25); blog 2021-12-12 and 2023-05-24.

QUOTE (blog "A Correction on my AI chip thoughts", 2021-12-12): "Neural networks are DSPs. All the loads and stores can be statically computed, which isn't even possible for GPU workloads, never mind CPU ones."

QUOTE (Latent Space): "Every single time you run the net, you do the exact same set of loads, stores, and arithmetic. The only thing that changes is the data."

Practice: capture and replay. speed.md says "tinygrad has the best execution speed of any framework because it usually bypasses the GPU driver and prebuilds the command queue". For us, MoE routing is the data-dependent exception to "identical compute every time", so the work actually done must be audited (stories S2, S3).

### P10. Own the foundation so every bug is yours
Sources: HN 2025-05-26 https://news.ycombinator.com/item?id=44094090; HN 2024-08-09 https://news.ycombinator.com/item?id=41203887; HN 2024-09-23 ...41627189 and ...41626068.

QUOTE: "We also have our own complete driver/runtime now in tinygrad; it's so much nicer to build on a foundation when you can blame yourself for the bugs."

QUOTE: "They don't have consumer GPUs in CI, they have no fuzz testing, and instead of root causing bugs they seem to just twiddle things until the application works."

QUOTE (2024-09-23): "Those drivers are faster than anything else when used to run fixed command queues (what neural network runs are)". To a skeptic: "This sounds like prejudice. Have you benchmarked it?"

### P11. Trust in tests means speed of progress
Sources: Lex #387 (02:07:39, 02:14:03); Latent Space (00:26:34).

QUOTE (Lex): "how much do you trust your tests? ... the more you trust your tests, the more you're like, "Oh, I got a pull request, and the tests past, I feel okay to merge that," the faster you can make progress."

QUOTE (Latent Space): "One of the best tests in TinyGrad is a file called testops.py [test_ops.py]. And it's just a hundred small examples written in TinyGrad and PyTorch, and it checks both the forwards and backwards to make sure they match."

QUOTE (Latent Space): "I want that green check to mean I can merge this ... if you somehow manage to introduce a bug and get the green check, okay, we're fixing the test, top priority."

### P12. Search beats hand-coded special cases, if the search space is simple and complete
Sources: blog 2024-03-27 "Where the Bitter Lesson ends"; blog 2025-07-06 "Can tinygrad win?"; HN 2025-09-14 https://news.ycombinator.com/item?id=45236404.

QUOTE (2025-07-06): "I suspect there is a simple formulation of the problem underlying all of the scheduling."

The three steps, as quoted:
- "Expose the underlying search problem spanning several orders of magnitude. Due to the execution of neural networks not being data dependent, this problem is very amenable to search."
- "Make sure your formulation is simple and complete. Fully capture all dimensions of the search space. The optimization goal is simple, run faster."
- "Apply the state of the art in search. Burn compute. Use LLMs to guide. Use SAT solvers. Reinforcement learning. It doesn't matter, there's no way to cheat this goal. Just see if it runs faster."

QUOTE (HN 2025-09-14): "I think most compilers do far too little search and opt for heuristics instead, often because they don't have an integrated runtime environment" and "my dream is a language where I can specify what it does instead of how it does it. Like the concept of Halide's schedule but more generic."

### P13. Say "I was wrong" in public, quickly
Sources: blog 2021-12-12; PR #12822; PR #7434.

QUOTE (2021-12-12): "...my thoughts on VLIW and big chips were wrong." and "I'm kicking myself for getting this wrong."

QUOTE (PR #12822, geohot, 2025-11-13): "Overall, I think I was wrong about this and it makes it less readable. Feel free to reopen if you feel strongly otherwise"

QUOTE (PR #7434, 2024-10-31, reverting a speed PR): "Ugh, it got slower in CI"

### P14. Order-of-magnitude arithmetic as a thinking tool
Sources: blog "Brain FLOPS" 2022-02-17; "A Person of Compute" 2023-04-26; "Will I ever own a zettaflop?" 2026-01-26.

SUMMARY: he back-of-envelopes whole systems.
- Brain FLOPS: from neuron firing energy, "That's only 162 GFLOPS".
- "one person of compute = 20 PFLOPS (64 A100s)".
- Zettaflop power: "currently B200s are getting 10 TFLOPS/W".

This is the same style as 400/2.6. If the arithmetic is off by an order of magnitude from the claim, the claim is wrong.

---

## 2. His approach to performance

### 2.1 How he reasons about the limit (recurring recipe, SUMMARY)
1. Name the bounding resource: bytes (HBM/RAM/L2/L1/LDS), FLOPs, issue slots (integer address math), or launch/Python overhead.
2. Compute the machine ratio. speed.md: a 4090 is ~160 TFLOPS over ~1 TB/s, so about 100 uses per loaded value. "For operations like BS=1 GEMV, the arithmetic intensity is 1, but for GEMMs and convs it can be much higher."
3. For LLM decode, the bound is `tok/s <= bandwidth / weight bytes read per token` (PR #9625). He asks for at least 50% of it.
4. Report the fraction of the bound, not seconds.
5. Register pressure is part of the model. speed.md: "be careful of making [UPCAST] too large, as if there's too much register pressure on the GPU the warp scheduler may not be able to fit many warps, or even worse, it could be spilling to local memory."

H200 TRANSLATION (mine; public NVIDIA numbers, verify against your own measurements):
- H200 is 141 GB HBM3e at ~4.8 TB/s, dense BF16 ~989 TFLOPS, dense FP8 ~1979 TFLOPS.
- That gives ridge points of ~206 FLOP/byte (BF16) and ~412 FLOP/byte (FP8).
- Below the ridge you are bandwidth bound. MoE W4A16 decode (a handful of tokens per expert) and FP8 attention decode (KV read) live there.
- Method: bytes the kernel must read / 4.8e12 = the floor in seconds, then report % of it.

### 2.2 How he decides what to optimise
- He optimises what the profile says dominates, after checking the work is necessary (P1, S2-S4).
- He splits speed in speed.md into Compile Speed (Python), Execution Speed (driver), Model Speed (scheduler) and Kernel Speed (codegen). He names the bottleneck first. Of execution speed: "There's very little to improve here, as this is almost never the bottleneck." Of the scheduler: "This is currently a big bottleneck of training speed."
- He deprioritises micro-tuning. PR #12822 (2025-10-23): "Many of these kernels are super short runtime so it doesn't really matter." Lex (01:07:03): "I'm much more of a build it the right way and worry about performance later."
- Constants need device coverage. PR #16072 (2026-05-08), geohot: "So tuning this stuff is really annoying. We don't have a good way to test it on many different devices." The contributor claimed 53 to 86 GB/s on Metal from `MV_ROWS_PER_THREAD` 4 to 16. The PR was closed, and the thread then asks for a GPU in CI.

### 2.3 Abstractions and frameworks
- Many "CISC" primitives are suspect. QUOTE (Lex 00:46:16): "Torch has like 2000 different kernels. PyTorch 2.0 introduced Prim Torch, which has only 250. Tinygrad has order of magnitude 25. ... You can think about it as kind of RISC versus SISC [CISC], right?"
- Matmul is shape ops plus a multiply and a reduce over the n-cube. There is no matmul primitive (Lex 00:45:32; blog 2023-05-24).
- He is fair to PyTorch. Lex 00:50:16: "By the way, I really like PyTorch. I think that it's actually a very good piece of software."
- Latent Space: "In Kama [comma], I felt like I was competing against a lot of idiots. Here, I'm competing against smart, very smart people who've made some, I think, different trade-offs."
- He judges frameworks by whether you can see and control the kernel. Of PyTorch's partial laziness: "you're getting the worst possible worlds."
- Frontends are cheap, backends are the work. HN 2025-12-30 (id=46439454): "we fully encourage the frontend to stay PyTorch ... all frontends are welcome."
- Boilerplate heuristic, Lex (01:22:47): "if Codex or copilot are helping you, that actually probably means that your framework or library is bad and there's too much boilerplate in it."

### 2.4 Hand-written kernels vs search (BEAM) vs compilers (his practice moved over time)
- **2022.** HN 2022-11-04 (id=33468274): "benchmark based compilation, that's already happening in the tinygrad compiler we use for openpilot to determine the local group size. ... The end dream is some ML guided search to optimize the kernels :)"
- **2023-24.** `BEAM=2` is "Powerful and fast kernel beam search" (v0.8.0 notes). Hand-coded opts and tensor cores were added to the search space (PR titles #2137 and #3275).
  - The stable-diffusion <324 ms bounty (PR #2129) was won with search plus heuristics. The search took ~80 minutes from scratch; knobs were `LATEBEAM=2` and an env-gated reduce-split threshold.
- **2025-11**, HN HipKittens thread (...id=45938961), QUOTE: "1 and 2 are supported, 1 [LDS layouts to avoid bank conflicts] you need to specify, 2 [L2 read patterns] will be found with BEAM. We are working on reimplementing HipKittens in tinygrad, all the stuff is there to do it. See the amd_uop_matmul example. tinygrad doesn't support 3 [warp specialisation] yet, it's not needed on any AMD GPUs, and not needed on NVIDIA consumer. It wouldn't be hard to add, but it's important to figure out how it best fits with the existing abstractions. I think everything will eventually move to a more producer-consumer model."
  - SUMMARY: he specifies the structure (layouts) by hand, searches the parameters, and refuses a one-off feature that breaks the abstraction.
- **2026**, repo:
  - Commit 0c0fadb51 (2026-09-23, Hotz): "RDNA4 LLM kernels ... fast kernels for the UD quants". This is `tinygrad/llm/kernels/amd.py`, ~900 lines.
  - Commit a0a901c8e: "faster qwen 3.8".
  - `extra/gemm/moe_gemm.py` wraps a HipKittens grouped MXFP8 MoE GEMM.
  - BEAM is still used: "restore qwen NV beam speed to 38 tok/s", "fix qwen 27b beam search".
  - INFERENCE: the practical rule is "state the kernel structure explicitly where search cannot find it or is too slow, but keep it in the same IR, profiler and test loop, and keep it small".
- **Assembly.** The tinygrad X post quoted by Phoronix (https://www.phoronix.com/news/Tiny-Sovereign-Stack-AMD-Close, 2025-01-15): "There's a $1,000 bounty for an RDNA3 assembler that comes within 10% of the LLVM based one." The goal was owning the stack (no LLVM/ROCm), not peak speed.
- **Compilers.** Blog 2025-09-12: "Or we could, you know, do the hard work and build better programming languages, compilers, and libraries."

---

## 3. Debugging and speed-bug stories
Evidence grade: [A] Hotz's own words or commit; [B] his review of someone else's finding; [C] third-party report involving his rules.

**S1. A "speedup" that could not be real (458 "GB/s") [B]** (PR #13379, 2025-11-21)
- Wrong: a CI benchmark claimed 458 tok/s for Llama-3.2-1B on CPU vs torch 3.9 tok/s.
- Found by: one-line bandwidth sanity check (P6).
- Lesson: before celebrating, convert tok/s to bytes/s and compare to the memory system.

**S2. MoE decode vs the bandwidth bound [B]** (PR #9625 "OLMoE speed up", 2025-03 to 2025-08)
- A contributor (geohotstan) got 3.7x to 5.4x from "magical fuse arange" (`FUSE_ARANGE`, which fuses the arange used for gather/index into the consumer kernel).
  - PR body numbers with `JITBEAM=2`: 34.77 ms to 9.49 ms at 2 layers; 123.58 ms to 23.02 ms at 8 layers.
  - The contributor's own worry, QUOTE: "Not sure if this fixes the "multiplying by 0 instead of not computing" gotta investigate." That is the MoE failure mode: compute every expert and mask, instead of reading only routed experts.
- Hotz measured 53.3 tok/s on M3 Max, computed the 153.8 tok/s bound, and set the bounty bar at 76.9+. Later he got "57.5 tok/s with FUSE_ARANGE=1" (2025-05-05).
- Final state (chenyu, 2025-08-08): master was at 69.2 tok/s, which is 45% of the bound. `FUSE_ARANGE` became the default. The bounty was unlocked for others, not paid on this PR.
- Lesson: turn "fast" into "% of bandwidth bound". The 50% bar was not demonstrably cleared in the thread.

**S3. "Dequantize only the routed experts" [C, exactly our domain]** (PR #17610, 2026-08-19)
- Wrong (EthanBnntt's PR text): "IQ3_XXS expert tensors were dequantized as a complete (experts, out, in) tensor before ExpertWeights selected the routed experts. Decode therefore expanded and read every expert even though only the top-k routes contributed to the output."
- Found by the "reported MB/token" traffic counter. Before and after on a Lambda H100 80GB, Qwen3.6-35B-A3B IQ3_XXS, CUDA:

  | | MB/token | ms/token |
  |---|---:|---:|
  | master | 83,525 | 222.23 |
  | patched | 3,412 | 161.49 |

  Traffic fell 24.5x and decode got 1.38x faster, with identical tokens.
- Hotz's reaction was "no AI" (the PR disclosed AI use). The process was rejected, not the technique.
- INFERENCE: a 24.5x traffic cut gave only 1.38x speed. After removing the waste, something else (launch overhead, dequant compute, attention, other layers) dominates. His method says to recompute the bound from the new byte count and re-profile.

**S4. Same kernel compiled N times because offsets leaked into ASTs [B]** (PR #15052, 2026-04-09)
- Claim (mantissabit): per-weight slice offsets fold into INDEX expressions, making each layer's `ast.key` unique, so ~146 kernels all miss the method cache.
- Hotz: "This should be found later by rewrite, you shouldn't need to do it explicitly. Is it not?" The contributor found it already was handled. The PR shrank to a 21-line regression test. Hotz: "That's a very complex looking test".
- Lesson: ask whether the compiler's general mechanism already handles it. If yes, add the test, not the special case. Count unique kernels as a metric.

**S5. Duplicate buffers in kernel args led to a redesign, not a patch [B]** (PR #1141, 2023-07)
- A +40/-31 line patch worked. Hotz refused it because the root cause was the abstraction (global, local and const buffers in one list). He did a refactor PR (#1256) so the fix became small. Quotes are in P3.

**S6. Python time is not GPU time: LLaMA 7B, 450 ms to 106 ms [A]** (PRs #812, #987, #1075, 2023-05 to 2023-07)
- Method: `DEBUG=1` separates Python from GPU time. On a 7900XTX the GPU part was 48.5 ms, so the rest was Python overhead in lazy.py and shapetracker.
- He posted 10-run `mean runtime` tables. PR #987: codegen 149 to 103 ms, methodcache 142 to 91 ms, profile 578 to 362 ms.
- His running log: "450 -> 400", "332.21 ms down from 450!", "286.11 ms!", "I saw a 106.61 ms!" (the bounty target was 100 ms).
- Lesson: once kernels are good, dispatch dominates, so replay a prebuilt queue.

**S7. int8 "faster because of bandwidth" was actually swapping [A]** (PR #1289, 2023-07-20)
- Contributor table: 30B fp16 1534 ms vs int8 259 ms per token on an M2 Max, attributed to memory bandwidth.
- Hotz: "The 30B speed isn't memory bandwidth, it's amount of memory. You are probably swapping to disk for the FP16 one."
- He also demanded a streaming quantizer ("I want to avoid high peak memory usage") and a real eval (MMLU, TriviaQA, HumanEval and others). He paid after "LGTM, will test".
- Lesson: when a speedup is bigger than your explanation, find the real cause.

**S8. Ring allreduce: "What's the units?", then variance [A]** (PR #3000, 2024-01)
- First run: ring 0.62 s vs naive 1.20 s average. Second run: "Hmm, ran a second time, slower."
- Proper script: naive 9.81 "GFLOP/s" / 117.75 Gb/s vs ring 118.18 "GFLOP/s" / 246.02 Gb/s. He then asked "Gb/s or GB/s? What should it be in theory?"
- Lesson: the first iteration (8.87 s, then 0.35 s) and run-to-run variance are noise. Use steady state, repeat, and compare to link bandwidth.

**S9. "Wow that seems like bug levels of worse" [A]** (PR #8759, 2025-01-27/28)
- A refactor raised Python time on BERT and resnet. Hotz asked "How much slower? Any idea why?", then "Wow that seems like bug levels of worse. Any idea why?"
- Lesson: an unexplained regression is a defect until explained.

**S10. BatchNorm vs GroupNorm, 70 ms [A]** (HN 2022-11-04, P8)
- Claimed cause: loop order in an off-the-beaten-path kernel, "no fundamental reason". "Several hours" on a custom `AllNorm` ended about 10% slower than BatchNorm. He reports failures too.

**S11. Qualcomm GPU: tinygrad 2x faster than the vendor library, 5x slower than PyTorch on NVIDIA [A]** (Lex 01:07:03; Latent Space 00:18:36, 2023)
- Lex QUOTE: "The only place where tinygrad is competitive performance wise right now is on Qualcomm GPUs. ... it's two x faster than Qualcomm's library."
- Reason given: image textures are faster than buffers on mobile GPUs. Hand-coded libraries (Xiaomi MACE) are good on 3x3 convs and dense matmul, but "the minute you go off the beaten path a tiny bit, well your performance is nothing."
- Latent Space QUOTE: "on NVIDIA, it's about 5x slower than PyTorch right now."

**S12. A driver that "fixes" by twiddling [A]** (blog 2023-06-07; HN 2024-04-01)
- "Before my big AMD driver rant, I built master from ROCK-Kernel-Driver (because it's rude to complain before you try master)." He reproduced kernel panics with AMD's demo apps in a loop.
- HN 2024-04-01: "unless you root cause issues and have a good CI, you'll never build a stable GPU for general compute."

**S13. Regression guards for speed numbers [A]** (commits 18327 and 18466, 2026-09)
- "restore qwen NV beam speed to 38 tok/s" touches `test/unit/test_gguf.py` (+7). "fix qwen 27b beam search" adds `test/opt/test_search.py` (+20).
- Pattern: fix the perf regression together with a test that fails on the old behaviour.

---

## 4. How he organises work and people

### 4.1 Bounties (mechanics from PR comments)
- **Claim by merge.** PR #728 (2023-03-25), geohot: "Claim the bounty by implementing this and having tinygrad generate GEMM kernels for M1 that are faster than torch/Accelerate.framework. Clean code only, must be merged to claim bounty."
- **Locks.** "Locked the bounty for you", "bounty lock is yours", "I'll lock the bounty, but you need to do a lot of cleanup." (#962, #1200, #14664). A lock reserves the work while in progress.
- **Acceptance is a number.**
  - "stable diffusion < 324ms" (#2129; chenyu: "Sub 324 we'll merge and the bounty is yours!")
  - "< 75 ms/step on 7900XTX ... while keeping 90%" (#1200)
  - "within 10% of the LLVM based one" (assembler)
  - "76.9+ tok/s" (#9625)
  - "faster than torch/Accelerate" (#728)
- **Value is integration.** #728: "The value in the bounty for me is the integration into tinygrad."
- **Shared and raised.** #10777: "If you two work together I'll up the bounty to a shared $300 (so $150 each)". #8846: "I'll raise the bounty to $500 since the scope increased".
- **Payment.** #10777 (2025-06-24): "Merged, bounty is yours! $150 each, reach out to (public contact) to claim, can pay via paypal or USDC".
- **Hiring by PR.** Blog 2023-05-24: "There's only one way to get hired at the tiny corp, and that's by submitting high quality pull requests to tinygrad. Job interviews are obsolete, prove you can do the job by doing the job!" Blog 2025-12-29: "People get hired by contributing to the repo. It's a very self directed job, with one meeting a week and a goal of making tinygrad better."
- **Bounties may not survive AI.** HN 2025-12-31 (id=46444740): "because of AI coding agents it's possible the bounty system will go away. The value today is much less in writing the code and much more in defining the tasks and validating the code. We don't have the bandwidth to read tons of AI slop ... The fix for this is probably a reputation or staking system."

### 4.2 CI and merge rules
- README Contributing: "If your PR looks "complex", is a big diff, or adds lots of lines, it won't be reviewed or merged. Consider breaking it up into smaller PRs that are individually clear wins. ... If you can (cleanly) refactor to the point that the feature is a 3 line change, this is great."
- "Bug fixes (with a regression test) are great!" and "Finding bugs, even writing broken tests (that should pass) with `@unittest.expectedFailure` is great. This is how we make progress."
- "Dead code removal from core `tinygrad/` folder." Docs and whitespace changes are closed unless you are a well-known contributor.
- **Process replay.** Refactors and speedups put `[pr]` in the title, and CI asserts the generated kernels equal master's. On #13780 (2025-12-20) he asked: "Did you test that it preserves GPU speed? I'm not sure [pr] will catch if it doesn't." He knows the guard's limits and asks for a measurement.
- **Real-hardware benchmarks in CI.** `benchmark.yml` runs model benchmarks, HLB-CIFAR10 (10 steps and a full run to 93%) and MLPerf, with process replay after each.
- **AI policy (README, current).** QUOTE: "If you are a new contributor with something that looks even close to AI written, it will be closed without feedback and you may be banned from our GitHub. No human should waste time reading AI slop. And for everyone, if you used AI, disclose what you used it for."
  - Reviews say "NO AI!" (#17459, #17097) and "no AI" (#17610).
  - His own commits carry model tags: "GLM 5.3 flash, just the model (opus)", "(kimi)", "(codex)", "(gpt-6)", and PR #17090 "(codex slop)".

### 4.3 Team
- QUOTE (2025-12-29): "The company is 6 people now. ... The tiny corp is a deconstructed company. We have almost nothing private, it's a Discord and GitHub. To fund the operation, we have a computer sales division that makes about $2M revenue a year."
- Remote-first, one meeting a week, a few in-person weeks per year (HN 2025-12-29, id=46424562).
- QUOTE (2025-07-06): "The main thing to watch is development speed. Our bet has to be that tinygrad's development speed is outpacing the others."
- Vendor negotiation in public: the AMD contract was "negotiated mostly in public on Twitter" (2025-12-29). "AMD passed my cultural test" (2025-03-08).

### 4.4 Being stuck or wrong
- **Stuck on a design:** restructure (P3), or find the smaller formulation.
- **Stuck on a vendor:** bypass the layer. HN 2024-08-09: "We didn't just bypass all of ROCm, we bypassed HSA! ... Between our PM4 backend and disabling CWSR, our AMD GPUs are now pretty stable."
- **Stuck on trust:** replace opinion with a test ("Have you benchmarked it?").
- **Wrong:** say it and revert it (P13). On his own code, HN 2023-08-11: "lol, you should see me bash my own code. I'm even more mean."
- **On giving up in public**, Latent Space 00:28:11: "Giving up is this wonderful thing. It's so liberating. And then, like, you can decide afterward if you really give up or not." Later, HN 2025-05-26: "Like it's strange people think I give up on things ... This is a 2+ year long project that I've worked on almost every day."

---

## 5. Contrasts, controversies, failures, critiques

### 5.1 Where he differs from conventional practice

| Conventional | Hotz |
|---|---|
| Abstract and plan for future features. Lex 02:06:09: "Not often do people tell you, "Do the simplest possible thing."" | Delete until only the goal is left. "You have never refactored enough." |
| Lines of code are irrelevant. | Lines and tokens/line are visible in CI and gate review. |
| Profile, then optimise. | Same, but the physical bound comes first and "can this fuse?" is the default question. |
| Depend on vendor stacks. | Replace them. Repo AGENTS.md: "tinygrad has user space PCI drivers for AMD and NVIDIA GPUs. Do not insert the unneeded kernel modules." |
| Interview, then hire. | A PR plus a bounty is the interview. |
| Review as coaching. | Short and blunt: "NO AI!". Suspected-AI PRs from new contributors are closed without feedback. |
| Hand-tune per kernel. | Make the search space complete, then burn compute. |

### 5.2 Where it failed or was contested
1. **Line-count drift.**
   - Dec 2025: "When this is finished and cleaned up, it'll be about 20,000 lines. And that's completely it." (18,935 at the time).
   - By v0.14.0 (2026-08-24) the count was 26,040, and the CI cap is 26,500.
   - `sz.py` excludes `tinygrad/llm`, `nn`, `renderer`, `runtime` and `viz` from "core", so the 20k and 26k figures may not be the same metric. Even so, the cap went 1000, then 5000, then 8000, then 26,500.
2. **Code-golf critique.** HN 2025-12-29 (id=46422757):
   - "I've looked at the "only 18,935 lines of code" python code and it made me want to poke my eyes out. Not sure what's the point of this extreme code-golfing."
   - "whenever someone incessantly focuses on lines of code as a metric (in either direction), I immediately start to take them less seriously."
   - Counter-voices in the same thread call it "dense but perfectly readable". Hotz conceded golf risk in 2023 ("became a little code golfy at the end").
   - Balanced view: the order-of-magnitude claim (a stack roughly 1000x smaller than PyTorch/LLVM/CUDA) is real. The per-line density is a style choice many find hard to read.
3. **Performance vs claims.**
   - 2023: "about 5x slower than PyTorch" on NVIDIA, and "build it the right way and worry about performance later".
   - 2025-12-29: "it outperforms PyTorch on many workloads", with no public table in that post.
   - The AMD MI350X Llama-405B MLPerf contract outcome is not established in my sources. The v0.14.0 notes list Llama 3.1 8B training and GPT-OSS 20B in MLPerf training.
4. **Bound not reached.** OLMoE decode sat at 45% of bandwidth after months (S2). The 50% bar was not demonstrably cleared in-thread.
5. **AMD bets.**
   - Driver problems cost months in 2023. Then "AMD YOLO" (2025-03-08): "Bought in for a quarter million. Long term."
   - Critics in the 2023-05-24 HN thread (id=36065175) said the $15k tinybox was a steep markup over ~$6k of GPUs. One questioned whether his skills "translate to other domains".
6. **Twitter episode (2022-11).**
   - Critics (HN 2025-12-29) say he "talked a big game ... then utterly failed ... gave up".
   - His answers: Lex 02:12:18, the day he quit Elon "came on my Twitter spaces afterward"; HN 2025-07-06, "left because I realized nothing was going to change". His refactor plan was tests and process replay first, then replace.
   - My sources show nothing either way about the code outcome.
7. **AI stance reversal.**
   - 2025-09-12 "AI Coding": "The best model of a programming AI is a compiler." and "That anyone uses LLMs to code is a testament to just how bad tooling and languages are."
   - 2026-07-12: "I might have been a little harsh in The Eternal Sloptember about models not being able to program ... I'm now pretty confident I'm getting better at using them and get some boost from the models."
   - He says "NO AI!" on outside PRs while tagging his own AI-assisted commits. EthanBnntt's honest disclosure on #17610 was rejected. His defence is review bandwidth.
8. **Tuning by constant lacks hardware coverage.** #16072 could not validate a 62-105% bandwidth win on Metal across devices. This is a process limit of a small team with a CI that has one GPU per vendor.
9. **Bounty volatility.** A bounty PR was closed while still a draft (#14664: "why did you close it? It was still draft PR."). A 2026 bounty submission reported "Benchmark Results (Simulated)" (#17097), which is the failure mode the "must be benchmarked" rule targets.

### 5.3 Less controversial
- The per-kernel bytes/FLOPs printout and dividing by the hardware bound are widely emulated. Contributor PRs now report GB/s tables by default.
- The fusion argument (the A*B+C round trip) is mainstream compiler wisdom. His contribution is making the dispatched kernel list a first-class artifact.

---

## 6. Quantisation, MoE, LLM inference, GPUs
- **Quantisation is weight compression and must be validated.** Latent Space (00:30:37): "Quantization is a form of weight compression, right? ... it's a lossy compressor. And how do you know that your loss isn't actually losing the power of the model? ... int4 65B llama is actually the same as FB16 7B llama, right? We don't know." On GGML 4-bit then: "I read what it was, and I was like, this isn't from any paper. ... you made up some quantization standards to make it run fast. ... where's, like, the Hellaswag number?" PR #1289 applied this: an int8 PR needed a streaming quantizer and the Llama benchmark suite before the bounty.
- **Where the gain is** (Latent Space, ~00:31:58): "doing your math in [int8], the big gain that people care about is actually having your weights in [int8], because weights in [int8] mean less memory and less memory bandwidth, whereas the math, keep it in FB32 [fp32]." This is weight-only quantisation for bandwidth, i.e. the W4A16 idea. On training: "if you're training, no one's gotten training to work with [int8] yet".
- **MoE.** The only evidence is in PRs (S2, S3):
  - Count active params for the bound (1.3B active at fp16 is 2.6 GB).
  - Ensure only routed experts are read and dequantised.
  - `extra/gemm/moe_gemm.py` has a HipKittens grouped MXFP8 GEMM, and `moe_routing.py` plus many `wozeparrot` commits (2026-09/10: "gptoss: fused router topk", "dispatch gather", "fp8 dispatch") cover MoE training kernels.
- **GPUs and vendors.**
  - "CUDA isn't really the moat people think it is, it is just an early ecosystem." (2025-03-08)
  - "AMD even got the double throughput Tensor Cores on RDNA4 (NVIDIA artificially halves this on their cards, soon market pressure will force them not to)." (same post)
  - speed.md: "Newer NVIDIA GPUs have a "Tensor Memory Accelerator" to assist with fast indexing, however, this is not supported in tinygrad yet."
  - HN 2025-11-15: "For training, it's NVIDIA and Google in first. AMD in second. And nobody in third."
- **All-reduce arithmetic example** (HN 2024-08-27, id=41368554): "Say you are training a 3B parameter model in BF16. That's 6GB of weights, as long as your step time is >=500ms you won't see a slowdown." (about a 16 GB/s link).
- **Static execution.** His dislike of data-dependent control flow in the hot path is P9.

---

## 7. The Hotz playbook for kernel speed work (36 rules, each tied to a source)

Source keys are in section 8. H200 notes are mine.

### A. Before touching code: bounds and claims
1. **Write the bound first:** `time_floor = bytes_that_must_move / bandwidth` and `flops / peak`. Decode: `tok/s <= BW / (active weight bytes + KV bytes)`. [PR #9625; speed.md] H200: BW 4.8 TB/s. For W4A16 experts use ~0.5 B/param plus scales, not 2 B/param.
2. **Know the ridge point** and which side each op is on. A 4090 needs ~100 FLOP/B from HBM and ~4 from L1. [speed.md] H200: ~206 (BF16) and ~412 (FP8) FLOP/B vs HBM.
3. **State every result as % of the bound, with a unit.** "Gb/s or GB/s? What should it be in theory?" [PR #3000]
4. **Set targets as fractions of the bound.** He asked 76.9 of 153.8. [PR #9625]
5. **If a number beats physics, the benchmark is broken.** Convert to bytes/s and compare to the memory system. [PR #13379]
6. **No speedup without a benchmark and a repro command.** Simulated numbers are rejected. [README; #17097]
7. **Ask "how much slower is the slow case, and why?"** An unexplained regression is a bug ("bug levels of worse"). [#8759, #3000]

### B. Find wasted work
8. **Print per-kernel FLOPs, bytes and GB/s every run** (DEBUG=2 equivalent) and sort by time. Attach analytic ops/mem estimates to custom kernels. [Lex 00:50:35; moe_gemm.py `Estimates`]
9. **Track bytes-per-token as a counter** and compare it to `active_params*bits/8`. 83,525 MB/token vs 3,412 MB was the whole MoE bug. [PR #17610]
10. **For MoE, prove only routed experts are read and dequantised** (no "multiplying by 0 instead of not computing"). Check both the dequant and the GEMM inputs. [#17610, #9625 body]
11. **Every intermediate written to HBM and read back is a suspect.** Fuse unless recompute costs more than store+load. Count reads and writes for both options (6R+2W vs 6R+3W). [Latent Space; speed.md]
12. **Count unique compiled kernels.** Same-shape layers must share one binary. Offsets or constants leaking into the kernel signature are a red flag. [PR #15052]
13. **Look off the beaten path:** the real M (1..16 tokens per expert), odd K and N, tail tiles, non-power-of-two heads. "GroupNorm is 2x slower than BatchNorm ... no fundamental reason." [HN 33468051]
14. **Separate launch/Python/dispatch time from GPU time** and replay a prebuilt queue (CUDA Graph / TinyJit analogue) for the fixed decode step. [PR #812; speed.md]
15. **When a gain is bigger than your explanation, find the true cause** (swap vs bandwidth). [PR #1289]
16. **When a 24x traffic cut gives 1.38x speed, recompute the bound with the new byte count and re-profile.** [INFERENCE from #17610]

### C. Make and test the change
17. **Reference-check every kernel before timing** against a trusted implementation, forward and backward where applicable (test_ops style: ~100 small cases vs torch). [Latent Space 00:26:34]
18. **If the diff touches many places, stop and refactor first.** The fix should then be tiny ("This should be 5 lines"). [PR #1141]
19. **Prefer changes that delete code and are faster** ("simpler order = more speed", -16 lines). [PR #9982]
20. **Revert fast when CI says slower.** [PR #7434] Small PRs make the revert one click.
21. **Pin a speed regression with a test** that fails on the old behaviour (tok/s floor, kernel count, bytes counter). [commits 18327, 18466]
22. **Refactors must be provably neutral:** diff generated kernels between commits and also measure speed. [process_replay README; PR #13780]
23. **Measure properly:** discard warm-up, repeat, report median or best, remember clocks ramp (first iteration 8.87 s then 0.35 s in #3000; a second run was slower than the first). [PR #3000]
24. **Record what you tried and rejected.** His own PR titles include "not worth the speed"; he noted "several hours" for a 10%-slower AllNorm. [PR #1584; HN 33471649]
25. **Tune constants per device and keep them env-overridable.** "Tuning this stuff is really annoying." [PR #16072; #2129 env-gated threshold]

### D. Search vs hand-written
26. **Make the search space complete and the objective "runs faster"; then burn compute.** Put hand-coded structure inside the search space. [blog 2025-07-06; PR #2137]
27. **Specify what search cannot find** (LDS layouts and swizzles) and let search find the schedule (L2 order, tile sizes). [HN 45938961]
28. **Hand-write only where format structure is too specific** (per-quant decode kernels). Keep it in the same IR and profiler, with a unit test per format. [commit 0c0fadb51]
29. **Do not add a feature (warp specialisation) just because another library has it.** Add it when it fits the abstractions and the hardware needs it. [HN 45938961]
30. **Write the kernel as spec plus schedule** ("specify what it does instead of how"), so the "how" can be re-searched when shapes or hardware change. [HN 45236404]

### E. Quantisation and numerics
31. **Treat W4/W8 as lossy compression; validate with a real eval** before calling a speed win a win. [Latent Space; PR #1289]
32. **Weights low-bit for bandwidth, math in higher precision where cheap** (the W4A16 pattern). [Latent Space ~00:31:58]
33. **Stream quantisation and dequantisation; avoid peak-memory spikes.** [PR #1289]

### F. Process and mindset
34. **Be suspicious of every layer you cannot read.** Reproduce, then bypass or own the lowest layer you depend on. [HN 44094090, 41203887]
35. **Say "I was wrong" in the PR and move on.** [PR #12822; blog 2021-12-12]
36. **Use AI as a tool, verified by reference and measurement.** Lex (01:21:03): "I spent 5% of time typing and 95% of time debugging. The last thing I want is close to correct code." AI patches must carry the same benchmark and bytes-moved evidence, and be labelled as AI-assisted. [Lex; README; commit log]

### G. Checklist for the H200 kernels (mine, derived from the above)
- **MoE W4A16 GEMM:** bytes = routed-expert weights (4-bit plus scales) + activations. Floor = bytes / 4.8 TB/s. Assert dequant touches only routed experts. Test M = 1, 2, 4, 8, 16 per expert. Count kernels and launches.
- **FP8 attention decode:** bytes = KV cache read (+ scales). Floor = KV bytes / 4.8 TB/s. Any extra pass over KV (separate dequant, separate softmax stats) is waste to delete. Split over KV for occupancy, but verify the reduction cost.
- **BF16 projections:** arithmetic intensity is roughly M FLOP/B, so they are bandwidth bound for M below ~200 and compute bound above. Report both % of 4.8 TB/s and % of 989 TFLOPS.
- **Whole decoder layer:** sum the per-kernel bounds. The gap between that sum and the measured layer is launch gaps, non-overlapped copies and fusion misses. Replay with graphs. Look for intermediates (RMSNorm output, rope, residual) that can fuse into neighbours.
- **Always:** reference-check first, benchmark second, one change per commit, revert on slower, keep the profile output in the PR.

---

## 8. Source index (raw files under `<workspace>`)

| Key | Source | Date | Raw file |
|---|---|---|---|
| Blog | https://geohot.github.io/blog/ (40 posts fetched) | various | `blog/*.txt`, `feed.xml` |
| Lex | Lex Fridman #387 transcript https://lexfridman.com/george-hotz-3-transcript | 2023-06-29 | `podcasts/lex387_hotz3.txt` |
| Latent Space | https://www.latent.space/p/geohot "Commoditizing the Petaflop" | 2023-06 | `podcasts/latentspace_geohot_2023.txt` |
| HN comments | all 111 comments by `georgehotz` (Algolia API) | 2020-12 to 2026-07 | `hn/georgehotz_hn_comments.txt` |
| HN threads | Five Years of Tinygrad (46422757); next Linux and LLVM (41622588); Tinygrad: simple and powerful NN framework (33462337); tiny corp raised $5.1M (36065175); Documentation for the AMD 7900XTX (39888020); HipKittens (45934212) | 2022-2025 | `hn/thread_*.txt` |
| Repo docs | README (also v0.8.0, v0.9.0), `docs/developer/speed.md`, `developer.md`, `env_vars.md`, viz README, process_replay README, AGENTS.md | repo @ d16046f, 2026-10-24 | `tinygrad_repo/` |
| Releases | v0.8.0 to v0.14.0 | 2024-01 to 2026-08 | `github/release_*.md` |
| PRs | #728, #812, #987, #1075 (HTML-parsed), #962, #1141, #1200, #1289, #1584, #1053, #2129, #2682, #3000, #4232, #4887, #5976, #7434, #8759, #8846, #9625, #9982, #10777, #12822, #13379, #13780, #14664, #15052, #16072, #17090, #17097, #17459, #17610, #17720 | 2023-2026 | `github/pr_*.txt`, `github/prhtml_*.txt` |
| Phoronix | https://www.phoronix.com/news/Tiny-Sovereign-Stack-AMD-Close | 2025-01-15 | `other/phoronix_sovereign.txt` |
| Repo clone | `scratch/hotz/tinygrad/repo` for commit messages (`tinygrad/llm/kernels/amd.py`, `extra/gemm/moe_gemm.py`, `sz.py`, CI workflows) | 2026-10-24 | not copied |

Reliability notes:
- Lex and Latent Space text is machine-transcribed ("indate" is int8, "SISC" is CISC, "FB-16" is FP16). I corrected only inside [brackets].
- Non-Hotz GitHub authors (EthanBnntt, geohotstan, kimjune01, chaosagent, Qazalin and others) are labelled as theirs.
- Dates for #1141, #812 and #987 are PR creation dates. Individual comment dates were lost in my HTML extraction.
- The 458 GB/s quote is exact. The bandwidth arithmetic around it is my inference.
- No private channels, credentials or paid services were used.

---

## Summary reply
**Sources used:**
- Hotz's blog (40 posts).
- Lex Fridman #387 and the Latent Space interview.
- All 111 HN comments by `georgehotz`, plus 6 HN threads.
- 34 GitHub PR threads where he reviews or authors.
- The tinygrad README, speed.md, CI workflows, release notes and repo commit log.
- The Phoronix article quoting a tinygrad X post.

**Where the files are:** the raw texts are in `<workspace>`. The ANALYSIS.md write was refused by the tool, so the complete analysis is the text above. Save it as `philosophy/ANALYSIS.md` if you want the file.

**The five most important findings:**
1. **Bound first, then fraction.** His review of the OLMoE MoE PR is the template: 1.3B active params gives 2.6 GB of weights, so 400/2.6 = 153.8 tok/s is the limit, and he set the bounty bar at 76.9 (50%). He also rejected a CI number (458 tok/s on a 1B model on CPU) as physically impossible ("clearly isn't running the model").
2. **The costliest MoE waste found in a related PR** was dequantising or reading all experts instead of the routed ones: 83,525 vs 3,412 MB/token. That cut traffic 24.5x but decode only 1.38x, so re-derive the bound after each fix. Hotz rejected that PR on AI-policy grounds, not technical ones.
3. **Visibility and instrumentation:** `DEBUG=2` prints FLOPs, bytes and GB/s per kernel. Hand-written kernels declare ops/mem estimates so they show utilisation too. Python/launch time is separated from GPU time and replayed with a prebuilt queue (S6: LLaMA 450 ms to 106 ms).
4. **Simplicity is enforced, with refactor first.** "This should be 5 lines"; "when you are struggling and needing to change lots of things, there's a conceptual issue." Speed PRs need benchmarks. Refactors are checked by kernel diff (process replay) plus a CI line/token budget. Unexplained regressions are bugs. He reverts quickly and says "I was wrong".
5. **Search plus structure, not dogma.** BEAM over a complete search space is the stated ideal. He specifies LDS layouts by hand and lets search find the schedule. By 2026 tinygrad also ships hand-written per-quant LLM and MoE kernels inside its own IR and profiler.

Critiques are in section 5. The main ones: the line-count promise drifted (20k promised, 26k in 2026, CI cap 26.5k), the code-golf complaint from HN, weak public evidence for "outperforms PyTorch", OLMoE decode left at 45% of its bound, AMD driver bets that cost months, and a "NO AI" gate on outside PRs while his own commits are AI-tagged.