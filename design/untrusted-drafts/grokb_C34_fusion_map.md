# UNTRUSTED (Grok Build, 2026-10-26): C34 fusion map for sonnet_v5f; verify before building.

# C34 fusion map — `sonnet_v5f` (job efc7af96)

R5: fuse a kernel **< 5 µs at T=1** that only r/w `[T,8192]`. R4: a producer that already holds `x` writes the consumer’s normalized copy. `nvjet_*` replace-only. No cache, no work outside `kernel_fn`. **S2 fused routing = failed class** (do not re-fuse routing into the router GEMV).

Spine T=1: `nvjet qkv`(+`splitKreduce`) → `_attn` → `_reduce` → `nvjet o` → **attn_bnd** → {`_gemv` pooler ∥ `_router_fix` → **routing**} → **pre_norm** → `smallm×2` → **epi** → `_unpool` → **ffn_bnd**. T=16: same with `_gemv` router, `nvjet` unpool, `_full_cat` on full.

| glue | T=1 / T=16 µs (full) | producer → consumer | fusion target | bytes saved | µs saved est. | numerics vs TASK.md | risk |
|---|---|---|---|---|---|---|---|
| attn_boundary `norm_residual` | 4.4 / 3.7 (4.1 / 3.8) | `nvjet o` `a` → pooler+router `h` (FP32 router_in T≤8); r/w `s` | **pooler+router prologue** (keep nvjet). Alt: **replace** nvjet o, boundary in **epilogue** | 32–48 KB / 0.5–0.8 MB (`a` round-trip; drop `h` store) | **3.5 / 2.7** (4.4−~1 dup math). Replace-o: 4–6 iff Triton o ≤9 µs | Reload BF16 `a`: **same** rounds. Replace-o: **remove 1 BF16 round of `a`** (FP32 acc) | Dup rmsnorm in two grids; one CTA stores `s`. Replace-o: Triton vs nvjet 7.0 |
| ffn_boundary `norm_residual` | 3.4 / 3.5 (3.5 / 3.6) | `_unpool` / `nvjet unpool` `f` → `(h_next, s)` | **unpool epilogue** T≤8 (FP32 `tot` in last-CTA). T=16: leave (nvjet) | 32 KB / 0 | **3.2 / 0** | **remove 1 BF16 round of `f`**; `rmsnorm(f*(1+w))` extra FP32; `h_next` still 1 BF16 store | Two row-reductions (f then s); pad cols of `f` are 0; T=16 nvjet cannot absorb |
| `_epilogue_kernel` | 4.5 / 4.9 (4.5 / 4.7) | `smallm` GEMM2 `y[T,8,4096]` → unpool | same-grid **unpool prologue** (materialize `y_sum`, ticket, then GEMV). T=16: leave or GEMM2 last-CTA | 16 KB / 256 KB of `y_sum` if L2-hot only | **1.5–2.0 / ~1.5** (math moves; GEMV needs full `y`) | **remove 1 BF16 round of weighted expert sum** only if GEMV reads FP32 `y_sum` | Not register-resident; G4 (tokens,4) layout; unpool smem 48 KB |
| `_pre_norm_kernel` (G2) | hidden ~2 / ~3 | pooler SPLIT=1 → `smallm` GEMM1 | **pooler epilogue** (atomic sumsq + scale `acc`) | 16 KB / 256 KB | **0 wall T=1** (behind router 7 µs) / **1.5–2.5 T=16** | **remove 1 BF16 round of pooled**; still 1 BF16 of normed x (G2) | T=1 wall 0; **do not grow pooler smem** (ps1 class) |
| `routing_kernel` | 2.7 / 2.7 | router_fix / `_gemv` → `smallm_route` | `smallm_route` **prologue** (logits→ids). **Not** router GEMV | ~1–2 KB logits | **0 wall T=1** (side 9.7 < pooler 19) / ~0–1 T=16 | top-8 ties (0.05 slack); FP32 sigmoid+bias | **S2 failed class**; skip |
| `_reduce_kernel` | hid~2.5 / 2.7 (2.9 / 3.8) | `_attn`/`_full_cat` partials → `nvjet o` | attn **epilogue** only if nsplit=1; else last-CTA merge in attn grid | 4 KB / 64 KB `[T,1024]` | **1.5–2.5 / 1.5** | split-merge order; 1 BF16 of `rmsnorm(O)*sigmoid(g)` kept | nsplit>1 on sliding 2K; ROWMAJOR full_cat; o is nvjet |
| `cublasLt::splitKreduce` | hid~1.8 / 1.8 | `nvjet qkv` splitK → attn | **replace** qkv with non-splitK nvjet or Triton GEMV | partials ~10–40 KB | **~1.5 / 1.5** iff GEMM not slower | qkv acc order vs split reduce; 1 BF16 of qkv kept | nvjet splitK is 12.5; non-split may lose |

R5 hits: **attn_boundary, ffn_boundary**. Others fail the `[T,8192]`-only test or sit off the wall (side stream).

## Top 3 by µs saved / builder-hour

### 1. Unpool epilogue ← ffn_boundary (~3.2 µs / ~4 h ≈ 0.8 µs/h)
1. Gate T≤8 (`_unpool_live`); T=16 keep `ffn_boundary`.
2. Last-CTA already holds FP32 `tot` = live 1024 of `f` (pad cols 0).
3. Local `sumsq += (tot*(1+w_post))^2`; `atomicAdd` per token; second ticket `== NB`.
4. `inv_f = rsqrt(sumsq/8192+eps)`; `s_new = α2*s + β2*tot*(1+w)*inv_f` (pad: `α2*s`).
5. Second `atomicAdd` of `s_new^2`; `h = s_new * rsqrt(mean+eps) * (1+w_next)`.
6. Store `h` BF16 and `s` FP32; **skip store of `f`**.
7. Host: `unpool_sliced` returns `(h_next,s)`; drop `state["ffn_boundary"]`.
8. Scratch: one `float[T]` sumsq + one `int32` ticket in `prepare()` (replay-zeroed).
9. Vector: vs v5f, extra FP32 on `f` only; 2060 13/13 + `okc check`.
10. Kill if Δ<1.5 µs (R5) or residual L2 fails.

### 2. Pooler+router prologue ← attn_boundary (~3.5 µs T=1 + 2.7 T=16 / ~8 h ≈ 0.4–0.8 µs/h)
1. Keep `nvjet o` (7.0 / ~14 µs). Do **not** replace it in v1.
2. Each pooler CTA already streams `h[T,8192]` as K. Replace that load with `a` BF16 + `s` FP32.
3. Prologue (both grids): `an = rmsnorm(a)`; `s_new = α1*s + β1*an`; `h = rmsnorm(s_new)*(1+w_pre)`.
4. Pooler GEMVs with BF16 `h`. Router T≤8 keeps `h` FP32 (`SPLIT_X`); T=16 uses BF16 `h`.
5. CTA 0 of pooler stores `s_new` (once). No `h` / router_in store.
6. Launch pooler∥router **after o-proj**, skip `attn_boundary`.
7. Same math as TASK (`a` is still the BF16 o-proj output).
8. Do **not** grow pooler smem (S7/ps1). Prologue is registers + the K loads.
9. Self-check: `h`/logits vs v5f within 1 BF16 ULP on `h`; routing slack unchanged.
10. Kill if pooler+router wall > old pair+boundary−1.5 µs, or T≤8 FP32 router path drifts.

### 3. Pooler epilogue ← pre_norm (~1.5–2.5 µs T=16, ~0 T=1 / ~3 h; bundle with #2)
1. Pooler is SPLIT=1: `acc` is the pooled row-slice in FP32.
2. `atomicAdd` of `acc^2` per token; ticket `== cdiv(4096,BN)`; `inv = rsqrt(mean+eps)`.
3. Store `acc * inv * w_pre` BF16; drop `_pre_norm_kernel` / `S9_NORM_EARLY`.
4. GEMM1 reads that tensor (same as G2). **Removes** the BF16 round of pooled.
5. T=1: wall 0 (already hidden); still drops a launch on every shape.
6. T=16: main = pooler+norm vs side = router+routing; this shortens main ~2 µs.
7. Same smem as today (epilogue after the K loop).
8. G2 already rejected folding into GEMM1 (3 CUDA paths × cross-CTA sumsq) — pooler is the one neighbour.
9. Vector V3: max 1 BF16 step vs `MuseSpark1xRMSNorm`.
10. Kill if T=16 Δ<1.5 µs and T=1 unchanged, or smem grows.

Do #2 and #3 in **one** pooler edit (R4 one-motion). Leave routing, reduce, splitKreduce, and epi-into-unpool off the first build card.
