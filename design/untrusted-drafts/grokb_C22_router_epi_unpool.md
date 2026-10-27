# Grok Build draft (UNTRUSTED, read-only lane, 2026-10-25) — C22: router last-CTA fixup + RPAR, epilogue grid (tokens,4), unpooler T<=8

I'll start by reading the design notes and the two kernel files so the drafts match the existing GEMV, router, and epilogue call sites.The decode-layer file is long; next I’ll pull the epilogue, unpooler, and `prepare()`/`kernel_fn` call sites the drafts have to match.I have the call sites. Next I’ll pull the original G3/G4/H13 notes and the unpooler/`prepare()` host path so the drafts stay drop-in.Paste-ready drafts for H13/RPAR, H9, and H8. They follow `_gemv_splitk` / `_CFG` in `director_v1b.py`, the last-CTA pattern in `_router_f2_kernel`, and the epilogue / unpooler call sites in `sonnet_v4.py`.

---

## 1. H13 / RPAR — last-CTA split-K router GEMV

**CTA counts at N = 256** (grid = `(cdiv(256, BN), SPLIT, cdiv(m, BM))`, BM = 16 for m ≤ 16 else 32):

| bucket | `_CFG` `(BN, BK, SPLIT, warps, stages)` | CTAs |
|---|---|---|
| m ≤ 8 | `(64, 128, 64, 4, 4)` | 4 × 64 × 1 = **256** |
| m ≤ 16 | `(64, 64, 64, 4, 4)` | **256** |
| m ≤ 32 | `(16, 128, 64, 4, 3)` | 16 × 64 × 1 = **1024** |

K = 8192, SPLIT = 64, BK ∈ {64, 128} → each split owns a disjoint 128-K strip (`k_per = 128`). HAS_LO stays inside the K loop (hi then lo `tl.dot`, FP32 acc). Fixup sums those FP32 partials in slice order `0 … SPLIT-1` and stores BF16/FP32. The output is a plain store; the only atomic is the per-tile int32 arrival counter.

### 1a. `director_v1b.py` — replace `_gemv_splitk`, `_CFG`, `prepare`, `_run`

```python
@triton.jit
def _gemv_splitk(x_ptr, w_ptr, wlo_ptr, out_ptr, part_ptr, done_ptr,
                 M, N, K, stride_xm, stride_wn, stride_om, part_m,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                 HAS_LO: tl.constexpr, FIXUP: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k_per = tl.cdiv(K, SPLIT * BK) * BK
    k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        rk = k0 + kk + tl.arange(0, BK)
        xm = (rm[:, None] < M) & (rk[None, :] < K)
        wm = (rn[:, None] < N) & (rk[None, :] < K)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=xm, other=0.0)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :], mask=wm, other=0.0)
        acc += tl.dot(x, tl.trans(w))
        if HAS_LO:
            wl = tl.load(wlo_ptr + rn[:, None] * stride_wn + rk[None, :], mask=wm, other=0.0)
            acc += tl.dot(x, tl.trans(wl))
    om = (rm[:, None] < M) & (rn[None, :] < N)
    optr = out_ptr + rm[:, None] * stride_om + rn[None, :]
    if FIXUP:
        # Publish this slice, then one GPU-scope ticket per (pid_m, pid_n) tile.
        tl.store(part_ptr + (pid_k * part_m + rm)[:, None] * N + rn[None, :], acc, mask=om)
        tl.debug_barrier()
        nt = tl.cdiv(N, BN)
        ticket = tl.atomic_add(done_ptr + pid_m * nt + pid_n, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT - 1:
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for s in range(SPLIT):  # fixed slice order; .cg skips a stale L1 line of `part`
                acc += tl.load(part_ptr + (s * part_m + rm)[:, None] * N + rn[None, :],
                               mask=om, other=0.0, cache_modifier=".cg")
            tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=om)
            tl.store(done_ptr + pid_m * nt + pid_n, 0)  # replay-ready
    else:
        tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=om)


# (BN, BK, SPLIT, warps, stages). RPAR: >= 256 CTAs at m <= 32.
_CFG = {
    ("router", 1): (64, 128, 64, 4, 4),    # 256 CTAs
    ("router", 8): (64, 128, 64, 4, 4),    # 256
    ("router", 16): (64, 64, 64, 4, 4),    # 256 (BK=64 kept from C9)
    ("router", 32): (16, 128, 64, 4, 3),   # 1024 (D4 alt: (64, 128, 64, 4, 3) = 256)
    ("router", 128): (32, 256, 8, 4, 3),
    ("router", 256): (32, 128, 8, 4, 3),
    ("pooler", 1): (32, 512, 1, 4, 4),
    ("pooler", 8): (32, 512, 1, 4, 4),
    ("pooler", 16): (32, 512, 1, 4, 4),
    ("pooler", 32): (32, 512, 1, 4, 3),
    ("o", 1): (32, 128, 1, 4, 4),
    ("o", 8): (32, 128, 1, 4, 4),
    ("o", 16): (32, 128, 1, 4, 4),
    ("o", 32): (64, 128, 1, 4, 4),
}
_ROUTER_TRITON_MAX_M = 256
_ROUTER_M_MAX = 256
_ROUTER_SPLIT_MAX = 64
_ROUTER_N = 256
_DONE_N = (triton.cdiv(_ROUTER_M_MAX, 16) * triton.cdiv(_ROUTER_N, 16))  # 16 x 16


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    dev = hi.device
    # last-CTA scratch: stable pointers, capture-safe; last CTA leaves `done` at 0
    state["router_part"] = torch.empty(
        (_ROUTER_SPLIT_MAX, _ROUTER_M_MAX, _ROUTER_N), dtype=torch.float32, device=dev)
    state["router_done"] = torch.zeros(_DONE_N, dtype=torch.int32, device=dev)
    return state


def _run(x, w, wlo, n, out_dtype, cfg, part, done, part_m):
    m, k = x.shape
    BN, BK, SPLIT, warps, stages = cfg
    BM = 16 if m <= 16 else (32 if m <= 32 else 64)
    fixup = SPLIT > 1
    # empty: last CTA writes every valid element of its tile; SPLIT==1 stores directly
    acc = torch.empty((m, n), device=x.device, dtype=out_dtype)
    n_tiles = triton.cdiv(m, BM) * triton.cdiv(n, BN)
    assert n_tiles <= done.numel() and SPLIT <= part.shape[0] and m <= part_m
    grid = (triton.cdiv(n, BN), SPLIT, triton.cdiv(m, BM))
    _gemv_splitk[grid](
        x, w, wlo if wlo is not None else w, acc, part, done,
        m, n, k, x.stride(0), w.stride(0), acc.stride(0), part_m,
        BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, HAS_LO=wlo is not None, FIXUP=fixup,
        num_warps=warps, num_stages=stages)
    return acc


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    b = _bucket(m)
    use_triton = _triton_ok(x.device)
    part, done, part_m = state["router_part"], state["router_done"], state["router_part"].shape[1]
    if layer == "router":
        if use_triton and m <= _ROUTER_TRITON_MAX_M:
            return _run(x, state["router_hi"], state["router_lo"], n, torch.float32,
                        _CFG[("router", b)], part, done, part_m)
        if m <= 8:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if use_triton and (layer, b) in _CFG:
        return _run(x, state[layer], None, n, torch.bfloat16, _CFG[(layer, b)], part, done, part_m)
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
```

### 1b. `sonnet_v4.py` — same fixup, keep `SPLIT_X` + `PDL`

Replace the body of `_gemv_splitk` after the K loop (signature gains `part_ptr, done_ptr, part_m`; `ATOMIC` becomes `FIXUP`). Keep the existing HAS_LO / SPLIT_X K-loop exactly.

```python
@triton.jit
def _gemv_splitk(x_ptr, w_ptr, wlo_ptr, out_ptr, part_ptr, done_ptr,
                 M, N, K, stride_xm, stride_wn, stride_om, part_m,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                 HAS_LO: tl.constexpr, FIXUP: tl.constexpr, SPLIT_X: tl.constexpr,
                 PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k_per = tl.cdiv(K, SPLIT * BK) * BK
    k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        rk = k0 + kk + tl.arange(0, BK)
        xm = (rm[:, None] < M) & (rk[None, :] < K)
        wm = (rn[:, None] < N) & (rk[None, :] < K)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=xm, other=0.0)
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :], mask=wm, other=0.0)
        if SPLIT_X:
            xh = x.to(w_ptr.dtype.element_ty)
            xl = (x - xh.to(tl.float32)).to(w_ptr.dtype.element_ty)
            acc += tl.dot(xh, tl.trans(w))
            acc += tl.dot(xl, tl.trans(w))
            wl = tl.load(wlo_ptr + rn[:, None] * stride_wn + rk[None, :], mask=wm, other=0.0)
            acc += tl.dot(xh, tl.trans(wl))
        else:
            acc += tl.dot(x, tl.trans(w))
            if HAS_LO:
                wl = tl.load(wlo_ptr + rn[:, None] * stride_wn + rk[None, :], mask=wm, other=0.0)
                acc += tl.dot(x, tl.trans(wl))
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    om = (rm[:, None] < M) & (rn[None, :] < N)
    optr = out_ptr + rm[:, None] * stride_om + rn[None, :]
    if FIXUP:
        tl.store(part_ptr + (pid_k * part_m + rm)[:, None] * N + rn[None, :], acc, mask=om)
        tl.debug_barrier()
        nt = tl.cdiv(N, BN)
        ticket = tl.atomic_add(done_ptr + pid_m * nt + pid_n, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT - 1:
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for s in range(SPLIT):
                acc += tl.load(part_ptr + (s * part_m + rm)[:, None] * N + rn[None, :],
                               mask=om, other=0.0, cache_modifier=".cg")
            tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=om)
            tl.store(done_ptr + pid_m * nt + pid_n, 0)
    else:
        tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=om)
```

`_CFG` router rows: same four RPAR tuples as above. `_run` / `_split_router_weights` / `_router_logits`:

```python
def _run(x, w, wlo, n, out_dtype, cfg, split_x=False, part=None, done=None, part_m=1):
    m, k = x.shape
    BN, BK, SPLIT, warps, stages = cfg
    BM = 16 if m <= 16 else (32 if m <= 32 else 64)
    fixup = SPLIT > 1
    acc = torch.empty((m, n), device=x.device, dtype=out_dtype)
    grid = (triton.cdiv(n, BN), SPLIT, triton.cdiv(m, BM))
    _gemv_splitk[grid](
        x, w, wlo if wlo is not None else w, acc, part, done,
        m, n, k, x.stride(0), w.stride(0), acc.stride(0), part_m,
        BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, HAS_LO=wlo is not None, FIXUP=fixup, SPLIT_X=split_x,
        num_warps=warps, num_stages=stages, **_pdl_kwargs(x.device))
    return acc


def _split_router_weights(router, bias):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    out = {"router": router.contiguous(), "router_hi": hi.contiguous(), "router_lo": lo.contiguous(),
           "router_hi_t": hi.t().contiguous(), "router_lo_t": lo.t().contiguous(),
           "router_bias": bias.float().contiguous(),
           "f2_done": torch.zeros(triton.cdiv(F2_MAX_TOKENS, F2_TOKEN_BLOCK), dtype=torch.int32, device=router.device),
           "router_part": torch.empty((64, 256, 256), dtype=torch.float32, device=router.device),
           "router_done": torch.zeros(16 * 16, dtype=torch.int32, device=router.device)}
    # ... existing sm_75 hi16/lo16 stand-in ...
    return out


def _router_logits(x, router_input, state, path):
    if path == "triton":
        cfg = _CFG[("router", _bucket(x.shape[0]))]
        part, done, part_m = state["router_part"], state["router_done"], state["router_part"].shape[1]
        src = router_input.contiguous() if router_input is not None else x.contiguous()
        return _run(src, state["router_hi"], state["router_lo"], N_EXPERTS, torch.float32, cfg,
                    split_x=router_input is not None, part=part, done=done, part_m=part_m)
    # ... existing fp32 / split paths unchanged ...
```

Pooler `_run(..., part=state["router_part"], done=state["router_done"], part_m=...)` is fine: pooler SPLIT = 1 so `FIXUP` is false and those pointers are unused.

### 1c. Test sketch

```python
def _ref_hi_lo(x, hi, lo):
    return (x @ hi.T).float() + (x @ lo.T).float()  # router 0.01 % bound

def test_h13_router_fixup(state, x):  # x: [m, 8192] bf16, m in {1,8,16,32,33,128,256}
    hi, lo = state["router_hi"], state["router_lo"]
    y = _run(x, hi, lo, 256, torch.float32, _CFG[("router", _bucket(x.shape[0]))],
             state["router_part"], state["router_done"], state["router_part"].shape[1])
    ref = _ref_hi_lo(x, hi, lo)
    rel = (y - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < 1e-4, rel  # 0.01 % class

    y2 = _run(x, hi, lo, 256, torch.float32, _CFG[("router", _bucket(x.shape[0]))],
              state["router_part"], state["router_done"], state["router_part"].shape[1])
    assert torch.equal(y, y2)  # counter reset: second launch, no host zero
    assert int(state["router_done"].abs().max()) == 0

    # ordered-slice twin (same k_per tiling) is bitwise with the kernel
    BN, BK, SPLIT, _, _ = _CFG[("router", _bucket(x.shape[0]))]
    k_per = ((8192 + SPLIT * BK - 1) // (SPLIT * BK)) * BK
    acc = torch.zeros_like(ref)
    for s in range(SPLIT):
        sl = slice(s * k_per, min((s + 1) * k_per, 8192))
        acc += x[:, sl].float() @ hi[:, sl].float().T
        acc += x[:, sl].float() @ lo[:, sl].float().T
    # tl.dot vs this torch sum: tolerance, not bitwise; both beats atomics on output
    assert (y - acc).abs().max() < 1e-2

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g.capture_begin()
        yg = _run(x, hi, lo, 256, torch.float32, _CFG[("router", _bucket(x.shape[0]))],
                  state["router_part"], state["router_done"], state["router_part"].shape[1])
        g.capture_end()
    torch.cuda.current_stream().wait_stream(s)
    g.replay(); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    assert int(state["router_done"].abs().max()) == 0
```

HAS_LO exactness: the K-loop still does `dot(x, hi)` then `dot(x, lo)` per BK tile, then the fixup only adds already-finished FP32 partials. SPLIT_X (layer T ≤ 8) keeps `xh.hi + xl.hi + xh.lo`.

### 1d. Risks

- **Graph capture.** `router_part` / `router_done` live in `prepare()` with stable pointers. `torch.empty` for the returned logits is capture-legal on the judge’s caching allocator; for the layer, prefer writing into a prepare()-allocated `router_out[256, 256]` view so the graph holds one pointer. Capturing a launch while `done` is nonzero (interrupted warmup) makes the next replay wait forever for `ticket == SPLIT-1`. Zero `done` in `prepare()`, and after every warmup print `done.max()`.
- **Counter reset under replay.** Last CTA stores `0` after the ordered sum. A second `kernel_fn` with no host `zero_` must be bitwise equal to the first (test above). If any CTA of a tile is skipped, the last ticket never hits and the next call hangs — grid is shape-only, so this only happens if SPLIT/BN at launch disagrees with the buffer math.
- **Scalar `atomic_add`.** Matches `_router_f2_kernel` (0-d pointer, `acq_rel` / `gpu`). Triton lowers that to one thread per CTA. If a future compiler emits a per-thread RMW, tickets explode and the last-CTA branch never fires. Guard: after one launch, `done` is all-zero and every output row is finite.
- **Visibility.** CTA `debug_barrier` + GPU-scope release atomic + `.cg` loads, same as F2. Missing a fence shows up as a 1-in-N garbage tile, not as a hang.
- **Extra traffic.** Partials are `SPLIT × m × 256 × 4` B (0.5 MB at m=8, SPLIT=64). That is the cost of dropping the 1 KB `FillFunctor` that currently stretches to 3.4 µs beside the pooler.
- **Bitwise vs today’s atomic path.** Ordered sum ≠ `atomic_add` into a zeroed buffer. Compare to hi+lo FP32 (0.01 %), and to a second launch of this kernel (bitwise).

---

## 2. H9 — epilogue grid `(tokens, 4)`

Each CTA recomputes `sumsq` over the full H = 4096 row (same `BLOCK_H` / `NCHUNK` / FP32 order as `_epilogue_kernel` today, so V4 is bitwise), then writes its 1024-column quarter. `BLOCK_H` still comes from `_marlin_knobs` (1024, or 2048 at M ≤ 8 under S1).

### 2a. Kernel + launch (replace `_epilogue_kernel` / `epilogue_triton`)

```python
@triton.jit
def _epilogue_kernel(
    y_ptr, w_ptr, s_ptr, ids_ptr, out_ptr, eps,
    TOPK: tl.constexpr, KP: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr,
    NPART: tl.constexpr, PDL: tl.constexpr = False,
):
    """out[t] = sum_k live_k * s[t,k] * rmsnorm(y[t,k] * w). Grid (tokens, NPART=4).
    sumsq walks the full row in the same BLOCK_H chunks as the 1-CTA kernel (V4 bitwise)."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    tok = tl.program_id(0)
    part = tl.program_id(1)
    NCHUNK: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    BLOCK_Q: tl.constexpr = H // NPART  # 1024 when H=4096, NPART=4 (power of two)
    slots = tl.arange(0, KP)
    in_k = slots < TOPK
    live = in_k & (tl.load(ids_ptr + tok * TOPK + slots, mask=in_k, other=-1) >= 0)
    sc = tl.load(s_ptr + tok * TOPK + slots, mask=live, other=0.0).to(tl.float32)
    row = (tok * TOPK + slots)[:, None] * H
    offs0 = tl.arange(0, BLOCK_H)

    sumsq = tl.zeros([KP], dtype=tl.float32)
    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK_H + offs0
        hm = offs < H
        w = tl.load(w_ptr + offs, mask=hm, other=0.0).to(tl.float32)
        x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None] & hm[None, :], other=0.0).to(tl.float32)
        yv = x * w[None, :]
        sumsq += tl.sum(yv * yv, axis=1)
    coef = tl.where(live, sc * tl.rsqrt(sumsq / H + eps), 0.0)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()

    offs = part * BLOCK_Q + tl.arange(0, BLOCK_Q)
    hm = offs < H
    w = tl.load(w_ptr + offs, mask=hm, other=0.0).to(tl.float32)
    x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None] & hm[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(x * coef[:, None], axis=0) * w
    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=hm)


def epilogue_triton(expert_out, topk_scores, slot_ids, norm_w, eps=POST_NORM_EPS, block_h=EPILOGUE_BLOCK_H,
                    warps=EPILOGUE_WARPS):
    y = expert_out.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = min(block_h, triton.next_power_of_2(hidden))
    npart = 4
    assert hidden % npart == 0
    _epilogue_kernel[(tokens, npart)](
        y, norm_w, topk_scores.contiguous(), slot_ids.contiguous(), out, eps,
        TOPK=topk, KP=triton.next_power_of_2(topk), H=hidden, BLOCK_H=block_h, NPART=npart,
        num_warps=warps, **_pdl_kwargs(y.device))
    return out
```

Call sites (`_experts` / `smallm` / Marlin) stay `epilogue_triton(...)`. No new prepare() buffer: `y`, scores, ids, `post_expert_w` / `norm_w` are already in play.

### 2b. Test sketch (V4)

```python
def test_h9_epilogue_bitwise(y, scores, ids, w, eps=1e-8):
    # y [T, 8, 4096] bf16, ids int32 with -1 for remote, w [4096] bf16
    tokens = y.shape[0]
    block_h = 2048 if tokens <= 8 else 1024
    a = epilogue_triton(y, scores, ids, w, eps=eps, block_h=block_h)           # grid (T, 4)
    b = epilogue_triton_1cta(y, scores, ids, w, eps=eps, block_h=block_h)      # today's (T,) kernel
    assert torch.equal(a, b)  # V4: same sumsq chunk order, same slot reduction, one BF16 store

    # dead slots (id < 0) contribute 0; a fully-remote token is a zero row
    remote = (ids < 0).all(-1)
    if remote.any():
        assert torch.equal(a[remote], torch.zeros_like(a[remote]))
```

Keep the current 1-CTA kernel under a private name for this test, or compile both `NPART=1` (BLOCK_Q = H, one write loop) and `NPART=4` from the same source.

### 2c. Risks

- **Bitwise.** `sumsq` must use the **same** `BLOCK_H` as the kernel you compare against. Forcing `BLOCK_H=1024` at M ≤ 8 (S1 uses 2048, `NCHUNK=2`) changes the FP32 reduction tree and fails V4.
- **Quarter width.** `H // 4` is a power of two (4096 / 4 = 1024). A hidden size that is not a multiple of 4×power-of-two breaks `tl.arange`.
- **L2 re-read.** 4 CTAs × 8 slots × 4096 × 2 B ≈ 256 KB of `y` per token, all from L2 after the first CTA. That is the occupancy win (4 CTAs vs 1) and the extra bytes.
- **Graph / counters.** No persistent counter. Grid is `(T, 4)` from the shape. Capture-safe as today’s epilogue.
- **PDL.** `gdc_wait` still first; `gdc_launch_dependents` after `coef`, before the quarter store — same as v4, so a dependent unpooler can start while quarters land.

---

## 3. H8 — unpooler T ≤ 8, GEMV N = 1024 into `[T, 8192]`

One fill of a prepare()-allocated `[8, 8192]` buffer; the GEMV writes columns 0:1024 with `stride_om = 8192`. Columns 1024:8192 stay the fill. No `F.pad`, no memcpy of the 1024-col slice.

### 3a. prepare() + launch (replace the `F.pad` block in `_kernel_sglang` / `_kernel_torch`)

```python
# in _prepare_sglang / prepare, next to dense weights:
state["unpool_out"] = torch.empty(
    (SLICED_UNPOOL_MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
# first-touch so the graph sees a defined allocation; kernel_fn zeros every call
state["unpool_out"].zero_()

_UNPOOL_CFG = (32, 256, 1, 4, 4)  # BN, BK, SPLIT, warps, stages → 32 CTAs, SPLIT=1


def _unpool_sliced(expert_output, state):
    """T <= 8: this GPU's 1024 rows of W, stored at columns 0:1024 of a zero [T, 8192]."""
    t, k = expert_output.shape  # k = 4096
    out = state["unpool_out"][:t]
    out.zero_()  # captured memset; the 7/8 "all-gather identity"
    w = state["dense"]["unpooler"]  # [8192, 4096]; N=1024 uses rows 0:1023
    BN, BK, SPLIT, warps, stages = _UNPOOL_CFG
    BM = 16
    part, done, part_m = state["router_part"], state["router_done"], state["router_part"].shape[1]
    grid = (triton.cdiv(UNPOOL_SLICE, BN), SPLIT, triton.cdiv(t, BM))  # (32, 1, 1)
    _gemv_splitk[grid](
        expert_output, w, w, out, part, done,
        t, UNPOOL_SLICE, D_EXPERT,
        expert_output.stride(0), w.stride(0), out.stride(0),  # stride_om == 8192
        part_m,
        BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, HAS_LO=False, FIXUP=False, SPLIT_X=False,
        num_warps=warps, num_stages=stages, **_pdl_kwargs(expert_output.device))
    return out  # view [T, 8192], live for this call only
```

```python
# _kernel_sglang, replacing:
#   unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
#   ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))
if tokens <= SLICED_UNPOOL_MAX_TOKENS:
    ffn_output = _unpool_sliced(expert_output, state)
else:
    ffn_output = _gemm(expert_output, dense["unpooler"])
```

Same swap in `_kernel_torch` (drop the `F.pad((y @ dense["unpooler"][:UNPOOL_SLICE].T).float(), ...)`).

`FIXUP=False` / SPLIT=1: dummy `part`/`done` are unused. If you have not yet landed H13, the current `_gemv_splitk` with `ATOMIC=False` and `stride_om=HIDDEN` is enough:

```python
_gemv_splitk[(UNPOOL_SLICE // 32, 1, 1)](
    expert_output, dense["unpooler"], dense["unpooler"], ffn_output,
    tokens, UNPOOL_SLICE, D_EXPERT,
    expert_output.stride(0), dense["unpooler"].stride(0), HIDDEN,
    BM=16, BN=32, BK=256, SPLIT=1, HAS_LO=False, ATOMIC=False, SPLIT_X=False,
    num_warps=4, num_stages=4, **_pdl_kwargs(expert_output.device))
```

### 3b. Test sketch

```python
def test_h8_unpooler(state, y):  # y [T, 4096] bf16, T in {1, 8}
    out = _unpool_sliced(y, state)
    assert out.shape == (y.shape[0], 8192) and out.stride(0) == 8192
    assert torch.equal(out[:, 1024:], torch.zeros_like(out[:, 1024:]))  # pad region

    # bitwise vs the same GEMV into a tight [T, 1024] (SPLIT=1, stride 1024)
    tight = torch.empty((y.shape[0], 1024), dtype=torch.bfloat16, device=y.device)
    _gemv_splitk[(32, 1, 1)](
        y, state["dense"]["unpooler"], state["dense"]["unpooler"], tight,
        state["router_part"], state["router_done"],
        y.shape[0], 1024, 4096, y.stride(0), state["dense"]["unpooler"].stride(0), 1024,
        state["router_part"].shape[1],
        BM=16, BN=32, BK=256, SPLIT=1, HAS_LO=False, FIXUP=False, SPLIT_X=False,
        num_warps=4, num_stages=4)
    assert torch.equal(out[:, :1024], tight)

    # vs cuBLAS + F.pad: bf16 1 % class (tl.dot vs nvjet)
    cublas = F.pad(_bf16_gemm_dispatch_impl(y, state["dense"]["unpooler"][:1024], None), (0, 7168))
    rel = (out.float() - cublas.float()).norm() / cublas.float().norm().clamp_min(1e-12)
    assert rel < 1e-2

    out2 = _unpool_sliced(y, state)
    assert torch.equal(out, out2)  # zero_ each call; leftover T=8 rows cannot leak into T=1
```

### 3c. Risks

- **Graph capture.** `unpool_out` is in `state` (stable pointer). `out.zero_()` must stay inside `kernel_fn` so the memset is in the graph. A prepare()-only zero leaves columns 1024:8192 holding the previous replay’s values.
- **Returning a view.** `return out` is `state["unpool_out"][:T]`. The FFN boundary consumes it in the same call. Do not stash that view across `kernel_fn` invocations.
- **T = 1 after T = 8.** `zero_()` covers the `[T, 8192]` view only. That is enough because the next consumer sees `[:T]`. The unused rows 1:8 of the buffer can stay dirty.
- **No counter.** SPLIT = 1, plain store. H13’s `done` is untouched (`FIXUP=False`).
- **Numerics.** Triton GEMV vs today’s cuBLAS slice is the bf16 1 % bound, bitwise in the pad region (true zeros). The tight-vs-strided self-check is bitwise.
- **Fill cost.** One 16 KB memset (~0.8 µs) replaces cuBLAS split-K reduce + `F.pad` memcpy (~5.4 µs on #0/#1/#6/#7). Under H6' this fill runs on the main stream after the experts, with no pooler beside it, so it should not stretch the way the 1 KB router fill does today.

---

## Wiring order in the layer (H6' context)

RPAR is the parallelism half of H6'; ORDER is the stream half. After both:

```python
logits = _router_logits(h_in, router_input, state, path)   # 256–1024 CTAs, last-CTA fixup
side.wait_stream(main)
with torch.cuda.stream(side):
    topk_w, topk_ids = state["routing"](logits, ...)
pooled = _pooler(h_in, state, pooler_path)                 # no HBM competitor
main.wait_stream(side)
# ... experts ...
y = epilogue_triton(...)                                   # grid (T, 4)
ffn = _unpool_sliced(y, state) if T <= 8 else _gemm(y, W)  # stride 8192
```

Kill / confirm (from D4 / D6): RPAR router alone ≤ 4 µs; H9 Δ ≥ 1.5 µs and V4 bitwise; H8 Δ ≥ 3 µs on #0/#1/#6/#7.