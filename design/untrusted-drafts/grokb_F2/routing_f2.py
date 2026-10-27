# okc: problem=muse_decode_layer
"""F2 router for T <= 32: GEMV hi+lo, biased top-8, renorm, local ids. One kernel."""

import torch
import triton
import triton.language as tl

_MULT = 0.1767766952966369  # reference ROUTER_MULTIPLIER, SGLang router_output_multiplier
_EPS = 1e-15                # router_score_division_eps / ROUTE_EPS


@triton.jit
def _fence(n: tl.constexpr):
    """acq_rel GPU fence on each lane of a 4-warp CTA (n = 128)."""
    v = tl.arange(0, n)
    return tl.inline_asm_elementwise(
        "fence.acq_rel.gpu; mov.u32 $0, $1;",
        "=r,r", [v], dtype=tl.int32, is_pure=False, pack=1,
    )


@triton.jit
def _dot_row0(x_row, w):
    """One token against w [BN, BK] bf16. Row 0 of a 16-row tl.dot, FP32 acc.

    Matches _gemv_splitk at T = 1, where BM = 16 and only row 0 is live.
    """
    rm = tl.arange(0, 16)
    z = x_row * 0
    xmat = tl.where(rm[:, None] == 0, x_row[None, :], z[None, :])
    y = tl.dot(xmat, tl.trans(w), out_dtype=tl.float32)
    return tl.sum(tl.where(rm[:, None] == 0, y, 0.0), 0)


@triton.jit
def _insert(vals, cols, wts, z, e, s):
    """Eight slots, higher biased score first, lower expert index on a tie.

    That is reference.py's biased.topk(8) order (sorted; equal values keep the
    smaller index). pos == 8 leaves the row unchanged.
    """
    i = tl.arange(0, 8)
    ahead = ((vals > z) | ((vals == z) & (cols < e))).to(tl.int32)
    pos = tl.sum(ahead)
    ii = i[:, None]
    jj = i[None, :]
    sel = (((ii < pos) & (jj == ii)) | ((ii > pos) & (jj == ii - 1))).to(tl.int32)
    nv = tl.sum(sel.to(tl.float32) * vals[None, :], 1)
    nc = tl.sum(sel * cols[None, :], 1)
    nw = tl.sum(sel.to(tl.float32) * wts[None, :], 1)
    take = i == pos
    return tl.where(take, z, nv), tl.where(take, e, nc), tl.where(take, s, nw)


@triton.jit
def _router_f2(
    x_ptr, hi_ptr, lo_ptr, bias_ptr, start_ptr, part_ptr, done_ptr,
    ids_ptr, w_ptr, loc_ptr, M, stride_x, stride_w,
    BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, SPLIT_X: tl.constexpr,
    MULT: tl.constexpr, EPS: tl.constexpr,
):
    s = tl.program_id(0)       # K-split
    token = tl.program_id(1)
    k_per = 8192 // SPLIT
    k0 = s * k_per
    for n0 in tl.static_range(0, 256, BN):
        acc = tl.zeros((BN,), dtype=tl.float32)
        rn = n0 + tl.arange(0, BN)
        nmask = rn[:, None] < 256
        for kk in range(0, k_per, BK):
            rk = k0 + kk + tl.arange(0, BK)
            x = tl.load(x_ptr + token * stride_x + rk, mask=rk < 8192, other=0.0)
            hi = tl.load(hi_ptr + rn[:, None] * stride_w + rk[None, :], mask=nmask, other=0.0)
            if SPLIT_X:
                # FP32 x = xh + xl (both bf16). Drop xl·Wl (~2^-32), as _gemv_splitk does.
                xh = x.to(tl.bfloat16)
                xl = (x - xh.to(tl.float32)).to(tl.bfloat16)
                acc += _dot_row0(xh, hi)
                acc += _dot_row0(xl, hi)
                lo = tl.load(lo_ptr + rn[:, None] * stride_w + rk[None, :], mask=nmask, other=0.0)
                acc += _dot_row0(xh, lo)
            else:
                acc += _dot_row0(x, hi)
                lo = tl.load(lo_ptr + rn[:, None] * stride_w + rk[None, :], mask=nmask, other=0.0)
                acc += _dot_row0(x, lo)
        tl.store(part_ptr + (s * M + token) * 256 + rn, acc, mask=rn < 256)
    # Publish this split's stores, then arrive. Last split reduces in split order.
    live = tl.sum(_fence(128) * 0)
    tl.debug_barrier()
    lane = tl.arange(0, 128)
    ticket = tl.atomic_add(done_ptr + token, 1, mask=(lane == 0), sem="acq_rel", scope="gpu")
    prev = tl.sum(tl.where(lane == 0, ticket, 0)) + live
    if prev == SPLIT - 1:
        live = tl.sum(_fence(128) * 0)
        vals = tl.full((8,), float("-inf"), tl.float32)
        cols = tl.full((8,), 32767, tl.int32)
        wts = tl.zeros((8,), tl.float32)
        for e in range(256):
            logit = live
            for ss in tl.static_range(SPLIT):
                logit += tl.load(part_ptr + (ss * M + token) * 256 + e)
            sig = tl.sigmoid(logit * MULT)
            vals, cols, wts = _insert(vals, cols, wts, sig + tl.load(bias_ptr + e), e, sig)
        i = tl.arange(0, 8)
        base = token * 8
        tl.store(ids_ptr + base + i, cols)
        tl.store(w_ptr + base + i, wts / (tl.sum(wts) + EPS))
        rel = (cols - tl.load(start_ptr)) & 255
        tl.store(loc_ptr + base + i, tl.where(rel < 32, rel, -1).to(tl.int32))


def _router_cfg(t):
    """(BN, BK, SPLIT, stages) — sonnet_v2._CFG['router']. Warps fixed at 4."""
    if t <= 8:
        return 64, 128, 16, 4
    if t <= 16:
        return 64, 64, 16, 4
    return 16, 128, 16, 3


def fused_router_f2(x, router_hi, router_lo, router_bias, local_start):
    """Route T <= 32 tokens. Graph-safe: grid and buffers depend only on T.

    Inputs
    ------
    x : [T, 8192] float32 if T <= 8 (boundary FP32 hidden), else bfloat16.
    router_hi, router_lo : [256, 8192] bfloat16, hi = bf16(router),
        lo = bf16(router - float(hi)). Same tensors as state['router_hi'/'router_lo'].
    router_bias : [256] float32. Added after sigmoid; not part of the weight.
    local_start : int32 device tensor, shape [1]. This GPU's first expert.

    Returns
    -------
    topk_ids : int32 [T, 8], global ids in torch.topk order on the biased scores.
    topk_weights : float32 [T, 8], sigmoid(mult * logit) / (sum + 1e-15).
    local_ids : int32 [T, 8], (id - local_start) mod 256 if that is < 32, else -1.
        This is what local_ids_triton writes and what moe_align_block_size reads.
    """
    if x.ndim != 2 or x.shape[1] != 8192 or not 1 <= x.shape[0] <= 32:
        raise ValueError("fused_router_f2 expects x [T, 8192] with 1 <= T <= 32")
    x = x.contiguous()
    router_hi = router_hi.contiguous()
    router_lo = router_lo.contiguous()
    router_bias = router_bias.contiguous()
    t = x.shape[0]
    bn, bk, split, stages = _router_cfg(t)
    part = torch.empty((split, t, 256), dtype=torch.float32, device=x.device)
    done = torch.zeros((t,), dtype=torch.int32, device=x.device)
    ids = torch.empty((t, 8), dtype=torch.int32, device=x.device)
    weights = torch.empty((t, 8), dtype=torch.float32, device=x.device)
    local = torch.empty((t, 8), dtype=torch.int32, device=x.device)
    _router_f2[(split, t)](
        x, router_hi, router_lo, router_bias, local_start, part, done,
        ids, weights, local, t, x.stride(0), router_hi.stride(0),
        BN=bn, BK=bk, SPLIT=split, SPLIT_X=x.dtype == torch.float32,
        MULT=_MULT, EPS=_EPS, num_warps=4, num_stages=stages,
    )
    return ids, weights, local
