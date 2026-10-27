# okc: problem=muse_bf16_proj
"""bf16_proj candidate v2 (card C23; Sonnet-A for the director). v1b (board entry c3b7c70b5edd43c5, 1.156x) + a per-(layer, bucket) PLAN table that can select the
cheap variants of fable-09 D4 section 4 (H13), measured by kernels/muse_bf16_proj/measure_cheap_v1.py on the judge's H200:

  direct  the GEMV with SPLIT = 1 and a direct store (no zero-fill, no atomics), e.g. the unpooler at m <= 8 (1024 rows x 4096)
  fix     split-K with the last-CTA fixup (kernel _gemv_fix): every CTA stores its FP32 partial into a workspace made in prepare(), takes an arrival
          ticket (CUTLASS's semaphore pattern, judge-proven in the decode layer's _router_fix_kernel), and the last CTA of each tile sums the partials in slice
          order, stores the output in its dtype and puts the counter back to zero. No torch.zeros, no FP32 atomics on the output, deterministic. BM = 0 means one
          M block (next power of two of m, >= 16): W is read once (router at m = 128 / 256); the grid has >= 256 CTAs at the router's 256 columns.
  tmm     torch.mm (cuBLAS) x @ W[:n].T.    gemv  v1b's own _gemv_splitk path (atomic when SPLIT > 1).    cublas  SGLang's dispatch (the baseline).

The shipped table _PLAN holds v1b's choices in every cell (so v2 == v1b: the same kernels, the same configs). _PLAN_NEW is the candidate set the measurement chooses
from: one entry per cell the card touches; tests flip cells to it and check them against the organisers' reference. Cells are chosen per (layer, shape bucket) only,
never from input values; a cell that cannot run (no sm_80, workspace too small, shape not divisible) falls back to the baseline path of v1b.
Predicted: v1b 1.156x until the measurement says otherwise (this file ships v1b's table).
"""

from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # in the judge's image
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
except Exception:  # local runs without SGLang
    def _bf16_gemm_dispatch_impl(x, w, b):
        return x @ w.T


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


@triton.jit
def _gemv_splitk(x_ptr, w_ptr, wlo_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                 HAS_LO: tl.constexpr, ATOMIC: tl.constexpr):
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
    if ATOMIC:
        tl.atomic_add(optr, acc, mask=om)
    else:
        tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=om)


@triton.jit(do_not_specialize=["M"])
def _gemv_fix(x_ptr, w_ptr, wlo_ptr, out_ptr, part_ptr, done_ptr, M, stride_x, stride_o,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, RG: tl.constexpr,
              HAS_LO: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    """out = x . (W (+ W_lo))^T, split-K over SPLIT CTAs with NO zero-fill and NO atomics on the output: every CTA stores its FP32 partial, takes an arrival
    ticket, and the last CTA of each (m-block, n-block) tile sums the partials in slice order (RG loads in flight) and stores the output in its dtype; that CTA
    puts the tile's counter back to zero. Needs K % (SPLIT * BK) == 0, N % BN == 0, SPLIT % RG == 0. Deterministic."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k_per: tl.constexpr = K // SPLIT
    k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        rk = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * stride_x + rk[None, :], mask=rm[:, None] < M, other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :])
        acc += tl.dot(x, tl.trans(w))
        if HAS_LO:
            wl = tl.load(wlo_ptr + rn[:, None] * K + rk[None, :])
            acc += tl.dot(x, tl.trans(wl))
    tl.store(part_ptr + (pid_k * M + rm)[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M)
    tl.debug_barrier()
    NB: tl.constexpr = N // BN
    ticket = tl.atomic_add(done_ptr + pid_m * NB + pid_n, 1, sem="acq_rel", scope="gpu")
    if ticket == SPLIT - 1:
        tot = tl.zeros((BM, BN), dtype=tl.float32)
        for s0 in range(0, SPLIT, RG):  # .cg: skip L1 so no stale line of this buffer can be read
            for s in tl.static_range(RG):
                tot += tl.load(part_ptr + ((s0 + s) * M + rm)[:, None] * N + rn[None, :], mask=rm[:, None] < M, other=0.0, cache_modifier=".cg")
        tl.store(out_ptr + rm[:, None] * stride_o + rn[None, :], tot.to(out_ptr.dtype.element_ty), mask=rm[:, None] < M)
        tl.store(done_ptr + pid_m * NB + pid_n, 0)


# (BN, BK, SPLIT, warps, stages) per (layer, bucket); BM follows from m. Best measured configs (C9 v1/v2). Unchanged from v1b.
_CFG = {
    ("router", 1): (64, 128, 16, 4, 4),
    ("router", 8): (64, 128, 16, 4, 4),
    ("router", 16): (64, 64, 16, 4, 4),
    ("router", 32): (16, 128, 16, 4, 3),
    ("router", 128): (32, 256, 8, 4, 3),
    ("router", 256): (32, 128, 8, 4, 3),
    ("pooler", 1): (32, 512, 1, 4, 4),
    ("pooler", 8): (32, 512, 1, 4, 4),  # v1b: the m=8 bucket regressed with BN16 (0.960x in check f046e5219a2f44ed)
    ("pooler", 16): (32, 512, 1, 4, 4),
    ("pooler", 32): (32, 512, 1, 4, 3),
    ("o", 1): (32, 128, 1, 4, 4),
    ("o", 8): (32, 128, 1, 4, 4),
    ("o", 16): (32, 128, 1, 4, 4),
    ("o", 32): (64, 128, 1, 4, 4),
}
_ROUTER_TRITON_MAX_M = 256
_BUCKETS = (1, 8, 16, 32, 128, 256)
_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")

# ---------------------------------------------------------------------------------------------------------------- the plan
#   ("gemv", BN, BK, SPLIT, warps, stages)            v1b's _gemv_splitk path (SPLIT > 1: torch.zeros + FP32 atomics)
#   ("direct", BN, BK, warps, stages)                 _gemv_splitk with SPLIT = 1: direct store, no zero-fill, no atomics
#   ("fix", BM, BN, BK, SPLIT, warps, stages, RG)     _gemv_fix, last-CTA fixup; BM 0 = one M block (next power of two of m, >= 16)
#   ("tmm",)                                          torch.mm(x, W[:n].T)
#   ("cublas",)                                       SGLang's dispatch (the baseline; for the router: F.linear FP32 at m <= 8, else two cuBLAS GEMMs on hi / lo)
# v1b's choices, cell by cell:
_PLAN = {}
for _l in _LAYERS:
    for _b in _BUCKETS:
        _PLAN[(_l, _b)] = ("gemv",) + _CFG[(_l, _b)] if (_l, _b) in _CFG else ("cublas",)
# Judge-measured winners (measure_cheap_v1, job cadb306e90e7428d, H200, 3 seeds; x = board/variant). Router m=32 keeps v1b (fix 0.94x).
_JUDGED = {
    ("unpooler", 1): ("fix", 16, 32, 128, 8, 4, 4, 8),     # 7.17 -> 5.25 us (x1.37)
    ("unpooler", 8): ("fix", 16, 32, 128, 8, 4, 4, 8),     # 6.82 -> 5.44 us (x1.25)
    ("router", 1): ("fix", 0, 32, 128, 32, 4, 4, 8),       # 6.66 -> 6.14 us (x1.09)
    ("router", 8): ("fix", 0, 32, 128, 32, 4, 4, 8),       # 6.66 -> 6.40 us (x1.05)
    ("router", 16): ("fix", 0, 32, 128, 32, 4, 4, 8),      # 7.39 -> 6.59 us (x1.12)
    ("router", 128): ("fix", 64, 32, 128, 8, 4, 3, 8),     # 10.24 -> 8.83 us (x1.15)
    ("router", 256): ("fix", 64, 32, 128, 8, 4, 3, 8),     # 12.58 -> 10.91 us (x1.15)
}
_PLAN.update(_JUDGED)

# Candidates the measurement chooses from (names = measure_cheap_v1.py's variants of the same cells). NOT shipped as defaults.
_PLAN_NEW = {
    ("unpooler", 1): ("direct", 32, 256, 4, 4),
    ("unpooler", 8): ("direct", 32, 256, 4, 4),
    ("router", 1): ("fix", 16, 64, 128, 64, 4, 4, 8),
    ("router", 8): ("fix", 16, 64, 128, 64, 4, 4, 8),
    ("router", 16): ("fix", 16, 64, 128, 64, 4, 4, 8),
    ("router", 32): ("fix", 32, 64, 128, 64, 4, 4, 8),
    ("router", 128): ("fix", 0, 32, 64, 32, 4, 4, 8),
    ("router", 256): ("fix", 0, 32, 64, 32, 4, 4, 8),
}
# Every variant the measurement times, per cell kind (tests run all of them against the reference).
_CANDIDATES = {
    "unpooler": [("direct", 32, 256, 4, 4), ("direct", 16, 256, 4, 4), ("direct", 32, 512, 4, 3), ("direct", 16, 512, 4, 3),
                 ("fix", 16, 32, 128, 4, 4, 4, 4), ("fix", 16, 16, 256, 4, 4, 4, 4), ("fix", 16, 32, 128, 8, 4, 4, 8), ("tmm",)],
    "router_small": [("fix", 0, 64, 128, 64, 4, 4, 8), ("fix", 0, 32, 128, 32, 4, 4, 8), ("fix", 0, 16, 128, 64, 4, 3, 8), ("fix", 0, 64, 128, 64, 4, 4, 1)],
    "router_big": [("fix", 0, 32, 64, 32, 4, 4, 8), ("fix", 0, 16, 64, 16, 4, 4, 8), ("fix", 0, 64, 64, 64, 8, 3, 8),
                   ("fix", 64, 32, 128, 8, 4, 3, 8), ("fix", 64, 32, 128, 16, 4, 3, 8), ("fix", 64, 64, 64, 16, 4, 4, 8)],
}
FIX_WS_FLOATS = 8 * 1024 * 1024   # split-K partials (FP32): SPLIT * m * n floats must fit (router 64 x 256 x 256, unpooler 4 x 8 x 1024)
FIX_TILES = 4096                  # arrival counters (int32), zero between launches


def _bucket(m):
    for b in _BUCKETS:
        if m <= b:
            return b
    return None


def _triton_ok(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = lo.contiguous()
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    dev = weights["router"].device
    # fixup workspace + arrival counters (made here, never in kernel_fn: no torch.zeros node in the graph); the last CTA of a tile resets its counter
    state["fix_ws"] = torch.empty(FIX_WS_FLOATS, device=dev, dtype=torch.float32)
    state["fix_sem"] = torch.zeros(FIX_TILES, device=dev, dtype=torch.int32)
    return state


def _run(x, w, wlo, n, out_dtype, cfg):
    m, k = x.shape
    BN, BK, SPLIT, warps, stages = cfg
    BM = 16 if m <= 16 else (32 if m <= 32 else 64)
    atomic = SPLIT > 1
    if atomic:
        acc = torch.zeros((m, n), device=x.device, dtype=torch.float32)
    else:
        acc = torch.empty((m, n), device=x.device, dtype=out_dtype)
    grid = (triton.cdiv(n, BN), SPLIT, triton.cdiv(m, BM))
    _gemv_splitk[grid](x, w, wlo if wlo is not None else w, acc, m, n, k, x.stride(0), w.stride(0), acc.stride(0),
                       BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, HAS_LO=wlo is not None, ATOMIC=atomic,
                       num_warps=warps, num_stages=stages)
    return acc if acc.dtype == out_dtype else acc.to(out_dtype)


def _fix_fits(state, m, n, k, bm, bn, bk, split, rg):
    """Shape-only test: can the fixup kernel run this cell (divisibility, workspace, counters)?"""
    if state.get("fix_ws") is None or n % bn or k % (split * bk) or split % rg:
        return False
    tiles = (n // bn) * triton.cdiv(m, bm)
    return split * m * n <= state["fix_ws"].numel() and tiles <= state["fix_sem"].numel()


def _run_fix(x, state, layer, n, out_dtype, var):
    _, bm, bn, bk, split, warps, stages, rg = var
    m, k = x.shape
    bm = bm or max(16, triton.next_power_of_2(m))
    if bm > 256 or not _fix_fits(state, m, n, k, bm, bn, bk, split, rg):
        return None
    lo = layer == "router"
    w = state["router_hi"] if lo else state[layer]
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    _gemv_fix[(n // bn, split, triton.cdiv(m, bm))](
        x, w, state["router_lo"] if lo else w, out, state["fix_ws"], state["fix_sem"], m, x.stride(0), out.stride(0),
        BM=bm, BN=bn, BK=bk, SPLIT=split, RG=rg, HAS_LO=lo, K=k, N=n, num_warps=warps, num_stages=stages)
    return out


def _baseline(x, state, layer, m, n):
    """v1b's non-Triton paths (also the fallback of every Triton cell)."""
    if layer == "router":
        if m <= 8:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    b = _bucket(m)
    var = _PLAN.get((layer, b), ("cublas",)) if b is not None else ("cublas",)
    kind = var[0]
    if kind != "cublas" and kind != "tmm" and not _triton_ok(x.device):
        return _baseline(x, state, layer, m, n)
    if kind == "gemv":
        if layer == "router":
            return _run(x, state["router_hi"], state["router_lo"], n, torch.float32, var[1:])
        return _run(x, state[layer], None, n, torch.bfloat16, var[1:])
    if kind == "direct":
        BN, BK, warps, stages = var[1:]
        return _run(x, state[layer], None, n, torch.bfloat16, (BN, BK, 1, warps, stages))
    if kind == "fix":
        out = _run_fix(x, state, layer, n, torch.float32 if layer == "router" else torch.bfloat16, var)
        if out is not None:
            return out
    if kind == "tmm" and layer != "router":
        return torch.mm(x, state[layer][:n].t())
    return _baseline(x, state, layer, m, n)
