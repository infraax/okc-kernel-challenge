# okc: problem=muse_bf16_proj
"""bf16_proj candidate v1b (director; v1 = check f046e5219a2f44ed: 1.140x [1.122,1.157]). Hypothesis: Triton split-K for the router (BF16 hi+lo, FP32 accumulate, at
every m) and for pooler / o at m <= 32; cuBLAS (SGLang's dispatch) for everything else.
Configs come from the judge's H200 via measurement checks b4f960c09e084964 (v1) and 26664259365345a9 (v2),
chosen per shape bucket only (never from input values).
Predicted (tools/score_sim.py what-if, judge baseline 728c9344 + same-method deltas): 1.166x (range 1.10-1.22).
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


# (BN, BK, SPLIT, warps, stages) per (layer, bucket); BM follows from m. Best measured configs (C9 v1/v2).
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


def _bucket(m):
    for b in (1, 8, 16, 32, 128, 256):
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


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, m, n = cfg["layer"], cfg["m"], cfg["n"]
    b = _bucket(m)
    use_triton = _triton_ok(x.device)
    if layer == "router":
        if use_triton and m <= _ROUTER_TRITON_MAX_M:
            return _run(x, state["router_hi"], state["router_lo"], n, torch.float32, _CFG[("router", b)])
        if m <= 8:
            return F.linear(x.float(), state["router"])
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if use_triton and (layer, b) in _CFG:
        return _run(x, state[layer], None, n, torch.bfloat16, _CFG[(layer, b)])
    return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
