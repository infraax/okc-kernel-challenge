# okc: problem=muse_bf16_proj
"""MEASUREMENT-ONLY candidate (card C9). Run as `okc check`, never submit.

kernel_fn is SGLang's own baseline (scores ~1.00x). prepare() runs once, untimed, and times a grid of Triton
split-K GEMV/GEMM configurations per scored shape on the judge's H200, on synthetic activations
(nothing derived from the judge's inputs), each as a CUDA-graph replay after an L2 flush, like the judge.
It prints one line per (shape, config): `MEAS <layer> m=<m> <cfg> us=<median> err=<rel_l2 vs fp64>`, plus the
baseline's time measured the same way, so the table is calibrated against the judge's own numbers.
"""

import math
import time
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # in the judge's image
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
except Exception:  # local dry runs
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
    x: torch.Tensor
    config: Config


ROUTER_FP32_MAX_TOKENS = 8
SCORED_M = (1, 8, 16, 32, 128, 256)
BUDGET_S = 240.0  # stay inside prepare()'s 300 s


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


def _make_variant(w, wlo, m, n, k, out_dtype, BM, BN, BK, SPLIT, warps, stages):
    """A callable computing x @ w[:n].T (+ lo) with one config; allocations happen here, not per call."""
    atomic = SPLIT > 1
    acc = torch.empty((m, n), device=w.device, dtype=torch.float32)
    out = torch.empty((m, n), device=w.device, dtype=out_dtype) if (atomic and out_dtype != torch.float32) else acc
    if not atomic and out_dtype != torch.float32:
        acc = out = torch.empty((m, n), device=w.device, dtype=out_dtype)
    grid = (triton.cdiv(n, BN), SPLIT, triton.cdiv(m, BM))

    def run(x):
        if atomic:
            acc.zero_()
        _gemv_splitk[grid](x, w, wlo if wlo is not None else w, acc, m, n, k, x.stride(0), w.stride(0), acc.stride(0),
                           BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, HAS_LO=wlo is not None, ATOMIC=atomic,
                           num_warps=warps, num_stages=stages)
        if out is not acc:
            out.copy_(acc)
        return out
    return run


def _time_graph(fn, x, flush, reps=25):
    """Median µs of CUDA-graph replays of fn(x), each after an L2 flush (as the judge does)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(x)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn(x)
    ts = []
    for _ in range(reps):
        flush.add_(1)  # touches 256 MB > 50 MB L2
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2], y


def _configs(m):
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    out = []
    for BN in (32, 64):
        for BK in (128, 256):
            for SPLIT in (1, 4, 8):
                out.append((bm, BN, BK, SPLIT, 4, 3))
    if m >= 128:
        out += [(64, 64, 64, 1, 4, 3), (64, 128, 64, 1, 8, 3), (128, 128, 64, 1, 8, 3)]
    return out


def _measure(weights):
    if not torch.cuda.is_available():
        print("MEAS skipped: no cuda")
        return
    t0 = time.time()
    dev = weights["qkv_gate"].device
    flush = torch.zeros(64 * 1024 * 1024, device=dev, dtype=torch.float32)
    router = weights["router"]
    r_hi = router.to(torch.bfloat16).contiguous()
    r_lo = (router - r_hi.float()).to(torch.bfloat16).contiguous()
    gen = torch.Generator(device=dev).manual_seed(1234)
    print(f"MEAS start gpu={torch.cuda.get_device_name()} triton={triton.__version__}")
    for m in SCORED_M:
        for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
            if time.time() - t0 > BUDGET_S:
                print("MEAS budget reached")
                return
            W = weights[layer]
            k = W.shape[1]
            n = 1024 if (layer == "unpooler" and m <= 8) else W.shape[0]
            x = torch.randn((m, k), generator=gen, device=dev).to(torch.bfloat16)
            ref = (x.double() @ W[:n].double().T)
            # baseline, measured the same way
            if layer == "router":
                if m <= ROUTER_FP32_MAX_TOKENS:
                    base = lambda x: F.linear(x.float(), router)
                else:
                    ht, lt = r_hi.t().contiguous(), r_lo.t().contiguous()
                    def base(x, ht=ht, lt=lt):
                        acc = torch.mm(x, ht, out_dtype=torch.float32)
                        acc += torch.mm(x, lt, out_dtype=torch.float32)
                        return acc
            else:
                base = lambda x, w=W[:n]: _bf16_gemm_dispatch_impl(x, w, None)
            us, y = _time_graph(base, x, flush)
            err = float((y.double() - ref).norm() / ref.norm())
            print(f"MEAS {layer} m={m} n={n} k={k} baseline us={us:.2f} err={err:.2e}")
            best = (us, "baseline")
            w_main = r_hi if layer == "router" else W
            w_lo = r_lo if layer == "router" else None
            odt = torch.float32 if layer == "router" else torch.bfloat16
            for cfg in _configs(m):
                if time.time() - t0 > BUDGET_S:
                    break
                try:
                    fn = _make_variant(w_main, w_lo, m, n, k, odt, *cfg)
                    us, y = _time_graph(fn, x, flush)
                    err = float((y.double() - ref).norm() / ref.norm())
                except Exception as e:  # a config that does not compile/run is data too
                    print(f"MEAS {layer} m={m} cfg={cfg} FAIL {type(e).__name__}: {str(e)[:80]}")
                    continue
                print(f"MEAS {layer} m={m} cfg=BM{cfg[0]}_BN{cfg[1]}_BK{cfg[2]}_S{cfg[3]}_w{cfg[4]}_s{cfg[5]} us={us:.2f} err={err:.2e}")
                if us < best[0] and err < (1e-4 if layer == "router" else 1e-2):
                    best = (us, cfg)
            print(f"MEAS BEST {layer} m={m} us={best[0]:.2f} cfg={best[1]}")
    print(f"MEAS done in {time.time() - t0:.1f}s")


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    _measure(weights)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return state


def kernel_fn(data, state):
    """SGLang's baseline, unchanged (this file only measures)."""
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][: data["config"]["n"]], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
