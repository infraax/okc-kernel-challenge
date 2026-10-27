# okc: problem=muse_decode_layer
# sonnet_v4 -- sonnet_v3_s1 (board entry 1.196x: sonnet_v2 + the judge-confirmed Marlin / epilogue knobs of the MoE problem's sonnet_v2b) plus S4, the
# MoE problem's SMALL-M CUDA PATH (its sonnet_v3b: sonnet_v3 judged 1.241x there, bs1 1.79x / bs8 1.47x / bs16 1.35x vs SGLang's Marlin path; v3b's tuned SMALLM_CFG: bs1 1.85x, bs8 1.45x, bs16 1.39x) for
# the layer's experts at <= 16 tokens (decode 1, verify 1 = 8 tokens, decode 16). Marlin + S1's knobs remain for > 16 tokens and as the fallback.
#   S4  Route + GEMM1 (SwiGLU fused, gate/up rows interleaved at repack time) + GEMM2 as three launches: smallm_route (local ids + per-expert token lists,
#       replaces our local-id kernel AND moe_align_block_size), a streaming split-K W4A16 kernel on BF16 mma.sync (weights are the mma A operand, 16 rows x
#       k16; E2M1 x E4M3 scale is exact in BF16; FP32 accumulate; last-CTA fixup adds the split-K partials in slice order, so it is deterministic), the same
#       kernel for GEMM2. It replaces align + Marlin GEMM1 + silu_and_mul + Marlin GEMM2 + the local-id kernel (5 launches -> 3). The layer's fused Triton
#       epilogue is KEPT (v3b's epilogue is the same kernel with the same BLOCK_H table as S1's _marlin_knobs: M <= 1 1024, M <= 8 2048, else 1024), so
#       the only change is upstream of it. Changes against v3b's code: smallm_route reads local_start from the layer's device tensor and applies the layer's
#       wrap rule ((id - local_start) mod 256 < 32), the host passes no Python ints from inputs; the self-check (below) compares against the layer's own
#       Marlin _experts; everything else is v3b's source verbatim (copied in: one file per candidate).
#       prepare() (untimed): repacks the 32 local experts into two tile-word copies beside the Marlin copy (~1.5 GB extra on the H200), builds the CUDA
#       extension with torch load_inline (one nvcc run; v3b's whole judged run took 115 s), and SELF-CHECKS the small-M experts against Marlin on seeded
#       synthetic inputs at M = 1, 8, 16 (rel L2 < 1e-2; v3b measured 6.9e-3 on the judge): if the build, the check or anything else fails it retries once with
#       FP32 products (-DSM_FP32_MUL=1) and otherwise leaves state["smallm"] = None, and Marlin serves every M. Needs ~67 KB dynamic smem per CTA (H200 only; the
#       2060 tests use the old v3 config table through a module override and an NVRTC build with the BF16 mma emulated).
#   Predicted per shape (us; board job 2cb883c4c64c45a2 -> v4; per-kernel table in design/breakdown/muse_decode_layer.md: decode-1 sliding Marlin 34.2 + epilogue 6.2
#   + ~7 us of align / silu / local-id launches in the hidden tail; the MoE problem's own v2b -> v3b deltas are bs1 49.5 -> ~43, bs8 110.8 -> ~98, bs16 152 -> ~135):
#     decode 1  sliding 129 -> ~120 (-9), full 142 -> ~133;   verify 1 sliding 187 -> ~174 (-13), full 201 -> ~188;
#     decode 16 sliding 202 -> ~185 (-17), full 357 -> ~340.  Decode 32 and verify 16 / 32 (32, 128, 256 tokens) are Marlin as before.
#     Load speedups: decode 1 ~1.32 -> ~1.42x, verify 1 ~1.40 -> ~1.51x, decode 16 ~1.26 -> ~1.35x; geo-mean over the 6 loads 1.196 -> ~1.23x (range 1.21-1.25).
#     Honest caveats: the saving scales with the number of active experts (the layer's routing is the reference's generate_input one, as in the MoE problem);
#     only the Marlin-side launches and kernels are replaced, the Triton epilogue (4.9-6.2 us), pooler, router and attention are untouched.
# Everything else is sonnet_v3_s1 (module-level switches S1_MOE_KNOBS = True, S2_FUSED_ROUTING = False, S3_PDL = False: S2 was refuted on the judge, S3 is noise).
# Rules: shapes / config / hardware only decide paths; local_start, lengths, positions, rows, scales are read on the device; no host sync in kernel_fn; the small-M
# scratch (route table, split-K partials, fixup counters) is allocated in prepare() and rewritten before it is read; CUDA-graph capturable; NVFP4 stays NVFP4
# (a bit permutation of the same nibbles and E4M3 bytes), activations BF16, FP32 accumulate, one BF16 rounding of each expert output; every key attended; KV cache FP8.
import functools
import math
import types
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # in the judge's image: SGLang's own modules, as baseline.py imports them
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sgl_kernel.scalar_type import scalar_types
    from torch import nn

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.kvcache.kvcache import store_cache
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
    from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
    from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.muse_spark_layers import (
        MuseSpark1xNormPreservingResidualConnection,
        MuseSpark1xRMSNorm,
        MuseSpark1xRMSNormWithInputScale,
    )
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

    HAVE_SGLANG = True
except ImportError:  # local runs without SGLang: plain PyTorch layer below
    HAVE_SGLANG = False

    def _bf16_gemm_dispatch_impl(x, w, b):
        return x @ w.T


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int
    q_len: int
    tokens: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float
    n_experts: int
    top_k: int
    local_experts: int


class Inputs(TypedDict):
    stream: torch.Tensor
    attn_input: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    local_start: torch.Tensor
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
ROUTER_FP32_MAX_TOKENS = 8       # SGLang's router: FP32 F.linear up to this many tokens, split BF16 GEMMs above
SLICED_UNPOOL_MAX_TOKENS = 8     # up to this many tokens the unpooler runs on this GPU's slice
UNPOOL_SLICE = HIDDEN // 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0
ROUTER_MULT = 0.1767766952966369  # reference ROUTER_MULTIPLIER (FusedMuseSparkRouting's router_output_multiplier)
ROUTE_EPS = 1e-15                 # router_score_division_eps
ROUTER_TRITON_MAX_TOKENS = 256   # director_v1b: Triton router up to here
POOLER_TRITON_MAX_TOKENS = 32    # director_v1b: Triton pooler up to here
EPILOGUE_BLOCK_H = 1024
EPILOGUE_WARPS = 8
EAGER_EPILOGUE_BELOW_TOKENS = 32  # SGLang's post-expert epilogue is eager below this many tokens (its _MIN_TOKENS_FOR_FUSED): we fuse it

# ---- sonnet_v3 step switches (module-level, read at call time; shapes / hardware only, never input values). With a switch off
# the code path is sonnet_v2's.
S1_MOE_KNOBS = True       # M <= 1: Marlin atomic reduce for both GEMMs; M <= 8: epilogue BLOCK_H 2048 (judge-confirmed, sonnet_v2b)
S2_FUSED_ROUTING = False  # router GEMV + sigmoid + bias + top-8 + renorm + local ids in one Triton kernel (T <= F2_MAX_TOKENS)
S3_PDL = False            # programmatic dependent launch on our Triton kernels (sm_90+, Triton with gdc_wait / launch_pdl)
F2_MAX_TOKENS = 32        # the fused routing kernel's token range; above it the v2 router + SGLang routing run
F2_TOKEN_BLOCK = 16       # tokens per CTA row-block (a tl.dot needs M >= 16)
# F2 per token bucket: (BN, BK, SPLIT, stages); 4 warps. CTAs = (256 / BN) * SPLIT * ceil(T / 16); per-CTA bytes BN * 8192 / SPLIT * 4.
_F2_CFG = {8: (64, 128, 16, 4), 32: (32, 64, 8, 4)}
# Marlin / epilogue knobs of S1 by M bucket: (use_atomic_add, use_fp32_reduce, epilogue BLOCK_H); v2's (False, True, 1024) elsewhere.
_S1_KNOBS = ((1, (True, False, 1024)), (8, (False, True, 2048)))


def _probe_pdl_api():
    """True when this Triton has tl.extra.cuda.gdc_wait / gdc_launch_dependents and a launch_pdl option (Triton >= 3.3 era)."""
    try:
        from triton.backends.nvidia.compiler import CUDAOptions
        return (hasattr(tl.extra.cuda, "gdc_wait") and hasattr(tl.extra.cuda, "gdc_launch_dependents")
                and "launch_pdl" in CUDAOptions.__dataclass_fields__)
    except Exception:
        return False


HAVE_PDL_API = _probe_pdl_api()


def _pdl_on(device):
    """S3 is live: switch on, API present, a Hopper-or-newer GPU (griddepcontrol is sm_90+). Hardware / config only."""
    return bool(S3_PDL and HAVE_PDL_API and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 9)


def _pdl_kwargs(device):
    """Launch kwargs of one of our Triton kernels: the PDL constexpr (gdc_wait first, dependents launched after the main loop)
    and the launch attribute. Empty when S3 is off, so the compiled kernels are sonnet_v2's."""
    return {"PDL": True, "launch_pdl": True} if _pdl_on(device) else {}


# =============================================================================================== projections (director_v1b)
@triton.jit
def _gemv_splitk(x_ptr, w_ptr, wlo_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                 HAS_LO: tl.constexpr, ATOMIC: tl.constexpr, SPLIT_X: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        tl.extra.cuda.gdc_wait()  # S3: first statement, before any load (the previous grid's outputs are this kernel's inputs)
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
            # FP32 input x = x_hi + x_lo (both BF16): x . (W_hi + W_lo) = x_hi.W_hi + x_hi.W_lo + x_lo.W_hi (+ x_lo.W_lo, 2^-32).
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
    if ATOMIC:
        tl.atomic_add(optr, acc, mask=om)
    else:
        tl.store(optr, acc.to(out_ptr.dtype.element_ty), mask=om)


# (BN, BK, SPLIT, warps, stages) per (layer, bucket); BM follows from m. Best configs measured on the H200 (C9 v1/v2).
_CFG = {
    ("router", 1): (64, 128, 16, 4, 4),
    ("router", 8): (64, 128, 16, 4, 4),
    ("router", 16): (64, 64, 16, 4, 4),
    ("router", 32): (16, 128, 16, 4, 3),
    ("router", 128): (32, 256, 8, 4, 3),
    ("router", 256): (32, 128, 8, 4, 3),
    ("pooler", 1): (32, 512, 1, 4, 4),
    ("pooler", 8): (32, 512, 1, 4, 4),
    ("pooler", 16): (32, 512, 1, 4, 4),
    ("pooler", 32): (32, 512, 1, 4, 3),
}


def _bucket(m):
    for b in (1, 8, 16, 32, 128, 256):
        if m <= b:
            return b
    return None


def _triton_ok(device):
    """Triton projections need BF16 tl.dot (sm_80+); hardware only, never input values."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8


def _run(x, w, wlo, n, out_dtype, cfg, split_x=False):
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
                       BM=BM, BN=BN, BK=BK, SPLIT=SPLIT, HAS_LO=wlo is not None, ATOMIC=atomic, SPLIT_X=split_x,
                       num_warps=warps, num_stages=stages, **_pdl_kwargs(x.device))
    return acc if acc.dtype == out_dtype else acc.to(out_dtype)


# =============================================================================================== fused routing (S2, F2)
@triton.jit
def _router_f2_kernel(
    x_ptr, hi_ptr, lo_ptr, bias_ptr, start_ptr, part_ptr, done_ptr, ids_ptr, w_ptr, loc_ptr,
    M, stride_x, stride_w,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, SPLIT_X: tl.constexpr,
    MULT: tl.constexpr, EPS: tl.constexpr, NE: tl.constexpr, COUNT: tl.constexpr, K: tl.constexpr,
    PDL: tl.constexpr = False,
):
    """Router logits (split-K GEMV on the BF16 hi + lo weights, FP32 accumulate), then, in the last CTA of each token block,
    sigmoid(MULT * logit) + bias -> exact top-8 -> weights / (sum + EPS) -> this GPU's expert index or -1. Grid
    (NE / BN, SPLIT, ceil(T / BM)). The split partials are summed in split order by one CTA, so the result is deterministic."""
    if PDL:
        tl.extra.cuda.gdc_wait()  # S3: x comes from the previous grid
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    NT: tl.constexpr = NE // BN
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k_per: tl.constexpr = K // SPLIT
    k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        rk = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * stride_x + rk[None, :], mask=rm[:, None] < M, other=0.0)
        w = tl.load(hi_ptr + rn[:, None] * stride_w + rk[None, :])
        if SPLIT_X:
            # FP32 input x = x_hi + x_lo (both 16-bit): x . (W_hi + W_lo) = x_hi.W_hi + x_lo.W_hi + x_hi.W_lo (x_lo.W_lo ~ 2^-32 dropped).
            xh = x.to(hi_ptr.dtype.element_ty)
            xl = (x - xh.to(tl.float32)).to(hi_ptr.dtype.element_ty)
            wl = tl.load(lo_ptr + rn[:, None] * stride_w + rk[None, :])
            acc += tl.dot(xh, tl.trans(w))
            acc += tl.dot(xl, tl.trans(w))
            acc += tl.dot(xh, tl.trans(wl))
        else:
            wl = tl.load(lo_ptr + rn[:, None] * stride_w + rk[None, :])
            xc = x.to(hi_ptr.dtype.element_ty)  # no-op for BF16 x with BF16 weights
            acc += tl.dot(xc, tl.trans(w))
            acc += tl.dot(xc, tl.trans(wl))
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(part_ptr + (pid_k * M + rm)[:, None] * NE + rn[None, :], acc, mask=rm[:, None] < M)
    # Publish the partials, then arrive (CUTLASS's semaphore pattern: barrier, then one release-acquire GPU-scope RMW by one thread).
    tl.debug_barrier()
    ticket = tl.atomic_add(done_ptr + pid_m, 1, sem="acq_rel", scope="gpu")
    if ticket == NT * SPLIT - 1:
        cols = tl.arange(0, NE)
        slot = tl.arange(0, 8)[None, :]
        tt = pid_m * BM + tl.arange(0, BM)
        tmask = tt < M
        logit = tl.zeros((BM, NE), dtype=tl.float32)
        for s in tl.static_range(SPLIT):  # .cg: skip L1 so no stale line of this buffer can be read
            logit += tl.load(part_ptr + (s * M + tt)[:, None] * NE + cols[None, :], mask=tmask[:, None], other=0.0,
                             cache_modifier=".cg")
        sig = tl.sigmoid(logit * MULT)
        work = sig + tl.load(bias_ptr + cols)[None, :]
        ids_t = tl.zeros((BM, 8), dtype=tl.int32)
        w_t = tl.zeros((BM, 8), dtype=tl.float32)
        for r in tl.static_range(8):
            # torch.topk order: highest biased score first; an exact tie goes to the lower expert index.
            top = tl.max(work, 1)
            idx = tl.min(tl.where(work == top[:, None], cols[None, :], NE), 1)
            hit = cols[None, :] == idx[:, None]
            ids_t = tl.where(slot == r, idx[:, None], ids_t)
            w_t = tl.where(slot == r, tl.sum(tl.where(hit, sig, 0.0), 1)[:, None], w_t)
            work = tl.where(hit, float("-inf"), work)
        w_t = w_t / (tl.sum(w_t, 1)[:, None] + EPS)
        rel = (ids_t - tl.load(start_ptr)) & (NE - 1)
        o = tt[:, None] * 8 + slot
        om = tmask[:, None] & (slot < 8)
        tl.store(ids_ptr + o, ids_t, mask=om)
        tl.store(w_ptr + o, w_t, mask=om)
        tl.store(loc_ptr + o, tl.where(rel < COUNT, rel, -1), mask=om)
        tl.store(done_ptr + pid_m, 0)  # ready for the next call (every CTA of this block has arrived)


def _f2_cfg(tokens):
    for limit in sorted(_F2_CFG):
        if tokens <= limit:
            return _F2_CFG[limit]
    return None


def fused_router_f2(x, router_hi, router_lo, router_bias, local_start, done):
    """Route 1 <= T <= F2_MAX_TOKENS tokens in one launch. Graph-safe: grid and buffers depend only on shapes.

    x: [T, 8192] FP32 (T <= 8: the boundary kernel's unrounded hidden, split hi + lo in registers) or BF16.
    router_hi / router_lo: [256, 8192] 16-bit (BF16 on the judge's GPU), hi = round(router), lo = round(router - hi).
    router_bias [256] FP32; local_start int32 [1] (device); done int32 [>= ceil(T / 16)] zeros, left zero by the kernel.
    Returns (topk_ids int32 [T, 8] global, topk_weights FP32 [T, 8] = sigmoid / (sum + 1e-15), local_ids int32 [T, 8])."""
    t, k = x.shape
    bn, bk, split, stages = _f2_cfg(t)
    part = torch.empty((split, t, N_EXPERTS), dtype=torch.float32, device=x.device)
    ids = torch.empty((t, TOP_K), dtype=torch.int32, device=x.device)
    weights = torch.empty((t, TOP_K), dtype=torch.float32, device=x.device)
    local = torch.empty((t, TOP_K), dtype=torch.int32, device=x.device)
    _router_f2_kernel[(N_EXPERTS // bn, split, triton.cdiv(t, F2_TOKEN_BLOCK))](
        x, router_hi, router_lo, router_bias, local_start, part, done, ids, weights, local,
        t, x.stride(0), router_hi.stride(0),
        BM=F2_TOKEN_BLOCK, BN=bn, BK=bk, SPLIT=split, SPLIT_X=x.dtype == torch.float32,
        MULT=ROUTER_MULT, EPS=ROUTE_EPS, NE=N_EXPERTS, COUNT=LOCAL_EXPERTS, K=k,
        num_warps=4, num_stages=stages, **_pdl_kwargs(x.device))
    return ids, weights, local


# =============================================================================================== attention (sonnet_v2)
BLOCK_N = 128           # keys per inner step
MAX_ROWS = 128          # L * (Hq / Hkv) rows per KV head the kernel supports (verify blocks: 8 * 8 = 64)
SM_COUNT = 132          # cost model only (H200); the plan never reads the device


# --------------------------------------------------------------------------------------------- FP8 helpers
@triton.jit
def _decode_e4m3(u):
    """uint8 E4M3FN codes -> fp32 (software path for GPUs without FP8 conversions, e.g. sm_75)."""
    u = u.to(tl.int32)
    e = (u >> 3) & 15
    m = u & 7
    normal = ((e + 120) << 23) | (m << 20)
    val = tl.where(e > 0, normal.to(tl.float32, bitcast=True), m.to(tl.float32) * 0.001953125)
    return tl.where((u & 128) != 0, -val, val)


@triton.jit
def _encode_e4m3(x):
    """fp32 -> uint8 E4M3FN codes, round-to-nearest-even, saturating at 448 (software path)."""
    sign = tl.where(x < 0, 128, 0)
    a = tl.minimum(tl.abs(x), 448.0)
    bits = a.to(tl.int32, bitcast=True)
    normal = ((bits + 0x7FFFF + ((bits >> 20) & 1)) >> 20) - 960
    sub = ((a * 512.0 + 12582912.0) - 12582912.0).to(tl.int32)
    code = tl.where(a < 0.015625, sub, normal)
    return (code | sign).to(tl.uint8)


# --------------------------------------------------------------------------------------------- RoPE helpers
@triton.jit
def _sincos(x):
    """(sin x, cos x) for fp32 |x| < ~4e6: round to a multiple of pi/2 with three FMA steps, Cephes polynomials on the
    remainder, quadrant by select. About 40 instructions per value and no slow path (libdevice's is ~1 KB of code)."""
    nf = (x * 0.6366197723675814 + 12582912.0) - 12582912.0
    r = tl.fma(nf, -1.5703125, x)
    r = tl.fma(nf, -4.837512969970703e-4, r)
    r = tl.fma(nf, -7.549789948768648e-8, r)
    z = r * r
    sr = ((-1.9515295891e-4 * z + 8.3321608736e-3) * z - 1.6666654611e-1) * z * r + r
    cr = ((2.443315711809948e-5 * z - 1.388731625493765e-3) * z + 4.166664568298827e-2) * z * z - 0.5 * z + 1.0
    q = nf.to(tl.int32) & 3
    swap = (q & 1) != 0
    s = tl.where(swap, cr, sr)
    c = tl.where(swap, sr, cr)
    s = tl.where((q & 2) != 0, -s, s)
    c = tl.where(((q + 1) & 2) != 0, -c, c)
    return s, c


@triton.jit
def _rope_tables(pos, log2_theta, d):
    """sin, cos of pos[r] * theta^(-2i/64), i = d // 2: fp32 [R, 64]."""
    inv_freq = tl.exp2(-((d // 2) * 2).to(tl.float32) * (log2_theta / 64.0))
    return _sincos(pos.to(tl.float32)[:, None] * inv_freq)


@triton.jit
def _load_norm_rope(base, ok, d, pos, scale, log2_theta, ROPE: tl.constexpr):
    """rows of 64 bf16 at base+d -> RMSNorm (eps 1e-5), x scale, interleaved RoPE if ROPE. fp32 [R, 64]."""
    x = tl.load(base + d, mask=ok, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / 64.0 + 1e-5)
    xn = x * inv[:, None]
    if ROPE:
        xp = tl.load(base + (d ^ 1), mask=ok, other=0.0).to(tl.float32) * inv[:, None]
        sin, cos = _rope_tables(pos, log2_theta, d)
        sgn = tl.where((d & 1) == 0, -1.0, 1.0)
        xn = xn * cos + sgn * xp * sin
    return xn * scale


@triton.jit
def _prep_q(qkv, positions, b, h, qk_factor, log2_theta,
            L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr,
            ROPE: tl.constexpr, BL: tl.constexpr, M: tl.constexpr, QK8: tl.constexpr, QK8_HW: tl.constexpr,
            CDTYPE: tl.constexpr):
    """The M = BL * GP query rows of one KV head (row r = token r // GP, head h*G + r % GP): RMSNorm, qk_factor, RoPE (tables
    computed once per token and broadcast over the group). QK8: rounded to FP8 E4M3 (round-to-nearest-even; software when
    QK8_HW is off), else CDTYPE. The logit scale is applied after the dot."""
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    d = tl.arange(0, 64)[None, :]
    r = tl.arange(0, M)
    j = r // GP
    rvalid = (j < L) & ((r % GP) < G)
    base = qkv + (b * L + j)[:, None] * QKV + (h * G + r % GP)[:, None] * 64
    x = tl.load(base + d, mask=rvalid[:, None], other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / 64.0 + 1e-5)
    xn = x * inv[:, None]
    if ROPE:
        xp = tl.load(base + (d ^ 1), mask=rvalid[:, None], other=0.0).to(tl.float32) * inv[:, None]
        jj = tl.arange(0, BL)
        pos = tl.load(positions + b * L + jj, mask=jj < L, other=0)
        sin, cos = _rope_tables(pos, log2_theta, d)
        sin = tl.reshape(tl.broadcast_to(sin[:, None, :], [BL, GP, 64]), [M, 64])
        cos = tl.reshape(tl.broadcast_to(cos[:, None, :], [BL, GP, 64]), [M, 64])
        sgn = tl.where((d & 1) == 0, -1.0, 1.0)
        xn = xn * cos + sgn * xp * sin
    xn = xn * qk_factor
    if QK8:
        if QK8_HW:
            q = xn.to(tl.float8e4nv)
        else:
            q = _encode_e4m3(xn).to(tl.float8e4nv, bitcast=True)
    else:
        q = xn.to(CDTYPE)
    return q, rvalid, j


@triton.jit
def _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE: tl.constexpr):
    """Online softmax step on raw logits sc (times c = sm_scale * k_scale * log2(e) gives the log2-domain logits)."""
    m_new = tl.maximum(m_i, tl.max(sc, 1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp2(sc * c - (m_use * c)[:, None])
    alpha = tl.exp2((m_i - m_use) * c)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = tl.dot(p.to(CDTYPE), vc, acc * alpha[:, None])
    return m_new, l_i, acc


# --------------------------------------------------------------------------------------------- kernel 1
@triton.jit
def _attn_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l,
    sm_scale, qk_factor, log2_theta, pt_stride, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    WINDOW: tl.constexpr, ROPE: tl.constexpr, EDGE: tl.constexpr, NATIVE_LOAD: tl.constexpr,
    NATIVE_STORE: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr,
    BL: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr, PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()  # S3: qkv comes from the previous grid
    s = tl.program_id(0)          # 0 .. NSPLIT-1: cached keys of one split; NSPLIT: the block's own keys
    h = tl.program_id(1)
    b = tl.program_id(2)
    d = tl.arange(0, 64)[None, :]
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64

    seq = tl.load(seq_lens + b).to(tl.int32)           # cached tokens; the block starts at this position
    lo = tl.maximum(seq - WINDOW + 1, 0) if WINDOW else 0
    start = lo + s * split_len
    end = tl.minimum(seq, start + split_len)
    if (s == NSPLIT) | (start < end):
        ks = tl.load(k_scale_ptr)
        c = sm_scale * ks * 1.4426950408889634        # logits in the log2 domain
        q, rvalid, j = _prep_q(qkv, positions, b, h, qk_factor, log2_theta, L, G, GP, HQ, HKV, ROPE, BL, M, QK8, NATIVE_STORE, CDTYPE)
        m_i = tl.full([M], float("-inf"), tl.float32)
        l_i = tl.zeros([M], tl.float32)
        acc = tl.zeros([M, 64], tl.float32)

        if s < NSPLIT:
            # ---- cached keys [start, end)
            ar = tl.arange(0, BN)
            nt = (end - start + BN - 1) // BN
            prow = page_table + b * pt_stride
            for i in range(nt):
                t = start + i * BN + ar
                tm = t < end
                pg = tl.load(prow + t // PAGE, mask=tm, other=0)
                off = (pg.to(tl.int64) * PAGE + t % PAGE) * (HKV * 64) + h * 64
                ku = tl.load(k_cache + off[:, None] + d, mask=tm[:, None], other=0, eviction_policy="evict_first")
                vu = tl.load(v_cache + off[:, None] + d, mask=tm[:, None], other=0, eviction_policy="evict_first")
                if NATIVE_LOAD:
                    vc = vu.to(tl.float8e4nv, bitcast=True).to(CDTYPE)
                else:
                    vc = _decode_e4m3(vu).to(CDTYPE)
                if QK8:
                    sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True)))
                elif NATIVE_LOAD:
                    sc = tl.dot(q, tl.trans(ku.to(tl.float8e4nv, bitcast=True).to(CDTYPE)))
                else:
                    sc = tl.dot(q, tl.trans(_decode_e4m3(ku).to(CDTYPE)))
                need_mask = start + (i + 1) * BN > end
                if EDGE:
                    need_mask = need_mask | ((i == 0) & (s == 0))
                if need_mask:
                    ok = tm[None, :]
                    if WINDOW:
                        ok = ok & (t[None, :] >= (seq + j - WINDOW + 1)[:, None])
                    sc = tl.where(ok, sc, float("-inf"))
                m_i, l_i, acc = _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE)
        else:
            # ---- the block's own keys / values: norm, RoPE, FP8 round + cache write, causal tile from registers
            vs = tl.load(v_scale_ptr)
            ti = tl.arange(0, BT)
            tv = ti < L
            tpos = tl.load(positions + b * L + ti, mask=tv, other=0)
            tbase = qkv + (b * L + ti)[:, None] * QKV
            kn = _load_norm_rope(tbase + HQ * 64 + h * 64, tv[:, None], d, tpos, 1.0, log2_theta, ROPE)
            vn = tl.load(tbase + (HQ + HKV) * 64 + h * 64 + d, mask=tv[:, None], other=0.0).to(tl.float32)
            slot = tl.load(cache_loc + b * L + ti, mask=tv, other=0)
            coff = slot[:, None] * (HKV * 64) + h * 64 + d
            if NATIVE_STORE:
                k8 = (kn / ks).to(tl.float8e4nv)
                v8 = (vn / vs).to(tl.float8e4nv)
                tl.store(k_cache + coff, k8.to(tl.uint8, bitcast=True), mask=tv[:, None])
                tl.store(v_cache + coff, v8.to(tl.uint8, bitcast=True), mask=tv[:, None])
                kt = k8 if QK8 else k8.to(CDTYPE)
                vt = v8.to(CDTYPE)
            else:
                ku8 = _encode_e4m3(kn / ks)
                vu8 = _encode_e4m3(vn / vs)
                tl.store(k_cache + coff, ku8, mask=tv[:, None])
                tl.store(v_cache + coff, vu8, mask=tv[:, None])
                kt = ku8.to(tl.float8e4nv, bitcast=True) if QK8 else _decode_e4m3(ku8).to(CDTYPE)
                vt = _decode_e4m3(vu8).to(CDTYPE)
            sc = tl.dot(q, tl.trans(kt))
            ok = tv[None, :] & (ti[None, :] <= j[:, None])
            if WINDOW:
                ok = ok & (ti[None, :] > j[:, None] - WINDOW)
            sc = tl.where(ok, sc, float("-inf"))
            m_i, l_i, acc = _softmax_step(sc, c, m_i, l_i, acc, vt, CDTYPE)

        if PDL:
            tl.extra.cuda.gdc_launch_dependents()  # S3: the reduce kernel may be scheduled while the partials are written
        r = tl.arange(0, M)
        pidx = ((b * HKV + h) * (NSPLIT + 1) + s) * M + r
        tl.store(part_m + pidx, m_i * c, mask=rvalid)
        tl.store(part_l + pidx, l_i, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, acc, mask=rvalid[:, None])


# --------------------------------------------------------------------------------------------- kernel 2
@triton.jit
def _reduce_kernel(
    part_o, part_m, part_l, qkv, seq_lens, v_scale_ptr, out, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, WINDOW: tl.constexpr,
    M: tl.constexpr, BS: tl.constexpr, PDL: tl.constexpr = False,
):
    if PDL:
        tl.extra.cuda.gdc_wait()  # S3: the partials come from the attention grid
    t = tl.program_id(0)        # b * L + j
    hq = tl.program_id(1)
    b = t // L
    j = t % L
    h = hq // G
    row = j * GP + hq % G
    d = tl.arange(0, 64)
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64

    seq = tl.load(seq_lens + b).to(tl.int32)
    lo = tl.maximum(seq - WINDOW + 1, 0) if WINDOW else 0
    n_active = (seq - lo + split_len - 1) // split_len      # cached splits with keys; the block's own one follows them
    base = (b * HKV + h) * (NSPLIT + 1) * M + row
    gate = tl.load(qkv + t * QKV + (HQ + 2 * HKV) * 64 + hq * 64 + d).to(tl.float32)

    Mx = tl.full([1], float("-inf"), tl.float32)
    Lsum = tl.zeros([1], tl.float32)
    O = tl.zeros([64], tl.float32)
    for s0 in range(0, n_active + 1, BS):
        si = s0 + tl.arange(0, BS)
        sv = si <= n_active
        pidx = base + tl.where(si == n_active, NSPLIT, si) * M
        ms = tl.load(part_m + pidx, mask=sv, other=float("-inf"))
        ls = tl.load(part_l + pidx, mask=sv, other=0.0)
        os_ = tl.load(part_o + pidx[:, None] * 64 + d[None, :], mask=sv[:, None], other=0.0)
        m_new = tl.maximum(Mx, tl.max(ms, 0))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(Mx - m_use)
        w = tl.exp2(ms - m_use)
        Lsum = Lsum * alpha + tl.sum(w * ls, 0)
        O = O * alpha + tl.sum(w[:, None] * os_, 0)
        Mx = m_new

    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    o = O / Lsum * tl.load(v_scale_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
    o = o * (1.0 / (1.0 + tl.exp(-gate)))
    tl.store(out + t * (HQ * 64) + hq * 64 + d, o.to(tl.bfloat16))


# --------------------------------------------------------------------------------------------- host
@functools.lru_cache(maxsize=None)
def _device_mode(device_type: str, index: int):
    """(V / fallback compute dtype, FP8 loads native, FP8 stores / q rounding native, q.K in FP8): hardware only, never input
    values. sm_89+ converts FP8 in hardware and has FP8 tensor-core MMAs (q rounded to E4M3, K used as cached, FP32
    accumulation). Elsewhere (the sm_75 dev GPU) FP8 is decoded / encoded in software and q.K runs in FP16 (exact E4M3
    products); under the Triton CPU interpreter loads and FP8 dots are exact but its fp32->e4m3 cast truncates near binade
    edges, so q rounding and stores are software there."""
    if device_type != "cuda":
        return tl.float32, True, False, True
    major, minor = torch.cuda.get_device_capability(index)
    native = (major, minor) >= (8, 9)
    return tl.float16, native, native, native    # fp16 mma operands: exact for E4M3, 3 more mantissa bits than BF16


def _cdiv(a, b):
    return -(-a // b)


def _plan(batch, hkv, span, rows, bn, occ):
    """Split length for a shape (shapes only, never values). Small cost model in microseconds: a program's time is a fixed
    chain (seq_lens -> page index -> K/V rows -> write) plus ~0.8 us per tile; the whole grid can't finish before the bytes
    are moved at ~3 TB/s (64-byte random rows) scaled by how full the machine is; the last programs to start finish up to
    half a program after the rest (tail: ~0.4 of a program's share of the bytes); every split adds a partial (rows x 64 fp32, written then read back) and reduce-loop
    iterations."""
    pairs = batch * hkv
    slots = SM_COUNT * occ
    best = None
    for mult in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048):
        split_len = mult * bn
        ns = _cdiv(span, split_len)
        progs = pairs * (ns + 1)
        waves = progs / slots
        tiles = split_len / bn
        t_lat = max(1.0, waves) * (3.0 + tiles * 0.8)
        util = min(1.0, waves) if waves < 1 else waves / math.ceil(waves)
        t_bw = pairs * span * 128 / 3.0e6
        per_slot = pairs * span / bn / slots                       # tiles each slot streams in total
        t_tail = 0.4 * t_bw * min(1.0, tiles / max(per_slot, 1e-9))
        t_part = progs * rows * 264 * 2 / 8.0e6
        t_red = 1.5 + _cdiv(ns + 1, 128) * 0.6
        est = max(t_lat, t_bw / util) + t_tail + t_part + t_red
        if best is None or est < best[0] * 0.995:
            best = (est, split_len, ns)
        if ns == 1:
            break
    return best[1], best[2]


def _attn_geometry(cfg):
    """(group, padded group, padded q_len, rows) of a layer's heads / q_len, or None when the Triton kernels don't cover it
    (config only)."""
    hq, hkv, d, length = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["q_len"]
    if d != 64 or hq % hkv:
        return None
    group = hq // hkv
    gp = triton.next_power_of_2(group)
    bl = max(triton.next_power_of_2(length), 16 // gp)
    if bl * gp > MAX_ROWS or length > BLOCK_N:
        return None
    return group, gp, bl, bl * gp


def _attention_decode(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page, window = cfg["page_size"], cfg["window"]
    geometry = _attn_geometry(cfg)
    if geometry is None:
        raise ValueError(f"unsupported config: head_dim {d}, {hq} q heads / {hkv} kv heads, q_len {length}")
    group, gp, bl, rows = geometry
    block_t = max(16, triton.next_power_of_2(length))
    qkv = data["qkv"]
    dev = qkv.device
    cdtype, native_load, native_store, qk8 = _device_mode(dev.type, dev.index or 0)

    max_pages = data["page_table"].shape[1]
    cap = max_pages * page                       # most tokens a request's page table can hold
    span = max(1, min(window - 1, cap) if window else cap)
    occ = 4 if rows <= 16 else 3
    split_len, nsplit = _plan(batch, hkv, span, rows, BLOCK_N, occ)

    part_o = torch.empty((batch, hkv, nsplit + 1, rows, 64), dtype=torch.float32, device=dev)
    part_m = torch.empty((batch, hkv, nsplit + 1, rows), dtype=torch.float32, device=dev)
    part_l = torch.empty_like(part_m)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    k_cache = data["k_cache"].view(torch.uint8)
    v_cache = data["v_cache"].view(torch.uint8)

    _attn_kernel[(nsplit + 1, hkv, batch)](
        qkv, data["positions"], data["cache_loc"], data["seq_lens"], data["page_table"], k_cache, v_cache,
        data["k_scale"], data["v_scale"], part_o, part_m, part_l,
        cfg["sm_scale"], cfg["qk_scale_factor"], math.log2(cfg["rope_theta"]) if cfg["rope_theta"] else 0.0,
        data["page_table"].stride(0), split_len, nsplit,
        L=length, G=group, GP=gp, HQ=hq, HKV=hkv, PAGE=page, WINDOW=window, ROPE=bool(cfg["rope_theta"]),
        EDGE=bool(window) and length > 1, NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype,
        BL=bl, M=rows, BN=BLOCK_N, BT=block_t,
        num_warps=4, num_stages=3 if split_len >= 3 * BLOCK_N else 1, **_pdl_kwargs(dev),
    )
    _reduce_kernel[(batch * length, hq)](
        part_o, part_m, part_l, qkv, data["seq_lens"], data["v_scale"], out, split_len, nsplit,
        L=length, G=group, GP=gp, HQ=hq, HKV=hkv, WINDOW=window, M=rows,
        BS=min(128, triton.next_power_of_2(nsplit + 1)), num_warps=4, num_stages=1, **_pdl_kwargs(dev),
    )
    return out


# =============================================================================================== dispatch
# Shapes, config and hardware only (never input values): which implementation runs each stage.
def _router_path(tokens, triton_proj):
    """"triton" = split-K GEMV on the BF16 hi+lo weights; at <= 8 tokens (ROUTER_FP32_MAX_TOKENS) its input is the baseline's
    FP32 hidden, split into BF16 hi+lo inside the kernel. Without Triton: SGLang's FP32 F.linear / two BF16 GEMMs."""
    if triton_proj and tokens <= ROUTER_TRITON_MAX_TOKENS:
        return "triton"
    return "fp32" if tokens <= ROUTER_FP32_MAX_TOKENS else "split"


def _f2_weights(tokens, device, state):
    """(hi, lo) operands of the fused routing kernel, or None when S2 is off / the shape is out of range / no suitable GPU.
    Hopper-class GPUs: the BF16 hi + lo pair (the rules' FP32-class router). Pre-sm_80 dev GPUs (no BF16 tl.dot): the FP16 hi + lo
    pair prepare() built there (22 mantissa bits in total; a local stand-in so the integration is testable, never built on the judge)."""
    if not S2_FUSED_ROUTING or not 1 <= tokens <= F2_MAX_TOKENS or device.type != "cuda":
        return None
    if _triton_ok(device):
        return state["router_hi"], state["router_lo"]
    if "router_hi16" in state:
        return state["router_hi16"], state["router_lo16"]
    return None


def _epilogue_path(tokens):
    """"triton" = our fused post-expert epilogue (SGLang is eager there); "sglang" = its own fused kernel (>= 32 tokens)."""
    return "triton" if tokens < EAGER_EPILOGUE_BELOW_TOKENS else "sglang"


def _pooler_path(tokens, triton_proj):
    return "triton" if triton_proj and tokens <= POOLER_TRITON_MAX_TOKENS else "cublas"


def _attention_path(cfg, is_cuda):
    """"triton" = sonnet_v2's fused kernels for every q_len they cover (decode 1, verify 8, sliding and full);
    "reference" = SGLang's FA3 chain (or the torch layer) for other head geometries and off-GPU runs."""
    return "triton" if is_cuda and _attn_geometry(cfg) is not None else "reference"


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _router_logits(x, router_input, state, path):
    """FP32 logits [T, 256]. router_input is the FP32 hidden the boundary kernel writes at <= 8 tokens (as in the baseline);
    when present, the Triton router splits it into BF16 hi + lo and keeps FP32-class precision."""
    if path == "triton":
        cfg = _CFG[("router", _bucket(x.shape[0]))]
        if router_input is not None:
            return _run(router_input.contiguous(), state["router_hi"], state["router_lo"], N_EXPERTS, torch.float32, cfg,
                        split_x=True)
        return _run(x.contiguous(), state["router_hi"], state["router_lo"], N_EXPERTS, torch.float32, cfg)
    if router_input is not None:
        return F.linear(router_input, state["router"])
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] < 8:  # CPU, sm_75 dev GPU: no BF16 GEMM with FP32 output
        xf = x.float()
        return xf @ state["router_hi_t"].float() + xf @ state["router_lo_t"].float()
    logits = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    logits += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return logits


def _pooler(x, state, path):
    if path == "triton":
        return _run(x.contiguous(), state["dense"]["pooler"], None, D_EXPERT, torch.bfloat16,
                    _CFG[("pooler", _bucket(x.shape[0]))])
    return _gemm(x, state["dense"]["pooler"])


def _split_router_weights(router, bias):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    out = {"router": router.contiguous(), "router_hi": hi.contiguous(), "router_lo": lo.contiguous(),
           "router_hi_t": hi.t().contiguous(), "router_lo_t": lo.t().contiguous(),
           "router_bias": bias.float().contiguous(),
           # arrival counters of the fused routing kernel (one per 16-token block); the kernel leaves them at zero
           "f2_done": torch.zeros(triton.cdiv(F2_MAX_TOKENS, F2_TOKEN_BLOCK), dtype=torch.int32, device=router.device)}
    if router.is_cuda and torch.cuda.get_device_capability(router.device)[0] < 8:  # dev GPU only: FP16 stand-in for BF16 tl.dot
        hi16 = router.half()
        out["router_hi16"] = hi16.contiguous()
        out["router_lo16"] = (router - hi16.float()).half().contiguous()
    return out


# =============================================================================================== SGLang layer (baseline.py)
def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def _prepare_sglang(weights):
    device = weights["qkv_gate"].device
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        **_split_router_weights(weights["router"], weights["router_bias"]),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    del freqs
    torch.cuda.synchronize()
    return state


@triton.jit
def _local_ids_kernel(ids_ptr, start_ptr, out_ptr, n, NE: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr,
                      PDL: tl.constexpr = False):
    """Global expert id -> this GPU's expert index, or -1: ((id - local_start) mod NE) if < COUNT. One launch for what the
    baseline does as a modulo (two eager ops) and an expert_map[...] gather. local_start is read on the device."""
    if PDL:
        tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    e = tl.load(ids_ptr + offs, mask=mask, other=0)
    rel = (e - tl.load(start_ptr)) & (NE - 1)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + offs, tl.where(rel < COUNT, rel, -1), mask=mask)


@triton.jit
def _epilogue_kernel(
    y_ptr, w_ptr, s_ptr, ids_ptr, out_ptr, eps,
    TOPK: tl.constexpr, KP: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr, PDL: tl.constexpr = False,
):
    """out[t] = sum_k live_k * s[t,k] * rmsnorm(y[t,k] * w) over the token's TOPK slots; FP32 throughout, one rounding at the
    store (SGLang's eager branch of fused_post_expert_norm_weighted_sum, in one kernel). Slots of experts this GPU does not
    hold (id < 0) are never loaded: Marlin leaves those rows undefined."""
    if PDL:
        tl.extra.cuda.gdc_wait()  # S3: y comes from the previous grid (Marlin)
    tok = tl.program_id(0)
    NCHUNK: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
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

    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK_H + offs0
        hm = offs < H
        w = tl.load(w_ptr + offs, mask=hm, other=0.0).to(tl.float32)
        x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None] & hm[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(x * coef[:, None], axis=0) * w
        tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=hm)


def local_ids_triton(topk_ids, local_start):
    """int32 [T, 8] local expert indices (-1: not on this GPU) from the global ids and the device tensor local_start."""
    ids = topk_ids.contiguous()
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    n = ids.numel()
    _local_ids_kernel[(triton.cdiv(n, 1024),)](ids, local_start, out, n, NE=N_EXPERTS, COUNT=LOCAL_EXPERTS, BLOCK=1024,
                                               num_warps=4, **_pdl_kwargs(ids.device))
    return out


def epilogue_triton(expert_out, topk_scores, slot_ids, norm_w, eps=POST_NORM_EPS, block_h=EPILOGUE_BLOCK_H,
                    warps=EPILOGUE_WARPS):
    """expert_out [M, K, H] BF16 (Marlin), topk_scores [M, K] FP32, slot_ids [M, K] int32 (< 0: not local), norm_w [H]."""
    y = expert_out.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = min(block_h, triton.next_power_of_2(hidden))
    _epilogue_kernel[(tokens,)](
        y, norm_w, topk_scores.contiguous(), slot_ids.contiguous(), out, eps,
        TOPK=topk, KP=triton.next_power_of_2(topk), H=hidden, BLOCK_H=block_h, num_warps=warps, **_pdl_kwargs(y.device))
    return out


def _marlin_knobs(num_tokens):
    """(use_atomic_add, use_fp32_reduce, epilogue BLOCK_H) of one Marlin GEMM pair + our epilogue, by token count. S1 = the judge-confirmed
    knobs of kernels/muse_moe_w4a16/sonnet_v2b.py (M <= 1: both GEMMs reduce by atomic add into C, which Marlin zero-initialises;
    M <= 8: epilogue BLOCK_H 2048); every other M keeps SGLang's FP32 reduce and v2's tiling."""
    if S1_MOE_KNOBS:
        for limit, knobs in _S1_KNOBS:
            if num_tokens <= limit:
                return knobs
    return False, True, EPILOGUE_BLOCK_H


# ============================================================================================ SMALLM BEGIN
# Small-M (M <= 16 tokens) streaming split-K W4A16 path: CUDA source (JIT-compiled by torch load_inline in the judge image, which has nvcc),
# the weight repack, the host-side plumbing. Layout / numerics are explained in the header of the muse_moe_w4a16 problem's sonnet_v3b.
SM_MAX_M = 16        # tokens handled by the small-M path
SM_TABW = 20         # route-table row: [expert, m_e, slot x 16, pad, pad]
SM_NEXP = 32         # local experts of the bank (ep8)
SM_TAB_INTS = (SM_NEXP + 1) * SM_TABW   # + one control row: [n_active, ...]
SM_BITS_SHIFT = 30   # E2M1 bit pattern x E4M3 scale lut: operands are true_value * 2**-SM_BITS_SHIFT, undone in the FP32 epilogue
SM_LUT_EXP = 96      # lut[b] = e4m3(b) * 2**96 as bf16 x 2;  raw E2M1 bf16 pattern = true * 2**-126  =>  product = true * 2**-30
SM_ROT = (0, 6, 12)  # word bit of operand reg r (r < 3) = rotr(word, SM_ROT[r]) & 0x81C081C0; reg 3 = ((w << 6) & 0x80C080C0) | ((w >> 2) & 0x01000100)
SM_NIB_TARGET = {3: 15, 2: 8, 1: 7, 0: 6}          # nibble bit (s e1 e0 m) -> bf16 bit of the low half (the high half is +16)
SM_REG3_SOURCE = {3: 9, 2: 10, 1: 1, 0: 0}         # nibble bit -> word bit (low half) of operand reg 3

_SMALLM_DEVICE_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef unsigned long long u64;

#define SM_D 4              // k64 iterations per register prefetch batch
#define SM_TABW 20
#define SM_MAXM 16
#define SM_NEXP 32
#define FULLMASK 0xffffffffu

// Programmatic dependent launch (sm_90): a dependent kernel may be scheduled while its predecessor finishes and runs its independent prologue
// (weight prefetch) before griddepcontrol.wait; without the launch attribute (or before sm_90) both are no-ops.
__device__ __forceinline__ void pdl_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ u32 smem_u32(const void* p) { return (u32)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldg_stream4(uint4& r, const void* p) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
}
__device__ __forceinline__ void ldg_stream2(uint2& r, const void* p) {
    asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p));
}
__device__ __forceinline__ uint4 ldcg4(const void* p) {
    uint4 r;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
    return r;
}

__device__ __forceinline__ u16 f2bf(float x) {
#if __CUDA_ARCH__ >= 800
    u16 r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
#else
    u32 u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (u16)0x7fc0u;
    return (u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
#endif
}

#ifndef SM_FP32_MUL
#define SM_FP32_MUL 0   // 1: form the products in FP32 (denormal-proof fallback, selected by prepare() if the bf16x2 multiply were to flush denormals)
#endif

__device__ __forceinline__ u32 pack_bf16x2(float lo, float hi) {
#if __CUDA_ARCH__ >= 800
    u32 d;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(lo), "f"(hi));
    return d;
#else
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);  // only used for exact values
#endif
}

// bf16x2 product (both operands exact in bf16, result exact in bf16; the raw E2M1 operand is a BF16 subnormal for the codes 0 and 0.5).
__device__ __forceinline__ u32 mul_bf16x2(u32 a, u32 b) {
#if __CUDA_ARCH__ >= 800 && !SM_FP32_MUL
    u32 d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
#else
    return pack_bf16x2(__uint_as_float(a << 16) * __uint_as_float(b << 16), __uint_as_float(a & 0xffff0000u) * __uint_as_float(b & 0xffff0000u));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(u32 (&r)[4], u32 addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

__device__ __forceinline__ float bflo(u32 p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float bfhi(u32 p) { return __uint_as_float(p & 0xffff0000u); }

// D[16 x 8] += A[16 x 16] * B[16 x 8], bf16 operands, FP32 accumulate. A: row-major regs a0..a3, B: b0, b1 (PTX m16n8k16 fragment layout).
__device__ __forceinline__ void mma16816(float (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    // sm_75 emulation of the same fragment layout with shuffles (development card without BF16 tensor cores).
    const int lane = threadIdx.x & 31, gid = lane >> 2, t = lane & 3;
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int sa = gid * 4 + p, sb0 = (2 * t) * 4 + p, sb1 = (2 * t + 1) * 4 + p;
        u32 a0 = __shfl_sync(FULLMASK, a[0], sa), a1 = __shfl_sync(FULLMASK, a[1], sa);
        u32 a2 = __shfl_sync(FULLMASK, a[2], sa), a3 = __shfl_sync(FULLMASK, a[3], sa);
        u32 x00 = __shfl_sync(FULLMASK, b0, sb0), x01 = __shfl_sync(FULLMASK, b1, sb0);
        u32 x10 = __shfl_sync(FULLMASK, b0, sb1), x11 = __shfl_sync(FULLMASK, b1, sb1);
        s0 += bflo(a0) * bflo(x00) + bfhi(a0) * bfhi(x00) + bflo(a2) * bflo(x01) + bfhi(a2) * bfhi(x01);
        s1 += bflo(a0) * bflo(x10) + bfhi(a0) * bfhi(x10) + bflo(a2) * bflo(x11) + bfhi(a2) * bfhi(x11);
        s2 += bflo(a1) * bflo(x00) + bfhi(a1) * bfhi(x00) + bflo(a3) * bflo(x01) + bfhi(a3) * bfhi(x01);
        s3 += bflo(a1) * bflo(x10) + bfhi(a1) * bfhi(x10) + bflo(a3) * bflo(x11) + bfhi(a3) * bfhi(x11);
    }
    c[0] += s0; c[1] += s1; c[2] += s2; c[3] += s3;
#endif
}

// One 32-bit word = the 8 E2M1 nibbles one lane needs for a 16-row x 16-k weight tile (A-fragment regs a0..a3, two nibbles each),
// bit-scattered so that three regs need one rotate + one AND and the fourth two shifts + two logic ops. Each result is the BF16 bit pattern
// of (E2M1 value * 2**-126), exact; the scale lut multiplies it into a normal BF16 number.
__device__ __forceinline__ void unpack_word(u32 w, u32 (&r)[4]) {
    r[0] = w & 0x81C081C0u;
    r[1] = __funnelshift_r(w, w, 6) & 0x81C081C0u;
    r[2] = __funnelshift_r(w, w, 12) & 0x81C081C0u;
    r[3] = ((w << 6) & 0x80C080C0u) | ((w >> 2) & 0x01000100u);
}

__device__ __forceinline__ void prefetch_batch(uint4 (&w)[SM_D], uint2 (&s)[SM_D], const uint4* wp, const uint2* sp) {
#pragma unroll
    for (int d = 0; d < SM_D; ++d) {
        ldg_stream4(w[d], wp + d * 32);
        ldg_stream2(s[d], sp + d * 8);
    }
}

// k-loop body for one prefetched batch of SM_D iterations (4 k16 tiles each).
template <int NT>
__device__ __forceinline__ void do_batch(const uint4 (&w)[SM_D], const uint2 (&s)[SM_D], int kk0, u32 lane_addr, int kp_bytes,
                                         const u32* s_lut, float (&acc)[2][4]) {
#pragma unroll
    for (int d = 0; d < SM_D; ++d) {
        const u32 wv[4] = {w[d].x, w[d].y, w[d].z, w[d].w};
        const u32 sx = s[d].x, sy = s[d].y;
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
            u32 bf[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                ldmatrix_x4(bf[nt], lane_addr + (u32)(nt * 8 * kp_bytes) + (u32)((kk0 + d * 4 + jp * 2) * 32));
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = jp * 2 + jj;
                u32 r[4];
                unpack_word(wv[j], r);
                const u32 sc0 = s_lut[(sx >> (8 * j)) & 255u];
                const u32 sc1 = s_lut[(sy >> (8 * j)) & 255u];
                const u32 a[4] = {mul_bf16x2(r[0], sc0), mul_bf16x2(r[1], sc1), mul_bf16x2(r[2], sc0), mul_bf16x2(r[3], sc1)};
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma16816(acc[nt], a, bf[nt][2 * jj], bf[nt][2 * jj + 1]);
            }
        }
    }
}

template <int NT>
__device__ __forceinline__ void k_loop(uint4 (&w0)[SM_D], uint2 (&s0)[SM_D], uint4 (&w1)[SM_D], uint2 (&s1)[SM_D], const uint4* wp, const uint2* sp,
                                       int nbatch, u32 lane_addr, int kp_bytes, const u32* s_lut, float (&acc)[2][4]) {
    for (int b = 0; b < nbatch; b += 2) {
        if (b + 1 < nbatch) prefetch_batch(w1, s1, wp + (b + 1) * SM_D * 32, sp + (b + 1) * SM_D * 8);
        do_batch<NT>(w0, s0, b * SM_D * 4, lane_addr, kp_bytes, s_lut, acc);
        if (b + 1 < nbatch) {
            if (b + 2 < nbatch) prefetch_batch(w0, s0, wp + (b + 2) * SM_D * 32, sp + (b + 2) * SM_D * 8);
            do_batch<NT>(w1, s1, (b + 1) * SM_D * 4, lane_addr, kp_bytes, s_lut, acc);
        }
    }
}

// Final stage for one warp's 16-row group: scale (global scale and the 2**30 operand scale), SwiGLU (GEMM1) or plain store (GEMM2), BF16 rounding.
template <bool FUSE, int NT>
__device__ __forceinline__ void write_out(const float (&acc)[2][4], float fscale, const int* trow, int me, int g, int gid, int t4,
                                          u16* out, int out_stride) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int tk = nt * 8 + 2 * t4 + c;
            if (tk < me) {
                const int slot = trow[2 + tk];
                u16* o = out + (u64)slot * out_stride;
                if (FUSE) {
                    const float gt = acc[nt][c] * fscale, up = acc[nt][2 + c] * fscale;
                    o[g * 8 + gid] = f2bf((gt / (1.0f + expf(-gt))) * up);
                } else {
                    o[g * 16 + gid] = f2bf(acc[nt][c] * fscale);
                    o[g * 16 + 8 + gid] = f2bf(acc[nt][2 + c] * fscale);
                }
            }
        }
    }
}

// grid (G / NW, KS, nz): CTA = NW warps = NW 16-row weight groups x one K slice of one active expert (z-th of the route table; z loops when
// the grid's z is smaller than the number of active experts, inactive z exit at once). Every warp streams its own contiguous weight bytes
// (uint4 per lane, 512 B per warp load, register double buffering), dequantizes in registers and multiplies with the activations staged in smem.
template <bool FUSE, int NW>
__global__ void __launch_bounds__(NW * 32)
w4_smallm_kernel(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ gsc, const u32* __restrict__ lut,
                 const u16* __restrict__ act, const int* __restrict__ tab, float4* __restrict__ part, int* __restrict__ cnt,
                 u16* __restrict__ out, int G, int NIT, int K, int KS, int out_stride, int act_shift) {
    extern __shared__ __align__(16) u8 smem_raw[];
    __shared__ int s_flag;
    u32* s_lut = reinterpret_cast<u32*>(smem_raw);
    u16* s_act = reinterpret_cast<u16*>(smem_raw + 1024);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    const int nb = blockIdx.x, ks = blockIdx.y, nbn = gridDim.x;
    if (FUSE) pdl_wait();  // GEMM1 reads the route table written by the route kernel; GEMM2 only waits (below) before it reads GEMM1's output
    if (FUSE) pdl_trigger();
    const int Kc = K / KS, nit = Kc >> 6, KP = Kc + 8, kp_bytes = KP * 2;
    const int nbatch = nit / SM_D;
    const int n_act = tab[SM_NEXP * SM_TABW];
    const int g = nb * NW + warp;
    const u32 lane_addr = smem_u32(s_act) + (u32)((((lane & 7) * KP) + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8) * 2);

    for (int z = blockIdx.z; z < n_act; z += gridDim.z) {
        const int* trow = tab + z * SM_TABW;
        const int e = trow[0];
        int me = trow[1];
        me = me > SM_MAXM ? SM_MAXM : me;
        const uint4* wp = Wq + (((u64)e * G + g) * NIT + (u64)ks * nit) * 32 + lane;
        const uint2* sp = Sq + (((u64)e * G + g) * NIT + (u64)ks * nit) * 8 + gid;
        uint4 w0[SM_D], w1[SM_D];
        uint2 s0[SM_D], s1[SM_D];
        prefetch_batch(w0, s0, wp, sp);

        __syncthreads();  // previous z iteration is done with s_act / s_flag
        for (int i = tid; i < 256; i += NW * 32) s_lut[i] = lut[i];
        if (!FUSE) pdl_wait();  // GEMM2: the intermediate (and the shared split-K scratch) is only touched after GEMM1 is complete
        const int nrows = me > 8 ? 16 : 8;
        const int cpr = Kc >> 3;  // 16-byte chunks per activation row
        for (int idx = tid; idx < nrows * cpr; idx += NW * 32) {
            const int row = idx / cpr, c = idx - row * cpr;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (row < me) v = reinterpret_cast<const uint4*>(act + (u64)(trow[2 + row] >> act_shift) * K + (u64)ks * Kc)[c];
            *reinterpret_cast<uint4*>(s_act + row * KP + c * 8) = v;
        }
        __syncthreads();

        float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        if (me <= 8) k_loop<1>(w0, s0, w1, s1, wp, sp, nbatch, lane_addr, kp_bytes, s_lut, acc);
        else k_loop<2>(w0, s0, w1, s1, wp, sp, nbatch, lane_addr, kp_bytes, s_lut, acc);

        const int nt_n = me > 8 ? 2 : 1;
        if (KS > 1) {
            // split-K: partials in the fragment layout (lane-contiguous float4), last arriving CTA of (z, nb) adds them in slice order.
            const u64 pbase = ((((u64)z * nbn + nb) * KS + ks) * NW + warp) * 64 + lane;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
                if (nt < nt_n) part[pbase + nt * 32] = make_float4(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3]);
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                const int prev = atomicAdd(&cnt[z * nbn + nb], 1);
                const int last = (prev == KS - 1);
                s_flag = last;
                if (last) cnt[z * nbn + nb] = 0;
            }
            __syncthreads();
            if (!s_flag) continue;
            __threadfence();
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                if (nt < nt_n) {
                    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                    for (int k2 = 0; k2 < KS; ++k2) {
                        const u64 pb = ((((u64)z * nbn + nb) * KS + k2) * NW + warp) * 64 + lane + nt * 32;
                        const uint4 v = ldcg4(part + pb);
                        a0 += __uint_as_float(v.x); a1 += __uint_as_float(v.y); a2 += __uint_as_float(v.z); a3 += __uint_as_float(v.w);
                    }
                    acc[nt][0] = a0; acc[nt][1] = a1; acc[nt][2] = a2; acc[nt][3] = a3;
                }
            }
        }
        const float fscale = gsc[e] * 1073741824.0f;  // global scale * 2**30 (undoes the operand scaling)
        if (me <= 8) write_out<FUSE, 1>(acc, fscale, trow, me, g, gid, t4, out, out_stride);
        else write_out<FUSE, 2>(acc, fscale, trow, me, g, gid, t4, out, out_stride);
    }
}

// One CTA: local ids (-1 = not local), and the route table: tab[z] = [expert, m_e, slots of its tokens (ascending), ...] for the z-th active
// expert (ascending expert id), rows >= n_active have expert -1, tab[32] = [n_active]. n <= 128 slots (M <= 16 tokens x 8).
__global__ void __launch_bounds__(128) smallm_route_kernel(const int* __restrict__ ids, int* __restrict__ local_ids, int* __restrict__ tab,
                                                           int n, const int* __restrict__ lo_ptr, int nloc) {
    __shared__ int wc[4][32];
    __shared__ int s_cidx[32], s_cnt[32], s_nact;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_trigger();
    const int lo = lo_ptr[0];  // local_start: a device tensor of the layer's inputs, never read on the host
    (&wc[0][0])[tid] = 0;
    __syncthreads();
    int e = -1;
    if (tid < n) {
        const int gl = ids[tid];
        const int rel = (gl - lo) & 255;  // the layer's local-id rule: ((id - local_start) mod 256) if < 32
        e = rel < nloc ? rel : -1;
        local_ids[tid] = e;
    }
    const u32 m = __match_any_sync(FULLMASK, e);
    const int rank_w = __popc(m & ((1u << lane) - 1u));
    if (e >= 0 && rank_w == 0) wc[warp][e] = __popc(m);
    __syncthreads();
    if (warp == 0) {
        const int c = wc[0][lane] + wc[1][lane] + wc[2][lane] + wc[3][lane];
        const u32 act = __ballot_sync(FULLMASK, c > 0);
        s_cidx[lane] = __popc(act & ((1u << lane) - 1u));
        s_cnt[lane] = c;
        if (lane == 0) s_nact = __popc(act);
    }
    __syncthreads();
    if (e >= 0) {
        int base = 0;
        for (int w = 0; w < warp; ++w) base += wc[w][e];
        const int rank = base + rank_w;
        if (rank < SM_MAXM) tab[s_cidx[e] * SM_TABW + 2 + rank] = tid;
    }
    if (tid < SM_NEXP) {
        if (s_cnt[tid] > 0) {
            tab[s_cidx[tid] * SM_TABW] = tid;
            tab[s_cidx[tid] * SM_TABW + 1] = s_cnt[tid];
        }
        if (tid >= s_nact) {
            tab[tid * SM_TABW] = -1;
            tab[tid * SM_TABW + 1] = 0;
        }
    }
    if (tid == 0) tab[SM_NEXP * SM_TABW] = s_nact;
}
"""

_SMALLM_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

static const int SM_SMEM_MAX = 98304;

template <bool FUSE, int NW>
static void launch_gemm_t(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor tab,
                          torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS,
                          int64_t nz, int64_t out_stride, int64_t act_shift) {
    const int Kc = (int)(K / KS);
    const size_t smem = 1024 + (size_t)16 * (Kc + 8) * 2;
    auto fn = w4_smallm_kernel<FUSE, NW>;
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX);
        inited = true;
    }
    const uint4* a_wq = reinterpret_cast<const uint4*>(wq.data_ptr());
    const uint2* a_sq = reinterpret_cast<const uint2*>(sq.data_ptr());
    const float* a_gs = gs.data_ptr<float>();
    const u32* a_lut = reinterpret_cast<const u32*>(lut.data_ptr());
    const u16* a_act = reinterpret_cast<const u16*>(act.data_ptr());
    const int* a_tab = tab.data_ptr<int>();
    float4* a_part = reinterpret_cast<float4*>(part.data_ptr());
    int* a_cnt = cnt.data_ptr<int>();
    u16* a_out = reinterpret_cast<u16*>(out.data_ptr());
    int i_G = (int)G, i_NIT = (int)NIT, i_K = (int)K, i_KS = (int)KS, i_os = (int)out_stride, i_as = (int)act_shift;
    void* args[] = {&a_wq, &a_sq, &a_gs, &a_lut, &a_act, &a_tab, &a_part, &a_cnt, &a_out, &i_G, &i_NIT, &i_K, &i_KS, &i_os, &i_as};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(G / NW), (unsigned)KS, (unsigned)nz);
    cfg.blockDim = dim3(NW * 32, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (at::cuda::getCurrentDeviceProperties()->major >= 9) {
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    cudaLaunchKernelExC(&cfg, (const void*)fn, args);
}

void smallm_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor tab,
                 torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz,
                 int64_t out_stride, int64_t act_shift, int64_t nw, bool fuse) {
    c10::cuda::CUDAGuard guard(act.device());
#define SM_CASE(F, W) launch_gemm_t<F, W>(wq, sq, gs, lut, act, tab, part, cnt, out, G, NIT, K, KS, nz, out_stride, act_shift)
    if (fuse) {
        if (nw == 2) SM_CASE(true, 2); else if (nw == 4) SM_CASE(true, 4); else SM_CASE(true, 8);
    } else {
        if (nw == 2) SM_CASE(false, 2); else if (nw == 4) SM_CASE(false, 4); else SM_CASE(false, 8);
    }
#undef SM_CASE
}

void smallm_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor tab, torch::Tensor lo, int64_t nloc) {
    c10::cuda::CUDAGuard guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    smallm_route_kernel<<<1, 128, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), tab.data_ptr<int>(), (int)ids.numel(), lo.data_ptr<int>(), (int)nloc);
}

void smallm_init() {
    cudaFuncSetAttribute(w4_smallm_kernel<true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX);
    cudaFuncSetAttribute(w4_smallm_kernel<true, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX);
    cudaFuncSetAttribute(w4_smallm_kernel<true, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX);
    cudaFuncSetAttribute(w4_smallm_kernel<false, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX);
    cudaFuncSetAttribute(w4_smallm_kernel<false, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX);
    cudaFuncSetAttribute(w4_smallm_kernel<false, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX);
}
"""
_SMALLM_CPP_SRC = (
    "void smallm_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor tab, "
    "torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz, "
    "int64_t out_stride, int64_t act_shift, int64_t nw, bool fuse);\n"
    "void smallm_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor tab, torch::Tensor lo, int64_t nloc);\n"
    "void smallm_init();\n")


# ---- repack (prepare(), untimed): NVFP4 bytes -> 16-row x 16-k tile words, lossless (a permutation of the same bits; scales stay E4M3 bytes) ----
def _nibble_bit_positions():
    """(reg r, half h, nibble bit j) -> bit of the 32-bit tile word. Regs 0..2: rotr(word, SM_ROT[r]) & 0x81C081C0; reg 3: see SM_REG3_SOURCE."""
    pos = {}
    for r in range(3):
        for h in range(2):
            for j in range(4):
                pos[(r, h, j)] = (SM_NIB_TARGET[j] + 16 * h + SM_ROT[r]) % 32
    for h in range(2):
        for j in range(4):
            pos[(3, h, j)] = SM_REG3_SOURCE[j] + 16 * h
    return pos


def pack_w4_tiles(packed, sf, interleave_gate_up=False):
    """packed uint8 [R, K/2] (two E2M1 per byte, low nibble first), sf float8_e4m3fn/uint8 [R, K/16] ->
    (words int32 [R/16, K/64, 32, 4], scales uint8 [R/16, K/64, 8, 2, 4]).
    Row group g = 16 consecutive rows (or, with interleave_gate_up, 8 gate rows then the 8 up rows of the same 8 output columns). Per k16 tile a lane
    (gid = lane // 4, t = lane % 4) owns the A-fragment elements a0 (row gid, k 2t..2t+1), a1 (row gid+8, same k), a2 (row gid, k 2t+8..), a3 (row gid+8,
    k 2t+8..) of mma.m16n8k16 as one 32-bit word; the 4 tiles of a k64 iteration are the 4 words of one 16-byte lane load."""
    rows, kh = packed.shape
    k = kh * 2
    assert rows % 16 == 0 and k % 64 == 0 and sf.shape == (rows, k // 16)
    wb = packed.view(torch.uint8)
    sb = sf.view(torch.uint8)
    if interleave_gate_up:
        half = rows // 2
        wb = torch.stack([wb[:half].reshape(half // 8, 8, kh), wb[half:].reshape(half // 8, 8, kh)], dim=1).reshape(rows, kh)
        sb = torch.stack([sb[:half].reshape(half // 8, 8, k // 16), sb[half:].reshape(half // 8, 8, k // 16)], dim=1).reshape(rows, k // 16)
    groups, kb, nit = rows // 16, k // 16, k // 64
    b = wb.reshape(groups, 2, 8, kb, 2, 4)  # [g, row half, gid, k16 tile, k half, t]
    pos = _nibble_bit_positions()
    words = torch.zeros((groups, kb, 8, 4), dtype=torch.int32, device=packed.device)
    for rh in range(2):
        for kk in range(2):
            r = rh + 2 * kk
            byte = b[:, rh, :, :, kk, :].permute(0, 2, 1, 3).to(torch.int32)  # [g, kb, gid, t]
            for h, q in enumerate((byte & 15, (byte >> 4) & 15)):
                for j in range(4):
                    words |= ((q >> j) & 1) << pos[(r, h, j)]
    words = words.reshape(groups, nit, 4, 8, 4).permute(0, 1, 3, 4, 2).contiguous()  # [g, it, gid, t, j] = [g, it, lane, j]
    scales = sb.reshape(groups, 2, 8, nit, 4).permute(0, 3, 2, 1, 4).contiguous()  # [g, it, gid, row half, j]
    return words.reshape(groups, nit, 32, 4), scales


def unpack_w4_tiles_torch(words, scales, lut):
    """Torch twin of the kernel's dequant (CPU-testable): returns float32 [R, K] = E2M1 * E4M3 scale * 2**-SM_BITS_SHIFT in the packed row order
    (the kernel's A operand, FP32 view of the exact BF16 values)."""
    def f32(bits):  # int64 holding a 32-bit pattern -> float32
        bits = bits & 0xFFFFFFFF
        return torch.where(bits >= 2 ** 31, bits - 2 ** 32, bits).to(torch.int32).view(torch.float32)

    def rotr(x, n):
        return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF

    groups, nit = words.shape[0], words.shape[1]
    w = words.reshape(groups, nit, 8, 4, 4).to(torch.int64) & 0xFFFFFFFF  # [g, it, gid, t, j]
    regs = [w & 0x81C081C0, rotr(w, 6) & 0x81C081C0, rotr(w, 12) & 0x81C081C0,
            ((w << 6) & 0x80C080C0) | ((w >> 2) & 0x01000100)]
    sc = scales.reshape(groups, nit, 8, 2, 4).to(torch.int64)  # [g, it, gid, rh, j]
    lutv = lut.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros((groups, nit, 4, 2, 8, 4, 2, 2), dtype=torch.float32, device=words.device)  # [g, it, j, rh, gid, t, kk, h]
    for r in range(4):
        rh, kk = r % 2, r // 2
        s_f = f32((lutv[sc[:, :, :, rh, :]] & 0xFFFF) << 16)  # [g, it, gid, j]
        for h in range(2):
            val = f32(((regs[r] >> (16 * h)) & 0xFFFF) << 16)  # bf16 pattern as fp32, [g, it, gid, t, j]
            out[:, :, :, rh, :, :, kk, h] = (val * s_f.unsqueeze(3)).permute(0, 1, 4, 2, 3)
    # element (row 16 g + 8 rh + gid, k 64 it + 16 j + 8 kk + 2 t + h)
    return out.permute(0, 3, 4, 1, 2, 6, 5, 7).reshape(groups * 16, nit * 64)


def make_scale_lut(device):
    """lut[b] = (bf16(e4m3(b) * 2**96)) in both halves of an int32: the BF16 block scale the dequant multiplies the raw E2M1 pattern with."""
    f = torch.arange(256, dtype=torch.uint8, device=device).view(torch.float8_e4m3fn).float() * (2.0 ** SM_LUT_EXP)
    bits = f.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    v = bits | (bits << 16)
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)


# ---- per-M launch configuration: (warps per CTA, K splits, grid z cap) for GEMM1 and GEMM2 ----
# Tuned on the judge H200 by its measure_smallm_v1 (job 7e8e3866f7f64f2a, generate_input routing, 3 seeds per M).
SMALLM_CFG = (
    # (largest M, nw1, ks1, nz1, nw2, ks2, nz2)
    (1, 8, 2, 2, 4, 2, 6),
    (2, 8, 2, 6, 4, 4, 12),
    (4, 8, 2, 2, 4, 4, 6),
    (8, 8, 2, 8, 8, 2, 4),
    (12, 8, 2, 8, 8, 2, 8),
    (16, 8, 2, 8, 8, 2, 16),
)


def smallm_cfg(m):
    for mx, nw1, ks1, nz1, nw2, ks2, nz2 in SMALLM_CFG:
        if m <= mx:
            return dict(nw1=nw1, ks1=ks1, nz1=nz1, nw2=nw2, ks2=ks2, nz2=nz2)
    return None


_SMALLM_EXT = {}


def smallm_ext(flags=()):
    """The compiled extension (module with smallm_gemm / smallm_route / smallm_init), built once per process and flag set by torch's load_inline."""
    key = tuple(flags)
    if key not in _SMALLM_EXT:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        src = _SMALLM_DEVICE_SRC + _SMALLM_HOST_SRC
        tag = hashlib.sha1((src + repr(key)).encode()).hexdigest()[:12]
        _SMALLM_EXT[key] = load_inline(
            name=f"okc_smallm_{tag}", cpp_sources=_SMALLM_CPP_SRC, cuda_sources=src,
            functions=["smallm_gemm", "smallm_route", "smallm_init"], extra_cuda_cflags=["-O3", *flags], verbose=False)
    return _SMALLM_EXT[key]


def prepare_smallm(weights, ext=None, flags=()):
    """Untimed. Repacks experts [0, 32) of w13 / w2 into tile words (gate/up interleaved for GEMM1) and allocates the scratch the small-M kernels
    use: the route table, the split-K partials and the fixup counters (zero here, reset to zero by the fixup CTAs after every use)."""
    dev = weights["w13"].device
    e_n = SM_NEXP
    inter, hid = weights["w13"].shape[1] // 2, weights["w2"].shape[1]
    k1, k2 = weights["w13"].shape[2] * 2, weights["w2"].shape[2] * 2
    g1, g2 = (2 * inter) // 16, hid // 16
    w1 = torch.empty((e_n, g1, k1 // 64, 32, 4), dtype=torch.int32, device=dev)
    s1 = torch.empty((e_n, g1, k1 // 64, 8, 2, 4), dtype=torch.uint8, device=dev)
    w2 = torch.empty((e_n, g2, k2 // 64, 32, 4), dtype=torch.int32, device=dev)
    s2 = torch.empty((e_n, g2, k2 // 64, 8, 2, 4), dtype=torch.uint8, device=dev)
    for e in range(e_n):
        w1[e], s1[e] = pack_w4_tiles(weights["w13"][e], weights["w13_sf"][e], interleave_gate_up=True)
        w2[e], s2[e] = pack_w4_tiles(weights["w2"][e], weights["w2_sf"][e])
    max_part = 0
    for mx, nw1, ks1, nz1, nw2, ks2, nz2 in SMALLM_CFG:
        max_part = max(max_part, g1 * ks1, g2 * ks2)
    sm = {
        "w1": w1, "s1": s1, "w2": w2, "s2": s2,
        "gs1": weights["w13_gs"][:e_n].contiguous(), "gs2": weights["w2_gs"][:e_n].contiguous(),
        "lut": make_scale_lut(dev), "tab": torch.zeros(SM_TAB_INTS, dtype=torch.int32, device=dev),
        "part": torch.empty(e_n * max_part * 64 * 4, dtype=torch.float32, device=dev),
        "cnt": torch.zeros((2, e_n * 1024), dtype=torch.int32, device=dev),
        "dims": dict(inter=inter, hid=hid, k1=k1, k2=k2, g1=g1, g2=g2),
        "ext": ext if ext is not None else smallm_ext(flags),
    }
    sm["ext"].smallm_init()
    return sm


def smallm_experts(x, topk_ids, sm, start, count, cfg):
    """Route + GEMM1 (SwiGLU fused) + GEMM2 for M <= 16 tokens. start: int32 device tensor [1] (local_start). Returns (y [M, 8, H] BF16,
    local_ids [M, 8] int32). Shape-only grids; no host sync."""
    m = x.shape[0]
    d, ext = sm["dims"], sm["ext"]
    assert m <= SM_MAX_M and count == SM_NEXP
    ids = topk_ids.to(torch.int32).contiguous()
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
    ext.smallm_route(ids, local_ids, sm["tab"], start, count)
    inter = torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
    nz = min(SM_NEXP, m * TOP_K)
    for k_dim, groups, ks, nw in ((d["k1"], d["g1"], cfg["ks1"], cfg["nw1"]), (d["k2"], d["g2"], cfg["ks2"], cfg["nw2"])):  # shapes only
        assert k_dim % ks == 0 and (k_dim // ks) % 256 == 0 and k_dim // ks <= 2048 and groups % nw == 0, (k_dim, groups, ks, nw)
    ext.smallm_gemm(sm["w1"], sm["s1"], sm["gs1"], sm["lut"], x.contiguous(), sm["tab"], sm["part"], sm["cnt"][0], inter,
                    d["g1"], d["k1"] // 64, d["k1"], cfg["ks1"], min(nz, cfg["nz1"]), d["inter"], 3, cfg["nw1"], True)
    ext.smallm_gemm(sm["w2"], sm["s2"], sm["gs2"], sm["lut"], inter, sm["tab"], sm["part"], sm["cnt"][1], y,
                    d["g2"], d["k2"] // 64, d["k2"], cfg["ks2"], min(nz, cfg["nz2"]), d["hid"], 0, cfg["nw2"], False)
    return y.view(m, TOP_K, d["hid"]), local_ids

# ---- the layer's small-M entry (S4) ----
S4_SMALLM = True   # M <= SM_MAX_M tokens: the streaming split-K W4A16 path instead of align + Marlin x2 + silu_and_mul (needs state["smallm"])


def _smallm_live(num_tokens, state):
    return S4_SMALLM and state.get("smallm") is not None and 1 <= num_tokens <= SM_MAX_M


def _experts_smallm(x, topk_weights, topk_ids, local_start, state):
    """smallm_experts (route + GEMM1 with SwiGLU + GEMM2) and the layer's fused Triton epilogue, tiled by S1's table."""
    m = x.shape[0]
    y, local_ids = smallm_experts(x.contiguous(), topk_ids, state["smallm"], local_start, LOCAL_EXPERTS, smallm_cfg(m))
    return epilogue_triton(y, topk_weights.to(torch.float32).contiguous(), local_ids, state["post_expert_w"],
                           block_h=_marlin_knobs(m)[2])


def _smallm_selfcheck_inputs(m, seed, device):
    """Seeded synthetic expert inputs (shapes only; nothing from the judge's inputs): unit-scale activations and a skewed routing."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, D_EXPERT), generator=g, device=device).to(torch.bfloat16)
    scores = torch.rand((m, N_EXPERTS), generator=g, device=device)
    z = torch.randn((N_EXPERTS,), generator=g, device=device)
    u = torch.rand((N_EXPERTS,), generator=g, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS   # the busiest 32-expert range becomes local
    return x, (routed.values / routed.values.sum(-1, keepdim=True)).float(), ids.to(torch.int32)


def _smallm_selfcheck(state):
    """Worst rel L2 of the small-M experts against the Marlin path over M in (1, 8, 16), or inf if anything raises."""
    worst = 0.0
    try:
        start = torch.zeros(1, dtype=torch.int32, device=state["post_expert_w"].device)
        for m in (1, 8, 16):
            x, w, ids = _smallm_selfcheck_inputs(m, 1000 + m, state["post_expert_w"].device)
            ref = _experts(x, w, ids, start, {**state, "smallm": None}).float()
            got = _experts_smallm(x, w, ids, start, state).float()
            torch.cuda.synchronize()
            err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
            worst = max(worst, err if err == err else float("inf"))
    except Exception as ex:  # noqa: BLE001
        print(f"okc-diag smallm selfcheck raised {type(ex).__name__}: {str(ex)[:160]}")
        return float("inf")
    return worst


def _prepare_smallm_for_layer(weights, state):
    """Build the small-M path (repack, JIT compile) and keep it only if its self-check against Marlin passes; otherwise (no nvcc, build
    error, out of memory, mismatch) state["smallm"] stays None and Marlin serves every M."""
    state["smallm"] = None
    if not (S4_SMALLM and torch.cuda.is_available()):
        return
    for flags in ((), ("-DSM_FP32_MUL=1",)):  # bf16x2 multiply first; FP32 products if its self-check fails
        try:
            state["smallm"] = prepare_smallm(weights, flags=flags)
        except Exception as ex:  # noqa: BLE001
            print(f"okc-diag smallm unavailable {flags}: {type(ex).__name__}: {str(ex)[:300]}")
            state["smallm"] = None
            return
        err = _smallm_selfcheck(state)
        print(f"okc-diag smallm selfcheck {flags} rel L2 vs Marlin: {err:.3e}")
        if err < SELFCHECK_TOL:
            break
        state["smallm"] = None
    print(f"okc-diag smallm active={state['smallm'] is not None}")


SELFCHECK_TOL = 1e-2


def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize, as kernels/muse_moe_w4a16/sonnet_v1.py:
    range-based local ids (one kernel), Marlin W4A16 GEMMs, and below 32 tokens (where SGLang runs ~8 eager ops) the fused
    Triton epilogue; from 32 tokens SGLang's own fused kernel."""
    if local_ids is None and _smallm_live(x.shape[0], state):
        return _experts_smallm(x, topk_weights, topk_ids, local_start, state)
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    if local_ids is None:  # S2 hands them over from the routing kernel
        local_ids = local_ids_triton(topk_ids, local_start)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    atomic, fp32_reduce, epilogue_block_h = _marlin_knobs(num_tokens)
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                  use_fp32_reduce=fp32_reduce, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if _epilogue_path(num_tokens) == "triton":
        return epilogue_triton(y, topk_weights, local_ids, state["post_expert_w"], block_h=epilogue_block_h)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _attention_fa3(data, qkv, state):
    """SGLang's attention chain (verify blocks, and everything off the Triton path): FusedQKNorm, FP8 cast + store_cache,
    FA3, FusedRMSNormGate. Returns the gated, normalized BF16 attention output [T, Hq*d]."""
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    return state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)


def _kernel_sglang(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    dense = state["dense"]
    device = data["attn_input"].device
    triton_proj = _triton_ok(device)
    # Attention: MuseSpark1xAttention.forward (the qkv_gate projection stays cuBLAS).
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if _attention_path(cfg, device.type == "cuda") == "triton":
        o = _attention_decode({**data, "qkv": qkv}).view(tokens, hq * d)
    else:
        o = _attention_fa3(data, qkv, state)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    rpath, ppath = _router_path(tokens, triton_proj), _pooler_path(tokens, triton_proj)
    # The baseline's boundary kernel also writes the FP32 router input up to 8 tokens; the Triton router splits it hi + lo.
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16,
        router_dtype=torch.float32 if tokens <= ROUTER_FP32_MAX_TOKENS else None)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _pooler(ffn_input, state, ppath)
    f2 = _f2_weights(tokens, device, state)
    local_ids = None
    with torch.cuda.stream(side_stream):
        if f2 is not None:  # S2: GEMV + sigmoid + bias + top-8 + renorm + local ids in one launch
            topk_ids, topk_weights, local_ids = fused_router_f2(
                (router_input if router_input is not None else ffn_input).contiguous(), f2[0], f2[1], state["router_bias"],
                data["local_start"], state["f2_done"])
        else:
            logits = _router_logits(ffn_input, router_input, state, rpath)
            topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids) + ((local_ids,) if local_ids is not None else ()):
            tensor.record_stream(main_stream)
    # all-reduce: identity. This GPU's experts start at local_start (read on the device by the local-id kernel).
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, topk_ids, data["local_start"], state,
                             local_ids=local_ids)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


# =============================================================================================== plain-PyTorch layer (no SGLang)
def _rms(x, eps=RMS_EPS):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _gain(w):
    return 1.0 + w.float()


def _rope(x, pos, theta):
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device).float() / d))
    ang = pos.float()[:, None, None] * inv_freq
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack([even * ang.cos() - odd * ang.sin(), even * ang.sin() + odd * ang.cos()], -1).flatten(-2)


def _residual_coefficients(gate):
    w = gate.float() / 0.3
    beta = torch.sigmoid(w)
    return torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3)), beta


def _attention_torch(data, qkv):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    q, k, v, g = qkv.split([hq * d, hkv * d, hkv * d, hq * d], dim=-1)
    pos = data["positions"]
    q = _rms(q.reshape(-1, hq, d))
    k = _rms(k.reshape(-1, hkv, d))
    if cfg["rope_theta"]:
        q, k = _rope(q, pos, cfg["rope_theta"]), _rope(k, pos, cfg["rope_theta"])
    q = (q * cfg["qk_scale_factor"]).to(torch.bfloat16)
    table = data["page_table"].long()
    k_cache = data["k_cache"].view(-1, hkv, d)
    v_cache = data["v_cache"].view(-1, hkv, d)
    k_scale, v_scale = data["k_scale"], data["v_scale"]
    k_cache[data["cache_loc"]] = (k / k_scale).to(FP8)
    v_cache[data["cache_loc"]] = (v.reshape(-1, hkv, d).float() / v_scale).to(FP8)
    t = torch.arange(table.shape[1] * page, device=qkv.device)
    pos = pos.view(batch, length)
    q = q.view(batch, length, hkv, hq // hkv, d).float()
    out = torch.empty((batch, length, hq, d), device=qkv.device)
    for b in range(batch):
        rows = table[b].clamp(min=0)[t // page] * page + t % page
        keys = k_cache[rows].float() * k_scale
        vals = v_cache[rows].float() * v_scale
        for c in range(0, length, 512):
            j = slice(c, min(c + 512, length))
            visible = t[None, :] <= pos[b, j][:, None]
            if cfg["window"]:
                visible &= t[None, :] > pos[b, j][:, None] - cfg["window"]
            att = torch.einsum("lkgd,skd->lkgs", q[b, j], keys) * cfg["sm_scale"]
            att = att.masked_fill(~visible[:, None, None, :], float("-inf")).softmax(-1)
            out[b, j] = torch.einsum("lkgs,skd->lkgd", att, vals).reshape(-1, hq, d)
    return (_rms(out).reshape(-1, hq * d) * torch.sigmoid(g.float())).to(torch.bfloat16)


def _experts_torch(x, ids, topk_weights, w):
    from nvfp4 import dequantize  # the judge's helper; only the no-SGLang path uses it

    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(LOCAL_EXPERTS):
        g = (topk_weights * (ids == e)).sum(-1)
        w13 = dequantize(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).to(torch.bfloat16)
        h = x @ w13.T
        inter = F.silu(h[:, :D_EXPERT]) * h[:, D_EXPERT:]
        w2 = dequantize(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).to(torch.bfloat16)
        y = (inter @ w2.T).float() * _gain(w["post_expert_norm"])
        out += g[:, None] * _rms(y.to(torch.bfloat16), POST_NORM_EPS)
    return out.to(torch.bfloat16)


def _kernel_torch(data, state):
    cfg = data["config"]
    tokens = cfg["tokens"]
    w, dense = state["weights"], state["dense"]
    device = data["attn_input"].device
    triton_proj = _triton_ok(device)
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if _attention_path(cfg, device.type == "cuda") == "triton":
        o = _attention_decode({**data, "qkv": qkv}).view(tokens, -1)
    else:
        o = _attention_torch(data, qkv)
    a = o @ dense["o"].T
    alpha, beta = _residual_coefficients(w["attn_residual_gate"])
    stream = alpha * data["stream"] + beta * _rms(a, POST_NORM_EPS)
    h32 = _rms(stream) * _gain(w["pre_ffn_norm"])
    h = h32.to(torch.bfloat16)
    rpath = _router_path(tokens, triton_proj)
    f2 = _f2_weights(tokens, device, state)
    if f2 is not None:  # S2 (same dispatch as the SGLang path)
        ids, topk_weights, local = fused_router_f2(
            h32.contiguous() if tokens <= ROUTER_FP32_MAX_TOKENS else h.contiguous(), f2[0], f2[1], state["router_bias"],
            data["local_start"], state["f2_done"])
    else:
        logits = _router_logits(h, h32 if tokens <= ROUTER_FP32_MAX_TOKENS else None, state, rpath)
        scores = torch.sigmoid(logits * 0.1767766952966369)
        ids = (scores + w["router_bias"]).topk(cfg["top_k"], dim=-1).indices
        picked = scores.gather(1, ids)
        topk_weights = picked / (picked.sum(-1, keepdim=True) + 1e-15)
        local = (ids - data["local_start"]) % N_EXPERTS
    p = (_rms(_pooler(h, state, _pooler_path(tokens, triton_proj))) * _gain(w["pre_expert_norm"])).to(torch.bfloat16)
    y = _experts_torch(p, local, topk_weights, w)
    if tokens > SLICED_UNPOOL_MAX_TOKENS:
        f = (y @ dense["unpooler"].T).float()
    else:
        f = F.pad((y @ dense["unpooler"][:UNPOOL_SLICE].T).float(), (0, HIDDEN - UNPOOL_SLICE))
    f = f * _gain(w["post_ffn_norm"])
    alpha, beta = _residual_coefficients(w["ffn_residual_gate"])
    stream = alpha * stream + beta * _rms(f, POST_NORM_EPS)
    hidden = (_rms(stream) * _gain(w["next_pre_attn_norm"])).to(torch.bfloat16)
    return hidden, stream, ids.to(torch.int32)


# =============================================================================================== entry points
def prepare(weights):
    """Called once, untimed. Weights are repacked losslessly: BF16 hi / lo halves of the FP32 router, as SGLang does."""
    print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")
    if HAVE_SGLANG:
        state = _prepare_sglang(weights)
        _prepare_smallm_for_layer(weights, state)
        return state
    return {"weights": weights,
            "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
            **_split_router_weights(weights["router"], weights["router_bias"])}


def kernel_fn(data, state):
    if HAVE_SGLANG:
        return _kernel_sglang(data, state)
    return _kernel_torch(data, state)
