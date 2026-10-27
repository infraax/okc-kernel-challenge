# okc: problem=muse_decode_layer
# sonnet_v1 -- SGLang's decoder layer (baseline.py, every module kept: cuBLAS qkv/o/unpooler, Marlin experts, JIT norms,
# side-stream overlap) with three judged kernels spliced in. PREDICTED speedup vs SGLang: ~1.05x geo-mean over the 6 loads
# (range 1.02-1.08). Unmeasured on the H200; per-load estimates below are SGLang-time saved / an assumed layer time of
# 150-560 us (the TASK.md floors are only lower bounds, so the percentages are guesses; the saved microseconds are measured).
#   1. Router: director_v1b's Triton split-K GEMV on the BF16 hi+lo weight split, FP32 accumulate, all T <= 256 (judged
#      1.5-4.1x per shape vs SGLang's FP32 F.linear <= 8 tokens / two BF16 GEMMs above). It runs on the side stream
#      under the pooler, so it only reaches the critical path when it is slower than the pooler: at 8 tokens SGLang's FP32
#      F.linear takes ~26 us vs the pooler's ~21 us -> verify 1 saves ~10 us/layer; elsewhere it is hidden (small gain from
#      less SM contention). With the Triton router the FP32 router input the boundary kernel writes <= 8 tokens is dropped.
#   2. Pooler projection: director_v1b's Triton config for T <= 32 (judged 1.10-1.12x, ~2 us/layer), cuBLAS above.
#   3. Attention, decode loads (q_len 1) only: sonnet_v1's 2-launch split-KV Triton kernel replaces FusedQKNorm + FP8 cast +
#      store_cache + FA3 + FusedRMSNormGate (judged attention loads: sliding 1.64-1.82x = -9..-12 us, full 1.10-1.40x =
#      -9/-19/-48 us for batch 1/16/32). Verify (q_len 8) keeps SGLang's FA3 chain: sonnet_v1 judged 0.62-0.76x there on
#      sliding layers and 0.79x at batch 32 full. (Knob SONNET_VERIFY_FULL_MAX_BATCH, default 0 = off, would route the
#      batch-1 full verify load, judged 1.42x, to the Triton kernel.)
# Per load (46x sliding + 16x full per step): decode 1 ~ +7% (attention -9 us, pooler -2 us), decode 16 ~ +8%,
#   decode 32 ~ +9% (full layer -48 us of ~560), verify 1 ~ +6% (router off the critical path), verify 16/32 ~ +1%
#   (pooler/router changes only; T=128/256 keep cuBLAS pooler). Geo-mean ~1.05x. Qkv, o, unpooler, experts, norms: unchanged.
# Without SGLang (local runs) the same dispatch drives a plain-PyTorch layer (the starter's math); on the judge the SGLang
# path runs. Rules: grid / splits / configs come from shapes and config only; lengths, positions, rows, scales are read on
# the device; no host syncs; every key attended; KV cache FP8, activations BF16, router BF16 hi+lo with FP32 accumulate.
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
ROUTER_TRITON_MAX_TOKENS = 256   # director_v1b: Triton router up to here
POOLER_TRITON_MAX_TOKENS = 32    # director_v1b: Triton pooler up to here
SONNET_VERIFY_FULL_MAX_BATCH = 0  # >0: full-attention verify blocks of up to this batch use the Triton attention (off)


# =============================================================================================== projections (director_v1b)
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


# =============================================================================================== attention decode (sonnet_v1)
LOG2E = 1.4426950408889634
BLOCK_N = 64            # keys per inner step
MIN_SPLIT = 128         # fewest keys a split handles
TARGET_PROGRAMS = 4 * 132  # attention programs to aim for (4 per H200 SM); fewer under the CPU interpreter
MAX_ROWS = 128          # L * (Hq / Hkv) rows per KV head the kernel supports (verify blocks: 8 * 8 = 64)


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


@triton.jit
def _load_norm_rope(base, ok, d, pos, scale, log2_theta, ROPE: tl.constexpr):
    """rows of 64 bf16 at base+d -> RMSNorm (eps 1e-5), x scale, interleaved RoPE if ROPE. fp32 [R, 64]."""
    x = tl.load(base + d, mask=ok, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=1) / 64.0 + 1e-5)
    xn = x * inv[:, None]
    if ROPE:
        xp = tl.load(base + (d ^ 1), mask=ok, other=0.0).to(tl.float32) * inv[:, None]
        inv_freq = tl.exp2(-((d // 2) * 2).to(tl.float32) * (log2_theta / 64.0))
        ang = pos.to(tl.float32)[:, None] * inv_freq
        sgn = tl.where((d & 1) == 0, -1.0, 1.0)
        xn = xn * tl.cos(ang) + sgn * xp * tl.sin(ang)
    return xn * scale


# --------------------------------------------------------------------------------------------- kernel 1
@triton.jit
def _attn_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l,
    sm_scale, qk_factor, log2_theta, pt_stride, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    WINDOW: tl.constexpr, ROPE: tl.constexpr, NATIVE_LOAD: tl.constexpr, NATIVE_STORE: tl.constexpr,
    CDTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_T: tl.constexpr,
):
    s = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    d = tl.arange(0, 64)[None, :]

    seq = tl.load(seq_lens + b).to(tl.int32)           # cached tokens; the block starts at this position
    lo = tl.maximum(seq - WINDOW + 1, 0) if WINDOW else 0
    start = lo + s * split_len
    end = tl.minimum(seq, start + split_len)
    ks = tl.load(k_scale_ptr)
    vs = tl.load(v_scale_ptr)
    c = sm_scale * ks * 1.4426950408889634              # logits in the log2 domain

    if (s == 0) | (start < end):
        r = tl.arange(0, BLOCK_M)
        j = r // G
        rvalid = r < L * G
        tk = b * L + j
        pos_r = tl.load(positions + tk, mask=rvalid, other=0)
        qbase = qkv + tk[:, None] * QKV + (h * G + r % G)[:, None] * 64
        q = _load_norm_rope(qbase, rvalid[:, None], d, pos_r, qk_factor, log2_theta, ROPE).to(CDTYPE)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, 64], tl.float32)

        # ---- cached keys [start, end) of this split
        for n0 in range(start, end, BLOCK_N):
            t = n0 + tl.arange(0, BLOCK_N)
            tm = t < end
            pg = tl.load(page_table + b * pt_stride + t // PAGE, mask=tm, other=0)
            row = pg.to(tl.int64) * PAGE + t % PAGE
            off = row[:, None] * (HKV * 64) + h * 64 + d
            ku = tl.load(k_cache + off, mask=tm[:, None], other=0)
            if NATIVE_LOAD:
                kc = ku.to(tl.float8e4nv, bitcast=True).to(CDTYPE)
            else:
                kc = _decode_e4m3(ku).to(CDTYPE)
            sc = tl.dot(q, tl.trans(kc)) * c
            ok = tm[None, :]
            if WINDOW:
                ok = ok & (t[None, :] >= (seq + j - WINDOW + 1)[:, None])
            sc = tl.where(ok, sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(sc - m_use[:, None])
            alpha = tl.exp2(m_i - m_use)
            l_i = l_i * alpha + tl.sum(p, 1)
            vu = tl.load(v_cache + off, mask=tm[:, None], other=0)
            if NATIVE_LOAD:
                vc = vu.to(tl.float8e4nv, bitcast=True).to(CDTYPE)
            else:
                vc = _decode_e4m3(vu).to(CDTYPE)
            acc = acc * alpha[:, None] + tl.dot(p.to(CDTYPE), vc)
            m_i = m_new

        # ---- split 0 also normalizes, rounds, stores and attends the block's own keys / values
        if s == 0:
            ti = tl.arange(0, BLOCK_T)
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
                kt = k8.to(CDTYPE)
                vt = v8.to(CDTYPE)
            else:
                ku8 = _encode_e4m3(kn / ks)
                vu8 = _encode_e4m3(vn / vs)
                tl.store(k_cache + coff, ku8, mask=tv[:, None])
                tl.store(v_cache + coff, vu8, mask=tv[:, None])
                kt = _decode_e4m3(ku8).to(CDTYPE)
                vt = _decode_e4m3(vu8).to(CDTYPE)
            sc = tl.dot(q, tl.trans(kt)) * c
            ok = tv[None, :] & (ti[None, :] <= j[:, None])
            if WINDOW:
                ok = ok & (ti[None, :] > j[:, None] - WINDOW)
            sc = tl.where(ok, sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, 1))
            m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp2(sc - m_use[:, None])
            alpha = tl.exp2(m_i - m_use)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(CDTYPE), vt)
            m_i = m_new

        pidx = ((b * HKV + h) * NSPLIT + s) * BLOCK_M + r
        tl.store(part_m + pidx, m_i)
        tl.store(part_l + pidx, l_i)
        tl.store(part_o + pidx[:, None] * 64 + d, acc)


# --------------------------------------------------------------------------------------------- kernel 2
@triton.jit
def _reduce_kernel(
    part_o, part_m, part_l, qkv, seq_lens, v_scale_ptr, out, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_S: tl.constexpr,
):
    t = tl.program_id(0)        # b * L + j
    hq = tl.program_id(1)
    b = t // L
    j = t % L
    h = hq // G
    row = j * G + hq % G
    d = tl.arange(0, 64)
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64

    seq = tl.load(seq_lens + b).to(tl.int32)
    lo = tl.maximum(seq - WINDOW + 1, 0) if WINDOW else 0
    n_active = tl.maximum((seq - lo + split_len - 1) // split_len, 1)
    base = (b * HKV + h) * NSPLIT * BLOCK_M + row

    M = tl.full([1], float("-inf"), tl.float32)
    Lsum = tl.zeros([1], tl.float32)
    O = tl.zeros([64], tl.float32)
    for s0 in range(0, n_active, BLOCK_S):
        si = s0 + tl.arange(0, BLOCK_S)
        sv = si < n_active
        pidx = base + si * BLOCK_M
        ms = tl.load(part_m + pidx, mask=sv, other=float("-inf"))
        ls = tl.load(part_l + pidx, mask=sv, other=0.0)
        os_ = tl.load(part_o + pidx[:, None] * 64 + d[None, :], mask=sv[:, None], other=0.0)
        m_new = tl.maximum(M, tl.max(ms, 0))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(M - m_use)
        w = tl.exp2(ms - m_use)
        Lsum = Lsum * alpha + tl.sum(w * ls, 0)
        O = O * alpha + tl.sum(w[:, None] * os_, 0)
        M = m_new

    o = O / Lsum * tl.load(v_scale_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / 64.0 + 1e-5)
    gate = tl.load(qkv + t * QKV + (HQ + 2 * HKV) * 64 + hq * 64 + d).to(tl.float32)
    o = o * (1.0 / (1.0 + tl.exp(-gate)))
    tl.store(out + t * (HQ * 64) + hq * 64 + d, o.to(tl.bfloat16))


# --------------------------------------------------------------------------------------------- host
@functools.lru_cache(maxsize=None)
def _device_mode(device_type: str, index: int):
    """(compute dtype, FP8 loads native, FP8 stores native, programs to aim for): hardware only, never input values.
    sm_89+ converts FP8 in hardware. Elsewhere (the sm_75 dev GPU) FP8 is decoded / encoded in software; under the
    Triton CPU interpreter loads are exact but its fp32->e4m3 cast truncates near binade edges, so stores are software."""
    if device_type != "cuda":
        return tl.float32, True, False, 8
    major, minor = torch.cuda.get_device_capability(index)
    cdtype = tl.float16     # mma operands: exact for E4M3 (one cvt.f16x2.e4m3x2), 3 more mantissa bits than BF16
    native = (major, minor) >= (8, 9)
    return cdtype, native, native, TARGET_PROGRAMS


def _plan(batch, hkv, span, target):
    """Splits along the sequence from shapes only: span is the most cached keys a request can have."""
    per_head = max(1, -(-target // (batch * hkv)))
    split_len = max(MIN_SPLIT, -(-span // per_head))
    split_len = -(-split_len // BLOCK_N) * BLOCK_N
    return split_len, max(1, -(-span // split_len))


def _attention_decode(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page, window = cfg["page_size"], cfg["window"]
    group = hq // hkv
    rows = length * group
    if d != 64 or hq % hkv or rows > MAX_ROWS:
        raise ValueError(f"unsupported config: head_dim {d}, {hq} q heads / {hkv} kv heads, q_len {length}")
    block_m = max(16, triton.next_power_of_2(rows))
    block_t = max(16, triton.next_power_of_2(length))
    qkv = data["qkv"]
    dev = qkv.device
    cdtype, native_load, native_store, target = _device_mode(dev.type, dev.index or 0)

    max_pages = data["page_table"].shape[1]
    cap = max_pages * page                       # most tokens a request's page table can hold
    span = max(1, min(window - 1, cap) if window else cap)
    split_len, nsplit = _plan(batch, hkv, span, target)

    part_o = torch.empty((batch, hkv, nsplit, block_m, 64), dtype=torch.float32, device=dev)
    part_m = torch.empty((batch, hkv, nsplit, block_m), dtype=torch.float32, device=dev)
    part_l = torch.empty_like(part_m)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    k_cache = data["k_cache"].view(torch.uint8)
    v_cache = data["v_cache"].view(torch.uint8)

    _attn_kernel[(nsplit, hkv, batch)](
        qkv, data["positions"], data["cache_loc"], data["seq_lens"], data["page_table"], k_cache, v_cache,
        data["k_scale"], data["v_scale"], part_o, part_m, part_l,
        cfg["sm_scale"], cfg["qk_scale_factor"], math.log2(cfg["rope_theta"]) if cfg["rope_theta"] else 0.0,
        data["page_table"].stride(0), split_len, nsplit,
        L=length, G=group, HQ=hq, HKV=hkv, PAGE=page, WINDOW=window, ROPE=bool(cfg["rope_theta"]),
        NATIVE_LOAD=native_load, NATIVE_STORE=native_store, CDTYPE=cdtype, BLOCK_M=block_m, BLOCK_N=BLOCK_N, BLOCK_T=block_t,
        num_warps=4, num_stages=3,
    )
    _reduce_kernel[(batch * length, hq)](
        part_o, part_m, part_l, qkv, data["seq_lens"], data["v_scale"], out, split_len, nsplit,
        L=length, G=group, HQ=hq, HKV=hkv, WINDOW=window, BLOCK_M=block_m,
        BLOCK_S=min(32, triton.next_power_of_2(nsplit)), num_warps=2,
    )
    return out


# =============================================================================================== dispatch
# Shapes, config and hardware only (never input values): which implementation runs each stage.
def _router_path(tokens, triton_proj):
    if triton_proj and tokens <= ROUTER_TRITON_MAX_TOKENS:
        return "triton"
    return "fp32" if tokens <= ROUTER_FP32_MAX_TOKENS else "split"


def _pooler_path(tokens, triton_proj):
    return "triton" if triton_proj and tokens <= POOLER_TRITON_MAX_TOKENS else "cublas"


def _attention_path(cfg, is_cuda):
    """"triton" = sonnet_v1's fused kernels (decode, q_len 1); "reference" = SGLang's FA3 chain (or the torch layer)."""
    if not is_cuda:
        return "reference"
    if cfg["q_len"] == 1:
        return "triton"
    if SONNET_VERIFY_FULL_MAX_BATCH and not cfg["window"] and cfg["batch"] <= SONNET_VERIFY_FULL_MAX_BATCH:
        return "triton"
    return "reference"


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _router_logits(x, router_input, state, path):
    """FP32 logits [T, 256]. router_input (FP32 hidden) exists only on the "fp32" path (and in the torch layer)."""
    if path == "triton":
        return _run(x.contiguous(), state["router_hi"], state["router_lo"], N_EXPERTS, torch.float32,
                    _CFG[("router", _bucket(x.shape[0]))])
    if router_input is not None:
        return F.linear(router_input, state["router"])
    logits = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    logits += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return logits


def _pooler(x, state, path):
    if path == "triton":
        return _run(x.contiguous(), state["dense"]["pooler"], None, D_EXPERT, torch.bfloat16,
                    _CFG[("pooler", _bucket(x.shape[0]))])
    return _gemm(x, state["dense"]["pooler"])


def _split_router_weights(router):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return {"router": router.contiguous(), "router_hi": hi.contiguous(), "router_lo": lo.contiguous(),
            "router_hi_t": hi.t().contiguous(), "router_lo_t": lo.t().contiguous()}


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
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
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
        **_split_router_weights(weights["router"]),
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


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
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
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16,
        router_dtype=torch.float32 if rpath == "fp32" else None)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _pooler(ffn_input, state, ppath)
    with torch.cuda.stream(side_stream):
        logits = _router_logits(ffn_input, router_input, state, rpath)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
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
    h = (_rms(stream) * _gain(w["pre_ffn_norm"])).to(torch.bfloat16)
    rpath = _router_path(tokens, triton_proj)
    logits = _router_logits(h, h.float(), state, rpath)
    scores = torch.sigmoid(logits * 0.1767766952966369)
    ids = (scores + w["router_bias"]).topk(cfg["top_k"], dim=-1).indices
    picked = scores.gather(1, ids)
    topk_weights = picked / (picked.sum(-1, keepdim=True) + 1e-15)
    p = (_rms(_pooler(h, state, _pooler_path(tokens, triton_proj))) * _gain(w["pre_expert_norm"])).to(torch.bfloat16)
    local = (ids - data["local_start"]) % N_EXPERTS
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
        return _prepare_sglang(weights)
    return {"weights": weights,
            "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
            **_split_router_weights(weights["router"])}


def kernel_fn(data, state):
    if HAVE_SGLANG:
        return _kernel_sglang(data, state)
    return _kernel_torch(data, state)
