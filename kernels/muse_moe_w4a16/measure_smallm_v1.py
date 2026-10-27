# okc: problem=muse_moe_w4a16
"""MEASUREMENT-ONLY candidate (sonnet_v3's small-M path; design/fable-07-out.md D1 option 3). Run as `okc check`, NEVER submit.

kernel_fn is sonnet_v2b's path, byte for byte (the region from `import re` to the SMALLM block and the kernel_fn at the end are verbatim copies of
kernels/muse_moe_w4a16/sonnet_v2b.py / sonnet_v3.py; tests/test_moe_sonnet_v3.py compares them), so the judge's own timing of this file re-measures our board
entry (v2b). prepare() repacks the experts as v2b does, builds sonnet_v3's small-M tile layout + CUDA extension, then runs `_measure(state)` once, untimed,
within a 170 s wall budget, on inputs of reference.generate_input's recipe (re-written in `_gen_input`: seeded top-8 of uniform scores + heavy-tailed expert
popularity, expert ids rotated so experts 0-31 are the busiest GPU's, real routing, nothing synthetic-skewed; 3 seeds per M). Nothing measured feeds kernel_fn.
Every timing is a CUDA-graph replay after a READ-ONLY L2 flush (128 MB sum), like the judge and like measure_v1. Nothing runs under graph capture (guarded).

What is measured, for M in (1, 2, 4, 8, 12, 16) and seeds (11, 12, 13):
  G1 / G2  sweep (seed 11 only) of the small-M kernel of GEMM1 (+SwiGLU) / GEMM2 over configs nw/ks/nz = warps per CTA (slab N = 16 x nw weight rows) /
           K splits (slice a multiple of 256, <= 2048) / grid-z cap on the number of active experts handled at once (CTAs per SM follow from nw and the slice):
           the six best and the two worst, us per call (the stage alone in its own graph), the other GEMM at its default config.
  B        per (M, seed): `marlin` = v2b's stages one by one (align = local ids + moe_align, w13, act, w2, epi; sum), then the small-M pipeline at the default config
           (`def`) and at the swept best (`best`): route, g1, g2, epi stage times, their sum, and err = rel L2 of its output against kernel_fn's (Marlin) output.
  T        whole-pipeline graph times alternated 3x: v2b kernel_fn vs the small pipeline at the swept best, ratio = v2b / small.
  C        best vs default config per M.  Header lines: gpu/torch, `pairs / distinct / maxm` = local pairs, distinct local experts, max tokens on one expert.
Log (tail < 10 KB): lines `G1 moes ...`, `G2 moes ...`, `B moes ...`, `T moes ...`, `C moes ...`, then `MEAS-SMALLM done <s>`.
"""
import re
import types
from typing import TypedDict

import torch
import triton
import triton.language as tl

try:  # in the judge's image: SGLang's own modules, as baseline.py imports them
    from sgl_kernel.scalar_type import scalar_types

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

    HAVE_SGLANG = True
except ImportError:  # local runs without SGLang: plain PyTorch experts below
    HAVE_SGLANG = False


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8
# SGLang's fused_post_expert_norm_weighted_sum is eager below this many tokens (its _MIN_TOKENS_FOR_FUSED); we fuse below it.
EAGER_BELOW_TOKENS = 32
EPILOGUE_BLOCK_H = 1024
EPILOGUE_WARPS = 8

# Per-M config table: (largest M of the bucket, config name); the last entry (None) takes every larger M. Source: see the header.
CONFIG_TABLE = (
    (1, "b8.a0.rtt.t1024w8"),
    (8, "b8.a0.rff.t2048w8"),
    (16, "b16.a2.rft.t1024w8"),
    (32, "b32.a2.rff.sg"),
    (128, "b64.a3.rff.sg"),
    (None, "b64.a2.rff.sg"),
)
REDUCE_FLAGS = {"f": (False, True), "n": (False, False), "t": (True, False)}  # (use_atomic_add, use_fp32_reduce)
_NAME_RE = re.compile(r"^b(8|16|32|48|64)\.(a[0-4])\.r([fnt])([fnt])\.(sg|t(\d+)w(\d+))$")


def parse_config(name):
    """'b<bm>.<a0..a4>.r<GEMM1 reduce><GEMM2 reduce>.<sg | t<BLOCK_H>w<warps>>' -> a plain dict."""
    mt = _NAME_RE.match(name)
    if mt is None:
        raise ValueError(f"bad config name {name!r}")
    bm, align, r1, r2, epi, bh, warps = mt.groups()
    return {"name": name, "bm": int(bm), "align": align, "atomic1": REDUCE_FLAGS[r1][0], "fp32_1": REDUCE_FLAGS[r1][1],
            "atomic2": REDUCE_FLAGS[r2][0], "fp32_2": REDUCE_FLAGS[r2][1], "epilogue": epi,
            "block_h": int(bh) if bh else None, "warps": int(warps) if warps else None}


def config_for(num_tokens):
    """v2b (director): keep only judge-confirmed knob changes (check 3c949ec07d51465f vs 786d9deec42c4f89):
    M <= 1 atomic reduce (decode-1 1.461x -> 1.613x), M <= 8 epilogue t2048w8 (1.270x -> 1.283x); every other M uses
    SGLang's baseline block-size rule with align a0, fp32 reduce, and our fused epilogue below 32 tokens (SGLang's
    fused kernel at >= 32). The sweep's other picks lost on the judge's routing (decode-32 0.80x, verify-16 0.87x)."""
    if num_tokens <= 1:
        return parse_config("b8.a0.rtt.t1024w8")
    if num_tokens <= 8:
        return parse_config("b8.a0.rff.t2048w8")
    bm = 64
    for cand in (8, 16, 32, 48, 64):
        if num_tokens * 8 / 32 / cand < 0.9:
            bm = cand
            break
    epi = "t1024w8" if num_tokens < 32 else "sg"
    return parse_config(f"b{bm}.a0.rff.{epi}")


# ============================================================================================ Triton kernels
@triton.jit
def _local_ids_kernel(ids_ptr, out_ptr, n, lo, hi, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    local = tl.where((e >= lo) & (e < hi), e - lo, -1)
    tl.store(out_ptr + offs, local, mask=mask)


@triton.jit
def _epilogue_kernel(
    y_ptr, w_ptr, s_ptr, ids_ptr, out_ptr, eps,
    TOPK: tl.constexpr, KP: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """out[t] = sum_k live_k * s[t,k] * rmsnorm(y[t,k] * w) over the token's TOPK slots; FP32 throughout, one rounding at the store."""
    tok = tl.program_id(0)
    NCHUNK: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    slots = tl.arange(0, KP)
    in_k = slots < TOPK
    live = in_k & (tl.load(ids_ptr + tok * TOPK + slots, mask=in_k, other=-1) >= 0)
    s = tl.load(s_ptr + tok * TOPK + slots, mask=live, other=0.0).to(tl.float32)
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
    coef = tl.where(live, s * tl.rsqrt(sumsq / H + eps), 0.0)

    for c in tl.static_range(NCHUNK):
        offs = c * BLOCK_H + offs0
        hm = offs < H
        w = tl.load(w_ptr + offs, mask=hm, other=0.0).to(tl.float32)
        x = tl.load(y_ptr + row + offs[None, :], mask=live[:, None] & hm[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(x * coef[:, None], axis=0) * w
        tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=hm)


# ============================================================================================ epilogue (Triton + torch twin)
def local_ids_triton(topk_ids, start, count):
    ids = topk_ids.contiguous()
    out = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    n = ids.numel()
    _local_ids_kernel[(triton.cdiv(n, 1024),)](ids, out, n, start, start + count, BLOCK=1024, num_warps=4)
    return out


def local_ids_torch(topk_ids, start, count):
    e = topk_ids.to(torch.int32)
    return torch.where((e >= start) & (e < start + count), e - start, torch.full_like(e, -1))


def epilogue_triton(expert_out, topk_scores, slot_ids, norm_w, eps=POST_NORM_EPS, block_h=EPILOGUE_BLOCK_H, warps=EPILOGUE_WARPS):
    """expert_out [M, K, H] (BF16 from Marlin), topk_scores [M, K] FP32, slot_ids [M, K] int32 (< 0: not local), norm_w [H]."""
    y = expert_out.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    if tokens == 0:
        return out
    block_h = min(block_h, triton.next_power_of_2(hidden))
    _epilogue_kernel[(tokens,)](
        y, norm_w, topk_scores.contiguous(), slot_ids.contiguous(), out, eps,
        TOPK=topk, KP=triton.next_power_of_2(topk), H=hidden, BLOCK_H=block_h, num_warps=warps)
    return out


def epilogue_torch(expert_out, topk_scores, slot_ids, norm_w, eps=POST_NORM_EPS):
    """The same math in PyTorch (SGLang's eager branch): mask, FP32 (y * w) RMSNorm, score-weighted sum, one rounding."""
    tokens, topk, _ = expert_out.shape
    live = (slot_ids.reshape(tokens, topk) >= 0).unsqueeze(-1)
    f = torch.where(live, expert_out, torch.zeros((), dtype=expert_out.dtype, device=expert_out.device)).float()
    f = f * norm_w.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
    return (f * topk_scores.float().unsqueeze(-1)).sum(dim=1).to(expert_out.dtype)


def epilogue(expert_out, topk_scores, slot_ids, norm_w, eps=POST_NORM_EPS):
    if expert_out.is_cuda:
        return epilogue_triton(expert_out, topk_scores, slot_ids, norm_w, eps)
    return epilogue_torch(expert_out, topk_scores, slot_ids, norm_w, eps)


# ============================================================================================ SGLang path
def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _prepare_sglang(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8 (as baseline.py).
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def _align(local_ids, conf, bank, is_ep):
    """moe_align for the config's variant: a0 = baseline call; a1 = num_experts = local count; a2/a3 = a0/a1 (a2 with 256 experts) with ignore_invalid_expert
    (non-local slots dropped); a4 = SGLang's single-token align (M = 1, no expert map)."""
    bm, al = conf["bm"], conf["align"]
    if al == "a4":
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        return moe_align_single_token(local_ids, bm)
    if al == "a0":
        n_exp = N_EXPERTS if is_ep else bank["count"]
    else:
        n_exp = N_EXPERTS if al == "a2" else bank["count"]
    if al in ("a2", "a3"):
        return moe_align_block_size(local_ids, bm, n_exp, ignore_invalid_expert=True)
    return moe_align_block_size(local_ids, bm, n_exp)


def _kernel_sglang(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    if expert_map is not None:
        local_ids = local_ids_triton(data["topk_ids"], cfg["local_start"], cfg["local_count"])  # == expert_map[topk_ids]
    else:
        local_ids = data["topk_ids"].to(torch.int32).contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    conf = config_for(num_tokens)
    block_size_m = conf["bm"]
    sorted_ids, block_experts, n_padded = _align(local_ids, conf, bank, expert_map is not None)

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k,
        use_atomic_add=conf["atomic1"], use_fp32_reduce=conf["fp32_1"], **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n,
        use_atomic_add=conf["atomic2"], use_fp32_reduce=conf["fp32_2"], **common).view(num_tokens, TOP_K, size_k)
    if conf["epilogue"] == "sg":
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
            slot_expert_ids=local_ids)
    return epilogue_triton(y, topk_weights, local_ids, state["norm_w"], block_h=conf["block_h"], warps=conf["warps"])


# ============================================================================================ SMALLM BEGIN
# Small-M (M <= 16 tokens) streaming split-K W4A16 path: CUDA source (JIT-compiled by torch load_inline in the judge image, which has nvcc),
# the weight repack, the host-side plumbing. Layout / numerics are explained in the header of sonnet_v3.py.
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
                                                           int n, int lo, int nloc) {
    __shared__ int wc[4][32];
    __shared__ int s_cidx[32], s_cnt[32], s_nact;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_trigger();
    (&wc[0][0])[tid] = 0;
    __syncthreads();
    int e = -1;
    if (tid < n) {
        const int gl = ids[tid];
        e = (gl >= lo && gl < lo + nloc) ? gl - lo : -1;
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

void smallm_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor tab, int64_t lo, int64_t nloc) {
    c10::cuda::CUDAGuard guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    smallm_route_kernel<<<1, 128, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), tab.data_ptr<int>(), (int)ids.numel(), (int)lo, (int)nloc);
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
    "void smallm_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor tab, int64_t lo, int64_t nloc);\n"
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
# Defaults chosen from byte/occupancy reasoning only (no GPU numbers yet): see measure_smallm_v1.py for the sweep that fixes them.
SMALLM_CFG = (
    # (largest M, nw1, ks1, nz1, nw2, ks2, nz2)
    (1, 4, 4, 3, 4, 4, 3),
    (8, 4, 4, 6, 4, 4, 6),
    (16, 4, 4, 12, 4, 4, 12),
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
    """Route + GEMM1 (SwiGLU fused) + GEMM2 for M <= 16 tokens. Returns (y [M, 8, H] BF16, local_ids [M, 8] int32). Shape-only grids; no host sync."""
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
# ============================================================================================ SMALLM END


# ============================================================================================ plain-PyTorch path (no SGLang)
def _experts_torch(x, local_ids, state):
    """Per-slot expert outputs [M, K, H] in BF16 (what Marlin's second GEMM produces): the starter's math, every local expert on every
    token, kept for the slots routed to it. Rows of non-local slots stay zero."""
    from nvfp4 import dequantize

    w = state["weights"]
    start, count, inter = state["start"], state["count"], w["w2"].shape[2] * 2
    y = torch.zeros((*local_ids.shape, w["w2"].shape[1]), dtype=torch.bfloat16, device=x.device)
    for e in range(count):
        w13 = dequantize(w["w13"][start + e], w["w13_sf"][start + e], w["w13_gs"][start + e]).to(torch.bfloat16)
        h = x @ w13.T
        mid = torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]
        w2 = dequantize(w["w2"][start + e], w["w2_sf"][start + e], w["w2_gs"][start + e]).to(torch.bfloat16)
        y = torch.where((local_ids == e).unsqueeze(-1), (mid @ w2.T).unsqueeze(1), y)
    return y


def _kernel_torch(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    start, count = cfg["local_start"], cfg["local_count"]
    state = {**state, "start": start, "count": count}
    ids = data["topk_ids"]
    local_ids = local_ids_triton(ids, start, count) if x.is_cuda else local_ids_torch(ids, start, count)
    y = _experts_torch(x, local_ids, state)
    scores = data["topk_weights"].float()
    if x.is_cuda and x.shape[0] < EAGER_BELOW_TOKENS:
        return epilogue_triton(y, scores, local_ids, state["norm_w"])
    return epilogue_torch(y, scores, local_ids, state["norm_w"])

# ============================================================================================ measurement (MEASUREMENT-ONLY)
import time

BUDGET_S = 170.0        # prepare() may run 300 s; the whole run has 15 min (compile of the extension comes first)
SCREEN_REPS = 21
FINAL_REPS = 101
FLUSH_MB = 128          # read-only L2 flush buffer (> the 50 MB L2), like the judge's cold L2 and like measure_v1
TOL = 1e-2              # rel L2 of the small-M path against the unchanged kernel_fn (Marlin) output
MS = (1, 2, 4, 8, 12, 16)
SEEDS = (11, 12, 13)    # routing seeds per M
NW_SET = (2, 4, 8)
KS_SET = (2, 4, 8, 16)
NZ_SET = (2, 3, 4, 6, 8, 12, 16, 32)
DEFAULT_REPS_ORDER = 3  # alternations of (v2b, small) in the final comparison


def _gen_input(bs, seed, dist, device):
    """reference.generate_input(bs, "ep8", seed, dist=dist, rank="busiest") re-written here (a submission file cannot import the reference): the same
    random calls in the same order, so the routing, activations and gates are bit-identical (tests/test_moe_sonnet_v3.py compares them)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    if dist == "normal":
        z = torch.randn((N_EXPERTS,), generator=gen, device=device)
        u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    if dist == "skewed":
        heat = torch.rand((N_EXPERTS,), generator=gen, device=device).pow(6) * 4
        scores = scores + heat
        k = x.shape[-1]
        n = max(1, int(k * 0.01))
        idx = torch.randperm(k, generator=gen, device=device)[:n]
        xo = x.clone()
        xo[..., idx] = (xo[..., idx].float() * 30.0).to(x.dtype)
        x = xo
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=N_EXPERTS // 32)
    ids = (ids - 32 * pairs.argmax()) % N_EXPERTS
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32), "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": bs, "d_hidden": 4096, "d_expert": 4096, "n_experts": N_EXPERTS, "local_start": 0, "local_count": 32,
                       "top_k": TOP_K, "sharding": "ep8"}}


def gemm_cfgs(m, k_dim, groups):
    """(nw, ks, nz) candidates for one GEMM at M tokens: warps per CTA, K splits (slice a multiple of 256, <= 2048), grid-z cap (<= min(32, 8 M))."""
    top = min(32, m * TOP_K)
    out = []
    for nw in NW_SET:
        if groups % nw:
            continue
        for ks in KS_SET:
            if k_dim % ks or (k_dim // ks) % 256 or k_dim // ks > 2048:
                continue
            for nz in NZ_SET:
                if nz <= top:
                    out.append((nw, ks, nz))
    return out


def cfg_str(c):
    return f"{c[0]}/{c[1]}/{c[2]}"


# ---- the two pipelines, stage by stage (each stage a closure over `st`; calling them in order leaves the output in st["out"]) ----
def _marlin_stages(data, state):
    """sonnet_v2b's _kernel_sglang cut into stages: ids+align, w13, act, w2, epi."""
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]
    st = {}
    conf = config_for(num_tokens)
    bm = conf["bm"]

    def f_align():
        st["ids"] = local_ids_triton(data["topk_ids"], cfg["local_start"], cfg["local_count"])
        st["tw"] = data["topk_weights"].to(torch.float32).contiguous()
        st["sorted"], st["bexp"], st["npad"] = _align(st["ids"], conf, bank, expert_map is not None)

    common = dict(moe_block_size=bm, mul_topk_weights=False, is_ep=expert_map is not None, b_q_type=scalar_types.float4_e2m1f,
                  is_k_full=True, is_zp_float=False)

    def f_w13():
        st["cache2"] = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
        cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
        st["cache13"] = cache13
        st["c1"] = moe_wna16_marlin_gemm(
            x, cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n), bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None,
            state["workspace"], st["sorted"], st["bexp"], st["npad"], st["tw"], top_k=TOP_K, size_m=num_tokens, size_n=gemm1_n, size_k=size_k,
            use_atomic_add=conf["atomic1"], use_fp32_reduce=conf["fp32_1"], **common)

    def f_act():
        silu_and_mul(st["c1"].view(-1, gemm1_n), st["cache2"], expert_ids=st["ids"].view(-1), expert_step=1)

    def f_w2():
        cache3 = st["cache13"][: num_tokens * TOP_K * size_k].view(-1, size_k)
        st["y"] = moe_wna16_marlin_gemm(
            st["cache2"], cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, state["workspace"],
            st["sorted"], st["bexp"], st["npad"], st["tw"], top_k=1, size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n,
            use_atomic_add=conf["atomic2"], use_fp32_reduce=conf["fp32_2"], **common).view(num_tokens, TOP_K, size_k)

    def f_epi():
        if conf["epilogue"] == "sg":
            st["out"] = fused_post_expert_norm_weighted_sum(
                st["y"], st["tw"], norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=st["ids"])
        else:
            st["out"] = epilogue_triton(st["y"], st["tw"], st["ids"], state["norm_w"], block_h=conf["block_h"], warps=conf["warps"])
        return st["out"]

    return [("align", f_align), ("w13", f_w13), ("act", f_act), ("w2", f_w2), ("epi", f_epi)], st


def _small_stages(data, state, c1, c2):
    """The small-M pipeline cut into stages: route, g1 (GEMM1 + SwiGLU), g2, epi. c1 / c2 = (nw, ks, nz) of GEMM1 / GEMM2."""
    sm = state["smallm"]
    d, ext = sm["dims"], sm["ext"]
    x = data["hidden_states"].contiguous()
    m = x.shape[0]
    nz = min(SM_NEXP, m * TOP_K)
    st = {}

    def f_route():
        st["local"] = torch.empty(data["topk_ids"].shape, dtype=torch.int32, device=x.device)
        ext.smallm_route(data["topk_ids"].contiguous(), st["local"], sm["tab"], data["config"]["local_start"], data["config"]["local_count"])

    def f_g1():
        st["inter"] = torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device)
        ext.smallm_gemm(sm["w1"], sm["s1"], sm["gs1"], sm["lut"], x, sm["tab"], sm["part"], sm["cnt"][0], st["inter"], d["g1"], d["k1"] // 64,
                        d["k1"], c1[1], min(nz, c1[2]), d["inter"], 3, c1[0], True)

    def f_g2():
        st["y"] = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
        ext.smallm_gemm(sm["w2"], sm["s2"], sm["gs2"], sm["lut"], st["inter"], sm["tab"], sm["part"], sm["cnt"][1], st["y"], d["g2"], d["k2"] // 64,
                        d["k2"], c2[1], min(nz, c2[2]), d["hid"], 0, c2[0], False)

    def f_epi():
        conf = config_for(m)
        st["out"] = epilogue_triton(st["y"].view(m, TOP_K, d["hid"]), data["topk_weights"].to(torch.float32).contiguous(), st["local"], state["norm_w"],
                                    block_h=conf["block_h"] or EPILOGUE_BLOCK_H, warps=conf["warps"] or EPILOGUE_WARPS)
        return st["out"]

    return [("route", f_route), ("g1", f_g1), ("g2", f_g2), ("epi", f_epi)], st


class _RealEnv:
    """The GPU side: seeded inputs of the generate_input recipe, graph timing after a read-only flush (like the judge)."""

    def __init__(self, state):
        self.state = state
        self.dev = state["norm_w"].device

    def available(self):
        return HAVE_SGLANG and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing() and self.state.get("smallm") is not None

    def clock(self):
        return time.time()

    def banner(self):
        return f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} cap={torch.cuda.get_device_capability()}"

    def setup(self):
        self.flush = torch.ones(FLUSH_MB * 1024 * 1024 // 4, device=self.dev, dtype=torch.float32)
        self.sink = torch.zeros((), device=self.dev, dtype=torch.float32)

    def release(self):
        for a in ("flush", "sink"):
            if hasattr(self, a):
                delattr(self, a)
        torch.cuda.empty_cache()

    def prep(self, m, seed):
        data = _gen_input(m, seed, "normal", self.dev)
        y0 = kernel_fn(data, self.state).clone()
        torch.cuda.synchronize()
        ids = data["topk_ids"]
        local = (ids >= 0) & (ids < 32)
        cnt = torch.bincount(ids[local].long(), minlength=32)
        info = f"pairs={int(local.sum())} distinct={int((cnt > 0).sum())} maxm={int(cnt.max())}"
        return dict(m=m, seed=seed, data=data, y0=y0, norm=float(y0.float().norm()), info=info)

    def err(self, ctx, y):
        n = float((y.float() - ctx["y0"].float()).norm())
        return n / max(ctx["norm"], 1e-12) if n == n else float("inf")

    def graph_time(self, fn, reps):
        """(median us of graph replays, each preceded by a read-only flush; the replayed output, cloned)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        ts = []
        for _ in range(reps):
            torch.sum(self.flush, dim=0, out=self.sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
        ts.sort()
        res = out.clone() if isinstance(out, torch.Tensor) else None
        del g
        return ts[len(ts) // 2], res

    def run_stages(self, fns, reps, name=None):
        """Eager run of all stages (fills `st`), then the graph time of one stage (name) or of all stages in one graph."""
        for _, f in fns:
            f()
        torch.cuda.synchronize()
        if name is None:
            def allf():
                r = None
                for _, f in fns:
                    r = f()
                return r
            return self.graph_time(allf, reps)
        f = dict(fns)[name]
        return self.graph_time(f, reps)

    def time_base(self, ctx, reps):
        return self.graph_time(lambda: kernel_fn(ctx["data"], self.state), reps)


def _short(e):
    return f"{type(e).__name__}: {str(e)[:80]}".replace("\n", " ")


def _sweep(env, ctx, which, base_c1, base_c2, deadline, reps):
    """Times one GEMM ('g1' or 'g2') over its configs with the other GEMM at its default; the stage fns run once eagerly first. Returns
    ([(us, cfg)] ascending, n_failed)."""
    d = env.state["smallm"]["dims"]
    cfgs = gemm_cfgs(ctx["m"], d["k1"] if which == "g1" else d["k2"], d["g1"] if which == "g1" else d["g2"])
    res, fails = [], 0
    for c in cfgs:
        if env.clock() > deadline:
            break
        c1, c2 = (c, base_c2) if which == "g1" else (base_c1, c)
        try:
            fns, st = _small_stages(ctx["data"], env.state, c1, c2)
            us, _ = env.run_stages(fns, reps, name=which)
            res.append((us, c))
        except Exception as ex:  # noqa: BLE001
            fails += 1
            if fails == 1:
                print(f"MEAS-SMALLM fail {which} {cfg_str(c)}: {_short(ex)}")
    res.sort()
    return res, fails


def _measure(state, env=None):
    """MEASUREMENT: see the module docstring. `env` is injected by tests (stubbed timing)."""
    env = env or _RealEnv(state)
    if not env.available():
        print("MEAS-SMALLM skipped: no cuda, no sglang or no small-M path")
        return
    t0 = env.clock()
    deadline = t0 + BUDGET_S
    print(f"MEAS-SMALLM {env.banner()} budget={BUDGET_S}s")
    env.setup()
    lines = []
    try:
        for m in MS:
            cfg0 = smallm_cfg(m)
            d1 = (cfg0["nw1"], cfg0["ks1"], cfg0["nz1"])
            d2 = (cfg0["nw2"], cfg0["ks2"], cfg0["nz2"])
            sweep_ctx = None
            best1, best2 = d1, d2
            for si, seed in enumerate(SEEDS):
                if env.clock() > deadline:
                    lines.append(f"B moes m={m} seed={seed} skipped (budget)")
                    continue
                ctx = env.prep(m, seed)
                if si == 0:
                    sweep_ctx = ctx
                    r1, f1 = _sweep(env, ctx, "g1", d1, d2, deadline, SCREEN_REPS)
                    r2, f2 = _sweep(env, ctx, "g2", d1, d2, deadline, SCREEN_REPS)
                    if r1:
                        best1 = r1[0][1]
                    if r2:
                        best2 = r2[0][1]
                    lines.append(f"G1 moes m={m} {ctx['info']} n={len(r1)} fail={f1} top: " + " ".join(f"{cfg_str(c)}:{u:.1f}" for u, c in r1[:6])
                                 + " | worst " + " ".join(f"{cfg_str(c)}:{u:.1f}" for u, c in r1[-2:]))
                    lines.append(f"G2 moes m={m} n={len(r2)} fail={f2} top: " + " ".join(f"{cfg_str(c)}:{u:.1f}" for u, c in r2[:6])
                                 + " | worst " + " ".join(f"{cfg_str(c)}:{u:.1f}" for u, c in r2[-2:]))
                # stage split of both pipelines at the default and at the swept-best config, then totals alternated (cancels drift)
                row = f"B moes m={m} seed={seed} {ctx['info']}"
                try:
                    mf, mst = _marlin_stages(ctx["data"], state)
                    mt = {nm: env.run_stages(mf, FINAL_REPS, name=nm)[0] for nm, _ in mf}
                    row += " | marlin " + " ".join(f"{nm}={v:.1f}" for nm, v in mt.items()) + f" sum={sum(mt.values()):.1f}"
                except Exception as ex:  # noqa: BLE001
                    row += f" | marlin fail {_short(ex)}"
                for tag, c1, c2 in (("def", d1, d2), ("best", best1, best2)):
                    try:
                        sf, sst = _small_stages(ctx["data"], state, c1, c2)
                        stt = {nm: env.run_stages(sf, FINAL_REPS, name=nm)[0] for nm, _ in sf}
                        _, out = env.run_stages(sf, 3)
                        row += f" | {tag} {cfg_str(c1)} {cfg_str(c2)} " + " ".join(f"{nm}={v:.1f}" for nm, v in stt.items()) \
                            + f" sum={sum(stt.values()):.1f} err={env.err(ctx, out):.1e}"
                    except Exception as ex:  # noqa: BLE001
                        row += f" | {tag} fail {_short(ex)}"
                    if best1 == d1 and best2 == d2:
                        break
                lines.append(row)
                try:
                    sf, _ = _small_stages(ctx["data"], state, best1, best2)
                    ta, tb = [], []
                    for _ in range(DEFAULT_REPS_ORDER):
                        ta.append(env.time_base(ctx, FINAL_REPS)[0])
                        tb.append(env.run_stages(sf, FINAL_REPS)[0])
                    ta.sort()
                    tb.sort()
                    lines.append(f"T moes m={m} seed={seed} v2b={ta[len(ta) // 2]:.1f} small={tb[len(tb) // 2]:.1f} ratio={ta[len(ta) // 2] / tb[len(tb) // 2]:.3f}")
                except Exception as ex:  # noqa: BLE001
                    lines.append(f"T moes m={m} seed={seed} fail {_short(ex)}")
            lines.append(f"C moes m={m} best g1={cfg_str(best1)} g2={cfg_str(best2)} default g1={cfg_str(d1)} g2={cfg_str(d2)}")
    finally:
        env.release()
    out = "\n".join(lines)
    print(out[-9000:])
    print(f"MEAS-SMALLM done {env.clock() - t0:.0f}s")


# ============================================================================================ entry points
def prepare(weights):
    """Called once, untimed. sonnet_v2b's repack (Marlin layout) and the small-M tile layout, then the synthetic sweep (MEASUREMENT-ONLY: kernel_fn stays v2b)."""
    print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")
    if HAVE_SGLANG:
        state = _prepare_sglang(weights)
        state["smallm"] = None
        if torch.cuda.is_available():
            try:
                state["smallm"] = prepare_smallm(weights)
            except Exception as ex:  # noqa: BLE001
                print(f"okc-diag smallm unavailable: {_short(ex)}")
    else:
        state = {"weights": weights, "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    _measure(state)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    if HAVE_SGLANG:
        return _kernel_sglang(data, state)
    return _kernel_torch(data, state)
