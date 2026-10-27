# okc: problem=muse_moe_w4a16
# sonnet_v6 -- sonnet_v5 (board 1.486x: v4's Marlin / small-M / large-M paths + the unified w4_moe5_kernel "Alt A") plus a DEEPER WEIGHT PIPELINE in the same
# template, w4_moe6_kernel<FUSE, NW, MR, D, RB, RS, LB, DBG> (card C21b; fable-09 D4 S1 decision (ii) after the C20 check a78c1b9c: WSTREAM 4.0 TB/s vs
# NOCOMPUTE 2.3 TB/s, occupancy twins < +15 %, stamps: per item stage 19 % / stream 67 % / wait 10 % / fix 5 % / epi 4 %, the stream phase itself at
# 9.6 GB/s per CTA x 396 resident = 3.8 TB/s). Numerics BIT-IDENTICAL to v5 = v4 at the same K split (same dequant, mma order, split-K fixup order, epilogue);
# only how the weight bytes arrive changes:
#   D / RB   register mode: batches of D k64 iterations per 16-row group, RB batches in flight beyond the one consumed (NBUF = RB + 1 buffers). v5 = D 4/MR, RB 1
#            (3 KB per warp in flight, 48 registers). v6 small M (4 warps x 1 group): D 1, RB 7 = 5.25 KB per warp at 153 registers, LB 3; large M alternative
#            (table switch): D 1, RB 3 = 4.5 KB per warp at 186 registers, LB 2.
#   RS       ring mode: a per-warp cp.async ring of RS stages in shared memory (576 B per tile = exactly the DRAM bytes; cp.async.cg 16 B per lane + 4 lanes for
#            the scales; wait_group RS-2; the dequant reads word + scales from the stage). v6 large M default: (4 x 2) D 1 RS 8 = 8 KB per warp in flight at
#            152 registers, 2 CTAs/SM by smem (104 KB): 64 KB per SM in flight (v5: 36 KB).
#   CONTINUITY  the first RB (RS-1) batches of the CTA's next work item are issued before the current item's fixup / epilogue / next staging: the stream never
#            drains at an item boundary (C20: ~38 % of an item was non-streaming).
#   NB 2     stage-once default at M >= 17 (C20: nb2 +8-13 %, nb4 no better); LB 3 kept where the registers allow (v5's 168-register cap holds for D 1 RB 3).
# prepare(): v5's preparation (v4's and v5's extensions + self-checks = the fallbacks), then the v6 extension (6 instantiations: the register default, the ring
# alternative for a table switch, the small-M default) and a BITWISE self-check of v6 against v5 (v4) on seeded inputs for M in (1, 2, 4, 8, 12, 16, 32, 33, 128, 256);
# a bucket whose output differs in any bit keeps v5's path (MOE6_ENABLE / state["moe6_ok"]). Marlin stays for M > 256 and for anything that fails.
#
# PREDICTION (est.; base = the C20 check's v5 numbers on seeds 11-13 and the judge medians of v5's submit 99f12b5d; measured by measure_ring_v1.py):
#   mechanism: the stream phase is already near the HBM ceiling while it runs; the gain is the duty cycle: continuity hides most of wait + fix + epi (19 % of
#   the item) and half of the staging (NB 2), and 1.5-2.7x the bytes in flight removes the exposed compute (NOCOMPUTE 0.81x) -> item 12.0 -> ~8.5-9.5 us.
#   bs 256: GEMM1 (seed 11-13) 225 -> 160-185 us, GEMM2 118 -> 85-100 | bs 128: 175 -> 125-145 / 93 -> 68-80 | bs 32: 100 -> 72-85 / 56 -> 42-50
#   bs 16: GEMM1 46 -> 36-41, GEMM2 34 -> 27-31 | bs 8: 50/36 -> 40-45/29-33 | bs 1: 26/36 -> 22-24/31-34
#   judge shapes (v5 submit medians): bs256 1.49 -> 1.8-2.0, bs128 1.55 -> 1.85-2.05, bs32 1.17 -> 1.35-1.5, bs16/8/1 +5-12 %  =>  moe 1.486 -> ~1.65-1.80
#   (kill if the measurement shows < +12 % on GEMM1 bs256 for every v6 variant; the ring/register twins then say whether depth or the fixup is the residual).
#
# DECISION RULE for the director after measure_ring_v1 (B lines: x = v5 / variant per GEMM, medians over 3 seeds; S lines: phase shares under v6):
#   (a) best v6 variant >= +12 % on GEMM1 bs256 and bitwise OK  -> promote this file with MOE6_CFG set to that variant (register vs ring, NB, LB);
#   (b) register and ring within noise of each other and both < +12 % while stream share stays > 60 %  -> the residual is the fixup/staging duty cycle, not depth:
#       next build = fixup overlap (FP32 atomic epilogue is NOT bitwise; instead KS 2 with the 16-row act class or a per-CTA partial ring);
#   (c) ring >= register by > 5 % at bs 256  -> the ring is the default; sweep RS 6/8/12 once;  (d) any `!INVALID` or bw != OK line  -> fix the harness first.
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

#if __CUDA_ARCH__ < 800
// sm_75 emulation of the mma fragment layout with shuffles; not inlined (keeps the development build of the many instantiations fast).
__device__ __noinline__ void mma_emul(float* c, const u32* a, u32 b0, u32 b1) {
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
}
#endif

// D[16 x 8] += A[16 x 16] * B[16 x 8], bf16 operands, FP32 accumulate. A: row-major regs a0..a3, B: b0, b1 (PTX m16n8k16 fragment layout).
__device__ __forceinline__ void mma16816(float (&c)[4], const u32 (&a)[4], u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    mma_emul(c, a, b0, b1);  // sm_75 emulation (development card without BF16 tensor cores)
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
# Tuned on the judge H200 by measure_smallm_v1.py (job 7e8e3866f7f64f2a, generate_input routing, 3 seeds per M).
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

# ============================================================================================ LARGEM BEGIN
# ---- large-M (M = 17..256 tokens) kernels: same tile layout, A-operand dequant and B-operand staging as the small-M kernel, but an expert's tokens are cut into
# chunks of <= LM_TC tokens (a chunk = one work item), NT = 1/2/4 n8 token tiles per chunk, MR 16-row groups per warp (the B fragments of a k16 tile are
# loaded once and used for MR weight tiles), split-K as before (the last CTA of a (chunk, n-block) adds the partials in slice order).
LM_TC = 32           # tokens per chunk
LM_CTW = 4           # chunk-table entry: [expert, tokens in chunk, offset into the slot list, pad]; ctab[0] = n_chunks, entries from ctab[4]
LM_MAX_M = 256       # tokens handled (<= 2048 slots)
LM_SMEM_MAX = 163840

_LARGEM_EXTRA_SRC = r"""
#define LM_TC 32
#define LM_CTW 4

template <int NT, int MR, int D>
__device__ __forceinline__ void do_batch_l(const uint4 (&w)[MR][D], const uint2 (&s)[MR][D], int kk0, u32 lane_addr, int kp_bytes,
                                           const u32* s_lut, float (&acc)[MR][NT][4]) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
            u32 bf[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                ldmatrix_x4(bf[nt], lane_addr + (u32)(nt * 8 * kp_bytes) + (u32)((kk0 + d * 4 + jp * 2) * 32));
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = jp * 2 + jj;
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    const u32 wv[4] = {w[r][d].x, w[r][d].y, w[r][d].z, w[r][d].w};
                    u32 q[4];
                    unpack_word(wv[j], q);
                    const u32 sc0 = s_lut[(s[r][d].x >> (8 * j)) & 255u];
                    const u32 sc1 = s_lut[(s[r][d].y >> (8 * j)) & 255u];
                    const u32 a[4] = {mul_bf16x2(q[0], sc0), mul_bf16x2(q[1], sc1), mul_bf16x2(q[2], sc0), mul_bf16x2(q[3], sc1)};
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) mma16816(acc[r][nt], a, bf[nt][2 * jj], bf[nt][2 * jj + 1]);
                }
            }
        }
    }
}

template <int MR, int D>
__device__ __forceinline__ void prefetch_batch_l(uint4 (&w)[MR][D], uint2 (&s)[MR][D], const uint4* wp, const uint2* sp, int wstride, int sstride) {
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int d = 0; d < D; ++d) {
            ldg_stream4(w[r][d], wp + (u64)r * wstride + d * 32);
            ldg_stream2(s[r][d], sp + (u64)r * sstride + d * 8);
        }
}

// One chunk of tokens x one 16*MR-row group of this warp x one K slice. Returns false if this CTA is not the last of its split-K group.
template <bool FUSE, int NW, int MR, int NT>
__device__ __forceinline__ void chunk_gemm(uint4 (&w0)[MR][4 / MR], uint2 (&s0)[MR][4 / MR], const uint4* wp, const uint2* sp, int wstride, int sstride, int nbatch,
                                           u32 lane_addr, int kp_bytes,
                                           const u32* s_lut, float4* part, int* cnt, u16* out, const int* sl, int mc, float fscale, int z, int nb, int nbn,
                                           int ks, int KS, int g0, int out_stride, int* s_flag) {
    constexpr int D = 4 / MR;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    uint4 w1[MR][D];
    uint2 s1[MR][D];
    float acc[MR][NT][4];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) acc[r][nt][0] = acc[r][nt][1] = acc[r][nt][2] = acc[r][nt][3] = 0.f;
    for (int b = 0; b < nbatch; b += 2) {
        if (b + 1 < nbatch) prefetch_batch_l<MR, D>(w1, s1, wp + (b + 1) * D * 32, sp + (b + 1) * D * 8, wstride, sstride);
        do_batch_l<NT, MR, D>(w0, s0, b * D * 4, lane_addr, kp_bytes, s_lut, acc);
        if (b + 1 < nbatch) {
            if (b + 2 < nbatch) prefetch_batch_l<MR, D>(w0, s0, wp + (b + 2) * D * 32, sp + (b + 2) * D * 8, wstride, sstride);
            do_batch_l<NT, MR, D>(w1, s1, (b + 1) * D * 4, lane_addr, kp_bytes, s_lut, acc);
        }
    }
    if (KS > 1) {
        const u64 pbase = (((((u64)z * nbn + nb) * KS + ks) * NW + warp) * MR) * 128 + lane;
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                part[pbase + r * 128 + nt * 32] = make_float4(acc[r][nt][0], acc[r][nt][1], acc[r][nt][2], acc[r][nt][3]);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int prev = atomicAdd(&cnt[z * nbn + nb], 1);
            const int last = (prev == KS - 1);
            *s_flag = last;
            if (last) cnt[z * nbn + nb] = 0;
        }
        __syncthreads();
        if (!*s_flag) return;
        __threadfence();
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint4 v[8];
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) {
                    if (k2 < KS) {
                        const u64 pb = (((((u64)z * nbn + nb) * KS + k2) * NW + warp) * MR) * 128 + lane + r * 128 + nt * 32;
                        v[k2] = ldcg4(part + pb);
                    }
                }
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) {
                    if (k2 < KS) {
                        a0 += __uint_as_float(v[k2].x); a1 += __uint_as_float(v[k2].y); a2 += __uint_as_float(v[k2].z); a3 += __uint_as_float(v[k2].w);
                    }
                }
                acc[r][nt][0] = a0; acc[r][nt][1] = a1; acc[r][nt][2] = a2; acc[r][nt][3] = a3;
            }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const int g = g0 + r;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tk = nt * 8 + 2 * t4 + c;
                if (tk < mc) {
                    u16* o = out + (u64)sl[tk] * out_stride;
                    if (FUSE) {
                        const float gt = acc[r][nt][c] * fscale, up = acc[r][nt][2 + c] * fscale;
                        o[g * 8 + gid] = f2bf((gt / (1.0f + expf(-gt))) * up);
                    } else {
                        o[g * 16 + gid] = f2bf(acc[r][nt][c] * fscale);
                        o[g * 16 + 8 + gid] = f2bf(acc[r][nt][2 + c] * fscale);
                    }
                }
            }
        }
    }
}

// grid (G / (NW * MR), KS, nz); work item z = chunk z of the chunk table (z loops when the grid's z is smaller than the number of chunks).
template <bool FUSE, int NW, int MR>
__global__ void __launch_bounds__(NW * 32)
w4_largem_kernel(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ gsc, const u32* __restrict__ lut,
                 const u16* __restrict__ act, const int* __restrict__ ctab, const int* __restrict__ slotl, float4* __restrict__ part,
                 int* __restrict__ cnt, u16* __restrict__ out, int G, int NIT, int K, int KS, int out_stride, int act_shift) {
    constexpr int D = 4 / MR;
    extern __shared__ __align__(16) u8 smem_raw[];
    __shared__ int s_flag;
    u32* s_lut = reinterpret_cast<u32*>(smem_raw);
    u16* s_act = reinterpret_cast<u16*>(smem_raw + 1024);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2;
    const int nb = blockIdx.x, ks = blockIdx.y, nbn = gridDim.x;
    if (FUSE) pdl_wait();
    if (FUSE) pdl_trigger();
    const int Kc = K / KS, nit = Kc >> 6, KP = Kc + 8, kp_bytes = KP * 2;
    const int n_chunks = ctab[0];
    const int g0 = (nb * NW + warp) * MR;
    const u32 lane_addr = smem_u32(s_act) + (u32)((((lane & 7) * KP) + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8) * 2);

    for (int z = blockIdx.z; z < n_chunks; z += gridDim.z) {
        const int* ct = ctab + 4 + LM_CTW * z;
        const int e = ct[0], mc = ct[1];
        const int* sl = slotl + ct[2];
        const uint4* wp = Wq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 32 + lane;
        const uint2* sp = Sq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 8 + gid;
        uint4 w0[MR][D];
        uint2 s0[MR][D];
        prefetch_batch_l<MR, D>(w0, s0, wp, sp, NIT * 32, NIT * 8);
        __syncthreads();  // previous chunk is done with s_act / s_flag
        for (int i = tid; i < 256; i += NW * 32) s_lut[i] = lut[i];
        if (!FUSE) pdl_wait();
        const int nrows = mc > 16 ? 32 : (mc > 8 ? 16 : 8);
        const int cpr = Kc >> 3;
        for (int idx = tid; idx < nrows * cpr; idx += NW * 32) {
            const int row = idx / cpr, c = idx - row * cpr;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (row < mc) v = reinterpret_cast<const uint4*>(act + (u64)(sl[row] >> act_shift) * K + (u64)ks * Kc)[c];
            *reinterpret_cast<uint4*>(s_act + row * KP + c * 8) = v;
        }
        __syncthreads();
        const float fscale = gsc[e] * 1073741824.0f;
        if (mc <= 8) chunk_gemm<FUSE, NW, MR, 1>(w0, s0, wp, sp, NIT * 32, NIT * 8, nit / D, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag);
        else if (mc <= 16) chunk_gemm<FUSE, NW, MR, 2>(w0, s0, wp, sp, NIT * 32, NIT * 8, nit / D, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag);
        else chunk_gemm<FUSE, NW, MR, 4>(w0, s0, wp, sp, NIT * 32, NIT * 8, nit / D, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag);
    }
}

// One CTA of 1024 threads: local ids, expert-sorted slot list (slots ascending within an expert), chunk table (experts ascending, <= LM_TC tokens per chunk).
// n <= 2048 slots. Warp w owns expert w: counts with ballots, then (after the offsets) writes its slots.
__global__ void __launch_bounds__(1024) largem_route_kernel(const int* __restrict__ ids, int* __restrict__ local_ids, int* __restrict__ ctab,
                                                            int* __restrict__ slotl, int n, int lo, int nloc) {
    __shared__ int s_e[2048];
    __shared__ int s_cnt[32], s_off[32], s_cb[32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_trigger();
    for (int s = tid; s < n; s += 1024) {
        const int gl = ids[s];
        const int e = (gl >= lo && gl < lo + nloc) ? gl - lo : -1;
        s_e[s] = e;
        local_ids[s] = e;
    }
    __syncthreads();
    int c = 0;
    for (int base = 0; base < n; base += 32) {
        const int s = base + lane;
        const int e = s < n ? s_e[s] : -1;
        c += __popc(__ballot_sync(FULLMASK, e == warp));
    }
    if (lane == 0) s_cnt[warp] = c;
    __syncthreads();
    if (warp == 0) {
        const int cn = s_cnt[lane];
        const int nch = (cn + LM_TC - 1) / LM_TC;
        int po = cn, pc = nch;  // inclusive scans
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int a = __shfl_up_sync(FULLMASK, po, d), b = __shfl_up_sync(FULLMASK, pc, d);
            if (lane >= d) { po += a; pc += b; }
        }
        s_off[lane] = po - cn;
        s_cb[lane] = pc - nch;
        for (int i = 0; i < nch; ++i) {
            int* t = ctab + 4 + LM_CTW * (pc - nch + i);
            t[0] = lane;
            t[1] = min(LM_TC, cn - i * LM_TC);
            t[2] = po - cn + i * LM_TC;
            t[3] = 0;
        }
        const int total = __shfl_sync(FULLMASK, pc, 31);
        if (lane == 0) ctab[0] = total;
    }
    __syncthreads();
    int run = 0;
    const int off = s_off[warp];
    for (int base = 0; base < n; base += 32) {
        const int s = base + lane;
        const int e = s < n ? s_e[s] : -1;
        const u32 m = __ballot_sync(FULLMASK, e == warp);
        if (e == warp) slotl[off + run + __popc(m & ((1u << lane) - 1u))] = s;
        run += __popc(m);
    }
}
"""

_LARGEM_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

static const int LM_SMEM_MAX = 163840;

template <bool FUSE, int NW, int MR>
static void launch_largem_t(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab,
                            torch::Tensor slotl, torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K,
                            int64_t KS, int64_t nz, int64_t out_stride, int64_t act_shift) {
    const int Kc = (int)(K / KS);
    const size_t smem = 1024 + (size_t)32 * (Kc + 8) * 2;
    auto fn = w4_largem_kernel<FUSE, NW, MR>;
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
        inited = true;
    }
    const uint4* a_wq = reinterpret_cast<const uint4*>(wq.data_ptr());
    const uint2* a_sq = reinterpret_cast<const uint2*>(sq.data_ptr());
    const float* a_gs = gs.data_ptr<float>();
    const u32* a_lut = reinterpret_cast<const u32*>(lut.data_ptr());
    const u16* a_act = reinterpret_cast<const u16*>(act.data_ptr());
    const int* a_ct = ctab.data_ptr<int>();
    const int* a_sl = slotl.data_ptr<int>();
    float4* a_part = reinterpret_cast<float4*>(part.data_ptr());
    int* a_cnt = cnt.data_ptr<int>();
    u16* a_out = reinterpret_cast<u16*>(out.data_ptr());
    int i_G = (int)G, i_NIT = (int)NIT, i_K = (int)K, i_KS = (int)KS, i_os = (int)out_stride, i_as = (int)act_shift;
    void* args[] = {&a_wq, &a_sq, &a_gs, &a_lut, &a_act, &a_ct, &a_sl, &a_part, &a_cnt, &a_out, &i_G, &i_NIT, &i_K, &i_KS, &i_os, &i_as};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(G / (NW * MR)), (unsigned)KS, (unsigned)nz);
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

void largem_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl,
                 torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz,
                 int64_t out_stride, int64_t act_shift, int64_t nw, int64_t mr, bool fuse) {
    c10::cuda::CUDAGuard guard(act.device());
#define LM_CASE(F, W, R) launch_largem_t<F, W, R>(wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, G, NIT, K, KS, nz, out_stride, act_shift)
    if (fuse) {
        if (nw == 4 && mr == 1) LM_CASE(true, 4, 1); else if (nw == 8 && mr == 1) LM_CASE(true, 8, 1);
        else if (nw == 4 && mr == 2) LM_CASE(true, 4, 2); else LM_CASE(true, 8, 2);
    } else {
        if (nw == 4 && mr == 1) LM_CASE(false, 4, 1); else if (nw == 8 && mr == 1) LM_CASE(false, 8, 1);
        else if (nw == 4 && mr == 2) LM_CASE(false, 4, 2); else LM_CASE(false, 8, 2);
    }
#undef LM_CASE
}

void largem_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc) {
    c10::cuda::CUDAGuard guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    largem_route_kernel<<<1, 1024, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctab.data_ptr<int>(), slotl.data_ptr<int>(),
                                                (int)ids.numel(), (int)lo, (int)nloc);
}

void largem_init() {
    cudaFuncSetAttribute(w4_largem_kernel<true, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
    cudaFuncSetAttribute(w4_largem_kernel<true, 8, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
    cudaFuncSetAttribute(w4_largem_kernel<true, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
    cudaFuncSetAttribute(w4_largem_kernel<true, 8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
    cudaFuncSetAttribute(w4_largem_kernel<false, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
    cudaFuncSetAttribute(w4_largem_kernel<false, 8, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
    cudaFuncSetAttribute(w4_largem_kernel<false, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
    cudaFuncSetAttribute(w4_largem_kernel<false, 8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, LM_SMEM_MAX);
}
"""
_LARGEM_CPP_SRC = (
    "void largem_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl, "
    "torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz, "
    "int64_t out_stride, int64_t act_shift, int64_t nw, int64_t mr, bool fuse);\n"
    "void largem_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc);\n"
    "void largem_init();\n")
# Large-M (17..256 tokens) path: Python side. The weights use the SMALLM tile layout (no second copy); only scratch buffers are new.
LM_ZMAX = 8 * LM_MAX_M // LM_TC + SM_NEXP   # bound on the number of chunks: sum over experts of ceil(m_e / 32) <= 8 M / 32 + 32
# bucket (largest M) -> launch configuration per GEMM: nw warps x mr 16-row groups per warp (rows per CTA = 16 nw mr), K splits and grid-z cap.
LARGEM_CFG = {
    32: dict(nw1=4, mr1=2, ks1=4, nz1=16, nw2=4, mr2=2, ks2=4, nz2=16),
    128: dict(nw1=4, mr1=2, ks1=4, nz1=32, nw2=4, mr2=2, ks2=4, nz2=32),
    256: dict(nw1=4, mr1=2, ks1=4, nz1=48, nw2=4, mr2=2, ks2=4, nz2=48),
}
LARGEM_ENABLE = {32: True, 128: True, 256: True}   # per bucket; False = Marlin (sonnet_v2b) for that bucket


def largem_bucket(m):
    for b in (32, 128, 256):
        if m <= b:
            return b
    return None


def largem_cfg(m):
    """Config of the bucket of m, or None (m <= 16 is the small-M path, m > 256 Marlin, bucket disabled)."""
    b = largem_bucket(m)
    if m <= SM_MAX_M or b is None or not LARGEM_ENABLE.get(b, False):
        return None
    return LARGEM_CFG[b]


_LARGEM_EXT = {}


def largem_ext(flags=()):
    """The compiled large-M extension (largem_gemm / largem_route / largem_init), built once per process and flag set."""
    key = tuple(flags)
    if key not in _LARGEM_EXT:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        src = _SMALLM_DEVICE_SRC + _LARGEM_EXTRA_SRC + _LARGEM_HOST_SRC
        tag = hashlib.sha1((src + repr(key)).encode()).hexdigest()[:12]
        _LARGEM_EXT[key] = load_inline(
            name=f"okc_largem_{tag}", cpp_sources=_LARGEM_CPP_SRC, cuda_sources=src,
            functions=["largem_gemm", "largem_route", "largem_init"], extra_cuda_cflags=["-O3", *key], verbose=False)
    return _LARGEM_EXT[key]


def prepare_largem(sm, ext=None, flags=(), max_m=LM_MAX_M, cfgs=None):  # cfgs: kept for callers; the buffers are sized for the worst case
    """Untimed. Scratch for the large-M kernels (chunk table, expert-sorted slot list, split-K partials, counters); the packed weights are sm's."""
    d, dev = sm["dims"], sm["w1"].device
    zmax = 8 * max_m // LM_TC + SM_NEXP
    need = max(d["g1"], d["g2"]) * 8        # partials for any K split up to 8 (the sweep of measure_largem_v1 goes there); touched memory is only what runs
    nbn = max(d["g1"], d["g2"]) // 4        # n-blocks per chunk for the smallest CTA (4 warps x 1 group)
    lm = {
        "ctab": torch.zeros(4 + LM_CTW * zmax, dtype=torch.int32, device=dev),
        "slotl": torch.zeros(8 * max_m, dtype=torch.int32, device=dev),
        "part": torch.empty(zmax * need * 128 * 4, dtype=torch.float32, device=dev),
        "cnt": torch.zeros((2, zmax * nbn), dtype=torch.int32, device=dev),
        "zmax": zmax, "max_m": max_m,
        "ext": ext if ext is not None else largem_ext(flags),
    }
    lm["ext"].largem_init()
    return lm


def largem_experts(x, topk_ids, sm, lm, start, count, cfg):
    """Route + GEMM1 (SwiGLU fused) + GEMM2 for 1..256 tokens, tokens of an expert cut into chunks of <= 32. Returns (y [M, 8, H] BF16, local_ids [M, 8])."""
    m = x.shape[0]
    d, ext = sm["dims"], lm["ext"]
    assert m <= lm["max_m"] and count == SM_NEXP
    ids = topk_ids.to(torch.int32).contiguous()
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
    ext.largem_route(ids, local_ids, lm["ctab"], lm["slotl"], start, count)
    inter = torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
    zb = min(lm["zmax"], 8 * m // LM_TC + SM_NEXP)  # shape-only bound on the chunk count
    for k_dim, groups, ks, rows in ((d["k1"], d["g1"], cfg["ks1"], cfg["nw1"] * cfg["mr1"]), (d["k2"], d["g2"], cfg["ks2"], cfg["nw2"] * cfg["mr2"])):
        assert k_dim % ks == 0 and (k_dim // ks) % 256 == 0 and k_dim // ks <= 2048 and groups % rows == 0 and ks <= 8, (k_dim, groups, ks, rows)
    ext.largem_gemm(sm["w1"], sm["s1"], sm["gs1"], sm["lut"], x.contiguous(), lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][0], inter,
                    d["g1"], d["k1"] // 64, d["k1"], cfg["ks1"], min(zb, cfg["nz1"]), d["inter"], 3, cfg["nw1"], cfg["mr1"], True)
    ext.largem_gemm(sm["w2"], sm["s2"], sm["gs2"], sm["lut"], inter, lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][1], y,
                    d["g2"], d["k2"] // 64, d["k2"], cfg["ks2"], min(zb, cfg["nz2"]), d["hid"], 0, cfg["nw2"], cfg["mr2"], False)
    return y.view(m, TOP_K, d["hid"]), local_ids
# ============================================================================================ LARGEM END


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


# ============================================================================================ small-M entry (self-check, kernel)
SELFCHECK = True      # prepare(): compare the small-M path with the Marlin path on seeded synthetic inputs (never on the judge's)
SELFCHECK_TOL = 1e-2


def _selfcheck_inputs(m, seed, device):
    """Seeded inputs of the recipe of reference.generate_input (dist="normal", rank="busiest") rewritten here: shapes only."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, 4096), generator=g, device=device).to(torch.bfloat16)
    scores = torch.rand((m, N_EXPERTS), generator=g, device=device)
    z = torch.randn((N_EXPERTS,), generator=g, device=device)
    u = torch.rand((N_EXPERTS,), generator=g, device=device).clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=N_EXPERTS // 32)
    ids = (ids - 32 * pairs.argmax()) % N_EXPERTS
    return {"hidden_states": x, "topk_ids": ids.to(torch.int32), "topk_weights": (routed.values / routed.values.sum(-1, keepdim=True)).float(),
            "config": {"bs": m, "d_hidden": 4096, "d_expert": 4096, "n_experts": N_EXPERTS, "local_start": 0, "local_count": 32,
                       "top_k": TOP_K, "sharding": "ep8"}}


def _smallm_selfcheck(state):
    """Worst rel L2 of the small-M path against the Marlin path over M in (1, 8, 16), or inf if anything raises."""
    worst = 0.0
    try:
        for m in (1, 8, 16):
            data = _selfcheck_inputs(m, 1000 + m, state["norm_w"].device)
            ref = _kernel_sglang(data, state).float()
            got = _kernel_smallm(data, state).float()
            torch.cuda.synchronize()
            err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
            worst = max(worst, err if err == err else float("inf"))
    except Exception as ex:  # noqa: BLE001
        print(f"okc-diag smallm selfcheck raised {type(ex).__name__}: {str(ex)[:160]}")
        return float("inf")
    return worst


def _kernel_smallm(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    y, local_ids = smallm_experts(x, data["topk_ids"], state["smallm"], cfg["local_start"], cfg["local_count"], smallm_cfg(m))
    conf = config_for(m)  # v2b's epilogue tiling for this M
    return epilogue_triton(y, data["topk_weights"].to(torch.float32).contiguous(), local_ids, state["norm_w"],
                           block_h=conf["block_h"] or EPILOGUE_BLOCK_H, warps=conf["warps"] or EPILOGUE_WARPS)


def _use_smallm(data, state):
    cfg = data["config"]
    return (state.get("smallm") is not None and data["hidden_states"].shape[0] <= SM_MAX_M and cfg["local_start"] == 0
            and cfg["local_count"] == SM_NEXP)


# ============================================================================================ large-M entry (self-check, kernel)
LARGEM_SELFCHECK_MS = (32, 128, 256)


def _epilogue_for(y, topk_weights, local_ids, state, m):
    """v2b's post-expert epilogue for m tokens: SGLang's fused kernel from 32 tokens on (as sonnet_v2b), our Triton kernel below."""
    conf = config_for(m)
    if HAVE_SGLANG and conf["epilogue"] == "sg":
        return fused_post_expert_norm_weighted_sum(
            y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)
    return epilogue_triton(y, topk_weights, local_ids, state["norm_w"], block_h=conf["block_h"] or EPILOGUE_BLOCK_H, warps=conf["warps"] or EPILOGUE_WARPS)


def _kernel_largem(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    y, local_ids = largem_experts(x, data["topk_ids"], state["smallm"], state["largem"], cfg["local_start"], cfg["local_count"], largem_cfg(m))
    return _epilogue_for(y, data["topk_weights"].to(torch.float32).contiguous(), local_ids, state, m)


def _use_largem(data, state):
    cfg = data["config"]
    m = data["hidden_states"].shape[0]
    b = largem_bucket(m)
    return (state.get("largem") is not None and largem_cfg(m) is not None and state.get("largem_ok", {}).get(b, False)
            and cfg["local_start"] == 0 and cfg["local_count"] == SM_NEXP)


def _largem_selfcheck(state):
    """{bucket M: worst rel L2 of the large-M path against the Marlin path on seeded synthetic inputs} (inf if anything raises)."""
    res = {}
    for m in LARGEM_SELFCHECK_MS:
        try:
            data = _selfcheck_inputs(m, 2000 + m, state["norm_w"].device)
            ref = _kernel_sglang(data, state).float()
            got = _kernel_largem(data, state).float()
            torch.cuda.synchronize()
            err = float((got - ref).norm() / ref.norm().clamp(min=1e-12))
            res[m] = err if err == err else float("inf")
        except Exception as ex:  # noqa: BLE001
            print(f"okc-diag largem selfcheck m={m} raised {type(ex).__name__}: {str(ex)[:160]}")
            res[m] = float("inf")
    return res


def _prepare_largem_state(state, flags):
    """Builds the large-M scratch + extension on top of state["smallm"] and keeps the buckets whose self-check passes."""
    state["largem"], state["largem_ok"] = None, {}
    if state.get("smallm") is None:
        return
    try:
        state["largem"] = prepare_largem(state["smallm"], flags=flags)
    except Exception as ex:  # noqa: BLE001  (build error, out of memory: Marlin serves M > 16)
        print(f"okc-diag largem unavailable: {type(ex).__name__}: {str(ex)[:300]}")
        state["largem"] = None
        return
    errs = _largem_selfcheck(state) if SELFCHECK else {m: 0.0 for m in LARGEM_SELFCHECK_MS}
    print("okc-diag largem selfcheck rel L2 vs Marlin: " + " ".join(f"m{m}={e:.2e}" for m, e in errs.items()))
    state["largem_ok"] = {m: bool(e < SELFCHECK_TOL) for m, e in errs.items()}
    if not any(state["largem_ok"].values()):
        state["largem"] = None
    print(f"okc-diag largem active={state['largem_ok']}")


# ============================================================================================ MOE5 BEGIN
# One streaming W4A16 GEMM template for 1..256 tokens (v4's small-M and large-M kernels fused; see the header). The packed weights and the
# large-M scratch (chunk table, slot list, split-K partials, counters) are v4's; only the extension is new.
M5_STAMPS, M5_NOCOMPUTE, M5_NOSTAGE = 1, 2, 4   # DBG template bits (debug twins; this file instantiates DBG = 0 only)
M5_OCC = 1024                                  # u64 slots at the head of the dbg buffer: per-SM concurrency counters (u32 cur[256], max[256])

_MOE5_EXTRA_SRC = r"""
#define M5_TC 32
#define M5_CTW 4
#define M5_STAMPS 1
#define M5_NOCOMPUTE 2
#define M5_NOSTAGE 4
#define M5_OCC 1024

__device__ __forceinline__ u64 m5_timer() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ u32 m5_smid() { u32 s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s; }

template <int MR, int D>
__device__ __forceinline__ void m5_prefetch(uint4 (&w)[MR][D], uint2 (&s)[MR][D], const uint4* wp, const uint2* sp, int wstride, int sstride) {
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int d = 0; d < D; ++d) {
            ldg_stream4(w[r][d], wp + (u64)r * wstride + d * 32);
            ldg_stream2(s[r][d], sp + (u64)r * sstride + d * 8);
        }
}

// k-loop body for one prefetched batch (D k64 iterations x MR 16-row groups): v4's do_batch_l; NOCOMPUTE folds the loaded words instead.
template <int NT, int MR, int D, int DBG>
__device__ __forceinline__ void m5_batch(const uint4 (&w)[MR][D], const uint2 (&s)[MR][D], int kk0, u32 lane_addr, int kp_bytes,
                                         const u32* s_lut, float (&acc)[MR][NT][4], u32& sink) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
#pragma unroll
        for (int jp = 0; jp < 2; ++jp) {
            u32 bf[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                ldmatrix_x4(bf[nt], lane_addr + (u32)(nt * 8 * kp_bytes) + (u32)((kk0 + d * 4 + jp * 2) * 32));
            if (DBG & M5_NOCOMPUTE) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) sink ^= bf[nt][0] ^ bf[nt][1] ^ bf[nt][2] ^ bf[nt][3];
#pragma unroll
                for (int r = 0; r < MR; ++r) sink ^= w[r][d].x ^ w[r][d].y ^ w[r][d].z ^ w[r][d].w ^ s[r][d].x ^ s[r][d].y;
                continue;
            }
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int j = jp * 2 + jj;
#pragma unroll
                for (int r = 0; r < MR; ++r) {
                    const u32 wv[4] = {w[r][d].x, w[r][d].y, w[r][d].z, w[r][d].w};
                    u32 q[4];
                    unpack_word(wv[j], q);
                    const u32 sc0 = s_lut[(s[r][d].x >> (8 * j)) & 255u];
                    const u32 sc1 = s_lut[(s[r][d].y >> (8 * j)) & 255u];
                    const u32 a[4] = {mul_bf16x2(q[0], sc0), mul_bf16x2(q[1], sc1), mul_bf16x2(q[2], sc0), mul_bf16x2(q[3], sc1)};
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) mma16816(acc[r][nt], a, bf[nt][2 * jj], bf[nt][2 * jj + 1]);
                }
            }
        }
    }
}

// One (chunk, n-block, K slice) work item of one CTA: the k-loop (register double buffering), the split-K fixup (last CTA adds the slices in
// order), the epilogue (SwiGLU + BF16 for GEMM1, BF16 for GEMM2). Identical in order of operations to v4's chunk_gemm.
template <bool FUSE, int NW, int MR, int NT, int DBG>
__device__ __forceinline__ void m5_chunk(uint4 (&w0)[MR][4 / MR], uint2 (&s0)[MR][4 / MR], const uint4* wp, const uint2* sp, int wstride, int sstride,
                                         int nbatch, u32 lane_addr, int kp_bytes, const u32* s_lut, float4* part, int* cnt, u16* out, const int* sl,
                                         int mc, float fscale, int z, int nb, int nbn, int ks, int KS, int g0, int out_stride, int* s_flag,
                                         u64* st, bool stamp, u32& sink) {
    constexpr int D = 4 / MR;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    uint4 w1[MR][D];
    uint2 s1[MR][D];
    float acc[MR][NT][4];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) acc[r][nt][0] = acc[r][nt][1] = acc[r][nt][2] = acc[r][nt][3] = 0.f;
    for (int b = 0; b < nbatch; b += 2) {
        if (b + 1 < nbatch) m5_prefetch<MR, D>(w1, s1, wp + (b + 1) * D * 32, sp + (b + 1) * D * 8, wstride, sstride);
        m5_batch<NT, MR, D, DBG>(w0, s0, b * D * 4, lane_addr, kp_bytes, s_lut, acc, sink);
        if (b + 1 < nbatch) {
            if (b + 2 < nbatch) m5_prefetch<MR, D>(w0, s0, wp + (b + 2) * D * 32, sp + (b + 2) * D * 8, wstride, sstride);
            m5_batch<NT, MR, D, DBG>(w1, s1, (b + 1) * D * 4, lane_addr, kp_bytes, s_lut, acc, sink);
        }
    }
    if ((DBG & M5_STAMPS) && stamp && tid == 0) st[3] = m5_timer();
    if (KS > 1) {
        const u64 pbase = (((((u64)z * nbn + nb) * KS + ks) * NW + warp) * MR) * 128 + lane;
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                part[pbase + r * 128 + nt * 32] = make_float4(acc[r][nt][0], acc[r][nt][1], acc[r][nt][2], acc[r][nt][3]);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int prev = atomicAdd(&cnt[z * nbn + nb], 1);
            const int last = (prev == KS - 1);
            *s_flag = last;
            if (last) cnt[z * nbn + nb] = 0;
        }
        __syncthreads();
        if ((DBG & M5_STAMPS) && stamp && tid == 0) st[4] = m5_timer();
        if (!*s_flag) return;
        __threadfence();
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint4 v[8];
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) {
                    if (k2 < KS) {
                        const u64 pb = (((((u64)z * nbn + nb) * KS + k2) * NW + warp) * MR) * 128 + lane + r * 128 + nt * 32;
                        v[k2] = ldcg4(part + pb);
                    }
                }
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) {
                    if (k2 < KS) {
                        a0 += __uint_as_float(v[k2].x); a1 += __uint_as_float(v[k2].y); a2 += __uint_as_float(v[k2].z); a3 += __uint_as_float(v[k2].w);
                    }
                }
                acc[r][nt][0] = a0; acc[r][nt][1] = a1; acc[r][nt][2] = a2; acc[r][nt][3] = a3;
            }
        if ((DBG & M5_STAMPS) && stamp && tid == 0) st[5] = m5_timer();
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const int g = g0 + r;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tk = nt * 8 + 2 * t4 + c;
                if (tk < mc) {
                    u16* o = out + (u64)sl[tk] * out_stride;
                    if (FUSE) {
                        const float gt = acc[r][nt][c] * fscale, up = acc[r][nt][2 + c] * fscale;
                        o[g * 8 + gid] = f2bf((gt / (1.0f + expf(-gt))) * up);
                    } else {
                        o[g * 16 + gid] = f2bf(acc[r][nt][c] * fscale);
                        o[g * 16 + 8 + gid] = f2bf(acc[r][nt][2 + c] * fscale);
                    }
                }
            }
        }
    }
    if ((DBG & M5_STAMPS) && stamp && tid == 0) st[6] = m5_timer();
}

// grid (G / (NW * MR) / NB, KS, nz): CTA = NW warps x MR 16-row weight groups per warp x one K slice; work item z = chunk z of the chunk table
// (z loops when the grid's z is smaller than the number of chunks; inactive z exit at once); the chunk's tokens (<= 32) are staged ONCE in smem
// and used for NB consecutive n-blocks. LB = min resident CTAs per SM asked from ptxas (0: no hint). DBG: debug twins (see the header).
template <bool FUSE, int NW, int MR, int LB, int DBG>
__global__ void __launch_bounds__(NW * 32, LB)
w4_moe5_kernel(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ gsc, const u32* __restrict__ lut,
               const u16* __restrict__ act, const int* __restrict__ ctab, const int* __restrict__ slotl, float4* __restrict__ part,
               int* __restrict__ cnt, u16* __restrict__ out, u64* __restrict__ dbg, int G, int NIT, int K, int KS, int NB, int out_stride, int act_shift) {
    constexpr int D = 4 / MR;
    constexpr int NTM = MR == 1 ? 2 : 4;  // MR 1 is the small-M geometry (M <= 16: a chunk holds <= 16 tokens): no 32-row code path
    extern __shared__ __align__(16) u8 smem_raw[];
    __shared__ int s_flag;
    u32* s_lut = reinterpret_cast<u32*>(smem_raw);
    u16* s_act = reinterpret_cast<u16*>(smem_raw + 1024);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2;
    const int ks = blockIdx.y, nbn = gridDim.x * NB;
    if (FUSE) pdl_wait();  // GEMM1 reads the chunk table written by the route kernel; GEMM2 only waits (below) before it reads GEMM1's output
    if (FUSE) pdl_trigger();
    const int Kc = K / KS, nit = Kc >> 6, KP = Kc + 8, kp_bytes = KP * 2;
    const int n_chunks = ctab[0];
    const u32 lane_addr = smem_u32(s_act) + (u32)((((lane & 7) * KP) + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8) * 2);
    u64* st = nullptr;
    u32 sink = 0, sm = 0;
    int cta = 0;
    if (DBG) {
        cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        st = dbg + M5_OCC + (u64)cta * 8;
        if ((DBG & M5_STAMPS) && tid == 0) {
            sm = m5_smid();
            u32* occ = reinterpret_cast<u32*>(dbg);
            const u32 cur = atomicAdd(&occ[sm], 1u) + 1u;
            atomicMax(&occ[256 + sm], cur);
        }
    }
    bool first = true;
    for (int z = blockIdx.z; z < n_chunks; z += gridDim.z) {
        const int* ct = ctab + 4 + M5_CTW * z;
        const int e = ct[0], mc = ct[1];
        const int* sl = slotl + ct[2];
        if ((DBG & M5_STAMPS) && first && tid == 0) st[0] = m5_timer();
        int nb = blockIdx.x * NB, g0 = (nb * NW + warp) * MR;
        const uint4* wp = Wq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 32 + lane;
        const uint2* sp = Sq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 8 + gid;
        uint4 w0[MR][D];
        uint2 s0[MR][D];
        m5_prefetch<MR, D>(w0, s0, wp, sp, NIT * 32, NIT * 8);
        __syncthreads();  // the previous work item is done with s_act / s_flag
        for (int i = tid; i < 256; i += NW * 32) s_lut[i] = lut[i];
        if (!FUSE) pdl_wait();  // GEMM2: the intermediate (and the shared split-K scratch) is only touched after GEMM1 is complete
        if ((DBG & M5_STAMPS) && first && tid == 0) st[1] = m5_timer();
        // STAGE8: the chunk's rows (zero-padded to the 8/16/32 class) in padded smem, 8 independent 16-byte loads per thread, then the 8 stores
        const int nrows = mc > 16 ? 32 : (mc > 8 ? 16 : 8);
        const int lcpr = __ffs(Kc) - 4;  // log2 of the 16-byte chunks per activation row (Kc / 8)
        const int cpr = 1 << lcpr;
        const int total = nrows * cpr;
        for (int base = tid; base < total; base += NW * 32 * 8) {
            uint4 v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int idx = base + u * NW * 32;
                const int row = idx >> lcpr, c = idx & (cpr - 1);
                v[u] = make_uint4(0u, 0u, 0u, 0u);
                if (idx < total && row < mc && !(DBG & M5_NOSTAGE))
                    v[u] = reinterpret_cast<const uint4*>(act + (u64)(sl[row] >> act_shift) * K + (u64)ks * Kc)[c];
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int idx = base + u * NW * 32;
                if (idx < total) *reinterpret_cast<uint4*>(s_act + (idx >> lcpr) * KP + (idx & (cpr - 1)) * 8) = v[u];
            }
        }
        __syncthreads();
        if ((DBG & M5_STAMPS) && first && tid == 0) st[2] = m5_timer();
        const float fscale = gsc[e] * 1073741824.0f;  // global scale * 2**30 (undoes the operand scaling)
        for (int nbi = 0; nbi < NB; ++nbi) {
            if (nbi > 0) {
                nb = blockIdx.x * NB + nbi;
                g0 = (nb * NW + warp) * MR;
                wp = Wq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 32 + lane;
                sp = Sq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 8 + gid;
                m5_prefetch<MR, D>(w0, s0, wp, sp, NIT * 32, NIT * 8);
            }
            if (mc <= 8) m5_chunk<FUSE, NW, MR, 1, DBG>(w0, s0, wp, sp, NIT * 32, NIT * 8, nit / D, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
            else if (NTM == 2 || mc <= 16) m5_chunk<FUSE, NW, MR, 2, DBG>(w0, s0, wp, sp, NIT * 32, NIT * 8, nit / D, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
            else m5_chunk<FUSE, NW, MR, 4, DBG>(w0, s0, wp, sp, NIT * 32, NIT * 8, nit / D, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
            first = false;
        }
    }
    if (DBG & M5_STAMPS) {
        if (tid == 0) {
            st[7] = m5_timer();
            atomicSub(reinterpret_cast<u32*>(dbg) + sm, 1u);
        }
    }
    if (DBG & M5_NOCOMPUTE) {  // the fold is stored (one word per warp) so the compiler cannot delete the loads
        if (lane == 0) dbg[M5_OCC + (u64)(gridDim.x * gridDim.y * gridDim.z) * 8 + (u64)cta * NW + warp] = (u64)sink;
    }
}

// M <= 16 (n <= 128 slots): the chunk table + slot list of largem_route_kernel (experts ascending, one chunk per active expert, slots ascending)
// from one 128-thread CTA with ballot/match counts, as smallm_route_kernel.
__global__ void __launch_bounds__(128) moe5_route_small_kernel(const int* __restrict__ ids, int* __restrict__ local_ids, int* __restrict__ ctab,
                                                               int* __restrict__ slotl, int n, int lo, int nloc) {
    __shared__ int wc[4][32];
    __shared__ int s_off[32];
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
        int po = c;  // inclusive scan of the counts over experts -> slot-list offsets
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int a = __shfl_up_sync(FULLMASK, po, d);
            if (lane >= d) po += a;
        }
        s_off[lane] = po - c;
        if (c > 0) {
            int* t = ctab + 4 + M5_CTW * __popc(act & ((1u << lane) - 1u));
            t[0] = lane;
            t[1] = c;
            t[2] = po - c;
            t[3] = 0;
        }
        if (lane == 0) ctab[0] = __popc(act);
    }
    __syncthreads();
    if (e >= 0) {
        int base = 0;
        for (int w = 0; w < warp; ++w) base += wc[w][e];
        slotl[s_off[e] + base + rank_w] = tid;
    }
}
"""

# Instantiations (FUSE, NW, MR, LB, DBG) of this file: GEMM1 small M (nw 8), GEMM2 small M (nw 8 / nw 4 at M <= 4), GEMM1 / GEMM2 large M (nw 4 x mr 2).
MOE5_INSTANCES = ((True, 8, 1, 2, 0), (False, 8, 1, 2, 0), (False, 4, 1, 4, 0), (True, 4, 2, 3, 0), (False, 4, 2, 3, 0))


def moe5_instance_names(instances):
    return [f"w4_moe5_kernel<{'true' if f else 'false'},{w},{r},{lb},{dbg}>" for f, w, r, lb, dbg in instances]


def moe5_host_src(instances):
    """The load_inline host source for a list of instantiations: moe5_gemm (dispatch by (fuse, nw, mr, lb, dbg)), moe5_route, moe5_init."""
    cases = "\n    ".join(f"M5_CASE({'true' if f else 'false'}, {w}, {r}, {lb}, {dbg})" for f, w, r, lb, dbg in instances)
    inits = "\n    ".join(f"cudaFuncSetAttribute(w4_moe5_kernel<{'true' if f else 'false'}, {w}, {r}, {lb}, {dbg}>, "
                          "cudaFuncAttributeMaxDynamicSharedMemorySize, M5_SMEM_MAX);\n    "
                          f"cudaFuncSetAttribute(w4_moe5_kernel<{'true' if f else 'false'}, {w}, {r}, {lb}, {dbg}>, "
                          "cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);"   # LB 3 x 68 KB needs the 228 KB split
                          for f, w, r, lb, dbg in instances)
    return _MOE5_HOST_TEMPLATE.replace("@CASES@", cases).replace("@INITS@", inits)


_MOE5_HOST_TEMPLATE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

static const int M5_SMEM_MAX = 163840;

template <bool FUSE, int NW, int MR, int LB, int DBG>
static void m5_launch(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab,
                      torch::Tensor slotl, torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor dbg, int64_t G, int64_t NIT,
                      int64_t K, int64_t KS, int64_t nz, int64_t NB, int64_t rows, int64_t out_stride, int64_t act_shift, bool pdl) {
    const int Kc = (int)(K / KS);
    const size_t smem = 1024 + (size_t)rows * (Kc + 8) * 2;
    auto fn = w4_moe5_kernel<FUSE, NW, MR, LB, DBG>;
    const uint4* a_wq = reinterpret_cast<const uint4*>(wq.data_ptr());
    const uint2* a_sq = reinterpret_cast<const uint2*>(sq.data_ptr());
    const float* a_gs = gs.data_ptr<float>();
    const u32* a_lut = reinterpret_cast<const u32*>(lut.data_ptr());
    const u16* a_act = reinterpret_cast<const u16*>(act.data_ptr());
    const int* a_ct = ctab.data_ptr<int>();
    const int* a_sl = slotl.data_ptr<int>();
    float4* a_part = reinterpret_cast<float4*>(part.data_ptr());
    int* a_cnt = cnt.data_ptr<int>();
    u16* a_out = reinterpret_cast<u16*>(out.data_ptr());
    u64* a_dbg = reinterpret_cast<u64*>(dbg.data_ptr());
    int i_G = (int)G, i_NIT = (int)NIT, i_K = (int)K, i_KS = (int)KS, i_NB = (int)NB, i_os = (int)out_stride, i_as = (int)act_shift;
    void* args[] = {&a_wq, &a_sq, &a_gs, &a_lut, &a_act, &a_ct, &a_sl, &a_part, &a_cnt, &a_out, &a_dbg, &i_G, &i_NIT, &i_K, &i_KS, &i_NB, &i_os, &i_as};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(G / (NW * MR) / NB), (unsigned)KS, (unsigned)nz);
    cfg.blockDim = dim3(NW * 32, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (pdl && at::cuda::getCurrentDeviceProperties()->major >= 9) {
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    cudaLaunchKernelExC(&cfg, (const void*)fn, args);
}

void moe5_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl,
               torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz,
               int64_t NB, int64_t rows, int64_t out_stride, int64_t act_shift, int64_t nw, int64_t mr, int64_t lb, int64_t dbg_flags, bool fuse, bool pdl) {
    c10::cuda::CUDAGuard guard(act.device());
#define M5_CASE(F, W, R, L, X) if (fuse == F && nw == W && mr == R && lb == L && dbg_flags == X) { \
        m5_launch<F, W, R, L, X>(wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, dbg, G, NIT, K, KS, nz, NB, rows, out_stride, act_shift, pdl); return; }
    @CASES@
#undef M5_CASE
    TORCH_CHECK(false, "moe5_gemm: no instantiation for fuse/nw/mr/lb/dbg = ", fuse, "/", nw, "/", mr, "/", lb, "/", dbg_flags);
}

void moe5_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc, bool small) {
    c10::cuda::CUDAGuard guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (small)
        moe5_route_small_kernel<<<1, 128, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctab.data_ptr<int>(), slotl.data_ptr<int>(),
                                                       (int)ids.numel(), (int)lo, (int)nloc);
    else
        largem_route_kernel<<<1, 1024, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctab.data_ptr<int>(), slotl.data_ptr<int>(),
                                                    (int)ids.numel(), (int)lo, (int)nloc);
}

void moe5_init() {
    @INITS@
}
"""
_MOE5_CPP_SRC = (
    "void moe5_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl, "
    "torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz, "
    "int64_t NB, int64_t rows, int64_t out_stride, int64_t act_shift, int64_t nw, int64_t mr, int64_t lb, int64_t dbg_flags, bool fuse, bool pdl);\n"
    "void moe5_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc, bool small);\n"
    "void moe5_init();\n")


def moe5_device_src():
    """The device source of the v5 extension: v4's helpers (dequant, mma, loads, PDL) + v4's large-M route kernel + the MOE5 template."""
    return _SMALLM_DEVICE_SRC + _LARGEM_EXTRA_SRC + _MOE5_EXTRA_SRC


# ---- per-M launch configuration: (nw, mr, ks, nz, nb, lb) per GEMM = warps x 16-row groups per warp, K splits, grid-z cap, n-blocks per staged tile,
# min CTAs per SM. The K split per bucket is v4's (SMALLM_CFG / LARGEM_CFG: judge-tuned), so the split-K sums are v4's and the output is bitwise v4's;
# nz as v4's; LB 2 for 8-warp CTAs (regs <= 128), 3 for the 4-warp x 2-group large-M CTAs (regs <= 168), 4 for the 4-warp x 1-group one; NB 4 / 2 for
# GEMM1 / GEMM2 at M >= 17 (the staged tile is reused by 4 / 2 weight groups; GEMM2 has half the n-blocks). Alt A's best guess; measure_altA_v1.py sweeps it.
MOE5_CFG = (
    # (largest M, GEMM1 (nw, mr, ks, nz, nb, lb), GEMM2 (nw, mr, ks, nz, nb, lb))
    (1, (8, 1, 2, 2, 1, 2), (4, 1, 2, 6, 1, 4)),
    (2, (8, 1, 2, 6, 1, 2), (4, 1, 4, 12, 1, 4)),
    (4, (8, 1, 2, 2, 1, 2), (4, 1, 4, 6, 1, 4)),
    (8, (8, 1, 2, 8, 1, 2), (8, 1, 2, 4, 1, 2)),
    (12, (8, 1, 2, 8, 1, 2), (8, 1, 2, 8, 1, 2)),
    (16, (8, 1, 2, 8, 1, 2), (8, 1, 2, 16, 1, 2)),
    (32, (4, 2, 4, 16, 4, 3), (4, 2, 4, 16, 2, 3)),
    (128, (4, 2, 4, 32, 4, 3), (4, 2, 4, 32, 2, 3)),
    (256, (4, 2, 4, 48, 4, 3), (4, 2, 4, 48, 2, 3)),
)
MOE5_ENABLE = {mx: True for mx, _c1, _c2 in MOE5_CFG}   # per bucket (largest M); False = v4's path for that bucket
MOE5_SELFCHECK_MS = (1, 2, 4, 8, 12, 16, 32, 33, 128, 256)


def moe5_bucket(m):
    for mx, _c1, _c2 in MOE5_CFG:
        if m <= mx:
            return mx
    return None


def moe5_cfg(m):
    """(GEMM1 cfg, GEMM2 cfg) of the bucket of m, or None (m > 256, bucket disabled)."""
    b = moe5_bucket(m)
    if b is None or not MOE5_ENABLE.get(b, False):
        return None
    for mx, c1, c2 in MOE5_CFG:
        if mx == b:
            return c1, c2
    return None


def moe5_rows(m):
    """Shape-only activation row class of the staged tile (a chunk holds <= min(m, 32) tokens)."""
    return 8 if m <= 8 else (16 if m <= 16 else 32)


def moe5_zbound(m, lm):
    """Shape-only bound on the number of chunks: <= 8 m slots; sum_e ceil(m_e / 32) <= 8 m / 32 + 32; <= 32 experts x ceil(m / 32) chunks each."""
    return min(lm["zmax"], 8 * m, 8 * m // LM_TC + SM_NEXP, SM_NEXP * ((m + LM_TC - 1) // LM_TC))


_MOE5_EXT = {}


def moe5_ext(flags=(), instances=MOE5_INSTANCES):
    """The compiled v5 extension (moe5_gemm / moe5_route / moe5_init), built once per process, flag set and instantiation list."""
    key = (tuple(flags), tuple(instances))
    if key not in _MOE5_EXT:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        src = moe5_device_src() + moe5_host_src(instances)
        tag = hashlib.sha1((src + repr(key)).encode()).hexdigest()[:12]
        _MOE5_EXT[key] = load_inline(
            name=f"okc_moe5_{tag}", cpp_sources=_MOE5_CPP_SRC, cuda_sources=src,
            functions=["moe5_gemm", "moe5_route", "moe5_init"], extra_cuda_cflags=["-O3", *flags], verbose=False)
    return _MOE5_EXT[key]


def prepare_moe5(sm, lm, ext=None, flags=()):
    """Untimed. The v5 extension on top of v4's packed weights (sm) and large-M scratch (lm); dbg = a dummy buffer (DBG = 0 kernels never touch it)."""
    m5 = {"ext": ext if ext is not None else moe5_ext(flags), "dbg": torch.zeros(8, dtype=torch.int64, device=sm["w1"].device)}
    m5["ext"].moe5_init()
    return m5


def moe5_gemm_args(m, k_dim, groups, ks, nz, nb, lm):
    """Shape-only launch arguments; raises on a geometry the kernel cannot run (caught by prepare's self-check)."""
    assert k_dim % ks == 0 and (k_dim // ks) % 256 == 0 and k_dim // ks <= 2048 and ks <= 8, (k_dim, ks)
    assert groups % nb == 0, (groups, nb)
    return dict(ks=ks, nz=min(moe5_zbound(m, lm), nz), nb=nb, rows=moe5_rows(m))


def moe5_experts(x, topk_ids, sm, lm, m5, start, count, cfg, pdl=(True, True), dbg=None, dbg_flags=(0, 0)):
    """Route + GEMM1 (SwiGLU fused) + GEMM2 for 1..256 tokens with the unified kernel. Returns (y [M, 8, H] BF16, local_ids [M, 8] int32).
    cfg = (GEMM1 (nw, mr, ks, nz, nb, lb), GEMM2 (...)). Shape-only grids; no host sync. dbg (one buffer or one per GEMM) / dbg_flags / pdl are the
    measurement file's knobs."""
    m = x.shape[0]
    d, ext = sm["dims"], m5["ext"]
    assert m <= lm["max_m"] and count == SM_NEXP
    ids = topk_ids.to(torch.int32).contiguous()
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
    ext.moe5_route(ids, local_ids, lm["ctab"], lm["slotl"], start, count, m <= SM_MAX_M)
    inter = torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
    dbg = m5["dbg"] if dbg is None else dbg
    dbgs = tuple(dbg) if isinstance(dbg, (tuple, list)) else (dbg, dbg)
    for i, (k_dim, groups, c, src, dst, os_, ash, fuse) in enumerate((
            (d["k1"], d["g1"], cfg[0], x.contiguous(), inter, d["inter"], 3, True),
            (d["k2"], d["g2"], cfg[1], inter, y, d["hid"], 0, False))):
        nw, mr, ks, nz, nb, lb = c
        assert groups % (nw * mr) == 0 and (mr == 2 or m <= SM_MAX_M), (groups, nw, mr, m)  # mr 1 = small-M geometry (chunks <= 16 tokens)
        a = moe5_gemm_args(m, k_dim, groups // (nw * mr), ks, nz, nb, lm)
        w, s, gs = (sm["w1"], sm["s1"], sm["gs1"]) if fuse else (sm["w2"], sm["s2"], sm["gs2"])
        ext.moe5_gemm(w, s, gs, sm["lut"], src, lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][i], dst, dbgs[i],
                      groups, k_dim // 64, k_dim, a["ks"], a["nz"], a["nb"], a["rows"], os_, ash, nw, mr, lb, dbg_flags[i], fuse, bool(pdl[i]))
    return y.view(m, TOP_K, d["hid"]), local_ids


def _kernel_moe5(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    y, local_ids = moe5_experts(x, data["topk_ids"], state["smallm"], state["largem"], state["moe5"], cfg["local_start"], cfg["local_count"], moe5_cfg(m))
    return _epilogue_for(y, data["topk_weights"].to(torch.float32).contiguous(), local_ids, state, m)


def _use_moe5(data, state):
    cfg = data["config"]
    m = data["hidden_states"].shape[0]
    return (state.get("moe5") is not None and state.get("largem") is not None and moe5_cfg(m) is not None
            and state.get("moe5_ok", {}).get(moe5_bucket(m), False) and cfg["local_start"] == 0 and cfg["local_count"] == SM_NEXP)


def _v4_path(data, state):
    """v4's own path for these inputs (the bitwise reference and the fallback): small-M kernel, large-M kernel or Marlin."""
    if _use_smallm(data, state):
        return _kernel_smallm(data, state)
    if _use_largem(data, state):
        return _kernel_largem(data, state)
    return None


def _moe5_selfcheck(state):
    """{bucket: True if v5's output is bit-identical to v4's on seeded inputs of every M of the bucket, False otherwise (or on any exception)}."""
    res = {}
    for m in MOE5_SELFCHECK_MS:
        b = moe5_bucket(m)
        try:
            data = _selfcheck_inputs(m, 3000 + m, state["norm_w"].device)
            ref = _v4_path(data, state)
            if ref is None:
                res[b] = False
                print(f"okc-diag moe5 selfcheck m={m}: v4 path unavailable")
                continue
            got = _kernel_moe5(data, state)
            torch.cuda.synchronize()
            ok = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
            res[b] = res.get(b, True) and ok
        except Exception as ex:  # noqa: BLE001
            print(f"okc-diag moe5 selfcheck m={m} raised {type(ex).__name__}: {str(ex)[:160]}")
            res[b] = False
    return res


def _prepare_moe5_state(state, flags):
    """Builds the v5 extension on top of state["smallm"] / state["largem"] and keeps the buckets that are bitwise v4."""
    state["moe5"], state["moe5_ok"] = None, {}
    if state.get("smallm") is None or state.get("largem") is None:
        print("okc-diag moe5 unavailable: v4's small-M / large-M state missing")
        return
    try:
        state["moe5"] = prepare_moe5(state["smallm"], state["largem"], flags=flags)
    except Exception as ex:  # noqa: BLE001  (build error: v4's paths serve every M)
        print(f"okc-diag moe5 unavailable: {type(ex).__name__}: {str(ex)[:300]}")
        state["moe5"] = None
        return
    oks = _moe5_selfcheck(state) if SELFCHECK else {mx: True for mx, _a, _b in MOE5_CFG}
    print("okc-diag moe5 bitwise vs v4 per bucket: " + " ".join(f"m{b}={'OK' if ok else 'DIFF'}" for b, ok in oks.items()))
    state["moe5_ok"] = oks
    if not any(oks.values()):
        state["moe5"] = None
    print(f"okc-diag moe5 active={state['moe5_ok']}")
# ============================================================================================ MOE5 END


# ============================================================================================ MOE6 BEGIN
# v5's unified kernel with a DEEPER WEIGHT PIPELINE (card C21b, fable-09 D4 S1 decision (ii)): the same tile layout, dequant, mma order, split-K fixup
# order and epilogue as v4/v5, only how the weight bytes arrive changes:
#   D     k64 iterations per batch per 16-row group (v5: 4 / MR, i.e. 3 KB per warp per batch);
#   RB    register mode: RB batches in flight beyond the one being consumed, NBUF = RB + 1 register buffers (v5 = RB 1, D 4 / MR);
#         finer batches buy depth at constant registers: (4 warps x 2 groups) D 1 RB 3 = 4 buffers x 2 tiles = v5's 48 registers, 4.5 KB per warp in flight;
#   RS    ring mode (RS > 0, RB ignored): a per-warp ring of RS stages in shared memory filled with cp.async (16 B per lane for the words, 4 lanes x 16 B
#         for the 64 B of scales of a tile = 576 B per tile, exactly the DRAM bytes), cp.async.commit_group per stage, cp.async.wait_group RS-2 before
#         consuming, __syncwarp around the slot reuse; the dequant reads its word (ld.shared.v4) and scale pair (ld.shared.v2) from the stage;
#   CONTINUITY  the pipeline never drains at a work-item boundary: the first RB (RS-1) batches of the CTA's NEXT work item (next n-block of the same
#         staged tile, or the next chunk) are issued while the current item's last batches, fixup and epilogue run.
# sm_75 (dev card): cp.async falls back to a synchronous copy (same bytes, same order; bitwise-testable, timing meaningless).
M6_TILE = 576       # bytes of one 16-row x k64 tile in a ring stage: 512 B of words + 64 B of E4M3 scales

_MOE6_EXTRA_SRC = r"""
#define M6_TILE 576

__device__ __forceinline__ void cp_async16(u8* sptr, const void* gptr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_u32(sptr)), "l"(gptr) : "memory");
#else
    uint4 v;
    ldg_stream4(v, gptr);
    *reinterpret_cast<uint4*>(sptr) = v;
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");
#endif
}

// One weight stream = the MR 16-row groups of one warp over one K slice of one expert: wp = words (+ lane), sp0 = scales (tile-contiguous, no lane offset).
struct M6Stream {
    const uint4* wp;
    const uint2* sp0;
};

template <int MR, int D>
__device__ __forceinline__ void m6_issue_reg(uint4 (&w)[MR][D], uint2 (&s)[MR][D], const M6Stream& st, int q, int wstride, int sstride, int gid) {
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int d = 0; d < D; ++d) {
            ldg_stream4(w[r][d], st.wp + (u64)r * wstride + (q * D + d) * 32);
            ldg_stream2(s[r][d], st.sp0 + (u64)r * sstride + (q * D + d) * 8 + gid);
        }
}

template <int MR, int D>
__device__ __forceinline__ void m6_issue_ring(u8* slot, const M6Stream& st, int q, int wstride, int sstride, int lane) {
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int d = 0; d < D; ++d) {
            u8* t = slot + (r * D + d) * M6_TILE;
            cp_async16(t + lane * 16, st.wp + (u64)r * wstride + (q * D + d) * 32);
            if (lane < 4) cp_async16(t + 512 + lane * 16, reinterpret_cast<const u8*>(st.sp0 + (u64)r * sstride + (q * D + d) * 8) + lane * 16);
        }
}

template <int MR, int D>
__device__ __forceinline__ void m6_load_ring(uint4 (&w)[MR][D], uint2 (&s)[MR][D], const u8* slot, int lane, int gid) {
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const u8* t = slot + (r * D + d) * M6_TILE;
            w[r][d] = *reinterpret_cast<const uint4*>(t + lane * 16);
            s[r][d] = *reinterpret_cast<const uint2*>(t + 512 + gid * 8);
        }
}

// Split-K fixup + epilogue of one work item (v5's m5_chunk after its k-loop; identical order of operations).
template <bool FUSE, int NW, int MR, int NT, int DBG>
__device__ __forceinline__ void m6_finish(float (&acc)[MR][NT][4], float4* part, int* cnt, u16* out, const int* sl, int mc, float fscale, int z, int nb, int nbn,
                                          int ks, int KS, int g0, int out_stride, int* s_flag, u64* st, bool stamp) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    if (KS > 1) {
        const u64 pbase = (((((u64)z * nbn + nb) * KS + ks) * NW + warp) * MR) * 128 + lane;
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                part[pbase + r * 128 + nt * 32] = make_float4(acc[r][nt][0], acc[r][nt][1], acc[r][nt][2], acc[r][nt][3]);
        __threadfence();
        __syncthreads();
        if (tid == 0) {
            const int prev = atomicAdd(&cnt[z * nbn + nb], 1);
            const int last = (prev == KS - 1);
            *s_flag = last;
            if (last) cnt[z * nbn + nb] = 0;
        }
        __syncthreads();
        if ((DBG & M5_STAMPS) && stamp && tid == 0) st[4] = m5_timer();
        if (!*s_flag) return;
        __threadfence();
#pragma unroll
        for (int r = 0; r < MR; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint4 v[8];
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) {
                    if (k2 < KS) {
                        const u64 pb = (((((u64)z * nbn + nb) * KS + k2) * NW + warp) * MR) * 128 + lane + r * 128 + nt * 32;
                        v[k2] = ldcg4(part + pb);
                    }
                }
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
                for (int k2 = 0; k2 < 8; ++k2) {
                    if (k2 < KS) {
                        a0 += __uint_as_float(v[k2].x); a1 += __uint_as_float(v[k2].y); a2 += __uint_as_float(v[k2].z); a3 += __uint_as_float(v[k2].w);
                    }
                }
                acc[r][nt][0] = a0; acc[r][nt][1] = a1; acc[r][nt][2] = a2; acc[r][nt][3] = a3;
            }
        if ((DBG & M5_STAMPS) && stamp && tid == 0) st[5] = m5_timer();
    }
#pragma unroll
    for (int r = 0; r < MR; ++r) {
        const int g = g0 + r;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tk = nt * 8 + 2 * t4 + c;
                if (tk < mc) {
                    u16* o = out + (u64)sl[tk] * out_stride;
                    if (FUSE) {
                        const float gt = acc[r][nt][c] * fscale, up = acc[r][nt][2 + c] * fscale;
                        o[g * 8 + gid] = f2bf((gt / (1.0f + expf(-gt))) * up);
                    } else {
                        o[g * 16 + gid] = f2bf(acc[r][nt][c] * fscale);
                        o[g * 16 + 8 + gid] = f2bf(acc[r][nt][2 + c] * fscale);
                    }
                }
            }
        }
    }
    if ((DBG & M5_STAMPS) && stamp && tid == 0) st[6] = m5_timer();
}

// Register mode: the k-loop of one work item over NBUF = RB + 1 register buffers. Precondition: batches 0..RB of this item are in flight in w[0..RB]
// (issued by the prologue or by the previous item's loop). After consuming batch b (buffer b % NBUF) batch b + RB + 1 is issued into that buffer: of this
// item, or of the next item (nxt) when past the end -> the stream never drains. nbatch % NBUF == 0 (host rule) keeps the buffer indices static.
template <bool FUSE, int NW, int MR, int D, int RB, int NT, int DBG>
__device__ __forceinline__ void m6_item_reg(uint4 (&w)[RB + 1][MR][D], uint2 (&s)[RB + 1][MR][D], const M6Stream& cur, const M6Stream& nxt, bool has_nxt,
                                            int wstride, int sstride, int nbatch, u32 lane_addr, int kp_bytes, const u32* s_lut,
                                            float4* part, int* cnt, u16* out, const int* sl, int mc, float fscale, int z, int nb, int nbn, int ks, int KS, int g0,
                                            int out_stride, int* s_flag, u64* st, bool stamp, u32& sink) {
    constexpr int NBUF = RB + 1;
    const int gid = (threadIdx.x & 31) >> 2;
    float acc[MR][NT][4];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) acc[r][nt][0] = acc[r][nt][1] = acc[r][nt][2] = acc[r][nt][3] = 0.f;
    for (int b = 0; b < nbatch; b += NBUF) {
#pragma unroll
        for (int u = 0; u < NBUF; ++u) {
            m5_batch<NT, MR, D, DBG>(w[u], s[u], (b + u) * D * 4, lane_addr, kp_bytes, s_lut, acc, sink);
            const int q = b + u + RB + 1;
            if (q < nbatch) m6_issue_reg<MR, D>(w[u], s[u], cur, q, wstride, sstride, gid);
            else if (has_nxt) m6_issue_reg<MR, D>(w[u], s[u], nxt, q - nbatch, wstride, sstride, gid);
        }
    }
    if ((DBG & M5_STAMPS) && stamp && threadIdx.x == 0) st[3] = m5_timer();
    m6_finish<FUSE, NW, MR, NT, DBG>(acc, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, s_flag, st, stamp);
}

// Ring mode: the k-loop of one work item over this warp's RS-stage ring. Precondition: batches 0..RS-2 of this item are in flight (RS - 1 committed
// groups; `head` = the slot of batch 0). Per batch: wait for its group, load it into registers, issue batch k + RS - 1 (this item or the next) into the slot
// consumed one step earlier, commit (an empty group past the last item keeps the count uniform), then dequant + mma from the registers.
template <bool FUSE, int NW, int MR, int D, int RS, int NT, int DBG>
__device__ __forceinline__ void m6_item_ring(u8* ring, int& head, const M6Stream& cur, const M6Stream& nxt, bool has_nxt, int wstride, int sstride, int nbatch,
                                             u32 lane_addr, int kp_bytes, const u32* s_lut, float4* part, int* cnt, u16* out, const int* sl, int mc, float fscale,
                                             int z, int nb, int nbn, int ks, int KS, int g0, int out_stride, int* s_flag, u64* st, bool stamp, u32& sink) {
    constexpr int SB = MR * D * M6_TILE;
    const int lane = threadIdx.x & 31, gid = lane >> 2;
    float acc[MR][NT][4];
#pragma unroll
    for (int r = 0; r < MR; ++r)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) acc[r][nt][0] = acc[r][nt][1] = acc[r][nt][2] = acc[r][nt][3] = 0.f;
    uint4 w[MR][D];
    uint2 s[MR][D];
    for (int k = 0; k < nbatch; ++k) {
        cp_async_wait<RS - 2>();
        __syncwarp();
        m6_load_ring<MR, D>(w, s, ring + head * SB, lane, gid);
        const int q = k + RS - 1;
        u8* slot = ring + ((head + RS - 1) % RS) * SB;
        if (q < nbatch) m6_issue_ring<MR, D>(slot, cur, q, wstride, sstride, lane);
        else if (has_nxt) m6_issue_ring<MR, D>(slot, nxt, q - nbatch, wstride, sstride, lane);
        cp_async_commit();
        head = (head + 1 == RS) ? 0 : head + 1;
        m5_batch<NT, MR, D, DBG>(w, s, k * D * 4, lane_addr, kp_bytes, s_lut, acc, sink);
    }
    if ((DBG & M5_STAMPS) && stamp && threadIdx.x == 0) st[3] = m5_timer();
    m6_finish<FUSE, NW, MR, NT, DBG>(acc, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, s_flag, st, stamp);
}

// grid (G / (NW * MR) / NB, KS, nz) as w4_moe5_kernel; smem = 1 KB lut | ring NW x RS x MR x D x 576 B (RS > 0) | staged activations rows x (Kc + 8) x 2.
template <bool FUSE, int NW, int MR, int D, int RB, int RS, int LB, int DBG>
__global__ void __launch_bounds__(NW * 32, LB)
w4_moe6_kernel(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const float* __restrict__ gsc, const u32* __restrict__ lut,
               const u16* __restrict__ act, const int* __restrict__ ctab, const int* __restrict__ slotl, float4* __restrict__ part,
               int* __restrict__ cnt, u16* __restrict__ out, u64* __restrict__ dbg, int G, int NIT, int K, int KS, int NB, int out_stride, int act_shift) {
    constexpr int NTM = MR == 1 ? 2 : 4;
    constexpr int NBUF = RB + 1;
    constexpr int SB = MR * D * M6_TILE;
    constexpr int RING_BYTES = RS > 0 ? NW * RS * SB : 0;
    extern __shared__ __align__(16) u8 smem_raw[];
    __shared__ int s_flag;
    u32* s_lut = reinterpret_cast<u32*>(smem_raw);
    u8* ring = smem_raw + 1024 + (threadIdx.x >> 5) * (RS > 0 ? RS * SB : 0);
    u16* s_act = reinterpret_cast<u16*>(smem_raw + 1024 + RING_BYTES);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2;
    const int ks = blockIdx.y, nbn = gridDim.x * NB;
    if (FUSE) pdl_wait();
    if (FUSE) pdl_trigger();
    const int Kc = K / KS, nit = Kc >> 6, KP = Kc + 8, kp_bytes = KP * 2;
    const int nbatch = nit / D, wstride = NIT * 32, sstride = NIT * 8;
    const int n_chunks = ctab[0];
    const u32 lane_addr = smem_u32(s_act) + (u32)((((lane & 7) * KP) + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8) * 2);
    u64* st = nullptr;
    u32 sink = 0, sm = 0;
    int cta = 0;
    if (DBG) {
        cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        st = dbg + M5_OCC + (u64)cta * 8;
        if ((DBG & M5_STAMPS) && tid == 0) {
            sm = m5_smid();
            u32* occ = reinterpret_cast<u32*>(dbg);
            const u32 cur = atomicAdd(&occ[sm], 1u) + 1u;
            atomicMax(&occ[256 + sm], cur);
        }
    }
    uint4 w[NBUF][MR][D];
    uint2 s[NBUF][MR][D];
    int head = 0;
    bool first = true, primed = false;
    for (int z = blockIdx.z; z < n_chunks; z += gridDim.z) {
        const int* ct = ctab + 4 + M5_CTW * z;
        const int e = ct[0], mc = ct[1];
        const int* sl = slotl + ct[2];
        const int zn = z + (int)gridDim.z;                      // the next chunk of this CTA (its first n-block is the item after this chunk's last)
        const bool has_zn = zn < n_chunks;
        const int en = has_zn ? ctab[4 + M5_CTW * zn] : 0;
        if ((DBG & M5_STAMPS) && first && tid == 0) st[0] = m5_timer();
        M6Stream cur, nxt;
        int nb = blockIdx.x * NB, g0 = (nb * NW + warp) * MR;
        cur.wp = Wq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 32 + lane;
        cur.sp0 = Sq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 8;
        if (!primed) {  // the CTA's first work item: fill the pipeline (RB + 1 register batches | RS - 1 ring stages)
            primed = true;
            if (RS > 0) {
#pragma unroll
                for (int q = 0; q < RS - 1; ++q) {
                    m6_issue_ring<MR, D>(ring + q * SB, cur, q, wstride, sstride, lane);
                    cp_async_commit();
                }
            } else {
#pragma unroll
                for (int q = 0; q < NBUF; ++q) m6_issue_reg<MR, D>(w[q], s[q], cur, q, wstride, sstride, gid);
            }
        }
        __syncthreads();  // the previous work item is done with s_act / s_flag
        for (int i = tid; i < 256; i += NW * 32) s_lut[i] = lut[i];
        if (!FUSE) pdl_wait();
        if ((DBG & M5_STAMPS) && first && tid == 0) st[1] = m5_timer();
        const int nrows = mc > 16 ? 32 : (mc > 8 ? 16 : 8);
        const int lcpr = __ffs(Kc) - 4;
        const int cpr = 1 << lcpr;
        const int total = nrows * cpr;
        for (int base = tid; base < total; base += NW * 32 * 8) {
            uint4 v[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int idx = base + u * NW * 32;
                const int row = idx >> lcpr, c = idx & (cpr - 1);
                v[u] = make_uint4(0u, 0u, 0u, 0u);
                if (idx < total && row < mc && !(DBG & M5_NOSTAGE))
                    v[u] = reinterpret_cast<const uint4*>(act + (u64)(sl[row] >> act_shift) * K + (u64)ks * Kc)[c];
            }
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int idx = base + u * NW * 32;
                if (idx < total) *reinterpret_cast<uint4*>(s_act + (idx >> lcpr) * KP + (idx & (cpr - 1)) * 8) = v[u];
            }
        }
        __syncthreads();
        if ((DBG & M5_STAMPS) && first && tid == 0) st[2] = m5_timer();
        const float fscale = gsc[e] * 1073741824.0f;
        for (int nbi = 0; nbi < NB; ++nbi) {
            nb = blockIdx.x * NB + nbi;
            g0 = (nb * NW + warp) * MR;
            cur.wp = Wq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 32 + lane;
            cur.sp0 = Sq + (((u64)e * G + g0) * NIT + (u64)ks * nit) * 8;
            // the item after this one: the next n-block of this chunk, else the first n-block of the next chunk
            bool has_nxt = true;
            int en_ = e, gn = ((nb + 1) * NW + warp) * MR;
            if (nbi + 1 == NB) {
                has_nxt = has_zn;
                en_ = en;
                gn = (blockIdx.x * NB * NW + warp) * MR;
            }
            nxt.wp = Wq + (((u64)en_ * G + gn) * NIT + (u64)ks * nit) * 32 + lane;
            nxt.sp0 = Sq + (((u64)en_ * G + gn) * NIT + (u64)ks * nit) * 8;
            if (RS > 0) {
                if (mc <= 8) m6_item_ring<FUSE, NW, MR, D, RS, 1, DBG>(ring, head, cur, nxt, has_nxt, wstride, sstride, nbatch, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
                else if (NTM == 2 || mc <= 16) m6_item_ring<FUSE, NW, MR, D, RS, 2, DBG>(ring, head, cur, nxt, has_nxt, wstride, sstride, nbatch, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
                else m6_item_ring<FUSE, NW, MR, D, RS, 4, DBG>(ring, head, cur, nxt, has_nxt, wstride, sstride, nbatch, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
            } else {
                if (mc <= 8) m6_item_reg<FUSE, NW, MR, D, RB, 1, DBG>(w, s, cur, nxt, has_nxt, wstride, sstride, nbatch, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
                else if (NTM == 2 || mc <= 16) m6_item_reg<FUSE, NW, MR, D, RB, 2, DBG>(w, s, cur, nxt, has_nxt, wstride, sstride, nbatch, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
                else m6_item_reg<FUSE, NW, MR, D, RB, 4, DBG>(w, s, cur, nxt, has_nxt, wstride, sstride, nbatch, lane_addr, kp_bytes, s_lut, part, cnt, out, sl, mc, fscale, z, nb, nbn, ks, KS, g0, out_stride, &s_flag, st, first, sink);
            }
            first = false;
        }
    }
    if (RS > 0) cp_async_wait<0>();
    if (DBG & M5_STAMPS) {
        if (tid == 0) {
            st[7] = m5_timer();
            atomicSub(reinterpret_cast<u32*>(dbg) + sm, 1u);
        }
    }
    if (DBG & M5_NOCOMPUTE) {
        if (lane == 0) dbg[M5_OCC + (u64)(gridDim.x * gridDim.y * gridDim.z) * 8 + (u64)cta * NW + warp] = (u64)sink;
    }
}
"""

# Instantiations (FUSE, NW, MR, D, RB, RS, LB, DBG) of this file: large M = 4 warps x 2 groups, the ring D 1 RS 8 (8 KB per warp in flight, 152 registers,
# 2 CTAs/SM by smem: 104 KB) and the register alternative D 1 RB 3 at LB 2 (4.5 KB per warp, 186 registers, no spills; at LB 3 the 168-register cap spills
# inside the k-loop); small M = 4 warps x 1 group, D 1 RB 7 (5.25 KB per warp, 153 registers, LB 3).
MOE6_INSTANCES = ((True, 4, 2, 1, 0, 8, 2, 0), (False, 4, 2, 1, 0, 8, 2, 0), (True, 4, 2, 1, 3, 0, 2, 0), (False, 4, 2, 1, 3, 0, 2, 0),
                  (True, 4, 1, 1, 7, 0, 3, 0), (False, 4, 1, 1, 7, 0, 3, 0))


def moe6_instance_names(instances):
    return [f"w4_moe6_kernel<{'true' if f else 'false'},{w},{r},{d},{rb},{rs},{lb},{dbg}>" for f, w, r, d, rb, rs, lb, dbg in instances]


def moe6_ring_bytes(nw, mr, d, rs):
    return nw * rs * mr * d * M6_TILE if rs else 0


def moe6_host_src(instances):
    """The load_inline host source for a list of instantiations: moe6_gemm (dispatch by (fuse, nw, mr, d, rb, rs, lb, dbg)), moe6_route, moe6_init."""
    def targs(f, w, r, d, rb, rs, lb, dbg):
        return f"{'true' if f else 'false'}, {w}, {r}, {d}, {rb}, {rs}, {lb}, {dbg}"
    cases = "\n    ".join(f"M6_CASE({targs(*i)})" for i in instances)
    inits = "\n    ".join(f"cudaFuncSetAttribute(w4_moe6_kernel<{targs(*i)}>, cudaFuncAttributeMaxDynamicSharedMemorySize, M6_SMEM_MAX);\n    "
                          f"cudaFuncSetAttribute(w4_moe6_kernel<{targs(*i)}>, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared);"
                          for i in instances)
    return _MOE6_HOST_TEMPLATE.replace("@CASES@", cases).replace("@INITS@", inits)


_MOE6_HOST_TEMPLATE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

static const int M6_SMEM_MAX = 163840;

template <bool FUSE, int NW, int MR, int D, int RB, int RS, int LB, int DBG>
static void m6_launch(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab,
                      torch::Tensor slotl, torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor dbg, int64_t G, int64_t NIT,
                      int64_t K, int64_t KS, int64_t nz, int64_t NB, int64_t rows, int64_t out_stride, int64_t act_shift, bool pdl) {
    const int Kc = (int)(K / KS);
    const size_t smem = 1024 + (size_t)(RS > 0 ? NW * RS * MR * D * M6_TILE : 0) + (size_t)rows * (Kc + 8) * 2;
    auto fn = w4_moe6_kernel<FUSE, NW, MR, D, RB, RS, LB, DBG>;
    const uint4* a_wq = reinterpret_cast<const uint4*>(wq.data_ptr());
    const uint2* a_sq = reinterpret_cast<const uint2*>(sq.data_ptr());
    const float* a_gs = gs.data_ptr<float>();
    const u32* a_lut = reinterpret_cast<const u32*>(lut.data_ptr());
    const u16* a_act = reinterpret_cast<const u16*>(act.data_ptr());
    const int* a_ct = ctab.data_ptr<int>();
    const int* a_sl = slotl.data_ptr<int>();
    float4* a_part = reinterpret_cast<float4*>(part.data_ptr());
    int* a_cnt = cnt.data_ptr<int>();
    u16* a_out = reinterpret_cast<u16*>(out.data_ptr());
    u64* a_dbg = reinterpret_cast<u64*>(dbg.data_ptr());
    int i_G = (int)G, i_NIT = (int)NIT, i_K = (int)K, i_KS = (int)KS, i_NB = (int)NB, i_os = (int)out_stride, i_as = (int)act_shift;
    void* args[] = {&a_wq, &a_sq, &a_gs, &a_lut, &a_act, &a_ct, &a_sl, &a_part, &a_cnt, &a_out, &a_dbg, &i_G, &i_NIT, &i_K, &i_KS, &i_NB, &i_os, &i_as};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(G / (NW * MR) / NB), (unsigned)KS, (unsigned)nz);
    cfg.blockDim = dim3(NW * 32, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (pdl && at::cuda::getCurrentDeviceProperties()->major >= 9) {
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    cudaLaunchKernelExC(&cfg, (const void*)fn, args);
}

void moe6_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl,
               torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz,
               int64_t NB, int64_t rows, int64_t out_stride, int64_t act_shift, int64_t nw, int64_t mr, int64_t d, int64_t rb, int64_t rs, int64_t lb,
               int64_t dbg_flags, bool fuse, bool pdl) {
    c10::cuda::CUDAGuard guard(act.device());
#define M6_CASE(F, W, R, DD, B, S, L, X) if (fuse == F && nw == W && mr == R && d == DD && rb == B && rs == S && lb == L && dbg_flags == X) { \
        m6_launch<F, W, R, DD, B, S, L, X>(wq, sq, gs, lut, act, ctab, slotl, part, cnt, out, dbg, G, NIT, K, KS, nz, NB, rows, out_stride, act_shift, pdl); return; }
    @CASES@
#undef M6_CASE
    TORCH_CHECK(false, "moe6_gemm: no instantiation for fuse/nw/mr/d/rb/rs/lb/dbg = ", fuse, "/", nw, "/", mr, "/", d, "/", rb, "/", rs, "/", lb, "/", dbg_flags);
}

void moe6_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc, bool small) {
    c10::cuda::CUDAGuard guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (small)
        moe5_route_small_kernel<<<1, 128, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctab.data_ptr<int>(), slotl.data_ptr<int>(),
                                                       (int)ids.numel(), (int)lo, (int)nloc);
    else
        largem_route_kernel<<<1, 1024, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctab.data_ptr<int>(), slotl.data_ptr<int>(),
                                                    (int)ids.numel(), (int)lo, (int)nloc);
}

void moe6_init() {
    @INITS@
}
"""
_MOE6_CPP_SRC = (
    "void moe6_gemm(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl, "
    "torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz, "
    "int64_t NB, int64_t rows, int64_t out_stride, int64_t act_shift, int64_t nw, int64_t mr, int64_t d, int64_t rb, int64_t rs, int64_t lb, "
    "int64_t dbg_flags, bool fuse, bool pdl);\n"
    "void moe6_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, int64_t lo, int64_t nloc, bool small);\n"
    "void moe6_init();\n")


def moe6_device_src():
    """The device source of the v6 extension: v5's device source (helpers, route kernels, the v5 template's batch body) + the MOE6 template."""
    return moe5_device_src() + _MOE6_EXTRA_SRC


# ---- per-M launch configuration: (nw, mr, d, rb, rs, ks, nz, nb, lb) per GEMM. K splits and z caps are v4's (bitwise contract); NB 2 at M >= 17 (C20: nb2 +8-13 %,
# nb4 no better); large M: the ring D 1 RS 8 at LB 2 (2 CTAs x 4 warps x 8 KB in flight per SM = 64 KB; v5: 3 x 4 x 3 KB = 36 KB) with continuity across items;
# small M: 4 warps x 1 group, D 1 RB 7, LB 3 (12 warps x 5.25 KB per SM; v5 had 2 CTAs x 8 warps x 3 KB at LB 2).
MOE6_CFG = (
    # (largest M, GEMM1 (nw, mr, d, rb, rs, ks, nz, nb, lb), GEMM2 (nw, mr, d, rb, rs, ks, nz, nb, lb))
    (1, (4, 1, 1, 7, 0, 2, 2, 1, 3), (4, 1, 1, 7, 0, 2, 6, 1, 3)),
    (2, (4, 1, 1, 7, 0, 2, 6, 1, 3), (4, 1, 1, 7, 0, 4, 12, 1, 3)),
    (4, (4, 1, 1, 7, 0, 2, 2, 1, 3), (4, 1, 1, 7, 0, 4, 6, 1, 3)),
    (8, (4, 1, 1, 7, 0, 2, 8, 1, 3), (4, 1, 1, 7, 0, 2, 4, 1, 3)),
    (12, (4, 1, 1, 7, 0, 2, 8, 1, 3), (4, 1, 1, 7, 0, 2, 8, 1, 3)),
    (16, (4, 1, 1, 7, 0, 2, 8, 1, 3), (4, 1, 1, 7, 0, 2, 16, 1, 3)),
    (32, (4, 2, 1, 0, 8, 4, 16, 2, 2), (4, 2, 1, 0, 8, 4, 16, 2, 2)),
    (128, (4, 2, 1, 0, 8, 4, 32, 2, 2), (4, 2, 1, 0, 8, 4, 32, 2, 2)),
    (256, (4, 2, 1, 0, 8, 4, 48, 2, 2), (4, 2, 1, 0, 8, 4, 48, 2, 2)),
)
MOE6_ENABLE = {mx: True for mx, _c1, _c2 in MOE6_CFG}
MOE6_SELFCHECK_MS = (1, 2, 4, 8, 12, 16, 32, 33, 128, 256)


def moe6_bucket(m):
    for mx, _c1, _c2 in MOE6_CFG:
        if m <= mx:
            return mx
    return None


def moe6_cfg(m):
    b = moe6_bucket(m)
    if b is None or not MOE6_ENABLE.get(b, False):
        return None
    for mx, c1, c2 in MOE6_CFG:
        if mx == b:
            return c1, c2
    return None


def moe6_smem(m, k_dim, c):
    nw, mr, d, rb, rs, ks, _nz, _nb, _lb = c
    return 1024 + moe6_ring_bytes(nw, mr, d, rs) + moe5_rows(m) * (k_dim // ks + 8) * 2


def moe6_check_cfg(m, k_dim, groups, c):
    """Geometry rules of the kernel (raise = caught by prepare's self-check): nbatch % (RB + 1) == 0 and >= 1 (register mode), RS >= 2 (ring), D | nit."""
    nw, mr, d, rb, rs, ks, nz, nb, lb = c
    assert groups % (nw * mr) == 0 and (groups // (nw * mr)) % nb == 0, (groups, nw, mr, nb)
    assert mr == 2 or m <= SM_MAX_M, (mr, m)
    assert k_dim % ks == 0 and (k_dim // ks) % 256 == 0 and k_dim // ks <= 2048 and ks <= 8, (k_dim, ks)
    nit = k_dim // ks // 64
    assert nit % d == 0, (nit, d)
    nbatch = nit // d
    if rs:
        assert rs >= 2 and rs - 1 <= nbatch, (rs, nbatch)
    else:
        assert nbatch % (rb + 1) == 0, (nbatch, rb)
    assert moe6_smem(m, k_dim, c) <= LM_SMEM_MAX, (m, c)


_MOE6_EXT = {}


def moe6_ext(flags=(), instances=MOE6_INSTANCES):
    key = (tuple(flags), tuple(instances))
    if key not in _MOE6_EXT:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        src = moe6_device_src() + moe6_host_src(instances)
        tag = hashlib.sha1((src + repr(key)).encode()).hexdigest()[:12]
        _MOE6_EXT[key] = load_inline(
            name=f"okc_moe6_{tag}", cpp_sources=_MOE6_CPP_SRC, cuda_sources=src,
            functions=["moe6_gemm", "moe6_route", "moe6_init"], extra_cuda_cflags=["-O3", *flags], verbose=False)
    return _MOE6_EXT[key]


def prepare_moe6(sm, lm, ext=None, flags=()):
    """Untimed. The v6 extension on top of v4's packed weights (sm) and large-M scratch (lm)."""
    m6 = {"ext": ext if ext is not None else moe6_ext(flags), "dbg": torch.zeros(8, dtype=torch.int64, device=sm["w1"].device)}
    m6["ext"].moe6_init()
    return m6


def moe6_experts(x, topk_ids, sm, lm, m6, start, count, cfg, pdl=(True, True), dbg=None, dbg_flags=(0, 0)):
    """Route + GEMM1 (SwiGLU fused) + GEMM2 for 1..256 tokens with the v6 kernel. cfg = (GEMM1 (nw, mr, d, rb, rs, ks, nz, nb, lb), GEMM2 (...))."""
    m = x.shape[0]
    d, ext = sm["dims"], m6["ext"]
    assert m <= lm["max_m"] and count == SM_NEXP
    ids = topk_ids.to(torch.int32).contiguous()
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
    ext.moe6_route(ids, local_ids, lm["ctab"], lm["slotl"], start, count, m <= SM_MAX_M)
    inter = torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
    dbg = m6["dbg"] if dbg is None else dbg
    dbgs = tuple(dbg) if isinstance(dbg, (tuple, list)) else (dbg, dbg)
    for i, (k_dim, groups, c, src, dst, os_, ash, fuse) in enumerate((
            (d["k1"], d["g1"], cfg[0], x.contiguous(), inter, d["inter"], 3, True),
            (d["k2"], d["g2"], cfg[1], inter, y, d["hid"], 0, False))):
        nw, mr, dd, rb, rs, ks, nz, nb, lb = c
        moe6_check_cfg(m, k_dim, groups, c)
        w, s, gs = (sm["w1"], sm["s1"], sm["gs1"]) if fuse else (sm["w2"], sm["s2"], sm["gs2"])
        ext.moe6_gemm(w, s, gs, sm["lut"], src, lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][i], dst, dbgs[i],
                      groups, k_dim // 64, k_dim, ks, min(moe5_zbound(m, lm), nz), nb, moe5_rows(m), os_, ash, nw, mr, dd, rb, rs, lb, dbg_flags[i], fuse, bool(pdl[i]))
    return y.view(m, TOP_K, d["hid"]), local_ids


def _kernel_moe6(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    y, local_ids = moe6_experts(x, data["topk_ids"], state["smallm"], state["largem"], state["moe6"], cfg["local_start"], cfg["local_count"], moe6_cfg(m))
    return _epilogue_for(y, data["topk_weights"].to(torch.float32).contiguous(), local_ids, state, m)


def _use_moe6(data, state):
    cfg = data["config"]
    m = data["hidden_states"].shape[0]
    return (state.get("moe6") is not None and state.get("largem") is not None and moe6_cfg(m) is not None
            and state.get("moe6_ok", {}).get(moe6_bucket(m), False) and cfg["local_start"] == 0 and cfg["local_count"] == SM_NEXP)


def _v5_path(data, state):
    """v5's path for these inputs (the bitwise reference and the fallback = the board kernel), else v4's."""
    if _use_moe5(data, state):
        return _kernel_moe5(data, state)
    return _v4_path(data, state)


def _moe6_selfcheck(state):
    """{bucket: True if v6's output is bit-identical to v5's (v4's) on seeded inputs of every M of the bucket}."""
    res = {}
    for m in MOE6_SELFCHECK_MS:
        b = moe6_bucket(m)
        try:
            data = _selfcheck_inputs(m, 4000 + m, state["norm_w"].device)
            ref = _v5_path(data, state)
            if ref is None:
                res[b] = False
                print(f"okc-diag moe6 selfcheck m={m}: v5/v4 path unavailable")
                continue
            got = _kernel_moe6(data, state)
            torch.cuda.synchronize()
            ok = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
            res[b] = res.get(b, True) and ok
        except Exception as ex:  # noqa: BLE001
            print(f"okc-diag moe6 selfcheck m={m} raised {type(ex).__name__}: {str(ex)[:160]}")
            res[b] = False
    return res


def _prepare_moe6_state(state, flags):
    state["moe6"], state["moe6_ok"] = None, {}
    if state.get("smallm") is None or state.get("largem") is None:
        print("okc-diag moe6 unavailable: v4's small-M / large-M state missing")
        return
    try:
        state["moe6"] = prepare_moe6(state["smallm"], state["largem"], flags=flags)
    except Exception as ex:  # noqa: BLE001
        print(f"okc-diag moe6 unavailable: {type(ex).__name__}: {str(ex)[:300]}")
        state["moe6"] = None
        return
    oks = _moe6_selfcheck(state) if SELFCHECK else {mx: True for mx, _a, _b in MOE6_CFG}
    print("okc-diag moe6 bitwise vs v5/v4 per bucket: " + " ".join(f"m{b}={'OK' if ok else 'DIFF'}" for b, ok in oks.items()))
    state["moe6_ok"] = oks
    if not any(oks.values()):
        state["moe6"] = None
    print(f"okc-diag moe6 active={state['moe6_ok']}")
# ============================================================================================ MOE6 END


# ============================================================================================ entry points
def prepare(weights):
    """Called once, untimed. v5's preparation (v4's extensions + self-checks, the v5 extension + its bitwise check: the fallbacks), then the v6 extension
    and its bitwise self-check against v5."""
    print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")
    if not HAVE_SGLANG:
        return {"weights": weights, "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    state = _prepare_sglang(weights)
    state["smallm"] = None
    if torch.cuda.is_available():
        for flags in ((), ("-DSM_FP32_MUL=1",)):
            try:
                state["smallm"] = prepare_smallm(weights, flags=flags)
            except Exception as ex:  # noqa: BLE001
                print(f"okc-diag smallm unavailable {flags}: {type(ex).__name__}: {str(ex)[:300]}")
                state["smallm"] = None
                break
            if not SELFCHECK:
                break
            err = _smallm_selfcheck(state)
            print(f"okc-diag smallm selfcheck {flags} rel L2 vs Marlin: {err:.3e}")
            if err < SELFCHECK_TOL:
                break
            state["smallm"] = None
        print(f"okc-diag smallm active={state['smallm'] is not None}")
        _prepare_largem_state(state, flags)
        _prepare_moe5_state(state, flags)
        _prepare_moe6_state(state, flags)
    return state


def kernel_fn(data, state):
    if HAVE_SGLANG:
        if _use_moe6(data, state):
            return _kernel_moe6(data, state)
        if _use_moe5(data, state):
            return _kernel_moe5(data, state)
        if _use_smallm(data, state):
            return _kernel_smallm(data, state)
        if _use_largem(data, state):
            return _kernel_largem(data, state)
        return _kernel_sglang(data, state)
    return _kernel_torch(data, state)
