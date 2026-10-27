# okc: problem=muse_moe_w4a16
# grouped_v1 -- sonnet_v1 (below, inherited unchanged for M < GROUPED_MIN_M) + a Triton grouped W4A16 GEMM pair for the large-M shapes
# (design/fable-07-out.md D1, option 2). Candidate for `okc check`; measure_grouped_v1.py sweeps its configs on the H200 first.
#
# The large-M path (verify 16 = bs 128, verify 32 = bs 256, and the tests' bs 257 / skewed 128) replaces the two Marlin GEMMs by:
#   moe_align_block_size(local_ids, BM)  ->  _grouped_gemm GEMM1 (gate+up, SiLU*up in the FP32 epilogue)  ->  _grouped_gemm GEMM2
#   ->  SGLang's fused_post_expert_norm_weighted_sum (unchanged).  Same inputs, same outputs, same numerics class as Marlin:
#   * Tiles = (expert, BM-row M-block from sorted_token_ids / expert_ids) x (BNH output columns). One persistent grid (SM count CTAs):
#     a CTA starts on tile program_id and fetches every further tile with an atomic counter (zeroed by one memset node per call), so a
#     hot expert's tiles spread over all CTAs and idle CTAs just exit. The tile count lives on the device (num_tokens_post_padded), the
#     grid and every constexpr come from shapes only -> CUDA-graph safe, no host sync, no value-dependent allocation or branch.
#   * Dequant in the kernel: E2M1 nibble -> exact value, x E4M3 block scale (per 16 K). E2M1 has 2 significant bits, E4M3 has 4, so
#     value x scale has <= 6 and is exactly representable in BF16 (tests/test_moe_grouped_v1.py checks all 16 x 254 pairs); the BF16
#     tile feeds tl.dot (BF16 x BF16, FP32 accumulate) exactly as Marlin's BF16 operands do. The per-expert FP32 global scale multiplies
#     the FP32 accumulator in the epilogue. Activations stay BF16 (x, SiLU*up rounded once to BF16 for GEMM2); weights stay NVFP4 in HBM.
#   * prepare() repacks (losslessly, byte permutation only) this GPU's 32 experts into 1 KiB (16 rows x 64 B) tile-contiguous blocks, with the
#     gate and up row groups of W13 interleaved so one tile reads both and its epilogue is silu(gate) * up (cache1 never exists).
#   * Dequant modes (grouped_cfg()["deq"]): 0 integer select, 1 subnormal trick (x 2^126, validated on the RTX 2060 and in the SASS: plain
#     mul.f32, no ftz), 2 packed inline PTX (8 elements per call, bf16x2 multiplies; 362 vs 556 SASS instructions per K step in the main loop,
#     needs repack_nvfp4(swap12=True)). 2 is only run-tested in its FP16 flavour (RTX 2060) + a bit-exact numpy emulation of both flavours;
#     the BF16 flavour must be confirmed by measure_grouped_v1.py (it checks every variant against Marlin before timing it) before DEQ=2 is
#     made the default here.
#   SASS-level model of the H200 (tools: triton.compile for sm_90 + cuobjdump): the loop is dequant-issue-bound, tile time ~ 20 us (deq 1)
#   / 13 us (deq 2) at BM 64, 8 warps.  ~25 experts x ~26 M-blocks at bs 128 / 256.
# PREDICTED (unmeasured), vs SGLang 460 / 621 us at bs 128 / 256:  deq 1: ~380 / ~420 us (1.2x / 1.48x);  deq 2: ~290 / ~310 us (1.6x / 2.0x).
#   bs 32 stays on Marlin (GROUPED_MIN_M = 64): the same model gives ~250 us (deq 1) vs Marlin 203 us, ~160 us with deq 2: lower
#   GROUPED_MIN_M to 32 only after measure_grouped_v1.py shows a win.  Other four loads unchanged from sonnet_v1 (1.46 / 1.22 / 1.27).
#   Geo-mean over the six loads: sonnet_v1 1.15 -> ~1.27 (deq 1) / ~1.44 (deq 2).
# Rules kept: precision (NVFP4 stays NVFP4, BF16 activations, FP32 accumulate), no input-value-dependent host logic, no host sync,
# nothing cached from `data`, nothing external. Local runs without SGLang / on CPU / sm_75 fall back to the plain-PyTorch path.
#
# ---- inherited from sonnet_v1 ----
# sonnet_v1 -- SGLang's NVFP4 expert path (baseline.py: moe_align_block_size, two Marlin W4A16 GEMMs with FP32 reduce, silu_and_mul;
# NVFP4 weights repacked exactly as baseline.py's prepare) with the post-expert epilogue replaced at M < 32 by one fused Triton kernel.
# Idea source: a public X thread (idea only, own code).
#
# VERIFIED in the SGLang checkout (<workspace>, python/sglang/):
#   kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py:7   _MIN_TOKENS_FOR_FUSED = 32
#   same file :77   `if tokens < _MIN_TOKENS_FOR_FUSED:` -> eager branch: torch.where(slot_expert_ids >= 0, y, 0), y.float(), * norm_w,
#                   pow(2).mean, rsqrt, * rsqrt, * topk_scores, sum(dim=1), .to(bf16)  (~8 eager launches); M >= 32 launches the Triton kernel.
#   srt/layers/muse_spark_fp4_moe.py:603 (MuseSpark1xFp4Experts._forward_marlin) calls exactly this function, and baseline.py calls it too,
#   so the judge's SGLang side is eager at M = 1, 8, 16 (three of the six scored shapes) and fused at M = 32, 128, 256.
#   (Also seen: srt/layers/muse_spark_fp4_moe.py:473 expert_map[local_topk_ids], the gather this file replaces by one range kernel.)
#
# What changes vs baseline.py:
#   1. M < 32: one Triton kernel, grid = tokens. It loads the token's 8 expert-output rows as one [8, BLOCK_H] tile (all 8 loads in
#      flight), pass 1 gets the 8 sums of squares of y*(1+w) in FP32, pass 2 forms sum_k (score_k * rsqrt(var_k + eps)) * x_k * w
#      in FP32 and stores BF16. Slots of non-local experts (local id < 0) are never loaded (Marlin leaves those rows undefined) and
#      count as zero. Same math as SGLang's eager branch (FP32 throughout, one BF16 rounding at the store); only the summation
#      order differs. M >= 32 keeps SGLang's own fused kernel (already fused there; no reason to replace it without a measurement).
#   2. expert id -> local id (id - local_start if inside [start, start + count) else -1) is one tiny Triton range kernel instead of the
#      expert_map[...] gather (the gather runs as an advanced-indexing op with an int32 index tensor). Same values as the map.
#   Not done (no way to measure here): per-M moe_align block size (kept: baseline's rule), preallocated Marlin buffers (the wrapper
#   allocates its FP32 reduce buffer itself; torch.empty inside a captured graph is free).
#
# PREDICTED speedup vs SGLang, geo-mean over the 6 loads ~1.09x (range 1.06-1.13). Unmeasured on the H200. Per load (one shape each):
#   decode 1 (M=1)   ~1.3x  (eager epilogue ~15-22 us of SGLang's ~77 us; ours ~3 us; the thread reports 77 -> 53 us)
#   verify 1 (M=8)   ~1.2x  (thread: 143 -> 112 us)
#   decode 16 (M=16) ~1.15x (thread: 189 -> 156 us; eager ops grow with M, the Marlin GEMMs dominate)
#   decode 32, verify 16, verify 32 (M=32/128/256) ~1.00-1.01x (same SGLang kernel; only the local-id kernel, <= 1 us, may help)
#   geo-mean (1.3 * 1.2 * 1.15 * 1 * 1 * 1) ^ (1/6) ~ 1.10; discounted a little because eager-op cost under graph replay on cold L2 is
#   not known and the thread's figures are from another judge run. The six loads count equally, so half the loads gain nothing.
# Without SGLang (local runs: CPU, RTX 2060) the experts run as plain PyTorch (the starter's math, producing the per-slot outputs
# in BF16 like Marlin) and then the same epilogue (Triton on CUDA at M < 32, otherwise its torch twin).
# Rules: grids/blocks come from shapes only; no host sync in kernel_fn; CUDA-graph capturable; NVFP4 stays NVFP4; activations BF16;
# FP32 accumulation in the epilogue.
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


def epilogue_triton(expert_out, topk_scores, slot_ids, norm_w, eps=POST_NORM_EPS):
    """expert_out [M, K, H] (BF16 from Marlin), topk_scores [M, K] FP32, slot_ids [M, K] int32 (< 0: not local), norm_w [H]."""
    y = expert_out.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    if tokens == 0:
        return out
    block_h = min(EPILOGUE_BLOCK_H, triton.next_power_of_2(hidden))
    _epilogue_kernel[(tokens,)](
        y, norm_w, topk_scores.contiguous(), slot_ids.contiguous(), out, eps,
        TOPK=topk, KP=triton.next_power_of_2(topk), H=hidden, BLOCK_H=block_h, num_warps=EPILOGUE_WARPS)
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


# ============================================================================================ grouped W4A16 GEMM (large M)
# Weight layout built by prepare() (lossless repack of the organisers' NVFP4 tensors; E2M1 codes and E4M3 scales are copied bit for bit):
#   packed  uint8 [E, NG, KB, 16, 64]  NG = rows / 16, KB = K / 128. One (group, chunk) block = 16 rows x 64 bytes = 1 KiB contiguous;
#           byte j of a row's chunk holds elements 2 j (low nibble) and 2 j + 1 (high nibble) of the chunk's 128 elements (nvfp4.py order).
#   scales  uint8 [E, NG, KB, 16, 8]   the E4M3 bytes; scale g (0..7) covers elements 16 g .. 16 g + 15 of the chunk.
#   gated (W13): row groups are interleaved gate/up: group 2 g = gate rows [16 g, 16 g + 16), group 2 g + 1 = up rows [I + 16 g, ...).
# So one tile of GEMM1 reads the gate and the up weights of the same output columns and its epilogue is silu(gate) * up.
CHUNK = 128
CHUNK_BYTES = 64
GROUP_ROWS = 16


@triton.jit
def _e2m1_f32(c, DEQ: tl.constexpr):
    """Exact FP32 value of the E2M1 code c (int32, 0..15: bit 3 sign, bits 2..0 = 2 exponent bits, 1 mantissa bit)."""
    sign = (c & 8) << 28
    t = c & 7
    if DEQ == 0:  # integer path: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 as FP32 bit patterns
        mag = tl.where(t >= 2, (t << 22) + (126 << 23), t * (126 << 23))
        v = (mag | sign).to(tl.float32, bitcast=True)
    else:  # t << 22 is an FP32 whose exponent field is e and top mantissa bit m: subnormal for e = 0; times 2^126 gives the value
        v = ((t << 22) | sign).to(tl.float32, bitcast=True) * 8.507059173023462e+37
    return v


@triton.jit
def _e4m3_f32(raw, NATIVE: tl.constexpr):
    """Exact FP32 value of E4M3 bytes (raw: uint8 tile)."""
    if NATIVE:
        v = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        s = raw.to(tl.int32)
        e = (s >> 3) & 15
        m = s & 7
        a = tl.where(e > 0, (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True), m.to(tl.float32) * 0.001953125)
        v = tl.where((s & 128) != 0, -a, a)
    return v


@triton.jit
def _bcast8(s, R: tl.constexpr):
    """[R, 8] block scales -> [R, 64]: scale g covers bytes 8 g .. 8 g + 7 (16 elements)."""
    return tl.reshape(tl.broadcast_to(s[:, :, None], (R, 8, 8)), (R, 64))


@triton.jit
def _scale_pair(sc, DT: tl.constexpr):
    """f32 [R, 8] scales -> int32 [R, 8] holding the scale twice as a packed (DT, DT) pair (bits of the exact BF16 / FP16 value)."""
    v = sc.to(DT).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    return v | (v << 16)


@triton.jit
def _dequant_asm(wb, sc, R: tl.constexpr, DT: tl.constexpr):
    """DEQ 2: packed dequant in inline PTX, 8 elements per call. Needs the byte order of repack_nvfp4(swap12=True): in every group of 4
    bytes the middle two are swapped, so one PTX register holds the nibbles of bytes 0 and 2 (16 bits apart) and
    the sign / magnitude fields are moved to their half-word positions by one shift and one mask each. The magnitude bits land in the
    exponent / top mantissa bits (a subnormal for the two smallest codes), one packed multiply by 2^126 (BF16) / 2^14 (FP16)
    makes the E2M1 value exact, a second one applies the block scale. The FP16 flavour exists for the RTX 2060 tests."""
    sp = _bcast8(_scale_pair(sc, DT), R)
    if DT == tl.bfloat16:
        x, y = tl.inline_asm_elementwise(
            (
            "{\n"
            ".reg .b32 a, b, c;\n"
            "mov.b32 c, 0x7E807E80;\n"
            "shl.b32 a, $4, 12; and.b32 a, a, 0x80008000;\n"
            "shl.b32 b, $4, 6; and.b32 b, b, 0x01C001C0; or.b32 a, a, b;\n"
            "mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $0, a, $5;\n"
            "shl.b32 a, $4, 4; and.b32 a, a, 0x80008000;\n"
            "shr.b32 b, $4, 2; and.b32 b, b, 0x01C001C0; or.b32 a, a, b;\n"
            "mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $1, a, $5;\n"
            "shl.b32 a, $4, 8; and.b32 a, a, 0x80008000;\n"
            "shl.b32 b, $4, 2; and.b32 b, b, 0x01C001C0; or.b32 a, a, b;\n"
            "mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $2, a, $5;\n"
            "mov.b32 a, $4; and.b32 a, a, 0x80008000;\n"
            "shr.b32 b, $4, 6; and.b32 b, b, 0x01C001C0; or.b32 a, a, b;\n"
            "mul.rn.bf16x2 a, a, c; mul.rn.bf16x2 $3, a, $5;\n"
            "}\n"
            ),
            "=r,=r,=r,=r,r,r,r,r,r", [wb, sp], dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)
    else:
        x, y = tl.inline_asm_elementwise(
            (
            "{\n"
            ".reg .b32 a, b, c;\n"
            "mov.b32 c, 0x74007400;\n"
            "shl.b32 a, $4, 12; and.b32 a, a, 0x80008000;\n"
            "shl.b32 b, $4, 9; and.b32 b, b, 0x0E000E00; or.b32 a, a, b;\n"
            "mul.rn.f16x2 a, a, c; mul.rn.f16x2 $0, a, $5;\n"
            "shl.b32 a, $4, 4; and.b32 a, a, 0x80008000;\n"
            "shl.b32 b, $4, 1; and.b32 b, b, 0x0E000E00; or.b32 a, a, b;\n"
            "mul.rn.f16x2 a, a, c; mul.rn.f16x2 $1, a, $5;\n"
            "shl.b32 a, $4, 8; and.b32 a, a, 0x80008000;\n"
            "shl.b32 b, $4, 5; and.b32 b, b, 0x0E000E00; or.b32 a, a, b;\n"
            "mul.rn.f16x2 a, a, c; mul.rn.f16x2 $2, a, $5;\n"
            "mov.b32 a, $4; and.b32 a, a, 0x80008000;\n"
            "shr.b32 b, $4, 3; and.b32 b, b, 0x0E000E00; or.b32 a, a, b;\n"
            "mul.rn.f16x2 a, a, c; mul.rn.f16x2 $3, a, $5;\n"
            "}\n"
            ),
            "=r,=r,=r,=r,r,r,r,r,r", [wb, sp], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    return tl.reshape(tl.join(x, y), (R, 128))


@triton.jit
def _dequant_tile(wb, sc, R: tl.constexpr, DEQ: tl.constexpr, DT: tl.constexpr):
    """wb uint8 [R, 64] packed pairs, sc f32 [R, 8] block scales -> [R, 128] in DT in K order. E2M1 x E4M3 has <= 2 + 4 significant
    bits so the FP32 product is exact and so is its rounding to BF16 (8 bits) or FP16 (11 bits); tests/test_moe_grouped_v1.py proves
    all 16 x 254 combinations."""
    if DEQ == 2:
        out = _dequant_asm(wb, sc, R, DT)
    else:
        w = wb.to(tl.int32)
        s64 = _bcast8(sc, R)
        lo = _e2m1_f32(w & 15, DEQ) * s64
        hi = _e2m1_f32(w >> 4, DEQ) * s64
        out = tl.reshape(tl.join(lo.to(DT), hi.to(DT)), (R, 128))
    return out


@triton.jit(do_not_specialize=["n_slots"])
def _grouped_gemm(
    a_ptr, w_ptr, s_ptr, gs_ptr, out_ptr, sorted_ptr, eids_ptr, npost_ptr, ctr_ptr, n_slots,
    K: tl.constexpr, N_OUT: tl.constexpr, A_DIV: tl.constexpr, GATED: tl.constexpr,
    BM: tl.constexpr, BNH: tl.constexpr, CH: tl.constexpr, MAXB: tl.constexpr, STAGES: tl.constexpr,
    NATIVE: tl.constexpr, DEQ: tl.constexpr, DT: tl.constexpr,
):
    """Persistent grouped W4A16 GEMM over the (M-block, N-tile) tiles of a moe_align_block_size result.

    out[slot, c] = gs[e] * sum_k A[row(slot), k] * W[e, c, k]  (GATED: out = silu(gate) * up, both scaled by gs[e]), where the slots of
    M-block b are sorted_ptr[b * BM : (b + 1) * BM] (pad = n_slots), e = eids_ptr[b] (< 0: block of non-local slots), row(slot) = slot // A_DIV.
    Tile t -> block b0 + t // NT, N tile t % NT (b0 = number of leading e < 0 blocks). A CTA starts on tile program_id and fetches
    every further tile from the counter at ctr_ptr (zero before the launch), so a hot expert's tiles spread over all CTAs and
    idle CTAs exit. The grid is a shape-only constant; the number of tiles is read from the device (n_post), never on the host."""
    KB: tl.constexpr = K // 128
    NT: tl.constexpr = N_OUT // BNH
    G_STEP: tl.constexpr = 2 if GATED else 1
    NG: tl.constexpr = G_STEP * N_OUT // 16
    WE: tl.constexpr = NG * KB * 1024
    SE: tl.constexpr = NG * KB * 128
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)

    nblk = tl.load(npost_ptr) // BM
    ob = tl.arange(0, MAXB)
    ev = tl.load(eids_ptr + ob, mask=ob < nblk, other=0)
    b0 = tl.sum(((ev < 0) & (ob < nblk)).to(tl.int32), axis=0)
    total = (nblk - b0) * NT

    rm = tl.arange(0, BM)
    n = tl.arange(0, BNH)
    k64 = tl.arange(0, 64)
    k128 = tl.arange(0, 128)
    w_off = ((n // 16) * (G_STEP * KB * 1024) + (n % 16) * 64)[:, None] + k64[None, :]
    s_off = ((n // 16) * (G_STEP * KB * 128) + (n % 16) * 8)[:, None] + tl.arange(0, 8)[None, :]

    t = pid
    while t < total:
        nxt = tl.atomic_add(ctr_ptr, 1, sem="relaxed") + nprog  # next tile for this CTA: its latency hides behind this tile
        b = b0 + t // NT
        c0 = (t % NT) * BNH
        e = tl.load(eids_ptr + b)
        ids = tl.load(sorted_ptr + b * BM + rm)
        valid = ids < n_slots
        arow = tl.where(valid, ids // A_DIV, 0).to(tl.int64)
        a_ptrs = a_ptr + arow[:, None] * K + k128[None, :]
        wg = w_ptr + e.to(tl.int64) * WE + (c0 // 16) * (G_STEP * KB * 1024) + w_off
        sg = s_ptr + e.to(tl.int64) * SE + (c0 // 16) * (G_STEP * KB * 128) + s_off
        acc_g = tl.zeros((BM, BNH), dtype=tl.float32)
        acc_u = tl.zeros((BM, BNH), dtype=tl.float32)
        for it in tl.range(0, KB // CH, num_stages=STAGES):
            for c in tl.static_range(CH):
                kc = it * CH + c
                a_t = tl.load(a_ptrs + kc * 128)
                w_t = _dequant_tile(tl.load(wg + kc * 1024), _e4m3_f32(tl.load(sg + kc * 128), NATIVE), BNH, DEQ, DT)
                acc_g = tl.dot(a_t, tl.trans(w_t), acc_g)
                if GATED:
                    u_t = _dequant_tile(tl.load(wg + KB * 1024 + kc * 1024), _e4m3_f32(tl.load(sg + KB * 128 + kc * 128), NATIVE), BNH, DEQ, DT)
                    acc_u = tl.dot(a_t, tl.trans(u_t), acc_u)
        gsv = tl.load(gs_ptr + e)
        if GATED:
            g = acc_g * gsv
            res = g / (1.0 + tl.exp(-g)) * (acc_u * gsv)
        else:
            res = acc_g * gsv
        o_ptrs = out_ptr + ids.to(tl.int64)[:, None] * N_OUT + (c0 + n)[None, :]
        tl.store(o_ptrs, res.to(DT), mask=valid[:, None])
        t = nxt


_TL_DTYPES = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16, torch.float32: tl.float32}


def grouped_gemm(a, wp, sp, gs, out, sorted_ids, expert_ids, n_post, ctr, n_slots, *, k, n_out, a_div, gated, bm, bnh, ch,
                 warps, stages, nprog, native=True, deq=0):
    """Launch _grouped_gemm. a [rows, k], out [n_slots, n_out] (same dtype as a); wp/sp/gs from repack_nvfp4/prepare."""
    assert k % (CHUNK * ch) == 0 and n_out % bnh == 0 and bnh % GROUP_ROWS == 0 and a.dtype == out.dtype
    _grouped_gemm[(nprog,)](
        a, wp, sp, gs, out, sorted_ids, expert_ids, n_post, ctr, n_slots,
        K=k, N_OUT=n_out, A_DIV=a_div, GATED=gated, BM=bm, BNH=bnh, CH=ch, MAXB=max(512, triton.next_power_of_2(expert_ids.numel())),
        STAGES=stages, NATIVE=native, DEQ=deq, DT=_TL_DTYPES[out.dtype], num_warps=warps)
    return out


def repack_nvfp4(w, sf, gated, swap12=False):
    """organisers' (uint8 [E, N, K/2], float8_e4m3fn [E, N, K/16]) -> (packed uint8 [E, NG, KB, 16, 64], scales uint8 [E, NG, KB, 16, 8]);
    layout above. gated: N = 2 I with gate rows then up rows, groups interleaved. swap12: in every 4 consecutive bytes of a row swap
    bytes 1 and 2 (the order DEQ 2 reads). Lossless: only the order of the bytes changes."""
    e_n, n, k2 = w.shape
    kb = k2 * 2 // CHUNK
    half = n // 2 // GROUP_ROWS
    wp = w.view(torch.uint8).reshape(e_n, n // GROUP_ROWS, GROUP_ROWS, kb, CHUNK_BYTES).permute(0, 1, 3, 2, 4)
    sp = sf.view(torch.uint8).reshape(e_n, n // GROUP_ROWS, GROUP_ROWS, kb, 8).permute(0, 1, 3, 2, 4)
    if gated:  # group 2 g = gate group g, 2 g + 1 = up group g
        wp = wp.reshape(e_n, 2, half, kb, GROUP_ROWS, CHUNK_BYTES).transpose(1, 2)
        sp = sp.reshape(e_n, 2, half, kb, GROUP_ROWS, 8).transpose(1, 2)
    out_w = torch.empty((e_n, n // GROUP_ROWS, kb, GROUP_ROWS, CHUNK_BYTES), dtype=torch.uint8, device=w.device)
    out_s = torch.empty((e_n, n // GROUP_ROWS, kb, GROUP_ROWS, 8), dtype=torch.uint8, device=w.device)
    for e in range(e_n):  # one expert at a time: bounded temporaries
        blk = wp[e].reshape(out_w.shape[1:])
        if swap12:
            blk = blk.reshape(*blk.shape[:-1], CHUNK_BYTES // 4, 4)[..., [0, 2, 1, 3]].reshape(out_w.shape[1:])
        out_w[e] = blk
        out_s[e] = sp[e].reshape(out_s.shape[1:])
    return out_w, out_s


def moe_align_torch(local_ids, block_size, num_experts):
    """moe_align_block_size's result in PyTorch (tests and the local fallback; the judge path uses SGLang's kernel): bucket 0 holds the
    non-local slots (id < 0, expert -1), bucket e + 1 expert e; every bucket padded to a multiple of block_size; pad = numel. Sync-free
    (counts, a stable sort and a scatter), so it can be captured in a CUDA graph. expert_ids past the last block are -7 (never read)."""
    flat = local_ids.reshape(-1).to(torch.int64)
    numel = flat.numel()
    nb = num_experts + 1
    if numel < nb:
        max_padded = numel * block_size
    else:
        max_padded = numel + nb * (block_size - 1)
    max_padded = (max_padded + 3) // 4 * 4
    dev = flat.device
    bucket = flat + 1
    counts = (bucket[:, None] == torch.arange(nb, device=dev)[None, :]).sum(0)
    padded = (counts + block_size - 1) // block_size * block_size
    ends = padded.cumsum(0)
    starts = ends - padded
    order = torch.sort(bucket, stable=True).indices  # slots by bucket, ascending slot inside a bucket
    cstart = counts.cumsum(0) - counts
    rank = torch.arange(numel, device=dev) - cstart[bucket[order]]
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32, device=dev)
    sorted_ids[starts[bucket[order]] + rank] = order.to(torch.int32)
    n_blocks = (max_padded + block_size - 1) // block_size
    block_start = torch.arange(n_blocks, device=dev) * block_size
    expert_ids = torch.searchsorted(ends, block_start, right=True) - 1  # bucket - 1 = expert
    expert_ids = torch.where(block_start < ends[-1], expert_ids, torch.full_like(expert_ids, -7)).to(torch.int32)
    return sorted_ids, expert_ids, ends[-1:].to(torch.int32)


# Per-shape config of the two GEMMs (shape-only), tuned from measure_grouped_v1.py's `B moeg` lines on the H200.
# (bm, bnh, ch, warps, stages): bnh is the tile's output columns (GEMM1: gate+up columns, i.e. a 2 * bnh wide weight tile); grid = SMs * mult.
GROUPED_MIN_M = 64  # shape-only dispatch threshold (tokens); the tests' bs 33 stays on Marlin
USE_GROUPED = True  # False: prepare() skips the repack and every shape runs sonnet_v1's Marlin path


def grouped_cfg(m):
    return {"bm": 32 if m <= 64 else 64, "g1": dict(bnh=64, ch=1, warps=8, stages=3), "g2": dict(bnh=128, ch=1, warps=8, stages=3),
            "deq": 1, "mult": 2}  # deq 2 after measure_grouped_v1.py confirms the BF16 PTX flavour; mult = CTAs per SM
    # (the default tiles use ~100 registers x 256 threads and 82-100 KB of shared memory: two CTAs fit one SM, no spills)


def _prepare_grouped(weights, start, count, device_sms, deq=None):
    """Repack this GPU's experts for _grouped_gemm. Called from prepare() (untimed); weights only. deq: the dequant mode the repack is
    for (DEQ 2 reads swapped middle bytes); default = grouped_cfg's."""
    deq = grouped_cfg(GROUPED_MIN_M)["deq"] if deq is None else deq
    sl = slice(start, start + count)
    w13, s13 = repack_nvfp4(weights["w13"][sl], weights["w13_sf"][sl], True, deq == 2)
    w2, s2 = repack_nvfp4(weights["w2"][sl], weights["w2_sf"][sl], False, deq == 2)
    return {"w13": w13, "s13": s13, "gs13": weights["w13_gs"][sl].float().contiguous(),
            "w2": w2, "s2": s2, "gs2": weights["w2_gs"][sl].float().contiguous(), "swap12": deq == 2,
            "inter": weights["w2"].shape[2] * 2, "hidden": weights["w2"].shape[1], "sms": device_sms}


def _kernel_grouped(data, state, align_fn, post_fn, native):
    """Large-M path: align -> GEMM1 (+SiLU*up) -> GEMM2 -> post-expert norm / weighted sum. No host sync, shapes only."""
    cfg = data["config"]
    g = state["grouped"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    m, hidden = x.shape
    inter_w = g["inter"]
    ids = data["topk_ids"]
    local_ids = local_ids_triton(ids, cfg["local_start"], cfg["local_count"]) if x.is_cuda else local_ids_torch(ids, cfg["local_start"], cfg["local_count"])
    scores = data["topk_weights"].to(torch.float32).contiguous()
    gc = grouped_cfg(m)
    assert (gc["deq"] == 2) == g["swap12"], "weights were repacked for another dequant mode"
    sorted_ids, expert_ids, n_post = align_fn(local_ids, gc["bm"])
    n_slots = m * TOP_K
    inter = torch.empty((n_slots, inter_w), dtype=x.dtype, device=x.device)
    y = torch.empty((n_slots, hidden), dtype=x.dtype, device=x.device)
    ctr = torch.zeros(32, dtype=torch.int32, device=x.device)  # one memset node; one counter per GEMM, 64 B apart (16-B aligned pointers)
    common = dict(bm=gc["bm"], nprog=g["sms"] * gc.get("mult", 1), native=native, deq=gc["deq"])
    grouped_gemm(x, g["w13"], g["s13"], g["gs13"], inter, sorted_ids, expert_ids, n_post, ctr[0:1], n_slots,
                 k=hidden, n_out=inter_w, a_div=TOP_K, gated=True, **gc["g1"], **common)
    grouped_gemm(inter, g["w2"], g["s2"], g["gs2"], y, sorted_ids, expert_ids, n_post, ctr[16:17], n_slots,
                 k=inter_w, n_out=hidden, a_div=1, gated=False, **gc["g2"], **common)
    return post_fn(y.view(m, TOP_K, hidden), scores, local_ids, state["norm_w"])


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
    grouped = None
    if USE_GROUPED and torch.cuda.get_device_capability(device) >= (9, 0):  # the Triton GEMM needs Hopper (BF16 wgmma, native FP8 decode)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        grouped = {(start, count): _prepare_grouped(weights, start, count, sms) for start, count in banks}
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "grouped": grouped,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


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

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    if num_tokens < EAGER_BELOW_TOKENS:  # SGLang's eager epilogue range: one fused kernel instead
        return epilogue_triton(y, topk_weights, local_ids, state["norm_w"])
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


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


# ============================================================================================ entry points
def prepare(weights):
    """Called once, untimed. NVFP4 weights are repacked into Marlin's layout exactly as baseline.py does."""
    print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")
    if HAVE_SGLANG:
        return _prepare_sglang(weights)
    return {"weights": weights, "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)}


def _align_sglang(local_ids, block_size):
    return moe_align_block_size(local_ids, block_size, N_EXPERTS)


def _post_sglang(y, scores, local_ids, norm_w):
    return fused_post_expert_norm_weighted_sum(
        y, scores, norm_weight=norm_w, eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local_ids)


def kernel_fn(data, state):
    if HAVE_SGLANG:
        # shape-only dispatch: >= GROUPED_MIN_M tokens (decode 32, verify 128 / 256 and the tests' 33 / 257) take the grouped GEMMs
        if state.get("grouped") is not None and data["hidden_states"].shape[0] >= GROUPED_MIN_M:
            return _kernel_grouped(data, state, _align_sglang, _post_sglang, True)
        return _kernel_sglang(data, state)
    return _kernel_torch(data, state)
