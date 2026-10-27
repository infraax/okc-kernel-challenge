# okc: problem=muse_moe_w4a16
# MEASUREMENT-ONLY (card C30): NEVER submit this file -- run it as `okc check` only; it prints a measurement, its kernel_fn is the board's sonnet_v5.
"""MEASUREMENT-ONLY candidate (card C30: the persistent warp-specialised kernel of sonnet_v7 against the board's v5; design/fable-10-out.md D3). Run as `okc check`,
NEVER submit.

kernel_fn is sonnet_v5's, byte for byte (the region from `import re` to v5's entry points and `def kernel_fn` are verbatim copies; tests/test_measure_ws_v1.py
compares them), so the judge's own timing of this file re-measures the board entry (v5, 1.486x). prepare() does v5's preparation (v4's extensions + self-checks,
the v5 extension + its bitwise check), builds the measurement extension with the 8 instantiations below (C31 adds no template: the twins are run-time modes of the DBG = 1 ones) (`-Xptxas -v`, verbose build log: regs / spills per
instantiation appear in the log BEFORE the summary; the build time is printed), runs sonnet_v7's bitwise self-check (v7 vs v5, real mma.sync, real mbarriers and
cp.async.bulk) and then `_measure(state)` once, untimed, inside prepare's 300 s (budget-aware: variants are dropped, never seeds, when the deadline nears), on
inputs of reference.generate_input's recipe (`_gen_input`, bit-identical routing; 3 seeds per M; M in 256, 128, 32, 16, 8, 1). Nothing measured feeds kernel_fn.
Every timing is a CUDA-graph replay after a READ-ONLY L2 flush (128 MB sum), median of REPS replays, like the judge. GEMM1 / GEMM2 of a v7 variant are timed in their
own graphs, each behind a 2-int memset that resets the item counters (the real pipeline's route kernel does that); the memset's own graph time is printed as zero=<us>
in the header for subtraction (v5's numbers carry no such node, so every v7 number is conservative by about that much).

Instantiations (FUSE, D, DBG) = {GEMM1 (true), GEMM2 (false)} x {D1, D2, D1 + twins, D2 + twins}; S (ring slots) and KS are runtime arguments:
  d1s8      WS D1 S8, the design: 8 slots of 18 KB at 32 activation rows, 80 bulk copies per stage, 97 KB of weights in flight         (bitwise v5 expected)
  d2s5      WS D2 S5: half the copies per byte (1 KB word copies), 5 slots of 35.5 KB, 110 KB of weights in flight                   (bitwise v5 expected)
  nocomp    d1s8 with the consumers folding the slot (a sink) instead of dequant + mma: the producer / ring ceiling (streamed TB/s)     nocomp2 = the same on d2s5
  consonly  d1s8 with every stage fetched from one L2-hot region (first tile of the first 24 groups): the consumer ceiling             consonly2 = the same on d2s5
  ks2/4/8   d1s8 with the K split changed (bs 256, 8, 1): not bitwise v5 (bw= shows the relative L2 vs v5; the gate bound is 3e-3)
  pdloff    d1s8 with GEMM2 launched without programmatic dependent launch (the weight prefetch before griddepcontrol.wait)
  co_*      C31, bs 256 / 128 / 16 / 1 only, TIMING ONLY (bw=-): the consumer-chain twins, each = consonly2 (D2 S5, the producer copies from one L2-hot region, so the producer is
            out of the picture) with ONE part of the consumer chain removed (run-time dbg modes of the DBG = 1 template, the sink keeps the removed parts' inputs alive):
              co_nounpack  no unpack_word / s_lut / mul_bf16x2: the raw lds words are the A fragment                         (b) the INT / ALU dequant
              co_nomma     lds + unpack + lut + mul stay, the mma becomes an XOR fold of A and B into the sink                 (c) the tensor-core rate
              co_noldm     B (the activations) is ldmatrix-ed once per ring stage instead of once per k16 pair                 the ldmatrix share of (a)
              co_nofix     no split-K fixup / epilogue (garbage output); the accumulators are folded into the sink              (d) fixup + epilogue
              co_ldsonly   the ring lds (words + scales) folded into the sink; no ldmatrix, unpack or mma                       the lds floor of (a)
              co_smem      lds + ldmatrix + fold (the NOCOMPUTE consumer): no unpack, no mma                                    the whole smem floor of (a)
  wstream   plain kernel streaming the same weight bytes of v5's chunks (16 loads in flight per lane, 4 CTAs/SM): the layout's ceiling
  stamps    d1s8 (stamps2: d2s5) with %globaltimer stamps per warp -> the S lines
Lines (summary block printed LAST = inside the 16 KB tail; <= 14 KB; S lines first):
  MEAS-WS gpu=.. cap=.. budget=..s bw_ach=<TB/s> (CAL: 64 MB read, 4 CTAs/SM, 16 B/lane) cal=<us> zero=<us> inst=8 build=<s> d1=<D>x<S> d2=<D>x<S> ctas=<n>
  S moe m=<M> <variant> g1 items/cta=<n> cons_wait_full=<%> prod_wait_empty=<%> fix_epi=<%> bw/sm=<GB/s> span=<us> tail=<us> cps=<bulk copies per stage> | g2 ..
      (cons_wait_full = the 12 consumer warps' time in try_wait(full) / their life; prod_wait_empty = the producer's time in try_wait(empty) / its life; fix_epi = split-K
       fixup + epilogue / consumer life; bw/sm = the producer's issued bytes / its life; tail = last CTA end - median CTA end; one seed, flushed, eager)
  I moe m=<M> seeds=11/12/13 pairs=.. distinct=.. chunks=.. uniq1=<MB> str1=<MB> uniq2=<MB> gat=<us>      (per seed; uniq = distinct experts x expert bytes, str = chunks x expert bytes;
      gat = the k-tiled activation gather pre-pass alone, seed 11 -- it is INSIDE every v7 g1 / pair number)
  B moe m=<M> v=<variant> | g1=<us>/<us>/<us> g2=<us>/<us>/<us> | ts=<TB/s g1>/<g2> x=<v5/variant g1>/<g2> bw=<OK | e=rel L2 vs v5> [pair=<us>..] [!INVALID]
      (ts = STREAMED TB/s = chunks x expert bytes / time -- the design's basis (510 MB at bs 256); !INVALID = above 4.5 TB/s unique or faster than wstream: the compiler
       deleted work (fable-09 P6))
  S moe m=<M> co-saves (of consonly2 g1=<us> g2=<us>, g1%/g2%): nounpack=<%>/<%> nomma=.. noldm=.. nofix=.. ldsonly=.. smem=..      (% of consonly2's time each twin SAVES; negative = slower)
  S verdict m=<M>: every rule that fires -- NOMMA saves >= 35 % -> TC-bound (wgmma, fable-10 D5-B); NOUNPACK >= 35 % -> ALU-bound (the dequant is the lever); the lds / lds + ldmatrix
      floor >= 60 % of the time or NOLDM >= 20 % -> smem-bound; NOFIX >= 20 % -> the fixup must overlap; none -> distributed chain (GEMM1 numbers)
  D (fable-10 D3) best=.. g1=.. nocomp=.. consonly=.. cons_wait_full=.. prod_wait_empty=.. bitwise=.. -> the decision-table row that fires
  MEAS-WS done <s> [skipped: ..]
Decision rule: fable-10 D3 (the header of sonnet_v7.py); the D line applies it to bs 256.
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



# ============================================================================================ MOE7 BEGIN
# Persistent warp-specialised unified kernel for 1..256 tokens (card C30, fable-10 D2): the header describes it. The packed weights and the large-M scratch
# (chunk table, slot list, split-K partials, counters) are v4's; the extension, the item counters and the ring are new.
import time
_MOE7_EXTRA_SRC = r"""
#define M7_MAXS 16
#define M7_NCONS 12
#define M7_THREADS 416
#define M7_GPI 24
#define M7_TILE 576
#define M7_APITCH 72     // activation row pitch in elements (64 + 8: 144 B, conflict-free for ldmatrix)
#define M7_STAMPS 1
#define M7_NOCOMPUTE 2
#define M7_CONSONLY 4
#define M7_COVER 8
#define M7_FAULT 16
#define M7_BOUND 32   // reserved: every wait is bounded
// consumer-chain twins (C31; DBG = 1 only, on top of CONSONLY so the producer is out of the picture; each removes ONE part of the chain)
#define M7_CO_NOUNPACK 64    // no unpack_word / s_lut / mul_bf16x2: the raw lds words are the A fragment (scales folded into the sink)
#define M7_CO_NOMMA 128      // lds + unpack + lut + mul stay, the mma is an XOR fold of A and B into the sink
#define M7_CO_NOLDM 256      // B (the activations) is ldmatrix-ed once per item into registers, not once per k16
#define M7_CO_NOFIX 512      // no split-K fixup / epilogue: the item ends after its last stage (acc folded into the sink)
#define M7_CO_LDSONLY 1024   // the ring lds only (words + scales folded into the sink): no ldmatrix, no unpack, no mma
#define M7_CO_ANY (M7_CO_NOUNPACK | M7_CO_NOMMA | M7_CO_NOLDM | M7_CO_NOFIX | M7_CO_LDSONLY)
#define M7_REC 8
#define M7_MAXCTA 256
#define M7_COV_MAX 8192
#define M7_COV_OFF_U64 (M5_OCC + M7_MAXCTA * 13 * M7_REC)

// ---- named barrier of the 12 consumer warps (the producer warp never joins; __syncthreads is not used in the kernel body) ----
__device__ __forceinline__ void m7_cbar() { asm volatile("bar.sync 1, 384;" ::: "memory"); }

// ---- ring synchronisation. M7_HW (sm_90): mbarrier (full: count 1 + expect_tx; empty: count 12) and cp.async.bulk 1-D copies. M7_MB without M7_HW (test builds
// with -DM7_EMU_MBAR=1 on the sm_75 dev card): the same mbarrier PROTOCOL (phase parity, arrival count, tx-count, expect_tx before the copies) on a software mbarrier in
// smem and synchronous copies that complete their bytes: tests the parity / byte accounting of the sm_90 call sites. Neither: smem counters (full[s] = completed fills,
// empty[s] = consumer arrivals), synchronous lane-loop copies, bounded spins -- the portable fallback. Same call sites in all three. ----
#if __CUDA_ARCH__ >= 900
#define M7_HW 1
#define M7_MB 1
#elif defined(M7_EMU_MBAR)
#define M7_HW 0
#define M7_MB 1
#else
#define M7_HW 0
#define M7_MB 0
#endif

__device__ __forceinline__ u32 m7_ld_vol(const u32* p) {
    u32 v;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory");
    return v;
}
__device__ __forceinline__ void m7_st_vol(u32* p, u32 v) { asm volatile("st.volatile.shared.u32 [%0], %1;" :: "r"(smem_u32(p)), "r"(v) : "memory"); }

#if M7_HW
__device__ __forceinline__ void m7_mbar_init(u64* b, u32 count) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(count) : "memory"); }
__device__ __forceinline__ void m7_mbar_arrive_tx(u64* b, u32 tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void m7_mbar_arrive(u64* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory"); }
__device__ __forceinline__ bool m7_try_wait(u64* b, u32 parity) {
    u32 ok;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");
    return ok != 0u;
}
#elif M7_MB
// software mbarrier: bits 0..31 tx-count (signed), 32..46 pending arrivals, 47 phase, 48..62 the arrival count of a phase. A phase completes when the pending
// arrivals AND the tx-count are zero (then the phase flips and the arrivals are re-armed), as the hardware's.
__device__ __forceinline__ void m7e_apply(u64* b, int d_pending, int d_tx) {
    unsigned long long* a = reinterpret_cast<unsigned long long*>(b);
    __threadfence_block();
    unsigned long long old = *reinterpret_cast<volatile unsigned long long*>(a);
    for (;;) {
        int pending = (int)((old >> 32) & 0x7fffu), tx = (int)(u32)old;
        u32 phase = (u32)((old >> 47) & 1u);
        const u32 count = (u32)((old >> 48) & 0x7fffu);
        pending += d_pending;
        tx += d_tx;
        if (pending == 0 && tx == 0) {
            phase ^= 1u;
            pending = (int)count;
        }
        const unsigned long long nw = ((unsigned long long)count << 48) | ((unsigned long long)phase << 47) | ((unsigned long long)(u32)pending << 32) | (unsigned long long)(u32)tx;
        const unsigned long long prev = atomicCAS(a, old, nw);
        if (prev == old) break;
        old = prev;
    }
}
__device__ __forceinline__ void m7_mbar_init(u64* b, u32 count) { *reinterpret_cast<volatile unsigned long long*>(b) = ((unsigned long long)count << 48) | ((unsigned long long)count << 32); }
__device__ __forceinline__ void m7_mbar_arrive_tx(u64* b, u32 tx) { m7e_apply(b, -1, (int)tx); }
__device__ __forceinline__ void m7_mbar_arrive(u64* b) { m7e_apply(b, -1, 0); }
__device__ __forceinline__ bool m7_try_wait(u64* b, u32 parity) {
    const unsigned long long v = *reinterpret_cast<volatile unsigned long long*>(b);
    return (u32)((v >> 47) & 1u) != parity;      // the phase of this parity has completed iff the barrier is in the other one
}
#endif

__device__ __forceinline__ void m7_init_bars(u64* bars, int S) {
#if M7_MB
    for (int s = 0; s < S; ++s) {
        m7_mbar_init(bars + s, 1u);
        m7_mbar_init(bars + M7_MAXS + s, (u32)M7_NCONS);
    }
#if M7_HW
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
#else
    u32* c = reinterpret_cast<u32*>(bars);
    for (int s = 0; s < 2 * M7_MAXS; ++s) c[s] = 0u;
#endif
}

// the stage use number `gen` of slot s has landed (full barrier); false = timed out. EVERY wait is a bounded spin (spin_max polls; a poll of try_wait is a
// hardware-suspended wait of up to a few microseconds on sm_90): a protocol bug ends the kernel with garbage output -- which prepare()'s self-check turns into a
// fallback to v5 -- instead of hanging prepare() or a timed call.
__device__ __forceinline__ bool m7_wait_full(u64* bars, int s, int gen, u32 spin_max) {
#if M7_MB
    const u32 par = (u32)(gen & 1);
    for (u32 i = 0; i < spin_max; ++i)
        if (m7_try_wait(bars + s, par)) {
#if !M7_HW
            __threadfence_block();
#endif
            return true;
        }
    return false;
#else
    const u32* c = reinterpret_cast<const u32*>(bars) + s;
    for (u32 i = 0; i < spin_max; ++i)
        if (m7_ld_vol(c) > (u32)gen) {
            __threadfence_block();
            return true;
        }
    return false;
#endif
}

// the slot s is free for its use number `gen` (the 12 consumer warps released use gen - 1); gen 0 is free at once
__device__ __forceinline__ bool m7_wait_empty(u64* bars, int s, int gen, u32 spin_max) {
#if M7_MB
    const u32 par = (u32)((gen & 1) ^ 1);
    for (u32 i = 0; i < spin_max; ++i)
        if (m7_try_wait(bars + M7_MAXS + s, par)) {
#if !M7_HW
            __threadfence_block();
#endif
            return true;
        }
    return false;
#else
    const u32* c = reinterpret_cast<const u32*>(bars) + M7_MAXS + s;
    for (u32 i = 0; i < spin_max; ++i)
        if (m7_ld_vol(c) >= (u32)gen * (u32)M7_NCONS) {
            __threadfence_block();
            return true;
        }
    return false;
#endif
}

__device__ __forceinline__ void m7_arrive_empty(u64* bars, int s) {
#if M7_MB
    m7_mbar_arrive(bars + M7_MAXS + s);
#else
    __threadfence_block();
    atomicAdd(reinterpret_cast<u32*>(bars) + M7_MAXS + s, 1u);
#endif
}

// producer, one elected lane: announce the bytes of the stage (the counter fallback needs no announcement)
__device__ __forceinline__ void m7_full_expect(u64* bars, int s, u32 bytes) {
#if M7_MB
    m7_mbar_arrive_tx(bars + s, bytes);
#endif
}

// producer, whole warp, after the copies of the stage were issued: the counter fallback's copies are synchronous, so it publishes the fill now
__device__ __forceinline__ void m7_full_commit(u64* bars, int s, int gen) {
#if !M7_MB
    __syncwarp();
    if ((threadIdx.x & 31) == 0) {
        __threadfence_block();
        m7_st_vol(reinterpret_cast<u32*>(bars) + s, (u32)gen + 1u);
    }
#endif
}

// 1-D copy global -> shared, completing on the full barrier of slot s. WARP-COLLECTIVE (all 32 lanes call it with the same arguments). sm_90: ONE elected lane
// issues the cp.async.bulk (a lane-varying bulk copy is lowered to an ELECT / R2UR / UBLKCP loop of ~10 instructions per active lane -- the first H200 run was
// bound by exactly that). Otherwise the lanes copy 16 B at a time in parallel and the software mbarrier is told the bytes landed.
__device__ __forceinline__ void m7_bulk(u8* dst, const void* src, u32 bytes, u64* bars, int s) {
#if M7_HW
    if ((threadIdx.x & 31) == 0)
        asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                     :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bars + s)) : "memory");
#else
    for (u32 o = (threadIdx.x & 31) * 16; o < bytes; o += 512)
        *reinterpret_cast<uint4*>(dst + o) = *reinterpret_cast<const uint4*>(reinterpret_cast<const u8*>(src) + o);
#if M7_MB
    __syncwarp();
    if ((threadIdx.x & 31) == 0) m7e_apply(bars + s, 0, -(int)bytes);
#endif
#endif
}

struct M7Pos {  // position in the ring: slot s, its use number gen, the byte offset of the slot
    int s, gen;
    u32 off;
};
__device__ __forceinline__ void m7_adv(M7Pos& p, int S, int slot_bytes) {
    p.off += (u32)slot_bytes;
    if (++p.s == S) {
        p.s = 0;
        ++p.gen;
        p.off = 0u;
    }
}

struct M7P {  // kernel arguments and derived constants shared by the roles
    u64* bars;
    u8* ring;
    float4* part;
    int* cnt;
    u16* out;
    const float* gsc;
    int* s_flag;
    const u32* s_lut;
    const int* slotl;
    const int* sloc;
    u64* dbg;
    int G, KS, NI, S, slot_bytes, out_stride, nst, R;
    int dm;
    u32 spin_max;
};
struct M7Item {
    int z, ks, ng, e, mc, g0, soff, nval;
    bool active;
};

// Split-K fixup + epilogue of one item, consumers only (named barrier). m6_finish's order of operations: partials (fragment layout), the last arriving
// item of (z, ng) adds them in slice order 0..KS-1 from 0.f, then SwiGLU + BF16 (GEMM1) or the plain BF16 store (GEMM2).
template <bool FUSE, int NT>
__device__ __forceinline__ void m7_finish(const M7P& P, const M7Item& it, float (&acc)[2][NT][4]) {
    const int tid = threadIdx.x, lane = tid & 31, gid = lane >> 2, t4 = lane & 3;
    const int G = P.G, KS = P.KS;
    if (KS > 1) {
        if (it.active) {
            const u64 pb = (((u64)it.z * G + it.g0) * KS + it.ks) * 128 + lane;
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    P.part[pb + (u64)r * KS * 128 + nt * 32] = make_float4(acc[r][nt][0], acc[r][nt][1], acc[r][nt][2], acc[r][nt][3]);
        }
        __threadfence();
        m7_cbar();
        if (tid == 32) {
            const int ci = it.z * P.NI + it.ng;
            const int prev = atomicAdd(&P.cnt[ci], 1);
            const int last = (prev == KS - 1);
            *P.s_flag = last;
            if (last) P.cnt[ci] = 0;
        }
        m7_cbar();
        if (!*P.s_flag) return;
        __threadfence();
        if (it.active) {
#pragma unroll
            for (int r = 0; r < 2; ++r)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    uint4 v[8];
#pragma unroll
                    for (int k2 = 0; k2 < 8; ++k2) {
                        if (k2 < KS) v[k2] = ldcg4(P.part + (((u64)it.z * G + it.g0 + r) * KS + k2) * 128 + lane + nt * 32);
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
    }
    if (!it.active) return;
    const int* sl = P.slotl + it.soff;                          // recomputed here: a 64-bit pointer and a float less live across the k-loop
    const float fscale = P.gsc[it.e] * 1073741824.0f;           // global scale * 2**30 (undoes the operand scaling)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int g = it.g0 + r;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int tk = nt * 8 + 2 * t4 + c;
                if (tk < it.mc) {
                    if (FUSE) {
                        // GEMM1 writes the intermediate straight into GEMM2's k-tiled activation layout [chunk][K/64][R rows][72] (the pad columns stay unwritten):
                        // column n = 8 g + gid of the slot's row sits in k-tile n / 64 at column n % 64; loc = chunk * 32 + row (written by the route kernel)
                        const int loc = P.sloc[sl[tk]];
                        u16* o = P.out + (((u64)(loc >> 5) * (P.G >> 3) + (g >> 3)) * P.R + (loc & 31)) * M7_APITCH + (g & 7) * 8 + gid;
                        const float gt = acc[r][nt][c] * fscale, up = acc[r][nt][2 + c] * fscale;
                        *o = f2bf((gt / (1.0f + expf(-gt))) * up);
                    } else {
                        u16* o = P.out + (u64)sl[tk] * P.out_stride;
                        o[g * 16 + gid] = f2bf(acc[r][nt][c] * fscale);
                        o[g * 16 + 8 + gid] = f2bf(acc[r][nt][2 + c] * fscale);
                    }
                }
            }
        }
    }
}

// The consumer-chain twins' k-loop body (C31): m5_batch's body with ONE part of the chain switched off by the run-time mode (uniform branches; only the DBG = 1
// template contains it). The sink keeps every removed part's inputs alive.
template <int NT, bool UNPACK, bool MMA>
__device__ __forceinline__ void m7_batch_twin(const uint4 (&w)[2][1], const uint2 (&s)[2][1], u32 la, int kp_bytes, const u32* s_lut, float (&acc)[2][NT][4],
                                              u32& sink) {
    constexpr bool nounpack = !UNPACK, nomma = !MMA;
#pragma unroll
    for (int jp = 0; jp < 2; ++jp) {
        u32 bf[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) ldmatrix_x4(bf[nt], la + (u32)(nt * 8 * kp_bytes) + (u32)(jp * 2 * 32));
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            const int j = jp * 2 + jj;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const u32 wv[4] = {w[r][0].x, w[r][0].y, w[r][0].z, w[r][0].w};
                u32 a[4];
                if (nounpack) {
                    a[0] = wv[j & 3]; a[1] = wv[(j + 1) & 3]; a[2] = wv[(j + 2) & 3]; a[3] = wv[(j + 3) & 3];
                    if (jj == 0 && jp == 0) sink ^= s[r][0].x ^ s[r][0].y;
                } else {
                    u32 q[4];
                    unpack_word(wv[j], q);
                    const u32 sc0 = s_lut[(s[r][0].x >> (8 * j)) & 255u];
                    const u32 sc1 = s_lut[(s[r][0].y >> (8 * j)) & 255u];
                    a[0] = mul_bf16x2(q[0], sc0); a[1] = mul_bf16x2(q[1], sc1); a[2] = mul_bf16x2(q[2], sc0); a[3] = mul_bf16x2(q[3], sc1);
                }
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    if (nomma) sink ^= a[0] ^ a[1] ^ a[2] ^ a[3] ^ bf[nt][2 * jj] ^ bf[nt][2 * jj + 1];
                    else mma16816(acc[r][nt], a, bf[nt][2 * jj], bf[nt][2 * jj + 1]);
                }
            }
        }
    }
}

// CO_NOLDM's k-loop body: m5_batch's compute with B (the activations) taken from registers loaded ONCE per ring stage, not once per k16 pair (the data of the later
// k16 pairs are wrong -- timing only). A separate code path (exclusive with the others) so the extra registers are not carried by the normal path.
template <int NT>
__device__ __forceinline__ void m7_batch_bfreg(const uint4 (&w)[2][1], const uint2 (&s)[2][1], const u32 (&bf)[NT][4], const u32* s_lut, float (&acc)[2][NT][4]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int jj = j & 1;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const u32 wv[4] = {w[r][0].x, w[r][0].y, w[r][0].z, w[r][0].w};
            u32 q[4];
            unpack_word(wv[j], q);
            const u32 sc0 = s_lut[(s[r][0].x >> (8 * j)) & 255u];
            const u32 sc1 = s_lut[(s[r][0].y >> (8 * j)) & 255u];
            const u32 a[4] = {mul_bf16x2(q[0], sc0), mul_bf16x2(q[1], sc1), mul_bf16x2(q[2], sc0), mul_bf16x2(q[3], sc1)};
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) mma16816(acc[r][nt], a, bf[nt][2 * jj], bf[nt][2 * jj + 1]);
        }
    }
}

// m5_batch with the twins selected at run time in the DBG = 1 template only (the DBG = 0 template never instantiates them).
template <int NT, int DBG>
struct M7Batch {
    static __device__ __forceinline__ void run(const uint4 (&wv)[2][1], const uint2 (&sv)[2][1], int kk0, u32 la, int kp_bytes, const u32* s_lut, float (&acc)[2][NT][4],
                                               u32& sink, int) {
        m5_batch<NT, 2, 1, 0>(wv, sv, kk0, la, kp_bytes, s_lut, acc, sink);
    }
};
template <int NT>
struct M7Batch<NT, 1> {
    static __device__ __forceinline__ void run(const uint4 (&wv)[2][1], const uint2 (&sv)[2][1], int kk0, u32 la, int kp_bytes, const u32* s_lut, float (&acc)[2][NT][4],
                                               u32& sink, int mode) {
        if (mode & M7_CO_LDSONLY) {   // the smem floor of the ring reads: lds.128 words + lds.64 scales, nothing else
#pragma unroll
            for (int r = 0; r < 2; ++r) sink ^= wv[r][0].x ^ wv[r][0].y ^ wv[r][0].z ^ wv[r][0].w ^ sv[r][0].x ^ sv[r][0].y;
        } else if (mode & M7_CO_NOUNPACK) m7_batch_twin<NT, false, true>(wv, sv, la, kp_bytes, s_lut, acc, sink);
        else if (mode & M7_CO_NOMMA) m7_batch_twin<NT, true, false>(wv, sv, la, kp_bytes, s_lut, acc, sink);
        else if (mode & M7_NOCOMPUTE) m5_batch<NT, 2, 1, M5_NOCOMPUTE>(wv, sv, kk0, la, kp_bytes, s_lut, acc, sink);
        else m5_batch<NT, 2, 1, 0>(wv, sv, kk0, la, kp_bytes, s_lut, acc, sink);
    }
};

// One work item of one consumer warp: nst stages, each = D k64 iterations of its 2 groups read from the ring slot (lds.128 words, lds.64 scales; the
// activation slice by ldmatrix) through m5_batch's body (v5's order of operations), then the slot is released. Returns false on a timed-out wait.
template <bool FUSE, int D, int NT, int DBG>
__device__ __forceinline__ bool m7_item(const M7P& P, M7Pos& pos, const M7Item& it, u64& tw, u64& tf, u32& sink) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2;
    constexpr int A_OFF = M7_GPI * D * M7_TILE;
    const int w = warp - 1;
    const u32 lane_rel = (u32)((((lane & 7) * M7_APITCH) + ((lane >> 4) & 1) * 16 + ((lane >> 3) & 1) * 8) * 2);
    const u32 d_bytes = (u32)(P.R * (M7_APITCH * 2));    // one k-tile of the activation slice: R rows x 144 B
    const bool stamps = DBG && (P.dm & M7_STAMPS), nofix = DBG && (P.dm & M7_CO_NOFIX);
    const int mode = DBG ? P.dm : 0;
    float acc[2][NT][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) acc[r][nt][0] = acc[r][nt][1] = acc[r][nt][2] = acc[r][nt][3] = 0.f;
    for (int q = 0; q < P.nst; ++q) {
        if (q > 0) {  // stage 0 of the item was waited for by the caller (it carries the descriptor)
            const u64 t = stamps ? m5_timer() : 0;
            if (!m7_wait_full(P.bars, pos.s, pos.gen, P.spin_max)) return false;
            if (stamps) tw += m5_timer() - t;
        }
        if (it.active) {
            const u8* slot = P.ring + pos.off;
            const u32 la = smem_u32(slot + A_OFF) + lane_rel;
            if (DBG && (mode & M7_CO_NOLDM)) {   // C31 CO_NOLDM: B is ldmatrix-ed ONCE per ring stage (the first k16 pair), then reused by every k16 pair of the stage
                u32 bfs[NT][4];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) ldmatrix_x4(bfs[nt], la + (u32)(nt * 8 * (M7_APITCH * 2)));
#pragma unroll
                for (int d = 0; d < D; ++d) {
                    uint4 wv[2][1];
                    uint2 sv[2][1];
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const u8* tile = slot + (d * it.nval + 2 * w + r) * M7_TILE;
                        wv[r][0] = *reinterpret_cast<const uint4*>(tile + lane * 16);
                        sv[r][0] = *reinterpret_cast<const uint2*>(tile + 512 + gid * 8);
                    }
                    m7_batch_bfreg<NT>(wv, sv, bfs, P.s_lut, acc);
                }
            } else {
#pragma unroll
            for (int d = 0; d < D; ++d) {   // the D k64 iterations of the stage, one at a time (m5_batch with D = 1): 12 registers of weights
                uint4 wv[2][1];
                uint2 sv[2][1];
#pragma unroll
                for (int r = 0; r < 2; ++r) {   // slot layout = the global blob's: [k64 iteration d][group][576 B tile = 512 B words + 64 B scales]
                    const u8* tile = slot + (d * it.nval + 2 * w + r) * M7_TILE;
                    wv[r][0] = *reinterpret_cast<const uint4*>(tile + lane * 16);
                    sv[r][0] = *reinterpret_cast<const uint2*>(tile + 512 + gid * 8);
                }
                M7Batch<NT, DBG>::run(wv, sv, 0, la + d * d_bytes, M7_APITCH * 2, P.s_lut, acc, sink, mode);
            }
            }
        }
        if (DBG) sink = (sink << 1) | (sink >> 31);   // (the twins' XOR folds of identical L2-hot stages must not cancel pairwise: rotate once per stage)
        __syncwarp();
        if (lane == 0) m7_arrive_empty(P.bars, pos.s);
        m7_adv(pos, P.S, P.slot_bytes);
    }
    const u64 t1 = stamps ? m5_timer() : 0;
    if (nofix) {   // C31 CO_NOFIX: no split-K fixup / epilogue (garbage output); the accumulators are folded so the k-loop stays alive
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) sink ^= __float_as_uint(acc[r][nt][0]) ^ __float_as_uint(acc[r][nt][1]) ^ __float_as_uint(acc[r][nt][2]) ^ __float_as_uint(acc[r][nt][3]);
    } else {
        m7_finish<FUSE, NT>(P, it, acc);
    }
    if (stamps) tf += m5_timer() - t1;
    return true;
}

// Persistent warp-specialised kernel: grid = SM count CTAs of 13 warps. Warp 0 = producer: takes items (z, ks, ng) from the device counter ctr[0] and fills
// the S-slot ring with cp.async.bulk copies (24 groups' words + scales, the chunk's activation rows); warps 1..12 = consumers, 2 groups each.
// DBG (template) = 1 builds the runtime-selected twins (dbg_mode bits M7_*); DBG 0 has none of them.
template <bool FUSE, int D, int DBG>
__global__ void __launch_bounds__(M7_THREADS, 1)
w4_moe7_kernel(const u8* __restrict__ Wb, const float* __restrict__ gsc, const u32* __restrict__ lut, const u16* __restrict__ actT, const int* __restrict__ ctab,
               const int* __restrict__ slotl, const int* __restrict__ sloc, float4* __restrict__ part, int* __restrict__ cnt, u16* __restrict__ out,
               int* __restrict__ ctr, u64* __restrict__ dbg, int G, int NIT, int K, int KS, int S, int slot_bytes, int R, int out_stride, int dbg_mode, int spin_max) {
    extern __shared__ __align__(16) u8 smem_raw[];
    __shared__ __align__(8) u64 s_bars[2 * M7_MAXS];
    __shared__ int s_desc[4][8];
    __shared__ M7P s_P;   // the consumers' constants live in smem (read at the use sites): the NT 4 path needs every register of the 128-register cap
    __shared__ int s_flag;
    constexpr int A_OFF = M7_GPI * D * M7_TILE;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int dm = DBG ? dbg_mode : 0;
    const bool stamps = (dm & M7_STAMPS) != 0, nocomp = (dm & (M7_NOCOMPUTE | M7_CO_ANY)) != 0;   // (nocomp: the sink is stored)
    u32* s_lut = reinterpret_cast<u32*>(smem_raw);
    u8* ring = smem_raw + 1024;
    const int Kc = K / KS, nit = Kc >> 6, nst = nit / D, NI = (G + M7_GPI - 1) / M7_GPI, KT = K >> 6;

    // ---- setup: barriers, the LUT, the consumers' constants, then one block barrier (the ring needs no initialisation: every stage is two full-size copies) ----
    if (tid == 0) {
        m7_init_bars(s_bars, S);
        s_P.bars = s_bars; s_P.ring = ring; s_P.part = part; s_P.cnt = cnt; s_P.out = out; s_P.gsc = gsc; s_P.s_flag = &s_flag; s_P.s_lut = s_lut;
        s_P.slotl = slotl; s_P.sloc = sloc; s_P.dbg = dbg; s_P.G = G; s_P.KS = KS; s_P.NI = NI; s_P.S = S; s_P.slot_bytes = slot_bytes; s_P.out_stride = out_stride;
        s_P.nst = nst; s_P.R = R; s_P.dm = dm; s_P.spin_max = (u32)spin_max;
    }
    for (int i = tid; i < 256; i += M7_THREADS) s_lut[i] = lut[i];
#if M7_HW
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
    __syncthreads();

    const u64 t_begin = ((DBG) && (stamps || nocomp)) ? m5_timer() : 0;
    u64 tw = 0, tf = 0, bytes_issued = 0;
    int items = 0, stages = 0;
    u32 sink = 0;

    if (warp == 0) {
        // ======================================================= producer: 2 bulk copies per stage, both issued by lane 0
        if (FUSE) {
            pdl_wait();  // GEMM1 reads the chunk table / activation tiles / item counter written by the route and gather kernels
            pdl_trigger();
        }
        const int n_items = ctab[0] * KS * NI;
        int s = 0, gen = 0;
        for (int seq = 0;; ++seq) {
            int i = 0;
            if (lane == 0) i = atomicAdd(ctr, 1);
            i = __shfl_sync(FULLMASK, i, 0);
            if (i >= n_items) {  // sentinel: an empty stage that wakes the consumers
                if (!m7_wait_empty(s_bars, s, gen, (u32)spin_max)) {
                    if (DBG && lane == 0) atomicExch(reinterpret_cast<u32*>(dbg) + 2, 1u);
                    break;
                }
                if (lane == 0) {
                    s_desc[seq & 3][0] = -1;
                    m7_full_expect(s_bars, s, 0u);
                }
                m7_full_commit(s_bars, s, gen);
                break;
            }
            const int per_z = KS * NI;
            const int z = i / per_z, rem = i - z * per_z;
            const int ks = rem / NI, ng = rem - ks * NI;
            const int* ct = ctab + 4 + M5_CTW * z;
            const int e = ct[0], mc = ct[1], soff = ct[2];
            const int nval = min(M7_GPI, G - ng * M7_GPI);
            const bool consonly = DBG && (dm & M7_CONSONLY);
            // the weight blob [expert][group block of 24][k64 iteration][group][576 B tile]: D consecutive iterations of one group block are ONE contiguous
            // D x nval x 576 B run (CONSONLY: every stage reads the first run of the first group block of the expert, L2-hot)
            const u8* wsrc = Wb + (((u64)e * G * NIT) + (u64)(consonly ? 0 : ng * M7_GPI) * NIT + (u64)(consonly ? 0 : ks * nit) * nval) * M7_TILE;
            const u32 wbytes = (u32)(D * nval * M7_TILE), abytes = (u32)(D * R * (M7_APITCH * 2));
            // the activation tiles [chunk][K/64][R rows][72]: D consecutive k-tiles of one chunk are ONE contiguous D x R x 144 B run
            const u8* asrc = reinterpret_cast<const u8*>(actT) + ((u64)z * KT + (u64)ks * nit) * (R * (M7_APITCH * 2));
            const u32 bytes_stage = wbytes + abytes;
            if (DBG && (dm & M7_COVER) && lane == 0 && n_items <= M7_COV_MAX)
                atomicAdd(reinterpret_cast<u32*>(dbg) + 2 * M7_COV_OFF_U64 + i, 1u);
            const int npre = (!FUSE && seq == 0) ? min(S - 1, nst) : 0;   // GEMM2: the weights of the first stages are fetched before griddepcontrol.wait
            for (int q = 0; q < nst; ++q) {
                if (DBG && (dm & M7_FAULT) && seq == 0 && q == 3) {   // test twin: a stage that is never filled (the consumers must time out)
                    if (++s == S) {
                        s = 0;
                        ++gen;
                    }
                    continue;
                }
                const u64 t = (DBG && stamps) ? m5_timer() : 0;
                if (!m7_wait_empty(s_bars, s, gen, (u32)spin_max)) {
                    if (DBG && lane == 0) atomicExch(reinterpret_cast<u32*>(dbg) + 2, 1u);
                    goto producer_done;
                }
                if (DBG && stamps) tw += m5_timer() - t;
                u8* slot = ring + (u64)s * slot_bytes;
                if (q == 0 && lane == 0) {
                    s_desc[seq & 3][0] = i; s_desc[seq & 3][1] = z; s_desc[seq & 3][2] = ks; s_desc[seq & 3][3] = ng;
                    s_desc[seq & 3][4] = e; s_desc[seq & 3][5] = mc; s_desc[seq & 3][6] = soff; s_desc[seq & 3][7] = nval;
                }
                if (lane == 0) m7_full_expect(s_bars, s, bytes_stage);
                __syncwarp();
                m7_bulk(slot, wsrc + (consonly ? 0 : (u64)q * wbytes), wbytes, s_bars, s);
                if (q >= npre) m7_bulk(slot + A_OFF, asrc + (u64)q * abytes, abytes, s_bars, s);
                if (q + 1 == npre) {  // the first npre stages hold weights only so far: wait for GEMM1, then add their activation tiles (slots 0..npre-1, use 0)
                    pdl_wait();
                    __syncwarp();
                    for (int p = 0; p < npre; ++p) m7_bulk(ring + (u64)p * slot_bytes + A_OFF, asrc + (u64)p * abytes, abytes, s_bars, p);
                    for (int p = 0; p < npre; ++p) m7_full_commit(s_bars, p, 0);   // (the fallback publishes a stage only when it is complete)
                }
                if (q >= npre) m7_full_commit(s_bars, s, gen);
                bytes_issued += bytes_stage;
                ++stages;
                if (++s == S) {
                    s = 0;
                    ++gen;
                }
            }
            ++items;
            if (DBG && (dm & M7_COVER) && lane == 0 && n_items <= M7_COV_MAX)
                atomicAdd(reinterpret_cast<u32*>(dbg) + 2 * M7_COV_OFF_U64 + M7_COV_MAX + i, (u32)nst);
        }
    producer_done:;
    } else {
        // ======================================================= consumers
        pdl_wait();  // GEMM2: GEMM1 is complete (the split-K scratch is shared; the intermediate is read through the producer)
        if (FUSE) pdl_trigger();
        const int w = warp - 1;
        const M7P& P = s_P;
        M7Pos pos;
        pos.s = 0; pos.gen = 0; pos.off = 0u;
        for (int seq = 0;; ++seq) {
            {
                const u64 t = stamps ? m5_timer() : 0;
                if (!m7_wait_full(s_bars, pos.s, pos.gen, (u32)spin_max)) {
                    if (DBG && lane == 0) atomicExch(reinterpret_cast<u32*>(dbg) + 2, 1u);
                    break;
                }
                if (DBG && stamps) tw += m5_timer() - t;
            }
            const int* d = s_desc[seq & 3];
            if (d[0] < 0) break;
            M7Item it;
            it.z = d[1]; it.ks = d[2]; it.ng = d[3]; it.e = d[4]; it.mc = d[5];
            const int nval = d[7];
            it.g0 = it.ng * M7_GPI + 2 * w;
            it.active = (2 * w < nval);
            it.soff = d[6];
            it.nval = nval;
            bool ok;
            if (it.mc <= 8) ok = m7_item<FUSE, D, 1, DBG>(P, pos, it, tw, tf, sink);
            else if (it.mc <= 16) ok = m7_item<FUSE, D, 2, DBG>(P, pos, it, tw, tf, sink);
            else ok = m7_item<FUSE, D, 4, DBG>(P, pos, it, tw, tf, sink);
            if (!ok) {
                if (DBG && lane == 0) atomicExch(reinterpret_cast<u32*>(dbg) + 2, 1u);
                break;
            }
            ++items;
            stages += nst;
            if (DBG && (dm & M7_COVER) && lane == 0 && d[0] < M7_COV_MAX) atomicAdd(reinterpret_cast<u32*>(dbg) + 2 * M7_COV_OFF_U64 + 2 * M7_COV_MAX + d[0], 1u);
        }
    }
    if (DBG && (stamps || nocomp) && lane == 0 && blockIdx.x < M7_MAXCTA) {
        u64* rec = dbg + M5_OCC + ((u64)blockIdx.x * 13 + warp) * M7_REC;
        rec[0] = t_begin; rec[1] = m5_timer(); rec[2] = tw; rec[3] = tf; rec[4] = (u64)items; rec[5] = (u64)stages; rec[6] = bytes_issued;
        rec[7] = nocomp ? (u64)sink : (u64)m5_smid();
    }
}

// The route kernels of v5 (M <= 16: one 128-thread CTA; larger M: largem_route_kernel's 1024-thread CTA) with one addition: the CTA's first threads zero the
// two item counters ctr[0] / ctr[1] (GEMM1 / GEMM2) of THIS call -- inside the timed call, so graph replays start from zero.
__global__ void __launch_bounds__(128) m7_route_small_kernel(const int* __restrict__ ids, int* __restrict__ local_ids, int* __restrict__ ctab,
                                                             int* __restrict__ slotl, int* __restrict__ sloc, int* __restrict__ ctr, int n, int lo, int nloc) {
    __shared__ int wc[4][32];
    __shared__ int s_off[32], s_cidx[32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_trigger();
    if (tid < 2) ctr[tid] = 0;
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
        int po = c;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int a = __shfl_up_sync(FULLMASK, po, d);
            if (lane >= d) po += a;
        }
        s_off[lane] = po - c;
        s_cidx[lane] = __popc(act & ((1u << lane) - 1u));
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
        sloc[tid] = (s_cidx[e] << 5) | (base + rank_w);   // slot -> (chunk, row in the chunk): the k-tiled activation layout of GEMM2's input
    }
}

__global__ void __launch_bounds__(1024) m7_route_large_kernel(const int* __restrict__ ids, int* __restrict__ local_ids, int* __restrict__ ctab,
                                                              int* __restrict__ slotl, int* __restrict__ sloc, int* __restrict__ ctr, int n, int lo, int nloc) {
    __shared__ int s_e[2048];
    __shared__ int s_cnt[32], s_off[32], s_cb[32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    pdl_trigger();
    if (tid < 2) ctr[tid] = 0;
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
        int po = cn, pc = nch;
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
        if (e == warp) {
            const int rk = run + __popc(m & ((1u << lane) - 1u));
            slotl[off + rk] = s;
            sloc[s] = ((s_cb[warp] + (rk >> 5)) << 5) | (rk & 31);   // slot -> (chunk, row in the chunk)
        }
        run += __popc(m);
    }
}

// Pre-pass of GEMM1 (inside the timed call): gathers every chunk's activation rows into the k-tiled layout [chunk][K/64][R rows][72] bf16 (144 B rows, rows >= mc
// zero, the 8 pad elements unwritten) so that ONE stage's activation slice (D consecutive k-tiles of a chunk) is ONE contiguous bulk copy. grid = (chunk bound, K/64/8).
__global__ void __launch_bounds__(256) m7_gather_kernel(const u16* __restrict__ x, const int* __restrict__ ctab, const int* __restrict__ slotl, u16* __restrict__ dst,
                                                        int K, int act_shift, int R) {
    pdl_wait();
    pdl_trigger();
    const int z = blockIdx.x;
    if (z >= ctab[0]) return;
    const int mc = ctab[4 + M5_CTW * z + 1], soff = ctab[4 + M5_CTW * z + 2];
    const int KT = K >> 6;
    const int kt1 = min(KT, (int)blockIdx.y * 8 + 8);
    for (int kt = blockIdx.y * 8; kt < kt1; ++kt) {
        for (int idx = threadIdx.x; idx < R * 8; idx += 256) {
            const int row = idx >> 3, c = idx & 7;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (row < mc) v = reinterpret_cast<const uint4*>(x + (u64)(slotl[soff + row] >> act_shift) * K + (u64)kt * 64)[c];
            *reinterpret_cast<uint4*>(dst + (((u64)z * KT + kt) * R + row) * M7_APITCH + c * 8) = v;
        }
    }
}
"""

# Instantiations (FUSE, D, DBG) of this file: D 1 (the design's) and D 2 (the board default, see the header); the measurement file adds D 4 and the DBG = 1 twins.
MOE7_INSTANCES = ((True, 1, 0), (False, 1, 0), (True, 2, 0), (False, 2, 0))


def moe7_instance_names(instances):
    return [f"w4_moe7_kernel<{'true' if f else 'false'},{d},{dbg}>" for f, d, dbg in instances] + ["m7_route_small_kernel", "m7_route_large_kernel", "m7_gather_kernel"]


def moe7_host_src(instances):
    """The load_inline host source for a list of instantiations: moe7_gemm (dispatch by (fuse, d, dbg)), moe7_route, moe7_init."""
    cases = "\n    ".join(f"M7_CASE({'true' if f else 'false'}, {d}, {dbg})" for f, d, dbg in instances)
    inits = "\n    ".join(f"m7_init_one((const void*)w4_moe7_kernel<{'true' if f else 'false'}, {d}, {dbg}>, \"w4_moe7_kernel<{'T' if f else 'F'},{d},{dbg}>\", optin, out);"
                          for f, d, dbg in instances)
    occ = "\n    ".join(f"M7_OCC_CASE({'true' if f else 'false'}, {d}, {dbg})" for f, d, dbg in instances)
    return _MOE7_HOST_TEMPLATE.replace("@CASES@", cases).replace("@INITS@", inits).replace("@OCC@", occ)


_MOE7_HOST_TEMPLATE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

static const int M7_SMEM_MAX = 232448;   // 227 KB = sharedMemPerBlockOptin of sm_90: the limit of STATIC + DYNAMIC shared memory of a block

// every CUDA call of the launch path is checked and names itself (a failed call that nobody reads poisons the next, unrelated torch call)
#define M7_CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { cudaGetLastError(); \
        TORCH_CHECK(false, "moe7: ", #call, " failed: ", cudaGetErrorName(e_), " (", cudaGetErrorString(e_), ")"); } } while (0)

// cudaFuncSetAttribute(MaxDynamicSharedMemorySize) must leave room for the static shared memory: dynamic <= optin - static (v5's 160 KB never met the limit; 227 KB
// dynamic + 480 B static is cudaErrorInvalidValue, and a launch with > 48 KB of dynamic smem and no attribute is the same error)
static void m7_init_one(const void* fn, const char* name, int optin, std::string& out) {
    cudaFuncAttributes a;
    M7_CK(cudaFuncGetAttributes(&a, fn));
    int dyn = optin - (int)a.sharedSizeBytes;
    if (dyn > M7_SMEM_MAX) dyn = M7_SMEM_MAX;
    M7_CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn));
    M7_CK(cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, (int)cudaSharedmemCarveoutMaxShared));
    cudaFuncAttributes b;
    M7_CK(cudaFuncGetAttributes(&b, fn));
    out += std::string(name) + " regs=" + std::to_string(b.numRegs) + " maxThreads=" + std::to_string(b.maxThreadsPerBlock) + " static=" + std::to_string(b.sharedSizeBytes)
         + " maxDynamic=" + std::to_string(b.maxDynamicSharedSizeBytes) + " local=" + std::to_string(b.localSizeBytes) + " optin=" + std::to_string(optin) + "\n";
}

template <bool FUSE, int D, int DBG>
static void m7_launch(torch::Tensor wb, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor sloc,
                      torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor ctr, torch::Tensor dbg, int64_t G,
                      int64_t NIT, int64_t K, int64_t KS, int64_t S, int64_t slot_bytes, int64_t R, int64_t out_stride, int64_t nctas,
                      int64_t dbg_mode, int64_t spin_max, bool pdl) {
    const size_t smem = 1024 + (size_t)S * (size_t)slot_bytes;
    TORCH_CHECK(smem <= (size_t)M7_SMEM_MAX && S >= 2 && S <= M7_MAXS && nctas >= 1 && nctas <= M7_MAXCTA, "moe7: bad ring/grid S=", S, " slot_bytes=", slot_bytes, " nctas=", nctas);
    auto fn = w4_moe7_kernel<FUSE, D, DBG>;
    const u8* a_wb = reinterpret_cast<const u8*>(wb.data_ptr());
    const float* a_gs = gs.data_ptr<float>();
    const u32* a_lut = reinterpret_cast<const u32*>(lut.data_ptr());
    const u16* a_act = reinterpret_cast<const u16*>(act.data_ptr());
    const int* a_ct = ctab.data_ptr<int>();
    const int* a_sl = slotl.data_ptr<int>();
    const int* a_loc = sloc.data_ptr<int>();
    float4* a_part = reinterpret_cast<float4*>(part.data_ptr());
    int* a_cnt = cnt.data_ptr<int>();
    u16* a_out = reinterpret_cast<u16*>(out.data_ptr());
    int* a_ctr = ctr.data_ptr<int>();
    u64* a_dbg = reinterpret_cast<u64*>(dbg.data_ptr());
    int i_G = (int)G, i_NIT = (int)NIT, i_K = (int)K, i_KS = (int)KS, i_S = (int)S, i_sb = (int)slot_bytes, i_R = (int)R, i_os = (int)out_stride,
        i_dm = (int)dbg_mode, i_sp = (int)spin_max;
    void* args[] = {&a_wb, &a_gs, &a_lut, &a_act, &a_ct, &a_sl, &a_loc, &a_part, &a_cnt, &a_out, &a_ctr, &a_dbg, &i_G, &i_NIT, &i_K, &i_KS, &i_S, &i_sb, &i_R, &i_os,
                    &i_dm, &i_sp};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)nctas);
    cfg.blockDim = dim3(M7_THREADS, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (pdl && at::cuda::getCurrentDeviceProperties()->major >= 9) {
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    const cudaError_t e = cudaLaunchKernelExC(&cfg, (const void*)fn, args);
    if (e != cudaSuccess) {
        cudaGetLastError();
        TORCH_CHECK(false, "moe7: cudaLaunchKernelExC failed: ", cudaGetErrorName(e), " (", cudaGetErrorString(e), ") fuse=", (int)FUSE, " D=", D, " dbg=", DBG, " grid=", nctas,
                    " block=", M7_THREADS, " dynamic_smem=", smem, " S=", S, " slot_bytes=", slot_bytes, " pdl_attrs=", cfg.numAttrs, " K=", K, " KS=", KS, " G=", G);
    }
}

void moe7_gemm(torch::Tensor wb, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor sloc,
               torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor ctr, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t K,
               int64_t KS, int64_t S, int64_t slot_bytes, int64_t R, int64_t out_stride, int64_t nctas, int64_t dbg_mode, int64_t spin_max,
               int64_t d, int64_t dbg_t, bool fuse, bool pdl) {
    c10::cuda::CUDAGuard guard(act.device());
#define M7_CASE(F, DD, X) if (fuse == F && d == DD && dbg_t == X) { \
        m7_launch<F, DD, X>(wb, gs, lut, act, ctab, slotl, sloc, part, cnt, out, ctr, dbg, G, NIT, K, KS, S, slot_bytes, R, out_stride, nctas, dbg_mode, spin_max, pdl); return; }
    @CASES@
#undef M7_CASE
    TORCH_CHECK(false, "moe7_gemm: no instantiation for fuse/d/dbg = ", fuse, "/", d, "/", dbg_t);
}

void moe7_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor sloc, torch::Tensor ctr, int64_t lo, int64_t nloc,
                bool small) {
    c10::cuda::CUDAGuard guard(ids.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (small)
        m7_route_small_kernel<<<1, 128, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctab.data_ptr<int>(), slotl.data_ptr<int>(), sloc.data_ptr<int>(),
                                                     ctr.data_ptr<int>(), (int)ids.numel(), (int)lo, (int)nloc);
    else
        m7_route_large_kernel<<<1, 1024, 0, stream>>>(ids.data_ptr<int>(), local_ids.data_ptr<int>(), ctab.data_ptr<int>(), slotl.data_ptr<int>(), sloc.data_ptr<int>(),
                                                      ctr.data_ptr<int>(), (int)ids.numel(), (int)lo, (int)nloc);
    const cudaError_t e = cudaGetLastError();
    TORCH_CHECK(e == cudaSuccess, "moe7: route kernel launch failed: ", cudaGetErrorName(e));
}

// GEMM1's pre-pass: the k-tiled gather of the chunks' activation rows (zb = the shape-only bound on the chunk count)
void moe7_gather(torch::Tensor x, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor dst, int64_t zb, int64_t K, int64_t act_shift, int64_t R, bool pdl) {
    c10::cuda::CUDAGuard guard(x.device());
    const u16* a_x = reinterpret_cast<const u16*>(x.data_ptr());
    const int* a_ct = ctab.data_ptr<int>();
    const int* a_sl = slotl.data_ptr<int>();
    u16* a_dst = reinterpret_cast<u16*>(dst.data_ptr());
    int i_K = (int)K, i_as = (int)act_shift, i_R = (int)R;
    void* args[] = {&a_x, &a_ct, &a_sl, &a_dst, &i_K, &i_as, &i_R};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)zb, (unsigned)((K / 64 + 7) / 8), 1);
    cfg.blockDim = dim3(256, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (pdl && at::cuda::getCurrentDeviceProperties()->major >= 9) {
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    const cudaError_t e = cudaLaunchKernelExC(&cfg, (const void*)m7_gather_kernel, args);
    TORCH_CHECK(e == cudaSuccess, "moe7: gather launch failed: ", cudaGetErrorName(e), " (", cudaGetErrorString(e), ") zb=", zb, " K=", K, " R=", R);
}

// the pre-flight of the Python side: how many CTAs of this instantiation fit an SM with `smem` bytes of dynamic shared memory (0 / -1 = the launch would fail)
int64_t moe7_occupancy(int64_t d, int64_t dbg_t, bool fuse, int64_t smem) {
#define M7_OCC_CASE(F, DD, X) if (fuse == F && d == DD && dbg_t == X) { int n = 0; \
        const cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void*)w4_moe7_kernel<F, DD, X>, M7_THREADS, (size_t)smem); \
        if (e != cudaSuccess) { cudaGetLastError(); return -1; } return n; }
    @OCC@
#undef M7_OCC_CASE
    return -2;
}

// sets the attributes of every instantiation; returns one diagnostic line per instantiation (registers, threads, static / maximum dynamic smem, the device's optin)
std::string moe7_init() {
    std::string out;
    int dev = 0;
    M7_CK(cudaGetDevice(&dev));
    int optin = 0;
    M7_CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    @INITS@
    return out;
}
"""
_MOE7_CPP_SRC = (
    "void moe7_gemm(torch::Tensor wb, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor sloc, "
    "torch::Tensor part, torch::Tensor cnt, torch::Tensor out, torch::Tensor ctr, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t K, int64_t KS, "
    "int64_t S, int64_t slot_bytes, int64_t R, int64_t out_stride, int64_t nctas, int64_t dbg_mode, int64_t spin_max, int64_t d, int64_t dbg_t, "
    "bool fuse, bool pdl);\n"
    "void moe7_route(torch::Tensor ids, torch::Tensor local_ids, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor sloc, torch::Tensor ctr, int64_t lo, "
    "int64_t nloc, bool small);\n"
    "void moe7_gather(torch::Tensor x, torch::Tensor ctab, torch::Tensor slotl, torch::Tensor dst, int64_t zb, int64_t K, int64_t act_shift, int64_t R, bool pdl);\n"
    "int64_t moe7_occupancy(int64_t d, int64_t dbg_t, bool fuse, int64_t smem);\n"
    "std::string moe7_init();\n")


def moe7_device_src():
    """The device source of the v7 extension: v5's device source (helpers, route kernels, m5_batch's body) + the MOE7 template."""
    return moe5_device_src() + _MOE7_EXTRA_SRC


# ---- per-M launch configuration: (d, s, ks) per GEMM = k64 iterations per ring stage, ring slots, K splits. The K split per bucket is v5's (MOE5_CFG), so the
# split-K sums are v5's and the output is bitwise v5's. D 2 instead of the design's D 1 / S 8 (see the header): half the bulk copies per byte.
M7_RING_LARGE = (2, 5)   # (D, S) at 17..256 tokens (32 activation rows): 5 x 35.5 KB = 178 KB of smem, 4 stages = 110 KB of weights in flight
M7_RING_SMALL = (2, 6)   # (D, S) at 1..16 tokens (8 / 16 activation rows): 6 x 29-31 KB = 175-188 KB
MOE7_CFG = (
    # (largest M, GEMM1 (d, s, ks), GEMM2 (d, s, ks)); ks = v5's per bucket
    (1, M7_RING_SMALL + (2,), M7_RING_SMALL + (2,)),
    (2, M7_RING_SMALL + (2,), M7_RING_SMALL + (4,)),
    (4, M7_RING_SMALL + (2,), M7_RING_SMALL + (4,)),
    (8, M7_RING_SMALL + (2,), M7_RING_SMALL + (2,)),
    (12, M7_RING_SMALL + (2,), M7_RING_SMALL + (2,)),
    (16, M7_RING_SMALL + (2,), M7_RING_SMALL + (2,)),
    (32, M7_RING_LARGE + (4,), M7_RING_LARGE + (4,)),
    (128, M7_RING_LARGE + (4,), M7_RING_LARGE + (4,)),
    (256, M7_RING_LARGE + (4,), M7_RING_LARGE + (4,)),
)
MOE7_ENABLE = {mx: True for mx, _c1, _c2 in MOE7_CFG}   # per bucket (largest M); False = v5's path for that bucket
MOE7_SELFCHECK_MS = (1, 2, 4, 8, 12, 16, 32, 33, 128, 256)
MOE7_MAX_FAILS = 2       # self-check sizes that may fail before the rest is skipped (see _moe7_selfcheck)
MOE7_TOL = 3e-3          # a bucket whose K split differs from v5's is gated on the relative L2 distance to v5 instead of the bit-identity
M7_TILE = 576            # bytes of one 16-row x k64 tile: 512 B of words + 64 B of E4M3 scales
M7_GPI = 24              # 16-row groups per item (12 consumer warps x 2)
M7_APITCH = 72           # activation row pitch in elements (64 + 8)
M7_COPIES_PER_STAGE = 2  # bulk copies per ring stage: the weight+scale run and the activation run (one elected lane issues both)
M7_MAXS = 16
M7_SMEM_MAX = 232448
M7_SPIN = 1 << 18        # bound of every spin loop in polls (262k polls = 8 ms .. 3 s depending on the poll's suspend time; a legitimate wait is a few hundred polls), see m7_wait_full
M7_STAMPS, M7_NOCOMPUTE, M7_CONSONLY, M7_COVER, M7_FAULT, M7_BOUND = 1, 2, 4, 8, 16, 32   # dbg_mode bits of the DBG = 1 twins (the measurement file / tests)
M7_CO_NOUNPACK, M7_CO_NOMMA, M7_CO_NOLDM, M7_CO_NOFIX, M7_CO_LDSONLY = 64, 128, 256, 512, 1024   # C31 consumer-chain twins (on top of CONSONLY): see measure_ws_v1.py
M7_MAXCTA, M7_REC, M7_COV_MAX = 256, 8, 8192
M7_COV_OFF = M5_OCC + M7_MAXCTA * 13 * M7_REC                   # u64 offset of the coverage arrays (u32 x 3 x M7_COV_MAX) in a DBG buffer
M7_DBG_U64 = M7_COV_OFF + 3 * M7_COV_MAX // 2                    # u64 words of a DBG buffer


def moe7_bucket(m):
    for mx, _c1, _c2 in MOE7_CFG:
        if m <= mx:
            return mx
    return None


def moe7_cfg(m):
    """(GEMM1 cfg, GEMM2 cfg) of the bucket of m, or None (m > 256, bucket disabled)."""
    b = moe7_bucket(m)
    if b is None or not MOE7_ENABLE.get(b, False):
        return None
    for mx, c1, c2 in MOE7_CFG:
        if mx == b:
            return c1, c2
    return None


def moe7_bitwise(m):
    """True if the K splits of m's bucket are v5's (then v7 must be bit-identical to v5; else the gate is the relative L2 bound)."""
    c7, c5 = moe7_cfg(m), moe5_cfg(m)
    return c7 is not None and c5 is not None and c7[0][2] == c5[0][2] and c7[1][2] == c5[1][2]


def moe7_slot_bytes(d, rows):
    """One ring slot = one stage: D k64 iterations of 24 groups (576 B tiles: 512 B words + 64 B E4M3 scales) = ONE contiguous run of the weight blob, then D k-tiles of
    the chunk's activations (`rows` rows x 144 B each) = ONE contiguous run of the k-tiled activation buffer; rounded up to 128 B."""
    return -(-(d * (M7_GPI * M7_TILE + rows * M7_APITCH * 2)) // 128) * 128


def moe7_smem(m, c):
    d, s, _ks = c
    return 1024 + s * moe7_slot_bytes(d, moe5_rows(m))


def moe7_check_cfg(m, k_dim, groups, c):
    """Geometry rules of the kernel (raise = caught by prepare's self-check): ring of 2..16 slots that fits 227 KB, K slice a multiple of 64 * D, enough stages
    per item that the 4-entry descriptor ring cannot be overwritten before it was read (4 * stages >= slots), 8 K splits at most."""
    d, s, ks = c
    assert d in (1, 2, 4) and 2 <= s <= M7_MAXS and 1 <= ks <= 8, c
    assert k_dim % ks == 0 and (k_dim // ks) % (64 * d) == 0 and (k_dim // ks) <= 2048, (k_dim, ks, d)
    nst = (k_dim // ks // 64) // d
    assert nst >= 1 and 4 * nst >= s, (nst, s)
    assert groups >= 1 and groups % 2 == 0, groups
    assert moe7_smem(m, c) <= M7_SMEM_MAX, (m, c)


_MOE7_EXT = {}


def moe7_ext(flags=(), instances=MOE7_INSTANCES):
    """The compiled v7 extension (moe7_gemm / moe7_route / moe7_init), built once per process, flag set and instantiation list."""
    key = (tuple(flags), tuple(instances))
    if key not in _MOE7_EXT:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        src = moe7_device_src() + moe7_host_src(instances)
        tag = hashlib.sha1((src + repr(key)).encode()).hexdigest()[:12]
        _MOE7_EXT[key] = load_inline(
            name=f"okc_moe7_{tag}", cpp_sources=_MOE7_CPP_SRC, cuda_sources=src,
            functions=["moe7_gemm", "moe7_route", "moe7_gather", "moe7_init", "moe7_occupancy"], extra_cuda_cflags=["-O3", *flags], verbose=False)
    return _MOE7_EXT[key]


def pack_w7_blob(words, scales):
    """The weight blob of the v7 kernel (prepare(), untimed, lossless: the same bits as v4's tile words and scales, regrouped). words int32 [E, G, NIT, 32, 4] (16 B per
    lane = 4 k16 tiles), scales uint8 [E, G, NIT, 8, 2, 4] -> uint8 [E, G * NIT * 576]. Per expert: group blocks of 24 groups (the last one holds G % 24 groups), inside a
    block k64 iteration it-major, then group, then the 576 B tile (512 B words + 64 B scales): one ring stage (D consecutive iterations of one block) is ONE contiguous
    D x nval x 576 B run. Block b of expert e starts at (e G NIT + 24 b NIT) 576; iteration it of a block of n groups at + it n 576."""
    e_n, g_n, nit = words.shape[:3]
    out = torch.empty((e_n, g_n * nit * M7_TILE), dtype=torch.uint8, device=words.device)
    full = (g_n // M7_GPI) * M7_GPI
    for e in range(e_n):
        t = torch.cat([words[e].contiguous().view(torch.uint8).reshape(g_n, nit, 512), scales[e].contiguous().reshape(g_n, nit, 64)], dim=2)     # [G, NIT, 576]
        parts = []
        if full:
            parts.append(t[:full].reshape(full // M7_GPI, M7_GPI, nit, M7_TILE).permute(0, 2, 1, 3).reshape(-1))
        if g_n > full:
            parts.append(t[full:].permute(1, 0, 2).reshape(-1))
        out[e] = torch.cat(parts) if len(parts) > 1 else parts[0]
    return out


M7_STATIC_SMEM = 512     # static shared memory of the kernel (480 B on sm_90) as the pre-flight assumes it when the extension reports none


def moe7_preflight(m7, d, dbg_t, fuse, smem):
    """None if a launch of the instantiation (fuse, d, dbg_t) with `smem` bytes of dynamic shared memory can run, else the reason (checked BEFORE any launch: a
    failed launch leaves an error that the next unrelated torch call reports). Needs: smem + static <= the device's sharedMemPerBlockOptin, and an occupancy of
    at least one CTA per SM from the CUDA occupancy calculator. Cached per key."""
    key = (d, dbg_t, bool(fuse), int(smem))
    cache = m7.setdefault("pf", {})
    if key in cache:
        return cache[key]
    why = None
    optin = m7.get("optin")
    if optin is not None and smem + m7.get("static", M7_STATIC_SMEM) > optin:
        why = f"dynamic smem {smem} + static {m7.get('static', M7_STATIC_SMEM)} > sharedMemPerBlockOptin {optin}"
    elif hasattr(m7["ext"], "moe7_occupancy"):
        occ = int(m7["ext"].moe7_occupancy(d, dbg_t, bool(fuse), int(smem)))
        if occ < 1:
            why = f"occupancy {occ} (cudaOccupancyMaxActiveBlocksPerMultiprocessor, dynamic smem {smem}) for fuse={int(bool(fuse))} D={d} dbg={dbg_t}"
    cache[key] = why
    return why


def moe7_check_launch(m7, m, c, fuse, dbg_t=0):
    """Geometry rules + the pre-flight for one GEMM launch; raises RuntimeError with the reason (callers: moe7_experts, the measurement's stages)."""
    why = moe7_preflight(m7, c[0], dbg_t, fuse, moe7_smem(m, c))
    if why is not None:
        raise RuntimeError(f"moe7 pre-flight: {why}")


def moe7_preflight_table(m7, cfg_table=None):
    """Pre-flight of every (bucket, GEMM) of the config table before the first launch. Returns ({bucket: reason}, lines to print)."""
    bad, lines = {}, []
    for mx, c1, c2 in (MOE7_CFG if cfg_table is None else cfg_table):
        if not MOE7_ENABLE.get(mx, True):
            continue
        for g, (c, fuse) in enumerate(((c1, True), (c2, False))):
            smem = moe7_smem(mx, c)
            why = moe7_preflight(m7, c[0], 0, fuse, smem)
            lines.append(f"okc-diag moe7 preflight m<={mx} g{g + 1} D{c[0]} S{c[1]} KS{c[2]} dynamic_smem={smem} " + ("ok" if why is None else f"FAIL: {why}"))
            if why is not None:
                bad[mx] = why
    return bad, lines


def prepare_moe7(sm, lm, ext=None, flags=(), nctas=None):
    """Untimed. The v7 extension on top of v4's packed weights (sm) and large-M scratch (lm): the item counters (zeroed by the route kernel in every call),
    a dummy dbg buffer (DBG = 0 kernels never touch it) and the persistent grid size = the SM count. Sets the function attributes of every instantiation
    (checked, with the device's sharedMemPerBlockOptin printed), then pre-flights every bucket's launch shape: a bucket that cannot launch is listed in
    m7["bad_buckets"] and never launched (a failed launch would poison the context for v5 and for the measurement)."""
    dev = sm["w1"].device
    props = torch.cuda.get_device_properties(dev)
    m7 = {"ext": ext if ext is not None else moe7_ext(flags), "dbg": torch.zeros(8, dtype=torch.int64, device=dev),
          "ctr": torch.zeros(2, dtype=torch.int32, device=dev),
          "nctas": int(nctas) if nctas is not None else int(props.multi_processor_count),
          "optin": int(getattr(props, "shared_memory_per_block_optin", 0)) or None}
    d_ = sm["dims"]
    m7["b1"], m7["b2"] = pack_w7_blob(sm["w1"], sm["s1"]), pack_w7_blob(sm["w2"], sm["s2"])
    m7["sloc"] = torch.zeros(8 * lm.get("max_m", 256), dtype=torch.int32, device=dev)
    assert d_["k2"] // 64 == d_["g1"] // 8, "the intermediate's k-tiles are the GEMM1 output groups / 8"
    info = m7["ext"].moe7_init()
    if isinstance(info, str) and info:
        print("okc-diag moe7 attributes (sharedMemPerBlockOptin=" + str(m7["optin"]) + "):\n" + info.rstrip("\n"))
        mt = re.search(r"static=(\d+)", info)
        if mt:
            m7["static"] = int(mt.group(1))
    m7["bad_buckets"], lines = moe7_preflight_table(m7)
    for ln in lines:
        print(ln)
    return m7


def moe7_tiled_elems(m, lm, k_dim):
    """Elements of a k-tiled activation buffer [chunk bound][K/64][R rows][72] for m tokens (shape-only)."""
    return moe5_zbound(m, lm) * (k_dim // 64) * moe5_rows(m) * M7_APITCH


def moe7_experts(x, topk_ids, sm, lm, m7, start, count, cfg, pdl=(True, True), dbg=None, dbg_mode=(0, 0), dbg_t=(0, 0), spin_max=None, nctas=None):
    """Route + gather + GEMM1 (SwiGLU fused) + GEMM2 for 1..256 tokens with the persistent warp-specialised kernel. cfg = (GEMM1 (d, s, ks), GEMM2 (...)). Returns
    (y [M, 8, H] BF16, local_ids [M, 8] int32). Shape-only launches; no host sync. The chunks' activations travel k-tiled ([chunk][K/64][R rows][72], so a ring stage's
    slice is one contiguous bulk copy): GEMM1's input is gathered by a pre-pass inside the call, GEMM1's epilogue writes GEMM2's input in the same layout. dbg (one
    buffer or one per GEMM) / dbg_mode / dbg_t / pdl are the measurement file's and the tests' knobs (nctas: the grid, default the SM count)."""
    m = x.shape[0]
    d, ext = sm["dims"], m7["ext"]
    spin_max = M7_SPIN if spin_max is None else spin_max
    nctas = m7["nctas"] if nctas is None else nctas
    assert m <= lm["max_m"] and count == SM_NEXP
    ids = topk_ids.to(torch.int32).contiguous()
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
    rows, zb = moe5_rows(m), moe5_zbound(m, lm)
    for i, (k_dim, groups, c, fuse) in enumerate(((d["k1"], d["g1"], cfg[0], True), (d["k2"], d["g2"], cfg[1], False))):   # checks first: nothing is launched on a bad config
        moe7_check_cfg(m, k_dim, groups, c)
        moe7_check_launch(m7, m, c, fuse, dbg_t[i])
    ext.moe7_route(ids, local_ids, lm["ctab"], lm["slotl"], m7["sloc"], m7["ctr"], start, count, m <= SM_MAX_M)
    xt = torch.empty(moe7_tiled_elems(m, lm, d["k1"]), dtype=torch.bfloat16, device=x.device)
    inter = torch.empty(moe7_tiled_elems(m, lm, d["k2"]), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
    ext.moe7_gather(x.contiguous(), lm["ctab"], lm["slotl"], xt, zb, d["k1"], 3, rows, bool(pdl[0]))
    dbg = m7["dbg"] if dbg is None else dbg
    dbgs = tuple(dbg) if isinstance(dbg, (tuple, list)) else (dbg, dbg)
    for i, (k_dim, groups, c, b, gsc, src, dst, fuse) in enumerate((
            (d["k1"], d["g1"], cfg[0], m7["b1"], sm["gs1"], xt, inter, True),
            (d["k2"], d["g2"], cfg[1], m7["b2"], sm["gs2"], inter, y, False))):
        dd, s, ks = c
        ext.moe7_gemm(b, gsc, sm["lut"], src, lm["ctab"], lm["slotl"], m7["sloc"], lm["part"], lm["cnt"][i], dst, m7["ctr"][i:], dbgs[i], groups, k_dim // 64, k_dim,
                      ks, s, moe7_slot_bytes(dd, rows), rows, 0 if fuse else d["hid"], nctas, dbg_mode[i], spin_max, dd, dbg_t[i], fuse, bool(pdl[i]))
    return y.view(m, TOP_K, d["hid"]), local_ids


def _kernel_moe7(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m = x.shape[0]
    y, local_ids = moe7_experts(x, data["topk_ids"], state["smallm"], state["largem"], state["moe7"], cfg["local_start"], cfg["local_count"], moe7_cfg(m))
    return _epilogue_for(y, data["topk_weights"].to(torch.float32).contiguous(), local_ids, state, m)


def _use_moe7(data, state):
    cfg = data["config"]
    m = data["hidden_states"].shape[0]
    return (state.get("moe7") is not None and state.get("largem") is not None and moe7_cfg(m) is not None
            and state.get("moe7_ok", {}).get(moe7_bucket(m), False) and cfg["local_start"] == 0 and cfg["local_count"] == SM_NEXP)


def _v5_path(data, state):
    """v5's path for these inputs (the bitwise reference and the fallback = the board kernel), else v4's."""
    if _use_moe5(data, state):
        return _kernel_moe5(data, state)
    return _v4_path(data, state)


def _moe7_selfcheck(state):
    """{bucket: True if v7's output is bit-identical to v5's (v4's) on seeded inputs of every M of the bucket -- or, for a bucket whose K split is not v5's, within
    MOE7_TOL relative L2 of it}. After MOE7_MAX_FAILS failing sizes the remaining sizes are not run (a broken ring protocol makes every launch run into the spin bound
    of M7_SPIN polls: prepare()'s budget must survive it) and count as failed."""
    res = {}
    fails = 0
    for m in MOE7_SELFCHECK_MS:
        b = moe7_bucket(m)
        if b in (state.get("moe7") or {}).get("bad_buckets", {}):
            res[b] = False
            continue
        if fails >= MOE7_MAX_FAILS:
            res[b] = False
            continue
        try:
            data = _selfcheck_inputs(m, 5000 + m, state["norm_w"].device)
            ref = _v5_path(data, state)
            if ref is None:
                res[b] = False
                fails += 1
                print(f"okc-diag moe7 selfcheck m={m}: v5/v4 path unavailable")
                continue
            got = _kernel_moe7(data, state)
            torch.cuda.synchronize()
            if moe7_bitwise(m):
                ok = bool(torch.equal(ref.view(torch.int16), got.view(torch.int16)))
            else:
                err = float((got.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12))
                ok = err == err and err < MOE7_TOL
            fails += 0 if ok else 1
            res[b] = res.get(b, True) and ok
        except Exception as ex:  # noqa: BLE001
            print(f"okc-diag moe7 selfcheck m={m} raised {type(ex).__name__}: {str(ex)[:160]}")
            res[b] = False
            fails += 1
    return res


def _prepare_moe7_state(state, flags, ext=None):
    state["moe7"], state["moe7_ok"] = None, {}
    if state.get("smallm") is None or state.get("largem") is None:
        print("okc-diag moe7 unavailable: v4's small-M / large-M state missing")
        return
    t0 = time.time()
    try:
        state["moe7"] = prepare_moe7(state["smallm"], state["largem"], ext=ext, flags=flags)
        print(f"okc-diag moe7 extension built in {time.time() - t0:.0f}s ({len(MOE7_INSTANCES)} instantiations, {state['moe7'].get('nctas')} CTAs)")
    except Exception as ex:  # noqa: BLE001  (build error: v5's paths serve every M)
        print(f"okc-diag moe7 unavailable: {type(ex).__name__}: {str(ex)[:300]}")
        state["moe7"] = None
        return
    oks = _moe7_selfcheck(state) if SELFCHECK else {mx: True for mx, _a, _b in MOE7_CFG}
    for b, why in (state["moe7"].get("bad_buckets") or {}).items():
        print(f"okc-diag moe7 bucket m<={b} inactive (v5 serves it): {why}")
    print("okc-diag moe7 vs v5 per bucket: " + " ".join(f"m{b}={'OK' if ok else 'DIFF'}" for b, ok in oks.items()) + f" ({time.time() - t0:.0f}s with the build)")
    for b in state["moe7"].get("bad_buckets") or {}:
        oks[b] = False
    state["moe7_ok"] = oks
    if not any(oks.values()):
        state["moe7"] = None
    print(f"okc-diag moe7 active={state['moe7_ok']}")
# ============================================================================================ MOE7 END


# ============================================================================================ MEASUREMENT (prepare() only; nothing here feeds kernel_fn)
import time

BUDGET_S = 200.0        # measurement wall budget inside prepare's 300 s (v4's / v5's builds and the repack come first)
PREPARE_LIMIT_S = 280.0
RESERVE_S = 8.0
REPS = 5                # graph replays per timing (median), each after a read-only L2 flush
FLUSH_MB = 128
CAL_MB = 64
MS = (256, 128, 32, 16, 8, 1)     # processing order: the decision shapes first
SEEDS = (11, 12, 13)
SUMMARY_BUDGET = 14800
# Ring shapes of this check (D, S): the dev-card tests patch them down to what 64 KB of smem holds.
WS_D1 = (1, 8)          # the design: 8 x 18 KB (32 rows), 97 KB of weights in flight, 80 bulk copies per stage
WS_D2 = (2, 5)          # half the copies per byte: 5 x 35.5 KB, 110 KB of weights in flight
DBG_MAX_CTA = M7_MAXCTA
WS_DBG_U64 = M5_OCC + 262144    # the WSTREAM twin writes one word per warp of up to 64 x 8 x 96 CTAs x 4 warps; CAL one per warp of 4 x SMs CTAs


def meas_instances():
    """The 8 template instantiations (FUSE, D, DBG) of this check (tests/test_measure_ws_v1.py compares): D 1 / D 2 as shipped and their DBG = 1 twins."""
    out = []
    for f in (True, False):
        out += [(f, 1, 0), (f, 2, 0), (f, 1, 1), (f, 2, 1)]
    return tuple(out)


MEAS_INSTANCES = meas_instances()

_MEAS_EXTRA_SRC = r"""
// WSTREAM twin: the weight bytes of the GEMM's chunks (words + scales, same addresses, same per-warp streams as v5's GEMM), no dequant / mma / staging / fixup;
// 8 k64 iterations (16 loads) in flight per lane, 4 CTAs per SM asked; the fold is stored by lane 0. The ceiling of THIS layout (v5's geometry 4 warps x 2 groups).
template <int NW, int MR>
__global__ void __launch_bounds__(NW * 32, 4)
meas_wstream_kernel(const uint4* __restrict__ Wq, const uint2* __restrict__ Sq, const int* __restrict__ ctab, u64* __restrict__ dbg, int G, int NIT, int KS) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2;
    const int nb = blockIdx.x, ks = blockIdx.y;
    const int nit = NIT / KS;
    const int n_chunks = ctab[0];
    const int g0 = (nb * NW + warp) * MR;
    u32 sink = 0;
    for (int z = blockIdx.z; z < n_chunks; z += gridDim.z) {
        const int e = ctab[4 + M5_CTW * z];
#pragma unroll
        for (int r = 0; r < MR; ++r) {
            const uint4* wp = Wq + (((u64)e * G + g0 + r) * NIT + (u64)ks * nit) * 32 + lane;
            const uint2* sp = Sq + (((u64)e * G + g0 + r) * NIT + (u64)ks * nit) * 8 + gid;
            for (int it = 0; it < nit; it += 8) {
                uint4 w[8];
                uint2 s[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    ldg_stream4(w[u], wp + (it + u) * 32);
                    ldg_stream2(s[u], sp + (it + u) * 8);
                }
#pragma unroll
                for (int u = 0; u < 8; ++u) sink ^= w[u].x ^ w[u].y ^ w[u].z ^ w[u].w ^ s[u].x ^ s[u].y;
            }
        }
    }
    if (lane == 0) dbg[M5_OCC + (u64)(blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) * NW + warp] = (u64)sink;
}

// CAL: achievable read bandwidth, 64 MB of uint4 streamed with 8 loads in flight per lane, 4 CTAs per SM, folded and stored (one word per warp).
__global__ void __launch_bounds__(256, 4) meas_cal_kernel(const uint4* __restrict__ buf, u64* __restrict__ dbg, int n16) {
    u32 sink = 0;
    const int stride = gridDim.x * 256;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n16; i += stride * 8) {
        uint4 v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int j = i + u * stride;
            v[u] = make_uint4(0u, 0u, 0u, 0u);
            if (j < n16) ldg_stream4(v[u], buf + j);
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) sink ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    if ((threadIdx.x & 31) == 0) dbg[M5_OCC + (u64)blockIdx.x * 8 + (threadIdx.x >> 5)] = (u64)sink;
}
"""

_MEAS_HOST_SRC = r"""
void meas_wstream(torch::Tensor wq, torch::Tensor sq, torch::Tensor ctab, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t KS, int64_t nz) {
    c10::cuda::CUDAGuard guard(wq.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid((unsigned)(G / 8), (unsigned)KS, (unsigned)nz);
    meas_wstream_kernel<4, 2><<<grid, 128, 0, stream>>>(reinterpret_cast<const uint4*>(wq.data_ptr()), reinterpret_cast<const uint2*>(sq.data_ptr()),
                                                        ctab.data_ptr<int>(), reinterpret_cast<u64*>(dbg.data_ptr()), (int)G, (int)NIT, (int)KS);
}

void meas_cal(torch::Tensor buf, torch::Tensor dbg, int64_t grid) {
    c10::cuda::CUDAGuard guard(buf.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    meas_cal_kernel<<<(unsigned)grid, 256, 0, stream>>>(reinterpret_cast<const uint4*>(buf.data_ptr()), reinterpret_cast<u64*>(dbg.data_ptr()),
                                                        (int)(buf.numel() * buf.element_size() / 16));
}
"""
_MEAS_CPP_SRC = _MOE7_CPP_SRC + (
    "void meas_wstream(torch::Tensor wq, torch::Tensor sq, torch::Tensor ctab, torch::Tensor dbg, int64_t G, int64_t NIT, int64_t KS, int64_t nz);\n"
    "void meas_cal(torch::Tensor buf, torch::Tensor dbg, int64_t grid);\n")


def meas_names(instances):
    return moe7_instance_names(instances) + ["meas_wstream_kernel<4,2>", "meas_cal_kernel"]


MEAS_NAMES = meas_names(MEAS_INSTANCES)


def meas_device_src():
    return moe7_device_src() + _MEAS_EXTRA_SRC


_MEAS_EXT = {}


def meas_ext(flags=(), instances=None):
    """The measurement extension: the 8 v7 instantiations + the WSTREAM / CAL kernels in ONE nvcc build, `-Xptxas -v` and a verbose build log (regs / spills per
    instantiation go to the log before the summary). Returns (extension, build seconds)."""
    instances = MEAS_INSTANCES if instances is None else tuple(instances)
    key = (tuple(flags), instances)
    if key not in _MEAS_EXT:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        t0 = time.time()
        src = meas_device_src() + moe7_host_src(instances) + _MEAS_HOST_SRC
        tag = hashlib.sha1((src + repr(key)).encode()).hexdigest()[:12]
        ext = load_inline(
            name=f"okc_measws_{tag}", cpp_sources=_MEAS_CPP_SRC, cuda_sources=src,
            functions=["moe7_gemm", "moe7_route", "moe7_gather", "moe7_init", "moe7_occupancy", "meas_wstream", "meas_cal"],
            extra_cuda_cflags=["-O3", "-Xptxas", "-v", "-diag-suppress", "550,128", *flags], verbose=True)   # the two warnings of v5's source ate the 16 KB tail
        _MEAS_EXT[key] = (ext, time.time() - t0)
    return _MEAS_EXT[key]


def _gen_input(bs, seed, dist, device):
    """reference.generate_input(bs, "ep8", seed, dist=dist, rank="busiest") re-written here (a submission file cannot import the reference): the same
    random calls in the same order, so the routing, activations and gates are bit-identical (tests/test_measure_ws_v1.py compares them)."""
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


# ---- variants: (name, GEMM1 cfg, GEMM2 cfg, dbg modes, dbg template flags, pdl) over sonnet_v7's (d, s, ks) tuples ----
def variants(m):
    """[(name, c1, c2, dbg_mode, dbg_t, pdl)] in priority order. ks = v5's K split of the bucket unless the name says ks<N> (then NOT bitwise v5). d1s8 = WS D1S8
    (the design), d2s5 = WS D2S5; nocomp / consonly = the twins of d1s8, nocomp2 / consonly2 = of d2s5; stamps / stamps2 feed the S lines."""
    c5 = moe5_cfg(m)
    k1, k2 = c5[0][2], c5[1][2]

    def cfg(ring, a=k1, b=k2):
        return (ring[0], ring[1], a), (ring[0], ring[1], b)

    d1, d2 = cfg(WS_D1), cfg(WS_D2)
    z = (0, 0)
    vs = [("d1s8", *d1, z, z, (True, True)), ("d2s5", *d2, z, z, (True, True)),
          ("nocomp", *d1, (M7_NOCOMPUTE, M7_NOCOMPUTE), (1, 1), (True, True)), ("consonly", *d1, (M7_CONSONLY, M7_CONSONLY), (1, 1), (True, True)),
          ("nocomp2", *d2, (M7_NOCOMPUTE, M7_NOCOMPUTE), (1, 1), (True, True)), ("consonly2", *d2, (M7_CONSONLY, M7_CONSONLY), (1, 1), (True, True))]
    if m in CO_MS:   # C31: the consumer-chain twins, all on top of CONSONLY at D2 S5 (the producer copies from one L2-hot region: out of the picture)
        for name, bit in CO_TWINS:
            vs.append((name, *d2, (M7_CONSONLY | bit,) * 2, (1, 1), (True, True)))
    if m in KS_MS:
        for ks in (2, 4, 8):
            if ks != k1:
                vs.append((f"ks{ks}", *cfg(WS_D1, ks, ks), z, z, (True, True)))
    vs.append(("pdloff", *d1, z, z, (True, False)))
    return vs


KS_MS = (256, 8)          # the K-split sweep (KS 2 / 4 / 8) runs at bs 8 and 256 (fable-10 D3; bs 1 dropped in C31 for the output budget)
# C31: the consumer-chain twins (bs in CO_MS), each = consonly2 (D2 S5, producer from an L2-hot region) with ONE part of the consumer chain removed, timing only:
CO_MS = (256, 128, 16, 1)
CO_TWINS = (("co_nounpack", M7_CO_NOUNPACK),              # no unpack_word / s_lut / mul_bf16x2: raw lds words are the A fragment            -> cost (b), the INT / ALU dequant
            ("co_nomma", M7_CO_NOMMA),                    # unpack + lut + mul stay, the mma is an XOR fold of A and B                        -> cost (c), the tensor-core rate
            ("co_noldm", M7_CO_NOLDM),                    # B ldmatrix-ed once per ring stage instead of once per k16 pair                    -> the ldmatrix share of (a)
            ("co_nofix", M7_CO_NOFIX),                    # no split-K fixup / epilogue (garbage output)                                      -> cost (d)
            ("co_ldsonly", M7_CO_LDSONLY),                # ring lds only (words + scales into the sink): no ldmatrix, unpack or mma          -> the lds floor of (a)
            ("co_smem", M7_NOCOMPUTE))                    # lds + ldmatrix + fold, no unpack / mma (the existing NOCOMPUTE on the consumer)  -> the whole smem floor of (a)
CO_NAMES = tuple(n for n, _b in CO_TWINS)
CO_RULES = (35.0, 35.0, 60.0, 20.0, 20.0)       # verdict thresholds: NOMMA saves >= 35 %, NOUNPACK saves >= 35 %, LDSONLY / smem floor >= 60 % of consonly2, NOFIX saves >= 20 %, NOLDM saves >= 20 %
PASS1 = ("d1s8", "d2s5", "nocomp", "consonly", "consonly2") + CO_NAMES   # measured for every M first; then wstream + stamps; then the rest


def expert_bytes(d):
    """(GEMM1, GEMM2) weight bytes of one expert: rows x K x (0.5 + 1/16) B (E2M1 + one E4M3 scale per 16)."""
    return 16 * d["g1"] * d["k1"] * 0.5625, 16 * d["g2"] * d["k2"] * 0.5625


# ---- the pipelines, stage by stage (closures; the buffers are allocated in the builder so the closures capture into a graph) ----
def _v5_stages(data, state):
    """The board's kernels (the pivot): v5's route, g1, g2 at MOE5_CFG through v5's extension. Returns (stages, st); stages carry 'pair' (g1 then g2)."""
    sm, lm, m5 = state["smallm"], state["largem"], state["moe5"]
    d, ext = sm["dims"], m5["ext"]
    x = data["hidden_states"].contiguous()
    m = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    c1, c2 = moe5_cfg(m)
    st = {"local": torch.empty(ids.shape, dtype=torch.int32, device=x.device),
          "inter": torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device),
          "y": torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)}
    a1 = moe5_gemm_args(m, d["k1"], d["g1"] // (c1[0] * c1[1]), c1[2], c1[3], c1[4], lm)
    a2 = moe5_gemm_args(m, d["k2"], d["g2"] // (c2[0] * c2[1]), c2[2], c2[3], c2[4], lm)

    def f_route():
        ext.moe5_route(ids, st["local"], lm["ctab"], lm["slotl"], 0, SM_NEXP, m <= SM_MAX_M)

    def f_g1():
        ext.moe5_gemm(sm["w1"], sm["s1"], sm["gs1"], sm["lut"], x, lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][0], st["inter"], m5["dbg"],
                      d["g1"], d["k1"] // 64, d["k1"], a1["ks"], a1["nz"], a1["nb"], a1["rows"], d["inter"], 3, c1[0], c1[1], c1[5], 0, True, True)

    def f_g2():
        ext.moe5_gemm(sm["w2"], sm["s2"], sm["gs2"], sm["lut"], st["inter"], lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][1], st["y"], m5["dbg"],
                      d["g2"], d["k2"] // 64, d["k2"], a2["ks"], a2["nz"], a2["nb"], a2["rows"], d["hid"], 0, c2[0], c2[1], c2[5], 0, False, True)

    def f_pair():
        f_g1()
        f_g2()

    return [("route", f_route), ("g1", f_g1), ("g2", f_g2), ("pair", f_pair)], st


def _v7_stages(data, state, c1, c2, dbg_mode=(0, 0), dbg_t=(0, 0), pdl=(True, True), dbgs=None):
    """The v7 pipeline cut into stages: route (zeroes the item counters, writes the slot -> (chunk, row) map), gather (the k-tiled activation pre-pass of GEMM1), g1, g2
    with per-GEMM (d, s, ks), twin flags, PDL. 'g1z' / 'g2z' are replayed in their own graphs, so they carry a counter reset (a 2-int memset node) in front, and g1z
    includes the gather (the pre-pass is part of GEMM1's cost); 'pair' = reset, gather, g1, g2 (the real thing minus route); the reset's own graph cost is printed
    in the header (zero=) for subtraction, the gather alone as gat= on the I line."""
    sm, lm, m7 = state["smallm"], state["largem"], state["moe7"]
    d, ext = sm["dims"], m7["ext"]
    x = data["hidden_states"].contiguous()
    m = x.shape[0]
    ids = data["topk_ids"].to(torch.int32).contiguous()
    dbgs = (m7["dbg"], m7["dbg"]) if dbgs is None else dbgs
    rows, zb = moe5_rows(m), moe5_zbound(m, lm)
    st = {"local": torch.empty(ids.shape, dtype=torch.int32, device=x.device),
          "xt": torch.empty(moe7_tiled_elems(m, lm, d["k1"]), dtype=torch.bfloat16, device=x.device),
          "inter": torch.empty(moe7_tiled_elems(m, lm, d["k2"]), dtype=torch.bfloat16, device=x.device),
          "y": torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)}
    for g, (c, groups, k, fuse) in enumerate(((c1, d["g1"], d["k1"], True), (c2, d["g2"], d["k2"], False))):
        moe7_check_cfg(m, k, groups, c)
        moe7_check_launch(m7, m, c, fuse, dbg_t[g])      # C30b: attributes / smem / occupancy pre-flight BEFORE any launch (a failed launch poisons the context)
    ctr = m7["ctr"]

    def f_route():
        ext.moe7_route(ids, st["local"], lm["ctab"], lm["slotl"], m7["sloc"], ctr, 0, SM_NEXP, m <= SM_MAX_M)

    def f_gather():
        ext.moe7_gather(x, lm["ctab"], lm["slotl"], st["xt"], zb, d["k1"], 3, rows, True)

    def f_g1():
        dd, s, ks = c1
        ext.moe7_gemm(m7["b1"], sm["gs1"], sm["lut"], st["xt"], lm["ctab"], lm["slotl"], m7["sloc"], lm["part"], lm["cnt"][0], st["inter"], ctr[0:], dbgs[0],
                      d["g1"], d["k1"] // 64, d["k1"], ks, s, moe7_slot_bytes(dd, rows), rows, 0, m7["nctas"], dbg_mode[0], M7_SPIN, dd, dbg_t[0], True, bool(pdl[0]))

    def f_g2():
        dd, s, ks = c2
        ext.moe7_gemm(m7["b2"], sm["gs2"], sm["lut"], st["inter"], lm["ctab"], lm["slotl"], m7["sloc"], lm["part"], lm["cnt"][1], st["y"], ctr[1:], dbgs[1],
                      d["g2"], d["k2"] // 64, d["k2"], ks, s, moe7_slot_bytes(dd, rows), rows, d["hid"], m7["nctas"], dbg_mode[1], M7_SPIN, dd, dbg_t[1], False, bool(pdl[1]))

    def zero():
        ctr.zero_()

    def f_g1z():
        zero()
        f_gather()
        f_g1()

    def f_g2z():
        zero()
        f_g2()

    def f_pair():
        zero()
        f_gather()
        f_g1()
        f_g2()

    return [("route", f_route), ("gather", f_gather), ("g1", f_g1), ("g2", f_g2), ("g1z", f_g1z), ("g2z", f_g2z), ("pair", f_pair)], st


def _ws_stages(data, state, c1, c2):
    """The WSTREAM twin of both GEMMs (the chunk table must be filled by a route stage first); v5's K split / z cap of the bucket, geometry (4 x 2)."""
    sm, lm, m7 = state["smallm"], state["largem"], state["moe7"]
    d, ext = sm["dims"], m7["ext"]
    m = data["hidden_states"].shape[0]
    zb = moe5_zbound(m, lm)
    dbg = m7["dbg_ws"]
    k1, k2 = c1[2], c2[2]
    nz1, nz2 = (min(zb, moe5_cfg(m)[0][3]), min(zb, moe5_cfg(m)[1][3]))

    def f_g1():
        ext.meas_wstream(sm["w1"], sm["s1"], lm["ctab"], dbg, d["g1"], d["k1"] // 64, k1, nz1)

    def f_g2():
        ext.meas_wstream(sm["w2"], sm["s2"], lm["ctab"], dbg, d["g2"], d["k2"] // 64, k2, nz2)

    return [("g1", f_g1), ("g2", f_g2)]


class _RealEnv:
    """The GPU side: seeded inputs of the generate_input recipe, graph timing after a read-only flush (like the judge), the dbg buffers, the stamps."""

    def __init__(self, state):
        self.state = state
        self.dev = state["norm_w"].device

    def available(self):
        return (HAVE_SGLANG and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing()
                and self.state.get("largem") is not None and self.state.get("moe5") is not None and self.state.get("moe7") is not None)

    def clock(self):
        return time.time()

    def banner(self):
        return f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} cap={torch.cuda.get_device_capability()}"

    def setup(self):
        self.flush = torch.ones(FLUSH_MB * 1024 * 1024 // 4, device=self.dev, dtype=torch.float32)
        self.sink = torch.zeros((), device=self.dev, dtype=torch.float32)
        self.cal = torch.ones(CAL_MB * 1024 * 1024 // 4, device=self.dev, dtype=torch.int32)
        self.nsm = torch.cuda.get_device_properties(self.dev).multi_processor_count
        self.dbg = torch.zeros(M7_DBG_U64, dtype=torch.int64, device=self.dev)
        self.dbg2 = torch.zeros(M7_DBG_U64, dtype=torch.int64, device=self.dev)
        self.dbg_ws = torch.zeros(WS_DBG_U64, dtype=torch.int64, device=self.dev)
        self.state["moe7"]["dbg"] = self.dbg
        self.state["moe7"]["dbg_ws"] = self.dbg_ws

    def release(self):
        for a in ("flush", "sink", "cal", "dbg", "dbg2", "dbg_ws"):
            if hasattr(self, a):
                delattr(self, a)
        torch.cuda.empty_cache()

    def prep(self, m, seed):
        data = _gen_input(m, seed, "normal", self.dev)
        ids = data["topk_ids"]
        local = (ids >= 0) & (ids < 32)
        cnt = torch.bincount(ids[local].long(), minlength=32)
        info = dict(pairs=int(local.sum()), distinct=int((cnt > 0).sum()), chunks=int(((cnt + 31) // 32).sum()), n_act=int((cnt > 0).sum()))
        fns, st = _v5_stages(data, self.state)
        for _, f in fns:
            f()
        torch.cuda.synchronize()
        live = st["local"].reshape(-1) >= 0
        y0 = st["y"].view(torch.int16)[live].clone()
        return dict(m=m, seed=seed, data=data, y0=y0, live=live, norm=float(st["y"].float()[live].norm()), info=info)

    def compare(self, ctx, y):
        """'OK' if y's live rows equal the pivot's bit for bit, else the relative L2 error."""
        got = y.view(torch.int16)[ctx["live"]]
        if torch.equal(got, ctx["y0"]):
            return "OK"
        n = float((y.float()[ctx["live"]] - ctx["y0"].view(torch.bfloat16).float()).norm())
        return f"e={n / max(ctx['norm'], 1e-12):.1e}" if n == n else "e=nan"

    def graph_time(self, fn, reps=REPS):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
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
        del g
        return ts[len(ts) // 2]

    def run_eager(self, fns):
        for _, f in fns:
            f()
        torch.cuda.synchronize()

    def calibrate(self):
        ext = self.state["moe7"]["ext"]
        grid = 4 * self.nsm
        us = self.graph_time(lambda: ext.meas_cal(self.cal, self.dbg_ws, grid))
        zero_us = self.graph_time(lambda: self.state["moe7"]["ctr"].zero_())
        return CAL_MB * 1024 * 1024 / us / 1e6, us, zero_us

    def stamps(self, ctx, c1, c2):
        """One eager, flushed run of the pipeline with STAMPS on both GEMMs (separate dbg buffers; the DBG = 1 template of the ring shape); returns the two stamp tables
        [ncta, 13, 8] (warp 0 = the producer) + the error flags."""
        for b in (self.dbg, self.dbg2):
            b.zero_()
        fns, _st = _v7_stages(ctx["data"], self.state, c1, c2, dbg_mode=(M7_STAMPS, M7_STAMPS), dbg_t=(1, 1), dbgs=(self.dbg, self.dbg2))
        stages = [f for f in fns if f[0] in ("route", "gather", "g1", "g2")]
        self.run_eager(stages)
        torch.sum(self.flush, dim=0, out=self.sink)
        for b in (self.dbg, self.dbg2):
            b.zero_()
        self.run_eager(stages)
        n = self.state["moe7"]["nctas"]
        return [(b[M5_OCC:M5_OCC + M7_MAXCTA * 13 * M7_REC].view(M7_MAXCTA, 13, M7_REC)[:n].clone(), int(b.view(torch.int32)[2])) for b in (self.dbg, self.dbg2)]


def _stamp_line(rec, err):
    """Per-CTA shares of the STAMPS twin. rec[cta, warp, 8] = (t_begin, t_end, wait, fix+epi, items, stages, bytes, smid); warp 0 = the producer. cons_wait_full = the
    consumers' time in try_wait(full) / their lifetime (all 12 warps pooled), prod_wait_empty = the producer's time in try_wait(empty) / its lifetime, fix_epi = split-K
    fixup + epilogue / consumer lifetime, items/cta = the producer's items per CTA (median), bw/sm = the producer's issued bytes / its lifetime (median over CTAs with
    items), span = last end - first begin, tail = last end - median end."""
    t = rec.to(torch.float64)
    act = t[:, 0, 0] > 0
    if not bool(act.any()):
        return "no stamps"
    t = t[act]
    life_c = (t[:, 1:, 1] - t[:, 1:, 0]).clamp(min=1.0)
    life_p = (t[:, 0, 1] - t[:, 0, 0]).clamp(min=1.0)
    cwf = 100.0 * float(t[:, 1:, 2].sum()) / float(life_c.sum())
    fix = 100.0 * float(t[:, 1:, 3].sum()) / float(life_c.sum())
    pwe = 100.0 * float(t[:, 0, 2].sum()) / float(life_p.sum())
    has = t[:, 0, 4] > 0
    bw = float((t[has, 0, 6] / life_p[has]).median()) if bool(has.any()) else 0.0
    ends = t[:, 0, 1]
    span = (float(ends.max()) - float(t[:, 0, 0].min())) / 1000.0
    tail = (float(ends.max()) - float(ends.median())) / 1000.0
    return (f"items/cta={float(t[:, 0, 4].median()):.0f} cons_wait_full={cwf:.0f}% prod_wait_empty={pwe:.0f}% fix_epi={fix:.0f}% bw/sm={bw:.1f}GB/s "
            f"span={span:.0f}us tail={tail:.1f}us cps={M7_COPIES_PER_STAGE}" + (" ERR" if err else ""))


def _med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def _fmt(xs, f="{:.1f}"):
    return "/".join(f.format(x) if x is not None else "-" for x in xs)


def _short(e):
    return f"{type(e).__name__}: {str(e)[:80]}".replace("\n", " ")


def _stamp_shares(rec):
    """(cons_wait_full %, prod_wait_empty %) of a stamp table, or None."""
    t = rec.to(torch.float64)
    t = t[t[:, 0, 0] > 0]
    if t.shape[0] == 0:
        return None
    life_c = (t[:, 1:, 1] - t[:, 1:, 0]).clamp(min=1.0)
    life_p = (t[:, 0, 1] - t[:, 0, 0]).clamp(min=1.0)
    return 100.0 * float(t[:, 1:, 2].sum()) / float(life_c.sum()), 100.0 * float(t[:, 0, 2].sum()) / float(life_p.sum())


def _decide(best_g1, best_name, nocomp_ts, consonly_ts, cwf, pwe, all_ok):
    """fable-10 D3's decision table on GEMM1 bs256 (medians over the seeds): best_g1 = the best real WS variant's GEMM1 us, nocomp_ts / consonly_ts = the twins'
    streamed TB/s (chunks x expert bytes / time, the design's basis), cwf / pwe = the stamp shares (%), all_ok = every bucket-identical variant printed bw=OK."""
    if best_g1 is None:
        return "D no WS result for bs256"
    head = f"D (fable-10 D3) best={best_name} g1={best_g1:.1f}us nocomp={nocomp_ts if nocomp_ts is None else round(nocomp_ts, 2)}TB/s consonly={consonly_ts if consonly_ts is None else round(consonly_ts, 2)}TB/s"
    head += f" cons_wait_full={'-' if cwf is None else f'{cwf:.0f}%'} prod_wait_empty={'-' if pwe is None else f'{pwe:.0f}%'} bitwise={'OK' if all_ok else 'DIFF'}"
    if not all_ok:
        return head + " -> (v) a bucket DIFFers: bug, that bucket stays v5; fix before promote"
    if nocomp_ts is not None and nocomp_ts < 3.2:
        return head + " -> (iv) NOCOMPUTE < 3.2: the bulk-copy producer cannot stream; D2 / D4 / more producer warps first, no verdict on the design"
    if best_g1 <= 175.0:
        return head + " -> (i) promote as the board kernel (set MOE7_CFG to the best variant)"
    if best_g1 <= 205.0 and cwf is not None and cwf >= 25.0 and nocomp_ts is not None and nocomp_ts >= 3.5:
        return head + " -> (ii) producer / DRAM side binds: D2S5 / 4-warp cp.async producer, one more check"
    if best_g1 <= 205.0 and pwe is not None and pwe >= 25.0:
        return head + " -> (iii) consumer side binds (CONSONLY says how far): wgmma consumers (step 2)"
    if best_g1 > 205.0 and nocomp_ts is not None and nocomp_ts >= 3.5 and consonly_ts is not None and consonly_ts >= 3.5:
        return head + " -> (vi) KILL: the sync protocol itself costs the time -> D5-A"
    return head + " -> (none of i-vi)"


def _co_shares(res, m):
    """{variant: (saved % of consonly2's GEMM1, saved % of its GEMM2)} of the C31 twins at bs m (medians over the seeds), the base times, or None."""
    base = res.get((m, "consonly2"))
    if not base or not base["g1"] or not base["g2"]:
        return None
    b1, b2 = _med(base["g1"]), _med(base["g2"])
    out = {}
    for name in CO_NAMES:
        r = res.get((m, name))
        if r and r["g1"] and r["g2"]:
            out[name] = (100.0 * (1.0 - _med(r["g1"]) / b1), 100.0 * (1.0 - _med(r["g2"]) / b2))
    return out, b1, b2


def _co_verdict(sh):
    """fable-10 D5 mapping of the C31 shares (GEMM1, saved % of consonly2): NOMMA >= 35 -> tensor-core bound (wgmma, D5-B); NOUNPACK >= 35 -> ALU bound (the dequant is the
    lever); the smem floor (LDSONLY / lds + ldmatrix) >= 60 % of the time (saves <= 40) or NOLDM >= 20 -> smem bound; NOFIX >= 20 -> the fixup must overlap. Every rule that
    fires is listed; none -> the chain is distributed."""
    nomma_t, nounpack_t, nofix_t, noldm_t = (sh.get(k, (None, None))[0] for k in ("co_nomma", "co_nounpack", "co_nofix", "co_noldm"))
    lds_t, smem_t = sh.get("co_ldsonly", (None, None))[0], sh.get("co_smem", (None, None))[0]
    fired = []
    if nomma_t is not None and nomma_t >= CO_RULES[0]:
        fired.append(f"TC-bound (NOMMA saves {nomma_t:.0f}%) -> wgmma consumers (fable-10 D5-B)")
    if nounpack_t is not None and nounpack_t >= CO_RULES[1]:
        fired.append(f"ALU-bound (NOUNPACK saves {nounpack_t:.0f}%) -> the dequant is the lever")
    floor = [x for x in (lds_t, smem_t) if x is not None]
    if (floor and 100.0 - min(floor) >= CO_RULES[2]) or (noldm_t is not None and noldm_t >= CO_RULES[4]):
        fired.append(f"smem-bound (lds-only floor {'-' if lds_t is None else f'{100 - lds_t:.0f}'}% / lds+ldmatrix floor {'-' if smem_t is None else f'{100 - smem_t:.0f}'}% of the time, NOLDM saves "
                     f"{'-' if noldm_t is None else f'{noldm_t:.0f}'}%)")
    if nofix_t is not None and nofix_t >= CO_RULES[3]:
        fired.append(f"fixup (NOFIX saves {nofix_t:.0f}%) -> the split-K fixup + epilogue must overlap the stream")
    return "; ".join(fired) if fired else "no single part reaches its threshold (distributed chain)"


def _measure(state, env=None, t_start=None, build_s=None):
    """MEASUREMENT: see the module docstring. `env` is injected by tests (stubbed timing)."""
    env = env or _RealEnv(state)
    if not env.available():
        print("MEAS-WS skipped: no cuda, no sglang, no v5 or no v7 extension")
        return
    t0 = env.clock()
    spent = (t0 - t_start) if t_start is not None else 0.0
    deadline = t0 + max(20.0, min(BUDGET_S, PREPARE_LIMIT_S - spent - RESERVE_S))
    env.setup()
    d = state["smallm"]["dims"]
    eb1, eb2 = expert_bytes(d)
    lines, skipped = [], []
    res = {}        # (m, name) -> {"g1": [us per seed], "g2": [...], "pair": [...], "bw": [..]}
    ctxs = {}       # (m, seed) -> ctx
    stamp_res = {}  # (m, name) -> (S line, (cwf, pwe) of GEMM1)
    gat = {}        # m -> graph time of the gather pre-pass alone (us)
    try:
        bw_ach, cal_us, zero_us = env.calibrate()
        head = (f"MEAS-WS {env.banner()} budget={deadline - t0:.0f}s (prepare so far {spent:.0f}s) bw_ach={bw_ach:.2f}TB/s cal={cal_us:.1f}us({CAL_MB}MB) zero={zero_us:.1f}us "
                f"inst={len(MEAS_INSTANCES)} cps={M7_COPIES_PER_STAGE} build={'-' if build_s is None else f'{build_s:.0f}s'} d1={WS_D1[0]}x{WS_D1[1]} d2={WS_D2[0]}x{WS_D2[1]} ctas={state['moe7']['nctas']}")
        lm = state["largem"]

        def timed(m, seed, name, stages, pair=False):
            if env.clock() > deadline:
                skipped.append(f"{name}@m{m}s{seed}")
                return
            ctx = ctxs[(m, seed)]
            try:
                fns, st = stages() if callable(stages) else stages      # a v7 stage set is built lazily: its pre-flight failure is a skip, not an abort
                env.run_eager([f for f in fns if f[0] in ("route", "gather", "g1", "g2")])
                r = res.setdefault((m, name), {"g1": [], "g2": [], "pair": [], "bw": []})
                r["bw"].append(env.compare(ctx, st["y"]) if (st is not None and not name.startswith("co_")) else "-")
                fd = dict(fns)
                r["g1"].append(env.graph_time(fd.get("g1z", fd["g1"])))
                r["g2"].append(env.graph_time(fd.get("g2z", fd["g2"])))
                if pair and "pair" in fd:
                    r["pair"].append(env.graph_time(fd["pair"]))
            except Exception as ex:  # noqa: BLE001
                skipped.append(f"{name}@m{m}s{seed}:{_short(ex)}")

        def v7(m, seed, v):
            name, c1, c2, fl, dbgt, pdl = v
            return _v7_stages(ctxs[(m, seed)]["data"], state, c1, c2, fl, dbgt, pdl)

        # pass 1, per M (the decision shape 256 first, so a deadline cuts the cheap shapes, never the decision): contexts (v5 pivot runs), the core variants,
        # wstream, stamps; pass 2: the rest (nocomp2 / consonly2 / ks sweep / pdloff)
        for m in MS:
            vs = variants(m)
            for seed in SEEDS:
                if env.clock() > deadline:
                    skipped.append(f"m{m}s{seed}")
                    continue
                ctxs[(m, seed)] = env.prep(m, seed)
                timed(m, seed, "v5", _v5_stages(ctxs[(m, seed)]["data"], state), pair=True)
                for v in vs:
                    if v[0] in PASS1:
                        timed(m, seed, v[0], lambda v=v: v7(m, seed, v), pair=(v[0] in ("d1s8", "d2s5")))
            vd = {v[0]: v for v in vs}
            for seed in SEEDS:
                if (m, seed) not in ctxs:
                    continue
                fns, _st = _v5_stages(ctxs[(m, seed)]["data"], state)
                dict(fns)["route"]()                                  # fills the chunk table for the wstream kernels (v5's route == v7's)
                if seed == SEEDS[0]:                                  # the k-tiled gather pre-pass alone (it is inside every v7 g1 / pair number)
                    try:
                        gf, _gs = v7(m, seed, vd["d1s8"])
                        dict(gf)["route"]()
                        gat[m] = env.graph_time(dict(gf)["gather"])
                    except Exception as ex:  # noqa: BLE001
                        skipped.append(f"gather@m{m}:{_short(ex)}")
                timed(m, seed, "wstream", (_ws_stages(ctxs[(m, seed)]["data"], state, vd["d1s8"][1], vd["d1s8"][2]), None))
            if (m, SEEDS[0]) in ctxs and env.clock() < deadline:
                for sname, vname in (("stamps", "d1s8"), ("stamps2", "d2s5")):
                    _n, a1, a2, _fl, _dt, _pdl = vd[vname]
                    try:
                        (t1, e1), (t2, e2) = env.stamps(ctxs[(m, SEEDS[0])], a1, a2)
                        stamp_res[(m, sname)] = (f"S moe m={m} {vname} g1 {_stamp_line(t1, e1)} | g2 {_stamp_line(t2, e2)}", _stamp_shares(t1))
                    except Exception as ex:  # noqa: BLE001
                        skipped.append(f"{sname}@m{m}:{_short(ex)}")
        for m in MS:
            for seed in SEEDS:
                if (m, seed) not in ctxs:
                    continue
                for v in variants(m):
                    if v[0] not in PASS1:
                        timed(m, seed, v[0], lambda v=v: v7(m, seed, v), pair=(v[0] == "pdloff"))
    finally:
        env.release()
    # ---- the summary block: S lines first, then I / B lines per M, the decision line ----
    lines.append(head)
    for m in MS:
        for sname in ("stamps", "stamps2"):
            if (m, sname) in stamp_res:
                lines.append(stamp_res[(m, sname)][0])
    for m in CO_MS:
        co = _co_shares(res, m)
        if co:
            sh, b1, b2 = co
            parts = " ".join(f"{n[3:]}={sh[n][0]:.0f}%/{sh[n][1]:.0f}%" for n in CO_NAMES if n in sh)
            lines.append(f"S moe m={m} co-saves (of consonly2 g1={b1:.1f}us g2={b2:.1f}us, g1%/g2%): {parts}")
            lines.append(f"S verdict m={m}: {_co_verdict(sh)}")
    verdict_inputs = {}
    for m in MS:
        seeds = [s for s in SEEDS if (m, s) in ctxs]
        if not seeds:
            lines.append(f"I moe m={m} skipped (budget)")
            continue
        inf = [ctxs[(m, s)]["info"] for s in seeds]
        lines.append(f"I moe m={m} seeds={_fmt(seeds, '{}')} pairs={_fmt([i['pairs'] for i in inf], '{}')} distinct={_fmt([i['distinct'] for i in inf], '{}')} "
                     f"chunks={_fmt([i['chunks'] for i in inf], '{}')} uniq1={_fmt([i['distinct'] * eb1 / 1e6 for i in inf], '{:.0f}')}MB "
                     f"str1={_fmt([i['chunks'] * eb1 / 1e6 for i in inf], '{:.0f}')}MB uniq2={_fmt([i['distinct'] * eb2 / 1e6 for i in inf], '{:.0f}')}MB"
                     + (f" gat={gat[m]:.1f}us" if m in gat else ""))
        piv = res.get((m, "v5"))
        ws = res.get((m, "wstream"))
        vnames = [v[0] for v in variants(m)]
        order = ["v5"] + [v for v in vnames if v in PASS1] + ["wstream"] + [v for v in vnames if v not in PASS1]
        for name in order:
            r = res.get((m, name))
            if not r or not r["g1"]:
                continue
            n = len(r["g1"])
            if name.startswith("co_"):   # C31 twins: timing only, medians only (the 14 KB budget)
                lines.append(f"B moe m={m} v={name} | g1={_med(r['g1']):.1f} g2={_med(r['g2']):.1f} | timing-only")
                continue
            ts = []
            for gi, eb in (("g1", eb1), ("g2", eb2)):
                ts.append(_med([inf[i]["chunks"] * eb / r[gi][i] / 1e6 for i in range(min(n, len(inf)))]))
            x = [_med([piv[gi][i] / r[gi][i] for i in range(min(n, len(piv[gi])))]) if piv and piv[gi] else None for gi in ("g1", "g2")]
            flag = ""
            ud = [inf[i]["distinct"] * eb1 / r["g1"][i] / 1e6 for i in range(min(n, len(inf)))]
            if _med(ud) > 4.5 or (ws and ws["g1"] and name not in ("v5", "wstream") and (_med(r["g1"]) < 0.98 * _med(ws["g1"]))):
                flag = " !INVALID"
            bw = "OK" if all(b == "OK" for b in r["bw"]) else ("-" if all(b == "-" for b in r["bw"]) else ",".join(b for b in r["bw"] if b != "OK")[:24])
            lines.append(f"B moe m={m} v={name} | g1={_fmt(r['g1'])} g2={_fmt(r['g2'])} | ts={ts[0]:.2f}/{ts[1]:.2f} x={_fmt(x, '{:.2f}')} bw={bw}"
                         + (f" pair={_fmt(r['pair'])}" if r["pair"] else "") + flag)
            verdict_inputs[(m, name)] = (_med(r["g1"]), ts[0], bw, flag)
    if 256 in MS and any(k[0] == 256 for k in verdict_inputs):
        real = [(verdict_inputs[(256, v)][0], v) for v in ("d1s8", "d2s5") if (256, v) in verdict_inputs and not verdict_inputs[(256, v)][3]]
        best = min(real) if real else (None, None)
        nc = verdict_inputs.get((256, "nocomp"), (None, None))[1]
        co = verdict_inputs.get((256, "consonly"), (None, None))[1]
        sh = stamp_res.get((256, "stamps" if best[1] != "d2s5" else "stamps2"), (None, None))[1]
        all_ok = all(verdict_inputs[(m, v)][2] == "OK" for (m, v) in verdict_inputs if v in ("d1s8", "d2s5"))
        lines.append(_decide(best[0], best[1], nc, co, None if sh is None else sh[0], None if sh is None else sh[1], all_ok))
    lines.append(f"MEAS-WS done {env.clock() - t0:.0f}s" + (f" skipped: {' '.join(skipped)[:600]}" if skipped else ""))
    out = "\n".join(lines)
    if len(out) > SUMMARY_BUDGET:
        out = out[:SUMMARY_BUDGET - 40] + "\n...(summary cut at budget)"
    print(out)


# ============================================================================================ entry points
def prepare(weights):
    """Called once, untimed. v5's preparation (v4's extensions + self-checks, the v5 extension + its bitwise check: kernel_fn is v5's), then the measurement
    extension (v7 + twins; the build time is printed), v7's self-check against v5 and the measurement (MEASUREMENT-ONLY: nothing of it feeds kernel_fn)."""
    t_start = time.time()
    print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")
    if not HAVE_SGLANG:
        return {"weights": weights, "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    state = _prepare_sglang(weights)
    state["smallm"], state["largem"], state["moe5"], state["moe7"] = None, None, None, None
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
        print(f"okc-diag v5 ready {time.time() - t_start:.0f}s after prepare start")
        build_s = None
        if state.get("smallm") is not None and state.get("largem") is not None and state.get("moe5") is not None:
            try:
                ext, build_s = meas_ext(flags)
                state["moe7"] = prepare_moe7(state["smallm"], state["largem"], ext=ext)
                print(f"okc-diag measws built in {build_s:.0f}s ({len(MEAS_INSTANCES)} instantiations), {time.time() - t_start:.0f}s after prepare start")
                oks = _moe7_selfcheck(state)
                print("okc-diag moe7 vs v5 per bucket: " + " ".join(f"m{b}={'OK' if ok else 'DIFF'}" for b, ok in oks.items()))
            except Exception as ex:  # noqa: BLE001
                print(f"okc-diag measws unavailable: {type(ex).__name__}: {str(ex)[:300]}")
                state["moe7"] = None
        _measure(state, t_start=t_start, build_s=build_s)
        state["moe7"] = None   # the measurement extension is not used by kernel_fn
        torch.cuda.synchronize()
    return state


def kernel_fn(data, state):
    if HAVE_SGLANG:
        if _use_moe5(data, state):
            return _kernel_moe5(data, state)
        if _use_smallm(data, state):
            return _kernel_smallm(data, state)
        if _use_largem(data, state):
            return _kernel_largem(data, state)
        return _kernel_sglang(data, state)
    return _kernel_torch(data, state)
