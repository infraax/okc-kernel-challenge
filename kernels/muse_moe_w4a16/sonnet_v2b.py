# okc: problem=muse_moe_w4a16
# sonnet_v2 -- sonnet_v1 (SGLang's NVFP4 expert path: moe_align_block_size, two Marlin W4A16 GEMMs, silu_and_mul, our fused Triton epilogue at
# M < 32 / SGLang's fused epilogue at M >= 32; NVFP4 weights repacked exactly as baseline.py's prepare) plus a PER-M CONFIG TABLE for Marlin's own
# knobs and our epilogue, chosen by M bucket only (M <= 1, <= 8, <= 16, <= 32, <= 128, else), from the judge's own sweep of measure_v1.py.
#
# CONFIG TABLE (name = b<bm>.<align>.r<GEMM1 reduce><GEMM2 reduce>.<epilogue>, grammar of measure_v1.shape_cfgs), source job: the judge run
# experiments/profiles/muse_moe_w4a16.8d094868ac244fcd.log (tools/parse_meas_moe.py with measure_v1.py), us per call, base -> best:
#   M <= 1    b8.a0.rtt.t1024w8   61.63 ->  57.73
#   M <= 8    b8.a0.rff.t2048w8   75.39 ->  73.02
#   M <= 16   b16.a2.rft.t1024w8  86.40 ->  64.26
#   M <= 32   b32.a2.rff.sg       88.03 ->  78.34
#   M <= 128  b64.a3.rff.sg      208.45 -> 176.58
#   else      b64.a2.rff.sg      316.54 -> 311.97   (M = 256 measured)
#   bm = Marlin/align moe_block_size; a0 = moe_align_block_size(ids, bm, 256); a2 = the same with ignore_invalid_expert=True (non-local slots
#   dropped, not padded); a3 = a2 with num_experts = 32; r: f = (use_atomic_add False, use_fp32_reduce True) as baseline, t = (True, False)
#   (atomic add into C, Marlin zero-initialises it), n = (False, False); epilogue sg = SGLang's fused_post_expert_norm_weighted_sum,
#   t<BLOCK_H>w<warps> = our Triton kernel with that BLOCK_H and num_warps.
# Everything else is sonnet_v1: the local-id range kernel, the weights repack, the activation, the epilogue math (FP32, one BF16 rounding).
#
# PREDICTION: ~1.20x (sonnet_v1 1.152x x same-method delta +4.7%: the table's gain over measure_v1's baseline config, geo-mean over the six
# loads, measured by the same judge timer). Unmeasured as a submission.
# Rules: bucket and all grids/blocks come from shapes only (M = hidden_states.shape[0]); no value-dependent host logic; no host sync in
# kernel_fn; CUDA-graph capturable; precision unchanged (only Marlin's own knobs and our epilogue's tiling); NVFP4 stays NVFP4; activations BF16.
# Any M gets its bucket's config (nearby shapes included); the Marlin kernel handles any moe_block_size in 8/16/32/48/64 for any M.
# Without SGLang (local runs: CPU, RTX 2060) the experts run as plain PyTorch like sonnet_v1 (the table is unused there).
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


def kernel_fn(data, state):
    if HAVE_SGLANG:
        return _kernel_sglang(data, state)
    return _kernel_torch(data, state)
