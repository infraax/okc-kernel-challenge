# okc: problem=muse_moe_w4a16
"""MEASUREMENT-ONLY candidate (card C13; design/fable-06-out.md D1). Run as `okc check`, NEVER submit.

kernel_fn is sonnet_v1's path, byte for byte (everything between the imports and the "measurement" section below is a
verbatim copy of kernels/muse_moe_w4a16/sonnet_v1.py; tests/test_moe_measure_v1.py compares the two), so the judge's own timing of
this file re-measures our board entry. prepare() repacks the experts as sonnet_v1 does, then runs `_measure(state)` once, untimed, within a
200 s wall budget, on SYNTHETIC inputs (shapes only: seeded skewed routing like reference.generate_input's dist="skewed" / rank="busiest",
random BF16 activations; nothing derived from the judge's inputs, nothing measured feeds kernel_fn). Every timing is a CUDA-graph replay after a
READ-ONLY L2 flush (128 MB sum), like the judge. Nothing runs under graph capture (guarded).

What is swept, for each scored M in (1, 8, 16, 32, 128, 256); a config is (bm, align, r1r2, epilogue):
  bm      Marlin/align block size moe_block_size 8/16/32/48/64 (baseline rule: smallest of 8,16,32,48,64 with M*8/32/bm < 0.9).
  align   a0 = baseline moe_align_block_size(ids, bm, 256)    a1 = same with num_experts=32 (33 buckets -> the CUDA small-batch align kernel
          for numel < 1024, instead of the 257-bucket path)   a2/a3 = a0/a1 with ignore_invalid_expert=True (non-local slots dropped, not padded)
          a4 (M=1 only) = moe_align_single_token(ids, bm).
  r1 r2   reduction of GEMM1 / GEMM2 (use_atomic_add, use_fp32_reduce): f = (False, True) baseline, n = (False, False) bf16 reduce in C,
          t = (True, False) atomic add (the kernel zero-initialises C itself when a slice is split), z = t plus an explicit C.zero_() first.
          Pairs timed: the nine of f/n/t, plus zf fz zz.
  epi     the post-expert epilogue: sg = SGLang's fused_post_expert_norm_weighted_sum (eager below 32 tokens), t<BLOCK_H>w<warps> = our Triton
          kernel (base at M < 32 is t1024w8, at M >= 32 sg); the 12 other variants are timed at the baseline values of the other axes.
The Python wrapper does not expose thread_k/thread_n/sms (they are -1/-1/device value: the CUDA side picks the thread config and blocks per SM)
and `workspace` is only the lock array (size sms*4 ints): neither is a speed knob.
Every variant's final output is checked against the unchanged kernel_fn output (rel L2 < 1e-2, first launch and graph replay); wrong ones are
counted as `bad`, not timed. Configs run in order of how many axes differ from the baseline (1 axis first) until the budget is used up.
The stage split (ids, align, w13, act, w2, epi: each its own graph, cold L2) is timed at the baseline config.

Log (tail < 12 KB): `CFGSIG`, `R moe m= pairs= local= distinct=` (the synthetic routing), one `T64 moe.m<M> <base64>` per M (uint16 LE, us x 100, in
`shape_cfgs` order; 0 = not run, 65535 = fail, 65534 = bad, 65533 = clamped; finals override), then per M
`B moe m=<M> base=<us> | top-5 <cfg>:<us> ... | ok= fail= bad= [skip=]` (base = kernel_fn as is),
`S moe m=<M> align=<us> w13=<us> act=<us> w2=<us> epi=<us> ids=<us> nop=<us> sum=<us>` (ids = local-id kernel; nop = an empty graph's replay) and
`K moe m=<M> ...` (best time per value of each axis). Decode with tools/parse_meas.py / tools/parse_meas_moe.py (the CFG block below is executed there).
"""
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


# ============================================================================================ measurement (MEASUREMENT-ONLY)
import base64
import re
import struct
import time

BUDGET_S = 200.0        # prepare() may run 300 s; the whole run has 15 min
FINAL_RESERVE_S = 14.0  # finals (151 reps x 6 timings x 6 M) + report
EARLY_END_S = 120.0     # configs that differ from the baseline in at most one axis stop here, the rest at BUDGET - reserve
SCREEN_REPS = 31
FINAL_REPS = 151
FLUSH_MB = 128          # read-only L2 flush buffer (> the 50 MB L2)
TOL = 1e-2              # rel L2 of a variant's output against the unchanged kernel_fn output
_DT = torch.bfloat16    # tests on the sm_75 dev card swap in float16

# ---- CFG BEGIN (pure python: tools/parse_meas.py executes this block; keep it stdlib-only) ----
import zlib

MS = (1, 8, 16, 32, 128, 256)
BMS = (8, 16, 32, 48, 64)
AXES = ("bm", "al", "r1", "r2", "ep")
RED_PAIRS = ("ff", "fn", "nf", "nn", "tf", "ft", "tn", "nt", "tt", "zf", "fz", "zz")
EPI_BH = (256, 512, 1024, 2048)
EPI_W = (2, 4, 8)
BASE_EPI_TRITON = "t1024w8"
FLAGS = {"f": (False, True), "n": (False, False), "t": (True, False), "z": (True, False)}  # (use_atomic_add, use_fp32_reduce); z = t + C.zero_()


def base_bm(m):
    """baseline.py's rule: the smallest candidate with M * 8 / 32 / bm < 0.9 (64 if none)."""
    for bm in (8, 16, 32, 48, 64):
        if m * 8 / 32 / bm < 0.9:
            return bm
    return 64


def base_epi(m):
    return "sg" if m >= 32 else BASE_EPI_TRITON


def cfg_name(bm, al, r1, r2, ep):
    return f"b{bm}.{al}.r{r1}{r2}.{ep}"


def shape_cfgs(layer, m):
    """Ordered configs of one M: (name, bm, align, r1, r2, epilogue). The Marlin x align x reduce product at the baseline epilogue,
    then the other epilogue variants at the baseline of everything else."""
    bbm, bep = base_bm(m), base_epi(m)
    out = []
    for bm in BMS:
        for al in ("a0", "a1", "a2", "a3") + (("a4",) if m == 1 else ()):
            for pair in RED_PAIRS:
                out.append((cfg_name(bm, al, pair[0], pair[1], bep), bm, al, pair[0], pair[1], bep))
    for ep in ["sg"] + [f"t{bh}w{w}" for bh in EPI_BH for w in EPI_W]:
        if ep != bep:
            out.append((cfg_name(bbm, "a0", "f", "f", ep), bbm, "a0", "f", "f", ep))
    return out


def base_index(m):
    bbm, bep = base_bm(m), base_epi(m)
    names = [c[0] for c in shape_cfgs("moe", m)]
    return names.index(cfg_name(bbm, "a0", "f", "f", bep))


def deviations(cfg, m):
    """How many axes differ from the baseline config of this M (r1 and r2 count separately)."""
    _, bm, al, r1, r2, ep = cfg
    return (bm != base_bm(m)) + (al != "a0") + (r1 != "f") + (r2 != "f") + (ep != base_epi(m))


def all_shapes():
    return [("moe", m) for m in MS]


def shape_id(layer, m):
    return f"{layer}.m{m}"


def cfg_sig():
    names = [c[0] for layer, m in all_shapes() for c in shape_cfgs(layer, m)]
    return len(names), zlib.crc32("\n".join(names).encode()) & 0xFFFFFFFF
# ---- CFG END ----

_ST_FAIL, _ST_BAD = 65535, 65534
_EPI_RE = re.compile(r"^t(\d+)w(\d+)$")


def _encode_t64(vals):
    """vals: per config None (not run) | float us | 'F' | 'B' -> base64 of uint16 LE (us x 100)."""
    u = []
    for v in vals:
        if v is None:
            u.append(0)
        elif v == "F":
            u.append(_ST_FAIL)
        elif v == "B":
            u.append(_ST_BAD)
        else:
            u.append(max(1, min(65533, int(round(v * 100)))))
    return base64.b64encode(struct.pack(f"<{len(u)}H", *u)).decode()


# ============================================================================================ the pipeline, stage by stage
def _epilogue_param(expert_out, topk_scores, slot_ids, norm_w, block_h, warps):
    """epilogue_triton with BLOCK_H / num_warps as arguments (same kernel)."""
    y = expert_out.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    _epilogue_kernel[(tokens,)](
        y, norm_w, topk_scores.contiguous(), slot_ids.contiguous(), out, POST_NORM_EPS,
        TOPK=topk, KP=triton.next_power_of_2(topk), H=hidden, BLOCK_H=min(block_h, triton.next_power_of_2(hidden)), num_warps=warps)
    return out


def _stage_fns(data, state, cfg):
    """sonnet_v1's _kernel_sglang cut into stages, with the swept knobs as arguments. Returns ([(name, fn)], st): calling the fns in order leaves
    the output in st["out"]; each fn reads the tensors the earlier ones left in `st`. With the baseline config this is _kernel_sglang's code."""
    _, bm, al, r1, r2, ep = cfg
    c = data["config"]
    bank = state["banks"][(c["local_start"], c["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]
    ws = state["workspace"]
    st = {}

    def common(r):
        atomic, fp32 = FLAGS[r]
        return dict(moe_block_size=bm, mul_topk_weights=False, is_ep=expert_map is not None,
                    b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=atomic,
                    use_fp32_reduce=fp32, is_zp_float=False)

    def f_ids():
        if expert_map is not None:
            st["ids"] = local_ids_triton(data["topk_ids"], c["local_start"], c["local_count"])
        else:
            st["ids"] = data["topk_ids"].to(torch.int32).contiguous()
        st["tw"] = data["topk_weights"].to(torch.float32).contiguous()

    def f_align():
        ids = st["ids"]
        if al == "a4":
            from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

            res = moe_align_single_token(ids, bm)
        elif al == "a0":
            res = moe_align_block_size(ids, bm, N_EXPERTS if expert_map is not None else bank["count"])
        else:
            n_exp = N_EXPERTS if al == "a2" else bank["count"]
            if al == "a1":
                res = moe_align_block_size(ids, bm, n_exp)
            else:
                res = moe_align_block_size(ids, bm, n_exp, ignore_invalid_expert=True)
        st["sorted"], st["bexp"], st["npad"] = res

    def f_w13():
        cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
        cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
        cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
        if r1 == "z":
            cache1.zero_()
        st["cache2"], st["cache13"] = cache2, cache13
        st["c1"] = moe_wna16_marlin_gemm(
            x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            st["sorted"], st["bexp"], st["npad"], st["tw"], top_k=TOP_K,
            size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common(r1))

    def f_act():
        silu_and_mul(st["c1"].view(-1, gemm1_n), st["cache2"], expert_ids=st["ids"].view(-1), expert_step=1)

    def f_w2():
        cache3 = st["cache13"][: num_tokens * TOP_K * size_k].view(-1, size_k)
        if r2 == "z":
            cache3.zero_()
        st["y"] = moe_wna16_marlin_gemm(
            st["cache2"], cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
            st["sorted"], st["bexp"], st["npad"], st["tw"], top_k=1,
            size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common(r2)).view(num_tokens, TOP_K, size_k)

    def f_epi():
        y, tw, ids = st["y"], st["tw"], st["ids"]
        if ep == "sg":
            st["out"] = fused_post_expert_norm_weighted_sum(
                y, tw, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=ids)
        elif ep == BASE_EPI_TRITON:
            st["out"] = epilogue_triton(y, tw, ids, state["norm_w"])
        else:
            bh, w = map(int, _EPI_RE.match(ep).groups())
            st["out"] = _epilogue_param(y, tw, ids, state["norm_w"], bh, w)

    fns = [("ids", f_ids), ("align", f_align), ("w13", f_w13), ("act", f_act), ("w2", f_w2), ("epi", f_epi)]
    return fns, st


def _routing(m, seed):
    """Seeded skewed routing for M tokens, the recipe of reference.generate_input(dist="skewed", rank="busiest") re-written here (shapes only):
    top-8 of uniform scores plus a per-expert heat, expert ids rotated so that experts 0-31 are the busiest GPU's."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand((m, N_EXPERTS), generator=g)
    heat = torch.rand((N_EXPERTS,), generator=g).pow(6) * 4
    routed = (scores + heat).topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=N_EXPERTS // 32)
    ids = (ids - 32 * pairs.argmax()) % N_EXPERTS
    return ids.to(torch.int32), (routed.values / routed.values.sum(-1, keepdim=True)).float()


class _RealEnv:
    """The GPU side: synthetic inputs, the unchanged output as reference, CUDA-graph timing after a read-only flush."""

    def __init__(self, state):
        self.state = state
        self.dev = state["norm_w"].device if "norm_w" in state else torch.device("cuda")

    def available(self):
        return HAVE_SGLANG and torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing()

    def clock(self):
        return time.time()

    def banner(self):
        import triton as _t

        return f"gpu={torch.cuda.get_device_name()} triton={_t.__version__} torch={torch.__version__}"

    def setup(self):
        self.flush = torch.ones(FLUSH_MB * 1024 * 1024 // 4, device=self.dev, dtype=torch.float32)
        self.sink = torch.zeros((), device=self.dev, dtype=torch.float32)
        self.gen = torch.Generator(device=self.dev).manual_seed(1234)

    def release(self):
        for a in ("flush", "sink", "gen"):
            if hasattr(self, a):
                delattr(self, a)
        torch.cuda.empty_cache()

    def prep(self, layer, m):
        ids, w = _routing(m, 4242 + m)
        x = torch.randn((m, 4096), generator=self.gen, device=self.dev).to(_DT)
        data = {"hidden_states": x, "topk_ids": ids.to(self.dev), "topk_weights": w.to(self.dev),
                "config": {"bs": m, "d_hidden": 4096, "d_expert": 4096, "n_experts": N_EXPERTS, "local_start": 0,
                           "local_count": 32, "top_k": TOP_K, "sharding": "ep8"}}
        y0 = kernel_fn(data, self.state).clone()
        torch.cuda.synchronize()
        local = (ids >= 0) & (ids < 32)
        info = f"pairs={int(local.sum())} local={int(local.any(dim=1).sum())} distinct={int(torch.unique(ids[local]).numel())}"
        return dict(layer=layer, m=m, data=data, y0=y0, norm=float(y0.float().norm()), info=info)

    def build(self, ctx, cfg):
        fns, st = _stage_fns(ctx["data"], self.state, cfg)

        def run():
            for _, f in fns:
                f()
            return st["out"]
        return run

    def _err(self, ctx, y):
        n = float((y.float() - ctx["y0"].float()).norm())
        return n / max(ctx["norm"], 1e-12) if n == n else float("inf")

    def check(self, ctx, run):
        """First launch of a variant. Returns the relative L2 error against the unchanged kernel_fn output."""
        y = run()
        torch.cuda.synchronize()
        return self._err(ctx, y)

    def _graph_time(self, fn, reps):
        """(median us of graph replays, each preceded by a read-only 128 MB flush; the replayed output, cloned)."""
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

    def time(self, ctx, run, reps):
        us, y = self._graph_time(run, reps)
        return us, self._err(ctx, y)

    def time_base(self, ctx, reps):
        """kernel_fn as is."""
        return self._graph_time(lambda: kernel_fn(ctx["data"], self.state), reps)[0]

    def stage_times(self, ctx, base_cfg, reps):
        """{ids, align, w13, act, w2, epi, nop}: each stage of the baseline config as its own graph (cold L2); nop = a one-element kernel."""
        fns, st = _stage_fns(ctx["data"], self.state, base_cfg)
        for _, f in fns:
            f()
        torch.cuda.synchronize()
        snap = dict(st)
        out = {}
        for name, f in fns:
            st.clear()
            st.update(snap)
            out[name] = self._graph_time(f, reps)[0]
        out["nop"] = self._graph_time(lambda: self.sink.add_(1.0), reps)[0]
        return out


def _short(e):
    return f"{type(e).__name__}: {str(e)[:80]}".replace("\n", " ")


def _marginals(tab, vals):
    """(axis, value) -> best time over the configs (floats in `vals`) that use that value."""
    best = {}
    for ci, v in enumerate(vals):
        if isinstance(v, float):
            for ax, val in zip(AXES, tab[ci][1:]):
                if (ax, val) not in best or v < best[(ax, val)]:
                    best[(ax, val)] = v
    return best


def _measure(state, env=None):
    """MEASUREMENT: see the module docstring. `env` is injected by tests (stubbed timing)."""
    env = env or _RealEnv(state)
    if not env.available():
        print("MEAS-MOE skipped: no cuda or no sglang")
        return
    t0 = env.clock()
    shapes = all_shapes()
    tabs = [shape_cfgs(layer, m) for layer, m in shapes]
    nres = [[None] * len(t) for t in tabs]     # screen value per config: None | us | "F" | "B"
    fin = [dict() for _ in shapes]             # finals by cfg name (table configs and the combo): us
    base_us = [None] * len(shapes)
    stage = [None] * len(shapes)
    ctxs = [None] * len(shapes)
    axok = {}                                  # (axis, value) -> [ok count, fail count]
    failmsg = {}                               # first message per exception type
    ncfg, crc = cfg_sig()
    print(f"MEAS-MOE start {env.banner()} budget={BUDGET_S:.0f}s screen={SCREEN_REPS} finals={FINAL_REPS}", flush=True)
    print(f"CFGSIG n={ncfg} crc={crc:08x} shapes={len(shapes)}")

    def axkeys(cfg):
        return list(zip(AXES, cfg[1:]))

    def attempt(si, cfg, reps):
        """build, check (first launch), time. -> ('ok', us) | ('B', None) | ('F', None); updates the per-axis ok/fail counts."""
        ctx = ctxs[si]
        keys = axkeys(cfg)
        try:
            run = env.build(ctx, cfg)
            if not (env.check(ctx, run) < TOL):
                res = ("B", None)
            else:
                us, err2 = env.time(ctx, run, reps)
                res = ("ok", us) if err2 < TOL else ("B", None)
            for k in keys:
                axok.setdefault(k, [0, 0])[0] += 1
            return res
        except Exception as e:
            for k in keys:
                axok.setdefault(k, [0, 0])[1] += 1
            failmsg.setdefault(type(e).__name__, _short(e))
            return ("F", None)

    def dead(cfg):
        return any(axok.get(k, [1, 0])[0] == 0 and axok.get(k, [1, 0])[1] >= 3 for k in axkeys(cfg))

    def phase(level, end_s, risky=False):
        """Configs that differ from the baseline in `level` axes; risky = the a4 (single-token align) ones, run after everything else."""
        for si, (layer, m) in enumerate(shapes):
            for ci, cfg in enumerate(tabs[si]):
                if deviations(cfg, m) != level or nres[si][ci] is not None or (cfg[2] == "a4") != risky:
                    continue
                if env.clock() - t0 > end_s:
                    return
                if dead(cfg):
                    nres[si][ci] = "F"
                    continue
                kind, us = attempt(si, cfg, SCREEN_REPS)
                nres[si][ci] = us if kind == "ok" else kind

    def finals(si):
        tab, m = tabs[si], shapes[si][1]
        top = sorted((v, ci) for ci, v in enumerate(nres[si]) if isinstance(v, float))[:5]
        for _, ci in top:
            kind, us = attempt(si, tab[ci], FINAL_REPS)
            if kind == "ok":
                fin[si][tab[ci][0]] = us
        base_us[si] = env.time_base(ctxs[si], FINAL_REPS)
        # combo: the best value of every axis (screen marginals), r1 and r2 chosen separately; skipped if it is a config already timed
        vals = [fin[si].get(tab[ci][0], v) for ci, v in enumerate(nres[si])]
        best = _marginals(tab, vals)
        pick = {}
        for ax in AXES:
            cand = [(us, val) for (a, val), us in best.items() if a == ax]
            if cand:
                pick[ax] = min(cand, key=lambda t: t[0])[1]
        if len(pick) == len(AXES):
            cfg = (cfg_name(pick["bm"], pick["al"], pick["r1"], pick["r2"], pick["ep"]), pick["bm"], pick["al"], pick["r1"], pick["r2"], pick["ep"])
            if cfg[0] not in fin[si]:
                kind, us = attempt(si, cfg, FINAL_REPS)
                if kind == "ok":
                    fin[si][cfg[0]] = us

    def progress(name):
        n = [sum(isinstance(v, float) for r in nres for v in r), sum(v is None for r in nres for v in r)]
        print(f"MEAS-MOE phase {name} t={env.clock() - t0:.0f}s timed={n[0]} unrun={n[1]}", flush=True)

    try:
        env.setup()
        for i, (layer, m) in enumerate(shapes):
            ctxs[i] = env.prep(layer, m)
            print(f"R {layer} m={m} {ctxs[i]['info']}")
            base_us[i] = env.time_base(ctxs[i], SCREEN_REPS)
            try:
                stage[i] = env.stage_times(ctxs[i], tabs[i][base_index(m)], FINAL_REPS)
            except Exception as e:
                failmsg.setdefault("stage:" + type(e).__name__, "stage " + _short(e))
        progress("setup")
        for level in range(0, 6):
            phase(level, EARLY_END_S if level <= 1 else BUDGET_S - FINAL_RESERVE_S)
            progress(f"dev{level}")
        for si in range(len(shapes)):
            if env.clock() - t0 > BUDGET_S:
                break
            finals(si)
        progress("finals")
        for level in range(1, 6):  # a4 last: an unexpected device fault there cannot cost the results above
            phase(level, BUDGET_S - 8.0, risky=True)
        if any(isinstance(v, float) for ci, v in enumerate(nres[0]) if tabs[0][ci][2] == "a4"):
            finals(0)
        progress("a4")
    except Exception as e:  # e.g. a device-side fault: report what we have
        print(f"MEAS-MOE aborted: {_short(e)}")
    finally:
        _report(env, shapes, tabs, nres, fin, base_us, stage, failmsg, t0)
        try:
            env.release()
        except Exception:
            pass


def _report(env, shapes, tabs, nres, fin, base_us, stage, failmsg, t0):
    for _, msg in sorted(failmsg.items()):
        print(f"FAIL {msg}")
    lines_b, lines_s, lines_k = [], [], []
    for si, (layer, m) in enumerate(shapes):
        tab = tabs[si]
        vals = [fin[si].get(c[0], nres[si][ci]) for ci, c in enumerate(tab)]
        print(f"T64 {shape_id(layer, m)} {_encode_t64(vals)}")
        ok = sorted((us, name) for name, us in fin[si].items())
        n_ok = sum(isinstance(v, float) for v in nres[si])
        n_fail = sum(v == "F" for v in nres[si])
        n_bad = sum(v == "B" for v in nres[si])
        n_skip = sum(v is None for v in nres[si])
        top = " ".join(f"{name}:{us:.2f}" for us, name in ok[:5]) or "-"
        b = base_us[si]
        line = f"B {layer} m={m} base={'-' if b is None else f'{b:.2f}'} | top-5 {top} | ok={n_ok} fail={n_fail} bad={n_bad}"
        lines_b.append(line + (f" skip={n_skip}" if n_skip else ""))
        s = stage[si]
        if s:
            tot = sum(s[k] for k in ("ids", "align", "w13", "act", "w2", "epi"))
            lines_s.append(f"S {layer} m={m} align={s['align']:.2f} w13={s['w13']:.2f} act={s['act']:.2f} w2={s['w2']:.2f} "
                           f"epi={s['epi']:.2f} ids={s['ids']:.2f} nop={s['nop']:.2f} sum={tot:.2f}")
        else:
            lines_s.append(f"S {layer} m={m} n/a")
        best = _marginals(tab, vals)
        parts = []
        for ax in AXES:
            items = sorted(((us, val) for (a, val), us in best.items() if a == ax))
            if ax == "ep":
                items = items[:4]
            parts.append(ax + " " + " ".join(f"{val}:{us:.1f}" for us, val in items))
        lines_k.append(f"K {layer} m={m} " + " | ".join(parts))
    print(f"MEAS-MOE done in {env.clock() - t0:.1f}s")
    for line in lines_k + lines_s + lines_b:  # last: if the tail is ever cut, the human lines survive
        print(line)


# ============================================================================================ entry points
def prepare(weights):
    """Called once, untimed. NVFP4 weights are repacked into Marlin's layout exactly as baseline.py does; then the synthetic sweep (MEASUREMENT-ONLY)."""
    print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")
    if HAVE_SGLANG:
        state = _prepare_sglang(weights)
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
