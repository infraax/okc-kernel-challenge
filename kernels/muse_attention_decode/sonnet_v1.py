# okc: problem=muse_attention_decode
# sonnet_v1 -- fused split-KV flash decoding in Triton, 2 launches instead of SGLang's ~6 (FusedQKNorm, store_cache,
# q->FP8, FA3 [+ its split combine], rms_norm_gate). PREDICTED speedup vs SGLang: ~1.25x geo-mean over the 6 loads
# (range 1.0-1.5x; per load ~2.4x at batch 1, ~1.0x at batch 16, ~0.9x at batch 32). Unmeasured: no H200 here.
# Why: kernel 1 = q/k RMSNorm + RoPE + FP8 cache write of the block's new k/v + attention over the FP8 paged cache,
#   split along the sequence so B*Hkv*splits ~ 4 waves of 132 SMs even with only 2 KV heads; the block's own keys are
#   attended from registers after the same FP8 rounding the cache gets (no HBM round trip, and every cached key is
#   read once). Kernel 2 merges the splits and applies RMSNorm x sigmoid(gate) -> BF16.
# Small loads (sliding layers x46, floors 0.1-3.7 us, batch 1 full 4.3 us) are launch/latency bound: ~4 fewer launches
#   is ~8-12 us a layer, x62 layers ~ 0.5-0.7 ms of SGLang's 3.77 ms decode step. The big full-attention loads (batch
#   16 / 32, floors 63 / 126 us) are HBM bound; FA3 is already close, our 64-byte-row Triton gather is not: expect <= 1x.
# Rules: grid, split size and buffers come from shapes / config only (the page table's width bounds the context);
#   lengths, positions, cache rows and scales are read on the device; no host syncs; every key of the window / context
#   is attended; operands are FP8 (cache) and FP16 (q, exact E4M3 products), accumulation FP32, output BF16.
import functools
import math

import torch
import triton
import triton.language as tl

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


def kernel_fn(data):
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
