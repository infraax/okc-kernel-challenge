# okc: problem=muse_attention_decode
# sonnet_v4 -- v3 with SINGLE-LAUNCH SLIDING attention (Move A of design/fable-07-out.md D2). Full-attention layers are v3's code unchanged.
# CHANGES (sliding layers only; v3's two launches _attn_kernel + _reduce_kernel become one launch + a 4-byte-per-(b, kv head) memset)
#   1. The reducer is folded into the split kernel (_attn_fused_kernel). Each program stores its partial [M rows x 64 fp32 + m + l] and does
#      tl.atomic_add(counter[b * HKV + h], 1, sem="acq_rel"); the program that sees (participants - 1) is the last arriver and merges all the
#      partials of that (b, kv head) (online max / sum), applies v_scale, the per-head RMSNorm and sigmoid(gate), and writes BF16.
#      Triton emits a CTA barrier before the scalar atomic (every thread's partial stores are ordered before thread 0's release) and the
#      result is broadcast to the CTA after the acquire; partials are read with .cg (L2) loads. Nothing spins: no residency assumption.
#   2. Counters are allocated in kernel_fn with torch.zeros (one memset graph node per call; nothing survives from a previous call,
#      the merger resets nothing). Participants = the splits that hold keys (from seq_lens on the device) + the program of the block's own keys.
#   3. Splits are sized from the window (shapes only): span 2047 keys -> decode 256 keys (8 splits, 2 tiles of 128), verify 512 keys (4 splits:
#      64 rows per program, one wave at 3 programs / SM); v3 used 16 / 8 / 6 splits for batch 1 / 16 / 32. Fewer splits = fewer partials for
#      the last arriver to merge. sonnet_v4b: 128 / 256 keys (more programs, more partials).
#   4. Window-edge masking: a window term is only needed on verify blocks (rows of one block start their window one key apart); for q_len 1 the
#      split bounds already are the window, so the compare is dropped. The tail mask stays (it is the split end vs seq_len).
# NOT folded: full layers. Their reducer merges up to 255 partials for each of 16 x B rows in parallel over >= 16 x B CTAs; one last arriver
#   per (b, kv head pair) would read ~1 MB serially (decode b1: 224 partials x 16 rows x 256 B), slower than the 2.6-3.4 us reducer.
#   (A two-level tree could do it; not worth it for 3 us on a 100-200 us kernel, 1.5 us only at batch 1 where full is 15 us.)
# PREDICTED (H200, not measurable here; v3 judged -> v4): sliding 7.98 / 13.65 / 16.43 / 9.49 / 17.30 / 25.37 us
#   -> 6.0 / 10.5 / 12.5 / 7.5 / 13.5 / 19 us (b1 decode, 16, 32, verify 1, 16, 32). Fable predicted 5.5 / 9 / 11 / 6.5 / 11 / 15; I do not
#   agree with the last three: the reducer launch (2.3-2.5 us in the profile of v3, plus ~1 us launch gap inside the graph) disappears, but the
#   memset node (~1 us), the last arriver's serial merge tail (L2 round trip of ~9 partials, ~1.2 us; ~2 us for the 64-row verify partials)
#   and the unchanged DRAM time stay. So ~ -2.5 us/shape on decode, -3..-6 us on verify-16/32 (v3's 8 / 6 split choice gives a much shorter tail
#   here). Per-load speedup vs SGLang (v3 judged -> v4): decode 1 2.32 -> ~2.44, decode 16 1.90 -> ~2.02, decode 32 1.95 -> ~2.05,
#   verify 1 2.06 -> ~2.17, verify 16 1.19 -> ~1.25, verify 32 1.10 -> ~1.17; geo-mean 1.689 -> ~1.85 (Fable: ~1.90).
# UNVERIFIABLE HERE: every H200 timing; whether the memset node costs less than 1 us in the graph; the cold-code cost of the larger fused kernel.
# Rules: as v3 (grid, split size and buffers from shapes / config only; seq_lens, positions, cache rows and scales read on the device; every key
#   of the window attended; K / V from the FP8 cache as is; q at most E4M3; FP32 accumulation; BF16 output; no state across calls).
import functools
import math

import torch
import triton
import triton.language as tl

BLOCK_N = 128           # keys per inner step (sliding layers, v2's kernel)
# Full-attention layers (window == 0), per q_len class. kernel "sep": _full_kernel (one [M, 64] problem per KV head, two heads per
# program); "cat": _full_cat_kernel (decode only: the head pair as one 16-row problem on [BN, 128] tiles).
FULL_CFG = {
    "decode": dict(kernel="cat", bn=64, stages=3, warps=2, programs=1056),   # 4 resident programs / SM -> two waves
    "verify": dict(kernel="sep", bn=64, stages=4, warps=4, programs=256),    # 2 resident programs / SM -> one wave
}
FULL_MIN_TILES = 4      # fewest tiles per split when a shape has few (batch x head pairs)
# Sliding layers (single launch): (keys per split, keys per tile) per q_len class; shapes only.
SLIDE_CFG = {"decode": dict(split=256, bn=128), "verify": dict(split=512, bn=128)}
SLIDE_MAX_SPLITS = 64   # bound on the partial buffers for odd (very wide window) configs
MAX_SPLITS = 255        # the reducer merges up to 256 partials (the splits + the token's own) in one pass (8 warps above 128)
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
    BL: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
):
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

        r = tl.arange(0, M)
        pidx = ((b * HKV + h) * (NSPLIT + 1) + s) * M + r
        tl.store(part_m + pidx, m_i * c, mask=rvalid)
        tl.store(part_l + pidx, l_i, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, acc, mask=rvalid[:, None])


# --------------------------------------------------------------------------------------------- sliding layers (v4): one launch
@triton.jit
def _attn_fused_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l, counter, out,
    sm_scale, qk_factor, log2_theta, pt_stride, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    WINDOW: tl.constexpr, ROPE: tl.constexpr, EDGE: tl.constexpr, NATIVE_LOAD: tl.constexpr,
    NATIVE_STORE: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr,
    BL: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
):
    """Sliding-window layers: v3's split kernel (_attn_kernel) with the reducer folded in. The last program of a (b, kv head) to
    finish merges the partials of all its rows and writes the BF16 output."""
    s = tl.program_id(0)          # 0 .. NSPLIT-1: cached keys of one split; NSPLIT: the block's own keys
    h = tl.program_id(1)
    b = tl.program_id(2)
    d = tl.arange(0, 64)[None, :]
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64

    seq = tl.load(seq_lens + b).to(tl.int32)           # cached tokens; the block starts at this position
    lo = tl.maximum(seq - WINDOW + 1, 0)
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
                    if EDGE:
                        # rows of a verify block start their window one key apart; for q_len 1 the split bounds are the window
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
            ok = ok & (ti[None, :] > j[:, None] - WINDOW)
            sc = tl.where(ok, sc, float("-inf"))
            m_i, l_i, acc = _softmax_step(sc, c, m_i, l_i, acc, vt, CDTYPE)

        r = tl.arange(0, M)
        pidx = ((b * HKV + h) * (NSPLIT + 1) + s) * M + r
        tl.store(part_m + pidx, m_i * c, mask=rvalid)
        tl.store(part_l + pidx, l_i, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, acc, mask=rvalid[:, None])

        # ---- arrive; the last of the (b, kv head)'s programs that hold a partial merges them all
        n_active = (seq - lo + split_len - 1) // split_len        # cached splits with keys; the block's own one follows them
        old = tl.atomic_add(counter + (b * HKV + h), 1, sem="acq_rel")
        if old == n_active:
            vscale = tl.load(v_scale_ptr)
            hq = h * G + r % GP
            tok = b * L + j
            base = (b * HKV + h) * (NSPLIT + 1) * M + r
            Mx = tl.full([M], float("-inf"), tl.float32)
            Lsum = tl.zeros([M], tl.float32)
            O = tl.zeros([M, 64], tl.float32)
            gate = tl.load(qkv + tok[:, None] * QKV + ((HQ + 2 * HKV) * 64 + hq * 64)[:, None] + d, mask=rvalid[:, None], other=0.0).to(tl.float32)
            for s2 in range(n_active + 1):
                pi = base + tl.where(s2 == n_active, NSPLIT, s2) * M
                ms = tl.load(part_m + pi, mask=rvalid, other=float("-inf"), cache_modifier=".cg")
                ls = tl.load(part_l + pi, mask=rvalid, other=0.0, cache_modifier=".cg")
                os_ = tl.load(part_o + pi[:, None] * 64 + d, mask=rvalid[:, None], other=0.0, cache_modifier=".cg")
                m_new = tl.maximum(Mx, ms)
                m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
                alpha = tl.exp2(Mx - m_use)
                w = tl.exp2(ms - m_use)
                Lsum = Lsum * alpha + w * ls
                O = O * alpha[:, None] + w[:, None] * os_
                Mx = m_new
            o = O / Lsum[:, None] * vscale
            o = o * tl.rsqrt(tl.sum(o * o, 1) / 64.0 + 1e-5)[:, None]
            o = o * (1.0 / (1.0 + tl.exp(-gate)))
            tl.store(out + tok[:, None] * (HQ * 64) + hq[:, None] * 64 + d, o.to(tl.bfloat16), mask=rvalid[:, None])


# --------------------------------------------------------------------------------------------- full layers (v3)
@triton.jit
def _pages(prow, t, cap, PAGE: tl.constexpr):
    """Page id of every key position t (bounded by the page table's width only, so it can be issued before seq_len is known)."""
    return tl.load(prow + t // PAGE, mask=t < cap, other=0)


@triton.jit
def _gather(k_cache, v_cache, pg, t, end, hb, d, PAGE: tl.constexpr, HKV: tl.constexpr):
    """K and V rows of key positions t for the KV heads hb/64 and hb/64 + 1 (adjacent: one 128-byte line per row):
    four uint8 [BN, 64] tiles, requested together; rows at or beyond `end` are not read."""
    if PAGE == 1:
        row = pg.to(tl.int64)
    else:
        row = pg.to(tl.int64) * PAGE + t % PAGE
    off = tl.multiple_of(row * (HKV * 64) + hb, 64)
    p = off[:, None] + d
    m = (t < end)[:, None]
    k0 = tl.load(k_cache + p, mask=m, other=0, eviction_policy="evict_first")
    k1 = tl.load(k_cache + (p + 64), mask=m, other=0, eviction_policy="evict_first")
    v0 = tl.load(v_cache + p, mask=m, other=0, eviction_policy="evict_first")
    v1 = tl.load(v_cache + (p + 64), mask=m, other=0, eviction_policy="evict_first")
    return k0, k1, v0, v1


@triton.jit
def _step(q, ku, vu, m_i, l_i, acc, c, tm, need_mask, NATIVE_LOAD: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr):
    """One tile of one KV head: logits, (masked on the tail tile only) online softmax, P.V."""
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
    if need_mask:
        sc = tl.where(tm[None, :], sc, float("-inf"))
    return _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE)


@triton.jit
def _own_head(qkv, positions, cache_loc, k_cache, v_cache, b, h, q, j, c, ks, vs, log2_theta, m_i, l_i, acc,
              L: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr, NATIVE_STORE: tl.constexpr,
              QK8: tl.constexpr, CDTYPE: tl.constexpr, BT: tl.constexpr):
    """The block's own keys / values of one KV head (norm, RoPE, FP8 round + cache write, causal tile from registers)."""
    d = tl.arange(0, 64)[None, :]
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64
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
    sc = tl.where(tv[None, :] & (ti[None, :] <= j[:, None]), sc, float("-inf"))
    return _softmax_step(sc, c, m_i, l_i, acc, vt, CDTYPE)


@triton.jit
def _full_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l,
    sm_scale, qk_factor, log2_theta, pt_stride, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    ROPE: tl.constexpr, NATIVE_LOAD: tl.constexpr, NATIVE_STORE: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr,
    BL: tl.constexpr, M: tl.constexpr, BN: tl.constexpr, BT: tl.constexpr,
):
    """Full-attention layers. One program = one split of one request for TWO adjacent KV heads. The keys of the split are
    streamed through a software pipeline kept in registers: the page ids of tile i+3 and the K / V rows of tile i+2 (both
    heads, 128-byte lines) are requested while tile i is computed, so the page-table -> row-gather chain of a tile is hidden
    behind two tiles of compute and three tiles' bytes are in flight per program."""
    s = tl.program_id(0)          # 0 .. NSPLIT-1: cached keys of one split; NSPLIT: the block's own keys
    hp = tl.program_id(1)
    b = tl.program_id(2)
    h0 = hp * 2
    d = tl.arange(0, 64)[None, :]
    ar = tl.arange(0, BN)
    cap = pt_stride * PAGE

    seq = tl.load(seq_lens + b).to(tl.int32)           # cached tokens; the block starts at this position
    start = s * split_len
    end = tl.minimum(seq, start + split_len)
    prow = page_table + b * pt_stride
    t0 = start + ar
    pgA = _pages(prow, t0, cap, PAGE)                  # page ids do not depend on seq_len: issued with its load and the q loads
    ahead = tl.load(prow + (start + BN + tl.arange(0, 2 * BN)) // PAGE, mask=(start + BN + tl.arange(0, 2 * BN)) < cap, other=0)
    ks = tl.load(k_scale_ptr)
    q0, rvalid, j = _prep_q(qkv, positions, b, h0, qk_factor, log2_theta, L, G, GP, HQ, HKV, ROPE, BL, M, QK8, NATIVE_STORE, CDTYPE)
    q1, rvalid, j = _prep_q(qkv, positions, b, h0 + 1, qk_factor, log2_theta, L, G, GP, HQ, HKV, ROPE, BL, M, QK8, NATIVE_STORE, CDTYPE)
    # `ahead` (the page ids of tiles 1 and 2) is only read to pull those lines of the page table into L1 / L2 now, in parallel with
    # the q loads, so the pipeline prologue's chain of page loads finds them hot; the * 0.0 term (IEEE: not folded) keeps the load
    # alive without changing the value, and is evaluated after q prep so nothing waits for the page table before the q loads issue.
    c = sm_scale * ks * 1.4426950408889634 + tl.sum(ahead.to(tl.float32), 0) * 0.0   # logits in the log2 domain
    m0 = tl.full([M], float("-inf"), tl.float32)
    l0 = tl.zeros([M], tl.float32)
    a0 = tl.zeros([M, 64], tl.float32)
    m1 = tl.full([M], float("-inf"), tl.float32)
    l1 = tl.zeros([M], tl.float32)
    a1 = tl.zeros([M, 64], tl.float32)

    if s < NSPLIT:
        # ---- cached keys [start, end): nothing to do (zero iterations, no bytes read) when the request ends before the split
        nt = (end - start + BN - 1) // BN
        hb = h0 * 64
        for i in range(nt):
            tc = t0 + i * BN
            kA0, kA1, vA0, vA1 = _gather(k_cache, v_cache, pgA, tc, end, hb, d, PAGE, HKV)
            pgA = _pages(prow, tc + BN, cap, PAGE)
            tm = tc < end
            need_mask = start + (i + 1) * BN > end
            m0, l0, a0 = _step(q0, kA0, vA0, m0, l0, a0, c, tm, need_mask, NATIVE_LOAD, QK8, CDTYPE)
            m1, l1, a1 = _step(q1, kA1, vA1, m1, l1, a1, c, tm, need_mask, NATIVE_LOAD, QK8, CDTYPE)
    else:
        vs = tl.load(v_scale_ptr)
        m0, l0, a0 = _own_head(qkv, positions, cache_loc, k_cache, v_cache, b, h0, q0, j, c, ks, vs, log2_theta, m0, l0, a0,
                               L, HQ, HKV, ROPE, NATIVE_STORE, QK8, CDTYPE, BT)
        m1, l1, a1 = _own_head(qkv, positions, cache_loc, k_cache, v_cache, b, h0 + 1, q1, j, c, ks, vs, log2_theta, m1, l1, a1,
                               L, HQ, HKV, ROPE, NATIVE_STORE, QK8, CDTYPE, BT)

    if (s == NSPLIT) | (start < end):
        r = tl.arange(0, M)
        pidx = ((b * HKV + h0) * M + r) * (NSPLIT + 1) + s     # [b, head, row, split]: a row's splits are contiguous for the reducer
        tl.store(part_m + pidx, m0 * c, mask=rvalid)
        tl.store(part_l + pidx, l0, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, a0, mask=rvalid[:, None])
        pidx = pidx + M * (NSPLIT + 1)
        tl.store(part_m + pidx, m1 * c, mask=rvalid)
        tl.store(part_l + pidx, l1, mask=rvalid)
        tl.store(part_o + pidx[:, None] * 64 + d, a1, mask=rvalid[:, None])


@triton.jit
def _gather_cat(k_cache, v_cache, pg, t, end, hb, c, PAGE: tl.constexpr, HKV: tl.constexpr):
    """K and V rows of key positions t for the KV heads hb/64 and hb/64 + 1 as ONE [BN, 128] tile each: the two heads' 64 bytes
    are adjacent in the cache row, so a row is one whole 128-byte line (16-byte accesses, 8 lanes per row)."""
    if PAGE == 1:
        row = pg.to(tl.int64)
    else:
        row = pg.to(tl.int64) * PAGE + t % PAGE
    p = tl.multiple_of(row * (HKV * 64) + hb, 128)[:, None] + c
    m = (t < end)[:, None]
    ku = tl.load(k_cache + p, mask=m, other=0, eviction_policy="evict_first")
    vu = tl.load(v_cache + p, mask=m, other=0, eviction_policy="evict_first")
    return ku, vu


@triton.jit
def _step_cat(q, ku, vu, m_i, l_i, acc, c, tm, need_mask, NATIVE_LOAD: tl.constexpr, QK8: tl.constexpr,
              CDTYPE: tl.constexpr):
    """One tile of a head pair, rows 0..7 = head 0, rows 8..15 = head 1 (q is zero outside its head's 64 columns): one logit
    dot and one softmax for both heads; P.V is computed for all 128 columns and only each head's own 64 are kept."""
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
    if need_mask:
        sc = tl.where(tm[None, :], sc, float("-inf"))
    return _softmax_step(sc, c, m_i, l_i, acc, vc, CDTYPE)


@triton.jit
def _full_cat_kernel(
    qkv, positions, cache_loc, seq_lens, page_table, k_cache, v_cache, k_scale_ptr, v_scale_ptr,
    part_o, part_m, part_l,
    sm_scale, qk_factor, pt_stride, split_len, NSPLIT,
    HQ: tl.constexpr, HKV: tl.constexpr, PAGE: tl.constexpr,
    NATIVE_LOAD: tl.constexpr, NATIVE_STORE: tl.constexpr, QK8: tl.constexpr, CDTYPE: tl.constexpr,
    BN: tl.constexpr, BT: tl.constexpr,
):
    """Full-attention decode (q_len 1, 8 query heads per KV head, no RoPE). One program = one split of one request for TWO
    adjacent KV heads, and the pair is ONE attention problem: 16 query rows (rows 0..7 head 0, 8..15 head 1) against
    [BN, 128] key / value tiles (both heads' 64 bytes of a cache row = one 128-byte line), with q zero outside its own
    head's columns. Same cp.async pipeline as _full_kernel (page ids one tile ahead in registers, 3 tiles in flight)."""
    s = tl.program_id(0)          # 0 .. NSPLIT-1: cached keys of one split; NSPLIT: the token's own key
    hp = tl.program_id(1)
    b = tl.program_id(2)
    h0 = hp * 2
    c128 = tl.arange(0, 128)[None, :]
    d = c128 % 64
    ar = tl.arange(0, BN)
    cap = pt_stride * PAGE
    QKV: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    G: tl.constexpr = HQ // HKV

    seq = tl.load(seq_lens + b).to(tl.int32)
    start = s * split_len
    end = tl.minimum(seq, start + split_len)
    prow = page_table + b * pt_stride
    t0 = start + ar
    pgA = _pages(prow, t0, cap, PAGE)
    ahead = tl.load(prow + (start + BN + tl.arange(0, 2 * BN)) // PAGE, mask=(start + BN + tl.arange(0, 2 * BN)) < cap, other=0)
    ks = tl.load(k_scale_ptr)

    # ---- q: row r = query head h0 * G + r (G = 8), RMSNorm, qk_factor, FP8 E4M3; columns outside its own head's 64 are zero
    r = tl.arange(0, 16)
    own = (c128 // 64) == (r // G)[:, None]
    x = tl.load(qkv + b * QKV + (h0 * G + r)[:, None] * 64 + d, mask=own, other=0.0).to(tl.float32)
    xn = x * tl.rsqrt(tl.sum(x * x, axis=1) / 64.0 + 1e-5)[:, None] * qk_factor
    if QK8:
        if NATIVE_STORE:
            q = xn.to(tl.float8e4nv)
        else:
            q = _encode_e4m3(xn).to(tl.float8e4nv, bitcast=True)
    else:
        q = xn.to(CDTYPE)
    c = sm_scale * ks * 1.4426950408889634 + tl.sum(ahead.to(tl.float32), 0) * 0.0   # see _full_kernel
    m_i = tl.full([16], float("-inf"), tl.float32)
    l_i = tl.zeros([16], tl.float32)
    acc = tl.zeros([16, 128], tl.float32)

    if s < NSPLIT:
        nt = (end - start + BN - 1) // BN
        hb = h0 * 64
        for i in range(nt):
            tc = t0 + i * BN
            ku, vu = _gather_cat(k_cache, v_cache, pgA, tc, end, hb, c128, PAGE, HKV)
            pgA = _pages(prow, tc + BN, cap, PAGE)
            m_i, l_i, acc = _step_cat(q, ku, vu, m_i, l_i, acc, c, tc < end, start + (i + 1) * BN > end, NATIVE_LOAD, QK8, CDTYPE)
    else:
        # ---- the token's own key / value (both heads): norm, FP8 round + cache write, attended from registers
        vs = tl.load(v_scale_ptr)
        ti = tl.arange(0, BT)
        tv = ti < 1
        base = qkv + (b + ti)[:, None] * QKV
        kx = tl.load(base + HQ * 64 + h0 * 64 + c128, mask=tv[:, None], other=0.0).to(tl.float32)
        k2 = tl.reshape(kx, [BT, 2, 64])
        k2 = k2 * tl.rsqrt(tl.sum(k2 * k2, axis=2) / 64.0 + 1e-5)[:, :, None]
        kn = tl.reshape(k2, [BT, 128])
        vn = tl.load(base + (HQ + HKV) * 64 + h0 * 64 + c128, mask=tv[:, None], other=0.0).to(tl.float32)
        slot = tl.load(cache_loc + b + ti, mask=tv, other=0)
        coff = slot[:, None] * (HKV * 64) + h0 * 64 + c128
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
        sc = tl.where(tv[None, :], sc, float("-inf"))
        m_i, l_i, acc = _softmax_step(sc, c, m_i, l_i, acc, vt, CDTYPE)

    if (s == NSPLIT) | (start < end):
        hh = r // G
        pidx = ((b * HKV + h0 + hh) * 16 + (r % G)) * (NSPLIT + 1) + s     # [b, head, row, split]
        tl.store(part_m + pidx, m_i * c)
        tl.store(part_l + pidx, l_i)
        tl.store(part_o + pidx[:, None] * 64 + d, acc, mask=own)


# --------------------------------------------------------------------------------------------- kernel 2
@triton.jit
def _reduce_kernel(
    part_o, part_m, part_l, qkv, seq_lens, v_scale_ptr, out, split_len, NSPLIT,
    L: tl.constexpr, G: tl.constexpr, GP: tl.constexpr, HQ: tl.constexpr, HKV: tl.constexpr, WINDOW: tl.constexpr,
    M: tl.constexpr, BS: tl.constexpr, ROWMAJOR: tl.constexpr,
):
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
    if ROWMAJOR:
        base = ((b * HKV + h) * M + row) * (NSPLIT + 1)        # [b, head, row, split]: one contiguous run per output row
    else:
        base = (b * HKV + h) * (NSPLIT + 1) * M + row          # [b, head, split, row] (v2's kernel)
    gate = tl.load(qkv + t * QKV + (HQ + 2 * HKV) * 64 + hq * 64 + d).to(tl.float32)

    Mx = tl.full([1], float("-inf"), tl.float32)
    Lsum = tl.zeros([1], tl.float32)
    O = tl.zeros([64], tl.float32)
    for s0 in range(0, n_active + 1, BS):
        si = s0 + tl.arange(0, BS)
        sv = si <= n_active
        if ROWMAJOR:
            pidx = base + tl.where(si == n_active, NSPLIT, si)
        else:
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


def _plan_full(batch, pairs, span, bn, programs):
    """Split length of the full-layer kernels (shapes only): about `programs` programs on the machine, at least FULL_MIN_TILES
    tiles per split, at most MAX_SPLITS splits (one reducer pass). A program streams its split through a 3-tile cp.async
    pipeline, so it is bandwidth-bound once running and costs ~3 us to start; more programs only add startup and partials."""
    ns = min(MAX_SPLITS, max(1, round(programs / (batch * pairs))))
    split_len = max(FULL_MIN_TILES * bn, _cdiv(_cdiv(span, ns), bn) * bn)
    return split_len, _cdiv(span, split_len)


def kernel_fn(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    page, window = cfg["page_size"], cfg["window"]
    group = hq // hkv
    gp = triton.next_power_of_2(group)
    bl = max(triton.next_power_of_2(length), 16 // gp)
    rows = bl * gp
    if d != 64 or hq % hkv or rows > MAX_ROWS or length > BLOCK_N:
        raise ValueError(f"unsupported config: head_dim {d}, {hq} q heads / {hkv} kv heads, q_len {length}")
    block_t = max(16, triton.next_power_of_2(length))
    qkv = data["qkv"]
    dev = qkv.device
    cdtype, native_load, native_store, qk8 = _device_mode(dev.type, dev.index or 0)

    max_pages = data["page_table"].shape[1]
    cap = max_pages * page                       # most tokens a request's page table can hold
    span = max(1, min(window - 1, cap) if window else cap)
    page_table = data["page_table"]
    k_cache = data["k_cache"].view(torch.uint8)
    v_cache = data["v_cache"].view(torch.uint8)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)
    theta = math.log2(cfg["rope_theta"]) if cfg["rope_theta"] else 0.0

    full_layer = not window and hkv % 2 == 0
    if full_layer:
        # ---- full-attention layers: both KV heads of a pair per program, software-pipelined gathers
        fc = dict(FULL_CFG["decode" if length == 1 else "verify"])
        # few programs (batch 1): every SM runs one or two, nothing shares the issue slots, so a program's own per-tile chain is what
        # counts -> 4 warps share each tile; with a saturated machine the 2-warp program's lower instruction count wins
        if fc["warps"] < 4 and batch * _plan_full(batch, hkv // 2, span, fc["bn"], fc["programs"])[1] <= 2 * SM_COUNT:
            fc["warps"] = 4
        split_len, nsplit = _plan_full(batch, hkv // 2, span, fc["bn"], fc["programs"])
        part_o = torch.empty((batch, hkv, nsplit + 1, rows, 64), dtype=torch.float32, device=dev)
        part_m = torch.empty((batch, hkv, nsplit + 1, rows), dtype=torch.float32, device=dev)
        part_l = torch.empty_like(part_m)
        if fc["kernel"] == "cat" and length == 1 and group == 8 and not cfg["rope_theta"]:
            _full_cat_kernel[(nsplit + 1, hkv // 2, batch)](
                qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
                data["k_scale"], data["v_scale"], part_o, part_m, part_l,
                cfg["sm_scale"], cfg["qk_scale_factor"], page_table.stride(0), split_len, nsplit,
                HQ=hq, HKV=hkv, PAGE=page, NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype,
                BN=fc["bn"], BT=block_t, num_warps=fc["warps"], num_stages=fc["stages"],
            )
        else:
            _full_kernel[(nsplit + 1, hkv // 2, batch)](
                qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
                data["k_scale"], data["v_scale"], part_o, part_m, part_l,
                cfg["sm_scale"], cfg["qk_scale_factor"], theta, page_table.stride(0), split_len, nsplit,
                L=length, G=group, GP=gp, HQ=hq, HKV=hkv, PAGE=page, ROPE=bool(cfg["rope_theta"]),
                NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype, BL=bl, M=rows,
                BN=fc["bn"], BT=block_t, num_warps=fc["warps"], num_stages=fc["stages"],
            )
    elif window:
        # ---- sliding layers: ONE launch, the last arriving program of a (b, kv head) merges (see the header)
        sc = SLIDE_CFG["decode" if length == 1 else "verify"]
        bn = sc["bn"]
        split_len = max(sc["split"], _cdiv(_cdiv(span, SLIDE_MAX_SPLITS), bn) * bn)
        nsplit = _cdiv(span, split_len)
        part_o = torch.empty((batch, hkv, nsplit + 1, rows, 64), dtype=torch.float32, device=dev)
        part_m = torch.empty((batch, hkv, nsplit + 1, rows), dtype=torch.float32, device=dev)
        part_l = torch.empty_like(part_m)
        counter = torch.zeros((batch * hkv,), dtype=torch.int32, device=dev)
        _attn_fused_kernel[(nsplit + 1, hkv, batch)](
            qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
            data["k_scale"], data["v_scale"], part_o, part_m, part_l, counter, out,
            cfg["sm_scale"], cfg["qk_scale_factor"], theta,
            page_table.stride(0), split_len, nsplit,
            L=length, G=group, GP=gp, HQ=hq, HKV=hkv, PAGE=page, WINDOW=window, ROPE=bool(cfg["rope_theta"]),
            EDGE=length > 1, NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype,
            BL=bl, M=rows, BN=bn, BT=block_t,
            num_warps=4, num_stages=3,
        )
        return out
    else:
        # ---- window 0 with an odd number of KV heads: v3's two-launch path
        occ = 4 if rows <= 16 else 3
        split_len, nsplit = _plan(batch, hkv, span, rows, BLOCK_N, occ)
        part_o = torch.empty((batch, hkv, nsplit + 1, rows, 64), dtype=torch.float32, device=dev)
        part_m = torch.empty((batch, hkv, nsplit + 1, rows), dtype=torch.float32, device=dev)
        part_l = torch.empty_like(part_m)
        _attn_kernel[(nsplit + 1, hkv, batch)](
            qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, k_cache, v_cache,
            data["k_scale"], data["v_scale"], part_o, part_m, part_l,
            cfg["sm_scale"], cfg["qk_scale_factor"], theta,
            page_table.stride(0), split_len, nsplit,
            L=length, G=group, GP=gp, HQ=hq, HKV=hkv, PAGE=page, WINDOW=window, ROPE=bool(cfg["rope_theta"]),
            EDGE=False, NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype,
            BL=bl, M=rows, BN=BLOCK_N, BT=block_t,
            num_warps=4, num_stages=3 if split_len >= 3 * BLOCK_N else 1,
        )
    _reduce_kernel[(batch * length, hq)](
        part_o, part_m, part_l, qkv, data["seq_lens"], data["v_scale"], out, split_len, nsplit,
        L=length, G=group, GP=gp, HQ=hq, HKV=hkv, WINDOW=window, M=rows, ROWMAJOR=full_layer,
        BS=min(256, triton.next_power_of_2(nsplit + 1)), num_warps=8 if nsplit + 1 > 128 else 4, num_stages=1,
    )
    return out
