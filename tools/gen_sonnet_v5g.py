#!/usr/bin/env python3
"""Regenerate kernels/muse_decode_layer/sonnet_v5g.py (C38) = sonnet_v5f.py + F1 (the FFN boundary folded into the unpooler kernel, T <= 8, S10_FFN_FUSE) + the C33 TOOLS_LIGHT rider
(S11_TOOLS_LIGHT: PDL + <= 64 KB per CTA L2 prefetch before gdc_wait on our weight-streaming Triton kernels). Everything is added to v5f's text by exact-match substitutions that fail
loudly when v5f changes; the twin kernels come from tools/gen_measure_corridor_v1.py's twins() (renamed _mc_ -> _tl_).
Run after any change to an input:   .venv/bin/python tools/gen_sonnet_v5g.py"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
SRC = ROOT / "kernels" / "muse_decode_layer" / "sonnet_v5f.py"
OUT = ROOT / "kernels" / "muse_decode_layer" / "sonnet_v5g.py"

HEADER = '''# okc: problem=muse_decode_layer
# sonnet_v5g (C38) -- sonnet_v5f (board entry 1.433x) plus two switchable changes (module-level switches, shapes / hardware only; each off = sonnet_v5f's code path, bit for bit).
#   F1 (S10_FFN_FUSE)  T <= SLICED_UNPOOL_MAX_TOKENS (8): the FFN boundary (SGLang's muse_spark_norm_residual: post-FFN norm + residual gate + the next layer's pre-attention norm, 3.4-4.4 us +
#       a launch gap) folded into the unpooler kernel. At T <= 8 the unpooler's output f is zero beyond column 1023 (TASK.md), so rmsnorm(f * (1 + w)) = f_live * (1 + w) * g with g = rsqrt(
#       sum(f_live^2 (1 + w)^2) / 8192 + 1e-8), and the residual update s' = alpha * s + beta * rmsnorm(f (1 + w)) and the next norm h = rmsnorm(s') * (1 + w_next) need only sums over columns:
#       sum(s'^2) = S1 + 2 g S2 + g^2 S3 with S1 = sum((alpha s)^2) over all 8192 columns, S2 = sum(alpha beta s f w) and S3 = sum((beta f w)^2) over the live columns. Every column tile's last CTA
#       (the one that already holds the FP32 split-K total of its 16 columns) publishes the four per-token partials of its 16 live + 112 pad columns, takes an arrival ticket on a counter that is
#       zeroed INSIDE the timed call (a memset on the side stream, hidden behind the pooler), waits until all 64 tiles arrived (bounded spin: the grid is 128 CTAs, all co-resident), sums the 64
#       partials in tile order (every CTA the same bits), and writes s' (FP32) and h (BF16) of its own columns. f is never stored (no BF16 round of f, no 7168 zero columns, no boundary kernel).
#       Numerics: the FP32 chain of reference.py (ref_back) with ONE BF16 rounding (h); the board's f went through BF16 before the boundary. tests: max rel error vs the reference chain / vs v5f.
#   S11_TOOLS_LIGHT  the C33 TOOLS_LIGHT variant as a rider (judge job fcbc85a47127453a: +2.0 us at T = 1, TOOLS_FULL -3.9 us = corridor congestion): PDL on for every Triton kernel of the layer (S3's
#       gdc_wait / launch_dependents) and, for _gemv_splitk, _router_fix_kernel, _unpool_kernel (and F1's), _pre_norm_kernel and _epilogue_kernel, the _tl_ twins: prefetch.global.L2 of the first
#       <= 64 KB of the CTA's weight slice BEFORE gdc_wait (weights never depend on the previous grid), then wait for the activations. Math unchanged (bitwise v5f). Inert before sm_90.
#   S12_ROUTE_SIDE  (C37 ROUTE_SIDE, judge job f31c983903fd4930: +1.1 / +1.2 us at T = 1, +0.6 at T = 16): the small-M route kernel (smallm_route: global ids -> local ids + the expert route table) runs on the
#       SIDE stream right after the routing (it needs only topk_ids and local_start), behind the pooler, instead of on the main stream between pre_norm and GEMM1 (+ record_stream / join as before).
#       Same kernel, same arguments: bitwise v5f. Off = v5f's order.
# PREDICTION: per shape (us; sonnet_v5f judge check 62a4b559a7f14ddc, load formula of TASK.md reproduces its 1.4300; the board job efc7af96's per-shape table is not in the repo). Central values:
#     #0 111.2 -> ~104.7 (F1 -3.0, TOOLS_LIGHT -2.4, ROUTE_SIDE -1.1)   #1 117.4 -> ~112.4 (-3.0, -0.8, -1.2)   #2 181.2 -> ~180.6 (ROUTE_SIDE -0.6)   #3 275.1 -> ~274.5 (-0.6)
#     #6 172.2 -> ~167.2 (-3.0, -1.0, -1.0)   #7 177.1 -> ~171.6 (-3.0, -1.5, -1.0)   #4 #5 #8 #9 #10 #11 unchanged (T >= 32 and T = 128 / 256: none of the three changes is live; the rider measured ~0 at T >= 16).
#     F1 = the boundary's 3.4-4.4 us (+ ~0.4 us launch gap) minus ~1.5 us of barrier + tail inside the unpooler; riders = C33 TOOLS_LIGHT (measured d1 sliding / full +2.4 / +0.8, v1 full +1.5, v1 sliding assumed +1.0)
#     and C37 ROUTE_SIDE (measured +1.1 / +1.2 us at T = 1, +0.6 at T = 16; T = 8 assumed +1.0). The three are not additive for certain (they move the same critical path).
#   Loads (speedup vs SGLang, v5f -> v5g central): decode 1 1.554 -> 1.644, decode 16 1.503 -> 1.507, decode 32 1.298, verify 1 1.596 -> 1.644, verify 16 1.357, verify 32 1.303.
#   PROBLEM 1.430 -> ~1.451 (range 1.441 - 1.458: low = F1 -1.5 and half the riders, high = F1 -4.4 and the riders in full).
#   Kill: F1 does not meet the tolerance (tests print the max rel error: vs the FP32 reference chain hidden 1.7e-3 / stream 5e-7, vs the board's BF16-f chain stream 9e-4 / hidden 1.9e-3; the layer bound is >= 3 %), or the check
#   shows the T <= 8 shapes slower than v5f; switch order: F1 (S10_FFN_FUSE) first, then S12_ROUTE_SIDE, then S11_TOOLS_LIGHT.
#   Risks: F1's in-kernel barrier needs the 64 tile-last CTAs to be co-resident with the 128 CTAs of the grid (H200: 132 SMs x 3 CTAs of the 48 KB config = 396 slots; nothing else runs then); a bounded spin (2^19 polls)
#   turns a missing CTA into a wrong answer, never a hang, and prepare()'s self-check G5 (reference chain + two identical runs + the counter) switches F1 off if it ever happens; the three gains were measured separately
#   and may not add up; the Triton compile of the fused kernel on the judge is covered by the same self-check.
# --- the header below is sonnet_v5f's ---
'''

SWITCHES = '''S11_EPI_GRID = True       # G4 (H9): T <= EPI_GRID_MAX_TOKENS: the MoE epilogue on grid (tokens, EPI_PARTS), one 1024-column quarter per CTA (bitwise v5b's epilogue)
S10_FFN_FUSE = True       # C38 F1: T <= SLICED_UNPOOL_MAX_TOKENS: the FFN boundary (post-FFN norm + residual + next pre-attention norm) folded into the unpooler kernel; off = v5f's SGLang boundary
S12_ROUTE_SIDE = True      # C38 rider (C37): smallm_route on the side stream right after the routing (T <= 16, small-M experts live); off = v5f (route on the main stream, in the experts)
S11_TOOLS_LIGHT = True    # C38 rider (C33 TOOLS_LIGHT): PDL on our Triton kernels + a <= 64 KB per CTA L2 prefetch of their weights before gdc_wait; off = v5f (S3_PDL decides PDL)
'''

TOOLS_LIGHT_BYTES = 65536

SHIM = '''
# =============================================================================================== C38 rider: the _tl_ twins behind the original names
_TL_LIGHT_BYTES = 65536   # TOOLS_LIGHT: the first <= 64 KB per CTA (C33: the whole slice, TOOLS_FULL, was 3.9 us SLOWER at T = 1: corridor congestion)


@functools.lru_cache(maxsize=None)
def _tools_hw(index):
    """Hopper-or-newer with Triton's PDL API (griddepcontrol is sm_90+); hardware only."""
    return bool(HAVE_PDL_API and torch.cuda.is_available() and torch.cuda.get_device_capability(index)[0] >= 9)


def _tools_live():
    return bool(S11_TOOLS_LIGHT and _tools_hw(torch.cuda.current_device() if torch.cuda.is_available() else 0))


def _tl_pfl(name, a, kw):
    """Prefetch lines (128 bytes, a power of two) per weight row of one launch: the first <= _TL_LIGHT_BYTES of the CTA's slice."""
    def cdiv(x, y):
        return -(-x // y)
    if name == "_gemv_splitk":
        bn, bk, split, k = kw["BN"], kw["BK"], kw["SPLIT"], a[6]
        rows, rowl, nmat = bn, cdiv(cdiv(k, split * bk) * bk * 2, 128), 2 if kw["HAS_LO"] else 1
    elif name == "_router_fix_kernel":
        rows, rowl, nmat = kw["BN"], cdiv(kw["K"] // kw["SPLIT"] * 2, 128), 2
    elif name == "_unpool_kernel":
        rows, rowl, nmat = kw["BN"], cdiv(kw["K"] // kw["SPLIT"] * 2, 128), 1
    elif name == "_pre_norm_kernel":
        rows, rowl, nmat = 1, cdiv(kw["N"] * 2, 128), 1
    else:   # _epilogue_kernel: the gain vector
        rows, rowl, nmat = 1, cdiv(kw["H"] * 2, 128), 1
    return triton.next_power_of_2(max(1, min(rowl, max(1, _TL_LIGHT_BYTES // (rows * 128 * nmat)))))


class _ToolsShim:
    """Stands in for one of our Triton kernels: with S11_TOOLS_LIGHT live (Hopper) the launch goes to its _tl_ twin with PDL and the prefetch depth, else to the original unchanged."""

    def __init__(self, name, orig, twin):
        self.name, self.orig, self.twin = name, orig, twin

    def __getitem__(self, grid):
        orig, twin, name = self.orig[grid], self.twin[grid], self.name

        def launch(*a, **kw):
            if not _tools_live():
                return orig(*a, **kw)
            kw = dict(kw)
            kw.update(PDL=True, launch_pdl=True, PFL=_tl_pfl(name, a, kw))
            return twin(*a, **kw)
        return launch


_gemv_splitk = _ToolsShim("_gemv_splitk", _gemv_splitk, _tl_gemv_splitk)
_router_fix_kernel = _ToolsShim("_router_fix_kernel", _router_fix_kernel, _tl_router_fix_kernel)
_unpool_kernel = _ToolsShim("_unpool_kernel", _unpool_kernel, _tl_unpool_kernel)
_pre_norm_kernel = _ToolsShim("_pre_norm_kernel", _pre_norm_kernel, _tl_pre_norm_kernel)
_epilogue_kernel = _ToolsShim("_epilogue_kernel", _epilogue_kernel, _tl_epilogue_kernel)
'''

F1 = '''
# =============================================================================================== C38 F1: the FFN boundary inside the unpooler kernel
@triton.jit
def _ffn_tail(tot, rm, rn, rmask, pid_n, s_ptr, stride_s, wf_ptr, al_ptr, be_ptr, wn_ptr, h_ptr, so_ptr, fpart_ptr, fctr_ptr,
              BM: tl.constexpr, BN: tl.constexpr, N: tl.constexpr, NB: tl.constexpr, NCOL: tl.constexpr, PADF: tl.constexpr, PADP: tl.constexpr, NSPIN: tl.constexpr):
    """One column tile's share of the FFN boundary, run by the last CTA of the tile (tot: its FP32 [BM, BN] columns of f = the unpooler output). s' = alpha s + beta g (f w), h = rmsnorm(s') w_next,
    g = rsqrt(mean((f w)^2) + 1e-8); the means need every tile: four per-token partials per tile, one arrival ticket, a bounded wait, a fixed-order sum (all CTAs the same bits)."""
    wf = tl.load(wf_ptr + rn)
    al = tl.load(al_ptr + rn)
    be = tl.load(be_ptr + rn)
    wn = tl.load(wn_ptr + rn)
    sl = tl.load(s_ptr + rm[:, None] * stride_s + rn[None, :], mask=rmask[:, None], other=0.0)
    fw = tot * wf[None, :]
    a_s = al[None, :] * sl
    bf = be[None, :] * fw
    pi = tl.arange(0, PADP)
    pc = N + pid_n * PADF + pi                       # the pad columns this tile owns: s' = alpha s there (f is zero)
    pm = (pi < PADF) & (pc < NCOL)
    alp = tl.load(al_ptr + pc, mask=pm, other=0.0)
    wnp = tl.load(wn_ptr + pc, mask=pm, other=0.0)
    sp = tl.load(s_ptr + rm[:, None] * stride_s + pc[None, :], mask=rmask[:, None] & pm[None, :], other=0.0)
    ap = alp[None, :] * sp
    pb = fpart_ptr + pid_n * (4 * BM)
    tl.store(pb + rm, tl.sum(fw * fw, axis=1))
    tl.store(pb + BM + rm, tl.sum(a_s * a_s, axis=1) + tl.sum(ap * ap, axis=1))
    tl.store(pb + 2 * BM + rm, tl.sum(a_s * bf, axis=1))
    tl.store(pb + 3 * BM + rm, tl.sum(bf * bf, axis=1))
    tl.debug_barrier()
    arrive = tl.atomic_add(fctr_ptr, 1, sem="acq_rel", scope="gpu")
    cnt = arrive + 1
    spins = 0
    while (cnt < NB) & (spins < NSPIN):
        cnt = tl.atomic_add(fctr_ptr, 0, sem="acquire", scope="gpu")
        spins += 1
    tl.debug_barrier()
    ti = tl.arange(0, NB)
    pidx = ti[:, None] * (4 * BM) + rm[None, :]
    c = tl.sum(tl.load(fpart_ptr + pidx, cache_modifier=".cg"), axis=0)
    s1 = tl.sum(tl.load(fpart_ptr + pidx + BM, cache_modifier=".cg"), axis=0)
    s2 = tl.sum(tl.load(fpart_ptr + pidx + 2 * BM, cache_modifier=".cg"), axis=0)
    s3 = tl.sum(tl.load(fpart_ptr + pidx + 3 * BM, cache_modifier=".cg"), axis=0)
    g = tl.rsqrt(c / NCOL + 1e-8)
    inv = tl.rsqrt((s1 + 2.0 * g * s2 + g * g * s3) / NCOL + 1e-5)
    sn = a_s + bf * g[:, None]
    tl.store(so_ptr + rm[:, None] * NCOL + rn[None, :], sn, mask=rmask[:, None])
    tl.store(h_ptr + rm[:, None] * NCOL + rn[None, :], (sn * inv[:, None] * wn[None, :]).to(tl.bfloat16), mask=rmask[:, None])
    tl.store(so_ptr + rm[:, None] * NCOL + pc[None, :], ap, mask=rmask[:, None] & pm[None, :])
    tl.store(h_ptr + rm[:, None] * NCOL + pc[None, :], (ap * inv[:, None] * wnp[None, :]).to(tl.bfloat16), mask=rmask[:, None] & pm[None, :])


@triton.jit(do_not_specialize=["M"])
def _unpool_ffn_kernel(x_ptr, w_ptr, s_ptr, wf_ptr, al_ptr, be_ptr, wn_ptr, h_ptr, so_ptr, part_ptr, done_ptr, fpart_ptr, fctr_ptr, M, stride_x, stride_s,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, K: tl.constexpr, N: tl.constexpr, NCOL: tl.constexpr,
                       PADF: tl.constexpr, PADP: tl.constexpr, NSPIN: tl.constexpr, PDL: tl.constexpr = False, PFL: tl.constexpr = 0):
    """_unpool_kernel's GEMV (grid (N / BN, SPLIT), split-K partials, last CTA of a tile sums them in slice order) whose last CTAs finish the layer: instead of storing f they write the new residual
    stream (FP32) and the next layer's attention input (BF16), see _ffn_tail. Requires M <= BM and N / BN a power of two."""
    if PFL > 0:
        _rn0 = tl.program_id(0) * BN + tl.arange(0, BN)
        _ko = tl.program_id(1) * (K // SPLIT) + tl.arange(0, PFL)[None, :] * 64
        _tl_pf(w_ptr + _rn0[:, None] * K + _ko)
    if PDL:
        tl.extra.cuda.gdc_wait()
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    NB: tl.constexpr = N // BN
    rm = tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rmask = rm < M
    k_per: tl.constexpr = K // SPLIT
    k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        rk = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * stride_x + rk[None, :], mask=rmask[:, None], other=0.0)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :])
        acc += tl.dot(x, tl.trans(w))
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    if SPLIT == 1:
        _ffn_tail(acc, rm, rn, rmask, pid_n, s_ptr, stride_s, wf_ptr, al_ptr, be_ptr, wn_ptr, h_ptr, so_ptr, fpart_ptr, fctr_ptr, BM, BN, N, NB, NCOL, PADF, PADP, NSPIN)
    else:
        tl.store(part_ptr + (pid_k * M + rm)[:, None] * N + rn[None, :], acc, mask=rmask[:, None])
        tl.debug_barrier()
        ticket = tl.atomic_add(done_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if ticket == SPLIT - 1:
            tot = tl.zeros((BM, BN), dtype=tl.float32)
            for s in range(SPLIT):  # .cg: skip L1 so no stale line of this buffer can be read
                tot += tl.load(part_ptr + (s * M + rm)[:, None] * N + rn[None, :], mask=rmask[:, None], other=0.0, cache_modifier=".cg")
            tl.store(done_ptr + pid_n, 0)
            _ffn_tail(tot, rm, rn, rmask, pid_n, s_ptr, stride_s, wf_ptr, al_ptr, be_ptr, wn_ptr, h_ptr, so_ptr, fpart_ptr, fctr_ptr, BM, BN, N, NB, NCOL, PADF, PADP, NSPIN)


FFN_SPIN = 1 << 19        # bounded wait of the in-kernel arrival barrier (each poll ~ 1 us): a CTA that never arrives gives a wrong answer, not a hang


def _f1_state(weights, device):
    """prepare(): the per-channel vectors of the FFN boundary (FP32, from the weights: 1 + post_ffn_norm, the residual gate's alpha / beta, 1 + next_pre_attn_norm) and the scratch of the
    in-kernel barrier (per-tile partials; the arrival counter is zeroed inside every call, see kernel_fn)."""
    alpha, beta = _residual_coefficients(weights["ffn_residual_gate"])
    return {"wf": (1.0 + weights["post_ffn_norm"].float()).contiguous(), "alpha": alpha.float().contiguous(), "beta": beta.float().contiguous(),
            "wn": (1.0 + weights["next_pre_attn_norm"].float()).contiguous(),
            "part": torch.zeros((UNPOOL_SLICE // 16, 4, 16), dtype=torch.float32, device=device), "ctr": torch.zeros(1, dtype=torch.int32, device=device)}


def _ffn_fuse_live(tokens, device, state, stream=None):
    """F1 is live for 1..SLICED_UNPOOL_MAX_TOKENS tokens wherever G3's unpooler kernel is live (shapes / config / dtype only)."""
    bn = _unpool_cfg(state)[0]
    nb = UNPOOL_SLICE // bn
    return bool(S10_FFN_FUSE and _unpool_live(tokens, device) and "f1" in state and "unpool_part" in state and (nb & (nb - 1)) == 0
                and (stream is None or (stream.dtype == torch.float32 and stream.dim() == 2 and stream.stride(-1) == 1)))


def unpool_ffn(y, stream, state, cfg=None):
    """(next attention input [T, 8192] BF16, new residual stream [T, 8192] FP32) = the unpooler slice GEMV and the whole FFN boundary in ONE launch. y [T, 4096], stream [T, 8192] FP32."""
    bn, bk, split, warps, stages = cfg or _unpool_cfg(state)
    y = y.contiguous()
    t = y.shape[0]
    f1 = state["f1"]
    assert t <= 16 and split <= state["unpool_part"].shape[0] and t <= state["unpool_part"].shape[1]
    h = torch.empty((t, HIDDEN), dtype=torch.bfloat16, device=y.device)
    so = torch.empty((t, HIDDEN), dtype=torch.float32, device=y.device)
    nb = UNPOOL_SLICE // bn
    padf = triton.cdiv(HIDDEN - UNPOOL_SLICE, nb)
    kw = {}
    if _tools_live():
        kw = dict(PDL=True, launch_pdl=True, PFL=_tl_pfl("_unpool_kernel", (), dict(BN=bn, SPLIT=split, K=D_EXPERT)))
    _unpool_ffn_kernel[(nb, split)](
        y, state["dense"]["unpooler"], stream, f1["wf"], f1["alpha"], f1["beta"], f1["wn"], h, so, state["unpool_part"], state["unpool_done"], f1["part"], f1["ctr"],
        t, y.stride(0), stream.stride(0), BM=16, BN=bn, BK=bk, SPLIT=split, K=D_EXPERT, N=UNPOOL_SLICE, NCOL=HIDDEN, PADF=padf, PADP=triton.next_power_of_2(padf),
        NSPIN=FFN_SPIN, num_warps=warps, num_stages=stages, **kw)
    return h, so
'''

SELFCHECK = '''    try:
        if S10_FFN_FUSE and _unpool_live(8, dev) and "f1" in state:   # G5 (F1): the fused boundary against the FP32 reference chain (reference.py ref_back) and against itself (deterministic)
            y = (0.5 * torch.randn((8, D_EXPERT), generator=g, device=dev)).to(torch.bfloat16)
            s = torch.randn((8, HIDDEN), generator=g, device=dev)
            f1 = state["f1"]
            f1["ctr"].zero_()
            h1, s1 = unpool_ffn(y, s, state)
            f1["ctr"].zero_()
            h2, s2 = unpool_ffn(y, s, state)
            f = F.pad((y.float() @ state["dense"]["unpooler"][:UNPOOL_SLICE].float().T), (0, HIDDEN - UNPOOL_SLICE)) * f1["wf"]
            sn = f1["alpha"] * s + f1["beta"] * _rms(f, POST_NORM_EPS)
            hn = _rms(sn) * f1["wn"]
            torch.cuda.synchronize()
            e_s = float((s1 - sn).norm() / sn.norm())
            e_h = float((h1.float() - hn).norm() / hn.norm())
            ok = e_s < 1e-3 and e_h < 1e-2 and bool(torch.equal(h1, h2)) and bool(torch.equal(s1, s2)) and int(f1["ctr"]) == UNPOOL_SLICE // _unpool_cfg(state)[0] and not int(state["unpool_done"].abs().sum())
            print(f"okc-diag G5 FFN fusion rel L2: stream {e_s:.2e} hidden {e_h:.2e} deterministic={bool(torch.equal(h1, h2))} ok={ok}")
            S10_FFN_FUSE = ok
    except Exception as ex:  # noqa: BLE001
        print(f"okc-diag G5 selfcheck raised {type(ex).__name__}: {str(ex)[:200]}")
        S10_FFN_FUSE = False
    try:
        if S12_ROUTE_SIDE and state.get("smallm") is not None:   # G6 (ROUTE_SIDE): the experts with the route done up front are bitwise the experts that route themselves
            x, w, ids = _smallm_selfcheck_inputs(8, 3008, dev)
            start = torch.zeros(1, dtype=torch.int32, device=dev)
            a = _experts_smallm(x, w, ids, start, state)
            b = _experts_smallm(x, w, ids, start, state, pre=smallm_route_side(ids, start, state))
            torch.cuda.synchronize()
            same = bool(torch.equal(a, b))
            print(f"okc-diag G6 route-side bitwise={same}")
            S12_ROUTE_SIDE = same
    except Exception as ex:  # noqa: BLE001
        print(f"okc-diag G6 selfcheck raised {type(ex).__name__}: {str(ex)[:200]}")
        S12_ROUTE_SIDE = False
    print(f"okc-diag glue switches: G2={S9_PRE_NORM} G3={S10_UNPOOL_GEMV} G4={S11_EPI_GRID} G5={S10_FFN_FUSE} G6={S12_ROUTE_SIDE} rider={S11_TOOLS_LIGHT}")
'''


def _sub(text, old, new, count=1):
    n = text.count(old)
    if n != count:
        raise SystemExit(f"gen_sonnet_v5g: expected {count} x {old[:80]!r} in sonnet_v5f.py, found {n}: v5f changed, update the substitutions")
    return text.replace(old, new)


def twins_block():
    import gen_measure_corridor_v1 as g
    t = g.twins()
    assert "_mc_" in t
    t = t.replace("_mc_", "_tl_")
    assert "_mc" not in t
    return "\n\n# =============================================================================================== C38 rider: twins of our weight-streaming Triton kernels (prefetch before gdc_wait)\n" + t + "\n"


def build():
    s = SRC.read_text()
    assert s.startswith("# okc: problem=muse_decode_layer\n")
    s = HEADER + s[len("# okc: problem=muse_decode_layer\n"):]
    # switches + PDL gate
    s = _sub(s, "S11_EPI_GRID = True       # G4 (H9): T <= EPI_GRID_MAX_TOKENS: the MoE epilogue on grid (tokens, EPI_PARTS), one 1024-column quarter per CTA (bitwise v5b's epilogue)\n", SWITCHES)
    s = _sub(s, "    return bool(S3_PDL and HAVE_PDL_API and device.type == \"cuda\" and torch.cuda.get_device_capability(device)[0] >= 9)",
             "    return bool((S3_PDL or S11_TOOLS_LIGHT) and HAVE_PDL_API and device.type == \"cuda\" and torch.cuda.get_device_capability(device)[0] >= 9)")
    # state: the F1 vectors and scratch
    s = _sub(s, "    del freqs\n    torch.cuda.synchronize()\n    return state\n",
             "    state[\"f1\"] = _f1_state(weights, device)   # C38 F1\n    del freqs\n    torch.cuda.synchronize()\n    return state\n")
    # self-check
    s = _sub(s, "    global S9_PRE_NORM, S10_UNPOOL_GEMV, S11_EPI_GRID\n", "    global S9_PRE_NORM, S10_UNPOOL_GEMV, S11_EPI_GRID, S10_FFN_FUSE, S12_ROUTE_SIDE\n")
    s = _sub(s, "    print(f\"okc-diag glue switches: G2={S9_PRE_NORM} G3={S10_UNPOOL_GEMV} G4={S11_EPI_GRID}\")\n", SELFCHECK)
    # twins + shim + F1 kernel, before the entry points
    marker = "def prepare(weights):\n"
    i = s.index(marker)
    assert s.count(marker) == 1
    s = s[:i] + twins_block() + SHIM + F1 + "\n\n" + s[i:]
    # kernel_fn: the zeroed arrival counter on the side stream, the fused boundary
    s = _sub(s, "    with torch.cuda.stream(side_stream):\n        if f2 is not None:",
             "    fuse = _ffn_fuse_live(tokens, device, state, stream)\n    with torch.cuda.stream(side_stream):\n        if fuse:\n            state[\"f1\"][\"ctr\"].zero_()  # F1's arrival counter, zeroed inside the timed call (side stream: hidden behind the pooler)\n        if f2 is not None:")
    old_tail = ("    if _unpool_live(tokens, device) and \"unpool_part\" in state:  # G3: slice GEMV + zero pad in one launch\n"
                "        ffn_output = unpool_sliced(expert_output, state)  # all-gather: identity\n")
    new_tail = ("    if fuse:  # C38 F1: slice GEMV + the whole FFN boundary in one launch (T <= 8)\n"
                "        next_attn_input, stream = unpool_ffn(expert_output, stream, state)  # all-gather: identity\n"
                "        return next_attn_input, stream, topk_ids\n"
                "    if _unpool_live(tokens, device) and \"unpool_part\" in state:  # G3: slice GEMV + zero pad in one launch\n"
                "        ffn_output = unpool_sliced(expert_output, state)  # all-gather: identity\n")
    s = _sub(s, old_tail, new_tail)
    # ---- S12 ROUTE_SIDE (C37): smallm_route on the side stream right after the routing
    s = _sub(s, "def smallm_experts(x, topk_ids, sm, start, count, cfg):", "def smallm_experts(x, topk_ids, sm, start, count, cfg, pre=None):")
    s = _sub(s, "    ids = topk_ids.to(torch.int32).contiguous()\n    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)\n    ext.smallm_route(ids, local_ids, sm[\"tab\"], start, count)\n",
             "    if pre is None:\n        ids = topk_ids.to(torch.int32).contiguous()\n        local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)\n"
             "        ext.smallm_route(ids, local_ids, sm[\"tab\"], start, count)\n    else:  # C38 S12: the route table and the local ids were built on the side stream\n        local_ids = pre\n")
    s = _sub(s, "def _experts_smallm(x, topk_weights, topk_ids, local_start, state):", "def _experts_smallm(x, topk_weights, topk_ids, local_start, state, pre=None):")
    s = _sub(s, "    y, local_ids = smallm_experts(x.contiguous(), topk_ids, state[\"smallm\"], local_start, LOCAL_EXPERTS, smallm_cfg(m))\n",
             "    y, local_ids = smallm_experts(x.contiguous(), topk_ids, state[\"smallm\"], local_start, LOCAL_EXPERTS, smallm_cfg(m), pre=pre)\n")
    s = _sub(s, "def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None):", "def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None, pre=None):")
    s = _sub(s, "        return _experts_smallm(x, topk_weights, topk_ids, local_start, state)\n    if local_ids is None and _largem_live",
             "        return _experts_smallm(x, topk_weights, topk_ids, local_start, state, pre=pre)\n    if local_ids is None and _largem_live")
    s = _sub(s, "    local_ids = None\n    fuse = _ffn_fuse_live(", "    local_ids = None\n    pre = None   # C38 S12: smallm_route's local ids, built on the side stream\n    fuse = _ffn_fuse_live(")
    s = _sub(s, "            topk_weights, topk_ids = state[\"routing\"](None, logits, TOP_K, True)\n",
             "            topk_weights, topk_ids = state[\"routing\"](None, logits, TOP_K, True)\n"
             "        if S12_ROUTE_SIDE and local_ids is None and _smallm_live(tokens, state):  # C38 S12: smallm_route needs only topk_ids + local_start: behind the pooler, off the main stream\n"
             "            pre = smallm_route_side(topk_ids, data[\"local_start\"], state)\n")
    s = _sub(s, "        for tensor in (topk_weights, topk_ids) + ((local_ids,) if local_ids is not None else ()):\n",
             "        for tensor in (topk_weights, topk_ids) + ((local_ids,) if local_ids is not None else ()) + ((pre,) if pre is not None else ()):\n")
    s = _sub(s, "    expert_output = _experts(expert_input if normed else _pre_expert(expert_input, state), topk_weights, topk_ids, data[\"local_start\"], state,\n                             local_ids=local_ids)\n",
             "    expert_output = _experts(expert_input if normed else _pre_expert(expert_input, state), topk_weights, topk_ids, data[\"local_start\"], state,\n                             local_ids=local_ids, pre=pre)\n")
    # smallm_route_side (placed with the F1 block)
    s = _sub(s, "\n# =============================================================================================== C38 F1: the FFN boundary inside the unpooler kernel\n",
             "\n# =============================================================================================== C38 S12: smallm_route on the side stream\ndef smallm_route_side(topk_ids, start, state):\n"
             "    \"\"\"The small-M route kernel on its own (it needs only topk_ids and local_start): local ids [T, 8] int32; the route table lands in the small-M state's tab.\"\"\"\n"
             "    sm = state[\"smallm\"]\n    ids = topk_ids.to(torch.int32).contiguous()\n    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)\n"
             "    sm[\"ext\"].smallm_route(ids, local_ids, sm[\"tab\"], start, LOCAL_EXPERTS)\n    return local_ids\n"
             "\n\n# =============================================================================================== C38 F1: the FFN boundary inside the unpooler kernel\n")
    return s


if __name__ == "__main__":
    text = build()
    compile(text, str(OUT), "exec")
    OUT.write_text(text)
    print("wrote", OUT, len(text), "bytes")
