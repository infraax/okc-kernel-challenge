# okc: problem=muse_bf16_proj
"""MEASUREMENT-ONLY candidate (card C9, v3; design/fable-06-out.md D1 + D2). Run as `okc check`, NEVER submit.

kernel_fn is SGLang's own baseline (scores ~1.00x). prepare() runs `_measure(weights)` once, untimed, within a 200 s wall
budget: it times Triton kernels per scored shape on the judge's H200, on synthetic activations (nothing derived from
the judge's inputs), each as a CUDA-graph replay after a READ-ONLY L2 flush (like the judge), then returns the plain
baseline state. Changes vs v2: (1) `do_not_specialize` on M/N/K/SPLIT/strides, SPLIT is a runtime argument -> one
compile per constexpr config serves every m and every shape (v2 spent most of its 200 s recompiling); (2) 11-rep screen,
51-rep finals for the top 5 per shape; (3) compact log: one human `B` line + one `T64` block per shape.

Experiments (all 30 scored shapes; the per-shape config list is `shape_cfgs(layer, m)`):
  E1 (fam A, B)  T1 single-launch streaming kernel, SPLIT=1, direct store, m <= 32 (20 shapes): masked 16-B loads with
                 alignment hints (A) vs TMA tensor descriptor on W (B; if TMA is unavailable it is recorded as FAIL).
  E2 (fam F, Z)  T2 split-K: in-kernel fixup (workspace + per-tile counter from prepare(); the last CTA sums the
                 partials in split order, writes the output and resets its counter) vs zero + atomic (+ fp32->bf16 copy):
                 router at every m, qkv_gate m=1, unpooler m <= 8.
  E3 (fam G, H)  T4 full-M split-K, BM=128, BN 64/128/256, SPLIT 2/4/8 (fixup G vs zero+atomic H): m=128 and m=256 for
                 qkv_gate, o, pooler, unpooler (cuBLAS measured the same way is the `base`).
Every variant is checked against an fp64 reference (rel L2 < 1e-2, router < 1e-4) before it is timed; wrong ones are
counted as `bad`, not timed.

Phases (absolute deadlines in PHASE_END_S): setup (30 fp64 references + baselines), A, G/H, F/Z, finals of those, TMA smoke
test, B, finals of B. The TMA family runs last so a device fault there cannot cost the other results.

Log (tail, < 15 KB): `CFG`/`CFGSIG` header + `MEAS3 phase` progress lines, one `T64 <layer>.m<m> <base64>` per shape (uint16 little endian, µs x 100, in
`shape_cfgs` order; 0 = not run, 65535 = fail, 65534 = bad, 65533 = clamped (>= 655.33 us); finals override the screen value for the top 5), then one
`B <layer> m=<m> base=<us> | top-5 <cfg>:<us> ... | ok= fail= bad= [skip=]` line per shape. Decode with
tools/parse_meas.py (it re-reads the config table from this file; CFGSIG must match).
"""

import base64
import itertools
import struct
import time
import zlib
from typing import TypedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:  # in the judge's image
    from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl
except Exception:  # local dry runs
    def _bf16_gemm_dispatch_impl(x, w, b):
        return x @ w.T


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


ROUTER_FP32_MAX_TOKENS = 8
BUDGET_S = 200.0        # prepare() may run 300 s; the whole run has 10 min
FINAL_RESERVE_S = 14.0  # finals (51 reps x top 5 x 30 shapes) + report
# absolute end of each phase (s since _measure started); E3 (few compile keys, the m=128 load) runs before E2
PHASE_END_S = {"A": 80.0, "GH": 100.0, "FZ": 135.0, "B": BUDGET_S - FINAL_RESERVE_S}
SCREEN_REPS = 11
FINAL_REPS = 51
FLUSH_MB = 128          # read-only L2 flush buffer (> the 50 MB L2)
WS_FLOATS = 20 * 1024 * 1024   # split-K workspace (fp32) allocated in prepare()
SEM_COUNT = 4096               # per-tile counters (int32), zero between launches

# ---- CFG BEGIN (pure python: tools/parse_meas.py executes this block; keep it stdlib-only) ----
import itertools
import zlib

LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
MS = (1, 8, 16, 32, 128, 256)
LAYER_NK = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
            "router": (256, 8192)}
SMEM_BUDGET = 200 * 1024
FAMILIES = {"A": "T1 masked", "B": "T1 TMA", "F": "T2 fixup", "Z": "T2 zero+atomic", "G": "T4 fixup",
            "H": "T4 zero+atomic"}


def shape_nk(layer, m):
    n, k = LAYER_NK[layer]
    if layer == "unpooler" and m <= 8:
        n = 1024
    return n, k


def _fits(bm, bn, bk, lo, stages):
    return stages * 2 * bk * (bm + bn * (2 if lo else 1)) <= SMEM_BUDGET


def shape_cfgs(layer, m):
    """Ordered configs of one shape: (name, fam, BM, BN, BK, SPLIT, warps, stages). name = fam+BM.BN.BK.SPLIT.w.s"""
    n, k = shape_nk(layer, m)
    lo = layer == "router"
    out = []

    def add(fam, bm, bn, bk, split, w, s):
        out.append((f"{fam}{bm}.{bn}.{bk}.{split}.{w}.{s}", fam, bm, bn, bk, split, w, s))

    if m <= 32:  # E1: the streaming kernel, SPLIT=1
        bms = (16,) if m <= 16 else (16, 32)
        for fam in ("A", "B"):
            for bm in bms:
                for bn in ((8, 16) if lo else (8, 16, 32, 64)):
                    for bk in (512, 1024, 2048):
                        if bk > k or k % bk:
                            continue
                        for w, s in itertools.product((4, 8), (3, 4, 5, 6)):
                            if _fits(bm, bn, bk, lo, s):
                                add(fam, bm, bn, bk, 1, w, s)
    if lo or (layer == "qkv_gate" and m == 1) or (layer == "unpooler" and m <= 8):  # E2
        bm = 16 if m <= 16 else (32 if m <= 32 else 64)
        for fam in ("F", "Z"):
            for bn, bk, (w, s), split in itertools.product((16, 32, 64), (64, 128, 256), ((4, 3), (4, 4)),
                                                           (8, 16) if lo else (4, 8, 16)):
                if split * bk <= k and _fits(bm, bn, bk, lo, s):
                    add(fam, bm, bn, bk, split, w, s)
    if m >= 128 and not lo:  # E3
        for fam in ("G", "H"):
            for bn, bk, (w, s), split in itertools.product((64, 128, 256), (64, 128), ((8, 3), (8, 4)), (2, 4, 8)):
                if split * bk <= k and _fits(128, bn, bk, False, s):
                    add(fam, 128, bn, bk, split, w, s)
    return out


def all_shapes():
    return [(layer, m) for m in MS for layer in LAYERS]


def shape_id(layer, m):
    return f"{layer}.m{m}"


def cfg_sig():
    names = [c[0] for layer, m in all_shapes() for c in shape_cfgs(layer, m)]
    return len(names), zlib.crc32("\n".join(names).encode()) & 0xFFFFFFFF
# ---- CFG END ----

_ST_FAIL, _ST_BAD = 65535, 65534
_BF = torch.bfloat16  # tests on the sm_75 dev card swap in float16


def _encode_t64(vals):
    """vals: per config None (not run) | float us | 'F' | 'B' -> base64 of uint16 LE (µs x 100)."""
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


# ----------------------------------------------------------------------------------------------------------------
# Kernels. M/N/K/SPLIT/strides are NOT specialised (no recompile per m); alignment is promised via the ALIGNED hint.

@triton.jit(do_not_specialize=["M", "N", "K", "stride_xm", "stride_wn", "stride_om"])
def _k_stream(x_ptr, w_ptr, wlo_ptr, out_ptr, M, N, K, stride_xm, stride_wn, stride_om,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_LO: tl.constexpr, TMA: tl.constexpr,
              EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, ALIGNED: tl.constexpr):
    """T1: out[m, n] = x @ W[n].T (+ lo), one CTA per (BN columns, BM rows), whole K in the loop, direct store."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    xo = rm[:, None] * stride_xm
    wo = rn[:, None] * stride_wn
    if ALIGNED:  # strides are multiples of 16 elements: 16-B (fp16/bf16: 32-B) vector loads
        xo = tl.multiple_of(xo, [16, 1])
        wo = tl.multiple_of(wo, [16, 1])
    mm = (rm < M)[:, None]
    nn = (rn < N)[:, None]
    if TMA:
        w_desc = tl.make_tensor_descriptor(w_ptr, shape=[N, K], strides=[stride_wn, 1], block_shape=[BN, BK])
        if HAS_LO:
            wl_desc = tl.make_tensor_descriptor(wlo_ptr, shape=[N, K], strides=[stride_wn, 1], block_shape=[BN, BK])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = tl.max_contiguous(tl.multiple_of(k0 + tl.arange(0, BK), BK), BK)
        if EVEN_K:
            x = tl.load(x_ptr + xo + rk[None, :], mask=mm, other=0.0)
        else:
            x = tl.load(x_ptr + xo + rk[None, :], mask=mm & (rk < K)[None, :], other=0.0)
        if TMA:
            w = w_desc.load([pid_n * BN, k0])
        elif EVEN_K and EVEN_N:
            w = tl.load(w_ptr + wo + rk[None, :])
        else:
            w = tl.load(w_ptr + wo + rk[None, :], mask=nn & (rk < K)[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        if HAS_LO:
            if TMA:
                wl = wl_desc.load([pid_n * BN, k0])
            elif EVEN_K and EVEN_N:
                wl = tl.load(wlo_ptr + wo + rk[None, :])
            else:
                wl = tl.load(wlo_ptr + wo + rk[None, :], mask=nn & (rk < K)[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(wl))
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty), mask=mm & (rn < N)[None, :])


@triton.jit(do_not_specialize=["M", "N", "K", "SPLIT", "stride_xm", "stride_wn", "stride_om"])
def _k_splitk(x_ptr, w_ptr, wlo_ptr, out_ptr, ws_ptr, sem_ptr, M, N, K, SPLIT, stride_xm, stride_wn, stride_om,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_LO: tl.constexpr, FIXUP: tl.constexpr,
              EVEN_K: tl.constexpr, EVEN_N: tl.constexpr, ALIGNED: tl.constexpr):
    """T2/T4: split-K over grid axis 1 (SPLIT is a runtime value). FIXUP=0: atomic_add into a zeroed fp32 buffer.
    FIXUP=1: every CTA stores its fp32 partial into the workspace, bumps a per-tile counter; the CTA that arrives last
    sums the partials in split order, writes `out` (any dtype) and resets the counter (no memset node needed)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    xo = rm[:, None] * stride_xm
    wo = rn[:, None] * stride_wn
    if ALIGNED:
        xo = tl.multiple_of(xo, [16, 1])
        wo = tl.multiple_of(wo, [16, 1])
    mm = (rm < M)[:, None]
    nn = (rn < N)[:, None]
    k_per = tl.cdiv(K, SPLIT * BK) * BK
    k_beg = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        rk = tl.max_contiguous(tl.multiple_of(k_beg + kk + tl.arange(0, BK), BK), BK)
        if EVEN_K:
            x = tl.load(x_ptr + xo + rk[None, :], mask=mm, other=0.0)
        else:
            x = tl.load(x_ptr + xo + rk[None, :], mask=mm & (rk < K)[None, :], other=0.0)
        if EVEN_K and EVEN_N:
            w = tl.load(w_ptr + wo + rk[None, :])
        else:
            w = tl.load(w_ptr + wo + rk[None, :], mask=nn & (rk < K)[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        if HAS_LO:
            if EVEN_K and EVEN_N:
                wl = tl.load(wlo_ptr + wo + rk[None, :])
            else:
                wl = tl.load(wlo_ptr + wo + rk[None, :], mask=nn & (rk < K)[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(wl))
    om = mm & (rn < N)[None, :]
    optr = out_ptr + rm[:, None] * stride_om + rn[None, :]
    if FIXUP:
        tile = pid_m * tl.num_programs(0) + pid_n
        base = (tile * SPLIT).to(tl.int64) * (BM * BN)
        loc = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.store(ws_ptr + base + pid_k * (BM * BN) + loc, acc)
        tl.debug_barrier()  # every thread's partial is stored before thread 0's release-atomic
        arrived = tl.atomic_add(sem_ptr + tile, 1)  # acq_rel, gpu scope
        if arrived == SPLIT - 1:  # last CTA of this tile
            total = tl.zeros((BM, BN), dtype=tl.float32)
            for s in range(0, SPLIT):
                total += tl.load(ws_ptr + base + s * (BM * BN) + loc, cache_modifier=".cg")
            tl.store(optr, total.to(out_ptr.dtype.element_ty), mask=om)
            tl.store(sem_ptr + tile, 0)  # ready for the next replay
    else:
        tl.atomic_add(optr, acc, mask=om)


# ----------------------------------------------------------------------------------------------------------------
# Host side

class _Unfit(Exception):
    """This config cannot run on this shape (workspace too small ...): a fail of the pair, not of the compile key."""


def _make_run(fam, bm, bn, bk, split, w_, s_, w, wlo, n, odt, ws=None, sem=None):
    """Callable run(x) -> out computing x @ w[:n].T (+ wlo) with one config. Allocations happen here, not per call.
    Stream-order safe: the fixup counters are zero on entry and are reset by the kernel."""
    k = w.shape[1]
    lo = wlo is not None
    dev = w.device
    aligned = (w.stride(0) % 16 == 0) and (k % 16 == 0)

    if fam in ("A", "B"):
        tma = fam == "B"

        def run(x):
            m = x.shape[0]
            o = run.buf
            if o is None or o.shape[0] != m:
                o = run.buf = torch.empty((m, n), device=dev, dtype=odt)
            _k_stream[(triton.cdiv(n, bn), triton.cdiv(m, bm))](
                x, w, wlo if lo else w, o, m, n, k, x.stride(0), w.stride(0), o.stride(0),
                BM=bm, BN=bn, BK=bk, HAS_LO=lo, TMA=tma, EVEN_K=(k % bk == 0), EVEN_N=(n % bn == 0),
                ALIGNED=aligned and x.stride(0) % 16 == 0, num_warps=w_, num_stages=s_)
            return o
        run.buf = None
        return run

    fixup = fam in ("F", "G")
    even_k = k % (split * bk) == 0
    even_n = n % bn == 0

    def run(x):
        m = x.shape[0]
        tiles_m, tiles_n = triton.cdiv(m, bm), triton.cdiv(n, bn)
        if run.buf is None or run.buf[0].shape[0] != m:
            acc = torch.empty((m, n), device=dev, dtype=torch.float32)
            out = acc if (not fixup and odt == torch.float32) else torch.empty((m, n), device=dev, dtype=odt)
            if fixup:
                if tiles_m * bm * tiles_n * bn * split > ws.numel() or tiles_m * tiles_n > sem.numel():
                    raise _Unfit("workspace")
                sem[: tiles_m * tiles_n].zero_()
            run.buf = (acc, out)
        acc, out = run.buf
        if fixup:
            tgt, wsp, smp = out, ws, sem
        else:
            acc.zero_()
            tgt, wsp, smp = acc, w, w  # unused pointers
        _k_splitk[(tiles_n, split, tiles_m)](
            x, w, wlo if lo else w, tgt, wsp, smp, m, n, k, split, x.stride(0), w.stride(0), tgt.stride(0),
            BM=bm, BN=bn, BK=bk, HAS_LO=lo, FIXUP=fixup, EVEN_K=even_k, EVEN_N=even_n,
            ALIGNED=aligned and x.stride(0) % 16 == 0, num_warps=w_, num_stages=s_)
        if not fixup and out is not acc:
            out.copy_(acc)
        return out
    run.buf = None
    return run


class _RealEnv:
    """The GPU side: synthetic inputs, fp64 references, baselines, CUDA-graph timing after a read-only flush."""

    def __init__(self, weights, scratch):
        self.weights = weights
        self.scratch = scratch
        self.dev = weights["qkv_gate"].device
        self.tma_msg = ""

    # --- lifecycle
    def available(self):
        return torch.cuda.is_available()

    def clock(self):
        return time.time()

    def banner(self):
        return f"gpu={torch.cuda.get_device_name()} triton={triton.__version__} torch={torch.__version__}"

    def setup(self):
        dev = self.dev
        self.flush = torch.ones(FLUSH_MB * 1024 * 1024 // 4, device=dev, dtype=torch.float32)
        self.sink = torch.zeros((), device=dev, dtype=torch.float32)
        router = self.weights["router"]
        self.r_hi = router.to(_BF).contiguous()
        self.r_lo = (router - self.r_hi.float()).to(_BF).contiguous()
        self.gen = torch.Generator(device=dev).manual_seed(1234)
        if self.scratch is None:
            self.scratch = _alloc_scratch(dev)
        try:  # in-kernel TMA descriptors need a global-scratch allocator
            triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device=dev, dtype=torch.int8))
        except Exception as e:
            self.tma_msg = f"set_allocator: {type(e).__name__}"

    def tma_smoke(self):
        """(ok, message). One tiny TMA launch, synchronised; any failure disables family B."""
        try:
            m, n, k = 4, 64, 512
            x = torch.randn((m, k), generator=self.gen, device=self.dev).to(_BF)
            w = torch.randn((n, k), generator=self.gen, device=self.dev).to(_BF)
            y = _make_run("B", 16, 16, 128, 1, 4, 3, w, None, n, _BF)(x)
            torch.cuda.synchronize()
            ref = x.double() @ w.double().T
            err = float((y.double() - ref).norm() / ref.norm())
            return (err < 1e-2), f"err={err:.1e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:90]}".replace("\n", " ")

    def release(self):
        for a in ("flush", "sink", "r_hi", "r_lo"):
            if hasattr(self, a):
                delattr(self, a)

    # --- per shape
    def prep(self, layer, m):
        W = self.weights[layer]
        n, k = shape_nk(layer, m)
        x = torch.randn((m, k), generator=self.gen, device=self.dev).to(_BF)
        ref = x.double() @ W[:n].double().T
        ctx = dict(layer=layer, m=m, n=n, k=k, x=x, ref=ref, rnorm=ref.norm(), tol=1e-4 if layer == "router" else 1e-2)
        if layer == "router":
            ctx.update(w=self.r_hi, wlo=self.r_lo, odt=torch.float32)
            if m <= ROUTER_FP32_MAX_TOKENS:
                ctx["base"] = lambda x, r=W: F.linear(x.float(), r)
            else:
                ht, lt = self.r_hi.t().contiguous(), self.r_lo.t().contiguous()

                def base(x, ht=ht, lt=lt):
                    acc = torch.mm(x, ht, out_dtype=torch.float32)
                    acc += torch.mm(x, lt, out_dtype=torch.float32)
                    return acc
                ctx["base"] = base
        else:
            ctx.update(w=W[:n], wlo=None, odt=_BF)
            ctx["base"] = lambda x, w=W[:n]: _bf16_gemm_dispatch_impl(x, w, None)
        return ctx

    def build(self, ctx, cfg):
        _, fam, bm, bn, bk, split, w_, s_ = cfg
        ws, sem = self.scratch
        return _make_run(fam, bm, bn, bk, split, w_, s_, ctx["w"], ctx["wlo"], ctx["n"], ctx["odt"], ws, sem)

    def _err(self, ctx, y):
        return float((y.double() - ctx["ref"]).norm() / ctx["rnorm"])

    def check(self, ctx, run):
        """First launch (compiles). Returns the relative L2 error against the fp64 reference."""
        y = run(ctx["x"])
        torch.cuda.synchronize()
        return self._err(ctx, y)

    def time_base(self, ctx, reps):
        return self.time(ctx, ctx["base"], reps)[0]

    def time(self, ctx, run, reps):
        """(median µs of graph replays each preceded by a read-only L2 flush, error of the replayed output)."""
        x = ctx["x"]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            run(x)
            run(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = run(x)
        ts = []
        for _ in range(reps):
            torch.sum(self.flush, dim=0, out=self.sink)  # read-only: 128 MB through L2
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
        ts.sort()
        err = self._err(ctx, y)
        del g
        return ts[len(ts) // 2], err


def _alloc_scratch(dev):
    """Split-K workspace (fp32) and per-tile counters (int32, zero). Allocated in prepare(), shape-indexed by the tile."""
    return torch.empty(WS_FLOATS, device=dev, dtype=torch.float32), torch.zeros(SEM_COUNT, device=dev, dtype=torch.int32)


def _short(e):
    return f"{type(e).__name__}: {str(e)[:80]}".replace("\n", " ")


def _key(cfg, layer):
    """Compile key of a config: what Triton compiles once and reuses for every shape/m (SPLIT is a runtime value)."""
    fam = cfg[1]
    kind = fam if fam in ("A", "B") else ("fix" if fam in ("F", "G") else "atom")
    return (kind,) + cfg[2:5] + cfg[6:] + (layer == "router",)


def _measure(weights, scratch=None, env=None):
    """MEASUREMENT: see the module docstring. `env` is injected by tests (stubbed timing on CPU)."""
    env = env or _RealEnv(weights, scratch)
    if not env.available():
        print("MEAS3 skipped: no cuda")
        return
    t0 = env.clock()
    shapes = all_shapes()
    tabs = [shape_cfgs(layer, m) for layer, m in shapes]
    nres = [[None] * len(t) for t in tabs]       # screen value per config: None | us | "F" | "B"
    fin = [dict() for _ in shapes]               # finals: cfg index -> us
    base_us = [None] * len(shapes)
    ctxs = [None] * len(shapes)
    poisoned = {}                                # compile key -> message (never retried)
    okkeys = set()                               # compile keys that worked at least once
    famfail = {}                                 # family -> (count, first message)
    ncfg, crc = cfg_sig()
    print(f"MEAS3 start {env.banner()} budget={BUDGET_S:.0f}s screen={SCREEN_REPS} finals={FINAL_REPS}", flush=True)
    print("CFG name=<fam><BM>.<BN>.<BK>.<SPLIT>.<warps>.<stages> fam: " + " ".join(f"{k}={v}" for k, v in FAMILIES.items()))
    print(f"CFGSIG n={ncfg} crc={crc:08x} shapes={len(shapes)}")

    def phase(fams, end_s):
        """Config-major: each compile key is compiled once (first use) and then timed on every shape it applies to."""
        order, jobs = {}, []
        for si, tab in enumerate(tabs):
            for ci, c in enumerate(tab):
                if c[1] in fams:
                    key = _key(c, shapes[si][0])
                    jobs.append((order.setdefault(key, len(order)), si, ci, key))
        jobs.sort()
        for _, si, ci, key in jobs:
            if env.clock() - t0 > end_s:
                return
            cfg, ctx = tabs[si][ci], ctxs[si]
            if key in poisoned:
                nres[si][ci] = "F"
                continue
            try:
                run = env.build(ctx, cfg)
                if not (env.check(ctx, run) < ctx["tol"]):
                    nres[si][ci] = "B"
                    continue
                us, err2 = env.time(ctx, run, SCREEN_REPS)
                nres[si][ci] = us if err2 < ctx["tol"] else "B"
                okkeys.add(key)
            except Exception as e:
                nres[si][ci] = "F"
                cnt, msg = famfail.get(cfg[1], (0, ""))
                famfail[cfg[1]] = (cnt + 1, msg or _short(e))
                if not isinstance(e, _Unfit) and key not in okkeys:
                    poisoned[key] = _short(e)

    def finals(fams, with_base):
        """Top 5 (screen) of the given families per shape, again with 51 reps; the baseline too."""
        for si in range(len(shapes)):
            if env.clock() - t0 > BUDGET_S:
                return
            top = sorted((v, ci) for ci, v in enumerate(nres[si]) if isinstance(v, float) and tabs[si][ci][1] in fams)[:5]
            for _, ci in top:
                try:
                    fin[si][ci] = env.time(ctxs[si], env.build(ctxs[si], tabs[si][ci]), FINAL_REPS)[0]
                except Exception:
                    pass
            if with_base:
                base_us[si] = env.time_base(ctxs[si], FINAL_REPS)

    def progress(name):
        n = [sum(isinstance(v, float) for r in nres for v in r), sum(v is None for r in nres for v in r)]
        print(f"MEAS3 phase {name} t={env.clock() - t0:.0f}s timed={n[0]} unrun={n[1]} poisoned={len(poisoned)}", flush=True)

    try:
        env.setup()
        for i, (layer, m) in enumerate(shapes):
            ctxs[i] = env.prep(layer, m)
            base_us[i] = env.time_base(ctxs[i], SCREEN_REPS)
        progress("setup")
        phase(("A",), PHASE_END_S["A"])
        progress("A")
        phase(("G", "H"), PHASE_END_S["GH"])
        progress("GH")
        phase(("F", "Z"), PHASE_END_S["FZ"])
        progress("FZ")
        finals(("A", "F", "Z", "G", "H"), True)  # before TMA: if that ever faults the device, these survive
        progress("finals1")
        tma_ok, tma_msg = env.tma_smoke()
        print(f"TMA smoke: {'ok' if tma_ok else 'FAIL'} {tma_msg} {env.tma_msg}".rstrip(), flush=True)
        if tma_ok:
            phase(("B",), PHASE_END_S["B"])
            progress("B")
            finals(("B",), False)
        else:
            for si, tab in enumerate(tabs):
                for ci, c in enumerate(tab):
                    if c[1] == "B":
                        nres[si][ci] = "F"
    except Exception as e:  # e.g. a device-side fault: report what we have
        print(f"MEAS3 aborted: {_short(e)}")
    finally:
        _report(env, shapes, tabs, nres, fin, base_us, famfail, t0)


def _report(env, shapes, tabs, nres, fin, base_us, famfail, t0):
    for fam, (cnt, msg) in sorted(famfail.items()):
        print(f"FAIL {fam} x{cnt} {msg}")
    lines = []
    for si, (layer, m) in enumerate(shapes):
        vals = list(nres[si])
        for ci, us in fin[si].items():
            vals[ci] = us
        print(f"T64 {shape_id(layer, m)} {_encode_t64(vals)}")
        ok = sorted((us, ci) for ci, us in fin[si].items())
        n_ok = sum(isinstance(v, float) for v in nres[si])
        n_fail = sum(v == "F" for v in nres[si])
        n_bad = sum(v == "B" for v in nres[si])
        n_skip = sum(v is None for v in nres[si])
        top = " ".join(f"{tabs[si][ci][0]}:{us:.2f}" for us, ci in ok[:5]) or "-"
        b = base_us[si]
        line = f"B {layer} m={m} base={b:.2f} | top-5 {top} | ok={n_ok} fail={n_fail} bad={n_bad}" if b is not None else \
            f"B {layer} m={m} base=- | top-5 {top} | ok={n_ok} fail={n_fail} bad={n_bad}"
        if n_skip:
            line += f" skip={n_skip}"
        lines.append(line)
    print(f"MEAS3 done in {env.clock() - t0:.1f}s")
    for line in lines:  # last: if the tail is ever cut, the human lines survive
        print(line)


def prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights["router"].to(torch.bfloat16)
    lo = (weights["router"] - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    scratch = _alloc_scratch(weights["qkv_gate"].device) if torch.cuda.is_available() else None
    _measure(weights, scratch)
    del scratch
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return state


def kernel_fn(data, state):
    """SGLang's baseline, unchanged (this file only measures)."""
    x, layer = data["x"], data["config"]["layer"]
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][: data["config"]["n"]], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc
