# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY block (tools/gen_measure_layer_v1.py appends it after the unchanged sonnet_v5 code and the inlined generate_input).
# Nothing here feeds kernel_fn: prepare() runs sonnet_v5's prepare(), then times the C22a' variants of fable-09 D4 section 2 on
# generate_input-recipe inputs (3 seeds per T, read-only L2 flush, CUDA-graph replays under CUPTI) and prints the B / A / K / C lines.
# ---------------------------------------------------------------------------------------------------------------------
import time as _ml_time
import threading as _ml_threading

_ML_T0 = _ml_time.perf_counter()
_ML_BUDGET_S = 262.0        # prepare() ends by here (the judge's limit is 300 s); variants / buckets that do not fit are dropped
_ML_FRONT_S = 205.0         # no new T bucket of the front study after this
_ML_C24_S = 235.0           # no new C24 bucket after this
_ML_TS = (1, 128, 256, 16, 32, 8)    # processing order = priority when the budget runs short (the output lists them sorted)
_ML_SHAPES = {1: (1, 1), 8: (1, 8), 16: (16, 1), 32: (32, 1), 128: (16, 8), 256: (32, 8)}   # T -> (batch, q_len): the judged loads
_ML_SEEDS = (11, 12, 13)
_ML_NAMES = ("base", "order", "rpar", "ordrpar", "psm", "psm2", "psmr", "psmrpar", "tma", "mm", "ordmm", "bm", "ordbm", "rs2")
_ML_CTX = 4096              # cache length of the study inputs: the FFN half of the layer does not depend on it
_ML_REPS = 12
_ML_FLUSH_MB = 128
_ML_MAX_BYTES = 13800
_ML_OUT = {"B": [], "A": [], "K": [], "C": [], "N": []}   # result lines by kind; "N" = notes


def _ml_left():
    return _ML_BUDGET_S - (_ml_time.perf_counter() - _ML_T0)


def _ml_elapsed():
    return _ml_time.perf_counter() - _ML_T0


def _ml_note(text):
    _ML_OUT["N"].append(" ".join(str(text).split())[:230])


def _ml_med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _ml_where(ex):
    tb = ex.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return f"{type(ex).__name__}@{tb.tb_lineno if tb else '?'}: {str(ex)[:110]}"


def _ml_us(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


# ===================================================================================================== kernels of the variants
@triton.jit
def _ml_router_fix(x_ptr, hi_ptr, lo_ptr, out_ptr, part_ptr, done_ptr, M, stride_x,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, SPLIT_X: tl.constexpr,
                   K: tl.constexpr, N: tl.constexpr):
    """RPAR / RBIG: router logits = x . (W_hi + W_lo)^T with FP32 accumulation, split-K over SPLIT CTAs, and NO zero-fill and NO atomics on the
    output: every CTA stores its FP32 partial, takes an arrival ticket (CUTLASS's semaphore pattern, as _router_f2_kernel), and the last CTA of
    each (m-block, n-block) tile sums the partials in slice order and stores the logits. The tile's counter is reset by that CTA. Deterministic."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    k_per: tl.constexpr = K // SPLIT
    k0 = pid_k * k_per
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, k_per, BK):
        rk = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * stride_x + rk[None, :], mask=rm[:, None] < M, other=0.0)
        w = tl.load(hi_ptr + rn[:, None] * K + rk[None, :])
        wl = tl.load(lo_ptr + rn[:, None] * K + rk[None, :])
        if SPLIT_X:
            xh = x.to(hi_ptr.dtype.element_ty)
            xl = (x - xh.to(tl.float32)).to(hi_ptr.dtype.element_ty)
            acc += tl.dot(xh, tl.trans(w))
            acc += tl.dot(xl, tl.trans(w))
            acc += tl.dot(xh, tl.trans(wl))
        else:
            xc = x.to(hi_ptr.dtype.element_ty)
            acc += tl.dot(xc, tl.trans(w))
            acc += tl.dot(xc, tl.trans(wl))
    tl.store(part_ptr + (pid_k * M + rm)[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M)
    tl.debug_barrier()
    NB: tl.constexpr = N // BN
    ticket = tl.atomic_add(done_ptr + pid_m * NB + pid_n, 1, sem="acq_rel", scope="gpu")
    if ticket == SPLIT - 1:
        tot = tl.zeros((BM, BN), dtype=tl.float32)
        for s in range(SPLIT):  # .cg: skip L1 so no stale line of this buffer can be read
            tot += tl.load(part_ptr + (s * M + rm)[:, None] * N + rn[None, :], mask=rm[:, None] < M, other=0.0, cache_modifier=".cg")
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], tot, mask=rm[:, None] < M)
        tl.store(done_ptr + pid_m * NB + pid_n, 0)


@triton.jit
def _ml_gemv_tma(x_ptr, w_desc, out_ptr, M, N, K, stride_xm, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """TMAPOOL: the pooler GEMV (split 1) with W streamed by a host TensorDescriptor (cp.async.bulk.tensor + mbarrier pipeline, num_stages deep)."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, K, BK):
        rk = kk + tl.arange(0, BK)
        x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        w = w_desc.load([pid_n * BN, kk])
        acc += tl.dot(x, tl.trans(w))
    rn = pid_n * BN + tl.arange(0, BN)
    om = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], acc.to(out_ptr.dtype.element_ty), mask=om)


# (name, BN, BK, SPLIT, warps, stages); BM = 16 / 32 (T <= 16 / <= 32) or T (RBIG: one M block, so W is read once)
_ML_RPAR_CFG = {
    1: (("A", 64, 128, 64, 4, 4), ("B", 32, 128, 32, 4, 4), ("C", 16, 128, 64, 4, 3)),      # A: 256 CTAs, B: 256, C: 1024
    8: (("A", 64, 128, 64, 4, 4), ("B", 32, 128, 32, 4, 4), ("C", 16, 128, 64, 4, 3)),
    16: (("A", 64, 64, 64, 4, 4), ("B", 32, 128, 32, 4, 4), ("C", 16, 128, 64, 4, 3)),
    32: (("A", 16, 128, 64, 4, 3), ("B", 64, 128, 64, 4, 3), ("C", 32, 128, 32, 4, 4)),
    128: (("A", 32, 32, 32, 4, 3), ("B", 16, 32, 16, 4, 3)),                                  # RBIG, BM = T
    256: (("A", 32, 32, 32, 4, 3), ("B", 16, 32, 16, 4, 3)),
}
_ML_TMA_CFG = (("4", 32, 256, 4, 4), ("6", 32, 256, 4, 6))        # (name, BN, BK, warps, stages)
_ML_POOLSMEM_CFG = (("ps1", 32, 256, 1, 4, 4), ("ps2", 32, 256, 1, 4, 3))   # POOLSMEM: the pooler GEMV with half the smem (offline ptxas, T=1 / T=32: 74 / 98 KB and 49 / 65 KB vs 147 / 197 KB)
_ML_OSHOT_CFG = (("A", 32, 1024, 8, 1), ("B", 32, 512, 4, 2), ("C", 16, 1024, 4, 1))   # (name, BN, BK, warps, stages), split 1


class _MLSt:
    pass


def _ml_setup(state):
    s = _MLSt()
    s.dev = torch.device("cuda")
    s.sg = HAVE_SGLANG
    s.tri = _triton_ok(s.dev)
    s.lp = torch.bfloat16 if s.tri else torch.float16           # the dev GPU (sm_75) runs the Triton kernels on FP16 stand-ins
    s.cap = torch.cuda.get_device_capability(s.dev)[0]
    s.rhi = state["router_hi"] if s.tri else state["router_hi16"]
    s.rlo = state["router_lo"] if s.tri else state["router_lo16"]
    s.rhi_t, s.rlo_t = state["router_hi_t"], state["router_lo_t"]
    s.pw = state["dense"]["pooler"] if s.tri else state["dense"]["pooler"].half()
    s.ow = state["dense"]["o"] if s.tri else state["dense"]["o"].half()
    s.bias = state["router_bias"]
    s.part = torch.empty((64, 256, 256), dtype=torch.float32, device=s.dev)      # RPAR scratch (stable pointers, counters left at zero)
    s.done = torch.zeros(4096, dtype=torch.int32, device=s.dev)
    s.side = state["side_stream"] if s.sg else torch.cuda.Stream()
    s.state = state
    if s.sg:
        s.routing = lambda lg: state["routing"](None, lg, TOP_K, True)
        s.norm = state["pre_expert_norm"]
    else:
        def routing(lg):
            sc = torch.sigmoid(lg * ROUTER_MULT)
            ids = (sc + s.bias).topk(TOP_K, dim=-1).indices
            picked = sc.gather(1, ids)
            return picked / (picked.sum(-1, keepdim=True) + ROUTE_EPS), ids.to(torch.int32)

        s.routing = routing
        s.norm = lambda x: F.layer_norm(x, (x.shape[-1],))
    s.tma_ok = False
    s.pdesc = {}
    if s.cap >= 9:
        try:
            from triton.tools.tensor_descriptor import TensorDescriptor

            for name, bn, bk, w, st in _ML_TMA_CFG:
                s.pdesc[name] = TensorDescriptor.from_tensor(s.pw, [bn, bk])
            s.tma_ok = True
        except Exception as ex:  # noqa: BLE001
            _ml_note(f"TMAPOOL dropped: no host TensorDescriptor ({type(ex).__name__}: {str(ex)[:80]})")
    else:
        _ml_note("TMAPOOL dropped: not an sm_90 GPU")
    return s


# ===================================================================================================== ops (each launches on the current stream)
def _ml_bm(t):
    return 16 if t <= 16 else (32 if t <= 32 else t)


def _ml_router_base(s, x, rin, stages=None):
    """sonnet_v5's router: _gemv_splitk (FP32 atomics into torch.zeros), FP32 hidden split hi+lo in registers at T <= 8. stages: POOLSMEM's smaller-smem router."""
    cfg = _CFG[("router", _bucket(x.shape[0]))]
    if stages is not None:
        cfg = cfg[:4] + (stages,)
    if rin is not None:
        return _run(rin.contiguous(), s.rhi, s.rlo, N_EXPERTS, torch.float32, cfg, split_x=True)
    return _run(x.contiguous(), s.rhi, s.rlo, N_EXPERTS, torch.float32, cfg)


def _ml_router_fix_op(s, x, rin, cfg):
    name, bn, bk, split, warps, stages = cfg
    src = rin.contiguous() if rin is not None else x.contiguous()
    t = x.shape[0]
    bm = _ml_bm(t)
    out = torch.empty((t, N_EXPERTS), dtype=torch.float32, device=x.device)
    grid = (N_EXPERTS // bn, split, triton.cdiv(t, bm))
    _ml_router_fix[grid](src, s.rhi, s.rlo, out, s.part, s.done, t, src.stride(0), BM=bm, BN=bn, BK=bk, SPLIT=split,
                         SPLIT_X=src.dtype == torch.float32, K=HIDDEN, N=N_EXPERTS, num_warps=warps, num_stages=stages)
    return out


def _ml_router_mm(s, x, rin):
    """RBIG (a): SGLang's own split path, two cuBLAS GEMMs on the BF16 hi / lo weights with FP32 output."""
    if s.tri:
        lg = torch.mm(x, s.rhi_t, out_dtype=torch.float32)
        lg += torch.mm(x, s.rlo_t, out_dtype=torch.float32)
        return lg
    xf = x.float()
    return xf @ s.rhi_t.float() + xf @ s.rlo_t.float()


def _ml_pool_base(s, x):
    t = x.shape[0]
    if t <= POOLER_TRITON_MAX_TOKENS:
        return _run(x.contiguous(), s.pw, None, D_EXPERT, s.lp, _CFG[("pooler", _bucket(t))])
    return _gemm(x, s.pw) if s.tri else x @ s.pw.T


def _ml_pool_tma(s, x, cfg):
    name, bn, bk, warps, stages = cfg
    t, k = x.shape
    bm = 16 if t <= 16 else 32
    out = torch.empty((t, D_EXPERT), dtype=s.lp, device=x.device)
    _ml_gemv_tma[(D_EXPERT // bn, 1, triton.cdiv(t, bm))](x.contiguous(), s.pdesc[name], out, t, D_EXPERT, k, x.stride(0), out.stride(0),
                                                           BM=bm, BN=bn, BK=bk, num_warps=warps, num_stages=stages)
    return out


def _ml_o_cublas(s, o):
    return _gemm(o, s.ow) if s.tri else o @ s.ow.T


def _ml_o_shot(s, o, cfg):
    name, bn, bk, warps, stages = cfg
    return _run(o.contiguous(), s.ow, None, HIDDEN, s.lp, (bn, bk, 1, warps, stages))


def _ml_routers(s, t):
    """router key -> callable(x, rin) -> FP32 logits [T, 256]."""
    d = {"base": lambda x, rin: _ml_router_base(s, x, rin), "s2": lambda x, rin: _ml_router_base(s, x, rin, stages=2)}
    if t <= ROUTER_TRITON_MAX_TOKENS:
        for cfg in _ML_RPAR_CFG[_bucket(t)]:
            d[("bm" if t > 32 else "rpar") + cfg[0]] = (lambda c: lambda x, rin: _ml_router_fix_op(s, x, rin, c))(cfg)
    if t > ROUTER_FP32_MAX_TOKENS:
        d["mm"] = lambda x, rin: _ml_router_mm(s, x, rin)
    return d


def _ml_poolers(s, t):
    d = {"base": lambda x: _ml_pool_base(s, x)}
    if t <= POOLER_TRITON_MAX_TOKENS:
        for cfg in _ML_POOLSMEM_CFG:
            d[cfg[0]] = (lambda c: lambda x: _run(x.contiguous(), s.pw, None, D_EXPERT, s.lp, c[1:]))(cfg)
    if t <= POOLER_TRITON_MAX_TOKENS and s.tma_ok:
        for cfg in _ML_TMA_CFG:
            d["tma" + cfg[0]] = (lambda c: lambda x: _ml_pool_tma(s, x, c))(cfg)
    return d


def _ml_front(s, x, rin, rf, pf, order):
    """The part of the FFN half sonnet_v5 runs between the attention boundary and the experts: router + routing (side stream) beside the pooler
    (main stream), join, pre-expert norm. order=False is sonnet_v5's topology; order=True runs the router first on the main stream, then
    only the routing goes to the side stream while the pooler streams (D4 section 2, H6')."""
    main, side = torch.cuda.current_stream(), s.side
    capturing = torch.cuda.is_current_stream_capturing()
    if order:
        logits = rf(x, rin)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            w, ids = s.routing(logits)
        pooled = pf(x)
        if not capturing:
            logits.record_stream(side)
    else:
        side.wait_stream(main)
        pooled = pf(x)
        with torch.cuda.stream(side):
            logits = rf(x, rin)
            w, ids = s.routing(logits)
    main.wait_stream(side)
    if not capturing:
        for t in (w, ids):
            t.record_stream(main)
    return s.norm(pooled), w, ids


# ===================================================================================================== inputs
def _ml_case(t, seed):
    b, q = _ML_SHAPES[t]
    return _tr_generate_input(batch=b, ctx=_ML_CTX, layer="sliding", seed=seed, q_len=q, spread="fixed", dist="soft", page=1,
                              local_start=0, device="cuda")


def _ml_head(s, data):
    """sonnet_v5's layer up to the attention boundary: -> (ffn_input in the study dtype [T, 8192], FP32 router input or None, attention output o [T, 1024])."""
    state, cfg = s.state, data["config"]
    tokens = cfg["tokens"]
    hq, d = cfg["num_q_heads"], cfg["head_dim"]
    dense = state["dense"]
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if _attention_path(cfg, True) == "triton":
        o = _attention_decode({**data, "qkv": qkv}).view(tokens, hq * d)
    elif s.sg:
        o = _attention_fa3(data, qkv, state)
    else:
        o = _attention_torch(data, qkv)
    if s.sg:
        a = _gemm(o, dense["o"])
        ffn, _, rin = state["attn_boundary"](data["stream"], a, branch_dtype=torch.bfloat16,
                                              router_dtype=torch.float32 if tokens <= ROUTER_FP32_MAX_TOKENS else None)
        return ffn.to(s.lp), rin, o.to(s.lp)
    w = state["weights"]
    a = o @ dense["o"].T
    alpha, beta = _residual_coefficients(w["attn_residual_gate"])
    stream = alpha * data["stream"] + beta * _rms(a, POST_NORM_EPS)
    h32 = _rms(stream) * _gain(w["pre_ffn_norm"])
    return h32.to(torch.bfloat16).to(s.lp), (h32 if tokens <= ROUTER_FP32_MAX_TOKENS else None), o.to(s.lp)


def _ml_fix_start(s, data, ids):
    """The judge gives each input the 32-expert range its routing sends the most pairs to; E = distinct experts of that range."""
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    start = LOCAL_EXPERTS * int(pairs.argmax())
    data["local_start"].fill_(start)
    rel = (ids.long() - start) % N_EXPERTS
    return start, int(torch.unique(rel[rel < LOCAL_EXPERTS]).numel())


# ===================================================================================================== profile session (after trace_runtime.py)
class _MLEnv:
    def __init__(self):
        self.flush = torch.ones(_ML_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        self.sink = torch.zeros((), device="cuda", dtype=torch.float32)

    def cold(self):
        torch.sum(self.flush, dim=0, out=self.sink)   # read-only: pushes the previous replay's data out of L2


_ml_marks = [0]


def _ml_mark():
    _ml_marks[0] += 1
    torch.cuda._sleep(1000)


def _ml_graph(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _ml_unit(env, units, key, run, reps=_ML_REPS, **meta):
    """Inside the profiling session: 2 warm runs, then reps x (L2 flush, marker, run, marker). The markers delimit the replay's kernels."""
    for _ in range(2):
        env.cold()
        run()
    torch.cuda.synchronize()
    lo = _ml_marks[0]
    for _ in range(reps):
        env.cold()
        _ml_mark()
        run()
        _ml_mark()
    torch.cuda.synchronize()
    units.append(dict(key=key, lo=lo, hi=_ml_marks[0], **meta))


def _ml_gpu_rows(prof):
    """[(start_ns, end_ns, stream, name)] of every kernel / memset / memcpy record of the current device."""
    try:
        kr = prof.profiler.kineto_results
        cuda = torch._C._autograd.DeviceType.CUDA
        dev = torch.cuda.current_device()
        rows = []
        for e in kr.events():
            if e.device_type() != cuda or e.device_index() != dev:
                continue
            kind = str(e.activity_type()).lower()
            if "kernel" not in kind and "memset" not in kind and "memcpy" not in kind:
                continue
            s0 = int(e.start_ns())
            rows.append((s0, s0 + int(e.duration_ns()), int(e.device_resource_id()), e.name()))
        if rows:
            return rows
    except Exception:  # noqa: BLE001  (private API moved: use the exported trace)
        pass
    import json as _json
    import tempfile as _tempfile

    dev = torch.cuda.current_device()
    rows = []
    with _tempfile.TemporaryDirectory() as d:
        path = d + "/t.json"
        prof.export_chrome_trace(path)
        with open(path) as fh:
            js = _json.load(fh)
    for e in js.get("traceEvents", []):
        if e.get("cat") not in ("kernel", "gpu_memset", "gpu_memcpy"):
            continue
        a = e.get("args", {})
        if a.get("device", dev) != dev:
            continue
        s0 = int(round(float(e["ts"]) * 1000.0))
        rows.append((s0, s0 + int(round(float(e["dur"]) * 1000.0)), int(a.get("stream", 0)), e["name"]))
    return rows


def _ml_replays(rows, units):
    """-> {unit key: [kernel rows of replay i]}; None when the marker count is off (CUPTI lost records)."""
    rows = sorted(rows, key=lambda r: (r[0], r[1]))
    marks = [r for r in rows if "spin_kernel" in r[3]]
    base = units[0]["lo"] if units else 0
    total = _ml_marks[0] - base
    if len(marks) != total:
        return None
    body = [r for r in rows if "spin_kernel" not in r[3]]
    res = {}
    for u in units:
        mm = marks[u["lo"] - base:u["hi"] - base]
        res[u["key"]] = [[r for r in body if mm[2 * i][1] <= r[0] and r[1] <= mm[2 * i + 1][0] + 1] for i in range(len(mm) // 2)]
    return res


def _ml_span(rows):
    return (max(r[1] for r in rows) - min(r[0] for r in rows)) / 1000.0 if rows else None


def _ml_alone(reps):
    """Unit of one op run alone: (median span first start -> last end in us, kernels per run)."""
    spans = [_ml_span(r) for r in reps if r]
    cnt = {}
    for r in reps:
        cnt[len(r)] = cnt.get(len(r), 0) + 1
    return _ml_med(spans), (max(cnt, key=lambda k: (cnt[k], k)) if cnt else 0)


def _ml_front_metrics(kern, order, nk_r, nk_g):
    """One replay of the front graph -> dict(ln, side_end, idle, pool, rout, span) in us. The last-ending kernel is the pre-expert norm (main stream)."""
    kern = sorted(kern, key=lambda r: (r[0], r[1]))
    t0 = kern[0][0]
    last = max(kern, key=lambda r: r[1])
    main = last[2]
    mrows = [r for r in kern if r[2] == main and r is not last]
    srows = [r for r in kern if r[2] != main]
    m = dict(ln=(last[0] - t0) / 1000.0, span=(last[1] - t0) / 1000.0, side_end=((max(r[1] for r in srows) - t0) / 1000.0) if srows else None,
             idle=max(0.0, (last[0] - max(r[1] for r in mrows)) / 1000.0) if mrows else None, pool=None, rout=None)
    if order:
        rr, pr = (mrows[:nk_r], mrows[nk_r:]) if nk_r and len(mrows) > nk_r else ([], [])
    else:
        rr, pr = (srows[:-nk_g], mrows) if nk_g and len(srows) > nk_g else ([], [])
    if rr and pr:
        m["pool"], m["rout"] = _ml_span(pr), _ml_span(rr)
    return m


def _ml_exp_metrics(kern):
    kern = sorted(kern, key=lambda r: (r[0], r[1]))
    big = [r for r in kern if "arlin" in r[3] or "w4_largem" in r[3]]
    m = dict(e2e=_ml_span(kern), g1=None, g2=None)
    if len(big) >= 2:
        m["g1"], m["g2"] = (big[0][1] - big[0][0]) / 1000.0, (big[-1][1] - big[-1][0]) / 1000.0
    return m


# ===================================================================================================== one T bucket of the front study
def _ml_chk_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


def _ml_ids_diff(i1, i0):
    return int((torch.sort(i1.long(), dim=-1).values != torch.sort(i0.long(), dim=-1).values).any(dim=-1).sum())


def _ml_variants(s, t):
    """[(name, order, router key, pooler key)] for the T bucket."""
    big = t > 32
    rs = ["mm", "bmA"] if big else ["rparA"]
    out = [("base", False, "base", "base"), ("order", True, "base", "base")]
    for r in rs:
        tag = {"mm": "mm", "bmA": "bm", "rparA": "rpar"}[r]
        out += [(tag, False, r, "base"), ("ord" + tag, True, r, "base")]
    if big:
        out += [("rs2", False, "s2", "base")]
    else:
        out += [("psm", False, "base", "ps1"), ("psm2", False, "base", "ps2"), ("psmr", False, "s2", "ps1"), ("psmrpar", False, "rparA", "ps1")]
        if s.tma_ok:
            out += [("tma", False, "base", "tma4")]
    return out


def _ml_validate(s, t, routers, poolers):
    """Launch every router / pooler variant once on synthetic inputs; one that does not compile or launch (smem, registers, a driver refusal) is dropped with a note."""
    g = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn((t, HIDDEN), generator=g, device="cuda").to(s.lp)
    rin = torch.randn((t, HIDDEN), generator=g, device="cuda") if t <= ROUTER_FP32_MAX_TOKENS else None
    for table, args in ((routers, (x, rin)), (poolers, (x,))):
        for k in list(table):
            try:
                table[k](*args)
                torch.cuda.synchronize()
            except Exception as ex:  # noqa: BLE001
                _ml_note(f"T={t} {k} dropped: {_ml_where(ex)}")
                del table[k]
    torch.cuda.synchronize()


def _ml_run_T(s, env, t, results, chk, a_rows):
    """Everything for one T (3 seeds) inside one profiling session; fills results[(t, seed)] after the session."""
    from torch.profiler import ProfilerActivity, profile

    routers, poolers = _ml_routers(s, t), _ml_poolers(s, t)
    _ml_validate(s, t, routers, poolers)
    variants = [v for v in _ml_variants(s, t) if v[2] in routers and v[3] in poolers]
    units, meta = [], {}
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for seed in _ML_SEEDS:
            if _ml_elapsed() > _ML_FRONT_S + 12:
                _ml_note(f"T={t} seed={seed} dropped:budget")
                continue
            data = _ml_case(t, seed)
            x, rin, o_attn = _ml_head(s, data)
            torch.cuda.synchronize()
            lg0 = routers["base"](x, rin)
            w0, ids0 = s.routing(lg0)
            pooled0 = poolers["base"](x)
            start, e_cnt = _ml_fix_start(s, data, ids0)
            meta[seed] = dict(E=e_cnt, start=start)
            # ---- correctness: every variant against the baseline path on these inputs
            ck = {}
            for rk, rf in routers.items():
                if rk == "base":
                    continue
                lg = rf(x, rin)
                lg2 = rf(x, rin)
                w1, i1 = s.routing(lg)
                torch.cuda.synchronize()
                ck["r" + rk] = (_ml_chk_rel(lg, lg0), _ml_ids_diff(i1, ids0), _ml_chk_rel(w1, w0), bool(torch.equal(lg, lg2)))
            for pk, pf in poolers.items():
                if pk != "base":
                    pp = pf(x)
                    torch.cuda.synchronize()
                    ck["p" + pk] = (_ml_chk_rel(pp, pooled0), bool(torch.equal(pp, pooled0)))
            xe0, wb, ib = _ml_front(s, x, rin, routers["base"], poolers["base"], False)
            xe1, wo, io = _ml_front(s, x, rin, routers["base"], poolers["base"], True)
            torch.cuda.synchronize()
            ck["order"] = (bool(torch.equal(xe0, xe1)), _ml_ids_diff(io, ib), _ml_chk_rel(wo, wb))
            if t <= 32:
                ref = _ml_o_cublas(s, o_attn)
                for cfg in _ML_OSHOT_CFG:
                    got = _ml_o_shot(s, o_attn, cfg)
                    torch.cuda.synchronize()
                    ck["o" + cfg[0]] = (_ml_chk_rel(got, ref),)
            chk.setdefault(t, []).append(ck)
            # ---- timed units
            for rk, rf in routers.items():
                _ml_unit(env, units, ("ra", t, seed, rk), (lambda f: lambda: f(x, rin))(rf), kind="alone")
            for pk, pf in poolers.items():
                _ml_unit(env, units, ("pa", t, seed, pk), (lambda f: lambda: f(x))(pf), kind="alone")
            _ml_unit(env, units, ("ga", t, seed), lambda: s.routing(lg0), kind="alone")
            _ml_unit(env, units, ("na", t, seed), lambda: s.norm(pooled0), kind="alone")
            if t <= 32:
                _ml_unit(env, units, ("oa", t, seed, "cublas"), lambda: _ml_o_cublas(s, o_attn), kind="alone")
                for cfg in _ML_OSHOT_CFG:
                    _ml_unit(env, units, ("oa", t, seed, cfg[0]), (lambda c: lambda: _ml_o_shot(s, o_attn, c))(cfg), kind="alone")
            for name, order, rk, pk in variants:
                try:
                    g = _ml_graph(lambda: _ml_front(s, x, rin, routers[rk], poolers[pk], order))
                except Exception as ex:  # noqa: BLE001
                    _ml_note(f"T={t} {name} graph failed: {_ml_where(ex)}")
                    continue
                _ml_unit(env, units, ("f", t, seed, name), g.replay, kind="front", order=order, rk=rk, pk=pk)
                del g
            del data, x, rin, o_attn, lg0
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
    reps = _ml_replays(_ml_gpu_rows(prof), units) if units else {}
    if reps is None:
        _ml_note(f"T={t}: CUPTI marker count off, no timings")
        return
    nk = {}
    for u in units:
        if u["kind"] == "alone":
            nk[u["key"]] = _ml_alone(reps[u["key"]])
    for u in units:
        k = u["key"]
        if u["kind"] != "front":
            continue
        t_, seed, name = k[1], k[2], k[3]
        nk_r = nk.get(("ra", t_, seed, u["rk"]), (None, 0))[1]
        nk_g = nk.get(("ga", t_, seed), (None, 0))[1]
        ms = [_ml_front_metrics(r, u["order"], nk_r, nk_g) for r in reps[k] if r]
        results[(t_, seed, name)] = {f: _ml_med([m[f] for m in ms]) for f in ("ln", "side_end", "idle", "pool", "rout", "span")}
    for key, (span, n) in nk.items():
        a_rows[key] = (span, n)
    for seed, m in meta.items():
        results[(t, seed, "_meta")] = m


# ===================================================================================================== C24-lite: Marlin + S1 against the large-M path on the layer's routing
LM_TC = 32
LM_CTW = 4
LM_MAX_M = 256
SM_NEXP_LM = 32
_ML_LM_CFG = {
    32: dict(nw1=4, mr1=2, ks1=4, nz1=16, nw2=4, mr2=2, ks2=4, nz2=16),
    128: dict(nw1=4, mr1=2, ks1=4, nz1=32, nw2=4, mr2=2, ks2=4, nz2=32),
    256: dict(nw1=4, mr1=2, ks1=4, nz1=48, nw2=4, mr2=2, ks2=4, nz2=48),
}
_ML_LM = {"ext": None, "err": None, "thread": None, "make_ext": None}   # make_ext: tests inject an NVRTC extension factory


def _ml_lm_build():
    """Compile the large-M extension (device source of the MoE problem's sonnet_v4, route kernel reading local_start from the device tensor
    with the layer's wrap rule). Runs in a thread beside the front study: nvcc is CPU work."""
    try:
        if _ML_LM["make_ext"] is not None:
            _ML_LM["ext"] = _ML_LM["make_ext"]()
            return
        import hashlib

        from torch.utils.cpp_extension import load_inline

        src = _SMALLM_DEVICE_SRC + _ML_LARGEM_EXTRA_SRC + _ML_LARGEM_HOST_SRC
        tag = hashlib.sha1(src.encode()).hexdigest()[:12]
        _ML_LM["ext"] = load_inline(
            name=f"okc_layer_largem_{tag}", cpp_sources=_ML_LARGEM_CPP_SRC, cuda_sources=src,
            functions=["largem_gemm", "largem_route", "largem_init"], extra_cuda_cflags=["-O3"], verbose=False)
    except Exception as ex:  # noqa: BLE001
        _ML_LM["err"] = f"{type(ex).__name__}: {str(ex)[:200]}"


def _ml_lm_scratch(sm, ext):
    d, dev = sm["dims"], sm["w1"].device
    zmax = 8 * LM_MAX_M // LM_TC + SM_NEXP_LM
    need = max(d["g1"], d["g2"]) * 8
    nbn = max(d["g1"], d["g2"]) // 4
    lm = {"ctab": torch.zeros(4 + LM_CTW * zmax, dtype=torch.int32, device=dev), "slotl": torch.zeros(8 * LM_MAX_M, dtype=torch.int32, device=dev),
          "part": torch.empty(zmax * need * 128 * 4, dtype=torch.float32, device=dev), "cnt": torch.zeros((2, zmax * nbn), dtype=torch.int32, device=dev),
          "zmax": zmax, "ext": ext}
    ext.largem_init()
    return lm


def _ml_lm_experts(x, topk_weights, topk_ids, local_start, state, lm, cfg):
    """Route (local_start read on the device, wrap rule) + GEMM1 (SwiGLU) + GEMM2 + the layer's epilogue for 17..256 tokens."""
    m = x.shape[0]
    sm = state["smallm"]
    d, ext = sm["dims"], lm["ext"]
    ids = topk_ids.to(torch.int32).contiguous()
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
    ext.largem_route(ids, local_ids, lm["ctab"], lm["slotl"], local_start, LOCAL_EXPERTS)
    inter = torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
    zb = min(lm["zmax"], 8 * m // LM_TC + SM_NEXP_LM)
    ext.largem_gemm(sm["w1"], sm["s1"], sm["gs1"], sm["lut"], x.contiguous(), lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][0], inter,
                    d["g1"], d["k1"] // 64, d["k1"], cfg["ks1"], min(zb, cfg["nz1"]), d["inter"], 3, cfg["nw1"], cfg["mr1"], True)
    ext.largem_gemm(sm["w2"], sm["s2"], sm["gs2"], sm["lut"], inter, lm["ctab"], lm["slotl"], lm["part"], lm["cnt"][1], y,
                    d["g2"], d["k2"] // 64, d["k2"], cfg["ks2"], min(zb, cfg["nz2"]), d["hid"], 0, cfg["nw2"], cfg["mr2"], False)
    y = y.view(m, TOP_K, d["hid"])
    w32 = topk_weights.to(torch.float32).contiguous()
    if _epilogue_path(m) == "triton":
        return epilogue_triton(y, w32, local_ids, state["post_expert_w"], block_h=_marlin_knobs(m)[2])
    return fused_post_expert_norm_weighted_sum(y, w32, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
                                               slot_expert_ids=local_ids)


def _ml_route_check(ext, lm):
    """The route kernel against a Python model of the layer's rule for starts that wrap (240, 250) and do not (0, 96)."""
    ids = torch.tensor([[250, 255, 0, 15, 16, 100, 239, 240], [3, 3, 3, 241, 245, 17, 18, 31]], dtype=torch.int32, device="cuda")
    bad = 0
    for start in (240, 250, 0, 96):
        st = torch.tensor([start], dtype=torch.int32, device="cuda")
        local = torch.empty_like(ids)
        ext.largem_route(ids, local, lm["ctab"], lm["slotl"], st, LOCAL_EXPERTS)
        torch.cuda.synchronize()
        rel = (ids.long() - start) % N_EXPERTS
        want = torch.where(rel < LOCAL_EXPERTS, rel, torch.full_like(rel, -1)).to(torch.int32)
        bad += int(not torch.equal(local, want))
    return bad


def _ml_run_c24(s, env, state, a_rows, c_lines):
    from torch.profiler import ProfilerActivity, profile

    if state.get("smallm") is None:
        _ml_note("C24 dropped: no smallm state (packed weights)")
        return
    th = _ML_LM["thread"]
    if th is not None:
        th.join(timeout=max(1.0, _ML_C24_S - _ml_elapsed() - 15))
        if th.is_alive():
            _ml_note("C24 dropped:budget (large-M extension still compiling)")
            return
    if _ML_LM["ext"] is None:
        _ml_note(f"C24 dropped: build failed {_ML_LM['err']}")
        return
    lm = _ml_lm_scratch(state["smallm"], _ML_LM["ext"])
    bad = _ml_route_check(_ML_LM["ext"], lm)
    c_lines.append(f"C24 route kernel vs wrap-rule model (starts 240 250 0 96): {'OK' if bad == 0 else f'{bad} MISMATCH'}")
    for t in (32, 128, 256):
        if _ml_elapsed() > _ML_C24_S:
            _ml_note(f"C24 T={t} dropped:budget")
            continue
        units, info = [], {}
        cfg = _ML_LM_CFG[32 if t <= 32 else (128 if t <= 128 else 256)]
        routers, poolers = _ml_routers(s, t), _ml_poolers(s, t)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for seed in _ML_SEEDS:
                data = _ml_case(t, seed)
                x, rin, _ = _ml_head(s, data)
                lg = routers["base"](x, rin)
                w, ids = s.routing(lg)
                start, e_cnt = _ml_fix_start(s, data, ids)
                xe = s.norm(poolers["base"](x))
                if s.sg:
                    xe = xe.to(torch.bfloat16)
                torch.cuda.synchronize()
                ref = _experts(xe, w, ids, data["local_start"], {**state, "smallm": None})
                try:
                    got = _ml_lm_experts(xe, w, ids, data["local_start"], state, lm, cfg)
                    torch.cuda.synchronize()
                    rel = _ml_chk_rel(got, ref)
                    g_m = _ml_graph(lambda: _experts(xe, w, ids, data["local_start"], {**state, "smallm": None}))
                    _ml_unit(env, units, ("e", t, seed, "marlin"), g_m.replay, kind="exp")
                    g_l = _ml_graph(lambda: _ml_lm_experts(xe, w, ids, data["local_start"], state, lm, cfg))
                    _ml_unit(env, units, ("e", t, seed, "largem"), g_l.replay, kind="exp")
                    del g_m, g_l
                except Exception as ex:  # noqa: BLE001
                    rel = None
                    _ml_note(f"C24 T={t} seed={seed} largem failed: {_ml_where(ex)}")
                info[seed] = (e_cnt, rel)
                del data, x, rin, lg, xe, ref
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
        reps = _ml_replays(_ml_gpu_rows(prof), units) if units else {}
        if reps is None:
            _ml_note(f"C24 T={t}: CUPTI marker count off")
            continue
        for seed, (e_cnt, rel) in info.items():
            rowm = {}
            for path in ("marlin", "largem"):
                key = ("e", t, seed, path)
                if key in reps:
                    ms = [_ml_exp_metrics(r) for r in reps[key] if r]
                    rowm[path] = {f: _ml_med([m[f] for m in ms]) for f in ("e2e", "g1", "g2")}
            c_lines.append((t, seed, e_cnt, rel, rowm))


# ===================================================================================================== output
def _ml_fmt_b(t, seed, e, name, m):
    f = f"B layer T={t} seed={seed} E={e} | {name} side_end={_ml_us(m['side_end'])} main_idle={_ml_us(m['idle'])} pool={_ml_us(m['pool'])} rout={_ml_us(m['rout'])} ln={_ml_us(m['ln'])}"
    return f


def _ml_render(results, a_rows, c_lines, cupti_ok):
    lines = []
    ms = lambda t, name, f: _ml_med([results[(t, sd, name)][f] for sd in _ML_SEEDS if (t, sd, name) in results])   # noqa: E731
    al = lambda t, key: _ml_med([a_rows[(key[0], t, sd) + key[1:]][0] for sd in _ML_SEEDS if (key[0], t, sd) + key[1:] in a_rows])   # noqa: E731
    # ---- summary block (first)
    lines.append(f"ML C22a' v1 gpu={_ml_gpu_name()} torch={torch.__version__} t={_ml_elapsed():.0f}s cupti={'ok' if cupti_ok else 'fail'} sglang={HAVE_SGLANG} "
                 f"triton={triton.__version__} reps={_ML_REPS} smem_offline_T1(KB): pooler 147 ps1 74 ps2 49 | router 127 s2 45 (ptxas, ptx_audit method)")
    lines.append("ML columns: ln = main-stream start of the pre-expert norm from the first kernel (the figure ORDER / RPAR / TMA must cut), side_end = end of the side stream's "
                 "last kernel, main_idle = ln - end of the main stream's last kernel before it, pool / rout = span of the pooler's / router's kernels in that graph; "
                 "medians over 12 replays and 3 seeds (summary) / per seed (B lines). A lines: ops run alone (first start -> last end). g1 g2 on C lines.")
    for t in sorted(_ML_TS):
        row = [f"T={t}"]
        for name in _ML_NAMES:
            v = ms(t, name, "ln")
            if v is not None:
                row.append(f"{name}:{v:.1f}/{_ml_us(ms(t, name, 'side_end'))}/{_ml_us(ms(t, name, 'idle'))}")
        if len(row) > 1:
            lines.append("ML ln/side_end/main_idle " + " ".join(row))
    # ---- rules of fable-09 D4 section 2, evaluated on the numbers
    v1 = ms(1, "order", "side_end")
    p1 = ms(1, "base", "pool")
    if v1 is not None:
        lines.append(f"RULE ORDER (adopt if side_end <= 48 at T=1): side_end={v1:.1f} ln base={_ml_us(ms(1, 'base', 'ln'))} order={_ml_us(ms(1, 'order', 'ln'))} -> "
                     f"{'ADOPT' if v1 <= 48 else 'KILL'}")
    r1 = al(1, ("ra", "rparA"))
    if r1 is not None:
        best = {t: min([x for x in (al(t, ("ra", k)) for k in ("rparA", "rparB", "rparC")) if x is not None], default=None) for t in (1, 8, 16, 32)}
        lines.append(f"RULE RPAR (adopt if router alone <= 4.0 us): best-of-3-cfg alone T=1/8/16/32 = " + "/".join(_ml_us(best[t]) for t in (1, 8, 16, 32))
                     + f" vs base " + "/".join(_ml_us(al(t, ("ra", "base"))) for t in (1, 8, 16, 32)))
    if any(("ra", t, sd, "mm") in a_rows for t in (128, 256) for sd in _ML_SEEDS):
        lines.append("RULE RBIG (adopt if <= 10 us): alone T=128/256 base " + "/".join(_ml_us(al(t, ("ra", "base"))) for t in (128, 256))
                     + " mm " + "/".join(_ml_us(al(t, ("ra", "mm"))) for t in (128, 256)) + " bmA " + "/".join(_ml_us(al(t, ("ra", "bmA"))) for t in (128, 256))
                     + " bmB " + "/".join(_ml_us(al(t, ("ra", "bmB"))) for t in (128, 256)))
    if any(("pa", t, sd, "ps1") in a_rows for t in (1, 8, 16, 32) for sd in _ML_SEEDS):
        lines.append("RULE POOLSMEM (pooler smem 147/197 KB -> 74/98 KB lets router CTAs co-reside; compare side_end / ln of psm* with base, order, rpar): pooler alone T=1/8/16/32 base "
                     + "/".join(_ml_us(al(t, ("pa", "base"))) for t in (1, 8, 16, 32)) + " ps1 " + "/".join(_ml_us(al(t, ("pa", "ps1"))) for t in (1, 8, 16, 32))
                     + " ps2 " + "/".join(_ml_us(al(t, ("pa", "ps2"))) for t in (1, 8, 16, 32)))
    if any(("pa", t, sd, "tma4") in a_rows for t in (1, 8, 16, 32) for sd in _ML_SEEDS):
        lines.append("RULE TMAPOOL (low priority: keep only if >= 1.5 us faster ALONE; PTX has cp.async.bulk.tensor): pooler alone T=1/8/16/32 base "
                     + "/".join(_ml_us(al(t, ("pa", "base"))) for t in (1, 8, 16, 32)) + " tma4 " + "/".join(_ml_us(al(t, ("pa", "tma4"))) for t in (1, 8, 16, 32))
                     + " tma6 " + "/".join(_ml_us(al(t, ("pa", "tma6"))) for t in (1, 8, 16, 32)))
    if any(("oa", t, sd, "A") in a_rows for t in (1, 8, 16, 32) for sd in _ML_SEEDS):
        lines.append("RULE OSHOT (adopt if <= 5.5 us): o-proj alone T=1/8/16/32 cublas " + "/".join(_ml_us(al(t, ("oa", "cublas"))) for t in (1, 8, 16, 32))
                     + " A(bk1024) " + "/".join(_ml_us(al(t, ("oa", "A"))) for t in (1, 8, 16, 32)) + " B(bk512) " + "/".join(_ml_us(al(t, ("oa", "B"))) for t in (1, 8, 16, 32))
                     + " C(bn16) " + "/".join(_ml_us(al(t, ("oa", "C"))) for t in (1, 8, 16, 32)))
    # ---- C24 rule
    c_rows = [c for c in c_lines if isinstance(c, tuple)]
    for t in (32, 128, 256):
        rr = [c for c in c_rows if c[0] == t and "marlin" in c[4] and "largem" in c[4]]
        if rr:
            ratios = [c[4]["largem"]["e2e"] / c[4]["marlin"]["e2e"] for c in rr if c[4]["largem"]["e2e"] and c[4]["marlin"]["e2e"]]
            ok = all(r <= 0.8 for r in ratios) and len(ratios) == len(_ML_SEEDS)
            lines.append(f"RULE C24 T={t} (largem on if <= 0.8 x Marlin on all 3 seeds): ratios " + " ".join(f"{r:.2f}" for r in ratios) + f" -> {'LARGEM' if ok else 'MARLIN'}")
    lines += [c for c in c_lines if isinstance(c, str)]
    lines += [f"NOTE {n}" for n in _ML_OUT["N"]]
    # ---- check summaries (worst over seeds), compact: <name> rel/ids_diff[/d1 = deterministic] ; pooled outputs: bit1 = bitwise equal to the baseline pooler
    for t, cks in sorted(CHK_ALL.items()):
        toks = []
        for k in sorted({k for ck in cks for k in ck}):
            vals = [ck[k] for ck in cks if k in ck]
            if k == "order":
                toks.append(f"order bit{int(all(v[0] for v in vals))}/ids{sum(v[1] for v in vals)}")
            elif k.startswith("r"):
                toks.append(f"{k[1:]} {max(v[0] for v in vals):.0e}/ids{sum(v[1] for v in vals)}/w{max(v[2] for v in vals):.0e}" + ("/d1" if all(v[3] for v in vals) else "/d0"))
            elif k.startswith("p"):
                toks.append(f"{k[1:]} {max(v[0] for v in vals):.0e}/bit{int(all(v[1] for v in vals))}")
            else:
                toks.append(f"{k} {max(v[0] for v in vals):.0e}")
        lines.append(f"K layer T={t} | " + " ".join(toks))
    # ---- A lines (ops alone, median over seeds)
    for t in sorted(_ML_TS):
        keys = sorted({k[0:1] + k[3:] for k in a_rows if k[1] == t and k[0] in ("ra", "pa", "oa")})
        toks = []
        for key in keys:
            v = al(t, key)
            if v is not None:
                toks.append(f"{key[0]}.{key[1]}={v:.1f}")
        for k2, nm in (("ga", "routing"), ("na", "norm")):
            v = _ml_med([a_rows[(k2, t, sd)][0] for sd in _ML_SEEDS if (k2, t, sd) in a_rows])
            if v is not None:
                toks.append(f"{nm}={v:.1f}")
        if toks:
            lines.append(f"A layer T={t} alone_us | " + " ".join(toks))
    # ---- C24 lines per seed
    for c in c_rows:
        t, sd, e, rel, rowm = c
        for path in ("marlin", "largem"):
            if path in rowm:
                m = rowm[path]
                tail = f" rel_vs_marlin={rel:.1e}" if path == "largem" and rel is not None else ""
                lines.append(f"B layer T={t} seed={sd} E={e} | c24_{path} e2e={_ml_us(m['e2e'])} g1={_ml_us(m['g1'])} g2={_ml_us(m['g2'])}{tail}")
    # ---- B lines: priority 1 = decision variants, seed 11; 2 = the same on seeds 12 / 13; 3 = the other variants (seed 11). Filled until the 14 KB tail is full.
    core = ("base", "order", "rpar", "ordrpar", "mm", "ordmm", "bm", "ordbm")
    cand = []
    for ti, t in enumerate(sorted(_ML_TS)):
        for si, sd in enumerate(_ML_SEEDS):
            meta = results.get((t, sd, "_meta"))
            if not meta:
                continue
            for ni, name in enumerate(_ML_NAMES):
                m = results.get((t, sd, name))
                if not m:
                    continue
                pr = (1 if si == 0 else 2) if name in core else (3 if si == 0 else 4)
                cand.append((pr, ti, si, ni, _ml_fmt_b(t, sd, meta["E"], name, m)))
    room = _ML_MAX_BYTES - len("\n".join(lines)) - 80
    keep, dropped = [], 0
    for item in sorted(cand, key=lambda c: (c[0], c[1], c[2], c[3])):
        if room >= len(item[4]) + 1:
            keep.append(item)
            room -= len(item[4]) + 1
        else:
            dropped += 1
    if dropped:
        lines.append(f"NOTE {dropped} B lines left out for the 14 KB log tail (seeds 12 / 13 of the secondary variants first); medians over the seeds are in the ML lines")
    lines += [c[4] for c in sorted(keep, key=lambda c: (c[1], c[2], c[3]))]
    return "\n".join(lines)


def _ml_gpu_name():
    return "_".join(torch.cuda.get_device_name().split())


CHK_ALL = {}


def _ml_main(state):
    if not torch.cuda.is_available():
        print("ML skip: no CUDA")
        return
    s = _ml_setup(state)
    env = _MLEnv()
    results, a_rows, c_lines = {}, {}, []
    if s.sg and state.get("smallm") is not None:
        th = _ml_threading.Thread(target=_ml_lm_build, daemon=True)
        _ML_LM["thread"] = th
        th.start()
    cupti_ok = True
    for t in _ML_TS:
        if _ml_elapsed() > _ML_FRONT_S:
            _ml_note(f"T={t} dropped:budget")
            continue
        n0 = len(results)
        try:
            _ml_run_T(s, env, t, results, _ML_CHK, a_rows)
        except Exception as ex:  # noqa: BLE001
            _ml_note(f"T={t} failed: {_ml_where(ex)}")
        if len(results) == n0:
            cupti_ok = cupti_ok and False
        print(f"ML progress T={t} t={_ml_elapsed():.0f}s", flush=True)
    CHK_ALL.update(_ML_CHK)
    if s.sg and state.get("smallm") is not None and _ml_elapsed() < _ML_C24_S:
        try:
            _ml_run_c24(s, env, state, a_rows, c_lines)
        except Exception as ex:  # noqa: BLE001
            _ml_note(f"C24 failed: {_ml_where(ex)}")
    elif not (s.sg and state.get("smallm") is not None):
        _ml_note("C24 dropped: needs SGLang's Marlin and the small-M packed weights (dev run)")
    else:
        _ml_note("C24 dropped:budget")
    print(_ml_render(results, a_rows, c_lines, cupti_ok), flush=True)


_ML_CHK = {}
_ml_orig_prepare = prepare


def prepare(weights):  # noqa: F811
    global _ML_T0
    _ML_T0 = _ml_time.perf_counter()
    state = _ml_orig_prepare(weights)
    try:
        _ml_main(state)
    except Exception as ex:  # noqa: BLE001
        print(f"ML failed: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
    return state
