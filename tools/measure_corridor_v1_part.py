# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY block (tools/gen_measure_corridor_v1.py appends it after the unchanged sonnet_v5f code, the generated _mc_ twins and the inlined generate_input).
# Nothing here feeds kernel_fn: prepare() runs sonnet_v5f's prepare(), then times the C33 variants of design/itachi-corridor.md on generate_input-recipe inputs
# (3 seeds per shape, read-only L2 flush, CUDA-graph replays of the whole layer, replay overhead subtracted) and prints the S / D / H / B lines.
# ---------------------------------------------------------------------------------------------------------------------
import statistics as _mc_stats
import threading as _mc_threading
import time as _mc_time

_MC_T0 = _mc_time.perf_counter()
_MC_BUDGET_S = 280.0         # prepare() ends by here (the judge's limit is 300 s); variants / shapes / seeds that do not fit are dropped, not run late
_MC_SEEDS = (11, 12, 13)
_MC_REPS = 5                 # timed replays per (variant, seed); the median is that seed's value
_MC_STAMP_REPS = 6
_MC_FLUSH_MB = 128           # read-only L2 flush buffer (> the 50 MB L2)
_MC_MAX_BYTES = 13800
_MC_LIGHT_BYTES = 65536      # TOOLS_LIGHT: first <= 64 KB per CTA
# judged shape index, label, case; processing order = priority when the budget runs short (the decision is taken on the first two)
_MC_SHAPES = [
    (0, "d1s", dict(batch=1, ctx=80000, layer="sliding", dist="soft", page=1)),
    (1, "d1f", dict(batch=1, ctx=80000, layer="full", dist="soft", page=1)),
    (7, "v1f", dict(batch=1, ctx=80000, layer="full", dist="soft", page=1, q_len=8)),
    (2, "d16s", dict(batch=16, ctx=80000, layer="sliding", dist="soft", page=1, spread="agentic")),
    (3, "d16f", dict(batch=16, ctx=80000, layer="full", dist="soft", page=1, spread="agentic")),
    (4, "d32s", dict(batch=32, ctx=80000, layer="sliding", dist="soft", page=1, spread="agentic")),
    (5, "d32f", dict(batch=32, ctx=80000, layer="full", dist="soft", page=1, spread="agentic")),
]
_MC_WEIGHT = {"d1s": 46, "d1f": 16}
# name -> (S3_PDL, prefetch mode 0 off / 1 light / 2 full, priority streams)
_MC_VARIANTS = [
    ("PDL_OFF", (False, 0, False)), ("WAITFIRST", (True, 0, False)), ("TOOLS_LIGHT", (True, 1, False)), ("TOOLS_FULL", (True, 2, False)),
    ("PRIO", (False, 0, True)), ("TOOLS_LIGHT+PRIO", (True, 1, True)),
]
_MC_STAMPED = ("PDL_OFF", "WAITFIRST", "TOOLS_LIGHT", "TOOLS_FULL")
_MC_TWIN_NAMES = ("_gemv_splitk", "_router_fix_kernel", "_unpool_kernel", "_pre_norm_kernel", "_epilogue_kernel")
_MC_NOTES = []


def _mc_elapsed():
    return _mc_time.perf_counter() - _MC_T0


def _mc_note(text):
    _MC_NOTES.append(" ".join(str(text).split())[:220])


def _mc_where(ex):
    return f"{type(ex).__name__}: {str(ex)[:90]}".replace("\n", " ")


def _mc_med(xs):
    xs = [x for x in xs if x is not None]
    return _mc_stats.median(xs) if xs else None


@triton.jit
def _mc_nop(p):
    tl.store(p, 0)


# ================================================================================================================ variants: the launch shims
_MC_TWINS = {n: globals()["_mc" + n] for n in _MC_TWIN_NAMES}
_MC_ORIG = {n: globals()[n] for n in _MC_TWIN_NAMES}
_MC_CUR = dict(pdl=False, pf=0)


def _mc_pfl(name, mode, a, kw):
    """Prefetch lines (128 bytes, a power of two) per weight row of this launch: LIGHT = the first <= 64 KB of the CTA's slice, FULL = the whole slice."""
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
    lines = rowl if mode == 2 else min(rowl, max(1, _MC_LIGHT_BYTES // (rows * 128 * nmat)))
    return triton.next_power_of_2(max(1, lines))


class _McShim:
    """Stands in for one of our Triton kernels in the module namespace: forwards the launch to its _mc_ twin with the variant's PDL attribute and prefetch depth."""

    def __init__(self, name):
        self.name = name

    def __getitem__(self, grid):
        twin = _MC_TWINS[self.name][grid]
        name = self.name

        def launch(*a, **kw):
            kw = dict(kw)
            if _MC_CUR["pdl"]:
                kw.update(PDL=True, launch_pdl=True)
            if _MC_CUR["pf"]:
                kw["PFL"] = _mc_pfl(name, _MC_CUR["pf"], a, kw)
            return twin(*a, **kw)
        return launch


def _mc_capable():
    """The variants differ only on a Hopper GPU with Triton's PDL API (griddepcontrol is sm_90+); elsewhere every variant is the board's code (the dev card)."""
    return bool(HAVE_PDL_API and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9)


_MC_EXT = {"orig": None, "twin": None, "err": None}
_MC_PLAIN = [None]


def _mc_apply(vname, state):
    """Flip the module switches for one variant (S3_PDL, the twin shims, the small-M extension, the side stream); the caller re-captures."""
    global S3_PDL
    pdl, pf, prio = dict(_MC_VARIANTS)[vname]
    S3_PDL = bool(pdl)
    cap = _mc_capable()
    _MC_CUR.update(pdl=bool(pdl) and cap, pf=pf if cap else 0)
    for n in _MC_TWIN_NAMES:
        globals()[n] = _McShim(n) if (pf and cap) else _MC_ORIG[n]
    sm = state.get("smallm")
    if sm is not None and _MC_EXT["orig"] is not None:
        if pf and _MC_EXT["twin"] is not None and cap:
            _MC_EXT["twin"].smallm_set_pref(pf)
            sm["ext"] = _MC_EXT["twin"]
        else:
            sm["ext"] = _MC_EXT["orig"]
    return prio


def _mc_build_twin_ext(state):
    """Second small-M extension (GEMM2 prefetch before pdl_wait), built in a thread beside the first shapes; failure leaves the original in every variant."""
    try:
        import hashlib

        from torch.utils.cpp_extension import load_inline

        src = _MC_SM_DEVICE_SRC + _MC_SM_HOST_SRC
        tag = hashlib.sha1(src.encode()).hexdigest()[:12]
        ext = load_inline(name=f"okc_smallm_c33_{tag}", cpp_sources=_MC_SM_CPP_SRC, cuda_sources=src,
                          functions=["smallm_gemm", "smallm_route", "smallm_init", "smallm_set_pref"], extra_cuda_cflags=["-O3", "-w"], verbose=False)
        ext.smallm_init()
        _MC_EXT["twin"] = ext
    except Exception as ex:  # noqa: BLE001
        _MC_EXT["err"] = _mc_where(ex)


# ================================================================================================================ graph inspection (priority / programmatic edges)
def _mc_graph_info(g):
    """(kernel nodes, {priority: count}, programmatic edges, edges) of a graph captured with keep_graph=True through libcudart (ctypes); None when unavailable."""
    try:
        import ctypes

        rt = None
        for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
            try:
                rt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if rt is None:
            return None
        h = ctypes.c_void_p(g.raw_cuda_graph())
        n = ctypes.c_size_t(0)
        if rt.cudaGraphGetNodes(h, None, ctypes.byref(n)) != 0:
            return None
        nodes = (ctypes.c_void_p * n.value)()
        rt.cudaGraphGetNodes(h, nodes, ctypes.byref(n))
        prio, kn = {}, 0
        for i in range(n.value):
            t = ctypes.c_int(-1)
            rt.cudaGraphNodeGetType(nodes[i], ctypes.byref(t))
            if t.value != 0:
                continue
            kn += 1
            val = (ctypes.c_int * 16)()
            r = rt.cudaGraphKernelNodeGetAttribute(nodes[i], 8, val)    # cudaLaunchAttributePriority
            key = val[0] if r == 0 else f"err{r}"
            prio[key] = prio.get(key, 0) + 1
        ne = ctypes.c_size_t(0)
        pe = None
        if rt.cudaGraphGetEdges(h, None, None, None, ctypes.byref(ne)) == 0:
            pe = 0
            if ne.value:
                fr, to, ed = (ctypes.c_void_p * ne.value)(), (ctypes.c_void_p * ne.value)(), (ctypes.c_ubyte * (8 * ne.value))()
                if rt.cudaGraphGetEdges(h, fr, to, ed, ctypes.byref(ne)) == 0:
                    pe = sum(1 for i in range(ne.value) if ed[8 * i + 2] == 1)    # cudaGraphEdgeTypeProgrammatic
                else:
                    pe = None
        return kn, prio, pe, ne.value
    except Exception:  # noqa: BLE001
        return None


# ================================================================================================================ timing / profile session (after trace_runtime.py)
class _McEnv:
    def __init__(self):
        self.flush = torch.ones(_MC_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        self.sink = torch.zeros((), device="cuda", dtype=torch.float32)
        self.hi = torch.cuda.Stream(priority=-1)
        self.lo = torch.cuda.Stream(priority=0)

    def cold(self):
        torch.sum(self.flush, dim=0, out=self.sink)   # read-only: pushes the previous replay's data out of L2

    def replay_us(self, g):
        self.cold()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1000.0

    def overhead(self, reps=21):
        """A graph replay's fixed cost (one empty kernel), which the judge subtracts (about 4 us). -> median us."""
        buf = torch.zeros(4, device="cuda", dtype=torch.int32)
        g = torch.cuda.CUDAGraph()
        _mc_nop[(1,)](buf)
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            _mc_nop[(1,)](buf)
        torch.cuda.synchronize()
        return _mc_med([self.replay_us(g) for _ in range(reps)])


_mc_marks = [0]


def _mc_mark():
    _mc_marks[0] += 1
    torch.cuda._sleep(1000)


def _mc_gpu_rows(prof):
    """[(start_ns, end_ns, stream, name)] of every kernel / memset / memcpy record of the current device."""
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
    return rows


def _mc_short(name):
    base = name.split("(")[0]
    base = base.split("<")[0] if "<" in base and not base.startswith("void") else base.replace("void ", "").split("<")[0]
    base = base.split("::")[-1] if "::" in base else base
    return (base or name)[:16]


def _mc_chain(rows):
    """One replay's kernels: [(short name, start, end, stream)] in start order, memsets / copies included."""
    return sorted(((_mc_short(r[3]), r[0], r[1], r[2]) for r in rows), key=lambda t: (t[1], t[2]))


def _mc_gaps(chain):
    """-> [(name, dur us, gap us)]: gap = start - end of the previous kernel on the SAME stream (negative: it started before its predecessor ended, i.e. a PDL overlap); first of a stream: gap to the replay's start."""
    t0 = chain[0][1]
    last_end, out = {}, []
    for name, s, e, st in chain:
        prev = last_end.get(st)
        out.append((name, (e - s) / 1000.0, (s - (prev if prev is not None else t0)) / 1000.0))
        last_end[st] = e
    return out


def _mc_stamps(env, graphs, seed_tag):
    """Profile session over the replays of `graphs` {variant: graph}: -> {variant: (n kernels, span us, [(name, dur, gap)] of the modal chain medians, sum gaps, n overlapped edges)}."""
    from torch.profiler import ProfilerActivity, profile

    units = []
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for vname, g in graphs.items():
            for _ in range(2):
                env.cold()
                g.replay()
            torch.cuda.synchronize()
            lo = _mc_marks[0]
            for _ in range(_MC_STAMP_REPS):
                env.cold()
                _mc_mark()
                g.replay()
                _mc_mark()
            torch.cuda.synchronize()
            units.append((vname, lo, _mc_marks[0]))
    rows = sorted(_mc_gpu_rows(prof), key=lambda r: (r[0], r[1]))
    marks = [r for r in rows if "spin_kernel" in r[3]]
    base = units[0][1]
    if len(marks) != _mc_marks[0] - base:
        raise RuntimeError(f"markers {len(marks)} of {_mc_marks[0] - base}")
    body = [r for r in rows if "spin_kernel" not in r[3]]
    res = {}
    for vname, lo, hi in units:
        mm = marks[lo - base:hi - base]
        reps = []
        for i in range(len(mm) // 2):
            k = [r for r in body if mm[2 * i][1] <= r[0] and r[1] <= mm[2 * i + 1][0] + 1]
            if k:
                reps.append(_mc_gaps(_mc_chain(k)))
        if not reps:
            continue
        sigs = [tuple(n for n, _, _ in r) for r in reps]
        best = max(set(sigs), key=sigs.count)
        use = [r for r, s in zip(reps, sigs) if s == best]
        med = [(best[i], _mc_med([r[i][1] for r in use]), _mc_med([r[i][2] for r in use])) for i in range(len(best))]
        res[vname] = dict(chain=med, sumdur=sum(d for _, d, _ in med), sumgap=sum(g for _, _, g in med), overlapped=sum(1 for _, _, g in med if g < -0.05), n=len(best))
    return res


# ================================================================================================================ one (shape, seed)
def _mc_case_data(case, seed, state):
    data = _tr_generate_input(**case, seed=seed, local_start=0, device="cuda")
    out = kernel_fn(data, state)                       # the judge gives each input the 32-expert range its routing sends the most pairs to
    ids = out[2]
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    data["local_start"].fill_(LOCAL_EXPERTS * int(pairs.argmax()))
    torch.cuda.synchronize()
    return data


def _mc_capture(env, data, state, vname, keep):
    """Apply the variant, two eager runs (compiles), capture the whole layer. -> (graph, outputs of the capture)."""
    prio = _mc_apply(vname, state)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        kernel_fn(data, state)
        kernel_fn(data, state)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(keep_graph=keep)
    with torch.cuda.graph(g, stream=env.hi if prio else env.lo):
        outs = kernel_fn(data, state)
    torch.cuda.synchronize()
    return g, outs


def _mc_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def _mc_seed(env, label, case, seed, res, st, ovh, deadline, stamp):
    data = _mc_case_data(case, seed, st["state"])
    graphs, outs = {}, {}
    names = [n for n, _ in _MC_VARIANTS if n not in st["dead"]]
    for vname in names:
        fresh = vname not in st["compiled"]
        if fresh and _mc_elapsed() + 1.3 * st["est"] > deadline:
            st["dead"][vname] = "skip budget"
            continue
        t1 = _mc_time.perf_counter()
        try:
            g, o = _mc_capture(env, data, st["state"], vname, keep=(stamp and seed == _MC_SEEDS[0]))
            graphs[vname], outs[vname] = g, o
            st["compiled"].add(vname)
        except Exception as ex:  # noqa: BLE001
            st["dead"][vname] = "fail " + _mc_where(ex)
            _mc_note(f"{vname}: {st['dead'][vname]}")
        finally:
            if fresh:
                st["est"] = max(st["est"], (_mc_time.perf_counter() - t1) / 1.0)
    # correctness: every variant's replay output against the board's (and the board against itself: PDL_OFF twice)
    chk = {}
    if "PDL_OFF" in graphs:
        graphs["PDL_OFF"].replay()
        torch.cuda.synchronize()
        ref = [t.clone() for t in outs["PDL_OFF"]]
        graphs["PDL_OFF"].replay()
        torch.cuda.synchronize()
        det = all(torch.equal(a, b) for a, b in zip(ref, outs["PDL_OFF"]))
        for vname, g in graphs.items():
            g.replay()
            torch.cuda.synchronize()
            same = all(torch.equal(a, b) for a, b in zip(ref, outs[vname]))
            chk[vname] = "OK" if same else f"DIFF({_mc_rel(outs[vname][0], ref[0]):.1e}{'' if det else ',board-self-DIFF'})"
        chk["_det"] = det
    # timing: interleaved rounds, the order reversed every other round, the first round is a warm-up
    samples = {n: [] for n in graphs}
    for rep in range(_MC_REPS + 1):
        order = list(graphs) if rep % 2 == 0 else list(graphs)[::-1]
        for vname in order:
            us = env.replay_us(graphs[vname])
            if rep > 0:
                samples[vname].append(us)
    for vname in graphs:
        r = res.setdefault(vname, dict(us=[], chk=[], seeds=[]))
        r["us"].append(max(0.01, _mc_med(samples[vname]) - ovh))
        r["chk"].append(chk.get(vname, "-"))
        r["seeds"].append(seed)
    info = {}
    if stamp and seed == _MC_SEEDS[0]:
        for vname, g in graphs.items():
            info[vname] = _mc_graph_info(g)
        try:
            res["_stamps"] = _mc_stamps(env, {n: graphs[n] for n in _MC_STAMPED if n in graphs}, label)
        except Exception as ex:  # noqa: BLE001
            _mc_note(f"stamps {label}: {_mc_where(ex)}")
        res["_info"] = info
    graphs.clear()
    outs.clear()
    del data
    torch.cuda.empty_cache()


# ================================================================================================================ output
def _mc_fmt_us(x):
    return "-" if x is None else f"{x:.1f}"


def _mc_decision(results):
    """The decision table of design/itachi-corridor.md section 7 on T = 1 decode (sliding : full = 46 : 16 by shape weight)."""
    gains = {}
    for vname, _ in _MC_VARIANTS:
        tot, wsum = 0.0, 0
        for label, w in _MC_WEIGHT.items():
            r = results.get(label)
            if not r or "PDL_OFF" not in r or vname not in r:
                tot = None
                break
            tot += w * (_mc_med(r["PDL_OFF"]["us"]) - _mc_med(r[vname]["us"]))
            wsum += w
        gains[vname] = None if tot is None else tot / wsum
    if gains.get("TOOLS_LIGHT") is None:
        return "D T1 decode: no reading (TOOLS_LIGHT / PDL_OFF missing)", gains
    gl, gf, gw, gp = (gains.get(k) for k in ("TOOLS_LIGHT", "TOOLS_FULL", "WAITFIRST", "PRIO"))
    acts = []
    if gl >= 3.0 and (gf is None or gf >= gl - 0.3):
        acts.append("TOOLS_LIGHT >= 3 us better, FULL no worse -> build sonnet_v5g with TOOLS_LIGHT on all our kernels")
    if gf is not None and gl - gf >= 1.0:
        acts.append("TOOLS_FULL clearly worse than LIGHT -> corridor congestion confirmed: keep LIGHT, cap smaller")
    if abs(gl) < 1.5 and abs(gw if gw is not None else 0.0) < 1.5:
        acts.append("WAITFIRST ~ TOOLS_LIGHT ~ OFF -> handovers are not the loss: ALT-1 fusion first (C34)")
    if gp is not None and gp >= 1.0:
        acts.append("PRIO >= 1 us -> add priority to v5g")
    if not acts:
        acts.append("no row of the table fires (0 < gain < 3 us): read the H stamps, then decide")
    g_txt = " ".join(f"{k}={v:+.1f}" for k, v in gains.items() if v is not None and k != "PDL_OFF")
    return f"D T1 decode gain vs PDL_OFF (us, weighted 46:16): {g_txt} => " + "; ".join(acts), gains


def _mc_render(results, meta):
    S, H, B = [], [], []
    names = [n for n, _ in _MC_VARIANTS]
    S.append(f"S corridor v1: gpu={meta['gpu']} used={meta['secs']:.0f}s of {_MC_BUDGET_S:.0f}s ovh={meta['ovh']:.2f}us(subtracted) seeds={len(_MC_SEEDS)} reps={_MC_REPS} capable={meta['capable']}"
             f" smallm_twin={meta['twin']} prefetch_kernels={','.join(n.lstrip('_') for n in _MC_TWIN_NAMES)}+small-M GEMM2")
    S.append("S untouched: cuBLAS nvjet / splitKreduce, SGLang norm_residual / routing kernels, Marlin (T=32), attention kernels, small-M GEMM1 (weights depend on the routing)")
    dead = meta.get("dead", {})
    if dead:
        S.append("S dropped variants: " + "; ".join(f"{k}: {v[:60]}" for k, v in dead.items()))
    for idx, label, case in _MC_SHAPES:
        r = results.get(label)
        if not r or "PDL_OFF" not in r:
            continue
        base = _mc_med(r["PDL_OFF"]["us"])
        parts = []
        for n in names:
            if n in r and r[n]["us"]:
                m = _mc_med(r[n]["us"])
                oks = r[n]["chk"]
                flag = "" if all(c in ("OK", "-") for c in oks) else " DIFF"
                parts.append(f"{n}={m:.1f}" + ("" if n == "PDL_OFF" else f"(x{base / m:.3f}{flag})"))
        S.append(f"S #{idx} {label} layer us: " + " ".join(parts))
    d_line, gains = _mc_decision(results)
    S.append(d_line)
    for label, r in results.items():
        info = r.get("_info") or {}
        if info:
            S.append(f"S graph {label}: " + " ".join(f"{n}:" + ("n/a" if v is None else f"k{v[0]} prio{dict(v[1])} pdl_edges={v[2]}/{v[3]}") for n, v in info.items() if n in ("PDL_OFF", "WAITFIRST", "TOOLS_LIGHT", "PRIO")))
    prio_seen = [(lab, r["_info"]["PRIO"]) for lab, r in results.items() if (r.get("_info") or {}).get("PRIO") is not None]
    if prio_seen:
        lab, v = prio_seen[0]
        hi = sum(c for k, c in v[1].items() if isinstance(k, int) and k < 0)
        S.append(f"S PRIO survives graph capture: {'YES' if hi else 'NO'} ({hi} of {v[0]} kernel nodes carry a high priority, {lab})")
    else:
        S.append("S PRIO survives graph capture: unchecked (graph inspection unavailable or the variant was dropped)")
    pe = [(lab, n, v[2]) for lab, r in results.items() for n, v in (r.get("_info") or {}).items() if v is not None and v[2] is not None and n in ("WAITFIRST", "TOOLS_LIGHT")]
    if pe:
        S.append("S programmatic (PDL) edges in the captured graphs: " + " ".join(f"{lab}/{n}={c}" for lab, n, c in pe))
    checks = {n: [c for lab, r in results.items() if n in r for c in r[n]["chk"]] for n in names}
    S.append("S bitwise vs the board: " + " ".join(f"{n}={'OK' if cs and all(c in ('OK', '-') for c in cs) else ('DIFF' if cs else '-')}" for n, cs in checks.items())
)
    for note in _MC_NOTES[:6]:
        S.append("S note: " + note)
    for label, r in results.items():
        st = r.get("_stamps") or {}
        for vname, v in st.items():
            cells = " ".join(f"{n}:{d:.1f}/{g:+.1f}" for n, d, g in v["chain"][:40])
            H.append(f"H {label} {vname} n={v['n']} sumdur={v['sumdur']:.1f} sumgap={v['sumgap']:+.1f} overlapped_edges={v['overlapped']} | {cells}")
    for idx, label, case in _MC_SHAPES:
        r = results.get(label)
        if not r:
            continue
        for k, seed in enumerate(_MC_SEEDS):
            for n in names:
                x = r.get(n)
                if not x or k >= len(x["us"]) or "PDL_OFF" not in r or k >= len(r["PDL_OFF"]["us"]):
                    continue
                B.append(f"B layer {label} seed={seed} | {n} us={x['us'][k]:.2f} x={r['PDL_OFF']['us'][k] / x['us'][k]:.3f} {x['chk'][k]}")
    head = "\n".join(S + H)
    out, dropped, size = [head], 0, len(head) + 1
    for line in B:
        if size + len(line) + 1 > _MC_MAX_BYTES - 80:
            dropped += 1
            continue
        out.append(line)
        size += len(line) + 1
    if dropped:
        out.insert(1, f"S {dropped} B lines dropped for the 14 KB cap (the S lines hold the seed medians)")
    text = "\n".join(out)
    if len(text.encode()) > _MC_MAX_BYTES:      # H lines too long: cut them
        keep = []
        size = len("\n".join(S)) + 1
        for line in H:
            if size + len(line) + 1 <= _MC_MAX_BYTES - 200:
                keep.append(line)
                size += len(line) + 1
        text = "\n".join(S + keep + ["S H lines cut for the 14 KB cap"])
    return text


# ================================================================================================================ the study
def _mc_main(state):
    results, meta = {}, dict(gpu="n/a", ovh=0.0, secs=0.0, capable=False, twin="no", dead={})
    try:
        if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
            print("S corridor v1: no CUDA device (or capturing): nothing measured", flush=True)
            return
        meta["gpu"] = "_".join(torch.cuda.get_device_name().split())
        meta["capable"] = _mc_capable()
        env = _McEnv()
        meta["ovh"] = env.overhead()
        _MC_PLAIN[0] = state.get("side_stream")
        sm = state.get("smallm")
        if sm is not None:
            _MC_EXT["orig"] = sm["ext"]
            if meta["capable"]:
                th = _mc_threading.Thread(target=_mc_build_twin_ext, args=(state,), daemon=True)
                th.start()
                _MC_EXT["thread"] = th
        st = dict(state=state, dead={}, compiled=set(), est=10.0)
        deadline = _MC_BUDGET_S - 14.0
        for idx, label, case in _MC_SHAPES:
            if _mc_elapsed() > deadline - 12.0:
                _mc_note(f"#{idx} {label} skipped (budget)")
                continue
            r = results.setdefault(label, {})
            for seed in _MC_SEEDS:
                if _mc_elapsed() > deadline - 8.0 and r:
                    _mc_note(f"#{idx} {label}: seeds after {seed} skipped (budget)")
                    break
                try:
                    _mc_seed(env, label, case, seed, r, st, meta["ovh"], deadline, stamp=label in ("d1s", "d1f"))
                except Exception as ex:  # noqa: BLE001
                    _mc_note(f"#{idx} {label} seed {seed}: {_mc_where(ex)}")
                    torch.cuda.empty_cache()
            print(f"MC progress {label} t={_mc_elapsed():.0f}s", flush=True)
        th = _MC_EXT.get("thread")
        if th is not None:
            meta["twin"] = "ok" if _MC_EXT["twin"] is not None else ("pending" if th.is_alive() else f"failed {_MC_EXT['err']}")
        meta["dead"] = st["dead"]
        _mc_apply("PDL_OFF", state)                      # leave the state as the board built it
        meta["secs"] = _mc_elapsed()
        print(_mc_render(results, meta), flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"S corridor v1 FAILED: {_mc_where(ex)}", flush=True)
        try:
            meta["secs"] = _mc_elapsed()
            print(_mc_render(results, meta), flush=True)
        except Exception:  # noqa: BLE001
            pass


_mc_orig_prepare = prepare


def prepare(weights):  # noqa: F811
    global _MC_T0
    _MC_T0 = _mc_time.perf_counter()
    state = _mc_orig_prepare(weights)
    if not globals().get("_MC_NO_AUTORUN"):
        try:
            _mc_main(state)
        except Exception as ex:  # noqa: BLE001
            print(f"S corridor v1 FAILED: {_mc_where(ex)}", flush=True)
    return state
