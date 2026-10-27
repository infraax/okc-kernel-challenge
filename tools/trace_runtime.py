# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY trace block (TRACE v1, design/breakdown/TRACE-FORMAT.md). Appended verbatim by tools/make_trace.py
# after the unchanged kernel file; the lines above this block (_TR_* settings, the inlined _tr_generate_input) are
# generated. Nothing here feeds kernel_fn: it only times it, in prepare() (or at import, for problems without prepare).
# ---------------------------------------------------------------------------------------------------------------------
import time as _tr_time

import torch as _tr_torch

_TR_T0 = _tr_time.perf_counter()
_TR_FLUSH_MB = 128          # read-only L2 flush buffer (> the H200's 50 MB L2), like the judge and measure_v3
_TR_NAME_CHARS = 90
_TR_MIN_REPLAYS = 5
_TR_FORCE_CHROME = False    # tests: read the profile from the chrome-trace export instead of kineto_results
_TR_FORCE_EAGER = False     # tests: profile eager calls instead of graph replays


def _tr_median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _tr_mode(xs):
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=lambda k: (counts[k], -k if isinstance(k, int) else 0))


def _tr_oneline(text, limit=160):
    return " ".join(str(text).split())[:limit]


def _tr_tok(text):
    return "_".join(str(text).split())


def _tr_label(case):
    parts = []
    for k, v in case.items():
        if k in ("seed", "weight", "repeats"):
            continue
        parts.append(f"{k}={'_'.join(str(v).split())}")
    return ",".join(parts)


def _tr_call(data, state):
    return kernel_fn(data, state) if _TR_KFN_ARGS == 2 else kernel_fn(data)


# ---------------------------------------------------------------------------------------------------- the profile
def _tr_gpu_rows_kineto(prof):
    """[(start_ns, end_ns, stream, name)] of every kernel / memset / memcpy record of the current device."""
    kr = prof.profiler.kineto_results
    cuda = _tr_torch._C._autograd.DeviceType.CUDA
    dev = _tr_torch.cuda.current_device()
    rows = []
    for e in kr.events():
        if e.device_type() != cuda or e.device_index() != dev:
            continue
        kind = str(e.activity_type()).lower()
        if "kernel" not in kind and "memset" not in kind and "memcpy" not in kind:
            continue
        s = int(e.start_ns())
        rows.append((s, s + int(e.duration_ns()), int(e.device_resource_id()), e.name()))
    return rows


def _tr_gpu_rows_chrome(prof):
    import json as _json
    import tempfile as _tempfile

    dev = _tr_torch.cuda.current_device()
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
        s = int(round(float(e["ts"]) * 1000.0))
        rows.append((s, s + int(round(float(e["dur"]) * 1000.0)), int(a.get("stream", 0)), e["name"]))
    return rows


def _tr_gpu_rows(prof):
    if not _TR_FORCE_CHROME:
        try:
            rows = _tr_gpu_rows_kineto(prof)
            if rows:
                return rows
        except Exception:  # noqa: BLE001  (private API moved: use the exported trace)
            pass
    return _tr_gpu_rows_chrome(prof)


def _tr_split(marks, lo, hi):
    """Marker rows [lo, hi) of one shape -> [(marker1_end, marker2_start)] per replay."""
    mm = marks[lo:hi]
    return [(mm[2 * i][1], mm[2 * i + 1][0]) for i in range(len(mm) // 2)]


def _tr_union(segs):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(segs):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def _tr_analyze(kern, m1_end, m2_start):
    """One replay: positions in start order, gap vs the previous end on ANY stream, overlap with other streams (ns)."""
    kern = sorted(kern, key=lambda r: (r[0], r[1]))
    t0 = kern[0][0]
    ranks = {}
    ks, runmax = [], None
    for i, (s, e, st, name) in enumerate(kern):
        ranks.setdefault(st, len(ranks))
        gap = 0 if runmax is None else max(0, s - runmax)
        runmax = e if runmax is None else max(runmax, e)
        segs = [(max(s2, s), min(e2, e)) for (s2, e2, st2, _) in kern if st2 != st and s2 < e and e2 > s]
        ks.append(dict(s=s - t0, d=e - s, g=gap, st=ranks[st], o=_tr_union(segs), name=name))
    end = max(r[1] for r in kern)
    return dict(kernels=ks, busy=_tr_union([(r[0], r[1]) for r in kern]), gaps=sum(k["g"] for k in ks),
                span=end - t0, head=kern[0][0] - m1_end, tail=m2_start - end)


def _tr_aggregate(reps):
    """Medians over the replays that share the modal kernel-name sequence."""
    sigs = [tuple(k["name"] for k in r["kernels"]) for r in reps]
    counts = {}
    for s in sigs:
        counts[s] = counts.get(s, 0) + 1
    best = max(counts, key=lambda s: (counts[s], -sigs.index(s)))
    use = [r for r, s in zip(reps, sigs) if s == best]
    ks = []
    for p, name in enumerate(best):
        col = [r["kernels"][p] for r in use]
        ks.append(dict(name=name, s=_tr_median([c["s"] for c in col]), d=_tr_median([c["d"] for c in col]),
                       g=_tr_median([c["g"] for c in col]), o=_tr_median([c["o"] for c in col]),
                       st=_tr_mode([c["st"] for c in col])))
    tot = {k: _tr_median([r[k] for r in use]) for k in ("busy", "gaps", "span", "head", "tail")}
    return dict(kernels=ks, rep_ok=len(use), rep_all=len(reps), **tot)


# ---------------------------------------------------------------------------------------------------- one shape
class _TrEnv:
    def __init__(self):
        self.dev = _tr_torch.cuda.current_device()
        self.flush = _tr_torch.ones(_TR_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=_tr_torch.float32)
        self.sink = _tr_torch.zeros((), device="cuda", dtype=_tr_torch.float32)
        self.cupti = None      # "graph" | "eager" | "fail:<why>"

    def cold(self):
        _tr_torch.sum(self.flush, dim=0, out=self.sink)   # read-only: pushes the previous replay's data out of L2


_tr_marks = [0]


def _tr_mark():
    _tr_marks[0] += 1
    _tr_torch.cuda._sleep(1000)


def _tr_replays(env, run, replays):
    """Inside the profiling session: replays x (L2 flush, marker, event, run, event, marker); the markers delimit each replay's
    kernels, the events give that replay's wall. -> (first marker index, end marker index, [wall us per replay])."""
    for _ in range(2):
        env.cold()
        run()
    _tr_torch.cuda.synchronize()
    lo, evs = _tr_marks[0], []
    for _ in range(replays):
        env.cold()
        _tr_mark()
        a, b = _tr_torch.cuda.Event(enable_timing=True), _tr_torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        _tr_mark()
        evs.append((a, b))
    _tr_torch.cuda.synchronize()
    return lo, _tr_marks[0], [a.elapsed_time(b) * 1000.0 for a, b in evs]


def _tr_wall(env, g, replays):
    for _ in range(3):
        env.cold()
        g.replay()
    ts = []
    for _ in range(replays):
        env.cold()
        a, b = _tr_torch.cuda.Event(enable_timing=True), _tr_torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    return _tr_median(ts)


def _tr_make_data(case):
    if _TR_LS_FROM_IDS:
        return _tr_generate_input(**{**case, "local_start": 0})
    return _tr_generate_input(**case)


def _tr_fix_local_start(data, state):
    """decode_layer: the judge gives each input the 32-expert range its routing sends the most pairs to. The kernel's own
    top-8 ids (global ids, independent of local_start) of one eager call pick the same range."""
    out = _tr_call(data, state)
    ids = out[2]
    pairs = _tr_torch.bincount((ids.long() // 32).flatten(), minlength=8)
    data["local_start"].fill_(32 * int(pairs.argmax()))
    _tr_torch.cuda.synchronize()


def _tr_shape(env, case, state, replays, mode):
    """Runs inside the profiling session. -> dict(wall=µs, marks=(lo, hi)). Raises on any failure."""
    data = _tr_make_data(case)
    if _TR_LS_FROM_IDS:
        _tr_fix_local_start(data, state)
    side = _tr_torch.cuda.Stream()
    side.wait_stream(_tr_torch.cuda.current_stream())
    with _tr_torch.cuda.stream(side):
        _tr_call(data, state)
        _tr_call(data, state)
    _tr_torch.cuda.current_stream().wait_stream(side)
    _tr_torch.cuda.synchronize()
    g = _tr_torch.cuda.CUDAGraph()
    with _tr_torch.cuda.graph(g):
        _tr_call(data, state)
    _tr_torch.cuda.synchronize()
    if mode == "none":                      # CUPTI unavailable: graph wall only
        return dict(wall=_tr_wall(env, g, max(replays, 10)))
    run = g.replay if mode == "graph" else (lambda: _tr_call(data, state))
    lo, hi, walls = _tr_replays(env, run, replays)
    res = dict(marks=(lo, hi), reps_wall=walls)
    res["wall"] = _tr_median(walls) if mode == "graph" else _tr_wall(env, g, max(replays, 10))
    del g, data
    return res


def _tr_selftest(env):
    """A tiny two-kernel graph (and an eager pair) in their own short session: does CUPTI see the kernels inside graph replays?
    -> "graph" | "eager" | "fail:<why>"."""
    from torch.profiler import ProfilerActivity, profile

    x = _tr_torch.ones(1024, device="cuda")
    side = _tr_torch.cuda.Stream()
    side.wait_stream(_tr_torch.cuda.current_stream())
    with _tr_torch.cuda.stream(side):
        (x * 2.0) + 1.0
    _tr_torch.cuda.current_stream().wait_stream(side)
    g = _tr_torch.cuda.CUDAGraph()
    with _tr_torch.cuda.graph(g):
        (x * 2.0) + 1.0
    _tr_torch.cuda.synchronize()
    seen = {}
    for mode, run in (("graph", g.replay), ("eager", lambda: (x * 2.0) + 1.0)):
        base = _tr_marks[0]
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            lo, hi, _ = _tr_replays(env, run, 2)
        rows = sorted(_tr_gpu_rows(prof), key=lambda r: (r[0], r[1]))
        marks = [r for r in rows if "spin_kernel" in r[3]]
        if len(marks) != hi - lo:
            return f"fail:markers_{mode}_{len(marks)}_of_{hi - lo}"
        n = sum(len(_tr_between(rows, a, b)) for a, b in _tr_split(marks, 0, hi - lo))
        seen[mode] = n
        if mode == "graph" and n > 0:
            return "graph"
    return "eager" if seen.get("eager", 0) > 0 else "fail:no_kernel_records"


def _tr_between(rows, m1_end, m2_start):
    return [r for r in rows if m1_end <= r[0] and r[1] <= m2_start + 1 and "spin_kernel" not in r[3]]


# ---------------------------------------------------------------------------------------------------- output
def _tr_us(ns):
    return f"{ns / 1000.0:.2f}"


def _tr_render(version, shapes, dropped, secs):
    """shapes: [(idx, case, res)] kept, in index order; dropped: [(idx, why)]. -> text."""
    names, order = {}, []
    lines = []
    for idx, case, res in shapes:
        agg = res.get("agg")
        w = case.get("weight", 1)
        if res.get("error"):
            lines.append(f"TE {idx} {res['error']}")
            continue
        n = len(agg["kernels"]) if agg else 0
        span = _tr_us(agg["span"]) if agg else "0.00"
        lines.append(f"TS {idx} w={w} wall={_tr_us(res['wall'] * 1000.0)} span={span} n={n} label={_tr_label(case)}"
                     + (f" rep={agg['rep_ok']}/{agg['rep_all']}" if agg else ""))
        if not agg:
            continue
        for p, k in enumerate(agg["kernels"]):
            nm = " ".join(k["name"].split())[:_TR_NAME_CHARS]
            if nm not in names:
                names[nm] = len(names)
                order.append(nm)
            line = f"TK {idx}.{p} {names[nm]} s={_tr_us(k['s'])} d={_tr_us(k['d'])} g={_tr_us(k['g'])}"
            if k["st"]:
                line += f" st={k['st']}"
            if k["o"] > 0:
                line += f" o={_tr_us(k['o'])}"
            lines.append(line)
        lines.append(f"TG {idx} gaps={_tr_us(agg['gaps'])} head={_tr_us(agg['head'])} tail={_tr_us(agg['tail'])} "
                     f"busy={_tr_us(agg['busy'])}")
    for idx, why in dropped:
        lines.append(f"TE {idx} {why}")
    head = [version] + [f"TN {i} {nm}" for i, nm in enumerate(order)]
    return "\n".join(head + lines + [f"TZ done {secs:.0f}"])


def _tr_emit(shapes, dropped, version, secs):
    """Fit the judge's 16 KB log tail: drop the lowest-weight shapes (the later index first on a tie) until it fits; a single
    shape that is still too long keeps its first kernels (TE <idx> truncated:size)."""
    kept = list(shapes)
    dropped = list(dropped)
    text = _tr_render(version, kept, dropped, secs)
    while len(text) > _TR_MAX_BYTES and len(kept) > 1:
        victim = min(kept, key=lambda t: (t[1].get("weight", 1), -t[0]))
        kept.remove(victim)
        dropped.append((victim[0], "dropped:size"))
        text = _tr_render(version, kept, dropped, secs)
    if len(text) > _TR_MAX_BYTES and kept and kept[0][2].get("agg"):
        idx, case, res = kept[0]
        agg = res["agg"]
        full = len(agg["kernels"])
        lo, hi = 0, full
        while lo < hi:
            mid = (lo + hi + 1) // 2
            trial = dict(res, agg=dict(agg, kernels=agg["kernels"][:mid]))
            if len(_tr_render(version, [(idx, case, trial)], dropped + [(idx, f"truncated:size n={full}")], secs)) <= _TR_MAX_BYTES:
                lo = mid
            else:
                hi = mid - 1
        kept[0] = (idx, case, dict(res, agg=dict(agg, kernels=agg["kernels"][:lo])))
        dropped.append((idx, f"truncated:size n={full}"))
        text = _tr_render(version, kept, dropped, secs)
    return text


def _tr_header(env, cupti, replays):
    return (f"TV 1 {_TR_PROBLEM} {_TR_FILE} torch={_tr_torch.__version__} cuda={_tr_torch.version.cuda} "
            f"gpu={'_'.join(_tr_torch.cuda.get_device_name().split())} cupti={cupti} replays={replays}")


def _tr_finish(rows, results, base):
    """After the session: turn each shape's marker window into the aggregated timeline."""
    from_rows = sorted(rows, key=lambda r: (r[0], r[1]))
    marks = [r for r in from_rows if "spin_kernel" in r[3]]
    total = _tr_marks[0] - base
    if len(marks) != total:
        raise RuntimeError(f"markers {len(marks)} of {total}")
    for res in results.values():
        if res.get("error") or "marks" not in res:
            continue
        lo, hi = res.pop("marks")
        reps = []
        for a, b in _tr_split(marks, lo - base, hi - base):
            kern = _tr_between(from_rows, a, b)
            if kern:
                reps.append(_tr_analyze(kern, a, b))
        if not reps:
            res["error"] = "no_kernel_records"
        else:
            res["agg"] = _tr_aggregate(reps)


def _tr_run(state):
    """Time every case; print the TRACE lines. Never raises."""
    t_end = _TR_T0 + _TR_BUDGET_S - 8.0          # the rest is for parsing the profile
    try:
        if not _tr_torch.cuda.is_available() or _tr_torch.cuda.is_current_stream_capturing():
            print(f"TV 1 {_TR_PROBLEM} {_TR_FILE} cupti=fail:no_cuda_or_capturing", flush=True)
            return
        from torch.profiler import ProfilerActivity, profile

        env = _TrEnv()
        try:
            cupti = "eager" if _TR_FORCE_EAGER else _tr_selftest(env)
        except Exception as ex:  # noqa: BLE001
            cupti = _tr_tok(f"fail:{type(ex).__name__}:{_tr_oneline(ex, 80)}")
        mode = cupti if cupti in ("graph", "eager") else "none"
        results, dropped = {}, []
        order = sorted(range(len(_TR_CASES)), key=lambda i: (-_TR_CASES[i].get("weight", 1), i))
        per_shape, base = None, _tr_marks[0]
        rows = []
        import contextlib as _contextlib
        if order:
            with (profile(activities=[ProfilerActivity.CUDA]) if mode != "none" else _contextlib.nullcontext()) as prof:
                for done, idx in enumerate(order):
                    case = _TR_CASES[idx]
                    left = t_end - _tr_time.perf_counter()
                    remaining = len(order) - done
                    if left < 3.0 or (per_shape is not None and left < 0.5 * per_shape):
                        dropped.append((idx, "dropped:budget"))
                        continue
                    replays = _TR_REPLAYS
                    if per_shape is not None and left < per_shape * remaining * 1.1:
                        replays = max(_TR_MIN_REPLAYS, replays // 2)
                    t1 = _tr_time.perf_counter()
                    try:
                        results[idx] = _tr_shape(env, case, state, replays, mode)
                    except Exception as ex:  # noqa: BLE001
                        results[idx] = dict(error=f"{type(ex).__name__}:{_tr_oneline(ex)}", wall=0.0)
                    _tr_torch.cuda.empty_cache()
                    dt = _tr_time.perf_counter() - t1
                    per_shape = dt if per_shape is None else 0.5 * (per_shape + dt)
                _tr_torch.cuda.synchronize()
            if mode != "none":
                try:
                    _tr_finish(_tr_gpu_rows(prof), results, base)
                except Exception as ex:  # noqa: BLE001  (the walls are still good)
                    cupti = _tr_tok(f"fail:{type(ex).__name__}:{_tr_oneline(ex, 80)}")
        text = _tr_emit([(i, _TR_CASES[i], results[i]) for i in sorted(results)], sorted(dropped),
                        _tr_header(env, cupti, _TR_REPLAYS), _tr_time.perf_counter() - _TR_T0)
        print(text, flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"TV 1 {_TR_PROBLEM} {_TR_FILE} cupti={_tr_tok('fail:' + type(ex).__name__ + ':' + _tr_oneline(ex, 120))}", flush=True)
        print(f"TZ done {_tr_time.perf_counter() - _TR_T0:.0f}", flush=True)


if _TR_HAS_PREPARE:
    _tr_orig_prepare = prepare

    def prepare(weights):  # noqa: F811
        global _TR_T0
        _TR_T0 = _tr_time.perf_counter()
        state = _tr_orig_prepare(weights)
        _tr_run(state)
        return state
elif not globals().get("_TR_NO_AUTORUN"):
    _tr_run(None)
