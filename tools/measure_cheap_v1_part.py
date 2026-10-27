# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY block (tools/gen_measure_cheap_v1.py appends it after the unchanged director_v1b code, the fixup kernel / candidate table of director_v2 and the
# inlined generate_input). Nothing here feeds kernel_fn: prepare() runs director_v1b's prepare(), then times the C23 variants of fable-09 D4 section 4 on
# generate_input-recipe inputs (3 seeds per shape, read-only L2 flush, CUDA-graph replays, interleaved rounds with the board kernel as pivot) and prints the S / B / K lines.
# ---------------------------------------------------------------------------------------------------------------------
import time as _mc_time

_MC_T0 = _mc_time.perf_counter()
_MC_BUDGET_S = 262.0        # prepare() ends by here (the judge's limit is 300 s); variants that do not fit are dropped, not run late
_MC_PRIO_S = 214.0          # no new-variant shape starts after this (the cheap reference shapes follow)
_MC_SEEDS = (11, 12, 13)
_MC_REPS = 5                # timed replays per (variant, seed); the median of those is that seed's value
_MC_FLUSH_MB = 128          # read-only L2 flush buffer (> the 50 MB L2)
_MC_MAX_BYTES = 13800
_MC_TOL = {"router": 1e-4}  # relative L2 bound; every other layer 1e-2 (problems/muse_bf16_proj/reference.py)
_MC_ROW_FACTOR, _MC_ROW_FLOOR, _MC_PROBE_TOL = 5.0, 0.1, 1e-3
_MC_MS = (1, 8, 16, 32, 128, 256)
_MC_LAYERS = ("qkv_gate", "o", "pooler", "unpooler", "router")
_MC_LOADS = ((1, "decode1"), (16, "decode16"), (32, "decode32"), (8, "verify1"), (128, "verify16"), (256, "verify32"))  # m of each judged load
_MC_PRIO = (("unpooler", 1), ("unpooler", 8), ("router", 1), ("router", 8), ("router", 16), ("router", 32), ("router", 128), ("router", 256))
_MC_NOTES = []


def _mc_elapsed():
    return _mc_time.perf_counter() - _MC_T0


def _mc_note(text):
    _MC_NOTES.append(" ".join(str(text).split())[:200])


def _mc_where(ex):
    return f"{type(ex).__name__}: {str(ex)[:90]}".replace("\n", " ")


def _mc_med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _mc_nk(layer, m):
    n, k = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096), "router": (256, 8192)}[layer]
    if layer == "unpooler" and m <= 8:
        n = 1024   # this GPU's slice (problems/muse_bf16_proj/reference.py UNPOOL_SLICE)
    return n, k


@triton.jit
def _mc_nop(p):
    tl.store(p, 0)


# ================================================================================================================ timing and checks
class _McEnv:
    def __init__(self, dev):
        self.dev = dev
        self.flush = torch.ones(_MC_FLUSH_MB * 1024 * 1024 // 4, device=dev, dtype=torch.float32)
        self.sink = torch.zeros((), device=dev, dtype=torch.float32)

    def replay_us(self, g):
        torch.sum(self.flush, dim=0, out=self.sink)   # read-only: 128 MB through L2, like the judge's flush
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1000.0

    def capture(self, fn, x):
        """Two eager runs on a side stream (compiles), then the graph. -> (graph, static output)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(x)
            fn(x)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = fn(x)
        return g, y

    def overhead(self, reps=21):
        """The fixed cost of a graph replay (one empty kernel), which the judge subtracts (about 4 us). -> raw median us."""
        buf = torch.zeros(4, device=self.dev, dtype=torch.int32)
        g, _ = self.capture(lambda x: (_mc_nop[(1,)](buf), buf)[1], buf)
        ts = [self.replay_us(g) for _ in range(reps)]
        return _mc_med(ts)


def _mc_check(y, ref, x, layer):
    """The organisers' bound (checks.compare + probe rows), re-implemented. -> (ok, share of the tightest bound used, why)."""
    tol = _MC_TOL.get(layer, 1e-2)
    if y.shape != ref.shape or y.dtype != ref.dtype:
        return False, 99.0, f"shape/dtype {tuple(y.shape)} {y.dtype}"
    out, r = y.float(), ref.float()
    if not bool(torch.isfinite(out).all()):
        return False, 99.0, "nan/inf"
    diff = out - r
    err = float(diff.norm() / r.norm().clamp(min=1e-12))
    rows_d, rows_r = diff.norm(dim=-1), r.norm(dim=-1)
    floor = (_MC_ROW_FLOOR * rows_r.mean()).clamp(min=1e-12)
    worst = float((rows_d / torch.maximum(rows_r, floor)).max())
    probes = ((x != 0).sum(-1) == 1).nonzero().flatten()
    perr = 0.0
    if len(probes):
        pd = (y[probes].double() - ref[probes].double()).norm(dim=-1) / ref[probes].double().norm(dim=-1).clamp(min=1e-30)
        perr = float(pd.max())
    use = max(err / tol, worst / (_MC_ROW_FACTOR * tol), perr / _MC_PROBE_TOL)
    return use <= 1.0, use, f"err={err:.2e} row={worst:.2e} probe={perr:.2e}"


# ================================================================================================================ variants
def _mc_smem_ok(bm, bn, bk, lo, stages):
    return stages * 2 * bk * (bm + bn * (2 if lo else 1)) <= 213 * 1024    # estimate of the pipeline's smem; Triton's real number is checked offline


def _mc_variants(layer, m, n, state, fstate):
    """-> [(name, kind, fn(x) -> out)], kind board | ref | new. The board kernel first (the pivot)."""
    cfg = {"m": m, "n": n, "k": _mc_nk(layer, m)[1], "layer": layer}
    out = [("board", "board", lambda x: kernel_fn({"x": x, "config": cfg}, state))]
    w = state[layer]
    if layer == "router":
        def sgl(x):
            if m <= 8:
                return F.linear(x.float(), state["router"])
            acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
            return acc

        def mm2(x):
            acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
            return acc

        out.append(("sgl", "ref", sgl))
        out.append(("mm2", "ref", mm2) if m <= 8 else ("lin", "ref", lambda x: F.linear(x.float(), state["router"])))
        cands = _MC_CANDIDATES["router_small"] if m <= 32 else _MC_CANDIDATES["router_big"]
    else:
        out.append(("sgl", "ref", lambda x: _bf16_gemm_dispatch_impl(x, w[:n], None)))
        out.append(("mm", "ref", lambda x: torch.mm(x, w[:n].t())))
        cands = _MC_CANDIDATES["unpooler"] if (layer == "unpooler" and m <= 8) else []
    for var in cands:
        if var[0] == "tmm":
            continue   # = the "mm" reference
        if var[0] == "direct":
            _, bn, bk, warps, stages = var
            if not _mc_smem_ok(16, bn, bk, False, stages):
                continue
            name = f"dir.{bn}.{bk}.{warps}.{stages}"
            out.append((name, "new", (lambda x, c=(bn, bk, 1, warps, stages): _run(x, w, None, n, torch.bfloat16, c))))
        else:
            _, bm, bn, bk, split, warps, stages, rg = var
            bmv = bm or max(16, triton.next_power_of_2(m))
            if not _mc_smem_ok(bmv, bn, bk, layer == "router", stages) or not _mc_fix_fits(fstate, m, n, _mc_nk(layer, m)[1], bmv, bn, bk, split, rg):
                continue
            name = f"fix.{'m' if bm == 0 else bm}.{bn}.{bk}.{split}.{warps}.{stages}" + (f".g{rg}" if rg != 8 else "")
            odt = torch.float32 if layer == "router" else torch.bfloat16
            out.append((name, "new", (lambda x, v=var, odt=odt: _mc_run_fix(x, fstate, layer, n, odt, v))))
    return out


# ================================================================================================================ one shape
def _mc_ref(w, layer, n, x):
    ref = x.double() @ w[:n].double().T
    return ref.to(torch.float32 if layer == "router" else torch.bfloat16)


def _mc_shape(env, layer, m, state, fstate, ovh, deadline):
    """-> dict name -> {kind, us (net, median over seeds), xs (per-seed board/variant), ok, use, why}; 'status' for non-timed ones."""
    n, k = _mc_nk(layer, m)
    seeds = [_tr_generate_input(m, layer, s, "normal", "cuda") for s in _MC_SEEDS]
    xs = [d["x"].contiguous() for d in seeds]
    refs = [_mc_ref(state[layer], layer, n, x) for x in xs]
    x = xs[0].clone()
    variants = _mc_variants(layer, m, n, state, fstate)
    res = {}
    graphs = []
    for name, kind, fn in variants:
        if name != "board" and _mc_elapsed() > deadline:
            res[name] = dict(kind=kind, status="skip")
            continue
        try:
            g, y = env.capture(fn, x)
            graphs.append((name, kind, g, y))
        except Exception as ex:  # noqa: BLE001
            res[name] = dict(kind=kind, status="fail " + _mc_where(ex))
            if name == "board":
                raise
    raw = {name: [] for name, _, _, _ in graphs}
    bad = {name: (0.0, "") for name, _, _, _ in graphs}
    okv = {name: True for name, _, _, _ in graphs}
    per_seed = {name: [] for name, _, _, _ in graphs}
    for si in range(len(_MC_SEEDS)):
        x.copy_(xs[si])
        samples = {name: [] for name, _, _, _ in graphs}
        for _ in range(_MC_REPS):
            for name, _, g, _ in graphs:             # board first in every round: the pivot
                samples[name].append(env.replay_us(g))
        for name, kind, g, y in graphs:
            per_seed[name].append(_mc_med(samples[name]))
            ok, use, why = _mc_check(y, refs[si], x, layer)
            okv[name] = okv[name] and ok
            if use >= bad[name][0]:
                bad[name] = (use, why)
    board = per_seed.get("board")
    for name, kind, g, y in graphs:
        net = [max(0.5, v - ovh) for v in per_seed[name]]
        bnet = [max(0.5, v - ovh) for v in board]
        res[name] = dict(kind=kind, us=_mc_med(net), xs=[b / v for b, v in zip(bnet, net)], ok=okv[name], use=bad[name][0], why=bad[name][1], status="ok")
    del graphs
    return res


# ================================================================================================================ report
def _mc_xs(r):
    return _mc_med(r["xs"])


def _mc_line(layer, m, name, r):
    if r.get("status") != "ok":
        return f"B bf16 {layer} m={m} | {name} {r['status'][:60]}"
    s = f"B bf16 {layer} m={m} | {name} us={r['us']:.2f} x={_mc_xs(r):.2f}"
    if not r["ok"]:
        s += f" BAD({r['use']:.1f})"
    elif r["use"] > 0.5 and r["kind"] == "new":
        s += f" b={r['use']:.2f}"
    return s


def _mc_best_new(shape_res):
    cands = [(r["us"], name) for name, r in shape_res.items() if r.get("kind") == "new" and r.get("status") == "ok" and r["ok"]]
    return min(cands) if cands else None


def _mc_score(results, pick):
    """Geomean over the six judged loads of (sum of the in-run baseline) / (sum of ours). pick(shape_res) -> our us."""
    import math
    vals, parts = [], []
    for m, lname in _MC_LOADS:
        tb = to = 0.0
        for layer in _MC_LAYERS:
            r = results.get((layer, m))
            if not r or "board" not in r or r["board"].get("status") != "ok":
                return None, []
            base = r["sgl"] if r.get("sgl", {}).get("status") == "ok" else r.get("mm") or r.get("lin") or r["board"]
            tb += base["us"]
            to += pick(r)
        vals.append(tb / to)
        parts.append(f"{lname}={tb / to:.3f}")
    return math.exp(sum(math.log(v) for v in vals) / len(vals)), parts


def _mc_render(results, ovh, dev_name, t_total):
    lines = []
    sgl_real = getattr(_bf16_gemm_dispatch_impl, "__module__", "").startswith("sglang")
    lines.append(f"S bf16 cheap v1: {dev_name} triton={triton.__version__} torch={torch.__version__} sgl={'real' if sgl_real else 'STUB'} replay_ovh={ovh:.2f}us "
                 f"seeds={len(_MC_SEEDS)}x{_MC_REPS} flush={_MC_FLUSH_MB}MB t={t_total:.0f}s")
    lines.append("S us = graph replay minus replay_ovh (median of seeds of medians of 5); x = board/variant (>1: faster than the board kernel); b = share of the correctness bound used; "
                 "BAD = outside the bound")
    sc_board, p_board = _mc_score(results, lambda r: r["board"]["us"])
    sc_best, p_best = _mc_score(results, lambda r: min([r["board"]["us"]] + [v["us"] for v in r.values() if v.get("kind") == "new" and v.get("status") == "ok" and v["ok"]]))
    if sc_board:
        lines.append(f"S est problem speedup vs the in-run baseline: board {sc_board:.3f} ({' '.join(p_board)}); board + best new variant per shape {sc_best:.3f} ({' '.join(p_best)})")
    else:
        lines.append("S est problem speedup: not available (some shape missing)")
    lines.append("S best new variant per shape (board us -> variant us, x):")
    nbad = 0
    for layer, m in _MC_PRIO:
        r = results.get((layer, m))
        if not r:
            lines.append(f"S  {layer} m={m}: not measured")
            continue
        bn = _mc_best_new(r)
        nbad += sum(1 for v in r.values() if v.get("status") == "ok" and not v["ok"])
        if bn is None:
            lines.append(f"S  {layer} m={m}: board {r['board'].get('us', 0):.2f}, no correct new variant")
        else:
            lines.append(f"S  {layer} m={m}: board {r['board']['us']:.2f} -> {bn[1]} {bn[0]:.2f} (x{_mc_xs(r[bn[1]]):.2f})")
    for layer in _MC_LAYERS:                      # reference-only shapes: board vs the baseline path vs torch.mm
        row = []
        for m in _MC_MS:
            r = results.get((layer, m))
            if r and (layer, m) not in _MC_PRIO and r["board"].get("status") == "ok":
                ref = r.get("sgl") if r.get("sgl", {}).get("status") == "ok" else None
                row.append(f"m{m}:{r['board']['us']:.1f}/{ref['us']:.1f}" if ref else f"m{m}:{r['board']['us']:.1f}")
        if row:
            lines.append(f"S  {layer} board/baseline us  " + " ".join(row))
    lines.append(f"S K: {nbad} new variants outside the bound (BAD in the B lines); every row below is a median over seeds {','.join(map(str, _MC_SEEDS))}")
    for note in _MC_NOTES:
        lines.append("S note: " + note)
    body = []
    for layer in _MC_LAYERS:
        for m in _MC_MS:
            r = results.get((layer, m))
            if not r:
                continue
            order = ["board"] + sorted((n for n, v in r.items() if v["kind"] == "new" and v.get("status") == "ok"), key=lambda n: r[n]["us"]) \
                + sorted(n for n, v in r.items() if v["kind"] == "new" and v.get("status") != "ok") + sorted(n for n, v in r.items() if v["kind"] == "ref")
            for name in order:
                body.append(_mc_line(layer, m, name, r[name]))
    room = _MC_MAX_BYTES - len("\n".join(lines)) - 40
    out = list(body)
    # over the cap: drop the reference-only lines of shapes without new variants first, then the slowest half of the new variants
    while len("\n".join(out)) > room:
        drop = [i for i, s in enumerate(out) if (" | mm " in s or " | mm2 " in s or " | lin " in s)]
        if not drop:
            drop = [i for i, s in enumerate(out) if " | board " not in s and " | sgl " not in s]
        if not drop:
            break
        out.pop(drop[-1])
    if len(out) < len(body):
        lines.append(f"S note: {len(body) - len(out)} lines dropped for the 14 KB cap")
    return "\n".join(lines + out)


# ================================================================================================================ main
def _mc_main(state):
    if not torch.cuda.is_available():
        print("MC skip: no CUDA")
        return
    dev = state["router"].device
    env = _McEnv(dev)
    fstate = dict(state)
    fstate["fix_ws"] = torch.empty(FIX_WS_FLOATS, device=dev, dtype=torch.float32)
    fstate["fix_sem"] = torch.zeros(FIX_TILES, device=dev, dtype=torch.int32)
    ovh = env.overhead()
    if not (1.0 < ovh < 8.0):
        _mc_note(f"replay overhead {ovh:.2f} us outside 1..8: x values unreliable")
    print(f"MC start t={_mc_elapsed():.0f}s replay_ovh={ovh:.2f}us", flush=True)
    results = {}
    order = list(_MC_PRIO) + [(l, m) for m in _MC_MS for l in _MC_LAYERS if (l, m) not in _MC_PRIO]
    try:
        for layer, m in order:
            if _mc_elapsed() > _MC_BUDGET_S - 6:
                _mc_note(f"{layer} m={m} dropped:budget")
                continue
            prio = (layer, m) in _MC_PRIO
            deadline = _MC_PRIO_S if prio else _MC_BUDGET_S - 8
            if prio and _mc_elapsed() > _MC_PRIO_S:
                _mc_note(f"{layer} m={m} new variants dropped:budget")
            try:
                results[(layer, m)] = _mc_shape(env, layer, m, state, fstate, ovh, deadline)
            except Exception as ex:  # noqa: BLE001
                _mc_note(f"{layer} m={m} failed: {_mc_where(ex)}")
            print(f"MC p {layer}.{m} {_mc_elapsed():.0f}s", flush=True)
    finally:
        print(_mc_render(results, ovh, "_".join(torch.cuda.get_device_name().split()), _mc_elapsed()), flush=True)


_mc_orig_prepare = prepare


def prepare(weights):  # noqa: F811
    global _MC_T0
    _MC_T0 = _mc_time.perf_counter()
    state = _mc_orig_prepare(weights)
    try:
        _mc_main(state)
    except Exception as ex:  # noqa: BLE001
        print(f"MC failed: {_mc_where(ex)}", flush=True)
    return state
