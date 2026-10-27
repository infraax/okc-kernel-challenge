# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY block (tools/gen_measure_verify_v1.py appends it after the unchanged sonnet_v3 code, the generated _mv_ twins of _full_kernel and the inlined generate_input).
# Nothing here feeds kernel_fn: at import it times the C26' variants on generate_input-recipe inputs (3 seeds per shape, read-only L2 flush, CUDA-graph replays, interleaved
# rounds with the board kernel as pivot) and prints the S / C / B lines.
# ---------------------------------------------------------------------------------------------------------------------
import statistics as _mv_stats
import time as _mv_time

_MV_T0 = _mv_time.perf_counter()
_MV_BUDGET_S = 240.0         # the study ends by here; variants that do not fit are dropped, not run late
_MV_SEEDS = (11, 12, 13)
_MV_REPS = 5                 # timed replays per (variant, seed); the median of those is that seed's value
_MV_FLUSH_MB = 128           # read-only L2 flush buffer (> the 50 MB L2)
_MV_MAX_BYTES = 13800
_MV_TOL = 0.03               # the smallest bound the attention check ever uses (checks.query_precision_bound floor); EXPSPLIT is held to it against base
_MV_ROW_FACTOR, _MV_ROW_FLOOR = 5.0, 0.1
_MV_NOTES = []

V_BASE, V_GATHER, V_MMAONLY, V_NOEXP, V_NOCVT, V_EXP25, V_EXP50 = (tl.constexpr(i) for i in range(7))
_MV_VAR_BASE = {"base": 0, "GATHER": 1, "MMAONLY": 2, "NOEXP": 3, "NOCVT": 4, "EXPSPLIT25": 5, "EXPSPLIT50": 6}
# name, overrides of FULL_CFG["verify"]; priority order (compiles happen on the first shape, in this order)
_MV_VARIANTS = [
    ("base", {}), ("GATHER", {"var": 1}), ("NOEXP", {"var": 3}), ("NOCVT", {"var": 4}), ("EXPSPLIT25", {"var": 5}), ("EXPSPLIT50", {"var": 6}),
    ("MMAONLY", {"var": 2}), ("RED", {"red": True}), ("PROGx2", {"programs": 512}), ("PROGd2", {"programs": 128}), ("ST3", {"stages": 3}),
    ("ST2", {"stages": 2}), ("BN128", {"bn": 128, "stages": 3}), ("W8", {"warps": 8}),
]
_MV_CHECKED = {"base", "EXPSPLIT25", "EXPSPLIT50", "PROGx2", "PROGd2", "ST3", "ST2", "BN128", "W8"}   # outputs that must equal base's (up to roundoff)


def _mv_elapsed():
    return _mv_time.perf_counter() - _MV_T0


def _mv_note(text):
    _MV_NOTES.append(" ".join(str(text).split())[:200])


def _mv_where(ex):
    return f"{type(ex).__name__}: {str(ex)[:80]}".replace("\n", " ")


def _mv_med(xs):
    xs = [x for x in xs if x is not None]
    return _mv_stats.median(xs) if xs else None


# ================================================================================================================ kernel helpers
@triton.jit
def _mv_split_exp(x, DIV: tl.constexpr):
    """exp2 of x [R, C] where 1 of every DIV columns (DIV 2: cols 0 mod 2; DIV 4: cols 0 mod 4) is evaluated by the polynomial, the rest by MUFU ex2.
    The pieces are separate tensors (reshape + split), so each element runs exactly one of the two paths."""
    R: tl.constexpr = x.shape[0]
    C: tl.constexpr = x.shape[1]
    if DIV == 2:
        a, b = tl.split(tl.reshape(x, [R, C // 2, 2]))
        p = tl.reshape(tl.join(_mv_exp2_poly(a), tl.exp2(b)), [R, C])
    else:
        e, o = tl.split(tl.reshape(x, [R, C // 2, 2]))                     # even / odd columns: thread-local in the MMA accumulator layout
        h = tl.permute(tl.reshape(e, [R, 2, C // 4]), (0, 2, 1))            # the even columns again by their high bit (also thread-local)
        e0, e1 = tl.split(h)
        ee = tl.reshape(tl.permute(tl.join(_mv_exp2_poly(e0), tl.exp2(e1)), (0, 2, 1)), [R, C // 2])
        p = tl.reshape(tl.join(ee, tl.exp2(o)), [R, C])
    return p


@triton.jit
def _mv_p(x, VAR: tl.constexpr):
    if VAR == V_NOEXP:
        p = x * 0.0078125 + 1.0                  # one FFMA instead of ex2 (still feeds P.V)
    elif VAR == V_EXP25:
        p = _mv_split_exp(x, 4)
    elif VAR == V_EXP50:
        p = _mv_split_exp(x, 2)
    else:
        p = tl.exp2(x)
    return p


@triton.jit
def _mv_gather(k_cache, v_cache, pg, t, end, hb, d, PAGE: tl.constexpr, HKV: tl.constexpr, VAR: tl.constexpr):
    """The board's _gather, or: GATHER (all four tiles as int32 words [BN, 16]), NOCVT / MMAONLY (K as the board, V as an FP16 view [BN, 32] of the same 64 bytes)."""
    if VAR == V_GATHER:
        if PAGE == 1:
            row = pg.to(tl.int64)
        else:
            row = pg.to(tl.int64) * PAGE + t % PAGE
        off = tl.multiple_of(row * (HKV * 16) + hb // 4, 16)
        w = off[:, None] + tl.arange(0, 16)[None, :]
        m = (t < end)[:, None]
        kw = k_cache.to(tl.pointer_type(tl.int32))
        vw = v_cache.to(tl.pointer_type(tl.int32))
        k0 = tl.load(kw + w, mask=m, other=0, eviction_policy="evict_first")
        k1 = tl.load(kw + (w + 16), mask=m, other=0, eviction_policy="evict_first")
        v0 = tl.load(vw + w, mask=m, other=0, eviction_policy="evict_first")
        v1 = tl.load(vw + (w + 16), mask=m, other=0, eviction_policy="evict_first")
    elif VAR == V_NOCVT or VAR == V_MMAONLY:
        if PAGE == 1:
            row = pg.to(tl.int64)
        else:
            row = pg.to(tl.int64) * PAGE + t % PAGE
        off = tl.multiple_of(row * (HKV * 64) + hb, 64)
        p = off[:, None] + d
        m = (t < end)[:, None]
        k0 = tl.load(k_cache + p, mask=m, other=0, eviction_policy="evict_first")
        k1 = tl.load(k_cache + (p + 64), mask=m, other=0, eviction_policy="evict_first")
        vh = v_cache.to(tl.pointer_type(tl.float16))
        ph = tl.multiple_of(off // 2, 32)[:, None] + tl.arange(0, 32)[None, :]
        v0 = tl.load(vh + ph, mask=m, other=0.0, eviction_policy="evict_first")
        v1 = tl.load(vh + (ph + 32), mask=m, other=0.0, eviction_policy="evict_first")
    else:
        k0, k1, v0, v1 = _gather(k_cache, v_cache, pg, t, end, hb, d, PAGE, HKV)
    return k0, k1, v0, v1


@triton.jit
def _mv_nop(p):
    tl.store(p, 0)


# ================================================================================================================ launch
def _mv_cfg(v):
    c = dict(FULL_CFG["verify"])
    c.update({k: x for k, x in v.items() if k in c})
    return c


def _mv_plan(data, cfg):
    c = data["config"]
    batch, length, hq, hkv = c["batch"], c["q_len"], c["num_q_heads"], c["num_kv_heads"]
    group = hq // hkv
    gp = triton.next_power_of_2(group)
    bl = max(triton.next_power_of_2(length), 16 // gp)
    span = max(1, data["page_table"].shape[1] * c["page_size"])
    split_len, nsplit = _plan_full(batch, hkv // 2, span, cfg["bn"], cfg["programs"])
    return dict(batch=batch, length=length, hq=hq, hkv=hkv, group=group, gp=gp, bl=bl, rows=bl * gp, split_len=split_len, nsplit=nsplit)


def _mv_bufs(data, pl):
    dev = data["qkv"].device
    part_o = torch.zeros((pl["batch"], pl["hkv"], pl["nsplit"] + 1, pl["rows"], 64), dtype=torch.float32, device=dev)
    part_m = torch.zeros((pl["batch"], pl["hkv"], pl["nsplit"] + 1, pl["rows"]), dtype=torch.float32, device=dev)
    return dict(part_o=part_o, part_m=part_m, part_l=torch.zeros_like(part_m), sink=torch.zeros(pl["batch"] * pl["hkv"] * (pl["nsplit"] + 1) + 8, dtype=torch.int32, device=dev),
                out=torch.empty((pl["batch"], pl["length"], pl["hq"] * 64), dtype=torch.bfloat16, device=dev))


def _mv_launch(data, cfg, var, pl, bufs, red_only=False):
    """kernel_fn's full-layer 'sep' branch with the twin kernel (and the board's reducer). -> the main kernel's compiled handle (None when only the reducer ran)."""
    c = data["config"]
    qkv = data["qkv"]
    dev = qkv.device
    cdtype, native_load, native_store, qk8 = _device_mode(dev.type, dev.index or 0)
    page_table = data["page_table"]
    h = None
    if not red_only:
        h = _mv_full_kernel[(pl["nsplit"] + 1, pl["hkv"] // 2, pl["batch"])](
            qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8),
            data["k_scale"], data["v_scale"], bufs["part_o"], bufs["part_m"], bufs["part_l"], bufs["sink"],
            c["sm_scale"], c["qk_scale_factor"], 0.0, page_table.stride(0), pl["split_len"], pl["nsplit"],
            L=pl["length"], G=pl["group"], GP=pl["gp"], HQ=pl["hq"], HKV=pl["hkv"], PAGE=c["page_size"], ROPE=bool(c["rope_theta"]),
            NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype, BL=pl["bl"], M=pl["rows"],
            BN=cfg["bn"], BT=max(16, triton.next_power_of_2(pl["length"])), VAR=var, ACOLS=32 if var in (2, 4) else 64,
            num_warps=cfg["warps"], num_stages=cfg["stages"],
        )
    _reduce_kernel[(pl["batch"] * pl["length"], pl["hq"])](
        bufs["part_o"], bufs["part_m"], bufs["part_l"], qkv, data["seq_lens"], data["v_scale"], bufs["out"], pl["split_len"], pl["nsplit"],
        L=pl["length"], G=pl["group"], GP=pl["gp"], HQ=pl["hq"], HKV=pl["hkv"], WINDOW=0, M=pl["rows"], ROWMAJOR=True,
        BS=min(256, triton.next_power_of_2(pl["nsplit"] + 1)), num_warps=8 if pl["nsplit"] + 1 > 128 else 4, num_stages=1,
    )
    return h


def _mv_counts(h):
    """ptx_audit-style counts of a compiled handle: regs / spills / smem, ex2, cp.async, wgmma, cvt, ld.global, bar.sync."""
    import re as _re
    if h is None:
        return None
    ptx = _re.sub(r"//[^\n]*", "", h.asm["ptx"])
    n = lambda pat: len(_re.findall(pat, ptx))
    return dict(regs=getattr(h, "n_regs", None), spills=getattr(h, "n_spills", None), smem=int(h.metadata.shared), ex2=n(r"\bex2\."), cpasync=n(r"\bcp\.async\.(?:ca|cg)\b"),
                wgmma=n(r"\bwgmma\.mma_async\b"), mma=n(r"\bmma\.sync\b"), cvt=n(r"\bcvt\."), ldg=n(r"\bld\.global\b"), bar=n(r"\bbar(?:rier)?\.(?:cta\.)?sync\b"))


# ================================================================================================================ timing and checks
class _MvEnv:
    def __init__(self):
        self.flush = torch.ones(_MV_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        self.sink = torch.zeros((), device="cuda", dtype=torch.float32)

    def replay_us(self, g):
        torch.sum(self.flush, dim=0, out=self.sink)   # read-only: 128 MB through L2, like the judge's flush
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) * 1000.0

    def capture(self, fn):
        """Two eager runs on a side stream (compiles), then the graph. -> (graph, what fn returned in the first eager run)."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            first = fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return g, first

    def overhead(self, reps=21):
        """A graph replay's fixed cost (one empty kernel), which the judge subtracts (about 4 us). -> median us."""
        buf = torch.zeros(4, device="cuda", dtype=torch.int32)
        g, _ = self.capture(lambda: _mv_nop[(1,)](buf))
        return _mv_med([self.replay_us(g) for _ in range(reps)])


def _mv_cmp(out, ref, tol=_MV_TOL):
    """checks.compare(out, ref, tol), re-implemented. -> (ok, relative L2 error, worst row error / (ROW_FACTOR * tol))."""
    o, r = out.float(), ref.float()
    if not bool(torch.isfinite(o).all()):
        return False, float("inf"), float("inf")
    diff = o - r
    err = float(diff.norm() / r.norm().clamp(min=1e-12))
    rows_d, rows_r = diff.reshape(-1, r.shape[-1]).norm(dim=-1), r.reshape(-1, r.shape[-1]).norm(dim=-1)
    floor = (_MV_ROW_FLOOR * rows_r.mean()).clamp(min=1e-12)
    worst = float((rows_d / torch.maximum(rows_r, floor)).max())
    return err <= tol and worst <= _MV_ROW_FACTOR * tol, err, worst / (_MV_ROW_FACTOR * tol)


# ================================================================================================================ one (shape, seed)
def _mv_seed(env, case, seed, shape_res, st, ovh, deadline):
    data = _tr_generate_input(**{**case, "seed": seed})
    verify = case.get("q_len", 1) > 1
    runners = []        # (name, graph, bufs)
    base_out = None
    base_pl = base_bufs = None
    # the board kernel, always
    g, _ = env.capture(lambda: kernel_fn(data))
    runners.append(("board", g, None))
    dead = st["dead"]
    todo = _MV_VARIANTS if verify else []      # decode-full shapes: the board kernel only (reference)
    for name, v in todo:
        res = shape_res.setdefault(name, dict(us=[], err=[], row=[], perr=[], ok=[], counts=None, status="ok"))
        if name in dead:
            res["status"] = dead[name]
            continue
        fresh = name not in st["compiled"] and name != "RED"
        if fresh and _mv_elapsed() + 1.3 * st["est"] > deadline:
            dead[name] = res["status"] = "skip budget"
            continue
        t1 = _mv_time.perf_counter()
        try:
            if v.get("red"):
                if base_pl is None:
                    base_pl = _mv_plan(data, _mv_cfg({}))
                    base_bufs = _mv_bufs(data, base_pl)
                    _mv_launch(data, _mv_cfg({}), 0, base_pl, base_bufs)
                    torch.cuda.synchronize()
                g, _ = env.capture(lambda: _mv_launch(data, _mv_cfg({}), 0, base_pl, base_bufs, True))
                runners.append((name, g, None))
                continue
            cfg = _mv_cfg(v)
            pl = _mv_plan(data, cfg)
            bufs = _mv_bufs(data, pl)
            var = v.get("var", 0)
            g, h = env.capture(lambda: _mv_launch(data, cfg, var, pl, bufs))
            if res["counts"] is None:
                res["counts"] = _mv_counts(h)
            runners.append((name, g, bufs))
            st["compiled"].add(name)
            if name == "base":
                base_pl, base_bufs = pl, bufs
                g.replay()
                torch.cuda.synchronize()
                base_out = bufs["out"].clone()
        except Exception as ex:  # noqa: BLE001
            dead[name] = res["status"] = "fail " + _mv_where(ex)
            continue
        finally:
            if fresh:
                st["est"] = max(st["est"], _mv_time.perf_counter() - t1)
    # outputs vs base
    errs = {}
    if base_out is not None:
        for name, g, bufs in runners:
            if name in _MV_CHECKED and bufs is not None:
                g.replay()
                torch.cuda.synchronize()
                ok, e, row = _mv_cmp(bufs["out"], base_out)
                pe = None
                if bufs["part_o"].shape == base_bufs["part_o"].shape:      # same split plan: the FP32 partials show what the bf16 output hides
                    pe = max(float((bufs[k] - base_bufs[k]).norm() / base_bufs[k].norm().clamp(min=1e-30)) for k in ("part_o", "part_l"))
                errs[name] = (ok, e, row, pe)
        bo = runners[0][1]
        bo.replay()
    # timing: interleaved rounds, the order reversed every other round
    samples = {n: [] for n, _, _ in runners}
    for rep in range(_MV_REPS + 1):
        order = runners if rep % 2 == 0 else runners[::-1]
        for name, g, _ in order:
            us = env.replay_us(g)
            if rep > 0:                     # the first round is a warm-up
                samples[name].append(us)
    for name, _, _ in runners:
        res = shape_res.setdefault(name, dict(us=[], err=[], row=[], perr=[], ok=[], counts=None, status="ok"))
        res["us"].append(max(0.01, _mv_med(samples[name]) - ovh))
        if name in errs:
            res["ok"].append(errs[name][0])
            res["err"].append(errs[name][1])
            res["row"].append(errs[name][2])
            res["perr"].append(errs[name][3])
    g = bo = None
    del runners, data, base_out, base_bufs
    torch.cuda.empty_cache()


# ================================================================================================================ output
def _mv_fmt_err(e):
    return "inf" if e == float("inf") else f"{e:.1e}"


def _mv_xs(shape_res, name, k):
    b, r = shape_res.get("base"), shape_res.get(name)
    if not b or not r or k >= len(b["us"]) or k >= len(r["us"]):
        return None
    return b["us"][k] / r["us"][k]


def _mv_render(results, meta):
    """results: {idx: dict(label, case, seeds, v: {name: res})}. -> text <= _MV_MAX_BYTES, S lines first."""
    S, C, B = [], [], []
    S.append(f"S attn verify v1: gpu={meta['gpu']} used={meta['secs']:.0f}s of {_MV_BUDGET_S:.0f}s ovh={meta['ovh']:.2f}us(subtracted) seeds={len(_MV_SEEDS)} reps={_MV_REPS}"
             f" shapes={len(results)}/{len(_MV_CASES)} variants={len(_MV_VARIANTS)}")
    worst = None
    for idx, r in results.items():
        v = r["v"]
        med = {n: _mv_med(x["us"]) for n, x in v.items()}
        if "board" not in med:
            continue
        parts = [f"board={med['board']:.1f}"]
        for n, _ in _MV_VARIANTS:
            if n in med and med[n] is not None and n != "base" and "base" in med:
                parts.append(f"{n}={med[n]:.1f}(x{med['base'] / med[n]:.2f})")
            elif n == "base" and med.get(n) is not None:
                parts.append(f"base={med[n]:.1f}(x{med['board'] / med[n]:.2f} board/base)")
            elif n in v and v[n]["status"] != "ok":
                parts.append(f"{n}={v[n]['status'][:28]}")
        S.append(f"S #{idx} {r['label']} us: " + " ".join(parts))
        b = med.get("base")
        if b and r["label"].startswith("verify"):
            def share(n):
                return f"{100 * (b - med[n]) / b:.0f}%" if med.get(n) else "n/a"
            red = med.get("RED")
            S.append(f"S #{idx} shares of base (save if removed): exp={share('NOEXP')} cvt={share('NOCVT')} compute(all but loads)={share('GATHER')} mma-only={share('MMAONLY')}"
                     f" split25={share('EXPSPLIT25')} split50={share('EXPSPLIT50')}" + (f" reducer={100 * red / b:.0f}%" if red else ""))
            floor = min([med[n] for n in ("GATHER", "MMAONLY") if med.get(n)] or [0.0])
            bad = [n for n in ("NOEXP", "NOCVT", "EXPSPLIT25", "EXPSPLIT50") if med.get(n) and floor and med[n] < floor * 0.98]
            if bad:
                S.append(f"S #{idx} INVALID twins faster than the GATHER / MMAONLY floor ({floor:.1f}): {','.join(bad)}")
        for n in ("EXPSPLIT25", "EXPSPLIT50"):
            x = v.get(n)
            if x and x["err"]:
                cand = (max(x["err"]), max(x["row"]), max([p for p in x["perr"] if p is not None] or [0.0]))
                worst = cand if worst is None else tuple(max(a, b) for a, b in zip(worst, cand))
    S.append(f"S EXPSPLIT vs base, worst over shapes/seeds/25-50: output rel L2 {_mv_fmt_err(worst[0])}, worst row {worst[1]:.2f} of its {_MV_ROW_FACTOR:g}x{_MV_TOL:g} limit, FP32 partials {_mv_fmt_err(worst[2])} (bound >= {_MV_TOL:g})"
             if worst else "S EXPSPLIT: no error measured")
    first = next(iter(results.values()), None)
    if first:
        for n, _ in _MV_VARIANTS:
            c = first["v"].get(n, {}).get("counts")
            if c:
                C.append(f"C {n} regs={c['regs']} spills={c['spills']} smem={c['smem']} ex2={c['ex2']} cp.async={c['cpasync']} wgmma={c['wgmma']} mma={c['mma']} cvt={c['cvt']} ldg={c['ldg']} bar={c['bar']}")
    for note in _MV_NOTES[:6]:
        S.append("S note: " + note)
    for idx, r in results.items():
        for k, seed in enumerate(r["seeds"]):
            for n in ["board"] + [n for n, _ in _MV_VARIANTS]:
                x = r["v"].get(n)
                if not x:
                    continue
                if k >= len(x["us"]):
                    if x["status"] != "ok" and k == 0:
                        B.append(f"B attn #{idx} seed={seed} | {n} {x['status'][:60]}")
                    continue
                sx = _mv_xs(r["v"], n, k)
                tail = "" if sx is None else f" x={sx:.2f}"
                if k < len(x["err"]):
                    tail += f" err={_mv_fmt_err(x['err'][k])}" + ("" if x["perr"][k] is None else f"/{_mv_fmt_err(x['perr'][k])}") + ("" if x["ok"][k] else " BAD")
                B.append(f"B attn #{idx} seed={seed} | {n} us={x['us'][k]:.2f}{tail}")
    head = "\n".join(S + C)
    out, dropped = [head], 0
    size = len(head) + 1
    for line in B:
        if size + len(line) + 1 > _MV_MAX_BYTES - 80:
            dropped += 1
            continue
        out.append(line)
        size += len(line) + 1
    if dropped:
        out.insert(1, f"S {dropped} B lines dropped for the 14 KB cap (the S lines hold the seed medians)")
    return "\n".join(out)


# ================================================================================================================ the study
def _mv_study():
    """Never raises. Prints the report."""
    results, meta = {}, dict(gpu="n/a", ovh=0.0, secs=0.0)
    try:
        if not torch.cuda.is_available() or torch.cuda.is_current_stream_capturing():
            print("S attn verify v1: no CUDA device (or capturing): nothing measured", flush=True)
            return
        meta["gpu"] = "_".join(torch.cuda.get_device_name().split())
        env = _MvEnv()
        meta["ovh"] = env.overhead()
        st = dict(dead={}, compiled=set(), est=8.0)
        deadline = _MV_BUDGET_S - 12.0
        for idx, label, case in _MV_CASES:
            if _mv_elapsed() > deadline - 10.0:
                _mv_note(f"#{idx} {label} skipped (budget)")
                continue
            r = results.setdefault(idx, dict(label=label, case=case, seeds=[], v={}))
            for seed in _MV_SEEDS:
                if _mv_elapsed() > deadline - 6.0 and r["seeds"]:
                    _mv_note(f"#{idx} {label}: seeds after {len(r['seeds'])} skipped (budget)")
                    break
                try:
                    _mv_seed(env, case, seed, r["v"], st, meta["ovh"], deadline)
                    r["seeds"].append(seed)
                except Exception as ex:  # noqa: BLE001
                    _mv_note(f"#{idx} {label} seed {seed}: {_mv_where(ex)}")
                    torch.cuda.empty_cache()
        meta["secs"] = _mv_elapsed()
        print(_mv_render(results, meta), flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"S attn verify v1 FAILED: {_mv_where(ex)}", flush=True)
        try:
            meta["secs"] = _mv_elapsed()
            print(_mv_render(results, meta), flush=True)
        except Exception:  # noqa: BLE001
            pass


if not globals().get("_MV_NO_AUTORUN"):
    _mv_study()
