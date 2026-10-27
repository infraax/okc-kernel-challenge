# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY block (tools/gen_measure_verify_v2.py appends it after the unchanged sonnet_v5 code and the inlined generate_input).
# Nothing here feeds kernel_fn: at import it times launch-config variants of the BOARD kernels _full_kernel (verify-full) and _full_cat_kernel (decode-full) on
# generate_input-recipe inputs (3 seeds per shape, read-only L2 flush, CUDA-graph replays, interleaved rounds, board kernel_fn as pivot) and prints the S / C / B lines.
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

# verify-full (_full_kernel): name -> overrides of FULL_BIG {bn, stages, warps, programs}; "B64S4W4" is v3's config (the old board for batch < 16)
_MV_VERIFY = [
    ("B128S3W4", {}), ("B64S4W4", {"bn": 64, "stages": 4}), ("B128S2W4", {"stages": 2}), ("B128S4W4", {"stages": 4}), ("B128S3W8", {"warps": 8}),
    ("B128S2W8", {"stages": 2, "warps": 8}), ("B128P384", {"programs": 384}), ("B128P192", {"programs": 192}), ("B256S2W8", {"bn": 256, "stages": 2, "warps": 8}), ("B256S3W8", {"bn": 256, "stages": 3, "warps": 8}), ("B128P512", {"programs": 512}),
]
# decode-full (_full_cat_kernel): name -> overrides of FULL_CFG["decode"]; "C64S3W2" is the board's config
_MV_DECODE = [
    ("C64S3W2", {}), ("C128S3W2", {"bn": 128}), ("C128S2W2", {"bn": 128, "stages": 2}), ("C128S3W4", {"bn": 128, "warps": 4}), ("C128S4W2", {"bn": 128, "stages": 4}),
    ("C128P528", {"bn": 128, "programs": 528}), ("C128P792", {"bn": 128, "programs": 792}), ("C64S4W2", {"stages": 4}), ("C64S3W4", {"warps": 4}),
    ("C128S2W4", {"bn": 128, "stages": 2, "warps": 4}), ("C128S4W4", {"bn": 128, "stages": 4, "warps": 4}), ("C128S3W4P792", {"bn": 128, "warps": 4, "programs": 792}),
]
_MV_REF = {"verify": "B128S3W4", "decode": "C64S3W2"}     # the variant the others are compared with (output error) and divided by (x)
_MV_NOTES = []


def _mv_elapsed():
    return _mv_time.perf_counter() - _MV_T0


def _mv_note(text):
    _MV_NOTES.append(" ".join(str(text).split())[:200])


def _mv_where(ex):
    return f"{type(ex).__name__}: {str(ex)[:80]}".replace("\n", " ")


def _mv_med(xs):
    xs = [x for x in xs if x is not None]
    return _mv_stats.median(xs) if xs else None


@triton.jit
def _mv_nop(p):
    tl.store(p, 0)


# ================================================================================================================ launch
def _mv_cfg(kind, v):
    c = dict(FULL_BIG if kind == "verify" else FULL_CFG["decode"])
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
    return dict(part_o=part_o, part_m=part_m, part_l=torch.zeros_like(part_m),
                out=torch.empty((pl["batch"], pl["length"], pl["hq"] * 64), dtype=torch.bfloat16, device=dev))


def _mv_launch(data, cfg, kind, pl, bufs):
    """kernel_fn's full-layer branch (the 'sep' kernel for verify, the 'cat' kernel for decode) with an explicit config. -> the main kernel's compiled handle."""
    c = data["config"]
    qkv = data["qkv"]
    dev = qkv.device
    cdtype, native_load, native_store, qk8 = _device_mode(dev.type, dev.index or 0)
    page_table = data["page_table"]
    kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
    grid = (pl["nsplit"] + 1, pl["hkv"] // 2, pl["batch"])
    bt = max(16, triton.next_power_of_2(pl["length"]))
    if kind == "verify":
        h = _full_kernel[grid](
            qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, kc, vc, data["k_scale"], data["v_scale"],
            bufs["part_o"], bufs["part_m"], bufs["part_l"], c["sm_scale"], c["qk_scale_factor"], 0.0, page_table.stride(0), pl["split_len"], pl["nsplit"],
            L=pl["length"], G=pl["group"], GP=pl["gp"], HQ=pl["hq"], HKV=pl["hkv"], PAGE=c["page_size"], ROPE=bool(c["rope_theta"]),
            NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype, BL=pl["bl"], M=pl["rows"], BN=cfg["bn"], BT=bt,
            num_warps=cfg["warps"], num_stages=cfg["stages"],
        )
    else:
        h = _full_cat_kernel[grid](
            qkv, data["positions"], data["cache_loc"], data["seq_lens"], page_table, kc, vc, data["k_scale"], data["v_scale"],
            bufs["part_o"], bufs["part_m"], bufs["part_l"], c["sm_scale"], c["qk_scale_factor"], page_table.stride(0), pl["split_len"], pl["nsplit"],
            HQ=pl["hq"], HKV=pl["hkv"], PAGE=c["page_size"], NATIVE_LOAD=native_load, NATIVE_STORE=native_store, QK8=qk8, CDTYPE=cdtype,
            BN=cfg["bn"], BT=bt, num_warps=cfg["warps"], num_stages=cfg["stages"],
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
    return dict(regs=getattr(h, "n_regs", None), spills=getattr(h, "n_spills", None), smem=int(h.metadata.shared), cpasync=n(r"\bcp\.async\.(?:ca|cg)\b"),
                wgmma=n(r"\bwgmma\.mma_async\b"), mma=n(r"\bmma\.sync\b"), ex2=n(r"\bex2\."), bar=n(r"\bbar(?:rier)?\.(?:cta\.)?sync\b"))


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
def _mv_variants(kind):
    return _MV_VERIFY if kind == "verify" else _MV_DECODE


def _mv_seed(env, case, kind, seed, shape_res, st, ovh, deadline):
    data = _tr_generate_input(**{**case, "seed": seed})
    runners = []        # (name, graph, bufs)
    g, _ = env.capture(lambda: kernel_fn(data))      # the board kernel (sonnet_v5), always
    runners.append(("board", g, None))
    for name, v in _mv_variants(kind):
        res = shape_res.setdefault(name, dict(us=[], err=[], perr=[], ok=[], counts=None, status="ok"))
        if name in st["dead"]:
            res["status"] = st["dead"][name]
            continue
        fresh = (kind, name) not in st["compiled"]
        if fresh and _mv_elapsed() + 1.3 * st["est"] > deadline:
            st["dead"][name] = res["status"] = "skip budget"
            continue
        t1 = _mv_time.perf_counter()
        try:
            cfg = _mv_cfg(kind, v)
            pl = _mv_plan(data, cfg)
            bufs = _mv_bufs(data, pl)
            g, h = env.capture(lambda: _mv_launch(data, cfg, kind, pl, bufs))
            if res["counts"] is None:
                res["counts"] = _mv_counts(h)
            runners.append((name, g, bufs))
            st["compiled"].add((kind, name))
        except Exception as ex:  # noqa: BLE001
            st["dead"][name] = res["status"] = "fail " + _mv_where(ex)
        finally:
            if fresh:
                st["est"] = max(st["est"], _mv_time.perf_counter() - t1)
    # outputs vs the reference variant (and the board's own)
    errs, ref_name = {}, _MV_REF[kind]
    refb = next((b for n, _, b in runners if n == ref_name), None)
    if refb is not None:
        for name, g, bufs in runners:
            if bufs is None:
                continue
            g.replay()
            torch.cuda.synchronize()
            ok, e, row = _mv_cmp(bufs["out"], refb["out"])
            pe = None
            if bufs["part_o"].shape == refb["part_o"].shape:
                pe = max(float((bufs[k] - refb[k]).norm() / refb[k].norm().clamp(min=1e-30)) for k in ("part_o", "part_l"))
            errs[name] = (ok, e, pe)
    samples = {n: [] for n, _, _ in runners}
    for rep in range(_MV_REPS + 1):
        order = runners if rep % 2 == 0 else runners[::-1]
        for name, g, _ in order:
            us = env.replay_us(g)
            if rep > 0:                     # the first round is a warm-up
                samples[name].append(us)
    for name, _, _ in runners:
        res = shape_res.setdefault(name, dict(us=[], err=[], perr=[], ok=[], counts=None, status="ok"))
        res["us"].append(max(0.01, _mv_med(samples[name]) - ovh))
        if name in errs:
            res["ok"].append(errs[name][0])
            res["err"].append(errs[name][1])
            res["perr"].append(errs[name][2])
    g = None
    del runners, data, refb
    torch.cuda.empty_cache()


# ================================================================================================================ output
def _mv_fmt_err(e):
    return "inf" if e == float("inf") else f"{e:.1e}"


def _mv_xs(shape_res, ref, name, k):
    b, r = shape_res.get(ref), shape_res.get(name)
    if not b or not r or k >= len(b["us"]) or k >= len(r["us"]):
        return None
    return b["us"][k] / r["us"][k]


def _mv_render(results, meta):
    """results: {idx: dict(label, kind, seeds, v: {name: res})}. -> text <= _MV_MAX_BYTES, S lines first."""
    S, C, B = [], [], []
    S.append(f"S attn verify v2: gpu={meta['gpu']} used={meta['secs']:.0f}s of {_MV_BUDGET_S:.0f}s ovh={meta['ovh']:.2f}us(subtracted) seeds={len(_MV_SEEDS)} reps={_MV_REPS}"
             f" shapes={len(results)}/{len(_MV_CASES)}; x = reference variant / variant (verify: {_MV_REF['verify']}, decode: {_MV_REF['decode']} = board)")
    for idx, r in results.items():
        v, ref = r["v"], _MV_REF[r["kind"]]
        med = {n: _mv_med(x["us"]) for n, x in v.items()}
        if "board" not in med:
            continue
        parts = [f"board={med['board']:.1f}"]
        for n, _ in _mv_variants(r["kind"]):
            if med.get(n) is not None and med.get(ref):
                parts.append(f"{n}={med[n]:.1f}(x{med[ref] / med[n]:.2f})")
            elif n in v and v[n]["status"] != "ok":
                parts.append(f"{n}={v[n]['status'][:30]}")
        S.append(f"S #{idx} {r['label']} us: " + " ".join(parts))
        errs = [max(x["err"]) for n, x in v.items() if x["err"]]
        if errs:
            S.append(f"S #{idx} max output rel L2 vs {ref}: {_mv_fmt_err(max(errs))}; BAD variants: " + (",".join(n for n, x in v.items() if x["ok"] and not all(x["ok"])) or "none"))
    for kind in ("verify", "decode"):
        first = next((r for r in results.values() if r["kind"] == kind), None)
        for n, _ in _mv_variants(kind) if first else []:
            c = first["v"].get(n, {}).get("counts")
            if c:
                C.append(f"C {n} regs={c['regs']} spills={c['spills']} smem={c['smem']} cp.async={c['cpasync']} wgmma={c['wgmma']} mma={c['mma']} ex2={c['ex2']} bar={c['bar']}")
    for note in _MV_NOTES[:6]:
        S.append("S note: " + note)
    for idx, r in results.items():
        ref = _MV_REF[r["kind"]]
        for k, seed in enumerate(r["seeds"]):
            for n in ["board"] + [n for n, _ in _mv_variants(r["kind"])]:
                x = r["v"].get(n)
                if not x:
                    continue
                if k >= len(x["us"]):
                    if x["status"] != "ok" and k == 0:
                        B.append(f"B attn #{idx} seed={seed} | {n} {x['status'][:60]}")
                    continue
                sx = _mv_xs(r["v"], ref, n, k)
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
            print("S attn verify v2: no CUDA device (or capturing): nothing measured", flush=True)
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
            kind = "verify" if case.get("q_len", 1) > 1 else "decode"
            r = results.setdefault(idx, dict(label=label, kind=kind, seeds=[], v={}))
            for seed in _MV_SEEDS:
                if _mv_elapsed() > deadline - 6.0 and r["seeds"]:
                    _mv_note(f"#{idx} {label}: seeds after {len(r['seeds'])} skipped (budget)")
                    break
                try:
                    _mv_seed(env, case, kind, seed, r["v"], st, meta["ovh"], deadline)
                    r["seeds"].append(seed)
                except Exception as ex:  # noqa: BLE001
                    _mv_note(f"#{idx} {label} seed {seed}: {_mv_where(ex)}")
                    torch.cuda.empty_cache()
        meta["secs"] = _mv_elapsed()
        print(_mv_render(results, meta), flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"S attn verify v2 FAILED: {_mv_where(ex)}", flush=True)
        try:
            meta["secs"] = _mv_elapsed()
            print(_mv_render(results, meta), flush=True)
        except Exception:  # noqa: BLE001
            pass


if not globals().get("_MV_NO_AUTORUN"):
    _mv_study()
