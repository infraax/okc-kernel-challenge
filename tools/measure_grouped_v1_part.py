# ============================================================================================ MEASUREMENT (appended by tools/gen_measure_grouped_v1.py)
import time  # noqa: E402
import traceback  # noqa: E402

BUDGET_S = 225.0        # prepare() may run 300 s; the whole run has 15 min (compiles included)
PHASE_END_S = {"A": 30.0, "B": 105.0, "C": 150.0, "D": 190.0}
SCREEN_REPS = 9
FINAL_REPS = 41
FLUSH_MB = 128          # read-only L2 flush buffer (> the 50 MB L2), like the judge
SHAPES = (32, 128, 256)  # decode 32, verify 16, verify 32
SEED = 20241024
TOL = 2e-2              # relative L2 of a grouped GEMM against Marlin's, same inputs (BF16 vs FP32 epilogue: ~1e-3 expected)
_BF = torch.bfloat16    # tests on the sm_75 dev card swap in float16

# GEMM1 (gated: bnh = output columns, the weight tile is gate + up) and GEMM2 (bnh = output columns) candidates:
# (bm, bnh, ch, warps, stages). Hand-ordered by prior; the sweep stops at the phase deadline. Compile key = (gated, these, deq).
G1_BIG = ((64, 64, 1, 8, 3), (64, 32, 1, 4, 3), (64, 64, 1, 4, 3), (64, 128, 1, 8, 3), (64, 32, 1, 8, 3), (128, 64, 1, 8, 3),
          (128, 32, 1, 8, 3), (32, 64, 1, 4, 3), (32, 128, 1, 8, 3), (64, 64, 1, 8, 4), (64, 64, 1, 8, 2), (64, 32, 1, 4, 4),
          (128, 32, 1, 8, 4), (64, 64, 2, 8, 2), (64, 32, 2, 8, 2), (128, 64, 1, 8, 2), (32, 32, 1, 4, 3), (32, 64, 1, 8, 3))
G2_BIG = ((64, 128, 1, 8, 3), (64, 64, 1, 4, 3), (64, 64, 1, 8, 3), (64, 256, 1, 8, 3), (128, 128, 1, 8, 3), (128, 64, 1, 8, 3),
          (32, 128, 1, 4, 3), (32, 128, 1, 8, 3), (64, 128, 1, 8, 4), (64, 128, 1, 4, 3), (64, 128, 1, 8, 2), (128, 64, 1, 8, 4),
          (64, 64, 2, 8, 2), (64, 128, 2, 8, 2), (32, 64, 1, 4, 3))
G1_SMALL = ((32, 64, 1, 4, 3), (32, 32, 1, 4, 3), (32, 64, 1, 8, 3), (16, 64, 1, 4, 3), (16, 32, 1, 4, 3), (64, 64, 1, 8, 3),
            (64, 32, 1, 4, 3), (32, 128, 1, 8, 3), (32, 64, 1, 4, 4), (16, 64, 1, 4, 4), (64, 64, 1, 4, 3))
G2_SMALL = ((32, 128, 1, 4, 3), (32, 64, 1, 4, 3), (32, 128, 1, 8, 3), (16, 128, 1, 4, 3), (16, 64, 1, 4, 3), (64, 128, 1, 8, 3),
            (64, 64, 1, 4, 3), (32, 256, 1, 8, 3), (32, 128, 1, 4, 4), (16, 128, 1, 4, 4), (64, 128, 1, 4, 3))
SMEM_BUDGET = 200 * 1024


def g_table(bs, gemm):
    """Candidate (bm, bnh, ch, warps, stages) tuples of GEMM `gemm` (1 or 2) at `bs` tokens that fit shared memory."""
    tab = {(1, True): G1_SMALL, (2, True): G2_SMALL, (1, False): G1_BIG, (2, False): G2_BIG}[(gemm, bs <= 64)]
    out = []
    for bm, bnh, ch, warps, stages in tab:
        w_rows = bnh * (2 if gemm == 1 else 1)
        smem = stages * (bm * 128 * ch * 2 + w_rows * 64 * ch) + w_rows * 128 * ch * 2
        if smem <= SMEM_BUDGET:
            out.append((bm, bnh, ch, warps, stages))
    return out


def cfg_name(c, deq):
    bm, bnh, ch, warps, stages = c
    return f"d{deq}b{bm}n{bnh}c{ch}w{warps}s{stages}"


def _short(e):
    return f"{type(e).__name__}: {str(e)[:70]}".replace("\n", " ")


def _routing(bs, dist, dev, gen):
    """Synthetic routing like reference.generate_input (not imported): top-8 of 256, ids rotated so experts 0-31 are the busiest GPU's."""
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=dev)
    if dist == "normal":
        z = torch.randn((N_EXPERTS,), generator=gen, device=dev)
        u = torch.rand((N_EXPERTS,), generator=gen, device=dev).clamp(min=1e-12)
        scores = scores + 0.03 * z / (-u.log()).sqrt()
    else:
        scores = scores + torch.rand((N_EXPERTS,), generator=gen, device=dev).pow(6) * 4
    routed = scores.topk(TOP_K, dim=-1)
    ids = routed.indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=N_EXPERTS // 32)
    ids = (ids - 32 * pairs.argmax()) % N_EXPERTS
    w = (routed.values / routed.values.sum(-1, keepdim=True)).float()
    return ids.to(torch.int32), w


class _RealEnv:
    """The GPU side: synthetic inputs, Marlin references and baselines, grouped closures, CUDA-graph timing after a read-only L2 flush."""

    def __init__(self, weights, state):
        self.weights = weights
        self.state = state
        self.dev = weights["w13"].device
        self.sms = torch.cuda.get_device_properties(self.dev).multi_processor_count
        self.native = torch.cuda.get_device_capability(self.dev) >= (8, 9)
        self.g = {}  # swap12 -> repacked weights

    def available(self):
        return torch.cuda.is_available() and HAVE_SGLANG

    def clock(self):
        return time.time()

    def banner(self):
        return f"gpu={torch.cuda.get_device_name()} sms={self.sms} triton={triton.__version__} torch={torch.__version__} fp8={int(self.native)}"

    def setup(self):
        self.flush = torch.ones(FLUSH_MB * 1024 * 1024 // 4, device=self.dev, dtype=torch.float32)
        self.sink = torch.zeros((), device=self.dev, dtype=torch.float32)
        self.gen = torch.Generator(device=self.dev).manual_seed(SEED)

    def weights_for(self, deq):
        key = deq == 2
        if key not in self.g:
            self.g[key] = _prepare_grouped(self.weights, 0, 32, self.sms, deq)
        return self.g[key]

    def sync(self):
        torch.cuda.synchronize()

    def release(self):
        for a in ("flush", "sink", "g"):
            if hasattr(self, a):
                delattr(self, a)
        torch.cuda.empty_cache()

    # --- one shape
    def prep(self, bs, dist="skewed"):
        dev = self.dev
        ids, tw = _routing(bs, dist, dev, self.gen)
        x = torch.randn((bs, self.weights["w13"].shape[2] * 2), generator=self.gen, device=dev).to(_BF)
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": tw,
                "config": {"local_start": 0, "local_count": 32, "bs": bs, "top_k": TOP_K, "n_experts": N_EXPERTS}}
        local_ids = local_ids_triton(ids, 0, 32)
        ctx = dict(bs=bs, dist=dist, data=data, x=x, local_ids=local_ids, tw=tw, slots=bs * TOP_K, aligns={})
        bank = self.state["banks"][(0, 32)]
        size_n, size_k = bank["size_n"], x.shape[1]
        gemm1_n = 2 * size_n
        bsm = next((b for b in BLOCK_SIZE_M_CANDIDATES if not bs * TOP_K / 32 / b < 0.9), BLOCK_SIZE_M_CANDIDATES[-1])
        s_ids, b_exp, n_pad = moe_align_block_size(local_ids, bsm, N_EXPERTS)
        cache2 = torch.empty((ctx["slots"], size_n), device=dev, dtype=x.dtype)
        cache13 = torch.empty((ctx["slots"] * max(gemm1_n, size_k),), device=dev, dtype=x.dtype)
        cache1 = cache13[: ctx["slots"] * gemm1_n].view(-1, gemm1_n)
        cache3 = cache13[: ctx["slots"] * size_k].view(-1, size_k)
        common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True, b_q_type=scalar_types.float4_e2m1f, is_k_full=True,
                      use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
        ws = self.state["workspace"]

        def m1():
            return moe_wna16_marlin_gemm(
                x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws, s_ids, b_exp, n_pad, tw, top_k=TOP_K,
                size_m=bs, size_n=gemm1_n, size_k=size_k, **common)

        def m2():
            return moe_wna16_marlin_gemm(
                cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws, s_ids, b_exp, n_pad, tw, top_k=1,
                size_m=ctx["slots"], size_n=size_k, size_k=size_n, **common)
        out1 = m1()
        silu_and_mul(out1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
        ref2 = m2().clone()
        ctx.update(m1=m1, m2=m2, cache2=cache2, live=(local_ids.view(-1) >= 0), ref1=cache2.clone(), ref2=ref2, size_n=size_n,
                   size_k=size_k, bsm=bsm)
        torch.cuda.synchronize()
        return ctx

    def align(self, ctx, bm):
        if bm not in ctx["aligns"]:
            ctx["aligns"][bm] = _align_sglang(ctx["local_ids"], bm)
        return ctx["aligns"][bm]

    # --- closures (nothing allocated at replay time)
    def g_run(self, ctx, gemm, c, deq, mult=2):
        bm, bnh, ch, warps, stages = c
        g = self.weights_for(deq)
        sorted_ids, expert_ids, n_post = self.align(ctx, bm)
        slots, hidden, inter = ctx["slots"], ctx["size_k"], ctx["size_n"]
        ctr = torch.zeros(32, dtype=torch.int32, device=self.dev)
        nprog = self.sms * mult
        if gemm == 1:
            out = torch.empty((slots, inter), dtype=ctx["x"].dtype, device=self.dev)
            a, w, s, gs, k, n_out, a_div, gated = ctx["x"], g["w13"], g["s13"], g["gs13"], hidden, inter, TOP_K, True
        else:
            out = torch.empty((slots, hidden), dtype=ctx["x"].dtype, device=self.dev)
            a, w, s, gs, k, n_out, a_div, gated = ctx["ref1"], g["w2"], g["s2"], g["gs2"], inter, hidden, 1, False

        def run():
            ctr.zero_()
            grouped_gemm(a, w, s, gs, out, sorted_ids, expert_ids, n_post, ctr[0:1], slots, k=k, n_out=n_out, a_div=a_div, gated=gated,
                         bm=bm, bnh=bnh, ch=ch, warps=warps, stages=stages, nprog=nprog, native=self.native, deq=deq)
            return out
        return run

    def e2e_run(self, ctx, c1, c2, deq, mult=1):
        """The whole large-M path of kernel_fn with these configs (grouped_cfg patched); c1 / c2 must share bm."""
        cfg = {"bm": c1[0], "g1": dict(bnh=c1[1], ch=c1[2], warps=c1[3], stages=c1[4]),
               "g2": dict(bnh=c2[1], ch=c2[2], warps=c2[3], stages=c2[4]), "deq": deq, "mult": mult}
        st = {"grouped": {(0, 32): self.weights_for(deq)}, "norm_w": self.state["norm_w"]}
        data, native = ctx["data"], self.native

        def run():
            globals()["grouped_cfg"] = lambda m: cfg
            return _kernel_grouped(data, st, _align_sglang, _post_sglang, native)
        return run

    def marlin_e2e(self, ctx):
        data, st = ctx["data"], self.state
        return lambda: _kernel_sglang(data, st)

    # --- checks / timing
    def _err(self, out, ref, live):
        d = (out[live].float() - ref[live].float())
        return float(d.norm() / ref[live].float().norm())

    def check(self, ctx, gemm, run):
        """First launch (compiles). Relative L2 error against Marlin's output on the same inputs (live slots only)."""
        out = run()
        torch.cuda.synchronize()
        return self._err(out, ctx["ref1"] if gemm == 1 else ctx["ref2"], ctx["live"])

    def time(self, fn, reps):
        """Median µs of graph replays, each preceded by a read-only L2 flush."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ts = []
        for _ in range(reps):
            torch.sum(self.flush, dim=0, out=self.sink)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) * 1000.0)
        ts.sort()
        del g
        return ts[len(ts) // 2]


def _measure(weights, state, env=None):
    """MEASUREMENT: see the module docstring. `env` is injected by tests (stubbed timing / GPU-free)."""
    env = env or _RealEnv(weights, state)
    if not env.available():
        print("MEASG skipped: no cuda / no SGLang")
        return
    t0 = env.clock()
    res = {}      # (bs, gemm) -> {(cfg tuple, deq): us | "F" | "B"}
    marlin = {}   # bs -> dict(m1=, m2=, e2e=)
    e2e = {}      # bs -> list of (us, name)
    notes = []
    poisoned = {}
    state_ = {"deq": 1}
    print(f"MEASG start {env.banner()} budget={BUDGET_S:.0f}s screen={SCREEN_REPS} finals={FINAL_REPS} tol={TOL}", flush=True)
    print("MEASG name=d<deq>b<BM>n<BNH>c<CH>w<warps>s<stages>[g<ctas/SM>]; us = graph-replay median after an L2 flush, Marlin timed the same way")

    def progress(name):
        n = sum(isinstance(v, float) for r in res.values() for v in r.values())
        print(f"MEASG phase {name} t={env.clock() - t0:.0f}s timed={n} poisoned={len(poisoned)}", flush=True)

    ctxs = {}

    def try_cfg(bs, gemm, c, deq, reps):
        key = (gemm, c, deq)
        if key in poisoned:
            return "F"
        ctx = ctxs[bs]
        try:
            run = env.g_run(ctx, gemm, c, deq)
            if not (env.check(ctx, gemm, run) < TOL):
                return "B"
            return env.time(run, reps)
        except Exception as e:
            poisoned[key] = _short(e)
            return "F"

    def sweep(gemm, end_s, deq, shapes):
        for bs in shapes:
            res.setdefault((bs, gemm), {})
        # config-major: a compile key is compiled once (first shape) and reused by the others
        seen = []
        for bs in shapes:
            for c in g_table(bs, gemm):
                if c not in seen:
                    seen.append(c)
        for c in seen:
            if env.clock() - t0 > end_s:
                return
            for bs in shapes:
                if c in g_table(bs, gemm) and (c, deq) not in res[(bs, gemm)]:
                    res[(bs, gemm)][(c, deq)] = try_cfg(bs, gemm, c, deq, SCREEN_REPS)

    def best(bs, gemm, bm=None):
        ok = [(v, k) for k, v in res.get((bs, gemm), {}).items() if isinstance(v, float) and (bm is None or k[0][0] == bm)]
        return min(ok) if ok else None

    try:
        env.setup()
        for bs in SHAPES:
            ctxs[bs] = env.prep(bs)
            marlin[bs] = dict(m1=env.time(ctxs[bs]["m1"], SCREEN_REPS), m2=env.time(ctxs[bs]["m2"], SCREEN_REPS),
                              e2e=env.time(env.marlin_e2e(ctxs[bs]), SCREEN_REPS))
        progress("setup")

        # A: which dequant mode, on the prior-best GEMM1 config at the biggest shape (bs 256), 2 / 1 / 0
        pbs = max(SHAPES)
        probe = g_table(pbs, 1)[0]
        modes = {}
        for deq in (2, 1, 0):
            modes[deq] = try_cfg(pbs, 1, probe, deq, SCREEN_REPS)
            res.setdefault((pbs, 1), {})[(probe, deq)] = modes[deq]
        good = [(v, d) for d, v in modes.items() if isinstance(v, float)]
        state_["deq"] = min(good)[1] if good else 1
        notes.append("modes " + " ".join(f"d{d}={v if isinstance(v, str) else format(v, '.1f')}" for d, v in sorted(modes.items())))
        progress("A")

        sweep(1, PHASE_END_S["B"], state_["deq"], SHAPES)
        progress("B")
        sweep(2, PHASE_END_S["C"], state_["deq"], SHAPES)
        progress("C")

        # D: whole large-M path per shape, per BM: the best GEMM1 and GEMM2 config of that BM; then 2 CTAs per SM on the winner
        for bs in SHAPES:
            e2e[bs] = []
            for bm in sorted({k[0][0] for k in res.get((bs, 1), {})} & {k[0][0] for k in res.get((bs, 2), {})}):
                b1, b2 = best(bs, 1, bm), best(bs, 2, bm)
                if not b1 or not b2 or env.clock() - t0 > PHASE_END_S["D"]:
                    continue
                for mult in (1, 2):
                    try:
                        run = env.e2e_run(ctxs[bs], b1[1][0], b2[1][0], state_["deq"], mult)
                        run()
                        env.sync()
                        e2e[bs].append((env.time(run, SCREEN_REPS), f"{cfg_name(b1[1][0], state_['deq'])}+{cfg_name(b2[1][0], state_['deq'])}g{mult}"))
                    except Exception as e:
                        notes.append(f"e2e bs={bs} bm={bm} g{mult} {_short(e)}")
                        break
        progress("D")

        # E: finals (more reps) for the top-3 GEMM configs per shape and for the best e2e; the other dequant modes on the best GEMM1
        for bs in SHAPES:
            for gemm in (1, 2):
                top = sorted((v, k) for k, v in res.get((bs, gemm), {}).items() if isinstance(v, float))[:3]
                for _, (c, deq) in top:
                    if env.clock() - t0 > BUDGET_S:
                        break
                    v = try_cfg(bs, gemm, c, deq, FINAL_REPS)
                    if isinstance(v, float):
                        res[(bs, gemm)][(c, deq)] = v
            b1 = best(bs, 1)
            if b1 and env.clock() - t0 < BUDGET_S:
                for deq in (0, 1, 2):
                    if deq != b1[1][1] and (b1[1][0], deq) not in res[(bs, 1)]:
                        res[(bs, 1)][(b1[1][0], deq)] = try_cfg(bs, 1, b1[1][0], deq, SCREEN_REPS)
            if e2e.get(bs) and env.clock() - t0 < BUDGET_S:
                e2e[bs].sort()
        progress("E")
    except Exception as e:  # e.g. a device-side fault: report what we have
        print(f"MEASG aborted: {_short(e)}")
        print("MEASG trace " + " / ".join(l.strip() for l in traceback.format_exc().splitlines()[-8:])[:900])
    finally:
        _report(env, t0, res, marlin, e2e, notes, poisoned)


def _report(env, t0, res, marlin, e2e, notes, poisoned):
    lines, s_lines = [], []

    def order(kv):
        return (0, kv[1]) if isinstance(kv[1], float) else (1, 0.0)
    for bs in SHAPES:
        for gemm in (1, 2):
            r = res.get((bs, gemm), {})
            toks = " ".join(f"{cfg_name(c, d)}:{('%.1f' % v) if isinstance(v, float) else v}" for (c, d), v in sorted(r.items(), key=order))
            s_lines.append(f"S moeg g{gemm} bs={bs} {toks}")
    for bs in SHAPES:
        m = marlin.get(bs)
        parts = [f"B moeg bs={bs}"]
        if m:
            parts.append(f"marlin g1={m['m1']:.1f} g2={m['m2']:.1f} e2e={m['e2e']:.1f}")
        for gemm in (1, 2):
            ok = sorted((v, k) for k, v in res.get((bs, gemm), {}).items() if isinstance(v, float))[:5]
            parts.append(f"g{gemm} top5 " + (" ".join(f"{cfg_name(c, d)}:{v:.1f}" for v, (c, d) in ok) or "-"))
            if ok and m:
                parts.append(f"g{gemm}_vs_marlin {m['m' + str(gemm)] / ok[0][0]:.2f}x")
        es = e2e.get(bs) or []
        if es and m:
            u, name = min(es)
            parts.append(f"e2e {name}:{u:.1f} vs marlin {m['e2e']:.1f} = {m['e2e'] / u:.2f}x")
        r = [v for g_ in (1, 2) for v in res.get((bs, g_), {}).values()]
        parts.append(f"ok={sum(isinstance(v, float) for v in r)} fail={sum(v == 'F' for v in r)} bad={sum(v == 'B' for v in r)}")
        lines.append(" | ".join(parts))
    for n in notes:
        print(f"MEASG {n}"[:300])
    seen = set()
    for k, msg in sorted(poisoned.items(), key=lambda kv: str(kv[0])):
        if msg not in seen and len(seen) < 6:
            seen.add(msg)
            print(f"FAIL {cfg_name(k[1], k[2])} gemm{k[0]} {msg}")
    for line in s_lines:  # the bulk first; the log tail (16 kB) keeps the B lines below
        print(line[:1500])
    print(f"MEASG done in {env.clock() - t0:.1f}s")
    for line in lines:  # last: if the tail is ever cut, the human lines survive
        print(line[:1200])


def prepare(weights):
    """Called once, untimed: build SGLang's Marlin state (the kernel_fn below is sonnet_v1's, i.e. the Marlin path for every shape), then
    measure."""
    state = _prepare_sglang(weights) if HAVE_SGLANG else {"weights": weights, "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    if HAVE_SGLANG and torch.cuda.is_available():
        _measure(weights, state)
        torch.cuda.synchronize()
    else:
        print("MEASG skipped: no cuda / no SGLang")
    return state
