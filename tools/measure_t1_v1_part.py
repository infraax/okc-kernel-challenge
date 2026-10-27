# ---------------------------------------------------------------------------------------------------------------------
# MEASUREMENT-ONLY block (tools/gen_measure_t1_v1.py appends it after the unchanged sonnet_v5f code, the derived _t1_kernel and the inlined generate_input).
# Nothing here feeds kernel_fn: prepare() runs sonnet_v5f's prepare(), then times the C37 variants on generate_input-recipe inputs (3 seeds, read-only L2 flush,
# CUDA-graph replays of the whole layer, the graph's fixed cost subtracted) and prints the S / B / D lines.
# ---------------------------------------------------------------------------------------------------------------------
import time as _t1_time

_T1_T0 = _t1_time.perf_counter()
_T1_BUDGET_S = 262.0        # prepare() ends by here (the judge's limit is 300 s); variants that do not fit are dropped
_T1_SEEDS = (11, 12, 13)
_T1_CTX = 80000             # the judged context
_T1_REPS = 9                # graph replays per timing (median), each after a read-only L2 flush
_T1_FLUSH_MB = 128
_T1_MAX_BYTES = 13800
# (name, generate_input arguments); the order is the priority when the budget runs short
_T1_SHAPES = (("dec1s", dict(batch=1, layer="sliding", q_len=1, spread="fixed")),
              ("dec1f", dict(batch=1, layer="full", q_len=1, spread="fixed")),
              ("dec16s", dict(batch=16, layer="sliding", q_len=1, spread="agentic")))
_T1_WEIGHTS = {"dec1s": 46, "dec1f": 16}      # the loads' weights of the two T = 1 shapes (TASK.md)
_T1_WIN_US = 2.0            # D line: a winner gains this much at T = 1 (weighted over sliding + full) ...
_T1_REG_PCT = 1.0           # ... and loses at most this much (percent) at decode 16
_T1_COMBO_US = 1.0          # a single variant enters the combo from this gain
_T1_NOTES = []


def _t1_left():
    return _T1_BUDGET_S - (_t1_time.perf_counter() - _T1_T0)


def _t1_note(text):
    _T1_NOTES.append(" ".join(str(text).split())[:200])


def _t1_secs(x):
    return '-' if x is None else '%.0fs' % x


def _t1_med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ===================================================================================================== the variants
class _T1V:
    """One variant = module switches on the small-M experts / the route placement (every one of them config, launch or stream placement: bitwise v5f).
    nz1   grid z of GEMM1 at M = 1 (SMALLM_CFG's row is 2)         trig_late  pdl_trigger after the K-loop instead of before it (needs the patched kernel)
    rows2 dynamic smem for 8 staged token rows instead of 16 at M <= 8 (the kernel stages 8 there; needs the patched host launch)
    route_side  smallm_route on the side stream right after the routing"""

    def __init__(self, name, nz1=None, trig_late=False, rows2=False, route_side=False, ext2=False):
        self.name, self.nz1, self.trig_late, self.rows2, self.route_side = name, nz1, trig_late, rows2, route_side
        self.ext2 = ext2 or trig_late or rows2          # runs through the patched extension (trig_late / rows are its extra launch arguments)

    @property
    def custom(self):
        return self.nz1 is not None or self.ext2 or self.route_side


_T1_SINGLES = (_T1V("base"), _T1V("nz8", nz1=8), _T1V("trig_early", ext2=True), _T1V("trig_late", trig_late=True), _T1V("rows2", rows2=True),
               _T1V("route_side", route_side=True))
_T1_NAMES = tuple(v.name for v in _T1_SINGLES) + ("combo",)


def _t1_combo(winners):
    """The combo of the winners (their switches OR-ed); none -> every switch together (so the check still says what the lot does)."""
    ws = [v for v in _T1_SINGLES if v.name in winners] or [v for v in _T1_SINGLES if v.name in ("nz8", "trig_late", "rows2", "route_side")]
    c = _T1V("combo", nz1=8 if any(v.nz1 for v in ws) else None, trig_late=any(v.trig_late for v in ws), rows2=any(v.rows2 for v in ws),
             route_side=any(v.route_side for v in ws))
    return c, tuple(v.name for v in ws)


# ===================================================================================================== the patched small-M extension (trig_late / rows)
def _t1_patch(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def _t1_device_src():
    """v5f's small-M device source with ONE extra kernel argument, trig_late: 0 = pdl_trigger before the K-loop (v5f), 1 = right after it."""
    s = _SMALLM_DEVICE_SRC
    s = _t1_patch(s, "u16* __restrict__ out, int G, int NIT, int K, int KS, int out_stride, int act_shift) {\n    extern __shared__ __align__(16) u8 smem_raw[];\n    __shared__ int s_flag;",
                  "u16* __restrict__ out, int G, int NIT, int K, int KS, int out_stride, int act_shift, int trig_late) {\n    extern __shared__ __align__(16) u8 smem_raw[];\n    __shared__ int s_flag;")
    s = _t1_patch(s, "    if (FUSE) pdl_trigger();\n    const int Kc = K / KS, nit = Kc >> 6, KP = Kc + 8, kp_bytes = KP * 2;",
                  "    if (FUSE && !trig_late) pdl_trigger();\n    const int Kc = K / KS, nit = Kc >> 6, KP = Kc + 8, kp_bytes = KP * 2;")
    s = _t1_patch(s, "        else k_loop<2>(w0, s0, w1, s1, wp, sp, nbatch, lane_addr, kp_bytes, s_lut, acc);\n",
                  "        else k_loop<2>(w0, s0, w1, s1, wp, sp, nbatch, lane_addr, kp_bytes, s_lut, acc);\n        if (FUSE && trig_late) pdl_trigger();   // C37 TRIG_LATE: the dependents launch when the K-loop is done\n")
    return s


_T1_HOST_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

static const int SM_SMEM_MAX = 98304;
#define T1_CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { cudaGetLastError(); \
        TORCH_CHECK(false, "t1: ", #call, " failed: ", cudaGetErrorName(e_), " (", cudaGetErrorString(e_), ")"); } } while (0)

// launch_gemm_t of v5f with two more arguments: rows (token rows of dynamic smem: 16 in v5f, 8 is enough at M <= 8) and trig_late
template <bool FUSE, int NW>
static void launch_gemm_t1(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor tab,
                           torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS,
                           int64_t nz, int64_t out_stride, int64_t act_shift, int64_t rows, int64_t trig_late) {
    const int Kc = (int)(K / KS);
    const size_t smem = 1024 + (size_t)rows * (Kc + 8) * 2;
    TORCH_CHECK(smem <= (size_t)SM_SMEM_MAX && (rows == 8 || rows == 16), "t1: bad smem ", smem, " rows ", rows);
    auto fn = w4_smallm_kernel<FUSE, NW>;
    static bool inited = false;
    if (!inited) {
        T1_CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SMEM_MAX));
        inited = true;
    }
    const uint4* a_wq = reinterpret_cast<const uint4*>(wq.data_ptr());
    const uint2* a_sq = reinterpret_cast<const uint2*>(sq.data_ptr());
    const float* a_gs = gs.data_ptr<float>();
    const u32* a_lut = reinterpret_cast<const u32*>(lut.data_ptr());
    const u16* a_act = reinterpret_cast<const u16*>(act.data_ptr());
    const int* a_tab = tab.data_ptr<int>();
    float4* a_part = reinterpret_cast<float4*>(part.data_ptr());
    int* a_cnt = cnt.data_ptr<int>();
    u16* a_out = reinterpret_cast<u16*>(out.data_ptr());
    int i_G = (int)G, i_NIT = (int)NIT, i_K = (int)K, i_KS = (int)KS, i_os = (int)out_stride, i_as = (int)act_shift, i_tl = (int)trig_late;
    void* args[] = {&a_wq, &a_sq, &a_gs, &a_lut, &a_act, &a_tab, &a_part, &a_cnt, &a_out, &i_G, &i_NIT, &i_K, &i_KS, &i_os, &i_as, &i_tl};
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(G / NW), (unsigned)KS, (unsigned)nz);
    cfg.blockDim = dim3(NW * 32, 1, 1);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    if (at::cuda::getCurrentDeviceProperties()->major >= 9) {
        cfg.attrs = attr;
        cfg.numAttrs = 1;
    }
    const cudaError_t e = cudaLaunchKernelExC(&cfg, (const void*)fn, args);
    if (e != cudaSuccess) {
        cudaGetLastError();
        TORCH_CHECK(false, "t1: launch failed: ", cudaGetErrorName(e), " fuse=", (int)FUSE, " nw=", NW, " grid=", G / NW, "x", KS, "x", nz, " smem=", smem);
    }
}

void smallm_gemm_t1(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor tab,
                    torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz,
                    int64_t out_stride, int64_t act_shift, int64_t nw, int64_t rows, int64_t trig_late, bool fuse) {
    c10::cuda::CUDAGuard guard(act.device());
#define T1_CASE(F, W) if (fuse == F && nw == W) { launch_gemm_t1<F, W>(wq, sq, gs, lut, act, tab, part, cnt, out, G, NIT, K, KS, nz, out_stride, act_shift, rows, trig_late); return; }
    T1_CASE(true, 8) T1_CASE(false, 4) T1_CASE(false, 8)
#undef T1_CASE
    TORCH_CHECK(false, "t1: no instantiation for fuse/nw = ", fuse, "/", nw);
}
"""
_T1_CPP_SRC = ("void smallm_gemm_t1(torch::Tensor wq, torch::Tensor sq, torch::Tensor gs, torch::Tensor lut, torch::Tensor act, torch::Tensor tab, "
               "torch::Tensor part, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t NIT, int64_t K, int64_t KS, int64_t nz, "
               "int64_t out_stride, int64_t act_shift, int64_t nw, int64_t rows, int64_t trig_late, bool fuse);\n")
_T1_EXT = {"ext": None, "err": None, "secs": None, "make": None}    # make: tests inject an NVRTC extension factory


def _t1_build_ext():
    """Builds the patched extension once (3 instantiations: <true,8>, <false,4>, <false,8>); returns it or None (the variants that need it are then dropped)."""
    if _T1_EXT["ext"] is not None or _T1_EXT["err"] is not None:
        return _T1_EXT["ext"]
    t0 = _t1_time.perf_counter()
    try:
        if _T1_EXT["make"] is not None:
            _T1_EXT["ext"] = _T1_EXT["make"](_t1_device_src())
        else:
            import hashlib

            from torch.utils.cpp_extension import load_inline

            src = _t1_device_src() + _T1_HOST_SRC
            tag = hashlib.sha1(src.encode()).hexdigest()[:12]
            _T1_EXT["ext"] = load_inline(name=f"okc_t1_{tag}", cpp_sources=_T1_CPP_SRC, cuda_sources=src, functions=["smallm_gemm_t1"],
                                         extra_cuda_cflags=["-O3", "-Xptxas", "-v", "-w"], verbose=True)
    except Exception as ex:  # noqa: BLE001
        _T1_EXT["err"] = f"{type(ex).__name__}: {str(ex)[:160]}"
        _t1_note(f"patched extension unavailable ({_T1_EXT['err']}): trig_early / trig_late / rows2 / combo dropped")
    _T1_EXT["secs"] = _t1_time.perf_counter() - t0
    return _T1_EXT["ext"]


# ===================================================================================================== the variants' expert path
def _t1_cfg(m, V):
    """smallm_cfg(m) with the variant's nz1 at M = 1 (the config row of M = 1 only; other M untouched)."""
    cfg = dict(smallm_cfg(m))
    if V.nz1 is not None and m == 1:
        cfg["nz1"] = V.nz1
    return cfg


def _t1_rows(m, V):
    """token rows of dynamic smem: v5f always allocates 16, the kernel stages 8 for a chunk of <= 8 tokens (M <= 8)."""
    return 8 if (V.rows2 and m <= 8) else 16


def _t1_route(topk_ids, start, state, V):
    """ROUTE_SIDE: smallm_route right after the routing (it needs only topk_ids and local_start): returns the local ids; the route table lands in sm['tab']."""
    sm = state["smallm"]
    ids = topk_ids.to(torch.int32).contiguous()
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    sm["ext"].smallm_route(ids, local_ids, sm["tab"], start, LOCAL_EXPERTS)
    return (local_ids,)


def _t1_smallm_experts(x, topk_ids, sm, start, count, cfg, V, pre, ext2):
    """smallm_experts of v5f with the variant's switches: nz1 (cfg), rows / trig_late (the patched extension), the route already done on the side stream (pre)."""
    m = x.shape[0]
    d, ext = sm["dims"], sm["ext"]
    assert m <= SM_MAX_M and count == SM_NEXP
    if pre is None:
        ids = topk_ids.to(torch.int32).contiguous()
        local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)
        ext.smallm_route(ids, local_ids, sm["tab"], start, count)
    else:
        local_ids = pre[0]
    inter = torch.empty((m * TOP_K, d["inter"]), dtype=torch.bfloat16, device=x.device)
    y = torch.empty((m * TOP_K, d["hid"]), dtype=torch.bfloat16, device=x.device)
    nz = min(SM_NEXP, m * TOP_K)
    for k_dim, groups, ks, nw in ((d["k1"], d["g1"], cfg["ks1"], cfg["nw1"]), (d["k2"], d["g2"], cfg["ks2"], cfg["nw2"])):
        assert k_dim % ks == 0 and (k_dim // ks) % 256 == 0 and k_dim // ks <= 2048 and groups % nw == 0, (k_dim, groups, ks, nw)
    xc = x.contiguous()
    if V.ext2:
        rows, tl_ = _t1_rows(m, V), int(V.trig_late)
        ext2.smallm_gemm_t1(sm["w1"], sm["s1"], sm["gs1"], sm["lut"], xc, sm["tab"], sm["part"], sm["cnt"][0], inter,
                            d["g1"], d["k1"] // 64, d["k1"], cfg["ks1"], min(nz, cfg["nz1"]), d["inter"], 3, cfg["nw1"], rows, tl_, True)
        ext2.smallm_gemm_t1(sm["w2"], sm["s2"], sm["gs2"], sm["lut"], inter, sm["tab"], sm["part"], sm["cnt"][1], y,
                            d["g2"], d["k2"] // 64, d["k2"], cfg["ks2"], min(nz, cfg["nz2"]), d["hid"], 0, cfg["nw2"], rows, tl_, False)
    else:
        ext.smallm_gemm(sm["w1"], sm["s1"], sm["gs1"], sm["lut"], xc, sm["tab"], sm["part"], sm["cnt"][0], inter,
                        d["g1"], d["k1"] // 64, d["k1"], cfg["ks1"], min(nz, cfg["nz1"]), d["inter"], 3, cfg["nw1"], True)
        ext.smallm_gemm(sm["w2"], sm["s2"], sm["gs2"], sm["lut"], inter, sm["tab"], sm["part"], sm["cnt"][1], y,
                        d["g2"], d["k2"] // 64, d["k2"], cfg["ks2"], min(nz, cfg["nz2"]), d["hid"], 0, cfg["nw2"], False)
    return y.view(m, TOP_K, d["hid"]), local_ids


def _t1_experts(x, topk_weights, topk_ids, local_start, state, local_ids=None, V=None, pre=None):
    """_experts of v5f; the small-M path takes the variant's switches (base: the unchanged _experts)."""
    m = x.shape[0]
    if V is not None and V.custom and local_ids is None and _smallm_live(m, state):
        y, lids = _t1_smallm_experts(x.contiguous(), topk_ids, state["smallm"], local_start, LOCAL_EXPERTS, _t1_cfg(m, V), V, pre, _T1_EXT["ext"] if V.ext2 else None)
        return epilogue_triton(y, topk_weights.to(torch.float32).contiguous(), lids, state["post_expert_w"], block_h=_marlin_knobs(m)[2])
    return _experts(x, topk_weights, topk_ids, local_start, state, local_ids=local_ids)


# ===================================================================================================== the study
def _t1_case(shape, seed):
    name, kw = next(s for s in _T1_SHAPES if s[0] == shape)
    return _tr_generate_input(batch=kw["batch"], ctx=_T1_CTX, layer=kw["layer"], seed=seed, q_len=kw["q_len"], spread=kw["spread"], dist="soft", page=1,
                              local_start=0, device="cuda")


def _t1_n_act(ids, start):
    rel = (ids.long() - start) % N_EXPERTS
    return int(torch.unique(rel[rel < LOCAL_EXPERTS]).numel())


class _T1Env:
    """The GPU side: inputs (the judge gives each input the 32-expert range its routing sends the most pairs to), graph timing after a read-only flush, bitwise compare."""

    def __init__(self, state):
        self.state = state
        self.flush = torch.ones(_T1_FLUSH_MB * 1024 * 1024 // 4, device="cuda", dtype=torch.float32)
        self.sink = torch.zeros((), device="cuda", dtype=torch.float32)

    def available(self):
        return HAVE_SGLANG and torch.cuda.is_available() and self.state.get("smallm") is not None

    def clock(self):
        return _t1_time.perf_counter()

    def banner(self):
        return f"gpu={torch.cuda.get_device_name()} torch={torch.__version__} cap={torch.cuda.get_device_capability()}"

    def case(self, shape, seed):
        data = _t1_case(shape, seed)
        out = kernel_fn(data, self.state)                      # the board's kernel_fn: routing -> the busiest local range
        torch.cuda.synchronize()
        ids = out[2]
        pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
        start = LOCAL_EXPERTS * int(pairs.argmax())
        data["local_start"].fill_(start)
        ref = kernel_fn(data, self.state)
        torch.cuda.synchronize()
        return dict(data=data, ref=tuple(t.clone() for t in ref), E=_t1_n_act(ids, start))

    def run(self, ctx, V):
        return _t1_kernel(ctx["data"], self.state, V)

    def same(self, ctx, out):
        return all(torch.equal(a, b) for a, b in zip(out, ctx["ref"]))

    def graph_time(self, fn, reps=None):
        reps = reps or _T1_REPS
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
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
        del g
        return _t1_med(ts)

    def overhead(self):
        """The judge subtracts a graph replay's fixed cost (about 4 us) from both sides: measured here as the replay of a graph with one tiny kernel."""
        x = torch.zeros(1, device="cuda")
        return self.graph_time(lambda: x.add_(1.0), reps=15)

    def release(self):
        del self.flush, self.sink
        torch.cuda.empty_cache()


def _t1_study(state, env=None, t_start=None):
    """Returns the result dict {'ovh', 'E': {(shape, seed): n}, 'l': {(shape, seed, variant): us}, 'bw': {...}, 'combo_of': names, 'skipped': [...]} and prints the report."""
    env = env or _T1Env(state)
    if not env.available():
        print("MEAS-T1 skipped: needs CUDA, SGLang and the small-M path")
        return None
    t0 = env.clock()
    deadline = (t_start if t_start is not None else t0) + _T1_BUDGET_S      # (the clocks of t_start and env are the same: perf_counter)
    res = {"l": {}, "bw": {}, "E": {}, "skipped": [], "combo_of": ()}
    try:
        res["ovh"] = env.overhead()
        ext_ok = _t1_build_ext() is not None
        singles = [v for v in _T1_SINGLES if ext_ok or not v.ext2]
        for v in _T1_SINGLES:
            if v not in singles:
                res["skipped"].append(f"{v.name}:no-ext")

        def time_variant(shape, seed, ctx, V, tag=None):
            name = tag or V.name
            if env.clock() > deadline:
                res["skipped"].append(f"{name}@{shape}s{seed}")
                return
            try:
                out = env.run(ctx, V)
                torch.cuda.synchronize()
                res["bw"][(shape, seed, name)] = "OK" if env.same(ctx, out) else "DIFF"
                res["l"][(shape, seed, name)] = env.graph_time(lambda: env.run(ctx, V))
            except Exception as ex:  # noqa: BLE001
                res["skipped"].append(f"{name}@{shape}s{seed}:{type(ex).__name__}: {str(ex)[:60]}")

        base = singles[0]
        for shape, _kw in _T1_SHAPES:                     # pass 1: every single variant (base twice: the second is the noise floor)
            for seed in _T1_SEEDS:
                if env.clock() > deadline:
                    res["skipped"].append(f"{shape}s{seed}")
                    continue
                ctx = env.case(shape, seed)
                res["E"][(shape, seed)] = ctx["E"]
                for V in singles:
                    time_variant(shape, seed, ctx, V)
                time_variant(shape, seed, ctx, base, tag="base2")
                del ctx
        winners = _t1_winners(res)
        combo, of = _t1_combo(winners[1])                     # pass 2: the combo of the singles that gain >= _T1_COMBO_US
        if combo.ext2 and not ext_ok:
            res["skipped"].append("combo:no-ext")
        else:
            res["combo_of"] = of
            for shape, _kw in _T1_SHAPES:
                for seed in _T1_SEEDS:
                    if (shape, seed, "base") not in res["l"]:
                        continue
                    ctx = env.case(shape, seed)
                    time_variant(shape, seed, ctx, combo)
                    del ctx
    finally:
        env.release()
    out = _t1_render(res, env.banner(), t0, env.clock())
    print(out, flush=True)
    return res


def _t1_gain(res, shape, name):
    """median over seeds of (base - variant) us at one shape, or None"""
    d = [res["l"][(shape, s, "base")] - res["l"][(shape, s, name)] for s in _T1_SEEDS if (shape, s, "base") in res["l"] and (shape, s, name) in res["l"]]
    return _t1_med(d)


def _t1_winners(res, min_us=_T1_COMBO_US):
    """(names that gain >= min_us at T = 1, weighted over sliding / full, and lose at most _T1_REG_PCT percent at decode 16), (the same for the combo's input list)"""
    out = []
    for V in list(_T1_SINGLES) + [_T1V("combo")]:
        if V.name == "base":
            continue
        gs, gf = _t1_gain(res, "dec1s", V.name), _t1_gain(res, "dec1f", V.name)
        if gs is None or gf is None:
            continue
        g = (_T1_WEIGHTS["dec1s"] * gs + _T1_WEIGHTS["dec1f"] * gf) / (_T1_WEIGHTS["dec1s"] + _T1_WEIGHTS["dec1f"])
        g16 = _t1_gain(res, "dec16s", V.name)
        b16 = _t1_med([res["l"][("dec16s", s, "base")] for s in _T1_SEEDS if ("dec16s", s, "base") in res["l"]])
        reg = 0.0 if (g16 is None or b16 is None) else max(0.0, -100.0 * g16 / b16)
        if g >= min_us and reg <= _T1_REG_PCT and res["bw"].get(("dec1s", _T1_SEEDS[0], V.name)) != "DIFF":
            out.append(V.name)
    return min_us, tuple(out)


def _t1_x(res, shape, name):
    xs = [res["l"][(shape, s, "base")] / res["l"][(shape, s, name)] for s in _T1_SEEDS if (shape, s, "base") in res["l"] and (shape, s, name) in res["l"]]
    return _t1_med(xs)


def _t1_render(res, banner, t0, t1):
    ovh = res.get("ovh") or 0.0
    names = [n for n in _T1_NAMES if any(k[2] == n for k in res["l"])] + (["base2"] if any(k[2] == "base2" for k in res["l"]) else [])
    sub = lambda us: None if us is None else us - ovh      # noqa: E731
    head = (f"MEAS-T1 {banner} ovh={ovh:.1f}us ctx={_T1_CTX} reps={_T1_REPS} seeds={'/'.join(map(str, _T1_SEEDS))} "
            f"E(dec1s)={'/'.join(str(res['E'].get(('dec1s', s), '-')) for s in _T1_SEEDS)} E(dec1f)={'/'.join(str(res['E'].get(('dec1f', s), '-')) for s in _T1_SEEDS)} "
            f"E(dec16s)={'/'.join(str(res['E'].get(('dec16s', s), '-')) for s in _T1_SEEDS)} ext2_build={_t1_secs(_T1_EXT['secs'])}")
    lines = [head]
    for n in names:                                         # S lines first: one per variant, the three shapes (layer us, ovh subtracted | x vs base | bitwise)
        cells = []
        for shape, _kw in _T1_SHAPES:
            ls = [res["l"][(shape, s, n)] for s in _T1_SEEDS if (shape, s, n) in res["l"]]
            m = _t1_med(ls)
            x = _t1_x(res, shape, n)
            cells.append(f"{shape}={'-' if m is None else f'{sub(m):.1f}'}us x{'-' if x is None else f'{x:.3f}'}")
        bw = [res["bw"].get((shape, s, n)) for shape, _kw in _T1_SHAPES for s in _T1_SEEDS if res["bw"].get((shape, s, n)) is not None]
        lines.append(f"S t1 {n:10s} | " + " ".join(cells) + f" | bw={'-' if not bw else ('OK' if all(b == 'OK' for b in bw) else 'DIFF')}")
    for n in names:                                         # B lines: per shape, the three seeds
        for shape, _kw in _T1_SHAPES:
            ls = [res["l"].get((shape, s, n)) for s in _T1_SEEDS]
            if all(l is None for l in ls):
                continue
            bw = "".join("o" if res["bw"].get((shape, s, n)) == "OK" else ("D" if res["bw"].get((shape, s, n)) == "DIFF" else "-") for s in _T1_SEEDS)
            lines.append(f"B t1 {shape} v={n} | us=" + "/".join('-' if l is None else f'{l - ovh:.1f}' for l in ls) + f" bw={bw}")
    # D line
    min_us, wins = _t1_winners(res, min_us=_T1_WIN_US)
    gains = []
    for n in [x for x in _T1_NAMES if x != "base"]:
        gs, gf, g16 = _t1_gain(res, "dec1s", n), _t1_gain(res, "dec1f", n), _t1_gain(res, "dec16s", n)
        if gs is not None and gf is not None:
            gains.append(f"{n} {gs:+.1f}/{gf:+.1f}/{'-' if g16 is None else f'{g16:+.1f}'}")
    nz = _t1_gain(res, "dec1s", "base2")
    lines.append(f"D t1 winners (>= {_T1_WIN_US:.0f}us weighted at T=1, T=16 regression <= {_T1_REG_PCT:.0f}%): {', '.join(wins) if wins else 'none'} | gains us dec1s/dec1f/dec16s: "
                 + "; ".join(gains) + f" | noise(base2-base dec1s)={'-' if nz is None else f'{-nz:+.2f}'}us | combo of {','.join(res.get('combo_of') or ()) or '-'}")
    for note in _T1_NOTES:
        lines.append("N " + note)
    lines.append(f"MEAS-T1 done {t1 - t0:.0f}s" + (f" skipped: {' '.join(res['skipped'])[:500]}" if res["skipped"] else ""))
    out = "\n".join(lines)
    if len(out.encode()) > _T1_MAX_BYTES:
        out = out[:_T1_MAX_BYTES - 40] + "\n...(report cut at budget)"
    return out


# ===================================================================================================== entry points
_t1_orig_prepare = prepare


def prepare(weights):  # noqa: F811
    global _T1_T0
    _T1_T0 = _t1_time.perf_counter()
    state = _t1_orig_prepare(weights)
    try:
        _t1_study(state, t_start=_T1_T0)
    except Exception as ex:  # noqa: BLE001
        print(f"MEAS-T1 failed: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
    return state
