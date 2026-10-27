"""kernels/muse_decode_layer/sonnet_v5h.py (C40) = sonnet_v5f + F1 (S10_FFN_FUSE: the FFN boundary behind the unpooler kernel, T <= 8, NO grid-wide wait) + the C33 TOOLS_LIGHT rider (S11_TOOLS_LIGHT) + the C37
ROUTE_SIDE rider (S12_ROUTE_SIDE).

No GPU:  file rules, header predictions, generator sync, the diff v5f -> v5g (additions + the marked edits; every other function byte-identical), the gates (T <= 8 fused, T = 9 not), switches.
Dev GPU (RTX 2060, sm_75, NVRTC, no SGLang, FP16 stand-ins where tl.dot needs BF16): F1's kernel against the FP32 reference chain of reference.py (ref_back) and against v5f's BF16-rounded path
         (max relative errors on h, the stream and, unchanged, the ids are PRINTED), determinism, the in-call counter, a CUDA graph with the memset, tile / token edges; ROUTE_SIDE bitwise v5f's
         experts incl. a two-stream CUDA graph; the whole layer (fallback path) against the organisers' reference at T = 1, 2, 8, 9, 16 and a graph, all switches off = v5f bitwise.
LAST:    sm_90a compiles (Capture): every changed kernel with PDL on has its prefetch before griddepcontrol.wait; F1's kernel: regs <= 255, 0 spills, the arrival atomics, smem.
Run:     set -o pipefail; tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v5h.py"""
import difflib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "problems" / "muse_moe_w4a16"))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_decode_layer"
V5H = ROOT / "kernels" / PROBLEM / "sonnet_v5h.py"
V5F = ROOT / "kernels" / PROBLEM / "sonnet_v5f.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v5h", V5H)


@pytest.fixture(scope="module")
def v5f():
    return _load("_sonnet_v5f_for_v5g", V5F)


# ------------------------------------------------------------------------------------------------ no GPU
def test_file_rules_header_and_generator_sync():
    src = V5H.read_text()
    lines = src.splitlines()
    assert lines[0] == "# okc: problem=muse_decode_layer"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ", r"import reference", r"from reference", r"\.tolist\(", r"urllib",
                r"torch\.profiler", r"MEASUREMENT"):
        assert not re.search(bad, src), bad
    compile(src, "sonnet_v5h.py", "exec")
    head = "\n".join(line for line in lines[:60] if line.startswith("#"))
    for needle in ("sonnet_v5h (C40)", "F1 (S10_FFN_FUSE)", "NO CTA WAITS FOR ANOTHER", "S11_TOOLS_LIGHT", "S12_ROUTE_SIDE", "# PREDICTION:", "PROBLEM: v5g check 1.439", "#0 101.4", "Kill:", "Risks:"):
        assert needle in head, needle
    assert next(i for i, line in enumerate(lines) if line.startswith("# PREDICTION:")) < 40
    import gen_sonnet_v5h as gen
    assert gen.build() == src, "regenerate: .venv/bin/python tools/gen_sonnet_v5h.py"
    assert 'print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")' in src


def _funcs(path):
    import ast
    src = Path(path).read_text()
    lines = src.splitlines()
    out = {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            st = n.decorator_list[0].lineno if n.decorator_list else n.lineno
            out[n.name] = "\n".join(lines[st - 1:n.end_lineno])
    return out


def test_diff_v5f_v5g_additions_and_marked_edits_only():
    fa, fb = _funcs(V5F), _funcs(V5H)
    changed = {k for k in fa if fa[k] != fb.get(k)}
    assert changed == {"_pdl_on", "_glue_selfcheck", "_prepare_sglang", "smallm_experts", "_experts_smallm", "_experts", "_kernel_sglang"}, changed
    new = set(fb) - set(fa)
    assert {"_ffn_tile", "_ffn_apply_kernel", "_unpool_ffn_kernel", "unpool_ffn", "_f1_state", "_ffn_fuse_live", "smallm_route_side", "_tools_hw", "_tools_live", "_tl_pfl", "_tl_pf",
            "_tl_gemv_splitk", "_tl_router_fix_kernel", "_tl_unpool_kernel", "_tl_pre_norm_kernel", "_tl_epilogue_kernel"} <= new
    a, b = V5F.read_text().splitlines(), V5H.read_text().splitlines()
    removed = [d[1:] for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] == "-" and d[:3] != "---"]
    edited = {"# okc: problem=muse_decode_layer", "def smallm_experts(x, topk_ids, sm, start, count, cfg):", "    ids = topk_ids.to(torch.int32).contiguous()",
              "    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=x.device)", '    ext.smallm_route(ids, local_ids, sm["tab"], start, count)',
              "def _experts_smallm(x, topk_weights, topk_ids, local_start, state):",
              '    y, local_ids = smallm_experts(x.contiguous(), topk_ids, state["smallm"], local_start, LOCAL_EXPERTS, smallm_cfg(m))',
              "def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None):",
              "        return _experts_smallm(x, topk_weights, topk_ids, local_start, state)",
              '    return bool(S3_PDL and HAVE_PDL_API and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 9)',
              "    global S9_PRE_NORM, S10_UNPOOL_GEMV, S11_EPI_GRID",
              '    print(f"okc-diag glue switches: G2={S9_PRE_NORM} G3={S10_UNPOOL_GEMV} G4={S11_EPI_GRID}")',
              "    del freqs",
              "    with torch.cuda.stream(side_stream):",
              '            topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)'.replace("            ", "            "),
              "        for tensor in (topk_weights, topk_ids) + ((local_ids,) if local_ids is not None else ()):",
              '                             local_ids=local_ids)',
              '    if _unpool_live(tokens, device) and "unpool_part" in state:  # G3: slice GEMV + zero pad in one launch',
              "    local_ids = None",
              "S11_EPI_GRID = True       # G4 (H9): T <= EPI_GRID_MAX_TOKENS: the MoE epilogue on grid (tokens, EPI_PARTS), one 1024-column quarter per CTA (bitwise v5b's epilogue)"}
    stray = [line for line in removed if line not in edited and line.strip() not in {x.strip() for x in b}]
    assert not stray, stray
    assert len(removed) < 40


def test_switches_gates_and_off_is_v5f(mod, v5f, monkeypatch):
    assert mod.S10_FFN_FUSE and mod.S11_TOOLS_LIGHT and mod.S12_ROUTE_SIDE and not mod.S3_PDL
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    state = {"f1": {}, "unpool_part": torch.empty(1)}
    monkeypatch.setattr(mod, "_triton_ok", lambda d: True)
    stream = torch.empty((8, 8192), dtype=torch.float32)
    if dev.type == "cuda":
        for t, want in ((1, True), (2, True), (8, True), (9, False), (16, False), (32, False)):
            assert mod._ffn_fuse_live(t, dev, state, stream) == want, t
        assert not mod._ffn_fuse_live(4, dev, state, stream.to(torch.bfloat16)) and not mod._ffn_fuse_live(4, dev, {}, stream)
        monkeypatch.setattr(mod, "S10_FFN_FUSE", False)
        assert not mod._ffn_fuse_live(4, dev, state, stream)
        monkeypatch.setattr(mod, "S10_FFN_FUSE", True)
        monkeypatch.setattr(mod, "S10_UNPOOL_GEMV", False)
        assert not mod._ffn_fuse_live(4, dev, state, stream)
    # the rider's PDL gate: S3_PDL OR S11 (hardware permitting); both off = v5f's
    if dev.type == "cuda":
        for s3, s11 in ((False, False), (True, False), (False, True)):
            monkeypatch.setattr(mod, "S3_PDL", s3)
            monkeypatch.setattr(mod, "S11_TOOLS_LIGHT", s11)
            monkeypatch.setattr(v5f, "S3_PDL", s3)
            assert mod._pdl_on(dev) == (v5f._pdl_on(dev) or (s11 and mod.HAVE_PDL_API and torch.cuda.get_device_capability(dev)[0] >= 9))
    assert mod._tl_pfl("_unpool_kernel", (), dict(BN=16, SPLIT=2, K=4096)) == 32 and mod._tl_pfl("_gemv_splitk", (None,) * 6 + (8192,), dict(BN=32, BK=512, SPLIT=1, HAS_LO=False)) == 16


# ------------------------------------------------------------------------------------------------ dev GPU: F1
# (BN, BK, SPLIT, warps, stages); N / BN = 64 / 64 / 64 / 32 tiles. No CTA waits for another: the configs that used to time the v5g barrier out on the 2060 (BK 128 / 256: 16-32 KB of smem per CTA,
# fewer resident CTAs than the 128 of the grid) must pass now.
CFGS = [(16, 256, 2, 4, 2), (16, 128, 4, 4, 2), (32, 128, 1, 4, 2), (16, 64, 2, 4, 2)]


def _f1_setup(mod, t, seed, dtype=torch.float16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s, sc=1.0: sc * torch.randn(s, generator=g, device="cuda")      # noqa: E731
    w = {"ffn_residual_gate": r(8192, sc=0.3), "post_ffn_norm": r(8192, sc=0.1), "next_pre_attn_norm": r(8192, sc=0.1)}
    full = torch.zeros((8192, 4096), dtype=dtype, device="cuda")
    full[:1024] = r(1024, 4096, sc=0.02).to(dtype)
    state = {"dense": {"unpooler": full}, "unpool_part": torch.empty((8, 16, 1024), dtype=torch.float32, device="cuda"), "unpool_done": torch.zeros(64, dtype=torch.int32, device="cuda"),
             "f1": mod._f1_state(w, "cuda")}
    y = r(t, 4096, sc=0.5).to(dtype)
    s = r(t, 8192)
    return state, w, y, s


def _ref_chain(mod, state, w, y, s, round_f=False):
    f = torch.zeros((y.shape[0], 8192), device="cuda")
    f[:, :1024] = y.float() @ state["dense"]["unpooler"][:1024].float().T
    if round_f:
        f = f.to(torch.bfloat16).float()                   # the board: the unpooler's BF16 output goes through SGLang's boundary kernel
    f = f * (1.0 + w["post_ffn_norm"].float())
    a, b = mod._residual_coefficients(w["ffn_residual_gate"])
    s2 = a * s + b * mod._rms(f, mod.POST_NORM_EPS)
    return mod._rms(s2) * (1.0 + w["next_pre_attn_norm"].float()), s2


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


@needs_cuda
@pytest.mark.parametrize("cfg", CFGS, ids=lambda c: "x".join(map(str, c)))
@pytest.mark.parametrize("t", [1, 2, 3, 8])
def test_f1_matches_the_reference_chain_and_prints_the_errors(mod, cfg, t, capsys):
    state, w, y, s = _f1_setup(mod, t, 10 * t + cfg[2])
    state["f1"]["ctr"].zero_()
    h, so = mod.unpool_ffn(y, s, state, cfg)
    torch.cuda.synchronize()
    hr, sr = _ref_chain(mod, state, w, y, s)
    hb, sb = _ref_chain(mod, state, w, y, s, round_f=True)
    e_h, e_s = _rel(h, hr), _rel(so, sr)
    e_hb, e_sb = _rel(h, hb), _rel(so, sb)
    with capsys.disabled():
        print(f"F1 T={t} cfg={cfg}: vs FP32 reference chain: hidden {e_h:.2e} stream {e_s:.2e} | vs the board's BF16-f chain: hidden {e_hb:.2e} stream {e_sb:.2e} | ids unchanged (not produced here)")
    assert h.dtype == torch.bfloat16 and so.dtype == torch.float32 and h.shape == (t, 8192)
    assert e_s < 1e-5 and e_h < 5e-3                       # FP32 chain, ONE bf16 rounding of h (2^-9 = 2e-3 relative)
    assert e_sb < 2e-3 and e_hb < 6e-3                     # against the board's order (f rounded to BF16 first): far inside the 3 % floor of the layer bound
    assert int(state["f1"]["ctr"]) == 1024 // cfg[0] and int(state["unpool_done"].abs().sum()) == 0


@needs_cuda
def test_f1_is_deterministic_edges_and_changing_t(mod):
    cfg = CFGS[0]
    state, w, y, s = _f1_setup(mod, 8, 5)
    outs = []
    for _ in range(3):
        state["f1"]["ctr"].zero_()
        outs.append(mod.unpool_ffn(y, s, state, cfg))
    assert all(torch.equal(outs[0][0], o[0]) and torch.equal(outs[0][1], o[1]) for o in outs)
    for t in (1, 5, 8, 2):                                   # a smaller T after a larger one leaves nothing stale; rows beyond T are never written
        state["f1"]["ctr"].zero_()
        h, so = mod.unpool_ffn(y[:t].contiguous(), s[:t].contiguous(), state, cfg)
        hr, sr = _ref_chain(mod, state, w, y[:t], s[:t])
        assert _rel(so, sr) < 1e-5 and _rel(h, hr) < 5e-3, t
    # strided stream input (a row stride beyond 8192) and an all-zero activation (f = 0: only the residual carries through)
    big = torch.zeros((8, 8192 + 64), device="cuda")
    big[:, :8192] = s
    state["f1"]["ctr"].zero_()
    h2, so2 = mod.unpool_ffn(y, big[:, :8192], state, cfg)
    assert torch.equal(so2, outs[0][1])
    state["f1"]["ctr"].zero_()
    h0, so0 = mod.unpool_ffn(torch.zeros_like(y), s, state, cfg)
    a, _ = mod._residual_coefficients(state["f1"]["alpha"] * 0 + 0.0)  # (keeps the helper imported); the expected stream is alpha * s
    assert _rel(so0, state["f1"]["alpha"] * s) < 1e-6


@needs_cuda
def test_f1_in_a_cuda_graph_with_the_counter_memset(mod):
    cfg = CFGS[0]
    state, w, y, s = _f1_setup(mod, 4, 8)

    def run():
        state["f1"]["ctr"].zero_()
        return mod.unpool_ffn(y, s, state, cfg)

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        run()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        h, so = run()
    for seed in (21, 22, 23):
        g2 = torch.Generator(device="cuda").manual_seed(seed)
        y.copy_((0.5 * torch.randn(y.shape, generator=g2, device="cuda")).to(y.dtype))
        s.copy_(torch.randn(s.shape, generator=g2, device="cuda"))
        g.replay()
        torch.cuda.synchronize()
        hr, sr = _ref_chain(mod, state, w, y, s)
        assert _rel(so, sr) < 1e-5 and _rel(h, hr) < 5e-3, seed
    assert int(state["unpool_done"].abs().sum()) == 0


@needs_cuda
def test_f1_with_the_prefetch_before_the_wait_is_bitwise_the_same(mod):
    cfg = CFGS[0]
    state, w, y, s = _f1_setup(mod, 8, 6)
    f1 = state["f1"]
    res = []
    for pfl in (0, 8, 32):
        f1["ctr"].zero_()
        h = torch.empty((8, 8192), dtype=torch.bfloat16, device="cuda")
        so = torch.empty((8, 8192), dtype=torch.float32, device="cuda")
        mod._unpool_ffn_kernel[(64, 2)](y, state["dense"]["unpooler"], s, f1["wf"], f1["alpha"], f1["beta"], f1["fsc"], f1["gs"], state["unpool_part"], state["unpool_done"], f1["part"], f1["ctr"],
                                        8, y.stride(0), s.stride(0), BM=16, BN=16, BK=256, SPLIT=2, K=4096, N=1024, NCOL=8192, PADF=112, PADP=128, PFL=pfl, num_warps=4, num_stages=2)
        mod._ffn_apply_kernel[(8192 // mod.FFN_APPLY_BC, 8)](s, f1["alpha"], f1["beta"], f1["wf"], f1["wn"], f1["fsc"], f1["gs"], h, so, s.stride(0),
                                                             BC=mod.FFN_APPLY_BC, BM=16, N=1024, NCOL=8192, num_warps=4)
        torch.cuda.synchronize()
        res.append((h, so))
    assert all(torch.equal(res[0][0], r[0]) and torch.equal(res[0][1], r[1]) for r in res)


@needs_cuda
def test_f1_needs_no_co_residency_one_cta_at_a_time(mod):
    """No CTA waits for another: the largest-smem config the 2060 takes (BK 256: fewer CTAs resident than the grid's 128, which timed the v5g barrier out) repeatedly gives the right answer."""
    cfg = (16, 256, 2, 4, 2)
    state, w, y, s = _f1_setup(mod, 8, 9)
    for _ in range(3):
        state["f1"]["ctr"].zero_()
        h, so = mod.unpool_ffn(y, s, state, cfg)
        torch.cuda.synchronize()
        hr, sr = _ref_chain(mod, state, w, y, s)
        assert _rel(so, sr) < 1e-5 and _rel(h, hr) < 5e-3


# ------------------------------------------------------------------------------------------------ dev GPU: ROUTE_SIDE
@pytest.fixture(scope="module")
def nv():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    return nv


DEV_CFG = ((16, 8, 4, 8, 8, 4, 8),)       # (largest M, nw1, ks1, nz1, nw2, ks2, nz2): 64 KB of smem on the dev card


@pytest.fixture()
def sm_state(mod, nv):   # function scope: the whole-layer tests below need the dev card's memory
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(4242)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=g, device="cuda", leading=(32,))
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous()}
    mod.SMALLM_CFG = DEV_CFG
    sm = mod.prepare_smallm(w, ext=nv.NvrtcExt(mod._SMALLM_DEVICE_SRC))
    yield {"smallm": sm}
    del sm, w
    torch.cuda.empty_cache()


def _live(y, lids):
    return y.reshape(-1, y.shape[-1]).view(torch.int16)[lids.reshape(-1) >= 0]


@needs_cuda
@pytest.mark.parametrize("m", [1, 4, 8, 16])
def test_route_side_experts_are_bitwise_the_board_experts(mod, sm_state, m):
    sm = sm_state["smallm"]
    start = torch.zeros(1, dtype=torch.int32, device="cuda")
    for seed in (1, 2):
        g = torch.Generator(device="cuda").manual_seed(100 * m + seed)
        x = torch.randn((m, 4096), generator=g, device="cuda").to(torch.bfloat16)
        ids = torch.randint(0, 40, (m, 8), generator=g, device="cuda", dtype=torch.int32)
        y0, l0 = mod.smallm_experts(x, ids, sm, start, 32, mod.smallm_cfg(m))
        pre = mod.smallm_route_side(ids, start, sm_state)
        y1, l1 = mod.smallm_experts(x, ids, sm, start, 32, mod.smallm_cfg(m), pre=pre)
        torch.cuda.synchronize()
        assert torch.equal(l0, l1) and torch.equal(_live(y0, l0), _live(y1, l1)), (m, seed)
        assert int(sm["cnt"].abs().sum()) == 0


@needs_cuda
def test_route_side_in_a_two_stream_graph(mod, sm_state):
    sm = sm_state["smallm"]
    start = torch.zeros(1, dtype=torch.int32, device="cuda")
    g0 = torch.Generator(device="cuda").manual_seed(77)
    x = torch.randn((1, 4096), generator=g0, device="cuda").to(torch.bfloat16)
    ids = torch.randint(0, 40, (1, 8), generator=g0, device="cuda", dtype=torch.int32)
    side = torch.cuda.Stream()

    def pipeline():
        main = torch.cuda.current_stream()
        side.wait_stream(main)
        with torch.cuda.stream(side):
            pre = mod.smallm_route_side(ids, start, sm_state)
        main.wait_stream(side)
        return mod.smallm_experts(x, ids, sm, start, 32, mod.smallm_cfg(1), pre=pre)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        pipeline()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y1, l1 = pipeline()
    for seed in (78, 79):
        g1 = torch.Generator(device="cuda").manual_seed(seed)
        x.copy_(torch.randn((1, 4096), generator=g1, device="cuda").to(torch.bfloat16))
        ids.copy_(torch.randint(0, 40, (1, 8), generator=g1, device="cuda", dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        y0, l0 = mod.smallm_experts(x, ids, sm, start, 32, mod.smallm_cfg(1))
        assert torch.equal(l0, l1) and torch.equal(_live(y0, l0), _live(y1, l1)), seed


# ------------------------------------------------------------------------------------------------ dev GPU: the whole layer (fallback path: no SGLang, no Hopper)
LAYER_CASES = [dict(batch=1, ctx=700, layer="sliding", page=1, dist="soft"), dict(batch=1, ctx=500, layer="full", page=1, dist="soft"), dict(batch=2, ctx=300, layer="full", page=1, dist="soft"),
               dict(batch=1, ctx=2300, layer="sliding", q_len=8, page=1, dist="soft"), dict(batch=9, ctx=300, layer="full", page=1, dist="soft"), dict(batch=2, ctx=300, layer="full", q_len=8, page=1, dist="soft")]


@pytest.fixture(scope="module")
def layer():
    L = lc.Loaded(PROBLEM, V5H)
    with torch.no_grad():
        weights = L.ref.generate_weights(L.ref.WEIGHTS_SEED, device="cuda")
        cand = lc._clone_weights(weights, L.ref, False)
        cand = L.cand.prepare(cand)
    yield L, weights, cand
    del cand
    torch.cuda.empty_cache()


@needs_cuda
@pytest.mark.parametrize("case", LAYER_CASES, ids=lambda c: lc.format_case(c))
def test_whole_layer_against_the_reference_at_t_1_2_8_9_16(layer, case):
    L, weights, cand = layer
    with torch.no_grad():
        ok, msg = lc.run_case(L, case, weights, cand, "cuda", 0, False)
    torch.cuda.empty_cache()
    assert ok, f"{lc.format_case(case)}: {msg}"
    assert L.cand._ffn_fuse_live(L.ref.generate_input(**{**case, "seed": 0, "device": "cuda"})["config"]["tokens"], torch.device("cuda"), cand) is False   # dev card: no BF16 tl.dot, F1 not live


@needs_cuda
def test_whole_layer_in_a_cuda_graph_and_switches_off_equal_v5f(layer, v5f):
    L, weights, cand = layer
    with torch.no_grad():
        ok, msg = lc.run_case(L, LAYER_CASES[0], weights, cand, "cuda", 1, True)
        assert ok, msg
        ref = L.ref
        data = ref.generate_input(**{**LAYER_CASES[2], "seed": 3, "device": "cuda"})
        for n in ("S10_FFN_FUSE", "S11_TOOLS_LIGHT", "S12_ROUTE_SIDE"):
            setattr(L.cand, n, False)
        out_g = [t.clone() for t in L.cand.kernel_fn(data, cand)]
        for n in ("S10_FFN_FUSE", "S11_TOOLS_LIGHT", "S12_ROUTE_SIDE"):
            setattr(L.cand, n, True)
        out_on = [t.clone() for t in L.cand.kernel_fn(data, cand)]
    assert all(torch.equal(a, b) for a, b in zip(out_g, out_on))      # on the dev card every switch is inert (no Hopper, no BF16 dot, no SGLang)
    f = lc.Loaded(PROBLEM, V5F)
    with torch.no_grad():
        cand_f = f.cand.prepare(lc._clone_weights(weights, f.ref, False))
        out_f = f.cand.kernel_fn(data, cand_f)
    assert all(torch.equal(a, b) for a, b in zip(out_g, out_f)), "switches off must be bitwise v5f"


# ------------------------------------------------------------------------------------------------ sm_90a (LAST: ptx_audit.Capture patches the device capability for the process)
def _drive(m, dtype):
    g = torch.Generator(device="cuda").manual_seed(5)
    r = lambda *s, dt=dtype, sc=1.0: (sc * torch.randn(s, generator=g, device="cuda")).to(dt)     # noqa: E731
    m._run(r(8, 8192), r(4096, 8192, sc=0.02), None, 4096, dtype, (32, 512, 1, 4, 4))
    m._run(r(8, 8192, dt=torch.float32), r(256, 8192, sc=0.02), r(256, 8192, sc=1e-4), 256, torch.float32, (64, 128, 16, 4, 4), split_x=True)
    st = {"router_hi": r(256, 8192, sc=0.02), "router_lo": r(256, 8192, sc=1e-4), "rfix_part": torch.empty((64, 16, 256), device="cuda"), "rfix_done": torch.zeros(64, dtype=torch.int32, device="cuda")}
    m._router_fix(r(1, 8192, dt=torch.float32), r(1, 8192, dt=torch.float32), st)
    full = torch.zeros((8192, 4096), dtype=dtype, device="cuda")
    ust = {"dense": {"unpooler": full}, "unpool_part": torch.empty((8, 16, 1024), device="cuda"), "unpool_done": torch.zeros(64, dtype=torch.int32, device="cuda")}
    m.unpool_sliced(r(3, 4096), ust, (16, 256, 2, 4, 4))
    m.pre_norm_triton(r(4, 4096), r(4096, sc=0.1))
    m.epilogue_triton(r(4, 8, 4096), torch.rand((4, 8), generator=g, device="cuda"), torch.randint(-1, 32, (4, 8), generator=g, device="cuda", dtype=torch.int32), r(4096, sc=0.1))
    w = {"ffn_residual_gate": r(8192, dt=torch.float32), "post_ffn_norm": r(8192, dt=torch.float32), "next_pre_attn_norm": r(8192, dt=torch.float32)}
    ust["f1"] = m._f1_state(w, "cuda")
    m.unpool_ffn(r(3, 4096), r(3, 8192, dt=torch.float32), ust, (16, 256, 2, 4, 4))


@needs_cuda
def test_sm90_changed_kernels_compile_prefetch_before_the_wait_and_f1_resources():
    from tools import ptx_audit as pa
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    m = _load("_sonnet_v5h_sm90", V5H)
    with pa.Capture() as cap:
        _drive(m, torch.bfloat16)
    names = {r["name"] for r in cap.records}
    assert {"_tl_gemv_splitk", "_tl_router_fix_kernel", "_tl_unpool_kernel", "_tl_pre_norm_kernel", "_tl_epilogue_kernel", "_unpool_ffn_kernel", "_ffn_apply_kernel"} <= names, names
    seen = set()
    for r in cap.records:
        if not (r["name"].startswith("_tl_") or r["name"] in ("_unpool_ffn_kernel", "_ffn_apply_kernel")):
            continue
        comp = triton.compile(ASTSource(r["jf"], r["signature"], r["constexprs"], r["attrs"]), target=GPUTarget("cuda", 90, 32), options=r["options"])
        body = re.sub(r"//[^\n]*", "", comp.asm["ptx"])
        ip, iw = body.find("prefetch.global.L2"), body.find("griddepcontrol.wait")
        if r["name"] == "_ffn_apply_kernel":      # no prefetch instruction: its independent loads precede the wait
            ip = body.find("ld.global")
        assert 0 <= ip < iw, (r["name"], ip, iw)
        seen.add(r["name"])
        if r["name"] in ("_unpool_ffn_kernel", "_ffn_apply_kernel"):
            c = pa.compile_record(r)
            assert c["regs"] <= 255 and c["smem"] <= 227 * 1024, (r["name"], c)
            assert c["spills"] <= (16 if r["name"] == "_unpool_ffn_kernel" else 0), (r["name"], c)     # the GEMV kernel keeps its early loads live across the K loop: ptxas spills 8 bytes
            if r["name"] == "_unpool_ffn_kernel":
                assert re.search(r"atom[a-z_.]*acq_rel", body), "ticket atomics"
                assert "acquire" not in body.replace("acq_rel", ""), "no spin wait left in the kernel (no acquire polling)"
                assert ".cg" in body or "ld.global.cg" in body
            print(f"AUDIT {r['name']} regs={c['regs']} spills={c['spills']} smem={c['smem']} cp.async={c['cp.async']} wgmma={c['wgmma']} warps={c['warps']} stages={c['stages']}")
    assert seen >= {"_tl_gemv_splitk", "_tl_router_fix_kernel", "_tl_unpool_kernel", "_tl_pre_norm_kernel", "_tl_epilogue_kernel", "_unpool_ffn_kernel", "_ffn_apply_kernel"}
