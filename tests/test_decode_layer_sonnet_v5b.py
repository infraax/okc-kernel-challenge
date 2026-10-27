"""kernels/muse_decode_layer/sonnet_v5b.py (C22b): sonnet_v5 plus S6 (router fixup at T = 1), S7 (pooler ps1 config at 2..8 tokens), S8 (large-M experts for 33..256 tokens).

No GPU:  file rules, v5b differs from v5 only by the additions and the switches, the path table (monotone in T), the large-M CUDA source is the MoE problem's sonnet_v4 with only the
         route kernel changed, prepare()'s fallbacks (build fails / self-check fails -> Marlin), the dispatch of _experts, sm_80 / sm_90 compiles.
Dev GPU (RTX 2060, sm_75; FP16 stand-ins, small configs): the router fixup kernel against the hi + lo reference (deterministic, counters back at zero, graph capture), the pooler
         accumulation order, the whole layer against the organisers' reference (local_check small --graph, 13/13), and with LM_NVRTC=1 the large-M experts through NVRTC.
Run:     tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v5b.py"""
import difflib
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_decode_layer"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v5b.py"
V5 = ROOT / "kernels" / PROBLEM / "sonnet_v5.py"
MOE4 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v4.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v5b", KERNEL)


@pytest.fixture(scope="module")
def ref():
    return lc.Loaded(PROBLEM, KERNEL).ref


@pytest.fixture(scope="module")
def weights(ref):
    return ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")


# ------------------------------------------------------------------------------------------------ file rules / diff
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_decode_layer"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ", r"import reference", r"from reference",
                r"\.tolist\(", r"urllib", r"torch\.profiler", r"MEASUREMENT"):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert 'print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")' in src
    header = "\n".join(line for line in src.splitlines() if line.startswith("#"))
    for needle in ("#0 122.2 -> ~116.7", "#8 502.2 -> ~382", "#10 704.3 -> ~549", "#11 1004.2 -> ~849", "PROBLEM 1.271 -> ~1.39", "T          router"):
        assert needle in header, needle
    compile(src, "sonnet_v5b.py", "exec")


def test_v5b_is_v5_plus_the_c22b_additions():
    a, b = V5.read_text().splitlines(), KERNEL.read_text().splitlines()
    removed = [d[1:] for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] == "-" and d[:3] != "---"]
    allowed = {
        "# okc: problem=muse_decode_layer",      # replaced by the same line + the v5b header
        "S5_FULL_CAT = True      # C19 switch (shapes / config only); off = sonnet_v4's attention",
        '        return _run(x.contiguous(), state["dense"]["pooler"], None, D_EXPERT, torch.bfloat16,',
        '                    _CFG[("pooler", _bucket(x.shape[0]))])',
        '           "f2_done": torch.zeros(triton.cdiv(F2_MAX_TOKENS, F2_TOKEN_BLOCK), dtype=torch.int32, device=router.device)}',
        '        cfg = _CFG[("router", _bucket(x.shape[0]))]',
    }
    extra = [line for line in removed if line not in allowed]
    # the lines above are re-added unchanged (diff context moved), so only genuinely changed lines may remain
    added = set(b)
    assert not [line for line in extra if line not in added], extra
    src = KERNEL.read_text()
    for flag in ("S1_MOE_KNOBS = True", "S2_FUSED_ROUTING = False", "S3_PDL = False", "S5_FULL_CAT = True", "S6_ROUTER_FIX = True", "S7_POOLER_PS1 = True", "S8_LARGEM = True"):
        assert flag in src, flag


def test_largem_source_is_the_moe_problems_except_the_route_kernels_local_start(mod):
    moe = _load("_moe4_for_v5b", MOE4)
    a = moe._LARGEM_EXTRA_SRC.splitlines() + moe._LARGEM_HOST_SRC.splitlines()
    b = mod._LARGEM_EXTRA_SRC.splitlines() + mod._LARGEM_HOST_SRC.splitlines()
    diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    assert len(diff) <= 20, "\n".join(diff)
    joined = "\n".join(diff)
    assert "lo_ptr" in joined and "& 255" in joined and "data_ptr<int>(), (int)nloc" in joined
    assert mod.LARGEM_CFG[128] == moe.LARGEM_CFG[128] and mod.LARGEM_CFG[256] == moe.LARGEM_CFG[256]
    assert mod.LM_TC == moe.LM_TC and mod.LM_CTW == moe.LM_CTW and mod.LM_MAX_M == moe.LM_MAX_M and mod.LM_ZMAX == moe.LM_ZMAX
    assert mod._LARGEM_CPP_SRC.count("torch::Tensor lo, int64_t nloc") == 1


# ------------------------------------------------------------------------------------------------ the path table
def _moe_path(mod, t, state):
    if mod._smallm_live(t, state):
        return "smallm"
    if mod._largem_live(t, state):
        return "largem128" if mod.largem_bucket(t) == 128 else "largem256"
    return "marlin"


GOOD = {"smallm": object(), "largem": object(), "largem_ok": {128: True, 256: True}}


@pytest.mark.parametrize("t,want", [(1, "smallm"), (8, "smallm"), (16, "smallm"), (17, "marlin"), (31, "marlin"), (32, "marlin"), (33, "largem128"), (100, "largem128"),
                                    (128, "largem128"), (129, "largem256"), (200, "largem256"), (256, "largem256"), (257, "marlin"), (1000, "marlin")])
def test_moe_path_table_by_tokens_only(mod, t, want):
    assert _moe_path(mod, t, GOOD) == want


def test_moe_path_is_monotone_in_t(mod):
    order = ["smallm", "marlin", "largem128", "largem256", "marlin"]
    seq = [_moe_path(mod, t, GOOD) for t in range(1, 400)]
    idx, last = 0, 0
    for p in seq:
        while order[idx] != p:
            idx += 1
            assert idx < len(order), (p, seq)
        assert idx >= last
        last = idx


@pytest.mark.parametrize("state,want", [({"smallm": object(), "largem": None, "largem_ok": {}}, "marlin"), ({"smallm": object(), "largem": object(), "largem_ok": {128: False, 256: True}}, "marlin"),
                                        ({"smallm": None, "largem": object(), "largem_ok": {128: True, 256: True}}, "marlin"), ({}, "marlin")])
def test_largem_needs_a_passing_build_and_self_check(mod, state, want):
    assert _moe_path(mod, 128, state) == want


def test_switches_turn_the_paths_off(mod, monkeypatch):
    monkeypatch.setattr(mod, "S8_LARGEM", False)
    assert _moe_path(mod, 128, GOOD) == "marlin"
    monkeypatch.setattr(mod, "S7_POOLER_PS1", False)
    assert mod._pooler_cfg(8) == mod._CFG[("pooler", 8)]
    monkeypatch.setattr(mod, "S6_ROUTER_FIX", False)
    assert not mod._router_fix_live(1)


@pytest.mark.parametrize("t,want", [(1, (32, 512, 1, 4, 4)), (2, (32, 256, 1, 4, 4)), (8, (32, 256, 1, 4, 4)), (9, (32, 512, 1, 4, 4)), (16, (32, 512, 1, 4, 4)), (32, (32, 512, 1, 4, 3))])
def test_pooler_config_table(mod, t, want):
    assert mod._pooler_cfg(t) == want


def test_router_fix_only_at_one_token(mod):
    assert [t for t in range(1, 300) if mod._router_fix_live(t)] == [1]
    assert mod.ROUTER_FIX_CFG == (64, 128, 64, 4, 4) and 256 // 64 * 64 >= 256     # 256 CTAs


# ------------------------------------------------------------------------------------------------ dispatch of _experts / prepare() fallbacks (no GPU)
def test_experts_dispatch_goes_to_largem_in_range(mod, monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "_experts_largem", lambda *a, **k: calls.append("large") or "L")
    monkeypatch.setattr(mod, "_experts_smallm", lambda *a, **k: calls.append("small") or "S")
    monkeypatch.setattr(mod, "local_ids_triton", lambda *a, **k: calls.append("marlin") or (_ for _ in ()).throw(RuntimeError("stop")))
    for t in (1, 16, 17, 32, 33, 128, 256, 257):
        x = torch.zeros(t, 16)
        try:
            out = mod._experts(x, torch.ones(t, 8), torch.zeros(t, 8, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), {**GOOD, "bank": {"size_n": 8}})
        except RuntimeError:
            out = None
        want = "S" if t <= 16 else ("L" if 33 <= t <= 256 else None)
        assert out == want, (t, out)
        assert calls[-1] == {"S": "small", "L": "large", None: "marlin"}[want], (t, calls)


def _prepared(mod, monkeypatch, build, errs):
    monkeypatch.setattr(mod, "prepare_largem", lambda sm, **k: build())
    monkeypatch.setattr(mod, "_largem_selfcheck", lambda st: errs)
    state = {"smallm": object()}
    mod._prepare_largem_for_layer(state)
    return state


def test_prepare_keeps_largem_when_selfcheck_passes(mod, monkeypatch):
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: 5e-3, 256: 6e-3})
    assert st["largem"] == {"x": 1} and st["largem_ok"] == {128: True, 256: True}


def test_prepare_keeps_only_the_passing_bucket(mod, monkeypatch):
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: 0.3, 256: 6e-3})
    assert st["largem"] == {"x": 1} and st["largem_ok"] == {128: False, 256: True}
    assert _moe_path(mod, 100, {**st, "smallm": object()}) == "marlin" and _moe_path(mod, 200, {**st, "smallm": object()}) == "largem256"


@pytest.mark.parametrize("errs", [{128: 0.3, 256: 0.4}, {128: float("nan"), 256: float("inf")}])
def test_prepare_falls_back_to_marlin_when_selfcheck_fails(mod, monkeypatch, errs):
    # NaN compares False against the tolerance, so it fails too
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, errs)
    assert st["largem"] is None and not any(st["largem_ok"].values())


def test_prepare_falls_back_to_marlin_when_build_fails(mod, monkeypatch):
    def boom():
        raise RuntimeError("no nvcc")

    st = _prepared(mod, monkeypatch, boom, {})
    assert st["largem"] is None and st["largem_ok"] == {}


def test_prepare_without_smallm_or_with_switch_off(mod, monkeypatch):
    st = {"smallm": None}
    mod._prepare_largem_for_layer(st)
    assert st["largem"] is None
    monkeypatch.setattr(mod, "S8_LARGEM", False)
    st = {"smallm": object()}
    mod._prepare_largem_for_layer(st)
    assert st["largem"] is None


def test_no_scratch_allocation_in_the_timed_largem_path(mod):
    import inspect

    src = inspect.getsource(mod.largem_experts) + inspect.getsource(mod._experts_largem) + inspect.getsource(mod._router_fix)
    assert ".zero_(" not in src and "torch.zeros" not in src and "synchronize" not in src


# ------------------------------------------------------------------------------------------------ offline compiles for the judge's GPU
@needs_cuda
def test_router_fix_kernel_compiles_for_sm90_within_smem(mod):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    bn, bk, split, warps, stages = mod.ROUTER_FIX_CFG
    for sx in (True, False):
        sig = {"x_ptr": "*fp32" if sx else "*bf16", "hi_ptr": "*bf16", "lo_ptr": "*bf16", "out_ptr": "*fp32", "part_ptr": "*fp32", "done_ptr": "*i32", "M": "i32", "stride_x": "i32"}
        consts = {"BM": 16, "BN": bn, "BK": bk, "SPLIT": split, "SPLIT_X": sx, "K": 8192, "N": 256}
        c = triton.compile(ASTSource(fn=mod._router_fix_kernel, signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32), options={"num_warps": warps, "num_stages": stages})
        assert c.metadata.shared < 200 * 1024 and "atom" in c.asm["ptx"]


@needs_cuda
@pytest.mark.parametrize("sm", ["sm_80", "sm_90"])
def test_largem_device_source_assembles_for_judge_arch(mod, sm):
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    cubin, low = nv.sass_check(mod._SMALLM_DEVICE_SRC + mod._LARGEM_EXTRA_SRC, sm, names=nv.LARGE_NAMES)
    assert len(cubin) > 1000 and len(low) == len(nv.LARGE_NAMES)


# ------------------------------------------------------------------------------------------------ dev GPU
def _fp16_state(mod, weights):
    s = mod._split_router_weights(weights["router"], weights["router_bias"])
    s["router_hi"], s["router_lo"] = s["router_hi16"], s["router_lo16"]
    return s


@needs_cuda
@pytest.mark.parametrize("t,fp32_in", [(1, True), (1, False), (8, True), (16, False)])
def test_router_fix_matches_the_hi_lo_reference(mod, weights, t, fp32_in):
    st = _fp16_state(mod, weights)
    g = torch.Generator(device="cuda").manual_seed(3 + t)
    h32 = torch.randn((t, 8192), generator=g, device="cuda")
    x = h32.to(torch.bfloat16).half()
    cfg = (32, 64, 32, 4, 2)           # small pipeline: the 2060 has 64 KB of smem
    y = mod._router_fix(x, h32 if fp32_in else None, st, cfg)
    y2 = mod._router_fix(x, h32 if fp32_in else None, st, cfg)
    torch.cuda.synchronize()
    src = h32 if fp32_in else x.float()
    want = src.double() @ (weights["router"].double().T)
    rel = float((y.double() - want).norm() / want.norm())
    assert y.shape == (t, 256) and y.dtype == torch.float32 and rel < 2e-4, rel
    assert torch.equal(y, y2), "not deterministic"
    assert int(st["rfix_done"].abs().sum()) == 0, "arrival counters not back at zero"
    # the same logits as v5's split-K kernel (atomics, summation order differs) to FP32 rounding
    v5 = _load("_v5_for_v5b", V5)
    old = v5._run(h32 if fp32_in else x, st["router_hi"], st["router_lo"], 256, torch.float32, (32, 64, 16, 4, 2), split_x=fp32_in)
    torch.cuda.synchronize()
    assert float((y - old).norm() / old.norm()) < 1e-5


@needs_cuda
def test_router_fix_in_a_cuda_graph(mod, weights):
    st = _fp16_state(mod, weights)
    h32 = torch.randn((1, 8192), device="cuda")
    x = h32.to(torch.bfloat16).half()
    cfg = (32, 64, 32, 4, 2)
    eager = mod._router_fix(x, h32, st, cfg)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mod._router_fix(x, h32, st, cfg)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod._router_fix(x, h32, st, cfg)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager) and int(st["rfix_done"].abs().sum()) == 0


@needs_cuda
def test_pooler_result_does_not_depend_on_bk(mod, weights):
    """S7 changes the pooler's BK (512 -> 256): the K loop accumulates into one FP32 accumulator in increasing k either way, so the result is bitwise the same."""
    w = weights["pooler"].half()
    g = torch.Generator(device="cuda").manual_seed(4)
    for t in (2, 5, 8):
        x = torch.randn((t, 8192), generator=g, device="cuda").half()
        a = mod._run(x, w, None, 4096, torch.float16, (32, 64, 1, 4, 2))
        b = mod._run(x, w, None, 4096, torch.float16, (32, 128, 1, 4, 2))
        torch.cuda.synchronize()
        assert torch.equal(a, b), t


@needs_cuda
def test_whole_layer_against_the_organisers_reference():
    passed, total = lc.run(PROBLEM, KERNEL, device="cuda", cases="small", graph=True, out=lambda *_: None)
    assert (passed, total) == (13, 13)


# ------------------------------------------------------------------------------------------------ dev GPU: large-M through NVRTC (opt in: ~12 min of NVRTC on sm_75 the first time)
@pytest.fixture(scope="module")
def lm_env(weights):
    if os.environ.get("LM_NVRTC") != "1":
        pytest.skip("set LM_NVRTC=1: NVRTC builds the large-M kernels for sm_75 with the BF16 mma emulated")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("needs NVRTC")
    mod = _load("_sonnet_v5b_lm", KERNEL)
    old = mod.SMALLM_CFG
    mod.SMALLM_CFG = ((1, 4, 4, 3, 4, 4, 3), (8, 4, 4, 6, 4, 4, 6), (16, 4, 4, 12, 4, 4, 12))
    sm = mod.prepare_smallm(weights, ext=nv.NvrtcExt(mod._SMALLM_DEVICE_SRC))
    mod.SMALLM_CFG = old
    lm = mod.prepare_largem(sm, ext=nv.NvrtcLargeExt(mod._SMALLM_DEVICE_SRC + mod._LARGEM_EXTRA_SRC))
    return mod, sm, lm


@needs_cuda
@pytest.mark.parametrize("m,seed,start", [(40, 3, 0), (48, 5, 96), (40, 7, 240)])
def test_largem_experts_match_the_references_moe_block(lm_env, ref, weights, m, seed, start):
    mod, sm, lm = lm_env
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, 4096), generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.randint(0, 256, (m, 8), generator=g, device="cuda")
    ids[:, :4] = (start + torch.randint(0, 32, (m, 4), generator=g, device="cuda")) % 256      # a busy 32-expert range (it wraps for start 240)
    w = torch.rand((m, 8), generator=g, device="cuda")
    w = w / w.sum(-1, keepdim=True)
    start_t = torch.tensor([start], dtype=torch.int32, device="cuda")
    cfg = dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24)      # 2060: K slices of 512 keep the smem under 64 KB
    y, local_ids = mod.largem_experts(x, ids.to(torch.int32), sm, lm, start_t, cfg)
    pe = (weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16)
    got = mod.epilogue_triton(y, w.float().contiguous(), local_ids, pe, block_h=1024)      # SGLang's fused epilogue (>= 32 tokens) is not on the dev box
    torch.cuda.synchronize()
    local = (ids - start) % 256
    full = {k: weights[k] for k in ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs", "post_expert_norm")}
    want = ref._experts(x.float(), local, w, full)
    rel = float((got.float() - want.float()).norm() / want.float().norm())
    assert got.shape == want.shape and rel < 1.5e-2, rel
