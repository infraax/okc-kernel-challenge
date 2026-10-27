"""kernels/muse_decode_layer/sonnet_v4.py: sonnet_v3_s1 plus S4, the MoE problem's small-M CUDA path (<= 16 tokens) for the layer's experts.

No GPU:       file rules, the CUDA source against the MoE problem's sonnet_v3b (only the route kernel's local_start handling differs), dispatch, prepare()'s
              fallbacks (build fails / self-check fails -> Marlin serves every M).
Dev GPU (RTX 2060, sm_75, no nvcc, no BF16 tensor cores): the CUDA device source runs through NVRTC with the BF16 mma emulated (tests/nvrtc_smallm.py), the
              module's SMALLM_CFG is overridden with the old v3 table (the tuned one needs ~67 KB of smem, the 2060 has 64 KB). Inputs come from the organisers'
              reference: generate_input's residual stream through the reference's router (real, skewed routing), the reference's own expert weights generator and its
              `_experts` as the oracle; sm_80 / sm_90 assembly of the device source (ptxas).
Run:          tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v4.py"""
import importlib.util
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
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v4.py"
V3B = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v3b.py"
S1 = ROOT / "kernels" / PROBLEM / "sonnet_v3_s1.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
OLD_TABLE = ((1, 4, 4, 3, 4, 4, 3), (8, 4, 4, 6, 4, 4, 6), (16, 4, 4, 12, 4, 4, 12))  # sonnet_v3's table: 34 KB smem


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v4_decode_layer", KERNEL)


@pytest.fixture(scope="module")
def v3b():
    return _load("_sonnet_v3b_for_v4", V3B)


@pytest.fixture(scope="module")
def ref():
    return lc.Loaded(PROBLEM, KERNEL).ref


# ------------------------------------------------------------------------------------------------ file rules
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_decode_layer"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket",
                r"requests", r"environ", r"import reference", r"from reference", r"\.tolist\(", r"urllib"):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert 'print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")' in src
    compile(src, "sonnet_v4.py", "exec")


def test_v3_s1_code_is_unchanged_outside_the_small_m_additions():
    """Everything of the board entry sonnet_v3_s1 is still in v4 verbatim except the three places that gain S4: _experts' first lines, prepare()'s SGLang branch,
    and the inserted block."""
    a, b = S1.read_text(), KERNEL.read_text()
    a_body = a[a.index("import functools"):]
    b_body = b[b.index("import functools"):]
    for cut in ('    if HAVE_SGLANG:\n        return _prepare_sglang(weights)\n    return {"weights": weights,',):
        assert cut in a_body
    a_pieces = a_body.split("def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None):")
    b_pieces = b_body.split("def _experts(x, topk_weights, topk_ids, local_start, state, local_ids=None):")
    assert len(a_pieces) == len(b_pieces) == 2
    assert b_pieces[0].startswith(a_pieces[0])                      # everything before _experts is s1 + the inserted block
    assert b_pieces[0][: len(a_pieces[0])] == a_pieces[0]
    s1_rest = a_pieces[1].replace('        return _prepare_sglang(weights)', '')
    b_rest = b_pieces[1].replace('        state = _prepare_sglang(weights)\n        _prepare_smallm_for_layer(weights, state)\n        return state', '')
    b_rest = b_rest.replace('    if local_ids is None and _smallm_live(x.shape[0], state):\n        return _experts_smallm(x, topk_weights, topk_ids, local_start, state)\n', '')
    assert b_rest == s1_rest
    assert "S1_MOE_KNOBS = True" in b and "S2_FUSED_ROUTING = False" in b and "S3_PDL = False" in b


def _diff_lines(a, b):
    al, bl = a.splitlines(), b.splitlines()
    import difflib

    return [d for d in difflib.unified_diff(al, bl, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]


def test_cuda_source_is_v3b_except_the_route_kernels_local_start(mod, v3b):
    diff = _diff_lines(v3b._SMALLM_DEVICE_SRC, mod._SMALLM_DEVICE_SRC)
    assert len(diff) <= 10, "\n".join(diff)
    assert any("lo_ptr" in d for d in diff) and any("& 255" in d for d in diff)
    assert mod.SMALLM_CFG == v3b.SMALLM_CFG        # the tuned table of the MoE problem
    for name in ("pack_w4_tiles", "make_scale_lut", "_nibble_bit_positions"):
        import inspect

        assert inspect.getsource(getattr(mod, name)) == inspect.getsource(getattr(v3b, name))


# ------------------------------------------------------------------------------------------------ dispatch / prepare (no GPU)
@pytest.mark.parametrize("tokens,live", [(1, True), (2, True), (8, True), (9, True), (16, True), (17, False), (32, False), (128, False), (256, False)])
def test_smallm_range(mod, tokens, live):
    assert mod._smallm_live(tokens, {"smallm": object()}) is live
    assert mod._smallm_live(tokens, {"smallm": None}) is False
    assert mod._smallm_live(tokens, {}) is False


def test_smallm_off_by_switch(mod, monkeypatch):
    monkeypatch.setattr(mod, "S4_SMALLM", False)
    assert mod._smallm_live(1, {"smallm": object()}) is False


@pytest.mark.parametrize("tokens", [1, 8, 16, 17, 32])
def test_experts_dispatch(mod, monkeypatch, tokens):
    calls = []
    monkeypatch.setattr(mod, "_experts_smallm", lambda *a, **k: calls.append("small") or "S")
    monkeypatch.setattr(mod, "local_ids_triton", lambda *a, **k: calls.append("marlin") or (_ for _ in ()).throw(RuntimeError("stop")))
    state = {"smallm": object(), "bank": {"size_n": 8}}
    x = torch.zeros(tokens, 16)
    try:
        out = mod._experts(x, torch.ones(tokens, 8), torch.zeros(tokens, 8, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), state)
    except RuntimeError:
        out = None
    assert (calls == ["small"] and out == "S") if tokens <= 16 else calls == ["marlin"]


def _prepared(mod, monkeypatch, build, errs):
    seen = []
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod, "prepare_smallm", lambda w, flags=(): seen.append(flags) or build(flags))
    it = iter(errs)
    monkeypatch.setattr(mod, "_smallm_selfcheck", lambda st: next(it))
    state = {}
    mod._prepare_smallm_for_layer({}, state)
    return state, seen


def test_prepare_keeps_smallm_when_selfcheck_passes(mod, monkeypatch):
    st, seen = _prepared(mod, monkeypatch, lambda f: {"v": f}, [3e-3])
    assert st["smallm"] == {"v": ()} and seen == [()]


def test_prepare_retries_with_fp32_products(mod, monkeypatch):
    st, seen = _prepared(mod, monkeypatch, lambda f: {"v": f}, [0.3, 2e-3])
    assert st["smallm"] == {"v": ("-DSM_FP32_MUL=1",)} and seen == [(), ("-DSM_FP32_MUL=1",)]


@pytest.mark.parametrize("errs", [[0.3, 0.4], [float("nan"), float("inf")]])
def test_prepare_falls_back_to_marlin_when_selfcheck_fails(mod, monkeypatch, errs):
    st, seen = _prepared(mod, monkeypatch, lambda f: {"v": f}, errs)
    assert st["smallm"] is None and len(seen) == 2


def test_prepare_falls_back_to_marlin_when_build_fails(mod, monkeypatch):
    def boom(flags):
        raise RuntimeError("no nvcc")

    st, seen = _prepared(mod, monkeypatch, boom, [])
    assert st["smallm"] is None and len(seen) == 1


def test_prepare_without_gpu_or_switch(mod, monkeypatch):
    monkeypatch.setattr(mod, "S4_SMALLM", False)
    st = {}
    mod._prepare_smallm_for_layer({}, st)
    assert st["smallm"] is None


def test_no_scratch_allocation_in_the_timed_path(mod):
    import inspect

    src = inspect.getsource(mod.smallm_experts) + inspect.getsource(mod._experts_smallm)
    assert "torch.zeros" not in src and ".zero_(" not in src and "synchronize" not in src


# ------------------------------------------------------------------------------------------------ GPU: NVRTC on the dev card
@pytest.fixture(scope="module")
def ext(mod):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    return nv.NvrtcExt(mod._SMALLM_DEVICE_SRC)


@needs_cuda
@pytest.mark.parametrize("sm,extra", [("sm_80", ()), ("sm_90", ()), ("sm_90", ("-DSM_FP32_MUL=1",))])
def test_device_source_assembles_for_judge_arch(mod, sm, extra):
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    cubin, low = nv.sass_check(mod._SMALLM_DEVICE_SRC, sm, extra=extra)
    assert len(cubin) > 1000 and len(low) == len(nv.KERNEL_NAMES)


def _reference_routing(ref, batch, q_len, seed, start_mode):
    """ids [T, 8] int32 (global), weights [T, 8] FP32 and local_start from generate_input's stream through the reference's router; start_mode 'busiest' (the
    judge's rule: the 32-expert range with the most routes), 'wrap' (that range moved to 240..255, 0..15) or an int (moved to start)."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    r = ref._router(gen, dev, None)
    data = ref.generate_input(batch=batch, ctx=64, layer="sliding", seed=seed, q_len=q_len, dist="soft", device=dev)
    s = data["stream"].float()
    h = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5) * (1.0 + 0.1 * torch.randn(8192, generator=gen, device=dev))
    scores, biased = ref.route(h, {"router": r["router"], "router_bias": r["router_bias"]})
    ids = biased.topk(8, dim=-1).indices
    picked = scores.gather(1, ids)
    w = picked / (picked.sum(-1, keepdim=True) + ref.ROUTE_EPS)
    counts = torch.bincount((ids // 32).flatten(), minlength=8)
    busiest = 32 * int(counts.argmax())
    start = {"busiest": busiest, "wrap": 240}.get(start_mode, start_mode) if not isinstance(start_mode, int) else start_mode
    ids = (ids - busiest + start) % 256          # the routing's busiest 32-expert range becomes this GPU's (it wraps past 255 for start 240)
    return ids.to(torch.int32), w.float(), start


@pytest.fixture(scope="module")
def full_weights(ref):
    """The reference's own generator for the 32 local experts (full size: ~0.8 GB) plus the post-expert norm."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(5)
    w13, w13_sf, gs13 = ref.random_weights(2 * ref.D_EXPERT, ref.D_EXPERT, generator=gen, device=dev, leading=(ref.LOCAL_EXPERTS,))
    w2, w2_sf, gs2 = ref.random_weights(ref.D_EXPERT, ref.D_EXPERT, generator=gen, device=dev, leading=(ref.LOCAL_EXPERTS,))
    gs13 = gs13 * torch.exp(torch.randn((ref.LOCAL_EXPERTS,), generator=gen, device=dev))
    gs2 = gs2 * torch.exp(torch.randn((ref.LOCAL_EXPERTS,), generator=gen, device=dev))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13, "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2,
            "post_expert_norm": (0.1 * torch.randn(ref.D_EXPERT, generator=gen, device=dev)).to(torch.bfloat16)}


@pytest.fixture(scope="module")
def sm_state(mod, ext, full_weights):
    old = mod.SMALLM_CFG
    mod.SMALLM_CFG = OLD_TABLE
    try:
        sm = mod.prepare_smallm(full_weights, ext=ext)
        yield {"smallm": sm, "post_expert_w": (full_weights["post_expert_norm"].float() + 1.0).to(torch.bfloat16)}
    finally:
        mod.SMALLM_CFG = old


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


@needs_cuda
@pytest.mark.parametrize("batch,q_len,mode", [(1, 1, "busiest"), (1, 1, "wrap"), (4, 1, "busiest"), (1, 8, "busiest"), (1, 8, "wrap"), (2, 8, "busiest"),
                                              (16, 1, "busiest"), (16, 1, "wrap"), (3, 1, 0), (5, 1, 224)])
def test_experts_match_the_references_moe_block(mod, ref, sm_state, full_weights, batch, q_len, mode):
    """_experts at 1..16 tokens against reference._experts (the organisers' experts: dequantize, SwiGLU, post-norm, weighted sum) on real routing."""
    ids, w, start = _reference_routing(ref, batch, q_len, 3 + batch * 5 + q_len, mode)
    t = batch * q_len
    g = torch.Generator(device="cuda").manual_seed(40 + t)
    x = torch.randn((t, ref.D_EXPERT), generator=g, device="cuda").to(torch.bfloat16)
    start_t = torch.tensor([start], dtype=torch.int32, device="cuda")
    old = mod.SMALLM_CFG
    mod.SMALLM_CFG = OLD_TABLE
    try:
        got = mod._experts(x, w, ids, start_t, sm_state)
        got2 = mod._experts(x, w, ids, start_t, sm_state)
    finally:
        mod.SMALLM_CFG = old
    torch.cuda.synchronize()
    local = (ids.long() - start) % 256
    want = ref._experts(x.float(), local, w, full_weights)
    assert got.shape == (t, ref.D_EXPERT) and got.dtype == torch.bfloat16 and not torch.isnan(got.float()).any()
    assert torch.equal(got.view(torch.int16), got2.view(torch.int16)), "not deterministic"
    assert int(((local >= 0) & (local < 32)).sum()) > 0           # the case exercises local experts
    assert _rel(got, want) < 1.5e-2, _rel(got, want)


@needs_cuda
def test_route_kernel_reads_local_start_from_the_device_with_wrap(mod, ext):
    ids = torch.tensor([[250, 255, 0, 15, 16, 100, 239, 240]], dtype=torch.int32, device="cuda")
    for start in (240, 0, 100, 250):
        s = torch.tensor([start], dtype=torch.int32, device="cuda")
        local = torch.empty_like(ids)
        tab = torch.zeros(mod.SM_TAB_INTS, dtype=torch.int32, device="cuda")
        ext.smallm_route(ids, local, tab, s, 32)
        torch.cuda.synchronize()
        rel = (ids.long() - start) % 256
        want = torch.where(rel < 32, rel, torch.full_like(rel, -1)).to(torch.int32)
        assert torch.equal(local, want), (start, local.tolist(), want.tolist())
        assert int(tab[32 * mod.SM_TABW]) == len({int(v) for v in want.flatten() if v >= 0})


@needs_cuda
def test_experts_graph_capture_and_replay(mod, ref, sm_state, full_weights):
    ids, w, start = _reference_routing(ref, 1, 8, 9, "busiest")
    x = torch.randn((8, ref.D_EXPERT), device="cuda").to(torch.bfloat16)
    start_t = torch.tensor([start], dtype=torch.int32, device="cuda")
    old = mod.SMALLM_CFG
    mod.SMALLM_CFG = OLD_TABLE
    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            eager = mod._experts(x, w, ids, start_t, sm_state)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = mod._experts(x, w, ids, start_t, sm_state)
        g.replay()
        g.replay()
        torch.cuda.synchronize()
    finally:
        mod.SMALLM_CFG = old
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
