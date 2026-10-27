"""kernels/muse_decode_layer/sonnet_v5c.py (C25): sonnet_v5b plus three switchable glue changes.
  G2 (S9_PRE_NORM, S9_NORM_EARLY)  the pre-expert RMSNorm(1 + w) as one Triton kernel, issued before the stream join   -> vector V3 (<= 1 BF16 step vs torch)
  G3 (S10_UNPOOL_GEMV)             the unpooler's 1024-column slice as one Triton GEMV that writes the zero-padded [T, 8192]
  G4 (S11_EPI_GRID)                the MoE epilogue on grid (tokens, 4)                                                   -> vector V4 (bitwise vs the 1-CTA kernel)

No GPU:  file rules, v5c differs from v5b only by the additions and the switches, every switch off is v5b's path, dispatch tables.
Dev GPU (RTX 2060, sm_75; FP16 stand-ins where tl.dot needs BF16): V3, V4, the unpooler GEMV (configs, determinism, counters, pad, graph capture),
         the whole layer against the organisers' reference (local_check small --graph, 13/13), offline sm_90 compiles.
Run:     tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v5c.py"""
import difflib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_decode_layer"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v5c.py"
V5B = ROOT / "kernels" / PROBLEM / "sonnet_v5b.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v5c", KERNEL)


@pytest.fixture(scope="module")
def weights():
    """Only the vectors these tests use (the full weight set would keep ~GB on the 6 GB dev card for the whole-layer tests that follow)."""
    g = torch.Generator(device="cuda").manual_seed(123)
    return {"pre_expert_norm": (0.1 * torch.randn(4096, generator=g, device="cuda")).to(torch.bfloat16),
            "unpooler": (0.02 * torch.randn((1024, 4096), generator=g, device="cuda")).to(torch.bfloat16)}


def _steps(a, b):
    """Distance in representable BF16 values (sign-magnitude order) between two BF16 tensors."""
    def key(x):
        i = x.view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return (key(a) - key(b)).abs()


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
    compile(src, "sonnet_v5c.py", "exec")


def test_header_has_the_predictions():
    header = "\n".join(line for line in KERNEL.read_text().splitlines() if line.startswith("#"))
    for needle in ("sonnet_v5c (C25)", "G2", "G3", "G4", "Predicted per shape", "PROBLEM 1.387", "switches", "kill"):
        assert needle in header, needle


def test_v5c_is_v5b_plus_the_c25_additions():
    a, b = V5B.read_text().splitlines(), KERNEL.read_text().splitlines()
    removed = [d[1:] for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] == "-" and d[:3] != "---"]
    added = set(b)
    # every removed line is either re-added unchanged (diff context moved / re-indented block) or one of the lines the three changes edit
    edited = {
        "# okc: problem=muse_decode_layer",
        '           "rfix_done": torch.zeros(64, dtype=torch.int32, device=router.device)}',
        "    _epilogue_kernel[(tokens,)](",
        "        TOPK=topk, KP=triton.next_power_of_2(topk), H=hidden, BLOCK_H=block_h, num_warps=warps, **_pdl_kwargs(y.device))",
        '    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, topk_ids, data["local_start"], state,',
        "    if tokens <= SLICED_UNPOOL_MAX_TOKENS:",
        '    p = (_rms(_pooler(h, state, _pooler_path(tokens, triton_proj))) * _gain(w["pre_expert_norm"])).to(torch.bfloat16)',
    }
    stray = [line for line in removed if line not in added and line not in edited and line.strip() not in {x.strip() for x in b}]
    assert not stray, stray
    src = KERNEL.read_text()
    for flag in ("S1_MOE_KNOBS = True", "S2_FUSED_ROUTING = False", "S3_PDL = False", "S5_FULL_CAT = True", "S6_ROUTER_FIX = True", "S7_POOLER_PS1 = True",
                 "S8_LARGEM = True", "S9_PRE_NORM = True", "S9_NORM_EARLY = True", "S10_UNPOOL_GEMV = True", "S11_EPI_GRID = True"):
        assert flag in src, flag


def test_all_unchanged_kernels_are_byte_identical_to_v5b(mod):
    """The CUDA sources, attention kernels and projection GEMV are v5b's, verbatim."""
    old = _load("_sonnet_v5b_for_v5c", V5B)
    for name in ("_SMALLM_DEVICE_SRC", "_SMALLM_HOST_SRC", "_LARGEM_EXTRA_SRC", "_LARGEM_HOST_SRC", "SMALLM_CFG", "LARGEM_CFG"):
        assert getattr(mod, name) == getattr(old, name), name
    import inspect
    for name in ("_gemv_splitk", "_router_fix_kernel", "_full_cat_kernel", "_attn_kernel", "_reduce_kernel", "_local_ids_kernel"):
        assert inspect.getsource(getattr(mod, name).fn) == inspect.getsource(getattr(old, name).fn), name


# ------------------------------------------------------------------------------------------------ switches / dispatch (no GPU)
def test_switches_default_on_and_off_is_v5b(mod, monkeypatch):
    assert mod.S9_PRE_NORM and mod.S9_NORM_EARLY and mod.S10_UNPOOL_GEMV and mod.S11_EPI_GRID
    monkeypatch.setattr(mod, "S11_EPI_GRID", False)
    assert [mod._epilogue_parts(t, 4096) for t in (1, 8, 16, 17, 32)] == [1] * 5
    monkeypatch.setattr(mod, "S11_EPI_GRID", True)
    assert [mod._epilogue_parts(t, 4096) for t in (1, 8, 16, 17, 32)] == [4, 4, 4, 1, 1]
    assert mod._epilogue_parts(4, 4097) == 1 and mod._epilogue_parts(4, 3 * 1024) == 1     # not a power-of-two quarter -> v5b's grid
    monkeypatch.setattr(mod, "S10_UNPOOL_GEMV", False)
    assert not mod._unpool_live(1, torch.device("cuda", 0)) if torch.cuda.is_available() else True


@needs_cuda
def test_unpool_is_live_only_up_to_8_tokens_on_bf16_hardware(mod, monkeypatch):
    monkeypatch.setattr(mod, "_triton_ok", lambda device: True)
    assert [t for t in range(1, 40) if mod._unpool_live(t, torch.device("cuda", 0))] == list(range(1, 9))
    monkeypatch.setattr(mod, "_triton_ok", lambda device: False)       # sm_75 dev card, CPU: cuBLAS + pad as in v5b
    assert not any(mod._unpool_live(t, torch.device("cuda", 0)) for t in range(1, 40))


def test_pre_norm_falls_back_to_the_module_when_off(mod, monkeypatch):
    calls = []
    state = {"pre_expert_norm": lambda x: calls.append("module") or x, "pre_expert_w": object()}
    monkeypatch.setattr(mod, "S9_PRE_NORM", False)
    x = torch.zeros(2, 4)
    assert mod._pre_expert(x, state) is x and calls == ["module"]
    monkeypatch.setattr(mod, "S9_PRE_NORM", True)
    assert mod._pre_expert(x, {"pre_expert_norm": state["pre_expert_norm"]}) is x     # CPU: no CUDA -> module


# ------------------------------------------------------------------------------------------------ V3: the norm
@needs_cuda
@pytest.mark.parametrize("t", [1, 2, 8, 16, 33, 128, 256])
@pytest.mark.parametrize("scale", [0.02, 1.0, 300.0])
def test_v3_pre_norm_within_one_bf16_step_of_torch(mod, weights, t, scale):
    g = torch.Generator(device="cuda").manual_seed(7 + t)
    w = (weights["pre_expert_norm"] + 1.0)                                  # bf16 effective weight, as MuseSpark1xRMSNorm binds it
    assert w.dtype == torch.bfloat16
    x = (scale * torch.randn((t, 4096), generator=g, device="cuda")).to(torch.bfloat16)
    x[0, ::97] *= 40                                                        # outliers
    got = mod.pre_norm_triton(x, w)
    want = F.rms_norm(x, (4096,), w, 1e-5)
    exact = (x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-5) * w.double()).to(torch.bfloat16)
    assert got.shape == x.shape and got.dtype == torch.bfloat16
    assert int(_steps(got, want).max()) <= 1 and int(_steps(got, exact).max()) <= 1
    assert float((_steps(got, want) > 0).float().mean()) < 0.01             # and almost always the very same BF16 value


@needs_cuda
def test_v3_pre_norm_edge_inputs(mod, weights):
    w = (weights["pre_expert_norm"] + 1.0)
    for x in (torch.zeros((2, 4096), device="cuda", dtype=torch.bfloat16),
              torch.full((2, 4096), 1e-20, device="cuda", dtype=torch.float32).to(torch.bfloat16),
              torch.full((2, 4096), 3e4, device="cuda", dtype=torch.float32).to(torch.bfloat16)):
        got = mod.pre_norm_triton(x, w)
        want = F.rms_norm(x, (4096,), w, 1e-5)
        assert torch.isfinite(got.float()).all() and int(_steps(got, want).max()) <= 1


@needs_cuda
def test_pre_norm_strided_input_and_graph(mod, weights):
    w = (weights["pre_expert_norm"] + 1.0)
    base = torch.randn((4, 8192), device="cuda").to(torch.bfloat16)
    x = base[:, :4096]                                                      # row stride 8192
    got = mod.pre_norm_triton(x, w)
    assert int(_steps(got, F.rms_norm(x.contiguous(), (4096,), w, 1e-5)).max()) <= 1
    mod.pre_norm_triton(x, w)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod.pre_norm_triton(x, w)
    base.copy_(torch.randn_like(base, dtype=torch.float32).to(torch.bfloat16))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, mod.pre_norm_triton(x, w))


# ------------------------------------------------------------------------------------------------ V4: the epilogue grid
def _epi_inputs(t, seed, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    y = (torch.randn((t, 8, 4096), generator=g, device=device) * 3).to(torch.bfloat16)
    sc = torch.rand((t, 8), generator=g, device=device)
    ids = torch.randint(-1, 32, (t, 8), generator=g, device=device).to(torch.int32)    # -1: slot of another GPU
    w = (1 + 0.3 * torch.randn(4096, generator=g, device=device)).to(torch.bfloat16)
    return y, sc, ids, w


@needs_cuda
@pytest.mark.parametrize("t", [1, 2, 3, 5, 8, 9, 13, 16])
@pytest.mark.parametrize("block_h", [1024, 2048])
@pytest.mark.parametrize("seed", [1, 2])
def test_v4_epilogue_grid_is_bitwise_the_one_cta_kernel(mod, monkeypatch, t, block_h, seed):
    y, sc, ids, w = _epi_inputs(t, seed)
    monkeypatch.setattr(mod, "S11_EPI_GRID", False)
    a = mod.epilogue_triton(y, sc, ids, w, block_h=block_h)
    monkeypatch.setattr(mod, "S11_EPI_GRID", True)
    b = mod.epilogue_triton(y, sc, ids, w, block_h=block_h)
    assert torch.equal(a, b)


@needs_cuda
def test_v4_remote_slots_and_remote_tokens(mod, monkeypatch):
    y, sc, ids, w = _epi_inputs(6, 5)
    ids[2] = -1                                                             # a token whose 8 experts are all on other GPUs
    y[2] = float("nan")                                                     # Marlin leaves those rows undefined: they must never be read
    got = mod.epilogue_triton(y, sc, ids, w, block_h=2048)
    assert torch.equal(got[2], torch.zeros_like(got[2])) and torch.isfinite(got.float()).all()
    monkeypatch.setattr(mod, "S11_EPI_GRID", False)
    assert torch.equal(got, mod.epilogue_triton(y, sc, ids, w, block_h=2048))


@needs_cuda
def test_epilogue_grid_matches_an_fp32_reference(mod):
    y, sc, ids, w = _epi_inputs(4, 9)
    got = mod.epilogue_triton(y, sc, ids, w, block_h=1024).float()
    live = (ids >= 0).float()
    yw = y.float() * w.float()
    coef = sc * live * torch.rsqrt(yw.pow(2).mean(-1) + mod.POST_NORM_EPS)
    want = ((y.float() * coef[..., None]).sum(1)) * w.float()
    assert float((got - want).norm() / want.norm()) < 5e-3


@needs_cuda
def test_epilogue_grid_in_a_cuda_graph(mod):
    y, sc, ids, w = _epi_inputs(8, 4)
    eager = mod.epilogue_triton(y, sc, ids, w, block_h=2048)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod.epilogue_triton(y, sc, ids, w, block_h=2048)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


# ------------------------------------------------------------------------------------------------ G3: the unpooler GEMV (FP16 stand-in for BF16 tl.dot on the 2060)
@pytest.fixture(scope="module")
def unpool_state(weights):
    w = weights["unpooler"][:1024].half()
    full = torch.zeros((8192, 4096), dtype=torch.half, device="cuda")
    full[:1024] = w
    return {"dense": {"unpooler": full},
            "unpool_part": torch.empty((8, 16, 1024), dtype=torch.float32, device="cuda"),
            "unpool_done": torch.zeros(64, dtype=torch.int32, device="cuda")}


UNPOOL_CFGS = [(16, 256, 2, 4, 2), (32, 128, 1, 4, 2), (16, 128, 4, 4, 2), (16, 128, 8, 4, 2), (32, 64, 2, 4, 3), (16, 64, 1, 4, 2)]


@needs_cuda
@pytest.mark.parametrize("cfg", UNPOOL_CFGS)
@pytest.mark.parametrize("t", [1, 3, 8, 16])
def test_unpool_matches_the_slice_gemm_and_zero_pads(mod, unpool_state, cfg, t):
    g = torch.Generator(device="cuda").manual_seed(11 + t)
    y = torch.randn((t, 4096), generator=g, device="cuda").half()
    out = mod.unpool_sliced(y, unpool_state, cfg)
    want = y.double() @ unpool_state["dense"]["unpooler"][:1024].double().T
    out2 = mod.unpool_sliced(y, unpool_state, cfg)
    torch.cuda.synchronize()
    assert out.shape == (t, 8192) and out.stride() == (8192, 1) and out.dtype == torch.half
    assert float((out[:, :1024].double() - want).norm() / want.norm()) < 1e-3
    assert torch.equal(out[:, 1024:], torch.zeros_like(out[:, 1024:]))
    assert torch.equal(out, out2), "not deterministic"
    assert int(unpool_state["unpool_done"].abs().sum()) == 0, "arrival counters not back at zero"


@needs_cuda
@pytest.mark.parametrize("cfg", UNPOOL_CFGS[:3])
def test_unpool_kernel_writes_the_whole_row_itself(mod, unpool_state, cfg):
    """The output comes from torch.empty in the layer: the kernel must overwrite every one of its T x 8192 elements (here pre-filled with NaN)."""
    import triton
    t = 5
    y = torch.randn((t, 4096), device="cuda").half()
    out = torch.full((t, 8192), float("nan"), device="cuda", dtype=torch.half)
    bn, bk, split, warps, stages = cfg
    ctas = (1024 // bn) * split
    mod._unpool_kernel[(1024 // bn, split)](y, unpool_state["dense"]["unpooler"], out, unpool_state["unpool_part"], unpool_state["unpool_done"], t,
                                           y.stride(0), out.stride(0), BM=16, BN=bn, BK=bk, SPLIT=split, K=4096, N=1024, NCOL=8192,
                                           PADB=triton.next_power_of_2(triton.cdiv(7168, ctas)), num_warps=warps, num_stages=stages)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    # rows beyond t of a larger buffer are not touched
    big = torch.full((8, 8192), 7.0, device="cuda", dtype=torch.half)
    mod._unpool_kernel[(1024 // bn, split)](y, unpool_state["dense"]["unpooler"], big, unpool_state["unpool_part"], unpool_state["unpool_done"], t,
                                           y.stride(0), big.stride(0), BM=16, BN=bn, BK=bk, SPLIT=split, K=4096, N=1024, NCOL=8192,
                                           PADB=triton.next_power_of_2(triton.cdiv(7168, ctas)), num_warps=warps, num_stages=stages)
    torch.cuda.synchronize()
    assert bool((big[t:] == 7.0).all()) and torch.equal(big[:t], out)


@needs_cuda
def test_unpool_split_k_is_deterministic_and_close_to_the_unsplit_result(mod, unpool_state):
    y = torch.randn((8, 4096), device="cuda").half()
    a = mod.unpool_sliced(y, unpool_state, (16, 128, 4, 4, 2))
    b = mod.unpool_sliced(y, unpool_state, (16, 128, 1, 4, 2))
    torch.cuda.synchronize()
    assert float((a.float() - b.float()).norm() / b.float().norm()) < 2e-3


@needs_cuda
def test_unpool_in_a_cuda_graph_and_changing_t(mod, unpool_state):
    cfg = UNPOOL_CFGS[0]
    y = torch.randn((8, 4096), device="cuda").half()
    eager = mod.unpool_sliced(y, unpool_state, cfg)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mod.unpool_sliced(y, unpool_state, cfg)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod.unpool_sliced(y, unpool_state, cfg)
    y.copy_(torch.randn_like(y, dtype=torch.float32).half())
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, mod.unpool_sliced(y, unpool_state, cfg)) and not torch.equal(out, eager)
    assert int(unpool_state["unpool_done"].abs().sum()) == 0
    for t in (1, 2, 7):                                                     # a smaller T after a larger one leaves no stale rows behind
        yt = torch.randn((t, 4096), device="cuda").half()
        o = mod.unpool_sliced(yt, unpool_state, cfg)
        assert o.shape[0] == t and torch.equal(o[:, 1024:], torch.zeros_like(o[:, 1024:]))


# ------------------------------------------------------------------------------------------------ offline compiles for the judge's GPU
def _compile90(kernel, sig, consts, warps, stages=3):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    return triton.compile(ASTSource(fn=kernel, signature=sig, constexprs=consts), target=GPUTarget("cuda", 90, 32), options={"num_warps": warps, "num_stages": stages})


@needs_cuda
def test_new_kernels_compile_for_sm90_within_smem(mod):
    c = _compile90(mod._pre_norm_kernel, {"x_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "eps": "fp32", "stride_x": "i32", "stride_o": "i32"},
                   {"N": 4096, "BLOCK": 4096}, mod.PRE_NORM_WARPS)
    assert c.metadata.shared < 64 * 1024 and "rsqrt" in c.asm["ptx"]
    bn, bk, split, warps, stages = mod.UNPOOL_CFG
    sig = {"x_ptr": "*bf16", "w_ptr": "*bf16", "out_ptr": "*bf16", "part_ptr": "*fp32", "done_ptr": "*i32", "M": "i32", "stride_x": "i32", "stride_o": "i32"}
    consts = {"BM": 16, "BN": bn, "BK": bk, "SPLIT": split, "K": 4096, "N": 1024, "NCOL": 8192, "PADB": 64}
    c = _compile90(mod._unpool_kernel, sig, consts, warps, stages)
    ptx = c.asm["ptx"]
    # (cp.async needs the 16-byte divisibility attributes of a real launch: tools/ptx_audit.py checks that: cp.async=32, smem 48 KB, 0 spills)
    assert c.metadata.shared < 200 * 1024 and ("atom" in ptx) == (split > 1) and "mma" in ptx
    for part in (4,):
        sig = {"y_ptr": "*bf16", "w_ptr": "*bf16", "s_ptr": "*fp32", "ids_ptr": "*i32", "out_ptr": "*bf16", "eps": "fp32"}
        for block_h in (1024, 2048):
            c = _compile90(mod._epilogue_kernel, sig, {"TOPK": 8, "KP": 8, "H": 4096, "BLOCK_H": block_h, "PDL": False, "NPART": part}, 8)
            assert c.metadata.shared < 64 * 1024


# ------------------------------------------------------------------------------------------------ dev GPU: the whole layer
@needs_cuda
def test_whole_layer_against_the_organisers_reference():
    passed, total = lc.run(PROBLEM, KERNEL, device="cuda", cases="small", graph=True, out=lambda *_: None)
    assert (passed, total) == (13, 13)


@needs_cuda
@pytest.mark.parametrize("off", [("S9_PRE_NORM", "S10_UNPOOL_GEMV", "S11_EPI_GRID"), ("S9_NORM_EARLY",)])
def test_whole_layer_with_switches_off(tmp_path, off):
    src = KERNEL.read_text()
    for name in off:
        src, n = re.subn(rf"^{name} = True", f"{name} = False", src, flags=re.M)
        assert n == 1, name
    path = tmp_path / "sonnet_v5c_off.py"
    path.write_text(src)
    passed, total = lc.run(PROBLEM, path, device="cuda", cases="small", graph=False, out=lambda *_: None)
    assert (passed, total) == (13, 13)


# ------------------------------------------------------------------------------------------------ prepare()'s self-check of the three switches
def test_glue_selfcheck_switches_a_failing_kernel_off(monkeypatch):
    m = _load("_sonnet_v5c_sc", KERNEL)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    w = torch.zeros(4096, dtype=torch.bfloat16, device=dev)
    state = {"post_expert_w": w, "pre_expert_w": w, "pre_expert_norm": lambda x: x, "dense": {"unpooler": torch.zeros((8192, 4096), dtype=torch.bfloat16, device=dev)}}
    def boom(*a, **k):
        raise RuntimeError("compile failed")
    monkeypatch.setattr(m, "pre_norm_triton", boom)
    monkeypatch.setattr(m, "unpool_sliced", boom)
    monkeypatch.setattr(m, "epilogue_triton", boom)
    monkeypatch.setattr(m, "_triton_ok", lambda device: True)
    if dev.type == "cpu":
        pytest.skip("generators need the device; CPU leg covered by the raising stubs only")
    m._glue_selfcheck(state)
    assert not m.S9_PRE_NORM and not m.S10_UNPOOL_GEMV and not m.S11_EPI_GRID


@needs_cuda
def test_glue_selfcheck_keeps_good_kernels_on(monkeypatch):
    m = _load("_sonnet_v5c_sc2", KERNEL)
    w = (torch.randn(4096, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    state = {"post_expert_w": w, "pre_expert_w": w, "pre_expert_norm": lambda x: F.rms_norm(x, (4096,), w, 1e-5),
             "dense": {"unpooler": (0.02 * torch.randn((8192, 4096), device="cuda")).half()}, "unpool_part": torch.empty((8, 16, 1024), device="cuda"),
             "unpool_done": torch.zeros(64, dtype=torch.int32, device="cuda")}
    monkeypatch.setattr(m, "_triton_ok", lambda device: False)       # sm_75: G3 not live here (BF16 tl.dot), the check leaves its switch alone
    m._glue_selfcheck(state)
    assert m.S9_PRE_NORM and m.S10_UNPOOL_GEMV and m.S11_EPI_GRID
