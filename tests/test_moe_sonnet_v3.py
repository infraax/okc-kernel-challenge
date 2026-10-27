"""kernels/muse_moe_w4a16/sonnet_v3.py: the small-M (M <= 16) streaming split-K W4A16 path.

CPU (no GPU, no SGLang): file rules, the bit plan of the repack, pack -> torch dequant twin == nvfp4.dequantize (exact), the scale lut, dispatch.
GPU dev card (RTX 2060, sm_75; no nvcc, no BF16 tensor cores): the CUDA device source is compiled with NVRTC (tests/nvrtc_smallm.py) and run through its
shuffle emulation of mma.m16n8k16 (same fragment layouts as the real instruction); inputs come from problems/muse_moe_w4a16/reference.generate_input
(real routing, dist normal and skewed), the check is reference.check_implementation (relative L2 1.5%, per row 5x). The sm_80 / sm_90 builds are assembled by NVRTC
(its ptxas) to validate every inline-asm string for the judge's GPU.

Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v3.py"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "problems" / "muse_moe_w4a16"))

KERNEL = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v3.py"
V2B = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v2b.py"

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v3_moe", KERNEL)


# --------------------------------------------------------------------------------------------- source rules
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\(", r"urllib"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert "okc-diag HAVE_SGLANG" in src
    compile(src, "sonnet_v3.py", "exec")


def test_v2b_code_is_unchanged_below_16_tokens_boundary():
    """Everything of sonnet_v2b outside its header and entry points is still in v3 verbatim (the M > 16 path is v2b)."""
    a, b = V2B.read_text(), KERNEL.read_text()
    lo = "import re\n"
    start_a = a.index(lo)
    end_a = a.index("# ============================================================================================ plain-PyTorch path")
    region = a[start_a:end_a]
    assert region in b
    tail_a = a[end_a:a.index("# ============================================================================================ entry points")]
    assert tail_a in b


def test_no_scratch_allocation_in_kernel_fn(mod):
    import inspect

    src = inspect.getsource(mod.smallm_experts)
    # only activation temporaries (route outputs, inter, y); the table, partials and counters come from prepare()
    assert "tab" in src and "torch.zeros" not in src and ".zero_(" not in src


# --------------------------------------------------------------------------------------------- bit plan / repack (CPU)
def test_bit_plan_tiles_the_word(mod):
    pos = mod._nibble_bit_positions()
    assert sorted(pos.values()) == list(range(32))
    # regs 0..2: bit (target + rot) of the word lands on its bf16 position under rotr; reg 3 by shifts
    for r in range(3):
        for h in range(2):
            for j in range(4):
                src = pos[(r, h, j)]
                assert (src - mod.SM_ROT[r]) % 32 == mod.SM_NIB_TARGET[j] + 16 * h
    for h in range(2):
        for j in range(4):
            src = pos[(3, h, j)]
            tgt = mod.SM_NIB_TARGET[j] + 16 * h
            shift = tgt - src
            assert shift in (6, -2)  # (w << 6) & 0x80C080C0 and (w >> 2) & 0x01000100 in unpack_word
            assert ((0x80C080C0 if shift == 6 else 0x01000100) >> tgt) & 1


def _random_nvfp4(rows, k, seed, device="cpu", e4m3_edge=False):
    from nvfp4 import random_weights

    g = torch.Generator(device=device).manual_seed(seed)
    packed, sf, gs = random_weights(rows, k, generator=g, device=device)
    if e4m3_edge:  # every non-NaN E4M3 byte pattern (subnormals, tiny, huge, negative zero ...) as block scales
        codes = torch.arange(256, dtype=torch.uint8)
        codes = codes[(codes & 0x7F) != 0x7F]
        idx = torch.randint(0, codes.numel(), sf.shape, generator=torch.Generator().manual_seed(seed + 1))
        sf = codes[idx].to(device).view(torch.float8_e4m3fn)
    return packed, sf, gs


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("edge", [False, True])
def test_pack_unpack_equals_dequantize_exactly(mod, interleave, edge):
    from nvfp4 import dequantize

    rows, k = 64, 256
    packed, sf, gs = _random_nvfp4(rows, k, 11, e4m3_edge=edge)
    words, scales = mod.pack_w4_tiles(packed, sf, interleave_gate_up=interleave)
    assert words.shape == (rows // 16, k // 64, 32, 4) and scales.shape == (rows // 16, k // 64, 8, 2, 4)
    lut = mod.make_scale_lut("cpu")
    got = mod.unpack_w4_tiles_torch(words, scales, lut)  # [rows, k] = value * 2**-30 (global scale excluded)
    ref = dequantize(packed, sf, torch.tensor(1.0))  # E2M1 * block scale
    if interleave:
        half = rows // 2
        order = torch.stack([torch.arange(half).view(-1, 8), torch.arange(half, rows).view(-1, 8)], dim=1).reshape(-1)
        ref = ref[order]
    exp = ref * (2.0 ** -mod.SM_BITS_SHIFT)
    assert torch.equal(got.float().view(torch.int32), exp.float().view(torch.int32)) or torch.equal(got, exp)


def test_scale_lut_values(mod):
    lut = mod.make_scale_lut("cpu").to(torch.int64) & 0xFFFFFFFF
    b = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    lo = ((lut & 0xFFFF) << 16).to(torch.int64)
    lo = torch.where(lo >= 2 ** 31, lo - 2 ** 32, lo).to(torch.int32).view(torch.float32)
    hi = (lut & 0xFFFF0000)
    hi = torch.where(hi >= 2 ** 31, hi - 2 ** 32, hi).to(torch.int32).view(torch.float32)
    ok = ~torch.isnan(b)
    assert torch.equal(lo[ok], (b * 2.0 ** 96)[ok]) and torch.equal(hi[ok], lo[ok])
    assert torch.isfinite(lo[ok]).all()


def test_exactness_of_operands_in_bf16(mod):
    """value * 2**-30 of every (E2M1 code, E4M3 scale) pair is a normal BF16 number: the product the kernel forms is exact."""
    codes = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    sc = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    sc = sc[~torch.isnan(sc)]
    prod = codes[:, None] * sc[None, :] * 2.0 ** -mod.SM_BITS_SHIFT
    assert torch.equal(prod.to(torch.bfloat16).float(), prod)
    nz = prod[prod != 0].abs()
    assert nz.min() >= 2.0 ** -126  # normal numbers


def test_smallm_cfg_buckets(mod):
    assert mod.smallm_cfg(1)["nz1"] == 3 and mod.smallm_cfg(2) == mod.smallm_cfg(8) and mod.smallm_cfg(16)["nz1"] == 12
    assert mod.smallm_cfg(17) is None
    for mx, nw1, ks1, nz1, nw2, ks2, nz2 in mod.SMALLM_CFG:
        assert nw1 in (2, 4, 8) and nw2 in (2, 4, 8) and (4096 // ks1) % 256 == 0 and (4096 // ks2) % 256 == 0
        assert 4096 // ks1 <= 2048 and 4096 // ks2 <= 2048 and 1 <= nz1 <= 32 and 1 <= nz2 <= 32


def test_dispatch_small_m_vs_marlin(mod, monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "HAVE_SGLANG", True)
    monkeypatch.setattr(mod, "_kernel_smallm", lambda d, s: calls.append("small") or "S")
    monkeypatch.setattr(mod, "_kernel_sglang", lambda d, s: calls.append("marlin") or "M")

    def data(m, start=0, count=32):
        return {"hidden_states": torch.zeros(m, 4), "config": {"local_start": start, "local_count": count}}

    st = {"smallm": object()}
    assert [mod.kernel_fn(data(m), st) for m in (1, 8, 16)] == ["S", "S", "S"]
    assert [mod.kernel_fn(data(m), st) for m in (17, 32, 128, 256)] == ["M", "M", "M", "M"]
    assert mod.kernel_fn(data(4, 32, 32), st) == "M" and mod.kernel_fn(data(4, 0, 256), st) == "M"
    assert mod.kernel_fn(data(4), {"smallm": None}) == "M"


def _prepare_with(mod, monkeypatch, build, errors):
    """prepare() with SGLang's repack and the small-M builder stubbed: build(flags) -> state or raises; errors = successive self-check results."""
    monkeypatch.setattr(mod, "HAVE_SGLANG", True)
    monkeypatch.setattr(mod, "_prepare_sglang", lambda w: {"norm_w": torch.zeros(1)})
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: True)
    seen = []
    monkeypatch.setattr(mod, "prepare_smallm", lambda w, flags=(): seen.append(flags) or build(flags))
    errs = iter(errors)
    monkeypatch.setattr(mod, "_smallm_selfcheck", lambda st: next(errs))
    st = mod.prepare({})
    return st, seen


def test_prepare_selects_variant_by_selfcheck(mod, monkeypatch):
    st, seen = _prepare_with(mod, monkeypatch, lambda f: {"v": f}, [1e-3])
    assert st["smallm"] == {"v": ()} and seen == [()]
    st, seen = _prepare_with(mod, monkeypatch, lambda f: {"v": f}, [0.2, 2e-3])  # bf16 multiply wrong -> FP32 products
    assert st["smallm"] == {"v": ("-DSM_FP32_MUL=1",)} and len(seen) == 2
    st, seen = _prepare_with(mod, monkeypatch, lambda f: {"v": f}, [0.2, 0.3])
    assert st["smallm"] is None
    st, seen = _prepare_with(mod, monkeypatch, lambda f: {"v": f}, [float("nan"), float("inf")])  # nan / inf are not < tol
    assert st["smallm"] is None and len(seen) == 2
    def boom(f):
        raise RuntimeError("no nvcc")
    st, seen = _prepare_with(mod, monkeypatch, boom, [])
    assert st["smallm"] is None and len(seen) == 1


# --------------------------------------------------------------------------------------------- GPU: NVRTC on the dev card
@pytest.fixture(scope="module")
def ext(mod):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    return nv.NvrtcExt(mod._SMALLM_DEVICE_SRC)


@needs_cuda
@pytest.mark.parametrize("sm", ["sm_80", "sm_90"])
def test_device_source_assembles_for_judge_arch(mod, sm):
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    cubin, low = nv.sass_check(mod._SMALLM_DEVICE_SRC, sm)
    assert len(cubin) > 1000 and len(low) == len(nv.KERNEL_NAMES)


@needs_cuda
def test_fp32_product_variant_assembles_for_sm90(mod):
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    cubin, low = nv.sass_check(mod._SMALLM_DEVICE_SRC, "sm_90", extra=("-DSM_FP32_MUL=1",))
    assert len(cubin) > 1000


def _route_ref(ids, lo, nloc):
    flat = ids.flatten().tolist()
    loc = [g - lo if lo <= g < lo + nloc else -1 for g in flat]
    exps = sorted({e for e in loc if e >= 0})
    rows = [(e, [s for s, v in enumerate(loc) if v == e]) for e in exps]
    return loc, rows


@needs_cuda
@pytest.mark.parametrize("dist", ["normal", "skewed"])
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 12, 16])
def test_route_table_matches_python(mod, ext, bs, dist):
    import reference

    d = reference.generate_input(bs, "ep8", 100 + bs, dist=dist)
    ids = d["topk_ids"].contiguous()
    local = torch.empty_like(ids)
    tab = torch.full((mod.SM_TAB_INTS,), 7777, dtype=torch.int32, device="cuda")
    ext.smallm_route(ids, local, tab, 0, 32)
    torch.cuda.synchronize()
    loc, rows = _route_ref(ids.cpu(), 0, 32)
    assert local.flatten().tolist() == loc
    t = tab.cpu().view(-1, mod.SM_TABW)
    assert int(t[32, 0]) == len(rows)
    for z, (e, slots) in enumerate(rows):
        assert int(t[z, 0]) == e and int(t[z, 1]) == len(slots) and t[z, 2:2 + len(slots)].tolist() == slots
    for z in range(len(rows), 32):
        assert int(t[z, 0]) == -1 and int(t[z, 1]) == 0


@needs_cuda
def test_route_table_nonlocal_range(mod, ext):
    ids = torch.tensor([[40, 41, 42, 43, 44, 45, 46, 47], [0, 33, 34, 35, 36, 37, 38, 39]], dtype=torch.int32, device="cuda")
    local = torch.empty_like(ids)
    tab = torch.zeros(mod.SM_TAB_INTS, dtype=torch.int32, device="cuda")
    ext.smallm_route(ids, local, tab, 32, 32)
    torch.cuda.synchronize()
    loc, rows = _route_ref(ids.cpu(), 32, 32)
    assert local.flatten().tolist() == loc
    assert int(tab[32 * mod.SM_TABW]) == len(rows)


def _small_weights(mod, inter, hid, k1, seed=5):
    """32 local experts with tiny dims (same code path, fast): w13 [32, 2*inter, k1/2], w2 [32, hid, inter/2]."""
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(2 * inter, k1, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(hid, inter, generator=g, device="cuda", leading=(32,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": (gs13 * (1 + torch.rand(32, device="cuda", generator=g))).float().expand(32).contiguous(),
            "w2": w2, "w2_sf": w2_sf, "w2_gs": (gs2 * (1 + torch.rand(32, device="cuda", generator=g))).float().expand(32).contiguous(),
            "post_norm_w": (0.1 * torch.randn(hid, device="cuda", generator=g)).to(torch.bfloat16)}


def _expert_ref(x, ids, w, inter):
    """FP32 reference of the per-slot expert outputs [M, 8, H] (zero for non-local slots) with the BF16 rounding of `inter` like the kernels."""
    from nvfp4 import dequantize

    m = x.shape[0]
    y = torch.zeros((m, 8, w["w2"].shape[1]), dtype=torch.float32, device=x.device)
    for e in sorted(set(ids[(ids >= 0) & (ids < 32)].flatten().tolist())):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok].float() @ dequantize(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        mid = (torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]).to(torch.bfloat16).float()
        y[tok, slot] = mid @ dequantize(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T
    return y


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


@needs_cuda
@pytest.mark.parametrize("nw1,ks1,nz1,nw2,ks2,nz2", [(4, 4, 3, 4, 4, 3), (2, 2, 32, 8, 1, 1), (8, 1, 2, 2, 2, 32), (4, 2, 1, 4, 1, 1)])
@pytest.mark.parametrize("bs,dist", [(1, "normal"), (4, "normal"), (8, "normal"), (16, "normal"), (16, "skewed"), (12, "skewed")])
def test_experts_small_dims_split_k_configs(mod, ext, bs, dist, nw1, ks1, nz1, nw2, ks2, nz2):
    """K = 1024 for both GEMMs (inter = 1024, hid = 128): every split-K / warps / z-cap combination gives the same experts' outputs."""
    import reference

    inter, hid, k1 = 1024, 128, 1024
    w = _small_weights(mod, inter, hid, k1)
    g1, g2 = 2 * inter // 16, hid // 16
    if g1 % nw1 or g2 % nw2 or (k1 // ks1) % 256 or (inter // ks2) % 256:
        pytest.skip("shape does not divide")
    d = reference.generate_input(bs, "ep8", 300 + bs, dist=dist)
    x = d["hidden_states"][:, :k1].contiguous()
    old = mod.SMALLM_CFG
    mod.SMALLM_CFG = ((16, nw1, ks1, nz1, nw2, ks2, nz2),)
    try:
        sm = mod.prepare_smallm(w, ext=ext)
        cfg = mod.smallm_cfg(bs)
        y, local = mod.smallm_experts(x, d["topk_ids"], sm, 0, 32, cfg)
        y2, _ = mod.smallm_experts(x, d["topk_ids"], sm, 0, 32, cfg)
    finally:
        mod.SMALLM_CFG = old
    torch.cuda.synchronize()
    live = (local >= 0)  # slots of non-local experts are never written (the epilogue masks them)
    assert torch.equal(y.view(torch.int16)[live], y2.view(torch.int16)[live]), "not deterministic"
    ref = _expert_ref(x, local, w, inter)
    assert _rel(y.float()[live], ref[live]) < 3e-3
    assert not torch.isnan(y.float()[live]).any()
    assert int(sm["cnt"].abs().sum()) == 0, "fixup counters are not back at zero"


# --------------------------------------------------------------------------------------------- GPU: full size, generate_input, the judge's check
@pytest.fixture(scope="module")
def full_state(mod, ext):
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(5252)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=g, device="cuda", leading=(32,))
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
         "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device="cuda")).to(torch.bfloat16)}
    sm = mod.prepare_smallm(w, ext=ext)
    state = {"smallm": sm, "norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    yield w, state
    del sm, state, w
    torch.cuda.empty_cache()


@needs_cuda
@pytest.mark.parametrize("dist", ["normal", "skewed"])
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 12, 16])
def test_full_size_against_reference(mod, full_state, bs, dist):
    import reference

    w, state = full_state
    d = reference.generate_input(bs, "ep8", 7000 + bs, dist=dist)
    out = mod._kernel_smallm(d, state)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(d, out, w)
    assert ok, msg
    out2 = mod._kernel_smallm(d, state)
    assert torch.equal(out.view(torch.int16), out2.view(torch.int16)), "not deterministic"


@needs_cuda
def test_full_size_graph_capture(mod, full_state):
    """The small-M path captures in a CUDA graph (no sync, no value-dependent host logic) and the replay matches the eager call, on fresh inputs."""
    import reference

    w, state = full_state
    d = reference.generate_input(8, "ep8", 9100)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod._kernel_smallm(d, state)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod._kernel_smallm(d, state)
    for seed in (9101, 9102):
        d2 = reference.generate_input(8, "ep8", seed)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            d[k].copy_(d2[k])
        g.replay()
        torch.cuda.synchronize()
        ref_out = mod._kernel_smallm(d2, state)
        assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))
        ok, msg = reference.check_implementation(d2, out, w)
        assert ok, msg


# --------------------------------------------------------------------------------------------- measure_smallm_v1.py (MEASUREMENT-ONLY)
MEAS = ROOT / "kernels" / "muse_moe_w4a16" / "measure_smallm_v1.py"


@pytest.fixture(scope="module")
def mm():
    return _load("_measure_smallm_v1", MEAS)


def test_measure_file_rules():
    src = MEAS.read_text()
    compile(src, "measure_smallm_v1.py", "exec")
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for bad in (r"subprocess", r"urllib", r"requests", r"socket", r"import os", r"environ", r"tolist\(", r"\.item\(", r"\.cpu\(",
                r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src


def test_measure_kernel_path_is_v2b_and_smallm_block_is_v3():
    a, b, m = V2B.read_text(), KERNEL.read_text(), MEAS.read_text()
    lo = "import re\n"
    hi = "# ============================================================================================ plain-PyTorch path"
    assert a[a.index(lo):a.index(hi)] in m                                     # v2b's expert path, verbatim
    s0, s1 = "# ============================================================================================ SMALLM BEGIN", "# ============================================================================================ SMALLM END"
    assert b[b.index(s0):b.index(s1)] == m[m.index(s0):m.index(s1)]              # the small-M block, verbatim
    tail_a = a[a.index("def kernel_fn"):]
    assert m[m.index("def kernel_fn"):] == tail_a


def test_gen_input_equals_reference(mm):
    import reference

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for bs, dist, seed in ((1, "normal", 11), (8, "normal", 12), (16, "skewed", 13), (12, "skewed", 14)):
        a = mm._gen_input(bs, seed, dist, dev)
        b = reference.generate_input(bs, "ep8", seed, dist=dist, device=dev)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            assert torch.equal(a[k], b[k]), (bs, dist, k)
        assert a["config"] == b["config"]


def test_gemm_cfgs_are_valid(mm):
    for m in mm.MS:
        for k, groups in ((4096, 512), (4096, 256)):
            cf = mm.gemm_cfgs(m, k, groups)
            assert cf and all(groups % nw == 0 and (k // ks) % 256 == 0 and k // ks <= 2048 and nz <= min(32, 8 * m) for nw, ks, nz in cf)
    d = mm.smallm_cfg
    for m in mm.MS:
        c = d(m)
        assert (c["nw1"], c["ks1"], c["nz1"]) in mm.gemm_cfgs(m, 4096, 512) and (c["nw2"], c["ks2"], c["nz2"]) in mm.gemm_cfgs(m, 4096, 256)


class _FakeEnv:
    """Stubbed timing environment: no GPU. Time = a deterministic function of the config so the sweep has a known winner."""

    def __init__(self, mm, state):
        self.mm, self.state, self.t = mm, state, 0.0
        self.log = []

    def available(self):
        return True

    def clock(self):
        self.t += 0.01
        return self.t

    def banner(self):
        return "gpu=fake"

    def setup(self):
        pass

    def release(self):
        pass

    def prep(self, m, seed):
        return dict(m=m, seed=seed, data={"config": {"local_start": 0, "local_count": 32}}, info=f"pairs={m * 3} distinct=3 maxm=2")

    def err(self, ctx, y):
        return 1e-3

    def run_stages(self, fns, reps, name=None):
        if name is None:
            return 10.0 + len(fns), 1
        return 5.0 + hash((name, ctx_id(self.state))) % 7, 1

    def time_base(self, ctx, reps):
        return 100.0, 1


def ctx_id(state):
    return 0


def test_measure_control_flow_with_fake_env(mm, monkeypatch, capsys):
    monkeypatch.setattr(mm, "_small_stages", lambda data, state, c1, c2: ([("route", lambda: None), ("g1", lambda: None), ("g2", lambda: None), ("epi", lambda: None)], {}))
    monkeypatch.setattr(mm, "_marlin_stages", lambda data, state: ([("align", lambda: None), ("w13", lambda: None), ("act", lambda: None), ("w2", lambda: None), ("epi", lambda: None)], {}))
    state = {"smallm": {"dims": {"k1": 4096, "k2": 4096, "g1": 512, "g2": 256}}}
    env = _FakeEnv(mm, state)
    mm._measure(state, env)
    out = capsys.readouterr().out
    assert len(out.encode()) < 10_000
    for tag in ("G1 moes", "G2 moes", "B moes", "T moes", "C moes", "MEAS-SMALLM done"):
        assert tag in out, tag
    assert out.count("B moes") >= len(mm.MS)


@needs_cuda
def test_measure_real_env_on_dev_card(mm, mod, ext, full_state, monkeypatch, capsys):
    """The real _RealEnv (graph capture, read-only flush, stage closures, sweeps) with the small-M kernels via NVRTC; Marlin stubbed by the small path itself."""
    w, state = full_state
    nm = mm
    sm = state["smallm"]
    st = {"smallm": sm, "norm_w": state["norm_w"]}
    monkeypatch.setattr(nm, "HAVE_SGLANG", True)
    monkeypatch.setattr(nm, "kernel_fn", lambda data, s: mod._kernel_smallm(data, s))
    monkeypatch.setattr(nm, "_marlin_stages", lambda data, s: ([("align", lambda: None), ("w13", lambda: None)], {}))
    monkeypatch.setattr(nm, "MS", (1, 16))
    monkeypatch.setattr(nm, "SEEDS", (11, 12))
    monkeypatch.setattr(nm, "NW_SET", (4,))
    monkeypatch.setattr(nm, "KS_SET", (4, 8))
    monkeypatch.setattr(nm, "NZ_SET", (3, 12))
    monkeypatch.setattr(nm, "FINAL_REPS", 5)
    monkeypatch.setattr(nm, "SCREEN_REPS", 3)
    nm._measure(st)
    out = capsys.readouterr().out
    assert "G1 moes m=1" in out and "G2 moes m=16" in out and "T moes m=16 seed=12" in out, out
    assert "fail" not in out.replace("fail=0", ""), out
    errs = [float(x) for x in re.findall(r"err=([0-9.e+-]+)", out)]
    assert errs and max(errs) < 1e-2, out
