"""kernels/muse_moe_w4a16/sonnet_v4.py: the large-M (17..256 tokens) path on top of the small-M path of sonnet_v3b, and measure_largem_v1.py.

CPU: file rules, bucket / enable table, dispatch, prepare() fallbacks. GPU dev card (RTX 2060, sm_75, no nvcc): the CUDA device source is compiled with NVRTC
(tests/nvrtc_smallm.py; BF16 mma emulated with shuffles in the same fragment layout) and checked against reference.generate_input routing and
reference.check_implementation at bs 17..256 (the 6 GB card holds the 32 local experts: 0.9 GB + the repack). The sm_80/sm_90 builds are assembled by NVRTC's ptxas.

Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v4.py"""
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

KERNEL = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v4.py"
V3B = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v3b.py"
MEAS = ROOT / "kernels" / "muse_moe_w4a16" / "measure_largem_v1.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v4_moe", KERNEL)


# --------------------------------------------------------------------------------------------- CPU
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\(", r"urllib"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert "okc-diag HAVE_SGLANG" in src
    compile(src, "sonnet_v4.py", "exec")


def test_v3b_python_is_kept(mod):
    """The small-M config table of v3b (judge-tuned) and the Marlin region are unchanged in v4."""
    b3, b4 = V3B.read_text(), KERNEL.read_text()
    lo, hi = "import re\n", "# ============================================================================================ SMALLM BEGIN"
    assert b3[b3.index(lo):b3.index(hi)] in b4
    t, u = "SMALLM_CFG = (", "def smallm_cfg"
    assert b3[b3.index(t):b3.index(u)] == b4[b4.index(t):b4.index(u)]


def test_buckets_and_enable_table(mod):
    assert [mod.largem_bucket(m) for m in (1, 16, 17, 32, 33, 128, 129, 256, 257)] == [32, 32, 32, 32, 128, 128, 256, 256, None]
    assert mod.largem_cfg(16) is None and mod.largem_cfg(257) is None
    assert mod.largem_cfg(17) is mod.LARGEM_CFG[32] and mod.largem_cfg(100) is mod.LARGEM_CFG[128] and mod.largem_cfg(256) is mod.LARGEM_CFG[256]
    old = dict(mod.LARGEM_ENABLE)
    try:
        mod.LARGEM_ENABLE[128] = False
        assert mod.largem_cfg(100) is None and mod.largem_cfg(17) is not None
    finally:
        mod.LARGEM_ENABLE.update(old)
    for b, c in mod.LARGEM_CFG.items():
        assert c["nw1"] in (4, 8) and c["mr1"] in (1, 2) and c["nw2"] in (4, 8) and c["mr2"] in (1, 2)
        assert 4096 // c["ks1"] % 256 == 0 and 4096 // c["ks2"] % 256 == 0
        assert 4096 // c["ks1"] <= 2048 and c["ks1"] <= 8 and c["ks2"] <= 8 and 512 % (c["nw1"] * c["mr1"]) == 0 and 256 % (c["nw2"] * c["mr2"]) == 0
        smem = 1024 + 32 * (4096 // min(c["ks1"], c["ks2"]) + 8) * 2
        assert smem <= mod.LM_SMEM_MAX


def test_dispatch(mod, monkeypatch):
    monkeypatch.setattr(mod, "HAVE_SGLANG", True)
    monkeypatch.setattr(mod, "_kernel_smallm", lambda d, s: "S")
    monkeypatch.setattr(mod, "_kernel_largem", lambda d, s: "L")
    monkeypatch.setattr(mod, "_kernel_sglang", lambda d, s: "M")

    def data(m, start=0, count=32):
        return {"hidden_states": torch.zeros(m, 4), "config": {"local_start": start, "local_count": count}}

    st = {"smallm": object(), "largem": object(), "largem_ok": {32: True, 128: True, 256: True}}
    assert [mod.kernel_fn(data(m), st) for m in (1, 8, 16)] == ["S"] * 3
    assert [mod.kernel_fn(data(m), st) for m in (17, 32, 33, 128, 129, 256)] == ["L"] * 6
    assert [mod.kernel_fn(data(m), st) for m in (257, 512)] == ["M", "M"]
    assert mod.kernel_fn(data(100, 32, 32), st) == "M" and mod.kernel_fn(data(100, 0, 256), st) == "M"
    st2 = dict(st, largem_ok={32: True, 128: False, 256: True})
    assert [mod.kernel_fn(data(m), st2) for m in (32, 100, 200)] == ["L", "M", "L"]
    assert mod.kernel_fn(data(100), dict(st, largem=None)) == "M"
    assert mod.kernel_fn(data(100), {"smallm": None, "largem": None}) == "M"


def test_prepare_largem_state_fallbacks(mod, monkeypatch):
    monkeypatch.setattr(mod, "prepare_largem", lambda sm, flags=(): {"x": 1})
    monkeypatch.setattr(mod, "_largem_selfcheck", lambda st: {32: 1e-3, 128: 0.5, 256: 2e-3})
    st = {"smallm": object(), "norm_w": None}
    mod._prepare_largem_state(st, ())
    assert st["largem"] == {"x": 1} and st["largem_ok"] == {32: True, 128: False, 256: True}
    monkeypatch.setattr(mod, "_largem_selfcheck", lambda st: {32: 0.5, 128: float("nan"), 256: float("inf")})
    st = {"smallm": object()}
    mod._prepare_largem_state(st, ())
    assert st["largem"] is None and not any(st["largem_ok"].values())

    def boom(sm, flags=()):
        raise RuntimeError("no nvcc")
    monkeypatch.setattr(mod, "prepare_largem", boom)
    st = {"smallm": object()}
    mod._prepare_largem_state(st, ())
    assert st["largem"] is None
    st = {"smallm": None}
    mod._prepare_largem_state(st, ())
    assert st["largem"] is None


def test_zmax_bounds_the_chunk_count(mod):
    """sum_e ceil(m_e / 32) <= 8 M / 32 + 32 for every routing with m_e <= M, sum m_e <= 8 M (checked on adversarial splits)."""
    import random

    rnd = random.Random(1)
    for m in (17, 32, 100, 256):
        cap = 8 * m // 32 + 32
        for _ in range(200):
            left, ch = 8 * m, 0
            for e in range(32):
                k = min(m, rnd.choice([1, 33, 65, m, rnd.randint(1, m)]), left)
                left -= k
                ch += -(-k // 32)
            assert ch <= cap
    assert mod.LM_ZMAX >= 8 * 256 // 32 + 32


# --------------------------------------------------------------------------------------------- GPU: NVRTC on the dev card
@pytest.fixture(scope="module")
def exts(mod):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    return nv.NvrtcExt(mod._SMALLM_DEVICE_SRC), nv.NvrtcLargeExt(mod._SMALLM_DEVICE_SRC + mod._LARGEM_EXTRA_SRC)


@needs_cuda
@pytest.mark.parametrize("sm", ["sm_80", "sm_90"])
@pytest.mark.parametrize("flags", [(), ("-DSM_FP32_MUL=1",)])
def test_large_source_assembles_for_judge_arch(mod, sm, flags):
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    cubin, low = nv.sass_check(mod._SMALLM_DEVICE_SRC + mod._LARGEM_EXTRA_SRC, sm, extra=flags, names=nv.LARGE_NAMES)
    assert len(cubin) > 1000 and len(low) == len(nv.LARGE_NAMES)


def _route_ref(ids, lo, nloc, tc=32):
    flat = ids.flatten().tolist()
    loc = [g - lo if lo <= g < lo + nloc else -1 for g in flat]
    slots, chunks = [], []
    for e in range(nloc):
        s = [i for i, v in enumerate(loc) if v == e]
        for i in range(0, len(s), tc):
            chunks.append((e, min(tc, len(s) - i), len(slots) + i))
        slots += s
    return loc, slots, chunks


@needs_cuda
@pytest.mark.parametrize("dist", ["normal", "skewed"])
@pytest.mark.parametrize("bs", [1, 17, 32, 33, 128, 256])
def test_route_chunk_table(mod, exts, bs, dist):
    import reference

    d = reference.generate_input(bs, "ep8", 400 + bs, dist=dist)
    ids = d["topk_ids"].contiguous()
    local = torch.empty_like(ids)
    zmax = 8 * 256 // 32 + 32
    ctab = torch.full((4 + 4 * zmax,), 7777, dtype=torch.int32, device="cuda")
    slotl = torch.full((8 * 256,), 7777, dtype=torch.int32, device="cuda")
    exts[1].largem_route(ids, local, ctab, slotl, 0, 32)
    torch.cuda.synchronize()
    loc, slots, chunks = _route_ref(ids.cpu(), 0, 32)
    assert local.flatten().tolist() == loc
    ct = ctab.cpu()
    assert int(ct[0]) == len(chunks)
    for i, (e, mc, off) in enumerate(chunks):
        assert ct[4 + 4 * i:7 + 4 * i].tolist() == [e, mc, off]
    assert slotl[:len(slots)].tolist() == slots


def _small_weights(inter, hid, k1, seed=5):
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(seed)
    w13, w13_sf, gs13 = random_weights(2 * inter, k1, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(hid, inter, generator=g, device="cuda", leading=(32,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": (gs13 * (1 + torch.rand(32, device="cuda", generator=g))).float().expand(32).contiguous(),
            "w2": w2, "w2_sf": w2_sf, "w2_gs": (gs2 * (1 + torch.rand(32, device="cuda", generator=g))).float().expand(32).contiguous(),
            "post_norm_w": (0.1 * torch.randn(hid, device="cuda", generator=g)).to(torch.bfloat16)}


def _expert_ref(x, ids, w, inter):
    from nvfp4 import dequantize

    y = torch.zeros((x.shape[0], 8, w["w2"].shape[1]), dtype=torch.float32, device=x.device)
    for e in sorted(set(ids[(ids >= 0) & (ids < 32)].flatten().tolist())):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        h = x[tok].float() @ dequantize(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]).T
        mid = (torch.nn.functional.silu(h[:, :inter]) * h[:, inter:]).to(torch.bfloat16).float()
        y[tok, slot] = mid @ dequantize(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]).T
    return y


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


@needs_cuda
@pytest.mark.parametrize("nw,mr,ks1,nz1,ks2,nz2", [(4, 2, 2, 3, 2, 3), (4, 1, 4, 64, 4, 64), (8, 1, 2, 1, 4, 2), (8, 2, 4, 2, 2, 5)])
@pytest.mark.parametrize("bs,dist", [(17, "normal"), (33, "normal"), (64, "skewed"), (100, "skewed"), (160, "normal")])
def test_experts_small_dims_all_configs(mod, exts, bs, dist, nw, mr, ks1, nz1, ks2, nz2):
    """K = 1024 for both GEMMs (inter = 1024, hid = 256): every nw / mr / split-K / z-cap combination (z loops, chunked hot experts) gives the experts' outputs."""
    import reference

    inter, hid, k1 = 1024, 256, 1024
    rows = nw * mr
    if (2 * inter // 16) % rows or (hid // 16) % rows:
        pytest.skip("shape does not divide")
    w = _small_weights(inter, hid, k1)
    d = reference.generate_input(bs, "ep8", 600 + bs, dist=dist)
    x = d["hidden_states"][:, :k1].contiguous()
    old = mod.SMALLM_CFG
    mod.SMALLM_CFG = ((16, 4, 4, 3, 4, 4, 3),)
    try:
        sm = mod.prepare_smallm(w, ext=exts[0])
    finally:
        mod.SMALLM_CFG = old
    cfg = dict(nw1=nw, mr1=mr, ks1=ks1, nz1=nz1, nw2=nw, mr2=mr, ks2=ks2, nz2=nz2)
    lm = mod.prepare_largem(sm, ext=exts[1], max_m=256, cfgs={32: cfg})
    y, local = mod.largem_experts(x, d["topk_ids"], sm, lm, 0, 32, cfg)
    y2, _ = mod.largem_experts(x, d["topk_ids"], sm, lm, 0, 32, cfg)
    torch.cuda.synchronize()
    live = local >= 0
    assert torch.equal(y.view(torch.int16)[live], y2.view(torch.int16)[live]), "not deterministic"
    ref = _expert_ref(x, local, w, inter)
    assert _rel(y.float()[live], ref[live]) < 3e-3
    assert not torch.isnan(y.float()[live]).any()
    assert int(lm["cnt"].abs().sum()) == 0, "fixup counters are not back at zero"


# --------------------------------------------------------------------------------------------- GPU: full size, generate_input, the judge's check
@pytest.fixture(scope="module")
def full_state(mod, exts):
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(5252)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=g, device="cuda", leading=(32,))
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
         "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device="cuda")).to(torch.bfloat16)}
    old_small = mod.SMALLM_CFG
    mod.SMALLM_CFG = ((16, 4, 4, 3, 4, 4, 3),)  # the 2060 has 64 KB of smem per block (v3b's judge-tuned table needs 67 KB)
    sm = mod.prepare_smallm(w, ext=exts[0])
    # large-M: K slices of 512 (8 splits)
    cfgs = {b: dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24) for b in (32, 128, 256)}
    old = dict(mod.LARGEM_CFG)
    mod.LARGEM_CFG.update(cfgs)
    lm = mod.prepare_largem(sm, ext=exts[1])
    state = {"smallm": sm, "largem": lm, "largem_ok": {32: True, 128: True, 256: True}, "norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    yield w, state
    mod.SMALLM_CFG = old_small
    mod.LARGEM_CFG.clear()
    mod.LARGEM_CFG.update(old)
    del sm, lm, state, w
    torch.cuda.empty_cache()


@needs_cuda
@pytest.mark.parametrize("dist", ["normal", "skewed"])
@pytest.mark.parametrize("bs", [17, 32, 33, 128, 256])
def test_full_size_against_reference(mod, full_state, bs, dist):
    import reference

    w, state = full_state
    d = reference.generate_input(bs, "ep8", 8000 + bs, dist=dist)
    out = mod._kernel_largem(d, state)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(d, out, w)
    assert ok, msg
    out2 = mod._kernel_largem(d, state)
    assert torch.equal(out.view(torch.int16), out2.view(torch.int16)), "not deterministic"


@needs_cuda
def test_full_size_graph_capture(mod, full_state):
    import reference

    w, state = full_state
    d = reference.generate_input(64, "ep8", 9300)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod._kernel_largem(d, state)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod._kernel_largem(d, state)
    for seed in (9301, 9302):
        d2 = reference.generate_input(64, "ep8", seed)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            d[k].copy_(d2[k])
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), mod._kernel_largem(d2, state).view(torch.int16))
        ok, msg = reference.check_implementation(d2, out, w)
        assert ok, msg


@needs_cuda
def test_small_path_still_correct_in_v4(mod, full_state):
    import reference

    w, state = full_state
    for bs in (1, 8, 16):
        d = reference.generate_input(bs, "ep8", 9500 + bs)
        ok, msg = reference.check_implementation(d, mod._kernel_smallm(d, state), w)
        assert ok, msg


# --------------------------------------------------------------------------------------------- measure_largem_v1.py (MEASUREMENT-ONLY)
@pytest.fixture(scope="module")
def mm():
    return _load("_measure_largem_v1", MEAS)


def test_measure_file_rules():
    src = MEAS.read_text()
    compile(src, "measure_largem_v1.py", "exec")
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for bad in (r"subprocess", r"urllib", r"requests", r"socket", r"import os", r"environ", r"tolist\(", r"\.item\(", r"\.cpu\(",
                r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src


def test_measure_kernel_fn_is_v3b_and_blocks_are_v4():
    a, b, m = V3B.read_text(), KERNEL.read_text(), MEAS.read_text()
    assert m[m.index("def kernel_fn"):] == a[a.index("def kernel_fn"):]
    for s0, s1 in (("# ============================================================================================ SMALLM BEGIN",
                    "# ============================================================================================ SMALLM END"),
                   ("# ============================================================================================ LARGEM BEGIN",
                    "# ============================================================================================ LARGEM END")):
        assert b[b.index(s0):b.index(s1)] == m[m.index(s0):m.index(s1)]


def test_gen_input_equals_reference(mm):
    import reference

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for bs, dist, seed in ((32, "normal", 11), (128, "normal", 12), (256, "skewed", 13), (100, "skewed", 14)):
        a = mm._gen_input(bs, seed, dist, dev)
        b = reference.generate_input(bs, "ep8", seed, dist=dist, device=dev)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            assert torch.equal(a[k], b[k]), (bs, dist, k)
        assert a["config"] == b["config"]


def test_gemm_cfgs_are_valid(mm):
    for m in mm.MS:
        zb = min(mm.LM_ZMAX, 8 * m // 32 + 32)
        for k, groups in ((4096, 512), (4096, 256)):
            cf = mm.gemm_cfgs_l(m, k, groups, zb)
            assert cf and all(groups % (nw * mr) == 0 and (k // ks) % 256 == 0 and k // ks <= 2048 and nz <= zb for nw, mr, ks, nz in cf)
        d1, d2 = mm._default_cfgs(m)
        assert d1[:3] in [c[:3] for c in mm.gemm_cfgs_l(m, 4096, 512, 96)] and d2[:3] in [c[:3] for c in mm.gemm_cfgs_l(m, 4096, 256, 96)]


class _FakeEnv:
    def __init__(self, state):
        self.state, self.t = state, 0.0

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
        return dict(m=m, seed=seed, data={"config": {"local_start": 0, "local_count": 32}}, info=f"pairs={m * 3} distinct=3 maxm=2 chunks=5")

    def err(self, ctx, y):
        return 1e-3

    def run_stages(self, fns, reps, name=None):
        return (10.0 + len(fns), 1) if name is None else (5.0 + hash(name) % 7, 1)

    def time_base(self, ctx, reps):
        return 100.0, 1


def test_measure_control_flow_with_fake_env(mm, monkeypatch, capsys):
    monkeypatch.setattr(mm, "_large_stages", lambda data, state, c1, c2: ([("route", lambda: None), ("g1", lambda: None), ("g2", lambda: None), ("epi", lambda: None)], {}))
    monkeypatch.setattr(mm, "_marlin_stages", lambda data, state: ([("align", lambda: None), ("w13", lambda: None), ("act", lambda: None), ("w2", lambda: None), ("epi", lambda: None)], {}))
    state = {"smallm": {"dims": {"k1": 4096, "k2": 4096, "g1": 512, "g2": 256}}, "largem": {"zmax": 96}}
    mm._measure(state, _FakeEnv(state), t_start=0.0)
    out = capsys.readouterr().out
    assert len(out.encode()) < 12_000
    for tag in ("G1 moel", "G2 moel", "B moel", "T moel", "C moel", "MEAS-LARGEM done"):
        assert tag in out, tag
    assert out.count("B moel") >= len(mm.MS)


@needs_cuda
def test_measure_real_env_on_dev_card(mm, mod, full_state, monkeypatch, capsys):
    """The real _RealEnv (graph capture, read-only flush, stage closures, sweeps) with the large-M kernels via NVRTC; Marlin stubbed by the large path itself."""
    w, state = full_state
    st = {"smallm": state["smallm"], "largem": state["largem"], "norm_w": state["norm_w"]}
    monkeypatch.setattr(mm, "HAVE_SGLANG", True)
    for b_, c_ in mod.LARGEM_CFG.items():  # the dev-card cfg of the fixture (64 KB smem)
        monkeypatch.setitem(mm.LARGEM_CFG, b_, c_)

    def stub_kernel_fn(data, s):  # stands in for Marlin: the large pipeline itself
        m = data["hidden_states"].shape[0]
        y, local = mm.largem_experts(data["hidden_states"], data["topk_ids"], s["smallm"], s["largem"], 0, 32, mm.LARGEM_CFG[mm.largem_bucket(m)])
        return mm._epilogue_for(y, data["topk_weights"].to(torch.float32).contiguous(), local, s, m)

    monkeypatch.setattr(mm, "kernel_fn", stub_kernel_fn)
    monkeypatch.setattr(mm, "fused_post_expert_norm_weighted_sum", lambda y, tw, norm_weight, eps, weight_first, slot_expert_ids:
                        mm.epilogue_triton(y, tw, slot_expert_ids, norm_weight), raising=False)
    monkeypatch.setattr(mm, "_marlin_stages", lambda data, s: ([("align", lambda: None), ("w13", lambda: None)], {}))
    monkeypatch.setattr(mm, "MS", (32, 128))
    monkeypatch.setattr(mm, "SEEDS", (11, 12))
    monkeypatch.setattr(mm, "VARIANTS", ((4, 2), (8, 1)))
    monkeypatch.setattr(mm, "KS_SET", (8,))
    monkeypatch.setattr(mm, "NZ_SET", (8, 32))
    monkeypatch.setattr(mm, "FINAL_REPS", 5)
    monkeypatch.setattr(mm, "SCREEN_REPS", 3)
    monkeypatch.setattr(mm, "ALTERNATIONS", 1)
    mm._measure(st)
    out = capsys.readouterr().out
    assert "G1 moel m=32" in out and "G2 moel m=128" in out and "T moel m=128 seed=12" in out, out
    assert "fail" not in out.replace("fail=0", ""), out
    errs = [float(x) for x in re.findall(r"err=([0-9.e+-]+)", out)]
    assert errs and max(errs) < 1e-2, out
    assert re.search(r"chunks=\d+", out)
