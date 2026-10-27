"""kernels/muse_moe_w4a16/sonnet_v5.py: the unified w4_moe5_kernel (C21 "Alt A") on top of sonnet_v4, and measure_altA_v1.py.

CPU: file rules, v4 region verbatim, config table / buckets / instantiations, dispatch, prepare() fallbacks, host-source generation.
GPU dev card (RTX 2060, sm_75, no nvcc): NVRTC builds of v4's and v5's device sources (BF16 mma emulated with shuffles in the same fragment layout);
V1 = v5 BITWISE v4 on reference.generate_input routing for M in {1, 4, 8, 16, 32, 33, 128, 256} x 3 seeds at the same K split (the 2060 holds the 32 local
experts; its 64 KB smem per block needs Kc <= 1024 for 16/32-row tiles, so the dev-card tables use ks 4 (M <= 16) / 8 (M > 16) for BOTH v4 and v5);
every scheduling flag (LB, NB, flattened z, PDL off) stays bitwise; the debug twins (STAMPS / NOCOMPUTE / NOSTAGE) run and report; the judge's check
passes; graph capture; the sm_80 / sm_90 assembly of every instantiation (regs / spills printed).

Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v5.py"""
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "problems" / "muse_moe_w4a16"))

KERNEL = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v5.py"
V4 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v4.py"
MEAS = ROOT / "kernels" / "muse_moe_w4a16" / "measure_altA_v1.py"
CUOBJDUMP = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# The measurement file's instantiation list (12 template instantiations, fable-09 D4 S1 budget); tests/test_moe_sonnet_v5 checks it equals the file's.
MEAS_INSTANCES = (
    (True, 4, 2, 0, 0), (False, 4, 2, 0, 0),      # base: v4's geometry, no launch-bounds hint (LB0); NB / Kc / nz runtime
    (True, 4, 2, 3, 0), (False, 4, 2, 3, 0),      # Alt A occupancy (LB3)
    (True, 4, 2, 3, 1), (False, 4, 2, 3, 1),      # Alt A + STAMPS
    (True, 4, 2, 0, 2), (False, 4, 2, 0, 2),      # NOCOMPUTE twin (LB0)
    (True, 4, 2, 0, 4), (False, 4, 2, 0, 4),      # NOSTAGE twin (LB0)
    (True, 8, 1, 2, 0), (False, 8, 1, 2, 0),      # small-M geometry (LB2)
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v5_moe", KERNEL)


@pytest.fixture(scope="module")
def v4():
    return _load("_sonnet_v4_moe_for_v5", V4)


# --------------------------------------------------------------------------------------------- CPU
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\(", r"urllib"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert "okc-diag HAVE_SGLANG" in src
    assert "PREDICTION" in src and "DECISION RULE" in src and "NOCOMPUTE >= 0.85" in src
    compile(src, "sonnet_v5.py", "exec")


def test_v4_region_is_verbatim():
    """Everything of v4 from `import re` up to its entry points (Marlin path, small-M + large-M blocks, self-checks) is in v5 unchanged."""
    a, b = V4.read_text(), KERNEL.read_text()
    lo, hi = "import re\n", "# ============================================================================================ entry points"
    assert a[a.index(lo):a.index(hi)] in b
    assert b.count("# ============================================================================================ MOE5 BEGIN") == 1


def test_cfg_table_buckets_and_instances(mod, v4):
    assert [mod.moe5_bucket(m) for m in (1, 2, 3, 4, 8, 9, 12, 16, 17, 32, 33, 128, 129, 256, 257)] == \
        [1, 2, 4, 4, 8, 12, 12, 16, 32, 32, 128, 128, 256, 256, None]
    assert mod.moe5_cfg(257) is None and mod.moe5_cfg(17) == mod.moe5_cfg(32) and mod.moe5_cfg(100) == mod.moe5_cfg(128)
    old = dict(mod.MOE5_ENABLE)
    try:
        mod.MOE5_ENABLE[128] = False
        assert mod.moe5_cfg(100) is None and mod.moe5_cfg(256) is not None
    finally:
        mod.MOE5_ENABLE.update(old)
    inst = set(mod.MOE5_INSTANCES)
    assert len(inst) == 5 and all(dbg == 0 for _f, _w, _r, _lb, dbg in inst)
    for mx, c1, c2 in mod.MOE5_CFG:
        for fuse, c, groups, k in ((True, c1, 512, 4096), (False, c2, 256, 4096)):
            nw, mr, ks, nz, nb, lb = c
            assert (fuse, nw, mr, lb, 0) in inst, (mx, c)
            assert groups % (nw * mr) == 0 and (groups // (nw * mr)) % nb == 0 and k % ks == 0 and (k // ks) % 256 == 0 and k // ks <= 2048 and ks <= 8
            assert mr == 2 or mx <= mod.SM_MAX_M                      # mr 1 = small-M geometry (chunks <= 16 tokens)
            assert 1024 + mod.moe5_rows(mx) * (k // ks + 8) * 2 <= mod.LM_SMEM_MAX
            if lb:                                                    # the register cap of the hint is what the sm_90 build reports (test below)
                assert 65536 // (nw * 32 * lb) >= 128
        # the K split is v4's for the bucket => the split-K sums are v4's (bitwise contract)
        if mx <= mod.SM_MAX_M:
            s = v4.smallm_cfg(mx)
            assert (c1[2], c2[2]) == (s["ks1"], s["ks2"]) and (c1[3], c2[3]) == (s["nz1"], s["nz2"]), mx
        else:
            l = v4.LARGEM_CFG[mx]
            assert (c1[2], c2[2]) == (l["ks1"], l["ks2"]) and (c1[3], c2[3]) == (l["nz1"], l["nz2"]), mx
    assert set(mod.moe5_bucket(m) for m in mod.MOE5_SELFCHECK_MS) == {mx for mx, _a, _b in mod.MOE5_CFG}


def test_zbound_and_rows(mod):
    lm = {"zmax": mod.LM_ZMAX}
    assert [mod.moe5_zbound(m, lm) for m in (1, 2, 4, 8, 16, 32, 128, 256)] == [8, 16, 32, 32, 32, 32, 64, 96]
    assert [mod.moe5_rows(m) for m in (1, 8, 9, 16, 17, 256)] == [8, 8, 16, 16, 32, 32]


def test_host_source_generation(mod):
    src = mod.moe5_host_src(mod.MOE5_INSTANCES)
    for f, w, r, lb, dbg in mod.MOE5_INSTANCES:
        assert f"M5_CASE({'true' if f else 'false'}, {w}, {r}, {lb}, {dbg})" in src
        assert f"w4_moe5_kernel<{'true' if f else 'false'}, {w}, {r}, {lb}, {dbg}>, cudaFuncAttributeMaxDynamicSharedMemorySize" in src
    assert "@CASES@" not in src and "@INITS@" not in src
    assert mod.moe5_instance_names([(True, 4, 2, 3, 1)]) == ["w4_moe5_kernel<true,4,2,3,1>"]
    dev = mod.moe5_device_src()
    assert dev.startswith(mod._SMALLM_DEVICE_SRC) and mod._LARGEM_EXTRA_SRC in dev and "w4_moe5_kernel" in dev
    for name in ("moe5_gemm", "moe5_route", "moe5_init"):
        assert name in mod._MOE5_CPP_SRC


def test_dispatch(mod, monkeypatch):
    monkeypatch.setattr(mod, "HAVE_SGLANG", True)
    monkeypatch.setattr(mod, "_kernel_moe5", lambda d, s: "5")
    monkeypatch.setattr(mod, "_kernel_smallm", lambda d, s: "S")
    monkeypatch.setattr(mod, "_kernel_largem", lambda d, s: "L")
    monkeypatch.setattr(mod, "_kernel_sglang", lambda d, s: "M")

    def data(m, start=0, count=32):
        return {"hidden_states": torch.zeros(m, 4), "config": {"local_start": start, "local_count": count}}

    ok = {mx: True for mx, _a, _b in mod.MOE5_CFG}
    st = {"smallm": object(), "largem": object(), "largem_ok": {32: True, 128: True, 256: True}, "moe5": object(), "moe5_ok": ok}
    assert [mod.kernel_fn(data(m), st) for m in (1, 8, 16, 17, 32, 33, 128, 129, 256)] == ["5"] * 9
    assert [mod.kernel_fn(data(m), st) for m in (257, 512)] == ["M", "M"]
    assert mod.kernel_fn(data(100, 32, 32), st) == "M" and mod.kernel_fn(data(100, 0, 256), st) == "M"
    st2 = dict(st, moe5_ok={**ok, 128: False, 8: False})
    assert [mod.kernel_fn(data(m), st2) for m in (8, 16, 100, 200)] == ["S", "5", "L", "5"]
    assert mod.kernel_fn(data(100), dict(st, moe5=None)) == "L"
    assert mod.kernel_fn(data(100), dict(st, moe5=None, largem=None)) == "M"
    assert mod.kernel_fn(data(8), dict(st, moe5=None, smallm=None)) == "M"
    assert mod.kernel_fn(data(8), dict(st, largem=None)) == "S"      # v5 needs v4's large-M scratch


def test_prepare_moe5_state_fallbacks(mod, monkeypatch):
    monkeypatch.setattr(mod, "prepare_moe5", lambda sm, lm, ext=None, flags=(): {"x": 1})
    monkeypatch.setattr(mod, "_moe5_selfcheck", lambda st: {1: True, 2: True, 4: False, 8: True, 12: True, 16: True, 32: True, 128: False, 256: True})
    st = {"smallm": object(), "largem": object(), "norm_w": None}
    mod._prepare_moe5_state(st, ())
    assert st["moe5"] == {"x": 1} and st["moe5_ok"][4] is False and st["moe5_ok"][128] is False and st["moe5_ok"][256] is True
    monkeypatch.setattr(mod, "_moe5_selfcheck", lambda st: {b: False for b in (1, 2, 4, 8, 12, 16, 32, 128, 256)})
    st = {"smallm": object(), "largem": object()}
    mod._prepare_moe5_state(st, ())
    assert st["moe5"] is None and not any(st["moe5_ok"].values())

    def boom(sm, lm, ext=None, flags=()):
        raise RuntimeError("no nvcc")
    monkeypatch.setattr(mod, "prepare_moe5", boom)
    st = {"smallm": object(), "largem": object()}
    mod._prepare_moe5_state(st, ())
    assert st["moe5"] is None
    for st in ({"smallm": None, "largem": object()}, {"smallm": object(), "largem": None}):
        mod._prepare_moe5_state(st, ())
        assert st["moe5"] is None and st["moe5_ok"] == {}


# --------------------------------------------------------------------------------------------- GPU: NVRTC on the dev card
@pytest.fixture(scope="module")
def nv():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    return nv


def _usage(nv, src, arch, names):
    """{name: (regs, stack bytes, local bytes)} from cuobjdump --dump-resource-usage of the NVRTC cubin for `arch`."""
    cubin, low = nv.compile_device(src, arch, names=names)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        r = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(p)], capture_output=True, text=True)
    use = {m.group(1): (int(m.group(2)), int(m.group(3)), int(m.group(5))) for m in
           re.finditer(r"Function (\S+?):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", r.stdout)}
    return {n: use.get(low[n].rstrip("\x00")) for n in names}


@needs_cuda
@pytest.mark.parametrize("sm", ["sm_80", "sm_90"])
def test_source_assembles_for_judge_arch(mod, nv, sm, capsys):
    """Every instantiation of the candidate and of the measurement file assembles for the judge's GPU; regs within the launch-bounds cap, no local memory."""
    inst = sorted(set(mod.MOE5_INSTANCES) | set(MEAS_INSTANCES))
    names = mod.moe5_instance_names(inst) + ["moe5_route_small_kernel", "largem_route_kernel"]
    use = _usage(nv, mod.moe5_device_src(), sm, names)
    with capsys.disabled():
        print(f"\n{sm} regs/stack/local per instantiation:")
    for (f, w, r, lb, dbg), n in zip(inst, names):
        regs, stack, local = use[n]
        with capsys.disabled():
            print(f"  {n:34s} regs={regs} stack={stack} local={local}")
        assert local == 0, n
        if lb:
            assert regs <= 65536 // (w * 32 * lb), (n, regs)
        assert stack <= 64, (n, stack)   # a few bytes of save/restore spills at the cap are fine; a spilling k-loop is not


@pytest.fixture(scope="module")
def exts(mod, v4, nv):
    inst = sorted(set(mod.MOE5_INSTANCES) | set(MEAS_INSTANCES))
    return (nv.NvrtcExt(v4._SMALLM_DEVICE_SRC), nv.NvrtcLargeExt(v4._SMALLM_DEVICE_SRC + v4._LARGEM_EXTRA_SRC),
            nv.NvrtcMoe5Ext(mod.moe5_device_src(), mod.moe5_instance_names(inst)))


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
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 13, 16])
def test_route_small_equals_largem_route(mod, exts, bs, dist):
    """The 128-thread route kernel writes the chunk table + slot list of largem_route_kernel (one chunk per active expert, experts and slots ascending)."""
    import reference

    e5 = exts[2]
    for seed in range(3):
        d = reference.generate_input(bs, "ep8", 700 + 10 * bs + seed, dist=dist)
        ids = d["topk_ids"].contiguous()
        zmax = mod.LM_ZMAX
        outs = []
        for small in (True, False):
            local = torch.full_like(ids, 7777)
            ctab = torch.full((4 + 4 * zmax,), 7777, dtype=torch.int32, device="cuda")
            slotl = torch.full((8 * 256,), 7777, dtype=torch.int32, device="cuda")
            e5.moe5_route(ids, local, ctab, slotl, 0, 32, small)
            torch.cuda.synchronize()
            outs.append((local, ctab, slotl))
        loc, slots, chunks = _route_ref(ids.cpu(), 0, 32)
        (l_s, c_s, s_s), (l_l, c_l, s_l) = outs
        n = len(chunks)
        assert l_s.flatten().tolist() == loc and torch.equal(l_s, l_l)
        assert int(c_s[0]) == n == int(c_l[0])
        assert torch.equal(c_s[4:4 + 4 * n], c_l[4:4 + 4 * n]) and [tuple(c_s[4 + 4 * i:7 + 4 * i].tolist()) for i in range(n)] == chunks
        assert torch.equal(s_s[:len(slots)], s_l[:len(slots)]) and s_s[:len(slots)].tolist() == slots


# dev-card launch tables (64 KB smem per block): ks 4 at M <= 16 (16-row tiles of Kc 1024), ks 8 at M > 16 (32-row tiles of Kc 512), nw 8 for both small GEMMs
DEV_SMALLM_CFG = ((16, 8, 4, 8, 8, 4, 8),)
DEV_LARGEM_CFG = {b: dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24) for b in (32, 128, 256)}
DEV_MOE5_CFG = tuple((mx, (8, 1, 4, 8, 1, 2), (8, 1, 4, 8, 1, 2)) if mx <= 16 else (mx, (4, 2, 8, 24, 4, 3), (4, 2, 8, 24, 2, 3))
                     for mx, _a, _b in ((1, 0, 0), (2, 0, 0), (4, 0, 0), (8, 0, 0), (12, 0, 0), (16, 0, 0), (32, 0, 0), (128, 0, 0), (256, 0, 0)))


def _full_weights():
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(5252)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=g, device="cuda", leading=(32,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
            "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device="cuda")).to(torch.bfloat16)}


@pytest.fixture(scope="module")
def full_state(mod, v4, exts):
    """v4's state (packed weights + scratch, NVRTC extensions) with the dev-card tables, and v5's extension on top; both modules see the same tables."""
    w = _full_weights()
    saved = (v4.SMALLM_CFG, dict(v4.LARGEM_CFG), mod.SMALLM_CFG, dict(mod.LARGEM_CFG), mod.MOE5_CFG)
    v4.SMALLM_CFG = mod.SMALLM_CFG = DEV_SMALLM_CFG
    for m_ in (v4, mod):
        m_.LARGEM_CFG.clear()
        m_.LARGEM_CFG.update(DEV_LARGEM_CFG)
    mod.MOE5_CFG = DEV_MOE5_CFG
    sm = v4.prepare_smallm(w, ext=exts[0])
    lm = v4.prepare_largem(sm, ext=exts[1])
    m5 = mod.prepare_moe5(sm, lm, ext=exts[2])
    state = {"smallm": sm, "largem": lm, "largem_ok": {32: True, 128: True, 256: True}, "moe5": m5,
             "moe5_ok": {mx: True for mx, _a, _b in DEV_MOE5_CFG}, "norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    yield w, state
    v4.SMALLM_CFG, l4, mod.SMALLM_CFG, l5, mod.MOE5_CFG = saved
    for m_, l_ in ((v4, l4), (mod, l5)):
        m_.LARGEM_CFG.clear()
        m_.LARGEM_CFG.update(l_)
    del sm, lm, m5, state, w
    torch.cuda.empty_cache()


def _v4_experts(v4, state, d):
    x, ids = d["hidden_states"], d["topk_ids"]
    m = x.shape[0]
    if m <= v4.SM_MAX_M:
        return v4.smallm_experts(x, ids, state["smallm"], 0, 32, v4.smallm_cfg(m))
    return v4.largem_experts(x, ids, state["smallm"], state["largem"], 0, 32, v4.LARGEM_CFG[v4.largem_bucket(m)])


def _live_rows(y, local):
    return y.reshape(-1, y.shape[-1]).view(torch.int16)[local.reshape(-1) >= 0]


@needs_cuda
@pytest.mark.parametrize("bs", [1, 4, 8, 16, 32, 33, 128, 256])
def test_v1_bitwise_vs_v4(mod, v4, full_state, bs):
    """V1: the unified kernel's expert outputs equal v4's bit for bit (3 generate_input seeds, normal + skewed), and the judge's check passes."""
    import reference

    w, state = full_state
    c1, c2 = mod.moe5_cfg(bs)
    for i, dist in enumerate(("normal", "normal", "skewed")):
        d = reference.generate_input(bs, "ep8", 8100 + bs + i, dist=dist)
        y4, l4 = _v4_experts(v4, state, d)
        y5, l5 = mod.moe5_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe5"], 0, 32, (c1, c2))
        torch.cuda.synchronize()
        assert torch.equal(l4, l5)
        assert torch.equal(_live_rows(y4, l4), _live_rows(y5, l5)), (bs, dist)
        assert int(state["largem"]["cnt"].abs().sum()) == 0 and int(state["smallm"]["cnt"].abs().sum()) == 0
        if i == 0:
            out = mod._kernel_moe5(d, dict(state))
            ok, msg = reference.check_implementation(d, out, w)
            assert ok, msg
            out2 = mod._kernel_moe5(d, dict(state))
            assert torch.equal(out.view(torch.int16), out2.view(torch.int16)), "not deterministic"


@needs_cuda
@pytest.mark.parametrize("bs", [16, 33, 256])
def test_scheduling_flags_stay_bitwise(mod, v4, full_state, bs):
    """LB 0 / 3, NB 1 / 2 / 4, z flattened to the shape bound, PDL off: scheduling only, every output bit equals v4's."""
    import reference

    w, state = full_state
    sm, lm, m5 = state["smallm"], state["largem"], state["moe5"]
    d = reference.generate_input(bs, "ep8", 8300 + bs)
    y4, l4 = _v4_experts(v4, state, d)
    ref = _live_rows(y4, l4)
    zb = mod.moe5_zbound(bs, lm)
    if bs <= 16:
        combos = [((8, 1, 4, zb, 1, 2), (8, 1, 4, zb, 1, 2)), ((8, 1, 4, 2, 1, 2), (8, 1, 4, 3, 1, 2)), ((8, 1, 4, 8, 2, 2), (8, 1, 4, 8, 2, 2))]
    else:
        combos = [((4, 2, 8, 24, nb1, lb), (4, 2, 8, 24, nb2, lb)) for lb in (0, 3) for nb1, nb2 in ((1, 1), (2, 2), (4, 4))]
        combos += [((4, 2, 8, zb, 4, 3), (4, 2, 8, zb, 2, 3)), ((4, 2, 8, 8, 1, 0), (4, 2, 8, 8, 1, 0))]
    for c1, c2 in combos:
        for pdl in ((True, True), (True, False)):
            y5, l5 = mod.moe5_experts(d["hidden_states"], d["topk_ids"], sm, lm, m5, 0, 32, (c1, c2), pdl=pdl)
            torch.cuda.synchronize()
            assert torch.equal(l4, l5) and torch.equal(ref, _live_rows(y5, l5)), (c1, c2, pdl)
    assert int(lm["cnt"].abs().sum()) == 0


def _dbg_buffer(mod, bs, cfg, nw):
    """A dbg buffer large enough for the stamps / occupancy / sink of the GEMM with the most CTAs."""
    ncta = max(512 // (c[0] * c[1]) // c[4] * c[2] * c[3] for c in cfg)
    return torch.zeros(mod.M5_OCC + ncta * 8 + ncta * 8, dtype=torch.int64, device="cuda"), ncta


@needs_cuda
@pytest.mark.parametrize("bs", [8, 33])
def test_debug_twins_run_and_report(mod, v4, full_state, bs):
    """STAMPS: t0..t7 monotone per CTA, per-SM concurrency counters back to 0 with max >= 1; NOCOMPUTE: sink words stored, output is not v4's;
    NOSTAGE: runs. The twins exist for the measurement file; the candidate never instantiates them."""
    import reference

    w, state = full_state
    sm, lm, m5 = state["smallm"], state["largem"], state["moe5"]
    d = reference.generate_input(bs, "ep8", 8400 + bs)
    y4, l4 = _v4_experts(v4, state, d)
    if bs <= 16:
        pytest.skip("the debug twins are instantiated for the large-M geometry only (12-instantiation budget)")
    cfg = ((4, 2, 8, 24, 4, 3), (4, 2, 8, 24, 2, 3))
    cfg0 = ((4, 2, 8, 24, 1, 0), (4, 2, 8, 24, 1, 0))
    dbg, _n = _dbg_buffer(mod, bs, cfg0, 4)        # sized for the largest grid of the variants below (NB 1): the kernels index it by CTA
    ncta = max(512 // 8 // c[4] * c[2] * c[3] for c in cfg)
    # STAMPS on both GEMMs
    y5, l5 = mod.moe5_experts(d["hidden_states"], d["topk_ids"], sm, lm, m5, 0, 32, cfg, dbg=dbg, dbg_flags=(1, 1))
    torch.cuda.synchronize()
    assert torch.equal(_live_rows(y4, l4), _live_rows(y5, l5))
    occ = dbg[: mod.M5_OCC].view(torch.int32)
    assert int(occ[:256].abs().sum()) == 0 and int(occ[256:512].max()) >= 1   # cur back to 0 after both kernels; max >= 1 (GEMM2's stamps overwrite GEMM1's)
    st = dbg[mod.M5_OCC: mod.M5_OCC + ncta * 8].view(-1, 8)
    active = st[:, 0] > 0
    assert int(active.sum()) > 0
    s = st[active]
    assert bool((s[:, 1] >= s[:, 0]).all() and (s[:, 2] >= s[:, 1]).all() and (s[:, 3] >= s[:, 2]).all() and (s[:, 7] >= s[:, 3]).all())
    assert bool((s[:, 4] >= s[:, 3]).all()) and int((s[:, 5] > 0).sum()) > 0 and bool((s[:, 6][s[:, 6] > 0] >= s[:, 3][s[:, 6] > 0]).all())
    # NOCOMPUTE: GEMM1 only; the fold is stored per warp
    dbg.zero_()
    y5, l5 = mod.moe5_experts(d["hidden_states"], d["topk_ids"], sm, lm, m5, 0, 32, cfg0, dbg=dbg, dbg_flags=(2, 0))
    torch.cuda.synchronize()
    g1_cta = 512 // 8 * 8 * 24
    sinks = dbg[mod.M5_OCC + g1_cta * 8: mod.M5_OCC + g1_cta * 8 + g1_cta * 4]
    assert int((sinks != 0).sum()) > 0 and not torch.equal(_live_rows(y4, l4), _live_rows(y5, l5))
    # NOSTAGE: both GEMMs run; GEMM2 reads GEMM1's (wrong) output
    y5, l5 = mod.moe5_experts(d["hidden_states"], d["topk_ids"], sm, lm, m5, 0, 32, cfg0, dbg=dbg, dbg_flags=(4, 4))
    torch.cuda.synchronize()
    assert torch.equal(l4, l5) and not torch.isnan(y5.float()[l5 >= 0]).any()
    assert int(lm["cnt"].abs().sum()) == 0


@needs_cuda
def test_selfcheck_and_graph_capture(mod, full_state):
    """prepare()'s bitwise self-check passes on the dev card with the dev-card tables; the v5 path captures and replays as a CUDA graph."""
    import reference

    w, state = full_state
    st = dict(state)
    assert all(mod._moe5_selfcheck(st).values())
    d = reference.generate_input(64, "ep8", 9300)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod._kernel_moe5(d, st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod._kernel_moe5(d, st)
    for seed in (9301, 9302):
        d2 = reference.generate_input(64, "ep8", seed)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            d[k].copy_(d2[k])
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), mod._kernel_moe5(d2, st).view(torch.int16))
        ok, msg = reference.check_implementation(d2, out, w)
        assert ok, msg


# --------------------------------------------------------------------------------------------- measure_altA_v1.py (MEASUREMENT-ONLY)
@pytest.fixture(scope="module")
def mm():
    return _load("_measure_altA_v1", MEAS)


def test_measure_file_rules():
    src = MEAS.read_text()
    compile(src, "measure_altA_v1.py", "exec")
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for bad in (r"subprocess", r"urllib", r"requests", r"socket", r"import os", r"environ", r"tolist\(", r"\.item\(", r"\.cpu\(",
                r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src


def test_measure_regions_are_verbatim(mm):
    """kernel_fn and v4's whole Python region are v4's byte for byte; the MOE5 block is sonnet_v5's; the instantiation list is the one the tests assemble."""
    mod = _load("_sonnet_v5_fresh", KERNEL)    # a fresh load: the GPU fixtures replace the module-level tables of the shared `mod`
    a, b, m = V4.read_text(), KERNEL.read_text(), MEAS.read_text()
    assert m[m.rindex("def kernel_fn"):] == a[a.index("def kernel_fn"):]
    lo, hi = "import re\n", "# ============================================================================================ entry points"
    assert a[a.index(lo):a.index(hi)] in m
    s0, s1 = ("# ============================================================================================ MOE5 BEGIN",
              "# ============================================================================================ MOE5 END")
    assert b[b.index(s0):b.index(s1)] == m[m.index(s0):m.index(s1)]
    assert tuple(mm.MEAS_INSTANCES) == MEAS_INSTANCES and len(set(mm.MEAS_INSTANCES)) == 12
    assert mm.moe5_device_src() == mod.moe5_device_src()
    assert mm.kernel_fn is not None and mm.MOE5_CFG != mod.MOE5_CFG      # the measurement table replaces the (4 x 1) GEMM2 of M <= 4 by the 8-warp one
    for (mx, c1, c2), (mx5, d1, d2) in zip(mm.MOE5_CFG, mod.MOE5_CFG):
        assert mx == mx5 and c1 == d1 and c2[2:5] == d2[2:5]


def test_measure_gen_input_equals_reference(mm):
    import reference

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for bs, dist, seed in ((32, "normal", 11), (128, "normal", 12), (256, "skewed", 13), (1, "normal", 11), (16, "skewed", 14)):
        a = mm._gen_input(bs, seed, dist, dev)
        b = reference.generate_input(bs, "ep8", seed, dist=dist, device=dev)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            assert torch.equal(a[k], b[k]), (bs, dist, k)
        assert a["config"] == b["config"]


def test_measure_variants_use_instantiated_kernels(mm):
    """Every variant of every M maps to one of the 12 instantiations, respects the dbg-buffer CTA bound and the kernel's geometry rules."""
    lm = {"zmax": mm.LM_ZMAX}
    inst = set(mm.MEAS_INSTANCES)
    for m in mm.MS:
        vs = mm.variants(m, mm.moe5_zbound(m, lm))
        names = [v[0] for v in vs]
        assert len(names) == len(set(names)) and "base" in names and "alt" in names and "zflat" in names and "pdloff" in names
        if m > mm.SM_MAX_M:
            assert {"nocomp", "nostage", "lb3", "kc", "nb2", "nb4", "altkc"} <= set(names)
        for name, c1, c2, fl, pdl in vs:
            for fuse, c, groups, k in ((True, c1, 512, 4096), (False, c2, 256, 4096)):
                nw, mr, ks, nz, nb, lb = c
                assert (fuse, nw, mr, lb, fl[0 if fuse else 1]) in inst, (m, name, c)
                assert mm._ncta(c, groups) <= mm.DBG_MAX_CTA and groups % (nw * mr * nb) == 0 and (mr == 2 or m <= 16)
                assert k % ks == 0 and (k // ks) % 256 == 0 and ks <= 8
        c1, c2 = mm._v4_cfgs(m)
        assert c1[4] == c2[4] == 1 and c1[5] == c2[5] == 0
    assert mm.expert_bytes({"g1": 512, "k1": 4096, "g2": 256, "k2": 4096}) == (8192 * 4096 * 0.5625, 4096 * 4096 * 0.5625)


class _FakeEnv:
    def __init__(self, state):
        self.state, self.t, self.n = state, 0.0, 0

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
        return dict(m=m, seed=seed, data={"hidden_states": torch.zeros(m, 8), "topk_ids": torch.zeros(m, 8, dtype=torch.int32)}, y0=None, live=None, norm=1.0,
                    info=dict(pairs=8 * m, distinct=min(32, 3 * m), chunks=min(32, 3 * m) + m // 64, n_act=min(32, 3 * m)))

    def compare(self, ctx, y):
        return "OK"

    def graph_time(self, fn, reps=5):
        self.n += 1
        return 100.0 + (self.n % 7)

    def run_eager(self, fns):
        for _, f in fns:
            f()

    def calibrate(self):
        return 4.05, 16.6

    def stamps(self, ctx, c1, c2):
        out = []
        for c, g in ((c1, 512), (c2, 256)):
            n = 64
            t = torch.arange(n, dtype=torch.int64).unsqueeze(1) * 1000 + torch.tensor([0, 100, 400, 1800, 1900, 2000, 2100, 2200], dtype=torch.int64)
            t[1::2, 5] = 0
            out.append((t, torch.tensor([2, 3, 3, 2]), 0))
        return out


def test_measure_control_flow_with_fake_env(mm, monkeypatch, capsys):
    """The budgeted printer: every M gets its I line, the v4 / base / alt / wstream / twins B lines and (large M) the S line; the summary stays under 14 KB."""
    y = torch.zeros(4, 4)

    def fake_stages(data, state, *a, **k):
        return [("route", lambda: None), ("g1", lambda: None), ("g2", lambda: None)], {"y": y}

    monkeypatch.setattr(mm, "_v4_stages", fake_stages)
    monkeypatch.setattr(mm, "_v5_stages", fake_stages)
    monkeypatch.setattr(mm, "_ws_stages", lambda data, state, c1, c2: [("g1", lambda: None), ("g2", lambda: None)])
    state = {"smallm": {"dims": {"k1": 4096, "k2": 4096, "g1": 512, "g2": 256}}, "largem": {"zmax": mm.LM_ZMAX}, "moe5": {"dbg": None}}
    mm._measure(state, _FakeEnv(state), t_start=0.0)
    out = capsys.readouterr().out
    assert len(out.encode()) < mm.SUMMARY_BUDGET
    assert out.startswith("MEAS-ALTA") and "bw_ach=4.05TB/s" in out and "MEAS-ALTA done" in out and "skipped" not in out
    for m in mm.MS:
        assert f"I moe m={m} seeds=11/12/13" in out
        for v in ("v4", "base", "alt", "wstream", "zflat", "pdloff"):
            assert re.search(rf"^B moe m={m} v={v} \| g1=[\d.]+/[\d.]+/[\d.]+ g2=[\d.]+/[\d.]+/[\d.]+ \| tb=[\d.]+/[\d.]+ ach=\d+/\d+% x=", out, re.M), (m, v)
        if m > 16:
            assert re.search(rf"^S moe m={m} g1 cta=[\d.]+us item=[\d.]+us span=\d+ stage=\d+% stream=\d+% wait=\d+% fix=\d+% epi=\d+% bw/cta=[\d.]+GB/s occ=3/2.5 \| g2 ", out, re.M)
            assert f"B moe m={m} v=nocomp" in out and f"B moe m={m} v=nostage" in out and f"B moe m={m} v=lb3" in out
        else:
            assert not re.search(rf"^S moe m={m} ", out, re.M)
    assert out.count("pair=") == 4 * len(mm.MS)     # v4, base, alt and pdloff carry the pair (GEMM1 + GEMM2 in one graph) time
    lines = out.splitlines()
    assert all(len(l) < 280 for l in lines) and len(lines) <= 6 * 16 + 3


def test_measure_budget_skips_variants_not_seeds(mm, monkeypatch, capsys):
    """Past the deadline the printer reports what was skipped and still prints the summary."""
    y = torch.zeros(4, 4)

    def fake_stages(data, state, *a, **k):
        return [("route", lambda: None), ("g1", lambda: None), ("g2", lambda: None)], {"y": y}

    monkeypatch.setattr(mm, "_v4_stages", fake_stages)
    monkeypatch.setattr(mm, "_v5_stages", fake_stages)
    monkeypatch.setattr(mm, "_ws_stages", lambda data, state, c1, c2: [("g1", lambda: None), ("g2", lambda: None)])
    env = _FakeEnv(None)
    env.clock = lambda: (setattr(env, "t", env.t + 2.0) or env.t)   # 2 s per clock call: the deadline (>= 20 s) passes after a few variants
    state = {"smallm": {"dims": {"k1": 4096, "k2": 4096, "g1": 512, "g2": 256}}, "largem": {"zmax": mm.LM_ZMAX}, "moe5": {"dbg": None}}
    mm._measure(state, env, t_start=0.0)
    out = capsys.readouterr().out
    assert "MEAS-ALTA done" in out and "skipped:" in out and "B moe m=256 v=v4" in out and len(out.encode()) < mm.SUMMARY_BUDGET


@needs_cuda
def test_measure_real_env_on_dev_card(mm, mod, v4, full_state, nv, monkeypatch, capsys):
    """The real _RealEnv on the 2060 (NVRTC build of the measurement extension incl. the WSTREAM / CAL kernels): graph timing after flushes, the variants,
    the stamps, the summary; every scheduling variant is bitwise v4 (bw=OK), the twins are not."""
    w, state = full_state
    ext = nv.NvrtcMeasExt(mm.meas_device_src(), mm.MEAS_NAMES)
    monkeypatch.setattr(mm, "HAVE_SGLANG", True)
    monkeypatch.setattr(mm, "SMALLM_CFG", DEV_SMALLM_CFG)
    monkeypatch.setattr(mm, "MOE5_CFG", DEV_MOE5_CFG)
    for b_, c_ in DEV_LARGEM_CFG.items():
        monkeypatch.setitem(mm.LARGEM_CFG, b_, c_)
    monkeypatch.setattr(mm, "MS", (32, 8))
    monkeypatch.setattr(mm, "SEEDS", (11,))
    monkeypatch.setattr(mm, "REPS", 3)
    st = {"smallm": state["smallm"], "largem": state["largem"], "norm_w": state["norm_w"], "moe5": mm.prepare_moe5(state["smallm"], state["largem"], ext=ext)}
    mm._measure(st)
    out = capsys.readouterr().out
    with capsys.disabled():
        print("\n" + out)
    assert out.startswith("MEAS-ALTA") and "MEAS-ALTA done" in out and len(out.encode()) < mm.SUMMARY_BUDGET
    assert re.search(r"bw_ach=[\d.]+TB/s", out) and "I moe m=32 seeds=11" in out and "I moe m=8 seeds=11" in out
    for v in ("v4", "base", "lb3", "alt", "nb2", "nb4", "zflat", "pdloff", "kc", "altkc"):   # kc = KS 8 = the dev-card base split: bitwise too
        mt = re.search(rf"^B moe m=32 v={v} \| .* bw=(\S+)", out, re.M)
        assert mt, v
        assert mt.group(1) == "OK", (v, mt.group(0))
    for v in ("nocomp", "nostage"):
        mt = re.search(rf"^B moe m=32 v={v} \| .* bw=(\S+)", out, re.M)
        assert mt and mt.group(1).startswith("e="), (v, mt.group(0) if mt else None)
    mt = re.search(r"^B moe m=8 v=kc \| .* bw=(\S+)", out, re.M)                              # m 8: KS 4 -> 8 changes the split-K sums
    assert mt and mt.group(1).startswith("e="), mt.group(0) if mt else None
    assert re.search(r"^B moe m=32 v=wstream \| g1=[\d.]+ g2=[\d.]+ \| tb=", out, re.M)
    assert re.search(r"^S moe m=32 g1 cta=[\d.]+us item=[\d.]+us span=\d+ stage=\d+% stream=\d+% .* occ=\d+/[\d.]+ \| g2 .* \| pdl g2t0=-?[\d.]+ g2t1=-?[\d.]+ g2end=-?[\d.]+$", out, re.M)
    for v in ("v4", "base", "zflat", "alt", "nb2", "zfnb2", "pdloff"):
        mt = re.search(rf"^B moe m=8 v={v} \| .* bw=(\S+)", out, re.M)
        assert mt and mt.group(1) == "OK", (v, mt.group(0) if mt else None)
    assert "skipped" not in out
    assert int(state["largem"]["cnt"].abs().sum()) == 0


@needs_cuda
@pytest.mark.parametrize("bs", [1, 4, 8])
def test_judge_small_m_config_kc2048_on_dev_card(mod, full_state, bs):
    """The judge's small-M configuration (K slice 2048 = v4's SMALLM_CFG, 8-row tiles: 34 KB smem, the only Kc 2048 case the 2060 can hold) passes the
    judge's check and is deterministic: the staging path with 256 16-byte chunks per row is exercised here, the bitwise check of it runs in prepare()."""
    import reference

    w, state = full_state
    judge_cfg = {1: ((8, 1, 2, 2, 1, 2), (4, 1, 2, 6, 1, 4)), 4: ((8, 1, 2, 2, 1, 2), (4, 1, 4, 6, 1, 4)), 8: ((8, 1, 2, 8, 1, 2), (8, 1, 2, 4, 1, 2))}[bs]
    for seed in (8500 + bs, 8600 + bs):
        d = reference.generate_input(bs, "ep8", seed)
        y, local = mod.moe5_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe5"], 0, 32, judge_cfg)
        y2, _ = mod.moe5_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe5"], 0, 32, judge_cfg)
        torch.cuda.synchronize()
        assert torch.equal(_live_rows(y, local), _live_rows(y2, local))
        out = mod.epilogue_triton(y, d["topk_weights"], local, state["norm_w"])
        ok, msg = reference.check_implementation(d, out, w)
        assert ok, msg
    assert int(state["largem"]["cnt"].abs().sum()) == 0
