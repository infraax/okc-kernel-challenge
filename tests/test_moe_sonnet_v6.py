"""kernels/muse_moe_w4a16/sonnet_v6.py: the deeper weight pipeline (C21b: register batches RB in flight | per-warp cp.async ring of RS stages, continuity across
work items) in the unified template, on top of sonnet_v5; and measure_ring_v1.py.

CPU: file rules, v5 region verbatim, config table / instantiations / geometry rules, dispatch, prepare() fallbacks, host-source generation.
GPU dev card (RTX 2060, sm_75, no nvcc, 64 KB smem per block, cp.async emulated by a synchronous copy): NVRTC builds of v4's and v6's device sources;
V1 = v6 BITWISE v4 (= v5) on reference.generate_input routing for M in {1, 4, 8, 16, 32, 33, 128, 256} x 3 seeds at the same K split (dev-card tables: ks 4
below 17 tokens, ks 8 above; rings of RS 4 / 3 stages instead of the judge's RS 8, which needs 104-141 KB of smem: test-only instantiations of the same
template); every scheduling flag (register vs ring, NB, flattened z, PDL off) stays bitwise; the STAMPS twin reports; the judge's check passes; graph capture;
the sm_80 / sm_90 assembly of every instantiation (regs / spills printed; no local memory, no spills in the ring / small-M kernels).

Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v6.py"""
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

KERNEL = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v6.py"
V5 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v5.py"
V4 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v4.py"
MEAS = ROOT / "kernels" / "muse_moe_w4a16" / "measure_ring_v1.py"
CUOBJDUMP = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/cuobjdump"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# The measurement file's instantiation list (FUSE, NW, MR, D, RB, RS, LB, DBG): 12 (fable-09 D4 S1 budget); the test checks it equals the file's.
MEAS_INSTANCES = (
    (True, 4, 2, 1, 0, 8, 2, 0), (False, 4, 2, 1, 0, 8, 2, 0),      # ring, large geometry (v6 default): 8 KB per warp in flight, 2 CTAs/SM
    (True, 4, 2, 1, 3, 0, 2, 0), (False, 4, 2, 1, 3, 0, 2, 0),      # register depth, large geometry: 4.5 KB per warp, 186 regs, 2 CTAs/SM
    (True, 8, 2, 1, 0, 8, 1, 0), (False, 8, 2, 1, 0, 8, 1, 0),      # ring, one 8-warp CTA per SM (141 KB smem)
    (True, 4, 1, 1, 7, 0, 3, 0), (False, 4, 1, 1, 7, 0, 3, 0),      # register depth, small geometry (v6 default): 5.25 KB per warp, 3 CTAs/SM
    (True, 8, 1, 2, 0, 8, 1, 0), (False, 8, 1, 2, 0, 8, 1, 0),      # ring, small geometry
    (True, 4, 2, 1, 0, 8, 2, 1), (False, 4, 2, 1, 0, 8, 2, 1),      # ring + STAMPS
)
# Dev-card-only instantiations (64 KB smem): the same template with RS 4 / 3 rings.
DEV_INSTANCES = (
    (True, 4, 2, 1, 0, 4, 2, 0), (False, 4, 2, 1, 0, 4, 2, 0), (True, 8, 2, 1, 0, 3, 1, 0), (False, 8, 2, 1, 0, 3, 1, 0),
    (True, 8, 1, 1, 0, 4, 1, 0), (False, 8, 1, 1, 0, 4, 1, 0), (True, 4, 2, 1, 0, 4, 2, 1), (False, 4, 2, 1, 0, 4, 2, 1),
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v6_moe", KERNEL)


@pytest.fixture(scope="module")
def v4():
    return _load("_sonnet_v4_moe_for_v6", V4)


# --------------------------------------------------------------------------------------------- CPU
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\(", r"urllib"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert "okc-diag HAVE_SGLANG" in src
    assert "PREDICTION" in src and "DECISION RULE" in src and "CONTINUITY" in src
    compile(src, "sonnet_v6.py", "exec")


def test_v5_region_is_verbatim():
    """Everything of v5 from `import re` up to its entry points (v4's region, the MOE5 block, its self-checks) is in v6 unchanged."""
    a, b = V5.read_text(), KERNEL.read_text()
    lo, hi = "import re\n", "# ============================================================================================ entry points"
    assert a[a.index(lo):a.index(hi)] in b
    assert b.count("# ============================================================================================ MOE6 BEGIN") == 1
    c = V4.read_text()
    assert c[c.index(lo):c.index(hi)] in b


def test_cfg_table_buckets_instances_and_geometry(mod, v4):
    assert [mod.moe6_bucket(m) for m in (1, 2, 3, 4, 8, 9, 12, 16, 17, 32, 33, 128, 129, 256, 257)] == \
        [1, 2, 4, 4, 8, 12, 12, 16, 32, 32, 128, 128, 256, 256, None]
    assert mod.moe6_cfg(257) is None and mod.moe6_cfg(17) == mod.moe6_cfg(32)
    old = dict(mod.MOE6_ENABLE)
    try:
        mod.MOE6_ENABLE[128] = False
        assert mod.moe6_cfg(100) is None and mod.moe6_cfg(256) is not None
    finally:
        mod.MOE6_ENABLE.update(old)
    inst = set(mod.MOE6_INSTANCES)
    assert len(inst) == 6 and all(dbg == 0 for *_r, dbg in inst)
    lm = {"zmax": mod.LM_ZMAX}
    for mx, c1, c2 in mod.MOE6_CFG:
        for fuse, c, groups, k in ((True, c1, 512, 4096), (False, c2, 256, 4096)):
            nw, mr, d, rb, rs, ks, nz, nb, lb = c
            assert (fuse, nw, mr, d, rb, rs, lb, 0) in inst, (mx, c)
            mod.moe6_check_cfg(mx, k, groups, c)
            assert nb == (2 if mx > mod.SM_MAX_M else 1)
            if lb:
                assert 65536 // (nw * 32 * lb) >= 128
        if mx <= mod.SM_MAX_M:
            s = v4.smallm_cfg(mx)
            assert (c1[5], c2[5]) == (s["ks1"], s["ks2"]) and (c1[6], c2[6]) == (s["nz1"], s["nz2"]), mx
        else:
            l = v4.LARGEM_CFG[mx]
            assert (c1[5], c2[5]) == (l["ks1"], l["ks2"]) and (c1[6], c2[6]) == (l["nz1"], l["nz2"]), mx
            assert c1[4] == 8 and c1[2] == 1 and c1[8] == 2             # the ring default: D 1 RS 8 LB 2
            assert mod.moe6_smem(mx, 4096, c1) <= 2 * 56 * 1024          # 2 CTAs per SM on 227 KB
    assert set(mod.moe6_bucket(m) for m in mod.MOE6_SELFCHECK_MS) == {mx for mx, _a, _b in mod.MOE6_CFG}
    # geometry rules
    with pytest.raises(AssertionError):
        mod.moe6_check_cfg(32, 4096, 512, (4, 2, 1, 2, 0, 4, 16, 2, 2))     # nbatch 16 % (RB + 1 = 3) != 0
    with pytest.raises(AssertionError):
        mod.moe6_check_cfg(32, 4096, 512, (4, 2, 1, 0, 1, 4, 16, 2, 2))     # RS >= 2
    with pytest.raises(AssertionError):
        mod.moe6_check_cfg(32, 4096, 512, (4, 2, 3, 0, 4, 4, 16, 2, 2))     # D must divide nit
    with pytest.raises(AssertionError):
        mod.moe6_check_cfg(32, 4096, 512, (4, 1, 1, 7, 0, 4, 16, 2, 3))     # mr 1 above 16 tokens
    with pytest.raises(AssertionError):
        mod.moe6_check_cfg(256, 4096, 512, (8, 2, 1, 0, 16, 4, 48, 2, 1))   # smem over the limit
    mod.moe6_check_cfg(8, 4096, 512, (4, 1, 1, 7, 0, 2, 8, 1, 3))           # nbatch 32 % 8
    assert mod.moe6_ring_bytes(4, 2, 1, 8) == 4 * 8 * 2 * 576 and mod.moe6_ring_bytes(4, 2, 1, 0) == 0
    assert [mod.moe6_zbound(m, lm) if hasattr(mod, "moe6_zbound") else mod.moe5_zbound(m, lm) for m in (1, 8, 256)] == [8, 32, 96]


def test_host_source_generation(mod):
    src = mod.moe6_host_src(mod.MOE6_INSTANCES)
    for f, w, r, d, rb, rs, lb, dbg in mod.MOE6_INSTANCES:
        t = f"{'true' if f else 'false'}, {w}, {r}, {d}, {rb}, {rs}, {lb}, {dbg}"
        assert f"M6_CASE({t})" in src and f"w4_moe6_kernel<{t}>, cudaFuncAttributeMaxDynamicSharedMemorySize" in src
        assert f"w4_moe6_kernel<{t}>, cudaFuncAttributePreferredSharedMemoryCarveout" in src
    assert "@CASES@" not in src and "@INITS@" not in src
    assert mod.moe6_instance_names([(True, 4, 2, 1, 0, 8, 2, 1)]) == ["w4_moe6_kernel<true,4,2,1,0,8,2,1>"]
    dev = mod.moe6_device_src()
    assert dev.startswith(mod.moe5_device_src()) and "w4_moe6_kernel" in dev and "cp.async.cg.shared.global" in dev and "cp.async.wait_group" in dev
    for name in ("moe6_gemm", "moe6_route", "moe6_init"):
        assert name in mod._MOE6_CPP_SRC


def test_dispatch(mod, monkeypatch):
    monkeypatch.setattr(mod, "HAVE_SGLANG", True)
    monkeypatch.setattr(mod, "_kernel_moe6", lambda d, s: "6")
    monkeypatch.setattr(mod, "_kernel_moe5", lambda d, s: "5")
    monkeypatch.setattr(mod, "_kernel_smallm", lambda d, s: "S")
    monkeypatch.setattr(mod, "_kernel_largem", lambda d, s: "L")
    monkeypatch.setattr(mod, "_kernel_sglang", lambda d, s: "M")

    def data(m, start=0, count=32):
        return {"hidden_states": torch.zeros(m, 4), "config": {"local_start": start, "local_count": count}}

    ok = {mx: True for mx, _a, _b in mod.MOE6_CFG}
    st = {"smallm": object(), "largem": object(), "largem_ok": {32: True, 128: True, 256: True}, "moe5": object(), "moe5_ok": dict(ok),
          "moe6": object(), "moe6_ok": ok}
    assert [mod.kernel_fn(data(m), st) for m in (1, 8, 16, 17, 32, 33, 128, 129, 256)] == ["6"] * 9
    assert [mod.kernel_fn(data(m), st) for m in (257, 512)] == ["M", "M"]
    assert mod.kernel_fn(data(100, 32, 32), st) == "M" and mod.kernel_fn(data(100, 0, 256), st) == "M"
    st2 = dict(st, moe6_ok={**ok, 128: False, 8: False})
    assert [mod.kernel_fn(data(m), st2) for m in (8, 16, 100, 200)] == ["5", "6", "5", "6"]
    assert mod.kernel_fn(data(100), dict(st, moe6=None)) == "5"
    assert mod.kernel_fn(data(100), dict(st, moe6=None, moe5=None)) == "L"
    assert mod.kernel_fn(data(8), dict(st, moe6=None, moe5=None, smallm=None)) == "M"
    assert mod.kernel_fn(data(8), dict(st, largem=None)) == "S"      # v6 (and v5) need v4's large-M scratch


def test_prepare_moe6_state_fallbacks(mod, monkeypatch):
    monkeypatch.setattr(mod, "prepare_moe6", lambda sm, lm, ext=None, flags=(): {"x": 1})
    monkeypatch.setattr(mod, "_moe6_selfcheck", lambda st: {1: True, 2: True, 4: False, 8: True, 12: True, 16: True, 32: True, 128: False, 256: True})
    st = {"smallm": object(), "largem": object(), "norm_w": None}
    mod._prepare_moe6_state(st, ())
    assert st["moe6"] == {"x": 1} and st["moe6_ok"][4] is False and st["moe6_ok"][128] is False and st["moe6_ok"][256] is True
    monkeypatch.setattr(mod, "_moe6_selfcheck", lambda st: {b: False for b in (1, 2, 4, 8, 12, 16, 32, 128, 256)})
    st = {"smallm": object(), "largem": object()}
    mod._prepare_moe6_state(st, ())
    assert st["moe6"] is None and not any(st["moe6_ok"].values())

    def boom(sm, lm, ext=None, flags=()):
        raise RuntimeError("no nvcc")
    monkeypatch.setattr(mod, "prepare_moe6", boom)
    st = {"smallm": object(), "largem": object()}
    mod._prepare_moe6_state(st, ())
    assert st["moe6"] is None
    for st in ({"smallm": None, "largem": object()}, {"smallm": object(), "largem": None}):
        mod._prepare_moe6_state(st, ())
        assert st["moe6"] is None and st["moe6_ok"] == {}


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
    cubin, low = nv.compile_device(src, arch, names=names)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        r = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(p)], capture_output=True, text=True)
    use = {m.group(1): (int(m.group(2)), int(m.group(3)), int(m.group(5))) for m in
           re.finditer(r"Function (\S+?):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", r.stdout)}
    return {n: use.get(low[n].rstrip("\x00")) for n in names}


ALL_INSTANCES = None


def _all_instances(mod):
    return sorted(set(mod.MOE6_INSTANCES) | set(MEAS_INSTANCES) | set(DEV_INSTANCES))


@needs_cuda
@pytest.mark.parametrize("sm", ["sm_80", "sm_90"])
def test_source_assembles_for_judge_arch(mod, nv, sm, capsys):
    """Every instantiation of the candidate, of the measurement file and of the dev-card tests assembles for the judge's GPU; regs within the cap; no local
    memory; the ring and small-M kernels without any stack frame (no spills)."""
    inst = _all_instances(mod)
    names = mod.moe6_instance_names(inst)
    use = _usage(nv, mod.moe6_device_src(), sm, names)
    with capsys.disabled():
        print(f"\n{sm} regs/stack/local per instantiation:")
    for (f, w, r, d, rb, rs, lb, dbg), n in zip(inst, names):
        regs, stack, local = use[n]
        with capsys.disabled():
            print(f"  {n:40s} regs={regs} stack={stack} local={local}")
        assert local == 0, n
        if lb:
            assert regs <= 65536 // (w * 32 * lb), (n, regs)
        if dbg == 0 and (rs > 0 or r == 1 or lb < 3):
            assert stack == 0, (n, stack)
        assert stack <= 64, (n, stack)


@pytest.fixture(scope="module")
def exts(mod, v4, nv):
    inst = _all_instances(mod)
    return (nv.NvrtcExt(v4._SMALLM_DEVICE_SRC), nv.NvrtcLargeExt(v4._SMALLM_DEVICE_SRC + v4._LARGEM_EXTRA_SRC),
            nv.NvrtcMoe6Ext(mod.moe6_device_src(), mod.moe6_instance_names(inst)))


# dev-card tables (64 KB smem per block): ks 4 at M <= 16 (16-row tiles of Kc 1024), ks 8 at M > 16 (32-row tiles of Kc 512); rings RS 4 / 3
DEV_SMALLM_CFG = ((16, 8, 4, 8, 8, 4, 8),)
DEV_LARGEM_CFG = {b: dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24) for b in (32, 128, 256)}
DEV_MOE6_CFG = tuple((mx, (4, 1, 1, 7, 0, 4, 8, 1, 3), (4, 1, 1, 7, 0, 4, 8, 1, 3)) if mx <= 16 else (mx, (4, 2, 1, 0, 4, 8, 24, 2, 2), (4, 2, 1, 0, 4, 8, 24, 2, 2))
                     for mx in (1, 2, 4, 8, 12, 16, 32, 128, 256))


def _full_weights():
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(5252)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=g, device="cuda", leading=(32,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
            "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device="cuda")).to(torch.bfloat16)}


@pytest.fixture(scope="module")
def full_state(mod, v4, exts):
    """v4's state (packed weights + scratch, NVRTC extensions) with the dev-card tables, and v6's extension on top (v5's extension is not built here:
    the bitwise reference is v4's kernels, which v5 equals bit for bit)."""
    w = _full_weights()
    saved = (v4.SMALLM_CFG, dict(v4.LARGEM_CFG), mod.SMALLM_CFG, dict(mod.LARGEM_CFG), mod.MOE6_CFG)
    v4.SMALLM_CFG = mod.SMALLM_CFG = DEV_SMALLM_CFG
    for m_ in (v4, mod):
        m_.LARGEM_CFG.clear()
        m_.LARGEM_CFG.update(DEV_LARGEM_CFG)
    mod.MOE6_CFG = DEV_MOE6_CFG
    sm = v4.prepare_smallm(w, ext=exts[0])
    lm = v4.prepare_largem(sm, ext=exts[1])
    m6 = mod.prepare_moe6(sm, lm, ext=exts[2])
    state = {"smallm": sm, "largem": lm, "largem_ok": {32: True, 128: True, 256: True}, "moe5": None, "moe6": m6,
             "moe6_ok": {mx: True for mx, _a, _b in DEV_MOE6_CFG}, "norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    yield w, state
    v4.SMALLM_CFG, l4, mod.SMALLM_CFG, l6, mod.MOE6_CFG = saved
    for m_, l_ in ((v4, l4), (mod, l6)):
        m_.LARGEM_CFG.clear()
        m_.LARGEM_CFG.update(l_)
    del sm, lm, m6, state, w
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
    """V1: v6's expert outputs (register mode below 17 tokens, ring above) equal v4's bit for bit on 3 generate_input seeds; the judge's check passes."""
    import reference

    w, state = full_state
    c1, c2 = mod.moe6_cfg(bs)
    for i, dist in enumerate(("normal", "normal", "skewed")):
        d = reference.generate_input(bs, "ep8", 8100 + bs + i, dist=dist)
        y4, l4 = _v4_experts(v4, state, d)
        y6, l6 = mod.moe6_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe6"], 0, 32, (c1, c2))
        torch.cuda.synchronize()
        assert torch.equal(l4, l6)
        assert torch.equal(_live_rows(y4, l4), _live_rows(y6, l6)), (bs, dist)
        assert int(state["largem"]["cnt"].abs().sum()) == 0 and int(state["smallm"]["cnt"].abs().sum()) == 0
        if i == 0:
            out = mod._kernel_moe6(d, dict(state))
            ok, msg = reference.check_implementation(d, out, w)
            assert ok, msg
            out2 = mod._kernel_moe6(d, dict(state))
            assert torch.equal(out.view(torch.int16), out2.view(torch.int16)), "not deterministic"


@needs_cuda
@pytest.mark.parametrize("bs", [8, 16, 33, 256])
def test_pipeline_flags_stay_bitwise(mod, v4, full_state, bs):
    """Register depth (RB 3 / 7), rings (RS 4 large, RS 3 in one 8-warp CTA, RS 4 small), NB 1 / 2, z flattened or looping over several chunks per CTA
    (continuity across chunks), PDL off: every output bit equals v4's."""
    import reference

    w, state = full_state
    sm, lm, m6 = state["smallm"], state["largem"], state["moe6"]
    d = reference.generate_input(bs, "ep8", 8300 + bs)
    y4, l4 = _v4_experts(v4, state, d)
    ref = _live_rows(y4, l4)
    zb = mod.moe5_zbound(bs, lm)
    if bs <= 16:
        combos = [((4, 1, 1, 7, 0, 4, zb, 1, 3), (4, 1, 1, 7, 0, 4, zb, 1, 3)), ((4, 1, 1, 7, 0, 4, 2, 1, 3), (4, 1, 1, 7, 0, 4, 3, 1, 3)),
                  ((4, 1, 1, 7, 0, 4, 8, 2, 3), (4, 1, 1, 7, 0, 4, 8, 2, 3)), ((8, 1, 1, 0, 4, 4, 8, 1, 1), (8, 1, 1, 0, 4, 4, 8, 1, 1)),
                  ((8, 1, 1, 0, 4, 4, 2, 2, 1), (8, 1, 1, 0, 4, 4, 2, 2, 1)), ((8, 1, 1, 0, 4, 4, zb, 1, 1), (4, 1, 1, 7, 0, 4, zb, 2, 3))]
    else:
        combos = [((4, 2, 1, 0, 4, 8, 24, nb1, 2), (4, 2, 1, 0, 4, 8, 24, nb2, 2)) for nb1, nb2 in ((1, 1), (2, 2), (2, 1))]
        combos += [((4, 2, 1, 3, 0, 8, 24, nb1, 2), (4, 2, 1, 3, 0, 8, 24, nb2, 2)) for nb1, nb2 in ((1, 1), (2, 2))]
        combos += [((8, 2, 1, 0, 3, 8, 24, 2, 1), (8, 2, 1, 0, 3, 8, 24, 1, 1)), ((4, 2, 1, 0, 4, 8, zb, 2, 2), (4, 2, 1, 0, 4, 8, zb, 2, 2)),
                   ((4, 2, 1, 0, 4, 8, 4, 2, 2), (4, 2, 1, 3, 0, 8, 4, 2, 2)), ((4, 2, 1, 3, 0, 8, 3, 1, 2), (8, 2, 1, 0, 3, 8, 3, 2, 1))]
    for c1, c2 in combos:
        for pdl in ((True, True), (True, False)):
            y6, l6 = mod.moe6_experts(d["hidden_states"], d["topk_ids"], sm, lm, m6, 0, 32, (c1, c2), pdl=pdl)
            torch.cuda.synchronize()
            assert torch.equal(l4, l6) and torch.equal(ref, _live_rows(y6, l6)), (c1, c2, pdl)
    assert int(lm["cnt"].abs().sum()) == 0


@needs_cuda
def test_stamps_twin_reports(mod, v4, full_state):
    """STAMPS on the ring kernel: t0..t7 monotone per CTA, t3 (end of the k-loop) after t2, per-SM concurrency counters back to 0; output still bitwise."""
    import reference

    w, state = full_state
    sm, lm, m6 = state["smallm"], state["largem"], state["moe6"]
    d = reference.generate_input(33, "ep8", 8433)
    y4, l4 = _v4_experts(v4, state, d)
    cfg = ((4, 2, 1, 0, 4, 8, 24, 2, 2), (4, 2, 1, 0, 4, 8, 24, 2, 2))
    ncta = 512 // 8 // 2 * 8 * 24
    dbgs = (torch.zeros(mod.M5_OCC + ncta * 16, dtype=torch.int64, device="cuda"), torch.zeros(mod.M5_OCC + ncta * 16, dtype=torch.int64, device="cuda"))
    y6, l6 = mod.moe6_experts(d["hidden_states"], d["topk_ids"], sm, lm, m6, 0, 32, cfg, dbg=dbgs, dbg_flags=(1, 1))   # one buffer per GEMM
    torch.cuda.synchronize()
    assert torch.equal(_live_rows(y4, l4), _live_rows(y6, l6))
    for dbg in dbgs:
        occ = dbg[: mod.M5_OCC].view(torch.int32)
        assert int(occ[:256].abs().sum()) == 0 and int(occ[256:512].max()) >= 1
        st = dbg[mod.M5_OCC: mod.M5_OCC + ncta * 8].view(-1, 8)
        s = st[st[:, 0] > 0]
        assert s.shape[0] > 0
        assert bool((s[:, 1] >= s[:, 0]).all() and (s[:, 2] >= s[:, 1]).all() and (s[:, 3] >= s[:, 2]).all() and (s[:, 4] >= s[:, 3]).all() and (s[:, 7] >= s[:, 4]).all())
        assert int((s[:, 5] > 0).sum()) > 0 and bool((s[:, 6][s[:, 6] > 0] >= s[:, 4][s[:, 6] > 0]).all())


@needs_cuda
def test_selfcheck_and_graph_capture(mod, full_state):
    """prepare()'s bitwise self-check (v6 vs v5-or-v4) passes with the dev-card tables; the v6 path captures and replays as a CUDA graph."""
    import reference

    w, state = full_state
    st = dict(state)
    assert all(mod._moe6_selfcheck(st).values())
    d = reference.generate_input(64, "ep8", 9300)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod._kernel_moe6(d, st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod._kernel_moe6(d, st)
    for seed in (9301, 9302):
        d2 = reference.generate_input(64, "ep8", seed)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            d[k].copy_(d2[k])
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), mod._kernel_moe6(d2, st).view(torch.int16))
        ok, msg = reference.check_implementation(d2, out, w)
        assert ok, msg


@needs_cuda
@pytest.mark.parametrize("bs", [1, 4, 8])
def test_judge_small_m_config_kc2048_on_dev_card(mod, full_state, bs):
    """The judge's small-M configuration (K slice 2048, 8-row tiles: 34 KB smem; nbatch 32 % 8) passes the judge's check and is deterministic."""
    import reference

    w, state = full_state
    judge_cfg = {1: ((4, 1, 1, 7, 0, 2, 2, 1, 3), (4, 1, 1, 7, 0, 2, 6, 1, 3)), 4: ((4, 1, 1, 7, 0, 2, 2, 1, 3), (4, 1, 1, 7, 0, 4, 6, 1, 3)),
                 8: ((4, 1, 1, 7, 0, 2, 8, 1, 3), (4, 1, 1, 7, 0, 2, 4, 1, 3))}[bs]
    for seed in (8500 + bs, 8600 + bs):
        d = reference.generate_input(bs, "ep8", seed)
        y, local = mod.moe6_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe6"], 0, 32, judge_cfg)
        y2, _ = mod.moe6_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe6"], 0, 32, judge_cfg)
        torch.cuda.synchronize()
        assert torch.equal(_live_rows(y, local), _live_rows(y2, local))
        out = mod.epilogue_triton(y, d["topk_weights"], local, state["norm_w"])
        ok, msg = reference.check_implementation(d, out, w)
        assert ok, msg
    assert int(state["largem"]["cnt"].abs().sum()) == 0


# --------------------------------------------------------------------------------------------- measure_ring_v1.py (MEASUREMENT-ONLY)
@pytest.fixture(scope="module")
def mm():
    return _load("_measure_ring_v1", MEAS)


def test_measure_file_rules():
    src = MEAS.read_text()
    compile(src, "measure_ring_v1.py", "exec")
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for bad in (r"subprocess", r"urllib", r"requests", r"socket", r"import os", r"environ", r"tolist\(", r"\.item\(", r"\.cpu\(",
                r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src


def test_measure_regions_are_verbatim(mm):
    """kernel_fn and v5's whole Python region are v5's byte for byte; the MOE6 block is sonnet_v6's; the instantiation list is the one the tests assemble."""
    mod = _load("_sonnet_v6_fresh", KERNEL)    # a fresh load: the GPU fixtures replace the module-level tables of the shared `mod`
    a, b, m = V5.read_text(), KERNEL.read_text(), MEAS.read_text()
    assert m[m.rindex("def kernel_fn"):] == a[a.rindex("def kernel_fn"):]
    lo, hi = "import re\n", "# ============================================================================================ entry points"
    assert a[a.index(lo):a.index(hi)] in m
    s0, s1 = ("# ============================================================================================ MOE6 BEGIN",
              "# ============================================================================================ MOE6 END")
    assert b[b.index(s0):b.index(s1)] == m[m.index(s0):m.index(s1)]
    assert set(mm.MEAS_INSTANCES) == set(MEAS_INSTANCES) and len(set(mm.MEAS_INSTANCES)) == 12
    assert mm.moe6_device_src() == mod.moe6_device_src() and mm.MOE6_CFG == mod.MOE6_CFG
    assert set(mod.MOE6_INSTANCES) <= set(mm.MEAS_INSTANCES)


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
    """Every variant of every M maps to one of the 12 instantiations and passes the kernel's geometry rules; the dev-card depths do the same with theirs."""
    lm = {"zmax": mm.LM_ZMAX}
    for depths in ((8, 8, (2, 8)), (4, 3, (1, 4))):
        rl, r82, rsm = depths
        mm.RING_LARGE_RS, mm.RING82_RS, mm.RING_SMALL_DR = rl, r82, rsm
        try:
            inst = set(mm.meas_instances())
            assert len(inst) == 12
            for m in mm.MS:
                vs = mm.variants(m, mm.moe5_zbound(m, lm))
                names = [v[0] for v in vs]
                assert len(names) == len(set(names)) and "zflat" in names and "pdloff" in names
                assert ({"ring", "rb3", "ring82", "ringnb1", "rb3nb1", "ringnb4"} <= set(names)) == (m > mm.SM_MAX_M)
                assert ({"rb7", "ring81", "rb7nb2", "ring81nb2"} <= set(names)) == (m <= mm.SM_MAX_M)
                for name, c1, c2, fl, pdl in vs:
                    for fuse, c, groups, k in ((True, c1, 512, 4096), (False, c2, 256, 4096)):
                        nw, mr, d, rb, rs, ks, nz, nb, lb = c
                        assert (fuse, nw, mr, d, rb, rs, lb, fl[0 if fuse else 1]) in inst, (depths, m, name, c)
                        if depths[0] == 8:
                            mm.moe6_check_cfg(m, k, groups, c)
                        assert mm._ncta(c, groups) <= mm.DBG_MAX_CTA
        finally:
            mm.RING_LARGE_RS, mm.RING82_RS, mm.RING_SMALL_DR = 8, 8, (2, 8)
    assert tuple(mm.meas_instances()) == mm.MEAS_INSTANCES
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


def _fake_stages_patch(mm, monkeypatch):
    y = torch.zeros(4, 4)

    def fake_stages(data, state, *a, **k):
        return [("route", lambda: None), ("g1", lambda: None), ("g2", lambda: None)], {"y": y}

    monkeypatch.setattr(mm, "_v5_stages", fake_stages)
    monkeypatch.setattr(mm, "_v6_stages", fake_stages)
    monkeypatch.setattr(mm, "_ws_stages", lambda data, state, c1, c2: [("g1", lambda: None), ("g2", lambda: None)])
    return {"smallm": {"dims": {"k1": 4096, "k2": 4096, "g1": 512, "g2": 256}}, "largem": {"zmax": mm.LM_ZMAX}, "moe6": {"dbg": None}}


def test_measure_control_flow_with_fake_env(mm, monkeypatch, capsys):
    """The budgeted printer: every M gets its I line, the v5 / variants / wstream B lines and (large M) the S line; the summary stays under 14 KB."""
    state = _fake_stages_patch(mm, monkeypatch)
    mm._measure(state, _FakeEnv(state), t_start=0.0)
    out = capsys.readouterr().out
    assert len(out.encode()) < mm.SUMMARY_BUDGET
    assert out.startswith("MEAS-RING") and "bw_ach=4.05TB/s" in out and "rs=8/8/2x8" in out and "MEAS-RING done" in out and "skipped" not in out
    for m in mm.MS:
        assert f"I moe m={m} seeds=11/12/13" in out
        names = ("v5", "ring", "rb3", "ring82", "wstream", "ringnb1", "zflat", "pdloff") if m > 16 else ("v5", "rb7", "ring81", "wstream", "rb7nb2", "zflat", "pdloff")
        for v in names:
            assert re.search(rf"^B moe m={m} v={v} \| g1=[\d.]+/[\d.]+/[\d.]+ g2=[\d.]+/[\d.]+/[\d.]+ \| tb=[\d.]+/[\d.]+ ach=\d+/\d+% x=", out, re.M), (m, v)
        if m > 16:
            assert re.search(rf"^S moe m={m} g1 cta=[\d.]+us item=[\d.]+us span=\d+ stage=\d+% stream=\d+% wait=\d+% fix=\d+% epi=\d+% bw/cta=[\d.]+GB/s occ=3/2.5 \| g2 ", out, re.M)
        else:
            assert not re.search(rf"^S moe m={m} ", out, re.M)
    assert out.count("pair=") == 3 * len(mm.MS)     # v5, the default (ring | rb7) and pdloff carry the pair time
    lines = out.splitlines()
    assert all(len(l) < 280 for l in lines) and len(lines) <= 6 * 14 + 3


def test_measure_budget_skips_variants_not_seeds(mm, monkeypatch, capsys):
    state = _fake_stages_patch(mm, monkeypatch)
    env = _FakeEnv(None)
    env.clock = lambda: (setattr(env, "t", env.t + 2.0) or env.t)
    mm._measure(state, env, t_start=0.0)
    out = capsys.readouterr().out
    assert "MEAS-RING done" in out and "skipped:" in out and "B moe m=256 v=v5" in out and len(out.encode()) < mm.SUMMARY_BUDGET


@needs_cuda
def test_measure_real_env_on_dev_card(mm, mod, v4, full_state, nv, monkeypatch, capsys):
    """The real _RealEnv on the 2060 with the dev-card ring depths (RS 4 / 3, small ring D 1 RS 4; NVRTC build of the measurement extension incl. the
    WSTREAM / CAL kernels): graph timing after flushes, the variants, the stamps, the summary; every variant is bitwise the pivot (bw=OK). The pivot is v5's
    kernel through v5's extension (built here by NVRTC too)."""
    import nvrtc_smallm as nvm

    w, state = full_state
    monkeypatch.setattr(mm, "RING_LARGE_RS", 4)
    monkeypatch.setattr(mm, "RING82_RS", 3)
    monkeypatch.setattr(mm, "RING_SMALL_DR", (1, 4))
    inst = mm.meas_instances()
    ext = nvm.NvrtcMeasExt(mm.meas_device_src(), mm.meas_names(inst))
    ext.moe6_gemm = nvm.NvrtcMoe6Ext.moe6_gemm.__get__(ext)
    ext.moe6_route = nvm.NvrtcMoe6Ext.moe6_route.__get__(ext)
    ext.moe6_init = lambda: None
    v5ext = nvm.NvrtcMoe5Ext(mm.moe5_device_src(), mm.moe5_instance_names([(True, 8, 1, 2, 0), (False, 8, 1, 2, 0), (True, 4, 2, 3, 0), (False, 4, 2, 3, 0)]))
    monkeypatch.setattr(mm, "HAVE_SGLANG", True)
    monkeypatch.setattr(mm, "SMALLM_CFG", DEV_SMALLM_CFG)
    monkeypatch.setattr(mm, "MOE6_CFG", DEV_MOE6_CFG)
    monkeypatch.setattr(mm, "MOE5_CFG", tuple((mx, (8, 1, 4, 8, 1, 2), (8, 1, 4, 8, 1, 2)) if mx <= 16 else (mx, (4, 2, 8, 24, 4, 3), (4, 2, 8, 24, 2, 3))
                                            for mx in (1, 2, 4, 8, 12, 16, 32, 128, 256)))
    for b_, c_ in DEV_LARGEM_CFG.items():
        monkeypatch.setitem(mm.LARGEM_CFG, b_, c_)
    monkeypatch.setattr(mm, "MS", (32, 8))
    monkeypatch.setattr(mm, "SEEDS", (11,))
    monkeypatch.setattr(mm, "REPS", 3)
    st = {"smallm": state["smallm"], "largem": state["largem"], "norm_w": state["norm_w"],
          "moe5": mm.prepare_moe5(state["smallm"], state["largem"], ext=v5ext), "moe6": mm.prepare_moe6(state["smallm"], state["largem"], ext=ext)}
    mm._measure(st)
    out = capsys.readouterr().out
    with capsys.disabled():
        print("\n" + out)
    assert out.startswith("MEAS-RING") and "MEAS-RING done" in out and len(out.encode()) < mm.SUMMARY_BUDGET
    assert re.search(r"bw_ach=[\d.]+TB/s", out) and "rs=4/3/1x4" in out and "I moe m=32 seeds=11" in out and "I moe m=8 seeds=11" in out
    for v in ("v5", "ring", "rb3", "ring82", "ringnb1", "rb3nb1", "ring82nb1", "ringnb4", "zflat", "pdloff"):
        mt = re.search(rf"^B moe m=32 v={v} \| .* bw=(\S+)", out, re.M)
        assert mt, v
        assert mt.group(1) == "OK", (v, mt.group(0))
    assert re.search(r"^B moe m=32 v=wstream \| g1=[\d.]+ g2=[\d.]+ \| tb=", out, re.M)
    assert re.search(r"^S moe m=32 g1 cta=[\d.]+us item=[\d.]+us span=\d+ stage=\d+% stream=\d+% .* occ=\d+/[\d.]+ \| g2 .* \| pdl g2t0=-?[\d.]+ g2t1=-?[\d.]+ g2end=-?[\d.]+$", out, re.M)
    for v in ("v5", "rb7", "ring81", "rb7nb2", "ring81nb2", "zflat", "pdloff"):
        mt = re.search(rf"^B moe m=8 v={v} \| .* bw=(\S+)", out, re.M)
        assert mt and mt.group(1) == "OK", (v, mt.group(0) if mt else None)
    assert "skipped" not in out
    assert int(state["largem"]["cnt"].abs().sum()) == 0
