"""kernels/muse_moe_w4a16/sonnet_v7.py: the persistent warp-specialised MoE kernel (card C30, design/fable-10-out.md D2 / D4) on top of sonnet_v5.

CPU: file rules, v5 region verbatim, config table / bitwise rule / ring geometry, host-source generation, dispatch, prepare() fallbacks, the self-check gate.
GPU dev card (RTX 2060, sm_75, no nvcc, 64 KB smem per block): NVRTC builds of v4's / v5's / v7's device sources. The sm_75 build takes v7's PORTABLE FALLBACK
(smem counters instead of mbarriers, lane-loop copies instead of cp.async.bulk, bounded spins); same call sites and the same control flow as the sm_90 build, so
the producer / consumer protocol, the item queue, the descriptor ring, the fixup and the epilogue are tested for real; timing there means nothing.
  V1  v7 BITWISE v5 on reference.generate_input routing for M in {1, 2, 4, 8, 12, 16, 32, 33, 128, 256} x 3 seeds (dev-card tables: K split 4 below 17 tokens,
      8 above; rings of 3 slots instead of the judge's 8, which needs 148 KB of smem), plus the ring depth / grid size / D sweeps;
  V2  the judge's check (reference.check_implementation) and graph capture / replay; v7's route kernel == v5's and zeroes the item counters;
  V3  producer coverage (DBG bitmap: every (z, ks, ng) issued exactly once, every stage once, every consumer warp once, the bytes add up);
  V4  bounded spins: no deadlock, a never-filled stage makes the consumers time out and flags an error instead of hanging;
  V5  the DBG twins (STAMPS, NOCOMPUTE, CONSONLY) run and report;
  V6  LAST: the sm_90 assembly of every instantiation (ptxas -v: registers, spills, smem; PTX: cp.async.bulk, mbarrier, no cp.async/ld.global.nc in the kernel).

Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v7.py"""
import importlib.util
import time
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "problems" / "muse_moe_w4a16"))

KERNEL = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v7.py"
V5 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v5.py"
V4 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v4.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# all template instantiations the tests assemble: the board's four (D 1, D 2), D 4 and the DBG = 1 twins (the measurement file's list, less D 4's twin)
ALL_INSTANCES = ((True, 1, 0), (False, 1, 0), (True, 2, 0), (False, 2, 0), (True, 4, 0), (False, 4, 0), (True, 1, 1), (False, 1, 1), (True, 2, 1), (False, 2, 1))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v7_moe", KERNEL)


@pytest.fixture(scope="module")
def v4():
    return _load("_sonnet_v4_moe_for_v7", V4)


# --------------------------------------------------------------------------------------------- CPU
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\(", r"urllib"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert "okc-diag HAVE_SGLANG" in src
    assert re.search(r"^# PREDICTION: GEMM1 bs256 234 -> 160 \(kill > 205\), GEMM2 124 -> 82; moe 1.486 -> 1.9; bitwise v5 on all buckets$", src, re.M)
    assert "DECISION RULE" in src and "w4_moe7_kernel" in src
    assert "__syncthreads" in src      # the setup barrier only: the kernel body's later barriers are the named barrier (checked below)
    compile(src, "sonnet_v7.py", "exec")


def test_kernel_body_has_one_block_barrier_and_no_spin_without_bound():
    """__syncthreads appears only in the MOE5 region (v5) and once in the v7 kernel (setup); the consumers' barriers are `bar.sync 1, 384`."""
    src = KERNEL.read_text()
    body = src[src.index("MOE7 BEGIN"):src.index("MOE7 END")]
    k = body[body.index("w4_moe7_kernel(const u8*"):body.index("m7_route_small_kernel(")]
    assert k.count("__syncthreads()") == 1
    assert "bar.sync 1, 384" in body and body.count("m7_cbar();") >= 2


def test_v5_region_is_verbatim():
    """Everything of v5 from `import re` up to its entry points (v4's region, the MOE5 block, its self-checks) is in v7 unchanged."""
    a, b = V5.read_text(), KERNEL.read_text()
    lo, hi = "import re\n", "# ============================================================================================ entry points"
    assert a[a.index(lo):a.index(hi)] in b
    assert b.count("# ============================================================================================ MOE7 BEGIN") == 1
    c = V4.read_text()
    assert c[c.index(lo):c.index(hi)] in b


def test_cfg_table_buckets_ring_geometry_and_smem(mod):
    assert [mod.moe7_bucket(m) for m in (1, 2, 3, 4, 8, 9, 12, 16, 17, 32, 33, 128, 129, 256, 257)] == [1, 2, 4, 4, 8, 12, 12, 16, 32, 32, 128, 128, 256, 256, None]
    assert mod.moe7_cfg(257) is None and mod.moe7_cfg(17) == mod.moe7_cfg(32)
    old = dict(mod.MOE7_ENABLE)
    try:
        mod.MOE7_ENABLE[128] = False
        assert mod.moe7_cfg(100) is None and mod.moe7_cfg(256) is not None
    finally:
        mod.MOE7_ENABLE.update(old)
    assert set(mod.MOE7_INSTANCES) == {(True, 1, 0), (False, 1, 0), (True, 2, 0), (False, 2, 0)}
    for mx, c1, c2 in mod.MOE7_CFG:
        c5 = mod.moe5_cfg(mx)
        assert (c1[2], c2[2]) == (c5[0][2], c5[1][2]), mx       # v5's K split => bitwise v5
        assert mod.moe7_bitwise(mx)
        for fuse, c, groups in ((True, c1, 512), (False, c2, 256)):
            assert (fuse, c[0], 0) in mod.MOE7_INSTANCES
            mod.moe7_check_cfg(mx, 4096, groups, c)
    assert set(mod.moe7_bucket(m) for m in mod.MOE7_SELFCHECK_MS) == {mx for mx, _a, _b in mod.MOE7_CFG}
    # design D2 s2.3: S 8 x (13.8 KB weights + scales + 4.6 KB activations) at 32 rows = 147 KB ring, 148 KB total with the LUT
    assert mod.moe7_slot_bytes(1, 32) == 18432 and mod.moe7_smem(256, (1, 8, 4)) == 1024 + 8 * 18432 <= 227 * 1024
    assert mod.moe7_slot_bytes(2, 32) == 36864 and 1024 + 5 * 36864 <= 227 * 1024                      # D 2, S 5: 185 KB
    for d_ in (1, 2, 4):
        for r_ in (8, 16, 32):
            assert mod.moe7_slot_bytes(d_, r_) >= d_ * (24 * 576 + r_ * 144) and mod.moe7_slot_bytes(d_, r_) % 128 == 0      # one weight run + one activation run per stage
    assert mod.moe7_slot_bytes(1, 8) < mod.moe7_slot_bytes(1, 16) < mod.moe7_slot_bytes(1, 32) and mod.moe7_slot_bytes(1, 8) % 128 == 0
    for mx, c1, c2 in mod.MOE7_CFG:
        for c in (c1, c2):
            assert mod.moe7_smem(mx, c) <= 227 * 1024
    # a bucket whose K split differs is not bitwise (gated on the relative L2 bound)
    old = mod.MOE7_CFG
    try:
        mod.MOE7_CFG = tuple((mx, (1, 8, 8), c2) if mx == 8 else (mx, c1, c2) for mx, c1, c2 in old)
        assert not mod.moe7_bitwise(8) and mod.moe7_bitwise(4)
    finally:
        mod.MOE7_CFG = old
    # geometry rules
    for bad in ((3, 8, 4), (1, 1, 4), (1, 17, 4), (1, 8, 9), (1, 8, 3)):
        with pytest.raises(AssertionError):
            mod.moe7_check_cfg(256, 4096, 512, bad)
    with pytest.raises(AssertionError):
        mod.moe7_check_cfg(256, 4096, 512, (2, 16, 4))     # smem over 227 KB
    with pytest.raises(AssertionError):
        mod.moe7_check_cfg(256, 4096, 512, (2, 16, 8))     # 4 * stages < slots: the descriptor ring could be overwritten
    mod.moe7_check_cfg(8, 4096, 512, (2, 5, 2))
    mod.moe7_check_cfg(256, 4096, 512, (2, 5, 4))


def test_host_source_generation(mod):
    src = mod.moe7_host_src(mod.MOE7_INSTANCES)
    for f, d, dbg in mod.MOE7_INSTANCES:
        t = f"{'true' if f else 'false'}, {d}, {dbg}"
        assert f"M7_CASE({t})" in src and f"m7_init_one((const void*)w4_moe7_kernel<{t}>," in src and f"M7_OCC_CASE({t})" in src
    # C30b: the attribute is optin - static (never the bare 227 KB), every CUDA call of the launch path is checked and names itself, the launch error carries its arguments
    assert "int dyn = optin - (int)a.sharedSizeBytes;" in src and "cudaFuncAttributeMaxDynamicSharedMemorySize, dyn" in src
    assert "cudaFuncAttributeMaxDynamicSharedMemorySize, M7_SMEM_MAX" not in src
    assert "cudaFuncAttributePreferredSharedMemoryCarveout" in src and "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in src
    assert "const cudaError_t e = cudaLaunchKernelExC" in src and "dynamic_smem=" in src and "cudaFuncGetAttributes" in src and "cudaDevAttrMaxSharedMemoryPerBlockOptin" in src
    assert "std::string moe7_init()" in src and "moe7_occupancy" in mod._MOE7_CPP_SRC
    assert "@CASES@" not in src and "@INITS@" not in src
    assert mod.moe7_instance_names([(True, 1, 1)])[0] == "w4_moe7_kernel<true,1,1>" and mod.moe7_instance_names([])[-3:] == ["m7_route_small_kernel", "m7_route_large_kernel", "m7_gather_kernel"]
    dev = mod.moe7_device_src()
    assert dev.startswith(mod.moe5_device_src())
    for tok in ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes", "mbarrier.try_wait.parity", "mbarrier.arrive.expect_tx", "mbarrier.init",
                "fence.proxy.async", "bar.sync 1, 384", "__launch_bounds__(M7_THREADS, 1)", "if (tid < 2) ctr[tid] = 0;"):
        assert tok in dev, tok
    assert "#if __CUDA_ARCH__ >= 900" in dev
    for name in ("moe7_gemm", "moe7_route", "moe7_gather", "moe7_init"):
        assert name in mod._MOE7_CPP_SRC
    assert "cudaLaunchKernelExC" in src and "(unsigned)nctas" in src and "M7_THREADS" in src


H200_OPTIN = 232448      # sharedMemPerBlockOptin of the H200 (sm_90)
STATIC_SMEM = 480        # the kernel's static smem (ptxas)


def test_h200_limits_every_configured_launch_fits(mod):
    """C30b: emulate the H200's limits on the CPU: for every (bucket, GEMM) of the board table AND every launch shape of the measurement file the requested dynamic
    smem + the static smem fits sharedMemPerBlockOptin = 232448; the attribute value the host code sets (optin - static) covers it; the instantiation exists. The first
    judge run died here: the attribute was set to 232448 with 480 B of static smem on top (cudaErrorInvalidValue), so every launch with > 48 KB failed."""
    mm = _load("_measure_ws_for_h200", ROOT / "kernels" / "muse_moe_w4a16" / "measure_ws_v1.py")
    attr = min(232448, H200_OPTIN - STATIC_SMEM)
    assert attr == 231968 and attr + STATIC_SMEM <= H200_OPTIN
    insts = set(mod.MOE7_INSTANCES)
    for mx, c1, c2 in mod.MOE7_CFG:
        for fuse, c in ((True, c1), (False, c2)):
            smem = mod.moe7_smem(mx, c)
            assert smem <= attr and smem + STATIC_SMEM <= H200_OPTIN, (mx, c, smem)
            assert (fuse, c[0], 0) in insts
    minst = set(mm.MEAS_INSTANCES)
    for m in mm.MS:
        for name, c1, c2, fl, dbgt, pdl in mm.variants(m):
            for g, (fuse, c) in enumerate(((True, c1), (False, c2))):
                smem = mod.moe7_smem(m, c)
                assert smem <= attr and smem + STATIC_SMEM <= H200_OPTIN, (m, name, c, smem)
                assert (fuse, c[0], dbgt[g]) in minst
    # the failure mode itself: the old attribute (227 KB) plus the static smem is over the limit
    assert 232448 + STATIC_SMEM > H200_OPTIN


class _FakeExt:
    """An extension double that applies the H200's rules: occupancy 0 when static + dynamic > optin; moe7_init reports like the real one."""

    def __init__(self, optin=H200_OPTIN):
        self.optin = optin
        self.inited = []

    def moe7_init(self):
        self.inited.append(1)
        return "".join(f"w4_moe7_kernel<T,{d},0> regs=122 maxThreads=416 static={STATIC_SMEM} maxDynamic={self.optin - STATIC_SMEM} optin={self.optin}\n" for d in (1, 2))

    def moe7_occupancy(self, d, dbg_t, fuse, smem):
        return 1 if smem + STATIC_SMEM <= self.optin else 0


def test_preflight_marks_unlaunchable_buckets_without_launching(mod, monkeypatch, capsys):
    """C30b: prepare_moe7's pre-flight runs BEFORE any launch: buckets whose smem + static exceeds the device's optin (or whose occupancy is 0) are listed in
    bad_buckets with the reason, the self-check skips them (False -> v5 serves them), the others stay active; the attribute-set path runs once."""
    class P:
        multi_processor_count = 132
        shared_memory_per_block_optin = 100 * 1024     # a device smaller than the H200: the 17..256 buckets (183 KB) cannot launch, the small ones (175-193 KB) neither
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: P)
    ext = _FakeExt(optin=100 * 1024)

    class Dev:
        pass
    sm = {"w1": torch.zeros(1), "s1": None, "w2": None, "s2": None, "dims": {"k2": 4096, "g1": 512}}
    monkeypatch.setattr(mod, "pack_w7_blob", lambda w, s_: None)
    m7 = mod.prepare_moe7(sm, {}, ext=ext, nctas=132)
    assert ext.inited == [1] and m7["static"] == STATIC_SMEM and m7["optin"] == 100 * 1024
    assert set(m7["bad_buckets"]) == {mx for mx, _a, _b in mod.MOE7_CFG}
    out = capsys.readouterr().out
    assert "FAIL: dynamic smem" in out and "sharedMemPerBlockOptin=102400" in out and "maxDynamic=" in out
    # the H200 itself: everything launches
    P.shared_memory_per_block_optin = H200_OPTIN
    m7 = mod.prepare_moe7(sm, {}, ext=_FakeExt(), nctas=132)
    assert m7["bad_buckets"] == {}
    assert "ok" in capsys.readouterr().out
    # occupancy 0 from the occupancy calculator is a failure too, and moe7_check_launch raises (no launch)
    ext0 = _FakeExt()
    ext0.moe7_occupancy = lambda d, dbg_t, fuse, smem: 0
    m70 = mod.prepare_moe7(sm, {}, ext=ext0, nctas=132)
    assert set(m70["bad_buckets"]) == {mx for mx, _a, _b in mod.MOE7_CFG} and "occupancy 0" in m70["bad_buckets"][256]
    with pytest.raises(RuntimeError, match="pre-flight"):
        mod.moe7_check_launch(m70, 256, mod.moe7_cfg(256)[0], True)
    mod.moe7_check_launch(m7, 256, mod.moe7_cfg(256)[0], True)


def test_selfcheck_skips_and_prepare_state_marks_bad_buckets(mod, monkeypatch):
    monkeypatch.setattr(mod, "prepare_moe7", lambda sm, lm, ext=None, flags=(), nctas=None: {"bad_buckets": {256: "x", 8: "y"}})
    monkeypatch.setattr(mod, "SELFCHECK", False)
    st = {"smallm": object(), "largem": object(), "norm_w": None}
    mod._prepare_moe7_state(st, ())
    assert st["moe7_ok"][256] is False and st["moe7_ok"][8] is False and st["moe7_ok"][128] is True
    monkeypatch.setattr(mod, "SELFCHECK", True)
    monkeypatch.setattr(mod, "_selfcheck_inputs", lambda m, seed, device: (_ for _ in ()).throw(RuntimeError("must not run for a bad bucket")) if m in (8, 256) else {"m": m})
    monkeypatch.setattr(mod, "_v5_path", lambda d, s: torch.zeros(2))
    monkeypatch.setattr(mod, "_kernel_moe7", lambda d, s: torch.zeros(2))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    res = mod._moe7_selfcheck({"norm_w": torch.zeros(1), "moe7": {"bad_buckets": {256: "x", 8: "y"}}})
    assert res[256] is False and res[8] is False and res[128] is True and res[1] is True


def test_dispatch(mod, monkeypatch):
    monkeypatch.setattr(mod, "HAVE_SGLANG", True)
    for n, tag in (("_kernel_moe7", "7"), ("_kernel_moe5", "5"), ("_kernel_smallm", "S"), ("_kernel_largem", "L"), ("_kernel_sglang", "M")):
        monkeypatch.setattr(mod, n, lambda d, s, tag=tag: tag)

    def data(m, start=0, count=32):
        return {"hidden_states": torch.zeros(m, 4), "config": {"local_start": start, "local_count": count}}

    ok = {mx: True for mx, _a, _b in mod.MOE7_CFG}
    st = {"smallm": object(), "largem": object(), "largem_ok": {32: True, 128: True, 256: True}, "moe5": object(), "moe5_ok": dict(ok),
          "moe7": object(), "moe7_ok": dict(ok)}
    assert [mod.kernel_fn(data(m), st) for m in (1, 8, 16, 17, 32, 33, 128, 129, 256)] == ["7"] * 9
    assert [mod.kernel_fn(data(m), st) for m in (257, 512)] == ["M", "M"]
    assert mod.kernel_fn(data(100, 32, 32), st) == "M" and mod.kernel_fn(data(100, 0, 256), st) == "M"      # local_start != 0 / count != 32: Marlin
    st2 = dict(st, moe7_ok={**ok, 128: False, 8: False})
    assert [mod.kernel_fn(data(m), st2) for m in (8, 16, 100, 200)] == ["5", "7", "5", "7"]                  # a failed bucket falls back to v5
    assert mod.kernel_fn(data(100), dict(st, moe7=None)) == "5"
    assert mod.kernel_fn(data(100), dict(st, moe7=None, moe5=None)) == "L"
    assert mod.kernel_fn(data(8), dict(st, moe7=None, moe5=None, smallm=None)) == "M"
    assert mod.kernel_fn(data(8), dict(st, largem=None)) == "S"          # v7 (and v5) need v4's large-M scratch


def test_prepare_moe7_state_fallbacks(mod, monkeypatch):
    monkeypatch.setattr(mod, "prepare_moe7", lambda sm, lm, ext=None, flags=(), nctas=None: {"x": 1})
    monkeypatch.setattr(mod, "_moe7_selfcheck", lambda st: {1: True, 2: True, 4: False, 8: True, 12: True, 16: True, 32: True, 128: False, 256: True})
    st = {"smallm": object(), "largem": object(), "norm_w": None}
    mod._prepare_moe7_state(st, ())
    assert st["moe7"] == {"x": 1} and st["moe7_ok"][4] is False and st["moe7_ok"][128] is False and st["moe7_ok"][256] is True
    monkeypatch.setattr(mod, "_moe7_selfcheck", lambda st: {b: False for b in (1, 2, 4, 8, 12, 16, 32, 128, 256)})
    st = {"smallm": object(), "largem": object()}
    mod._prepare_moe7_state(st, ())
    assert st["moe7"] is None and not any(st["moe7_ok"].values())

    def boom(sm, lm, ext=None, flags=(), nctas=None):
        raise RuntimeError("no nvcc")
    monkeypatch.setattr(mod, "prepare_moe7", boom)
    st = {"smallm": object(), "largem": object()}
    mod._prepare_moe7_state(st, ())
    assert st["moe7"] is None
    for st in ({"smallm": None, "largem": object()}, {"smallm": object(), "largem": None}):
        mod._prepare_moe7_state(st, ())
        assert st["moe7"] is None and st["moe7_ok"] == {}


def test_selfcheck_gate_logic(mod, monkeypatch):
    """The per-bucket gate: bit-identity where the K split is v5's, the relative L2 bound otherwise; any exception or a missing v5/v4 path = False."""
    dev = torch.device("cpu")
    base = torch.randn(4, 8).to(torch.bfloat16)

    def run(got_fn, bitwise):
        monkeypatch.setattr(mod, "_selfcheck_inputs", lambda m, seed, device: {"m": m})
        monkeypatch.setattr(mod, "_v5_path", lambda d, s: base)
        monkeypatch.setattr(mod, "_kernel_moe7", lambda d, s: got_fn())
        monkeypatch.setattr(mod, "moe7_bitwise", lambda m: bitwise)
        monkeypatch.setattr(mod, "MOE7_SELFCHECK_MS", (1, 8))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
        return mod._moe7_selfcheck({"norm_w": torch.zeros(1, device=dev)})

    assert run(lambda: base.clone(), True) == {1: True, 8: True}
    flipped = base.clone()
    flipped.view(torch.int16)[0, 0] ^= 1
    assert run(lambda: flipped, True) == {1: False, 8: False}                 # one bit differs: DIFF where bitwise is required
    assert run(lambda: flipped, False) == {1: True, 8: True}                  # ... but inside the L2 bound where the K split differs
    far = base.clone()
    far[0] = far[0] * 1.5 + 1
    assert run(lambda: far, False) == {1: False, 8: False}
    monkeypatch.setattr(mod, "_v5_path", lambda d, s: None)
    monkeypatch.setattr(mod, "_kernel_moe7", lambda d, s: base)
    assert mod._moe7_selfcheck({"norm_w": torch.zeros(1)}) == {1: False, 8: False}

    def boom():
        raise RuntimeError("launch failed")
    assert run(boom, True) == {1: False, 8: False}
    # after MOE7_MAX_FAILS failing sizes the rest is not run (a deadlocking ring would burn prepare()'s budget) and counts as failed
    calls = []
    monkeypatch.setattr(mod, "_selfcheck_inputs", lambda m, seed, device: calls.append(m) or {"m": m})
    monkeypatch.setattr(mod, "_v5_path", lambda d, s: base)
    monkeypatch.setattr(mod, "_kernel_moe7", lambda d, s: flipped)
    monkeypatch.setattr(mod, "moe7_bitwise", lambda m: True)
    monkeypatch.setattr(mod, "MOE7_SELFCHECK_MS", (1, 2, 4, 8, 12, 16))
    res = mod._moe7_selfcheck({"norm_w": torch.zeros(1)})
    assert calls == [1, 2] and res == {1: False, 2: False, 4: False, 8: False, 12: False, 16: False}


# --------------------------------------------------------------------------------------------- GPU: NVRTC on the dev card
@pytest.fixture(scope="module")
def nv():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    return nv


V5_DEV_INSTANCES = [(True, 8, 1, 2, 0), (False, 8, 1, 2, 0), (True, 4, 2, 3, 0), (False, 4, 2, 3, 0)]
# dev-card tables (64 KB smem per block): K split 4 at M <= 16 (16-row tiles of Kc 1024), 8 above (32-row tiles of Kc 512); v7: 3-slot rings
DEV_SMALLM_CFG = ((16, 8, 4, 8, 8, 4, 8),)
DEV_LARGEM_CFG = {b: dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24) for b in (32, 128, 256)}
DEV_MOE5_CFG = tuple((mx, (8, 1, 4, 8, 1, 2), (8, 1, 4, 8, 1, 2)) if mx <= 16 else (mx, (4, 2, 8, 24, 4, 3), (4, 2, 8, 24, 2, 3))
                     for mx in (1, 2, 4, 8, 12, 16, 32, 128, 256))
DEV_RING = 3
DEV_MOE7_CFG = tuple((mx, (1, DEV_RING, 4), (1, DEV_RING, 4)) if mx <= 16 else (mx, (1, DEV_RING, 8), (1, DEV_RING, 8)) for mx in (1, 2, 4, 8, 12, 16, 32, 128, 256))


def _full_weights():
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(5252)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=g, device="cuda", leading=(32,))
    return {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
            "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device="cuda")).to(torch.bfloat16)}


@pytest.fixture(scope="module")
def full_state(mod, nv):
    """v4's state (packed weights + scratch, NVRTC extensions) with the dev-card tables, v5's extension (the bitwise reference) and v7's, 7 CTAs (fewer than
    items, more than SMs would be fine too: the CTAs only talk through the item counter)."""
    w = _full_weights()
    saved = (mod.SMALLM_CFG, dict(mod.LARGEM_CFG), mod.MOE5_CFG, mod.MOE7_CFG, mod.M7_SPIN)
    mod.M7_SPIN = 1 << 27       # the dev card's emulated mma / lane-loop copies make a legitimate wait 1000x longer than an H200's
    mod.SMALLM_CFG = DEV_SMALLM_CFG
    mod.LARGEM_CFG.clear()
    mod.LARGEM_CFG.update(DEV_LARGEM_CFG)
    mod.MOE5_CFG, mod.MOE7_CFG = DEV_MOE5_CFG, DEV_MOE7_CFG
    sm = mod.prepare_smallm(w, ext=nv.NvrtcExt(mod._SMALLM_DEVICE_SRC))
    lm = mod.prepare_largem(sm, ext=nv.NvrtcLargeExt(mod._SMALLM_DEVICE_SRC + mod._LARGEM_EXTRA_SRC))
    m5 = mod.prepare_moe5(sm, lm, ext=nv.NvrtcMoe5Ext(mod.moe5_device_src(), mod.moe5_instance_names(V5_DEV_INSTANCES)))
    m7 = mod.prepare_moe7(sm, lm, ext=nv.NvrtcMoe7Ext(mod.moe7_device_src(), mod.moe7_instance_names(ALL_INSTANCES)), nctas=7)
    m7e = mod.prepare_moe7(sm, lm, ext=nv.NvrtcMoe7Ext(mod.moe7_device_src(), mod.moe7_instance_names(ALL_INSTANCES), extra=("-DM7_EMU_MBAR=1",)), nctas=7)
    state = {"moe7_emu": m7e, "smallm": sm, "largem": lm, "largem_ok": {32: True, 128: True, 256: True}, "moe5": m5, "moe5_ok": {mx: True for mx, _a, _b in DEV_MOE5_CFG},
             "moe7": m7, "moe7_ok": {mx: True for mx, _a, _b in DEV_MOE7_CFG}, "norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    yield w, state
    mod.SMALLM_CFG, l, mod.MOE5_CFG, mod.MOE7_CFG, mod.M7_SPIN = saved
    mod.LARGEM_CFG.clear()
    mod.LARGEM_CFG.update(l)
    del sm, lm, m5, m7, state, w
    torch.cuda.empty_cache()


def _v5_experts(mod, state, d):
    return mod.moe5_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe5"], 0, 32, mod.moe5_cfg(d["hidden_states"].shape[0]))


def _live_rows(y, local):
    return y.reshape(-1, y.shape[-1]).view(torch.int16)[local.reshape(-1) >= 0]


def _v7_experts(mod, state, d, cfg=None, flavor="counters", **kw):
    """flavor: "counters" = the portable fallback (smem counters); "emu" = the software-mbarrier build (the sm_90 protocol on synchronous copies)."""
    m = d["hidden_states"].shape[0]
    return mod.moe7_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe7_emu" if flavor == "emu" else "moe7"], 0, 32,
                            mod.moe7_cfg(m) if cfg is None else cfg, **kw)


FLAVORS = ["counters", "emu"]


@needs_cuda
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("bs", [1, 2, 4, 8, 12, 16, 32, 33, 128, 256])
def test_v1_bitwise_vs_v5(mod, full_state, bs, flavor):
    """V1: v7's expert outputs equal v5's bit for bit on 3 generate_input seeds (normal, normal, skewed); the counters are back to zero; deterministic."""
    import reference

    w, state = full_state
    for i, dist in enumerate(("normal", "normal", "skewed")):
        d = reference.generate_input(bs, "ep8", 8100 + bs + i, dist=dist)
        y5, l5 = _v5_experts(mod, state, d)
        y7, l7 = _v7_experts(mod, state, d, flavor=flavor)
        torch.cuda.synchronize()
        assert torch.equal(l5, l7)
        assert torch.equal(_live_rows(y5, l5), _live_rows(y7, l7)), (bs, dist)
        assert int(state["largem"]["cnt"].abs().sum()) == 0 and int(state["smallm"]["cnt"].abs().sum()) == 0
        if i == 0:
            y7b, _ = _v7_experts(mod, state, d, flavor=flavor)
            assert torch.equal(_live_rows(y7, l7), _live_rows(y7b, l7)), "not deterministic"


@needs_cuda
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("bs", [4, 33, 128])
def test_v1b_ring_depth_grid_and_d_sweeps_stay_bitwise(mod, full_state, bs, flavor):
    """Ring depth 2 / 3 / 5 (S is a runtime knob), grid sizes 1 / 3 / 7 / 40 CTAs (the CTAs only talk through the item counter), D 2 (one stage = two k64
    iterations; the dev card holds it at 8 activation rows), PDL off: every output bit equals v5's."""
    import reference

    w, state = full_state
    d = reference.generate_input(bs, "ep8", 8200 + bs)
    y5, l5 = _v5_experts(mod, state, d)
    ref = _live_rows(y5, l5)
    ks = 4 if bs <= 16 else 8
    combos = [((1, 2, ks), (1, 2, ks), 7, (True, True)), ((1, 3, ks), (1, 3, ks), 1, (True, True)), ((1, 3, ks), (1, 3, ks), 3, (True, False)),
              ((1, 3, ks), (1, 3, ks), 40, (False, False))]
    smax = max(s_ for s_ in range(2, 17) if mod.moe7_smem(bs, (1, s_, ks)) <= 64000)         # the dev card's 64 KB: 4 slots at 8 rows, 3 at 16 / 32 rows
    if smax > 3:
        combos += [((1, smax, ks), (1, smax - 1, ks), 7, (True, True))]
    if bs <= 8:
        combos += [((2, 2, ks), (2, 2, ks), 7, (True, True))]
    for c1, c2, nctas, pdl in combos:
        y7, l7 = _v7_experts(mod, state, d, cfg=(c1, c2), pdl=pdl, nctas=nctas, flavor=flavor)
        torch.cuda.synchronize()
        assert torch.equal(l5, l7) and torch.equal(ref, _live_rows(y7, l7)), (c1, c2, nctas, pdl)
    assert int(state["largem"]["cnt"].abs().sum()) == 0


@needs_cuda
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("bs", [1, 4, 8])
def test_v1d_judge_small_m_k_split_2048(mod, full_state, bs, flavor):
    """The judge's small-M configuration (K split 2: a K slice of 2048, 32 stages of D 1 / 16 of D 2 per item) is bitwise v5 at the same split (the dev card holds it
    at 8 activation rows: v5's staged tile is 33 KB, v7's D 2 ring 2 x 29.8 KB)."""
    import reference

    w, state = full_state
    for seed in (8800 + bs, 8900 + bs):
        d = reference.generate_input(bs, "ep8", seed)
        y5, l5 = mod.moe5_experts(d["hidden_states"], d["topk_ids"], state["smallm"], state["largem"], state["moe5"], 0, 32,
                                  ((8, 1, 2, 8, 1, 2), (8, 1, 2, 8, 1, 2)))
        for c in ((2, 2, 2), (1, 4, 2)):
            y7, l7 = _v7_experts(mod, state, d, cfg=(c, c), flavor=flavor)
            torch.cuda.synchronize()
            assert torch.equal(l5, l7) and torch.equal(_live_rows(y5, l5), _live_rows(y7, l7)), (bs, seed, c)
    assert int(state["largem"]["cnt"].abs().sum()) == 0


@needs_cuda
@pytest.mark.parametrize("bs,ks", [(8, 2), (8, 8), (1, 8), (33, 4), (33, 2)])
def test_v1c_other_k_splits_inside_the_l2_bound(mod, full_state, bs, ks):
    """A K split that is not v5's is not bitwise but stays inside the relative L2 bound the per-bucket gate uses (MOE7_TOL), and the output is deterministic."""
    import reference

    w, state = full_state
    d = reference.generate_input(bs, "ep8", 8300 + bs + ks)
    y5, l5 = _v5_experts(mod, state, d)
    y7, l7 = _v7_experts(mod, state, d, cfg=((1, 3, ks), (1, 3, ks)))
    y7b, _ = _v7_experts(mod, state, d, cfg=((1, 3, ks), (1, 3, ks)))
    torch.cuda.synchronize()
    a, b = _live_rows(y5, l5).view(torch.bfloat16).float(), _live_rows(y7, l7).view(torch.bfloat16).float()
    assert float((a - b).norm() / a.norm()) < mod.MOE7_TOL
    assert torch.equal(_live_rows(y7, l7), _live_rows(y7b, l7))
    assert int(state["largem"]["cnt"].abs().sum()) == 0


@needs_cuda
@pytest.mark.parametrize("bs", [1, 4, 33, 128])
def test_v2_judge_check_and_item_counters(mod, full_state, bs):
    """V2: the judge's check (reference.check_implementation) passes on v7's whole kernel_fn path; the route kernel zeroes the item counters inside the call."""
    import reference

    w, state = full_state
    d = reference.generate_input(bs, "ep8", 8400 + bs)
    st = dict(state)
    st["moe7"] = dict(state["moe7"])
    st["moe7"]["ctr"] = torch.full((2,), 12345, dtype=torch.int32, device="cuda")    # garbage from a "previous call": the route kernel must reset it
    out = mod._kernel_moe7(d, st)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(d, out, w)
    assert ok, msg
    nch = int(state["largem"]["ctab"][0])
    ks = mod.moe7_cfg(bs)[0][2]
    ctr = st["moe7"]["ctr"].tolist()
    assert ctr[0] == nch * ks * 22 + 7 and ctr[1] == nch * mod.moe7_cfg(bs)[1][2] * 11 + 7, (ctr, nch)   # items + one sentinel fetch per CTA, from 0
    out2 = mod._kernel_moe7(d, st)
    assert torch.equal(out.view(torch.int16), out2.view(torch.int16))


@needs_cuda
def test_dev_card_attribute_rule_and_preflight(mod, nv, full_state):
    """C30b on the real driver: (1) setting MaxDynamicSharedMemorySize to the device's optin with static smem on top is INVALID_VALUE -- the H200 failure,
    reproduced on the 2060 -- and optin - static is accepted (what moe7_init does); (2) moe7_init reports every instantiation; (3) the occupancy calculator says >= 1
    for the dev configs and 0 for a shape over the limit; (4) a bucket over the limit is refused by the pre-flight and the context stays usable."""
    from cuda.bindings import driver as cu

    w, state = full_state
    m7 = state["moe7"]
    ext = m7["ext"]
    optin = ext._opt_in()
    fn = ext.fn["w4_moe7_kernel<false,1,0>"]
    static = nv._chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn))
    assert static > 0
    res = cu.cuFuncSetAttribute(fn, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin)
    assert res[0] == cu.CUresult.CUDA_ERROR_INVALID_VALUE
    info = ext.moe7_init()
    for f, d, dbg in ALL_INSTANCES:
        assert f"w4_moe7_kernel<{'true' if f else 'false'},{d},{dbg}> regs=" in info
    assert f"maxDynamic={optin - static}" in info and "maxThreads=416" in info
    assert ext.moe7_occupancy(1, 0, True, 1024 + 3 * 18432) >= 1
    assert ext.moe7_occupancy(1, 0, True, optin) <= 0
    assert mod.moe7_preflight(m7, 1, 0, True, optin) is not None and mod.moe7_preflight(m7, 1, 0, True, 1024 + 3 * 18432) is None
    assert m7["bad_buckets"] == {}
    with pytest.raises(RuntimeError, match="pre-flight"):
        mod.moe7_check_launch(m7, 256, (1, 16, 8), True)
    torch.zeros(1, device="cuda").add_(1)
    torch.cuda.synchronize()


@needs_cuda
def test_emu_build_runs_the_mbarrier_protocol(mod, nv):
    """The `emu` flavor really is the mbarrier-protocol build (64-bit CAS state machine, parity waits) and the `counters` flavor is not."""
    name = mod.moe7_instance_names([(False, 1, 0)])[0]
    a, _ = nv.compile_device(mod.moe7_device_src(), "compute_75", want="ptx", names=[name])
    b, _ = nv.compile_device(mod.moe7_device_src(), "compute_75", want="ptx", names=[name], extra=("-DM7_EMU_MBAR=1",))
    assert b"atom.shared.cas.b64" in b and b"atom.shared.cas.b64" not in a
    assert b"ld.volatile.shared.u32" in a


@needs_cuda
@pytest.mark.parametrize("bs", [1, 9, 16, 17, 100, 256])
def test_v2b_route_kernels_equal_v5s(mod, full_state, bs):
    import reference

    w, state = full_state
    sm, lm, m5, m7 = state["smallm"], state["largem"], state["moe5"], state["moe7"]
    d = reference.generate_input(bs, "ep8", 8500 + bs)
    ids = d["topk_ids"].to(torch.int32).contiguous()
    small = bs <= mod.SM_MAX_M
    loc5 = torch.empty_like(ids)
    m5["ext"].moe5_route(ids, loc5, lm["ctab"], lm["slotl"], 0, 32, small)
    t5, s5 = lm["ctab"].clone(), lm["slotl"].clone()
    lm["ctab"].zero_()
    lm["slotl"].zero_()
    loc7 = torch.empty_like(ids)
    ctr = torch.full((2,), 99, dtype=torch.int32, device="cuda")
    sloc = torch.full((2048,), -7, dtype=torch.int32, device="cuda")
    m7["ext"].moe7_route(ids, loc7, lm["ctab"], lm["slotl"], sloc, ctr, 0, 32, small)
    torch.cuda.synchronize()
    assert torch.equal(loc5, loc7) and ctr.tolist() == [0, 0]
    n = int(t5[0])
    pairs = int((loc5 >= 0).sum())
    assert int(lm["ctab"][0]) == n and torch.equal(t5[4:4 + 4 * n], lm["ctab"][4:4 + 4 * n]) and torch.equal(s5[:pairs], lm["slotl"][:pairs])
    # sloc: every live slot maps to (chunk, row) with that chunk's expert == the slot's expert and slotl[offset + row] == the slot
    ct, sl, lo_, sc, lc = lm["ctab"].cpu(), lm["slotl"].cpu(), loc7.reshape(-1).cpu(), sloc.cpu(), None
    for slot in range(bs * 8):
        if int(lo_[slot]) >= 0:
            z, row = int(sc[slot]) >> 5, int(sc[slot]) & 31
            assert z < n and int(ct[4 + 4 * z]) == int(lo_[slot]) and row < int(ct[5 + 4 * z]) and int(sl[int(ct[6 + 4 * z]) + row]) == slot, (slot, z, row)


@needs_cuda
def test_selfcheck_and_graph_capture(mod, full_state):
    """prepare()'s self-check (v7 vs v5) passes with the dev-card tables; the v7 path captures and replays as a CUDA graph (no host sync, counters reset in the call)."""
    import reference

    w, state = full_state
    st = dict(state)
    assert all(mod._moe7_selfcheck(st).values())
    d = reference.generate_input(64, "ep8", 9300)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        mod._kernel_moe7(d, st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod._kernel_moe7(d, st)
    for seed in (9301, 9302, 9303):
        d2 = reference.generate_input(64, "ep8", seed)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            d[k].copy_(d2[k])
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), mod._kernel_moe7(d2, st).view(torch.int16))
        ok, msg = reference.check_implementation(d2, out, w)
        assert ok, msg
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    assert int(state["largem"]["cnt"].abs().sum()) == 0


@needs_cuda
def test_c30c_weight_blob_layout(mod, full_state):
    """C30c: the weight blob is the v4 tile words + scales regrouped, lossless: block b (24 groups, the last one G % 24) of expert e starts at (e G NIT + 24 b NIT) 576 B,
    iteration it of a block of n groups at + it n 576, group gl at + gl 576: 512 B words then 64 B scales. D consecutive iterations of a block are one run."""
    w, state = full_state
    sm, m7 = state["smallm"], state["moe7"]
    for blob, words, scales, G in ((m7["b1"], sm["w1"], sm["s1"], 512), (m7["b2"], sm["w2"], sm["s2"], 256)):
        nit = words.shape[2]
        assert blob.shape == (32, G * nit * 576) and blob.dtype == torch.uint8
        gen = torch.Generator().manual_seed(7)
        for _ in range(300):
            e, g, it = (int(torch.randint(0, n, (1,), generator=gen)) for n in (32, G, nit))
            b, gl = divmod(g, 24)
            nv = min(24, G - b * 24)
            off = (b * 24 * nit + it * nv) * 576 + gl * 576
            tile = blob[e, off:off + 576]
            assert torch.equal(tile[:512], words[e, g, it].reshape(-1).view(torch.uint8)) and torch.equal(tile[512:], scales[e, g, it].reshape(-1))
    assert mod.pack_w7_blob(sm["w2"][:2], sm["s2"][:2]).shape == (2, 256 * 64 * 576)


@needs_cuda
@pytest.mark.parametrize("bs", [5, 33])
def test_c30c_gather_kernel_tiles(mod, full_state, bs):
    """C30c: the gather pre-pass writes [chunk][K/64][R][72] with the chunk's rows (token = slot >> 3), zero rows beyond mc; the pad columns are left alone."""
    import reference

    w, state = full_state
    sm, lm, m7 = state["smallm"], state["largem"], state["moe7"]
    d = reference.generate_input(bs, "ep8", 8950 + bs)
    ids = d["topk_ids"].to(torch.int32).contiguous()
    loc = torch.empty_like(ids)
    m7["ext"].moe7_route(ids, loc, lm["ctab"], lm["slotl"], m7["sloc"], m7["ctr"], 0, 32, bs <= mod.SM_MAX_M)
    R, zb = mod.moe5_rows(bs), mod.moe5_zbound(bs, lm)
    xt = torch.full((mod.moe7_tiled_elems(bs, lm, 4096),), 7.0, device="cuda", dtype=torch.bfloat16)
    m7["ext"].moe7_gather(d["hidden_states"].contiguous(), lm["ctab"], lm["slotl"], xt, zb, 4096, 3, R, True)
    torch.cuda.synchronize()
    ct, sl = lm["ctab"].cpu(), lm["slotl"].cpu()
    v = xt.view(zb, 64, R, 72).cpu().float()
    x = d["hidden_states"].cpu().float()
    for z in range(int(ct[0])):
        mc, off = int(ct[5 + 4 * z]), int(ct[6 + 4 * z])
        for row in range(R):
            want = x[int(sl[off + row]) >> 3].reshape(64, 64) if row < mc else torch.zeros(64, 64)
            assert torch.equal(v[z, :, row, :64], want), (z, row)
            assert bool((v[z, :, row, 64:] == 7.0).all())


def test_c30c_two_bulk_copies_per_stage(mod):
    """C30c: the producer issues TWO bulk copies per stage (one weight+scale run, one activation run), both from the elected lane, no per-lane copy loop; the
    constant the S line / the report quote."""
    assert mod.M7_COPIES_PER_STAGE == 2
    src = mod.moe7_device_src()
    k = src[src.index("w4_moe7_kernel(const u8*"):src.index("// ======================================================= consumers")]
    assert k.count("m7_bulk(") == 3                      # weights, activations, and the activations of the GEMM2 pre-wait stages
    assert "lane < nval" not in k and "lane < mc" not in k
    b = src[src.index("__device__ __forceinline__ void m7_bulk"):]
    assert "if ((threadIdx.x & 31) == 0)" in b[:400]


# ---------------------------------------------------------------------------------------- DBG twins (V3 coverage, V4 bounded spins, V5 twins)
def _dbg_bufs(mod):
    return tuple(torch.zeros(mod.M7_DBG_U64, dtype=torch.int64, device="cuda") for _ in range(2))


@needs_cuda
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("bs", [8, 33])
def test_v3_producer_coverage(mod, full_state, bs, flavor):
    """V3: with the COVER twin every item (z, ks, ng) of both GEMMs is fetched by the producer exactly once, with all its stages, and processed by all 12
    consumer warps; the producer's issued bytes add up to the items' bytes; no error flag; the output is still bitwise v5."""
    import reference

    w, state = full_state
    d = reference.generate_input(bs, "ep8", 8600 + bs)
    y5, l5 = _v5_experts(mod, state, d)
    dbgs = _dbg_bufs(mod)
    c1, c2 = mod.moe7_cfg(bs)
    y7, l7 = _v7_experts(mod, state, d, dbg=dbgs, dbg_mode=(mod.M7_COVER | mod.M7_STAMPS, mod.M7_COVER | mod.M7_STAMPS), dbg_t=(1, 1), flavor=flavor)
    torch.cuda.synchronize()
    assert torch.equal(_live_rows(y5, l5), _live_rows(y7, l7))
    ct = state["largem"]["ctab"].cpu()
    nch = int(ct[0])
    mcs = [int(ct[5 + 4 * z]) for z in range(nch)]
    for dbg, c, groups, k_dim in ((dbgs[0], c1, 512, 4096), (dbgs[1], c2, 256, 4096)):
        dd, s, ks = c
        ni = -(-groups // 24)
        nst = (k_dim // ks // 64) // dd
        n_items = nch * ks * ni
        u32 = dbg.view(torch.int32)
        base = 2 * mod.M7_COV_OFF
        picked = u32[base: base + n_items].cpu()
        staged = u32[base + mod.M7_COV_MAX: base + mod.M7_COV_MAX + n_items].cpu()
        consumed = u32[base + 2 * mod.M7_COV_MAX: base + 2 * mod.M7_COV_MAX + n_items].cpu()
        assert bool((picked == 1).all()), "an item was fetched 0 or 2+ times"
        assert bool((staged == nst).all()) and bool((consumed == 12).all())
        assert int(u32[base + n_items: base + mod.M7_COV_MAX].abs().sum()) == 0        # nothing beyond the items
        assert int(u32[2]) == 0                                                          # no timeout flag
        rec = dbg[mod.M5_OCC: mod.M5_OCC + mod.M7_MAXCTA * 13 * mod.M7_REC].view(mod.M7_MAXCTA, 13, mod.M7_REC)[: state["moe7"]["nctas"]]
        want = 0
        for z in range(nch):
            for ng in range(ni):
                nval = min(24, groups - ng * 24)
                want += ks * nst * dd * (nval * 576 + mod.moe5_rows(bs) * 144)
        assert int(rec[:, 0, 6].sum()) == want                                           # the producer warp's issued bytes
        assert int(rec[:, 0, 4].sum()) == n_items and int(rec[:, 1:, 4].sum()) == 12 * n_items
        assert int(rec[:, 0, 5].sum()) == n_items * nst
        t0, t1 = rec[:, :, 0], rec[:, :, 1]
        assert bool((t1 >= t0).all()) and bool((rec[:, :, 2] >= 0).all()) and bool((rec[:, 1:, 3] >= 0).all())


@needs_cuda
def test_v3b_every_cta_gets_items_or_a_sentinel(mod, full_state):
    """More CTAs than items (bs 1 GEMM2 has few): the surplus CTAs take the sentinel at once and exit; the output is unchanged."""
    import reference

    w, state = full_state
    d = reference.generate_input(1, "ep8", 8700)
    y5, l5 = _v5_experts(mod, state, d)
    y7, l7 = _v7_experts(mod, state, d, nctas=200)
    torch.cuda.synchronize()
    assert torch.equal(_live_rows(y5, l5), _live_rows(y7, l7))


@needs_cuda
@pytest.mark.parametrize("flavor", FLAVORS)
def test_v4_bounded_spin_no_deadlock(mod, full_state, flavor):
    """V4: (a) a normal run with the BOUND twin finishes and flags nothing; (b) a stage that is never filled (FAULT twin) makes the consumers time out and the
    producer's next wait time out as well: the kernel RETURNS (no hang) with the error flag set, the counters of the fixup are left alone."""
    import reference

    w, state = full_state
    d = reference.generate_input(8, "ep8", 8800)
    y5, l5 = _v5_experts(mod, state, d)
    dbgs = _dbg_bufs(mod)
    y7, l7 = _v7_experts(mod, state, d, dbg=dbgs, dbg_mode=(mod.M7_BOUND, mod.M7_BOUND), dbg_t=(1, 1), flavor=flavor)
    torch.cuda.synchronize()
    assert torch.equal(_live_rows(y5, l5), _live_rows(y7, l7)) and int(dbgs[0].view(torch.int32)[2]) == 0 and int(dbgs[1].view(torch.int32)[2]) == 0
    dbgs = _dbg_bufs(mod)

    t = time.time()
    # both GEMMs with the fault; a small spin bound keeps the test fast (a spin iteration on the dev card is ~100 ns)
    y7, l7 = _v7_experts(mod, state, d, dbg=dbgs, dbg_mode=(mod.M7_FAULT | mod.M7_BOUND, mod.M7_FAULT | mod.M7_BOUND), dbg_t=(1, 1), spin_max=1 << 16, flavor=flavor)
    torch.cuda.synchronize()
    assert time.time() - t < 60
    assert int(dbgs[0].view(torch.int32)[2]) == 1 and int(dbgs[1].view(torch.int32)[2]) == 1
    # the kernel left the ring in an arbitrary state, but a following normal call is unaffected (the ring, the barriers and the counters are per launch / per call)
    state["largem"]["cnt"].zero_()
    state["smallm"]["cnt"].zero_()
    y7, l7 = _v7_experts(mod, state, d, flavor=flavor)
    torch.cuda.synchronize()
    assert torch.equal(_live_rows(y5, l5), _live_rows(y7, l7))


@needs_cuda
@pytest.mark.parametrize("flavor", FLAVORS)
def test_v5_twins_run_and_report(mod, full_state, flavor):
    """V5: STAMPS (wait / fix+epi shares are reported per warp), NOCOMPUTE (consumers fold instead of dequant + mma: the output differs, the sink is stored) and
    CONSONLY (the producer re-reads one L2-hot region) run to completion."""
    import reference

    w, state = full_state
    d = reference.generate_input(33, "ep8", 8900)
    y5, l5 = _v5_experts(mod, state, d)
    ref = _live_rows(y5, l5)
    for mode, same in ((mod.M7_STAMPS, True), (mod.M7_NOCOMPUTE, False), (mod.M7_CONSONLY, False), (mod.M7_NOCOMPUTE | mod.M7_STAMPS, False)):
        dbgs = _dbg_bufs(mod)
        y7, l7 = _v7_experts(mod, state, d, dbg=dbgs, dbg_mode=(mode, mode), dbg_t=(1, 1), flavor=flavor)
        torch.cuda.synchronize()
        assert bool(torch.equal(ref, _live_rows(y7, l7))) == same, mode
        assert int(dbgs[0].view(torch.int32)[2]) == 0
        rec = dbgs[0][mod.M5_OCC: mod.M5_OCC + mod.M7_MAXCTA * 13 * mod.M7_REC].view(mod.M7_MAXCTA, 13, mod.M7_REC)[: state["moe7"]["nctas"]]
        if mode & (mod.M7_STAMPS | mod.M7_NOCOMPUTE):
            assert int(rec[:, 1:, 4].sum()) > 0
        if mode & mod.M7_STAMPS:
            assert bool((rec[:, :, 1] > rec[:, :, 0]).all())
            span = (rec[:, :, 1] - rec[:, :, 0]).float()
            cons_wait = (rec[:, 1:, 2].float() / span[:, 1:]).mean()
            assert 0.0 <= float(cons_wait) <= 1.0
        if mode & mod.M7_NOCOMPUTE:
            assert int(rec[:, 1:, 7].abs().sum()) != 0


@needs_cuda
@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("bs", [8, 33])
def test_c31_consumer_twins_remove_exactly_their_part(mod, full_state, bs, flavor):
    """C31: the consumer-chain twins (on top of CONSONLY) run to completion, report a non-zero sink, and remove what they claim: NOMMA / LDSONLY leave the accumulators at
    zero (live output rows all zero: the mma really is gone), NOUNPACK / NOLDM change the numbers (not zero, not v5's), NOFIX never touches the fixup counters / partials and
    writes no output; the plain CONSONLY run stays unharmed (no sink record)."""
    import reference

    w, state = full_state
    d = reference.generate_input(bs, "ep8", 9100 + bs)
    y5, l5 = _v5_experts(mod, state, d)
    ref = _live_rows(y5, l5)
    cons = mod.M7_CONSONLY
    modes = {"nounpack": mod.M7_CO_NOUNPACK, "nomma": mod.M7_CO_NOMMA, "noldm": mod.M7_CO_NOLDM, "nofix": mod.M7_CO_NOFIX, "ldsonly": mod.M7_CO_LDSONLY,
             "smem": mod.M7_NOCOMPUTE}
    for name, bit in modes.items():
        dbgs = _dbg_bufs(mod)
        state["largem"]["cnt"].zero_()
        y7, l7 = _v7_experts(mod, state, d, dbg=dbgs, dbg_mode=(cons | bit, cons | bit), dbg_t=(1, 1), flavor=flavor)
        torch.cuda.synchronize()
        assert int(dbgs[0].view(torch.int32)[2]) == 0 and int(dbgs[1].view(torch.int32)[2]) == 0, name
        rec = dbgs[0][mod.M5_OCC: mod.M5_OCC + mod.M7_MAXCTA * 13 * mod.M7_REC].view(mod.M7_MAXCTA, 13, mod.M7_REC)[: state["moe7"]["nctas"]]
        if name != "noldm":   # (NOLDM keeps the real mma chain: its result is the output, there is nothing to fold)
            assert int(rec[:, 1:, 7].abs().sum()) != 0, name                                  # the sink is stored: the twin's inputs are alive
        assert int(state["largem"]["cnt"].abs().sum()) == 0, name
        if name in ("nomma", "ldsonly", "smem"):
            assert int(_live_rows(y7, l7).abs().sum()) == 0, name                        # accumulators never updated -> SwiGLU(0) = 0 -> 0
        elif name in ("nounpack", "noldm"):
            got = _live_rows(y7, l7)
            assert not torch.equal(got, ref) and int(got.abs().sum()) != 0, name
    dbgs = _dbg_bufs(mod)
    y7, l7 = _v7_experts(mod, state, d, dbg=dbgs, dbg_mode=(cons, cons), dbg_t=(1, 1), flavor=flavor)
    torch.cuda.synchronize()
    rec = dbgs[0][mod.M5_OCC: mod.M5_OCC + mod.M7_MAXCTA * 13 * mod.M7_REC].view(mod.M7_MAXCTA, 13, mod.M7_REC)
    assert int(rec.abs().sum()) == 0 and int(_live_rows(y7, l7).abs().sum()) != 0         # plain CONSONLY: no record, real numbers (the L2-hot source makes them differ from v5's)


def test_c31_twin_bits_are_distinct_and_in_the_source(mod):
    bits = [mod.M7_STAMPS, mod.M7_NOCOMPUTE, mod.M7_CONSONLY, mod.M7_COVER, mod.M7_FAULT, mod.M7_BOUND, mod.M7_CO_NOUNPACK, mod.M7_CO_NOMMA, mod.M7_CO_NOLDM, mod.M7_CO_NOFIX,
            mod.M7_CO_LDSONLY]
    assert len(set(bits)) == len(bits) and all(b & (b - 1) == 0 for b in bits)
    src = mod.moe7_device_src()
    for n, b in (("NOUNPACK", mod.M7_CO_NOUNPACK), ("NOMMA", mod.M7_CO_NOMMA), ("NOLDM", mod.M7_CO_NOLDM), ("NOFIX", mod.M7_CO_NOFIX), ("LDSONLY", mod.M7_CO_LDSONLY)):
        assert f"#define M7_CO_{n} {b}" in src
    assert mod.MOE7_INSTANCES == ((True, 1, 0), (False, 1, 0), (True, 2, 0), (False, 2, 0))           # the twins are run-time modes: no new instantiation


# --------------------------------------------------------------------------------------------- sm_90 audit (LAST in the file)
def _ptxas_log(nv, src, name, arch):
    from cuda.bindings import nvrtc

    # NVRTC caches a compile of identical text and then prints no ptxas -v lines: a nonce global (a comment would be stripped before the cache key) makes every audit a fresh compile
    prog = nv._chk(nvrtc.nvrtcCreateProgram((src + f"\n__device__ long long m7_audit_nonce = {time.time_ns()}LL;\n").encode(), b"k.cu", 0, [], []))
    nv._chk(nvrtc.nvrtcAddNameExpression(prog, name.encode()))
    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17", b"-default-device", b"--ptxas-options=-v"]
    res = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    n = nv._chk(nvrtc.nvrtcGetProgramLogSize(prog))
    log = b" " * n
    nv._chk(nvrtc.nvrtcGetProgramLog(prog, log))
    assert res[0] == nvrtc.nvrtcResult.NVRTC_SUCCESS, log.decode(errors="replace")[:2000]
    return log.decode(errors="replace")


@needs_cuda
@pytest.mark.parametrize("inst", ALL_INSTANCES)
def test_v6_sm90_ptxas_audit(mod, nv, inst, capsys):
    """V6 (LAST): every instantiation assembles for the judge's sm_90 with ptxas: the consumer path fits the register cap (13 warps = 4 on one scheduler => 128
    registers per thread, 65536 / (4 x 32) -- the design assumed 152, which 13 warps cannot have without setmaxnreg), NO spills, static smem as designed, and
    the PTX carries the bulk copies and the mbarriers but no cp.async / ld.global.nc (the weights arrive only through the producer's bulk copies)."""
    name = mod.moe7_instance_names([inst])[0]
    log = _ptxas_log(nv, mod.moe7_device_src(), name, "sm_90")
    mt = re.search(r"Compiling entry function '(_Z14w4_moe7_kernel[^']*)' for 'sm_90'(.*?)Compile time", log, re.S)
    assert mt, log[-600:]
    body = mt.group(2)
    regs = int(re.search(r"Used (\d+) registers", body).group(1))
    smem = int(re.search(r"(\d+) bytes smem", body).group(1))
    sp = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
    with capsys.disabled():
        print(f"\n  {name:28s} regs={regs} stack={sp.group(1)} spill_st={sp.group(2)} spill_ld={sp.group(3)} static_smem={smem}")
    assert regs <= 128 and smem <= 1024
    if inst[2] == 0 or inst[1] == 1:
        assert sp.group(1) == "0" and sp.group(2) == "0" and sp.group(3) == "0", (name, sp.groups())    # ratchet R3: no spills
    if inst[2] == 1 and inst[1] == 2:      # the C31 consumer twins' kernel (run-time modes of the D2 DBG = 1 template): a few bytes at most, none in the plain paths
        assert int(sp.group(2)) <= 16 and int(sp.group(3)) <= 32, (name, sp.groups())
    ptx, _low = nv.compile_device(mod.moe7_device_src(), "compute_90", want="ptx", names=[name])
    ptx = ptx.decode(errors="replace")
    k = ptx[ptx.index(".entry _Z14w4_moe7_kernel"):]
    nxt = k.find(".visible .entry", 10)
    k = k if nxt < 0 else k[:nxt]
    assert len(re.findall(r"cp\.async\.bulk\.shared::cluster\.global\.mbarrier::complete_tx::bytes", k)) >= 2       # weights + activations (+ the GEMM2 pre-wait activations, unrolled); the per-stage count is checked on the source
    assert "mbarrier.try_wait.parity.shared::cta.b64" in k and "mbarrier.arrive.expect_tx" in k and "mbarrier.init" in k
    assert "bar.sync 1, 384" in k and "fence.proxy.async" in k and "griddepcontrol.wait" in k
    assert not re.search(r"cp\.async\.(ca|cg)\b", k) and not re.search(r"ld\.global\.nc(\.L1::no_allocate)?\.v[24]", k)   # (the 256-word LUT is a scalar ld.global.nc.u32)
    assert not re.search(r"ld\.global[^;]*\.v4", k.replace("ld.global.cg.v4", ""))        # the fixup's ld.global.cg.v4 partials are the only 16-B global loads
    assert "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32" in k and "ldmatrix.sync.aligned.m8n8.x4.shared.b16" in k
    if inst[0]:
        assert "griddepcontrol.launch_dependents" in k


@needs_cuda
def test_v6b_sm90_ring_smem_fits_one_cta_per_sm():
    """The designed smem: D 1 / S 8 at 32 rows = 148480 B dynamic + 480 B static (+ 1 KB reserved) <= 227 KB, and two CTAs cannot share an SM (so the grid of 132
    CTAs is one per SM and the GEMM2 CTAs of a PDL launch wait for a free SM instead of co-residing)."""
    mod = _load("_sonnet_v7_fresh", KERNEL)       # a fresh load: the GPU fixtures replace the module-level dev-card tables of the shared `mod` until teardown
    dyn = mod.moe7_smem(256, (1, 8, 4))
    assert dyn == 148480 and dyn + 480 + 1024 <= 227 * 1024 and 2 * (dyn + 480 + 1024) > 228 * 1024      # the design's D 1 / S 8
    assert mod.moe7_smem(8, (1, 8, 2)) * 2 > 228 * 1024            # also at 8 rows (121 KB): still one CTA per SM
    for m_ in (1, 8, 16, 17, 256):                                    # the board's own table
        for c in mod.moe7_cfg(m_):
            assert mod.moe7_smem(m_, c) + 480 + 1024 <= 227 * 1024 and 2 * (mod.moe7_smem(m_, c) + 480 + 1024) > 228 * 1024, (m_, c)
