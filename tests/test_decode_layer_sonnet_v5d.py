"""kernels/muse_decode_layer/sonnet_v5d.py (H24): sonnet_v5c with the large-M experts (T = 33..256) on the MoE problem's sonnet_v5 unified kernel w4_moe5_kernel.

No GPU:  file rules, v5d is v5c plus additions only (no line removed), the MOE5 CUDA block is the MoE problem's verbatim, the route kernel is v5c's (device local_start, wrap rule),
         MOE5_CFG = the MoE problem's rows with v4's K splits, the dispatch table by T, prepare()'s fallbacks (build fails / mismatch / switch off -> v5c's path).
Dev GPU (RTX 2060, sm_75): NVRTC assembly for sm_80 / sm_90 (regs within the launch-bounds cap, no local memory), the whole layer against the organisers' reference (local_check small
         --graph, 13/13), and with LM_NVRTC=1 the moe5 experts BITWISE equal to v5c's large-M experts through NVRTC (BF16 mma emulated).
Run:     tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v5d.py     (LM_NVRTC=1 for the bitwise run)"""
import difflib
import importlib.util
import inspect
import os
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

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_decode_layer"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v5d.py"
V5C = ROOT / "kernels" / PROBLEM / "sonnet_v5c.py"
MOE5 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v5.py"
CUOBJDUMP = Path("/usr/local/cuda/bin/cuobjdump")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v5d", KERNEL)


@pytest.fixture(scope="module")
def old():
    return _load("_sonnet_v5c_for_v5d", V5C)


@pytest.fixture(scope="module")
def moe():
    return _load("_moe5_for_v5d", MOE5)


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
    compile(src, "sonnet_v5d.py", "exec")


def test_header_has_the_predictions():
    header = "\n".join(line for line in KERNEL.read_text().splitlines() if line.startswith("#"))
    for needle in ("sonnet_v5d (H24)", "w4_moe5_kernel", "Predicted per shape", "#8 393.5 -> ~383.5", "#11 883.2 -> ~858", "PROBLEM 1.423 -> ~1.437", "range 1.425-1.448", "Kill below",
                   "BITWISE", "T = 64 / 128 / 256", "Risks", "device local_start"):
        assert needle in header, needle


def test_v5d_is_v5c_plus_additions_only():
    a, b = V5C.read_text().splitlines(), KERNEL.read_text().splitlines()
    ops = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    assert not [d for d in ops if d[0] == "-"], "v5d removes a v5c line"
    # outside the header comment and the MOE5 block only the dispatch + prepare lines are new
    begin = next(i for i, line in enumerate(b) if "MOE5 BEGIN (H24)" in line)
    end = next(i for i, line in enumerate(b) if "MOE5 END" in line)
    first_code = next(i for i, line in enumerate(b) if line.startswith("import functools"))
    outside = [line for i, line in enumerate(b) if (i >= first_code and not begin <= i <= end) and line not in set(a)]
    assert outside == ["    if local_ids is None and _moe5_live(x.shape[0], state):   # H24: the moe problem's unified kernel, bitwise the large-M path's output",
                       "        return _experts_moe5(x, topk_weights, topk_ids, local_start, state)",
                       "        _prepare_moe5_for_layer(state)"], outside


def test_unchanged_sources_and_kernels_are_v5cs(mod, old):
    for name in ("_SMALLM_DEVICE_SRC", "_SMALLM_HOST_SRC", "_LARGEM_EXTRA_SRC", "_LARGEM_HOST_SRC", "SMALLM_CFG", "LARGEM_CFG"):
        assert getattr(mod, name) == getattr(old, name), name
    for name in ("_gemv_splitk", "_router_fix_kernel", "_full_cat_kernel", "_attn_kernel", "_reduce_kernel", "_local_ids_kernel", "_pre_norm_kernel", "_unpool_kernel", "_epilogue_kernel"):
        assert inspect.getsource(getattr(mod, name).fn) == inspect.getsource(getattr(old, name).fn), name
    for name in ("largem_experts", "_experts_largem", "_largem_selfcheck", "_prepare_largem_for_layer", "prepare_largem", "_largem_live", "_smallm_selfcheck_inputs", "_experts_smallm"):
        assert inspect.getsource(getattr(mod, name)) == inspect.getsource(getattr(old, name)), name
    assert "local_start" in mod._LARGEM_EXTRA_SRC or "lo_ptr" in mod._LARGEM_EXTRA_SRC


def test_moe5_block_is_the_moe_problems_and_the_route_is_v5cs(mod, old, moe):
    assert mod._MOE5_EXTRA_SRC == moe._MOE5_EXTRA_SRC
    assert mod.moe5_device_src() == old._SMALLM_DEVICE_SRC + old._LARGEM_EXTRA_SRC + moe._MOE5_EXTRA_SRC
    assert "lo_ptr" in mod.moe5_device_src() and "& 255" in mod.moe5_device_src()     # v5c's patched route: device local_start, wrap rule
    # the host source is the MoE problem's for the same instantiations, except the route launch (device tensor instead of a host int)
    inst = mod.MOE5_INSTANCES
    a = moe.moe5_host_src(inst).splitlines()
    b = mod.moe5_host_src(inst).splitlines()
    diff = [d for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] in "+-" and d[:3] not in ("+++", "---")]
    joined = "\n".join(diff)
    assert len(diff) <= 20, joined
    assert "lo.data_ptr<int>()" in joined and "moe5_route_small_kernel" in joined and "TORCH_CHECK(!small" in joined
    assert mod._MOE5_CPP_SRC.count("torch::Tensor lo, int64_t nloc, bool small") == 1
    assert mod.MOE5_INSTANCES == ((True, 4, 2, 3, 0), (False, 4, 2, 3, 0)) and set(mod.MOE5_INSTANCES) <= set(moe.MOE5_INSTANCES)
    assert mod.moe5_instance_names(inst) == moe.moe5_instance_names(inst)


def test_host_source_has_the_carveout_and_two_cases(mod):
    src = mod.moe5_host_src(mod.MOE5_INSTANCES)
    assert src.count("M5_CASE(") == 2 + 1 and "@CASES@" not in src and "@INITS@" not in src       # 2 cases + the #define
    assert src.count("cudaSharedmemCarveoutMaxShared") == 2
    assert src.count("largem_route_kernel<<<1, 1024") == 1


def test_cfg_is_the_moe_rows_with_v5cs_k_splits(mod, old, moe):
    rows = {mx: (c1, c2) for mx, c1, c2 in moe.MOE5_CFG}
    for b in (128, 256):
        assert mod.MOE5_CFG[b] == rows[b], b
        c1, c2 = mod.MOE5_CFG[b]
        lc_cfg = old.LARGEM_CFG[b]
        # the same K splits / warp geometry / z caps as v5c's large-M kernel => the same split-K sums (bitwise)
        assert (c1[0], c1[1], c1[2], c1[3]) == (lc_cfg["nw1"], lc_cfg["mr1"], lc_cfg["ks1"], lc_cfg["nz1"])
        assert (c2[0], c2[1], c2[2], c2[3]) == (lc_cfg["nw2"], lc_cfg["mr2"], lc_cfg["ks2"], lc_cfg["nz2"])
        assert (c1[4], c1[5], c2[4], c2[5]) == (4, 3, 2, 3)        # NB 4 / 2, LB 3
        assert (c1[0], c1[1], c1[5]) + (c2[0], c2[1], c2[5]) in {(4, 2, 3, 4, 2, 3)}
    assert mod.MOE5_SELFCHECK_MS == (64, 128, 256) and mod.MOE5_SELFCHECK_STARTS == (0, 100)
    assert mod.moe5_cfg(33) == mod.MOE5_CFG[128] and mod.moe5_cfg(128) == mod.MOE5_CFG[128]
    assert mod.moe5_cfg(129) == mod.MOE5_CFG[256] and mod.moe5_cfg(256) == mod.MOE5_CFG[256]
    assert mod.moe5_cfg(32) is None and mod.moe5_cfg(257) is None and mod.moe5_cfg(1) is None


def test_zbound_and_rows(mod, moe):
    lm = {"zmax": moe.LM_ZMAX}
    for m in (33, 64, 100, 128, 129, 200, 256):
        assert mod.moe5_zbound(m) == moe.moe5_zbound(m, lm)
        assert mod.moe5_rows(m) == moe.moe5_rows(m) == 32
    assert mod.LM_ZMAX == moe.LM_ZMAX


# ------------------------------------------------------------------------------------------------ the path table (T only)
def _moe_path(mod, t, state):
    if mod._smallm_live(t, state):
        return "smallm"
    if mod._moe5_live(t, state):
        return "moe5_128" if mod.largem_bucket(t) == 128 else "moe5_256"
    if mod._largem_live(t, state):
        return "largem128" if mod.largem_bucket(t) == 128 else "largem256"
    return "marlin"


GOOD = {"smallm": object(), "largem": object(), "largem_ok": {128: True, 256: True}, "moe5": object(), "moe5_ok": {128: True, 256: True}}


@pytest.mark.parametrize("t,want", [(1, "smallm"), (8, "smallm"), (16, "smallm"), (17, "marlin"), (31, "marlin"), (32, "marlin"), (33, "moe5_128"), (100, "moe5_128"),
                                    (128, "moe5_128"), (129, "moe5_256"), (200, "moe5_256"), (256, "moe5_256"), (257, "marlin"), (1000, "marlin")])
def test_moe_path_table_by_tokens_only(mod, t, want):
    assert _moe_path(mod, t, GOOD) == want


def test_moe_path_is_monotone_in_t(mod):
    order = ["smallm", "marlin", "moe5_128", "moe5_256", "marlin"]
    seq = [_moe_path(mod, t, GOOD) for t in range(1, 400)]
    idx = last = 0
    for p in seq:
        while order[idx] != p:
            idx += 1
            assert idx < len(order), (p, seq)
        assert idx >= last
        last = idx


@pytest.mark.parametrize("state,t,want", [
    ({**GOOD, "moe5": None}, 128, "largem128"),                                                   # no moe5 build -> v5c's large-M kernel
    ({**GOOD, "moe5_ok": {128: False, 256: True}}, 100, "largem128"),                              # a bucket that differs in a bit
    ({**GOOD, "moe5_ok": {128: False, 256: True}}, 200, "moe5_256"),
    ({**GOOD, "moe5_ok": {}}, 200, "largem256"),
    ({**GOOD, "moe5": None, "largem_ok": {128: False, 256: True}}, 100, "marlin"),                 # v5c's own fallback chain is untouched
    ({**GOOD, "largem": None}, 128, "marlin"),
    ({**GOOD, "smallm": None}, 128, "marlin"),
    ({}, 128, "marlin")])
def test_fallback_chain(mod, state, t, want):
    assert _moe_path(mod, t, state) == want


def test_switch_off_is_v5cs_path(mod, monkeypatch):
    monkeypatch.setattr(mod, "S12_MOE5", False)
    assert _moe_path(mod, 128, GOOD) == "largem128" and _moe_path(mod, 256, GOOD) == "largem256"
    assert mod.S8_LARGEM and mod.S4_SMALLM


def test_experts_dispatch_by_t(mod, monkeypatch):
    calls = []
    monkeypatch.setattr(mod, "_experts_moe5", lambda *a, **k: calls.append("moe5") or "5")
    monkeypatch.setattr(mod, "_experts_largem", lambda *a, **k: calls.append("large") or "L")
    monkeypatch.setattr(mod, "_experts_smallm", lambda *a, **k: calls.append("small") or "S")
    monkeypatch.setattr(mod, "local_ids_triton", lambda *a, **k: calls.append("marlin") or (_ for _ in ()).throw(RuntimeError("stop")))
    for t in (1, 16, 17, 32, 33, 128, 129, 256, 257):
        x = torch.zeros(t, 16)
        try:
            out = mod._experts(x, torch.ones(t, 8), torch.zeros(t, 8, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), {**GOOD, "bank": {"size_n": 8}})
        except RuntimeError:
            out = None
        want = "S" if t <= 16 else ("5" if 33 <= t <= 256 else None)
        assert out == want, (t, out)
        assert calls[-1] == {"S": "small", "5": "moe5", None: "marlin"}[want], (t, calls)
    monkeypatch.setattr(mod, "S12_MOE5", False)
    assert mod._experts(torch.zeros(64, 16), torch.ones(64, 8), torch.zeros(64, 8, dtype=torch.int32), torch.zeros(1, dtype=torch.int32), {**GOOD, "bank": {"size_n": 8}}) == "L"


# ------------------------------------------------------------------------------------------------ prepare()'s fallbacks
def _prepared(mod, monkeypatch, build, oks, **state0):
    monkeypatch.setattr(mod, "prepare_moe5", lambda **k: build())
    monkeypatch.setattr(mod, "_moe5_selfcheck", lambda st: oks)
    state = {"smallm": object(), "largem": object(), "largem_ok": {128: True, 256: True}, "post_expert_w": torch.zeros(1), **state0}
    mod._prepare_moe5_for_layer(state)
    return state


def test_prepare_keeps_moe5_when_bitwise(mod, monkeypatch):
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: True, 256: True})
    assert st["moe5"] == {"x": 1} and st["moe5_ok"] == {128: True, 256: True}


def test_prepare_keeps_only_the_bitwise_bucket(mod, monkeypatch):
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: False, 256: True})
    assert st["moe5"] == {"x": 1} and st["moe5_ok"] == {128: False, 256: True}
    assert _moe_path(mod, 100, st) == "largem128" and _moe_path(mod, 200, st) == "moe5_256"


def test_prepare_drops_moe5_when_nothing_matches(mod, monkeypatch):
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: False, 256: False})
    assert st["moe5"] is None and not any(st["moe5_ok"].values())
    assert _moe_path(mod, 100, st) == "largem128"


def test_prepare_needs_the_v5c_bucket_to_be_good_too(mod, monkeypatch):
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: True, 256: True}, largem_ok={128: False, 256: True})
    assert st["moe5_ok"] == {128: False, 256: True}


def test_prepare_falls_back_when_the_build_fails(mod, monkeypatch):
    def boom():
        raise RuntimeError("no nvcc")

    st = _prepared(mod, monkeypatch, boom, {})
    assert st["moe5"] is None and st["moe5_ok"] == {}


def test_prepare_without_v5c_largem_or_with_switch_off(mod, monkeypatch):
    for kw in ({"largem": None}, {"smallm": None}, {"largem_ok": {128: False, 256: False}}):
        st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: True, 256: True}, **kw)
        assert st["moe5"] is None, kw
    monkeypatch.setattr(mod, "S12_MOE5", False)
    st = _prepared(mod, monkeypatch, lambda: {"x": 1}, {128: True, 256: True})
    assert st["moe5"] is None


def test_selfcheck_is_bitwise_and_swallows_exceptions(mod, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("generators need the device")
    state = {"post_expert_w": torch.zeros(4, device="cuda")}
    seen = []

    def fake_inputs(m, seed, dev):
        seen.append(m)
        return torch.zeros(m, 4, device=dev), torch.ones(m, 8, device=dev), torch.zeros(m, 8, dtype=torch.int32, device=dev)

    same = torch.ones(2, 2, dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(mod, "_smallm_selfcheck_inputs", fake_inputs)
    monkeypatch.setattr(mod, "_experts_largem", lambda *a: same)
    monkeypatch.setattr(mod, "_experts_moe5", lambda *a: same.clone())
    assert mod._moe5_selfcheck(state) == {128: True, 256: True} and seen == [64, 128, 256]
    one_bit = same.clone()
    one_bit.view(torch.int16)[0, 0] ^= 1
    monkeypatch.setattr(mod, "_experts_moe5", lambda *a: one_bit)
    assert mod._moe5_selfcheck(state) == {128: False, 256: False}
    monkeypatch.setattr(mod, "_experts_moe5", lambda *a: (_ for _ in ()).throw(RuntimeError("launch")))
    assert mod._moe5_selfcheck(state) == {128: False, 256: False}


def test_no_scratch_allocation_in_the_timed_moe5_path(mod):
    src = inspect.getsource(mod.moe5_experts) + inspect.getsource(mod._experts_moe5)
    assert ".zero_(" not in src and "torch.zeros" not in src and "synchronize" not in src and ".item(" not in src and ".cpu(" not in src


def test_moe5_experts_only_reads_local_start_on_the_device(mod):
    src = inspect.getsource(mod.moe5_experts)
    assert "ext.moe5_route(ids, local_ids, lm[\"ctab\"], lm[\"slotl\"], start, SM_NEXP, False)" in src
    assert "int(start" not in src and "start.item" not in src


# ------------------------------------------------------------------------------------------------ offline compiles for the judge's GPU
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
    if not CUOBJDUMP.exists():
        return {n: None for n in names}
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.cubin"
        p.write_bytes(cubin)
        r = subprocess.run([str(CUOBJDUMP), "--dump-resource-usage", str(p)], capture_output=True, text=True)
    use = {m.group(1): (int(m.group(2)), int(m.group(3)), int(m.group(5))) for m in
           re.finditer(r"Function (\S+?):\s*\n\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", r.stdout)}
    return {n: use.get(low[n].rstrip("\x00")) for n in names}


@needs_cuda
@pytest.mark.parametrize("sm", ["sm_80", "sm_90"])
def test_moe5_device_source_assembles_for_judge_arch(mod, nv, sm):
    """NVRTC (with its own ptxas) for the judge's arch: every inline-asm string assembles; regs within the launch-bounds cap and no local memory where cuobjdump is available."""
    names = mod.moe5_instance_names(mod.MOE5_INSTANCES) + ["moe5_route_small_kernel", "largem_route_kernel"]
    cubin, low = nv.sass_check(mod.moe5_device_src(), sm, names=names)
    assert len(cubin) > 1000 and len(low) == len(names)
    use = _usage(nv, mod.moe5_device_src(), sm, names)
    for (f, w, r, lb, dbg), n in zip(mod.MOE5_INSTANCES, names):
        if use[n] is None:
            continue
        regs, stack, local = use[n]
        assert local == 0 and regs <= 65536 // (w * 32 * lb) and stack <= 64, (n, use[n])


# ------------------------------------------------------------------------------------------------ dev GPU
@needs_cuda
def test_whole_layer_against_the_organisers_reference():
    passed, total = lc.run(PROBLEM, KERNEL, device="cuda", cases="small", graph=True, out=lambda *_: None)
    assert (passed, total) == (13, 13)


@needs_cuda
def test_whole_layer_with_the_switch_off(tmp_path):
    src, n = re.subn(r"^S12_MOE5 = True", "S12_MOE5 = False", KERNEL.read_text(), flags=re.M)
    assert n == 1
    path = tmp_path / "sonnet_v5d_off.py"
    path.write_text(src)
    passed, total = lc.run(PROBLEM, path, device="cuda", cases="small", graph=False, out=lambda *_: None)
    assert (passed, total) == (13, 13)


# ------------------------------------------------------------------------------------------------ dev GPU: moe5 vs v5c's large-M through NVRTC (opt in)
@pytest.fixture(scope="module")
def lm_env():
    if os.environ.get("LM_NVRTC") != "1":
        pytest.skip("set LM_NVRTC=1: NVRTC builds the large-M and moe5 kernels for sm_75 with the BF16 mma emulated")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("needs NVRTC")
    ref = lc.Loaded(PROBLEM, KERNEL).ref
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")
    mod = _load("_sonnet_v5d_lm", KERNEL)
    old = mod.SMALLM_CFG
    mod.SMALLM_CFG = ((1, 4, 4, 3, 4, 4, 3), (8, 4, 4, 6, 4, 4, 6), (16, 4, 4, 12, 4, 4, 12))
    sm = mod.prepare_smallm(weights, ext=nv.NvrtcExt(mod._SMALLM_DEVICE_SRC))
    mod.SMALLM_CFG = old
    lm = mod.prepare_largem(sm, ext=nv.NvrtcLargeExt(mod._SMALLM_DEVICE_SRC + mod._LARGEM_EXTRA_SRC))
    m5 = mod.prepare_moe5(ext=nv.NvrtcMoe5Ext(mod.moe5_device_src(), mod.moe5_instance_names(mod.MOE5_INSTANCES)), device="cuda")
    return mod, sm, lm, m5


def _inputs(m, seed, start):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, 4096), generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.randint(0, 256, (m, 8), generator=g, device="cuda")
    ids[:, :4] = (start + torch.randint(0, 32, (m, 4), generator=g, device="cuda")) % 256      # a busy 32-expert range (it wraps for start 240)
    return x, ids.to(torch.int32)


@needs_cuda
@pytest.mark.parametrize("m,seed,start", [(40, 3, 0), (64, 4, 100), (128, 5, 0), (130, 6, 100), (256, 7, 240)])
def test_moe5_experts_are_bitwise_v5cs_largem_experts(lm_env, m, seed, start):
    mod, sm, lm, m5 = lm_env
    x, ids = _inputs(m, seed, start)
    st = torch.tensor([start], dtype=torch.int32, device="cuda")
    # 2060: K slices of 512 (ks 8) keep the smem under 64 KB; the same ks in both kernels => the same split-K sums
    lcfg = dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24)
    mcfg = ((4, 2, 8, 24, 4, 3), (4, 2, 8, 24, 2, 3))
    y0, l0 = mod.largem_experts(x, ids, sm, lm, st, lcfg)
    y1, l1 = mod.moe5_experts(x, ids, sm, lm, m5, st, mcfg)
    torch.cuda.synchronize()
    assert torch.equal(l0, l1)
    keep = (l1 >= 0).unsqueeze(-1)      # remote slots are never written (torch.empty); the epilogue skips them (slot id -1)
    assert torch.equal(y0.view(torch.int16).masked_fill(~keep, 0), y1.view(torch.int16).masked_fill(~keep, 0))
    assert (l1 >= 0).any() and (l1 < 0).any()        # both local and remote slots are exercised


@needs_cuda
def test_moe5_experts_in_a_cuda_graph_and_changing_start(lm_env):
    mod, sm, lm, m5 = lm_env
    x, ids = _inputs(64, 11, 0)
    st = torch.zeros(1, dtype=torch.int32, device="cuda")
    mcfg = ((4, 2, 8, 24, 4, 3), (4, 2, 8, 24, 2, 3))
    lcfg = dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24)
    mod.moe5_experts(x, ids, sm, lm, m5, st, mcfg)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y, local = mod.moe5_experts(x, ids, sm, lm, m5, st, mcfg)
    for s0 in (0, 100, 250):
        st.fill_(s0)
        g.replay()
        y0, l0 = mod.largem_experts(x, ids, sm, lm, st, lcfg)
        torch.cuda.synchronize()
        keep = (local >= 0).unsqueeze(-1)
        assert torch.equal(local, l0) and torch.equal(y.view(torch.int16).masked_fill(~keep, 0), y0.view(torch.int16).masked_fill(~keep, 0)), s0
