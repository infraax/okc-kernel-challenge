"""kernels/muse_moe_w4a16/measure_ws_v1.py (MEASUREMENT-ONLY, card C30): file rules, verbatim regions (v5's kernel_fn and Python region, sonnet_v7's MOE7 block), the
variant table against the 8 instantiations, the input generator, the stamp line / decision rule, the budgeted printer with a stubbed environment (<= 14 KB, S lines first),
and the real environment end to end on the RTX 2060 (NVRTC build of the measurement extension, the sm_75 fallback ring, graph timing, stamps, wstream / cal); the sm_90
assembly of the whole measurement source is LAST.

Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_measure_ws_v1.py"""
import importlib.util
import re
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "problems" / "muse_moe_w4a16"))

MEAS = ROOT / "kernels" / "muse_moe_w4a16" / "measure_ws_v1.py"
V7 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v7.py"
V5 = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v5.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# the measurement file's instantiation list (FUSE, D, DBG): 8 = {T, F} x {D1, D2, D1 + twins, D2 + twins}
MEAS_INSTANCES = tuple((f, d, dbg) for f in (True, False) for d, dbg in ((1, 0), (2, 0), (1, 1), (2, 1)))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mm():
    return _load("_measure_ws_v1", MEAS)


# --------------------------------------------------------------------------------------------- CPU
def test_measure_file_rules():
    src = MEAS.read_text()
    compile(src, "measure_ws_v1.py", "exec")
    lines = src.splitlines()
    assert lines[0] == "# okc: problem=muse_moe_w4a16"
    assert "NEVER submit" in lines[1] and "MEASUREMENT-ONLY" in lines[1]
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for bad in (r"subprocess", r"urllib", r"requests", r"socket", r"import os", r"environ", r"tolist\(", r"\.item\(", r"\.cpu\(",
                r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src


def test_measure_regions_are_verbatim(mm):
    """kernel_fn and v5's whole Python region are v5's byte for byte; the MOE7 block is sonnet_v7's; the instantiation list is the one the tests assemble."""
    mod = _load("_sonnet_v7_fresh_for_meas", V7)
    a, b, m = V5.read_text(), V7.read_text(), MEAS.read_text()
    assert m[m.rindex("def kernel_fn"):] == a[a.rindex("def kernel_fn"):]
    lo, hi = "import re\n", "# ============================================================================================ entry points"
    assert a[a.index(lo):a.index(hi)] in m
    s0, s1 = ("# ============================================================================================ MOE7 BEGIN",
              "# ============================================================================================ MOE7 END")
    assert b[b.index(s0):b.index(s1)] == m[m.index(s0):m.index(s1)]
    assert set(mm.MEAS_INSTANCES) == set(MEAS_INSTANCES) and len(set(mm.MEAS_INSTANCES)) == 8
    assert mm.moe7_device_src() == mod.moe7_device_src() and mm.MOE7_CFG == mod.MOE7_CFG
    assert set(mod.MOE7_INSTANCES) <= set(mm.MEAS_INSTANCES)
    assert mm.meas_names(mm.MEAS_INSTANCES)[-2:] == ["meas_wstream_kernel<4,2>", "meas_cal_kernel"]
    assert "-Xptxas" in m and "verbose=True" in m and "build=" in m


def test_sync_tool_is_idempotent_on_the_committed_file():
    """tools/sync_measure_ws_v1.py rebuilds the file from sonnet_v5 / sonnet_v7 and its own header + measurement section: nothing to do on a synced tree."""
    sync = _load("_sync_measure_ws", ROOT / "tools" / "sync_measure_ws_v1.py")
    assert sync.build(MEAS.read_text()) == MEAS.read_text()


def test_measure_gen_input_equals_reference(mm):
    import reference

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for bs, dist, seed in ((32, "normal", 11), (128, "normal", 12), (256, "skewed", 13), (1, "normal", 11), (16, "skewed", 14)):
        a = mm._gen_input(bs, seed, dist, dev)
        b = reference.generate_input(bs, "ep8", seed, dist=dist, device=dev)
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            assert torch.equal(a[k], b[k]), (bs, dist, k)
        assert a["config"] == b["config"]


def test_variants_use_instantiated_kernels_and_valid_geometry(mm):
    """Every variant of every M maps to one of the 8 instantiations and passes the kernel's geometry rules; the ks sweep exists at bs 256 / 8 / 1 only; the shapes are
    the design's (D1S8, D2S5)."""
    inst = set(mm.meas_instances())
    assert len(inst) == 8 and mm.WS_D1 == (1, 8) and mm.WS_D2 == (2, 5) and mm.MS == (256, 128, 32, 16, 8, 1) and mm.SEEDS == (11, 12, 13)
    for m in mm.MS:
        vs = mm.variants(m)
        names = [v[0] for v in vs]
        assert len(names) == len(set(names)) and names[:4] == ["d1s8", "d2s5", "nocomp", "consonly"] and "pdloff" in names
        assert set(mm.PASS1) - set(mm.CO_NAMES) <= set(names) and (set(mm.CO_NAMES) <= set(names)) == (m in mm.CO_MS)
        assert (any(n.startswith("ks") for n in names)) == (m in (256, 8))
        c5 = mm.moe5_cfg(m)
        for name, c1, c2, fl, dbgt, pdl in vs:
            for fuse, c, groups, k, g in ((True, c1, 512, 4096, 0), (False, c2, 256, 4096, 1)):
                mm.moe7_check_cfg(m, k, groups, c)
                assert (fuse, c[0], dbgt[g]) in inst, (m, name, c)
                if name in ("d1s8", "nocomp", "consonly", "pdloff", "stamps"):
                    assert c[:2] == (1, 8)
                if name in ("d2s5", "nocomp2", "consonly2"):
                    assert c[:2] == (2, 5)
                if not name.startswith("ks"):
                    assert c[2] == c5[g][2], (m, name)           # v5's K split: bitwise v5 expected
                else:
                    assert c[2] == int(name[2:]) and c[2] != c5[g][2]
                assert (dbgt[g] == 1) == (fl[g] != 0)
    assert {n for n in [v[0] for v in mm.variants(256)] if n.startswith("ks")} == {"ks2", "ks8"}
    assert {v[0] for v in mm.variants(1)} >= set(mm.CO_NAMES) and not any(n in [v[0] for v in mm.variants(32)] for n in mm.CO_NAMES)
    assert {n for n in [v[0] for v in mm.variants(8)] if n.startswith("ks")} == {"ks4", "ks8"}
    assert mm.expert_bytes({"g1": 512, "k1": 4096, "g2": 256, "k2": 4096}) == (8192 * 4096 * 0.5625, 4096 * 4096 * 0.5625)


def _rec(n=4, cons_wait=100, prod_wait=50, fix=20, items=10, life=1000, bytes_=10000, skew=0):
    """a synthetic stamp table [n, 13, 8]: warp 0 = producer"""
    t = torch.zeros(n, 13, 8, dtype=torch.int64)
    for c in range(n):
        for w in range(13):
            t[c, w, 0] = 1000 + c
            t[c, w, 1] = 1000 + c + life + (skew if c == n - 1 else 0)
        t[c, 0, 2], t[c, 0, 4], t[c, 0, 6] = prod_wait, items, bytes_
        t[c, 1:, 2], t[c, 1:, 3] = cons_wait, fix
    return t


def test_stamp_line_and_shares(mm):
    s = mm._stamp_line(_rec(), 0)
    assert "items/cta=10" in s and "cons_wait_full=10%" in s and "prod_wait_empty=5%" in s and "fix_epi=2%" in s and "bw/sm=10.0GB/s" in s and "tail=" in s and "ERR" not in s
    assert mm._stamp_shares(_rec()) == pytest.approx((10.0, 5.0))
    assert mm._stamp_line(_rec(), 1).endswith("ERR") and mm._stamp_line(torch.zeros(2, 13, 8, dtype=torch.int64), 0) == "no stamps"
    assert "tail=0.5us" in mm._stamp_line(_rec(skew=500), 0)


def _co_res(mm, m, base, saves):
    """res dict with consonly2 = base us (g1 and g2 x 0.5) and each co_ twin saving the given % (of g1; g2 the same)"""
    res = {(m, "consonly2"): {"g1": [base] * 3, "g2": [base / 2] * 3, "pair": [], "bw": ["OK"] * 3}}
    for n, sv in saves.items():
        res[(m, n)] = {"g1": [base * (1 - sv / 100)] * 3, "g2": [base / 2 * (1 - sv / 100)] * 3, "pair": [], "bw": ["-"] * 3}
    return res


def test_co_shares_and_verdict_rules(mm):
    """C31: the twins' saved shares of consonly2 and the verdict mapping (NOMMA >= 35 -> TC; NOUNPACK >= 35 -> ALU; smem floor >= 60 % of the time or NOLDM >= 20 ->
    smem; NOFIX >= 20 -> fixup; every rule that fires is listed)."""
    full = {"co_nounpack": 10.0, "co_nomma": 50.0, "co_noldm": 5.0, "co_nofix": 8.0, "co_ldsonly": 70.0, "co_smem": 60.0}
    sh, b1, b2 = mm._co_shares(_co_res(mm, 256, 270.0, full), 256)
    assert b1 == 270.0 and b2 == 135.0 and sh["co_nomma"][0] == pytest.approx(50.0) and sh["co_ldsonly"][1] == pytest.approx(70.0)
    v = mm._co_verdict(sh)
    assert "TC-bound (NOMMA saves 50%)" in v and "wgmma" in v and "ALU-bound" not in v and "smem-bound" not in v and "fixup" not in v
    assert "ALU-bound (NOUNPACK saves 40%)" in mm._co_verdict(dict(sh, co_nounpack=(40.0, 40.0)))
    assert "smem-bound" in mm._co_verdict(dict(sh, co_ldsonly=(30.0, 30.0)))               # the lds-only floor is 70 % of the time
    assert "smem-bound" not in mm._co_verdict(dict(sh, co_ldsonly=(70.0, 70.0), co_smem=(70.0, 70.0)))
    assert "smem-bound" in mm._co_verdict(dict(sh, co_noldm=(22.0, 22.0)))
    assert "fixup (NOFIX saves 23%)" in mm._co_verdict(dict(sh, co_nofix=(23.0, 23.0)))
    both = mm._co_verdict(dict(sh, co_nofix=(23.0, 0.0), co_nounpack=(36.0, 0.0)))
    assert "TC-bound" in both and "ALU-bound" in both and "fixup" in both and both.count(";") == 2
    assert "distributed" in mm._co_verdict({k: (10.0, 10.0) for k in mm.CO_NAMES} | {"co_ldsonly": (60.0, 60.0), "co_smem": (60.0, 60.0)})
    assert mm._co_shares({}, 256) is None and mm._co_shares({(256, "consonly2"): {"g1": [], "g2": [], "bw": []}}, 256) is None


def test_decision_rule_rows(mm):
    d = mm._decide
    assert "(i) promote" in d(160.0, "d2s5", 3.8, 4.2, 12.0, 8.0, True)
    assert "(v)" in d(160.0, "d2s5", 3.8, 4.2, 12.0, 8.0, False)
    assert "(iv)" in d(190.0, "d1s8", 3.0, 4.2, 40.0, 5.0, True) and "(iv)" in d(120.0, "d1s8", 3.1, 4.0, 10.0, 5.0, True)
    assert "(ii)" in d(190.0, "d1s8", 3.6, 4.2, 30.0, 5.0, True)
    assert "(iii)" in d(190.0, "d1s8", 3.6, 4.2, 10.0, 30.0, True)
    assert "(vi) KILL" in d(210.0, "d1s8", 3.6, 3.7, 10.0, 10.0, True)
    assert "(none" in d(190.0, "d1s8", 3.6, 4.2, 10.0, 10.0, True) and "no WS result" in d(None, None, None, None, None, None, True)
    assert "nocomp=None" in d(210.0, "d1s8", None, None, None, None, True)


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
        return 4.05, 16.6, 1.9

    def stamps(self, ctx, c1, c2):
        return [(_rec(n=8), 0), (_rec(n=8), 0)]


def _fake_patch(mm, monkeypatch):
    def fake_stages(data, state, *a, **k):
        names = ("route", "gather", "g1", "g2", "g1z", "g2z", "pair")
        return [(n, lambda: None) for n in names], {"y": torch.zeros(4, 4)}

    monkeypatch.setattr(mm, "_v5_stages", fake_stages)
    monkeypatch.setattr(mm, "_v7_stages", fake_stages)
    monkeypatch.setattr(mm, "_ws_stages", lambda data, state, c1, c2: [("g1", lambda: None), ("g2", lambda: None)])
    return {"smallm": {"dims": {"k1": 4096, "k2": 4096, "g1": 512, "g2": 256}}, "largem": {"zmax": mm.LM_ZMAX}, "moe7": {"dbg": None, "nctas": 132}}


def test_measure_control_flow_with_fake_env(mm, monkeypatch, capsys):
    """The budgeted printer: S lines first, every M gets its I line and the v5 / variants / wstream B lines, the D line, the summary stays under 14 KB."""
    state = _fake_patch(mm, monkeypatch)
    mm._measure(state, _FakeEnv(state), t_start=0.0, build_s=57.0)
    out = capsys.readouterr().out
    assert len(out.encode()) < mm.SUMMARY_BUDGET
    assert out.startswith("MEAS-WS") and "bw_ach=4.05TB/s" in out and "zero=1.9us" in out and "build=57s" in out and "d1=1x8 d2=2x5" in out and "cps=2" in out and "MEAS-WS done" in out
    assert "skipped" not in out
    ls = out.splitlines()
    first_i = next(i for i, l in enumerate(ls) if l.startswith("I moe"))
    assert all(l.startswith("S ") for l in ls[1:first_i]) and sum(l.startswith("S verdict m=") for l in ls) == len(mm.CO_MS) and sum(" co-saves " in l for l in ls) == len(mm.CO_MS) and first_i == 1 + 2 * len(mm.MS) + 2 * len(mm.CO_MS)       # S lines first (stamps + stamps2 per M, co-saves + verdict per twin bs)
    for m in mm.MS:
        assert f"I moe m={m} seeds=11/12/13" in out
        names = ["v5", "d1s8", "d2s5", "nocomp", "consonly", "wstream", "nocomp2", "consonly2", "pdloff"] + (["ks2", "ks8"] if m == 256 else ["ks4", "ks8"] if m == 8 else []) + (list(mm.CO_NAMES) if m in mm.CO_MS else [])
        for v in names:
            if v.startswith("co_"):
                assert re.search(rf"^B moe m={m} v={v} \| g1=[\d.]+ g2=[\d.]+ \| timing-only$", out, re.M), (m, v)
            else:
                assert re.search(rf"^B moe m={m} v={v} \| g1=[\d.]+/[\d.]+/[\d.]+ g2=[\d.]+/[\d.]+/[\d.]+ \| ts=[\d.]+/[\d.]+ x=[\d.-]+/[\d.-]+ bw=\S+", out, re.M), (m, v)
        assert f"S moe m={m} d1s8 g1 items/cta=10 cons_wait_full=10%" in out and f"S moe m={m} d2s5 g1" in out
    assert re.search(r"^D ", out, re.M)
    assert out.count(" gat=") == len(mm.MS)       # the gather pre-pass alone, once per M
    assert all(len(l) < 300 for l in ls) and len(ls) <= 6 * 14 + 14 + 6 * len(mm.CO_MS) + 8


def test_measure_budget_skips_variants_not_seeds(mm, monkeypatch, capsys):
    state = _fake_patch(mm, monkeypatch)
    env = _FakeEnv(None)
    env.clock = lambda: (setattr(env, "t", env.t + 2.0) or env.t)
    mm._measure(state, env, t_start=0.0)
    out = capsys.readouterr().out
    assert "MEAS-WS done" in out and "skipped:" in out and "B moe m=256 v=v5" in out and len(out.encode()) < mm.SUMMARY_BUDGET


def test_measure_skips_without_extension(mm, capsys):
    mm._measure({"smallm": None, "largem": None, "moe5": None, "moe7": None, "norm_w": torch.zeros(1)})
    assert "MEAS-WS skipped" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------- GPU: the real environment on the dev card
@pytest.fixture(scope="module")
def nv():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    nv = pytest.importorskip("nvrtc_smallm")
    if not nv.HAVE_NVRTC:
        pytest.skip("cuda-python / NVRTC not available")
    return nv


DEV_SMALLM_CFG = ((16, 8, 4, 8, 8, 4, 8),)
DEV_LARGEM_CFG = {b: dict(nw1=4, mr1=2, ks1=8, nz1=24, nw2=4, mr2=2, ks2=8, nz2=24) for b in (32, 128, 256)}
DEV_MOE5_CFG = tuple((mx, (8, 1, 4, 8, 1, 2), (8, 1, 4, 8, 1, 2)) if mx <= 16 else (mx, (4, 2, 8, 24, 4, 3), (4, 2, 8, 24, 2, 3))
                     for mx in (1, 2, 4, 8, 12, 16, 32, 128, 256))
DEV_MOE7_CFG = tuple((mx, (1, 3, 4), (1, 3, 4)) if mx <= 16 else (mx, (1, 3, 8), (1, 3, 8)) for mx in (1, 2, 4, 8, 12, 16, 32, 128, 256))


@pytest.fixture(scope="module")
def dev(mm, nv):
    from nvfp4 import random_weights

    g = torch.Generator(device="cuda").manual_seed(5252)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=g, device="cuda", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=g, device="cuda", leading=(32,))
    w = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(32).contiguous(),
         "post_norm_w": (0.1 * torch.randn((4096,), generator=g, device="cuda")).to(torch.bfloat16)}
    saved = (mm.SMALLM_CFG, dict(mm.LARGEM_CFG), mm.MOE5_CFG, mm.MOE7_CFG, mm.M7_SPIN)
    mm.M7_SPIN = 1 << 27       # the dev card's emulated mma / lane-loop copies make a legitimate wait 1000x longer than an H200's
    mm.SMALLM_CFG = DEV_SMALLM_CFG
    mm.LARGEM_CFG.clear()
    mm.LARGEM_CFG.update(DEV_LARGEM_CFG)
    mm.MOE5_CFG, mm.MOE7_CFG = DEV_MOE5_CFG, DEV_MOE7_CFG
    sm = mm.prepare_smallm(w, ext=nv.NvrtcExt(mm._SMALLM_DEVICE_SRC))
    lm = mm.prepare_largem(sm, ext=nv.NvrtcLargeExt(mm._SMALLM_DEVICE_SRC + mm._LARGEM_EXTRA_SRC))
    m5 = mm.prepare_moe5(sm, lm, ext=nv.NvrtcMoe5Ext(mm.moe5_device_src(), mm.moe5_instance_names([(True, 8, 1, 2, 0), (False, 8, 1, 2, 0), (True, 4, 2, 3, 0), (False, 4, 2, 3, 0)])))
    ext = nv.NvrtcMeasWsExt(mm.meas_device_src(), mm.meas_names(mm.MEAS_INSTANCES))
    m7 = mm.prepare_moe7(sm, lm, ext=ext, nctas=7)
    state = {"smallm": sm, "largem": lm, "largem_ok": {32: True, 128: True, 256: True}, "moe5": m5, "moe5_ok": {mx: True for mx, _a, _b in DEV_MOE5_CFG},
             "moe7": m7, "moe7_ok": {mx: True for mx, _a, _b in DEV_MOE7_CFG}, "norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}
    yield w, state
    mm.SMALLM_CFG, l, mm.MOE5_CFG, mm.MOE7_CFG, mm.M7_SPIN = saved
    mm.LARGEM_CFG.clear()
    mm.LARGEM_CFG.update(l)
    del sm, lm, m5, m7, state, w
    torch.cuda.empty_cache()


@needs_cuda
def test_measure_real_env_on_dev_card(mm, dev, monkeypatch, capsys):
    """The real _RealEnv on the 2060 with the dev-card rings (D1 S3; D2 slot = D1 S2 here: 64 KB holds no 36 KB x 2 ring at 32 rows): graph timing after flushes, the
    variants incl. the twins, the counter resets in the graphs, stamps, wstream / cal, the summary; the bitwise variants print bw=OK. M 8 uses the real D2 S2."""
    w, state = dev
    monkeypatch.setattr(mm, "HAVE_SGLANG", True)
    monkeypatch.setattr(mm, "WS_D1", (1, 3))
    monkeypatch.setattr(mm, "WS_D2", (1, 2))
    monkeypatch.setattr(mm, "MS", (33, 8))
    monkeypatch.setattr(mm, "SEEDS", (11,))
    monkeypatch.setattr(mm, "KS_MS", ())
    monkeypatch.setattr(mm, "CO_MS", (33, 8))
    monkeypatch.setattr(mm, "REPS", 3)
    monkeypatch.setattr(mm, "FLUSH_MB", 32)
    t0 = time.time()
    mm._measure(state, t_start=t0 - 10.0, build_s=12.0)
    out = capsys.readouterr().out
    with capsys.disabled():
        print("\n" + out)
    assert out.startswith("MEAS-WS") and "MEAS-WS done" in out and len(out.encode()) < mm.SUMMARY_BUDGET and "zero=" in out and "build=12s" in out
    assert "skipped" not in out, out[-400:]
    assert out.splitlines()[1].startswith("S moe m=33 d1s8 g1 items/cta=")
    assert re.search(r"^I moe m=33 .* gat=[\d.]+us$", out, re.M) and "cps=2" in out.splitlines()[0]
    for m in (33, 8):
        assert f"I moe m={m} seeds=11" in out
        for v in ("d1s8", "d2s5", "pdloff"):
            mt = re.search(rf"^B moe m={m} v={v} \| .* bw=(\S+)", out, re.M)
            assert mt and mt.group(1) == "OK", (m, v, mt.group(0) if mt else None)
        for v in ("nocomp", "consonly", "nocomp2", "consonly2"):
            assert re.search(rf"^B moe m={m} v={v} \| g1=", out, re.M), v
        assert re.search(rf"^B moe m={m} v=wstream \| g1=[\d.]+ g2=[\d.]+ \| ts=", out, re.M)
        assert re.search(rf"^S moe m={m} d1s8 g1 items/cta=\d+ cons_wait_full=\d+% prod_wait_empty=\d+% fix_epi=\d+% bw/sm=[\d.]+GB/s span=\d+us tail=[\d.]+us cps=2 \| g2 ", out, re.M)
    for m in (33, 8):                       # C31: the consumer-chain twins ran (timing only), their shares and the verdict are printed
        for v in mm.CO_NAMES:
            assert re.search(rf"^B moe m={m} v={v} \| g1=[\d.]+ g2=[\d.]+ \| timing-only$", out, re.M), (m, v)
        assert re.search(rf"^S moe m={m} co-saves \(of consonly2 g1=[\d.]+us g2=[\d.]+us, g1%/g2%\): nounpack=-?\d+%/-?\d+% nomma=.* smem=-?\d+%/-?\d+%$", out, re.M)
        assert re.search(rf"^S verdict m={m}: \S", out, re.M)
    assert int(state["largem"]["cnt"].abs().sum()) == 0


@needs_cuda
def test_measure_real_env_ks_sweep_and_d2_on_dev_card(mm, dev, monkeypatch, capsys):
    """bs 8 with the real D2 S2 ring and the ks sweep (ks 4 is v5's on the dev tables: ks2 / ks8 differ and print their relative L2, never OK-by-accident)."""
    w, state = dev
    monkeypatch.setattr(mm, "HAVE_SGLANG", True)
    monkeypatch.setattr(mm, "WS_D1", (1, 3))
    monkeypatch.setattr(mm, "WS_D2", (2, 2))
    monkeypatch.setattr(mm, "MS", (8,))
    monkeypatch.setattr(mm, "SEEDS", (11,))
    monkeypatch.setattr(mm, "KS_MS", (8,))
    monkeypatch.setattr(mm, "REPS", 3)
    monkeypatch.setattr(mm, "FLUSH_MB", 32)
    mm._measure(state, t_start=time.time() - 10.0)
    out = capsys.readouterr().out
    assert "skipped" not in out, out[-500:]
    for v, ok in (("d1s8", True), ("d2s5", True), ("ks2", False), ("ks8", False)):
        mt = re.search(rf"^B moe m=8 v={v} \| .* bw=(\S+)", out, re.M)
        assert mt, v
        assert (mt.group(1) == "OK") == ok, (v, mt.group(0))
        if not ok:
            assert mt.group(1).startswith("e=") and float(mt.group(1)[2:]) < mm.MOE7_TOL


@needs_cuda
def test_measure_preflight_skips_an_unlaunchable_variant_and_keeps_going(mm, dev, monkeypatch, capsys):
    """C30b: a variant whose ring does not fit the device (D2 S5 at 32 rows = 183 KB on a 64 KB card -- the H200 failure's class) is refused by the pre-flight BEFORE any
    launch: it is listed as skipped with the reason, nothing is poisoned, and the other variants and v5 / wstream are measured as usual."""
    w, state = dev
    monkeypatch.setattr(mm, "HAVE_SGLANG", True)
    monkeypatch.setattr(mm, "WS_D1", (1, 3))
    monkeypatch.setattr(mm, "WS_D2", (2, 5))
    monkeypatch.setattr(mm, "MS", (33,))
    monkeypatch.setattr(mm, "SEEDS", (11,))
    monkeypatch.setattr(mm, "KS_MS", ())
    monkeypatch.setattr(mm, "REPS", 3)
    monkeypatch.setattr(mm, "FLUSH_MB", 32)
    mm._measure(state, t_start=time.time() - 10.0)
    out = capsys.readouterr().out
    assert "d2s5@m33s11:RuntimeError: moe7 pre-flight" in out
    assert not re.search(r"^B moe m=33 v=d2s5 ", out, re.M)
    mt = re.search(r"^B moe m=33 v=d1s8 \| .* bw=(\S+)", out, re.M)
    assert mt and mt.group(1) == "OK"
    assert re.search(r"^B moe m=33 v=v5 ", out, re.M) and re.search(r"^B moe m=33 v=wstream ", out, re.M)
    torch.zeros(1, device="cuda").add_(1)
    torch.cuda.synchronize()


@needs_cuda
def test_item_counters_reset_inside_the_graph(mm, dev):
    """The g1 / g2 closures of a v7 stage set are NOT idempotent (item queue) -- g1z / g2z / pair carry the reset: replayed twice they produce identical outputs."""
    import reference

    w, state = dev
    d = mm._gen_input(8, 11, "normal", "cuda")
    c1, c2 = (1, 3, 4), (1, 3, 4)
    fns, st = mm._v7_stages(d, state, c1, c2)
    fd = dict(fns)
    fd["route"]()
    fd["pair"]()
    torch.cuda.synchronize()
    y1 = st["y"].clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fd["pair"]()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    live = st["local"].reshape(-1) >= 0
    assert torch.equal(st["y"].view(torch.int16)[live], y1.view(torch.int16)[live])
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        fd["g1z"]()
    g2.replay()
    g2.replay()
    torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------- sm_90 assembly (LAST in the file)
@needs_cuda
def test_measurement_source_assembles_for_sm90(mm, nv):
    """The whole measurement device source (8 instantiations, WSTREAM, CAL) assembles for the judge's sm_90 (NVRTC runs ptxas): every inline-asm string is accepted."""
    names = mm.meas_names(mm.MEAS_INSTANCES)
    cubin, low = nv.compile_device(mm.meas_device_src(), "sm_90", names=names)
    assert len(cubin) > 100000 and set(low) == set(names)


def test_meas_ext_exports_every_moe7_binding():
    """C30c judge run 4030a692fde24e74 died on `no attribute 'moe7_gather'`: the measurement extension must export every
    binding the v7 block exports and every `.moe7_*(` / `.meas_*(` name the file calls (no nvcc on the dev box catches this)."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "kernels/muse_moe_w4a16/measure_ws_v1.py").read_text()
    lists = [set(re.findall(r'"(\w+)"', m)) for m in re.findall(r"functions=\[([^\]]*)\]", src)]
    v7 = next(l for l in lists if "moe7_gemm" in l and "meas_cal" not in l)
    meas = next(l for l in lists if "meas_cal" in l)
    assert v7 <= meas, sorted(v7 - meas)
    called = set(re.findall(r"\.((?:moe7|meas)_\w+)\(", src))
    assert called <= (v7 | meas), sorted(called - (v7 | meas))
