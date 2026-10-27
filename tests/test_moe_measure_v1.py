"""kernels/muse_moe_w4a16/measure_v1.py (MEASUREMENT-ONLY): the config table, the _measure control flow with a stubbed timing environment,
the real _RealEnv path on a CUDA card with stubbed SGLang ops, and the log parsers.

No GPU needed (all but the last group):   tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_measure_v1.py
sm_75 dev card (graph capture, Triton epilogue variants, stub Marlin) runs the `gpu` tests too."""
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KERNEL = ROOT / "kernels" / "muse_moe_w4a16" / "measure_v1.py"
SONNET = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v1.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


mv = _load("measure_moe_v1", KERNEL)

from tools import parse_meas  # noqa: E402
from tools import parse_meas_moe  # noqa: E402


# --------------------------------------------------------------------------------------------- source rules

def test_source_compiles_and_is_measurement_only():
    src = KERNEL.read_text()
    compile(src, "f", "exec")
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for banned in ("subprocess", "urllib", "requests", "socket", "os.system", ".item()", ".cpu()", "import os", "environ", "tolist("):
        assert banned not in src, banned
    assert "kernels.muse_" not in src and "import reference" not in src and "from reference" not in src


def test_kernel_path_is_a_verbatim_copy_of_sonnet_v1():
    a, b = SONNET.read_text(), KERNEL.read_text()
    lo, hi = "import types\n", "# ============================================================================================ "
    def region(t, end_marker):
        return t[t.index(lo): t.index(end_marker)]
    assert region(a, hi + "entry points") == region(b, hi + "measurement (MEASUREMENT-ONLY)")
    # kernel_fn itself too
    assert a[a.index("def kernel_fn"):] == b[b.index("def kernel_fn"):]


# --------------------------------------------------------------------------------------------- config table

def test_config_table():
    seen = 0
    for layer, m in mv.all_shapes():
        tab = mv.shape_cfgs(layer, m)
        names = [c[0] for c in tab]
        assert len(names) == len(set(names))
        seen += len(names)
        bbm, bep = mv.base_bm(m), mv.base_epi(m)
        for name, bm, al, r1, r2, ep in tab:
            assert name == f"b{bm}.{al}.r{r1}{r2}.{ep}"
            assert bm in mv.BMS and al in ("a0", "a1", "a2", "a3", "a4") and r1 + r2 in mv.RED_PAIRS + ("ff",)
        assert ("a4" in {c[2] for c in tab}) == (m == 1)
        base = tab[mv.base_index(m)]
        assert base == (f"b{bbm}.a0.rff.{bep}", bbm, "a0", "f", "f", bep)
        assert mv.deviations(base, m) == 0
        assert {c[1] for c in tab} == set(mv.BMS)
        # 12 epilogue variants at the baseline of everything else, the rest is the product with the baseline epilogue
        extra = [c for c in tab if c[5] != bep]
        assert len(extra) == 12 and all(c[1:5] == (bbm, "a0", "f", "f") for c in extra)
        assert {c[5] for c in tab} == {"sg"} | {f"t{bh}w{w}" for bh in mv.EPI_BH for w in mv.EPI_W}
        assert len(tab) == len(mv.BMS) * (5 if m == 1 else 4) * len(mv.RED_PAIRS) + 12
    assert [mv.base_bm(m) for m in mv.MS] == [8, 8, 8, 16, 48, 64]  # baseline.py's rule at the six scored M
    assert [mv.base_epi(m) for m in mv.MS] == ["t1024w8"] * 3 + ["sg"] * 3
    assert mv.cfg_sig()[0] == seen == 1572
    assert mv.cfg_sig() == mv.cfg_sig()


def test_base_bm_matches_baseline_loop():
    for m in (1, 4, 8, 16, 32, 33, 128, 257, 256):
        for bm in mv.BLOCK_SIZE_M_CANDIDATES:
            if m * mv.TOP_K / 32 / bm < 0.9:
                break
        assert mv.base_bm(m) == bm, m


def test_cfg_block_is_pure_python():
    ns = parse_meas.load_cfg(KERNEL)
    assert ns["cfg_sig"]() == mv.cfg_sig()
    assert ns["shape_cfgs"]("moe", 1) == mv.shape_cfgs("moe", 1)


def test_routing_is_seeded_skewed_and_local():
    a = mv._routing(128, 4242 + 128)
    b = mv._routing(128, 4242 + 128)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    ids, w = a
    assert ids.shape == (128, 8) and ids.dtype == torch.int32 and ids.min() >= 0 and ids.max() < 256
    assert all(len(set(r.tolist())) == 8 for r in ids)
    assert torch.allclose(w.sum(-1), torch.ones(128))
    per_gpu = torch.bincount((ids // 32).flatten(), minlength=8)
    assert per_gpu.argmax() == 0 and per_gpu[0] > 2 * per_gpu.float().mean()  # GPU 0 (experts 0-31) is the busiest, skewed
    for m in mv.MS:
        i, _ = mv._routing(m, 4242 + m)
        assert ((i >= 0) & (i < 32)).any()


# --------------------------------------------------------------------------------------------- _measure control flow, stub env

class StubEnv:
    """Deterministic timings with no GPU. build() returns the cfg, check() a fake error, time() a fake median; the clock advances per call."""

    def __init__(self, step=0.004, base_us=60.0, scale=1.0, bad_every=0, fail_al=None, fail_all_ep=None, stage_fail=False):
        self.t, self.step, self.base_us, self.scale = 0.0, step, base_us, scale
        self.bad_every, self.fail_al, self.fail_all_ep, self.stage_fail = bad_every, fail_al, fail_all_ep, stage_fail
        self.timed, self.built, self.released, self.order = 0, 0, False, []

    def available(self): return True
    def clock(self): return self.t
    def banner(self): return "gpu=stub"
    def setup(self): pass
    def release(self): self.released = True

    def prep(self, layer, m):
        return dict(layer=layer, m=m, info=f"pairs={m * 2} local={m} distinct={min(32, m * 2)}")

    def build(self, ctx, cfg):
        self.t += self.step
        self.built += 1
        self.order.append(cfg[2])
        if cfg[2] == self.fail_al:
            raise RuntimeError("stub: bad align")
        if cfg[5] == self.fail_all_ep:
            raise RuntimeError("stub: epilogue compile error")
        return (ctx["m"], cfg)

    def _h(self, cfg):
        return sum(map(ord, cfg[0])) % 97

    def check(self, ctx, run):
        self.t += self.step
        if self.bad_every and self._h(run[1]) % self.bad_every == 0:
            return 0.5
        return 0.0

    def _us(self, run):
        m, cfg = run
        return self.base_us * self.scale + 0.03 * m ** 0.5 + 0.1 * self._h(cfg) % 5 - (3.0 if cfg[2] == "a1" else 0.0) + 0.37 * (cfg[1] != mv.base_bm(m))

    def time(self, ctx, run, reps):
        self.t += self.step * (1 + reps / 40)
        self.timed += 1
        return self._us(run), 0.0

    def time_base(self, ctx, reps):
        self.t += self.step
        return self.base_us * self.scale + ctx["m"] % 7

    def stage_times(self, ctx, base_cfg, reps):
        self.t += self.step * 6
        if self.stage_fail:
            raise RuntimeError("stub: stage graph")
        return dict(ids=3.1, align=5.5, w13=40.2, act=6.0, w2=38.7, epi=3.3, nop=1.9)


def _dry(capsys, **kw):
    env = StubEnv(**kw)
    mv._measure({}, env)
    return capsys.readouterr().out, env


def test_dry_run_structure_and_print_size(capsys):
    out, env = _dry(capsys)
    n = len(out.encode())
    assert n < 12000, n
    lines = out.splitlines()
    assert lines[0].startswith("MEAS-MOE start") and lines[1].startswith("CFGSIG ")
    assert sum(l.startswith("T64 ") for l in lines) == 6 and sum(l.startswith("R moe") for l in lines) == 6
    for tag in "BSK":
        tail = [l for l in lines if l.startswith(tag + " ")]
        assert len(tail) == 6
    assert lines[-6].startswith("B ") and lines[-12].startswith("S ") and lines[-18].startswith("K ")  # human lines last
    assert "MEAS-MOE done" in out and "aborted" not in out and env.released
    for l in lines:
        if l.startswith("B "):
            assert re.match(r"^B moe m=\d+ base=[\d.]+ \| top-5 (\S+:[\d.]+ ?){1,5} \| ok=\d+ fail=\d+ bad=\d+$", l), l
        if l.startswith("S "):
            assert re.match(r"^S moe m=\d+ align=[\d.]+ w13=[\d.]+ act=[\d.]+ w2=[\d.]+ epi=[\d.]+ ids=[\d.]+ nop=[\d.]+ sum=[\d.]+$", l), l
    print(f"dry-run print size: {n} bytes")


def test_dry_run_times_every_config_once_and_finals_add_up(capsys):
    out, env = _dry(capsys)
    total = mv.cfg_sig()[0]
    assert env.timed >= total  # screen of every config + finals (top 5 and the combo per M)
    assert env.timed <= total + 6 * 6
    assert f"CFGSIG n={total}" in out


def test_dry_run_failures_bad_and_dead_axis(capsys):
    out, env = _dry(capsys, bad_every=11, fail_al="a1", fail_all_ep="t256w2")
    assert re.search(r"^FAIL RuntimeError: stub: ", out, re.M)
    assert len(out.encode()) < 12000
    full = parse_meas.parse_full(out, KERNEL)
    cnt = full["meta"]["counts"]["moe.m8"]
    assert cnt["fail"] > 0 and cnt["bad"] > 0
    assert not any(".a1." in k for k in full["data"]["moe.m8"] if k != "base")
    assert not any(k.endswith(".t256w2") for k in full["data"]["moe.m8"] if k != "base")
    # a1 failed 3 times with no success -> the rest of a1 is marked F without being built
    assert env.built < mv.cfg_sig()[0]


def test_a4_runs_after_everything_else_and_gets_final_timings(capsys):
    out, env = _dry(capsys)
    first = env.order.index("a4")
    # all non-a4 screening (1512 configs) and the finals of every M happen before the first a4 build
    assert first >= 1512
    assert env.order[first:].count("a4") >= 60
    full = parse_meas.parse_full(out, KERNEL)
    assert sum(".a4." in k for k in full["data"]["moe.m1"]) == 60
    assert "MEAS-MOE phase a4" in out


def test_dry_run_stage_failure_is_reported_not_fatal(capsys):
    out, env = _dry(capsys, stage_fail=True)
    assert "S moe m=1 n/a" in out and "FAIL stage RuntimeError" in out and "MEAS-MOE done" in out
    assert sum(l.startswith("B ") for l in out.splitlines()) == 6


def test_dry_run_budget_cut_reports_skips(capsys):
    out, env = _dry(capsys, step=0.15)
    assert "skip=" in out and "MEAS-MOE done" in out
    assert len(out.encode()) < 12000
    # one-axis configs come before multi-axis ones: the baseline and its single-axis neighbours are timed first
    full = parse_meas.parse_full(out, KERNEL)
    d = full["data"]["moe.m16"]
    assert "base" in d and len(d) > 3


def test_dry_run_worst_case_widths(capsys):
    out, env = _dry(capsys, scale=10.0, base_us=640.0)
    assert len(out.encode()) < 12000
    print(f"dry-run worst-case print size: {len(out.encode())} bytes")


def test_skips_without_cuda_or_sglang(capsys):
    class NoEnv(StubEnv):
        def available(self): return False
    mv._measure({}, NoEnv())
    assert "skipped" in capsys.readouterr().out
    mv._measure({})  # the real env without SGLang / CUDA
    assert "skipped" in capsys.readouterr().out or torch.cuda.is_available()


# --------------------------------------------------------------------------------------------- parsers

def test_roundtrip_with_parsers(capsys):
    out, env = _dry(capsys)
    full = parse_meas.parse_full(out, KERNEL)
    data = full["data"]
    assert len(data) == 6 and full["meta"]["sig_ok"] is True
    for m in mv.MS:
        d = data[f"moe.m{m}"]
        assert "base" in d
        assert len(d) - 1 == len(mv.shape_cfgs("moe", m))  # every config has a value (nothing cut, nothing failed)
    best = parse_meas.best_per_shape(data)
    row = next(r for r in best if r["shape"] == "moe.m8")
    assert row["us"] == min(v for k, v in data["moe.m8"].items() if k != "base")
    res = parse_meas_moe.parse(out, KERNEL)
    assert set(res["stage"]) == {f"moe.m{m}" for m in mv.MS} and res["stage"]["moe.m16"]["w13"] == 40.2
    assert res["stage"]["moe.m1"]["nop"] == 1.9
    assert res["routing"]["moe.m8"]["pairs"] == 16
    assert res["axes"]["moe.m8"]["bm"]
    text = parse_meas_moe.render(res, KERNEL)
    assert "moe.m256" in text and "w13=40.20" in text


# --------------------------------------------------------------------------------------------- real env on a CUDA card, stub SGLang

def _install_stubs(monkeypatch, fail_a1=False):
    """Torch stand-ins for the SGLang ops so the real _RealEnv (graph capture, flush, events) runs on any CUDA card."""
    dt = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16
    monkeypatch.setattr(mv, "_DT", dt)
    monkeypatch.setattr(mv, "HAVE_SGLANG", True)
    monkeypatch.setattr(mv, "scalar_types", types.SimpleNamespace(float4_e2m1f=object()), raising=False)

    def align(ids, bm, n_exp, ignore_invalid_expert=False):
        if fail_a1 and n_exp == 32:
            raise RuntimeError("stub align")
        numel = ids.numel()
        padded = (numel + (n_exp + 1) * (bm - 1) + 3) // 4 * 4
        return (torch.zeros(padded, dtype=torch.int32, device=ids.device), torch.zeros((padded + bm - 1) // bm, dtype=torch.int32, device=ids.device),
                torch.zeros(1, dtype=torch.int32, device=ids.device))

    def marlin(a, c, *args, top_k, size_m, size_n, size_k, moe_block_size, use_atomic_add, use_fp32_reduce, **kw):
        rows = c.shape[0]
        rep = a.float().repeat_interleave(rows // a.shape[0], 0)
        base = rep.repeat(1, size_n // rep.shape[1]) * (1 + torch.arange(size_n, device=a.device) * 1e-3)
        if not use_atomic_add and not use_fp32_reduce:  # pretend the bf16 reduction is wrong: those configs must come out `bad`
            base = base.flip(1)
        c.copy_(base.to(c.dtype))
        return c

    def silu(x, out, expert_ids=None, expert_step=1):
        i = out.shape[1]
        out.copy_((torch.nn.functional.silu(x[:, :i].float()) * x[:, i:].float()).to(out.dtype))

    def fused(y, tw, norm_weight, eps, weight_first, slot_expert_ids):
        return mv.epilogue_torch(y, tw, slot_expert_ids, norm_weight, eps)

    monkeypatch.setattr(mv, "moe_align_block_size", align, raising=False)
    monkeypatch.setattr(mv, "moe_wna16_marlin_gemm", marlin, raising=False)
    monkeypatch.setattr(mv, "silu_and_mul", silu, raising=False)
    monkeypatch.setattr(mv, "fused_post_expert_norm_weighted_sum", fused, raising=False)
    dev = torch.device("cuda")
    emap = torch.full((256,), -1, dtype=torch.int32, device=dev)
    emap[:32] = torch.arange(32, dtype=torch.int32, device=dev)
    z = torch.zeros(1, device=dev)
    return {"banks": {(0, 32): {"w13": z, "w13_s": z, "w13_gs": z, "w2": z, "w2_s": z, "w2_gs": z, "expert_map": emap, "count": 32,
                                "size_n": 4096}},
            "workspace": torch.zeros(1056, dtype=torch.int32, device=dev),
            "norm_w": (0.1 * torch.randn(4096, device=dev)).to(dt) + 1}


@needs_cuda
def test_real_env_with_stub_ops_end_to_end(monkeypatch, capsys):
    state = _install_stubs(monkeypatch)
    for k, v in dict(SCREEN_REPS=3, FINAL_REPS=5, FLUSH_MB=8, BUDGET_S=150.0, EARLY_END_S=100.0).items():
        monkeypatch.setattr(mv, k, v)
    mv._measure(state)
    out = capsys.readouterr().out
    assert len(out.encode()) < 12000, len(out.encode())
    assert "aborted" not in out and "MEAS-MOE done" in out, out[-3000:]
    full = parse_meas.parse_full(out, KERNEL)
    cnt = full["meta"]["counts"]
    for m in mv.MS:
        c = cnt[f"moe.m{m}"]
        assert c["ok"] > 0 and c["skip"] == 0
        assert c["bad"] >= 0.2 * len(mv.shape_cfgs("moe", m)) * 0.5  # the "n" reduce modes of the stub are wrong -> bad
        d = full["data"][f"moe.m{m}"]
        assert d["base"] > 0
        # no variant that uses the wrong "n" reduce was timed
        offending = [k for k in d if k != "base" and re.search(r"\.r(n.|.n)\.", k)]
        assert not offending, offending[:5]
    res = parse_meas_moe.parse(out, KERNEL)
    for m in mv.MS:
        s = res["stage"][f"moe.m{m}"]
        assert all(s[k] > 0 for k in ("ids", "align", "w13", "act", "w2", "epi", "nop"))
    print(out[-2500:])


@needs_cuda
def test_real_env_failing_align_variant_is_counted(monkeypatch, capsys):
    state = _install_stubs(monkeypatch, fail_a1=True)
    for k, v in dict(SCREEN_REPS=2, FINAL_REPS=3, FLUSH_MB=8, BUDGET_S=150.0, EARLY_END_S=100.0).items():
        monkeypatch.setattr(mv, k, v)
    monkeypatch.setattr(mv, "MS", (1, 8))
    mv._measure(state)
    out = capsys.readouterr().out
    assert "FAIL RuntimeError: stub align" in out and "MEAS-MOE done" in out


@needs_cuda
def test_baseline_cfg_stages_reproduce_kernel_fn_bit_for_bit(monkeypatch):
    state = _install_stubs(monkeypatch)
    env = mv._RealEnv(state)
    env.setup()
    for m in (1, 16, 33):
        ctx = env.prep("moe", m)
        base = mv.shape_cfgs("moe", m)[mv.base_index(m)]
        assert env.check(ctx, env.build(ctx, base)) == 0.0
        # the other epilogues and the align variants agree with it to rounding; the explicit-zero / atomic stand-ins too
        for cfg in mv.shape_cfgs("moe", m):
            if cfg[3] + cfg[4] in ("ff", "tt", "zz") and cfg[2] in ("a0", "a1", "a2", "a3"):
                assert env.check(ctx, env.build(ctx, cfg)) < mv.TOL, cfg
    env.release()
