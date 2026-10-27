"""kernels/muse_bf16_proj/measure_v3.py (MEASUREMENT-ONLY): kernel families, config tables, the _measure control flow.

CPU (Triton interpreter, FP16/FP32 stand-ins; the interpreter garbles BF16):
    TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_measure_v3.py
Any CUDA GPU (also the sm_75 dev card, FP16; compiles the real way, graph replay, fixup counters):
    tools/heavy.sh .venv/bin/python -m pytest -q tests/test_measure_v3.py
The dry-run tests (config tables, print size, parser round trip) need neither."""
import importlib.util
import os
import re
import sys
import zlib
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KERNEL = ROOT / "kernels" / "muse_bf16_proj" / "measure_v3.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="needs TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")

_spec = importlib.util.spec_from_file_location("measure_v3", KERNEL)
mv3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mv3)

from tools import parse_meas  # noqa: E402


# --------------------------------------------------------------------------------------------- source / tables

def test_source_compiles_and_is_measurement_only():
    src = KERNEL.read_text()
    compile(src, "f", "exec")
    assert src.startswith("# okc: problem=muse_bf16_proj")
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for banned in ("subprocess", "urllib", "requests", "socket", "os.system", ".item()"):
        assert banned not in src


def test_do_not_specialize_covers_m_n_k_split_and_strides():
    src = KERNEL.read_text()
    assert src.count("do_not_specialize") >= 2
    assert '["M", "N", "K", "stride_xm", "stride_wn", "stride_om"]' in src
    assert '["M", "N", "K", "SPLIT", "stride_xm", "stride_wn", "stride_om"]' in src


def test_config_tables():
    names_seen = 0
    fam_shapes = {f: set() for f in mv3.FAMILIES}
    for layer, m in mv3.all_shapes():
        tab = mv3.shape_cfgs(layer, m)
        names = [c[0] for c in tab]
        assert len(names) == len(set(names)), (layer, m)
        names_seen += len(names)
        n, k = mv3.shape_nk(layer, m)
        for name, fam, bm, bn, bk, split, w, s in tab:
            fam_shapes[fam].add((layer, m))
            assert name == f"{fam}{bm}.{bn}.{bk}.{split}.{w}.{s}"
            assert split * bk <= k and bn <= max(n, 64) and mv3._fits(bm, bn, bk, layer == "router", s)
            assert (split == 1) == (fam in ("A", "B"))
            assert (bm == 128) == (fam in ("G", "H"))
    assert len(mv3.all_shapes()) == 30
    # E1: all 20 shapes with m <= 32, both loaders
    e1 = {(l, m) for l, m in mv3.all_shapes() if m <= 32}
    assert len(e1) == 20 and fam_shapes["A"] == e1 and fam_shapes["B"] == e1
    # E2: router at every m, qkv_gate m=1, unpooler m <= 8
    e2 = {(l, m) for l, m in mv3.all_shapes() if l == "router" or (l, m) in (("qkv_gate", 1), ("unpooler", 1), ("unpooler", 8))}
    assert fam_shapes["F"] == e2 and fam_shapes["Z"] == e2
    assert {m for l, m in e2 if l == "router"} == {1, 8, 16, 32, 128, 256}
    # E3: m=128/256 for the four BF16 layers
    e3 = {(l, m) for l, m in mv3.all_shapes() if m >= 128 and l != "router"}
    assert len(e3) == 8 and fam_shapes["G"] == e3 and fam_shapes["H"] == e3
    # E3 grid: BN {64,128,256} x SPLIT {2,4,8}
    tab = mv3.shape_cfgs("qkv_gate", 128)
    assert {c[3] for c in tab} == {64, 128, 256} and {c[5] for c in tab if c[1] == "G"} == {2, 4, 8}
    # E1 grid on a decode shape
    tab = mv3.shape_cfgs("pooler", 32)
    assert {c[2] for c in tab} == {16, 32} and {c[3] for c in tab} >= {8, 16, 32} and {c[4] for c in tab} >= {512, 1024}
    assert {c[7] for c in tab} >= {3, 4, 5, 6} and {c[6] for c in tab} == {4, 8}
    assert names_seen == mv3.cfg_sig()[0]
    assert mv3.cfg_sig() == mv3.cfg_sig()


def test_cfg_block_is_pure_python():
    ns = parse_meas.load_cfg(KERNEL)
    assert ns["cfg_sig"]() == mv3.cfg_sig()
    assert ns["shape_cfgs"]("router", 256) == mv3.shape_cfgs("router", 256)


def test_scratch_capacity_covers_every_config():
    for layer, m in mv3.all_shapes():
        n, k = mv3.shape_nk(layer, m)
        for name, fam, bm, bn, bk, split, w, s in mv3.shape_cfgs(layer, m):
            if fam in "FG":
                tiles_m, tiles_n = -(-m // bm), -(-n // bn)
                assert tiles_m * bm * tiles_n * bn * split <= mv3.WS_FLOATS, name
                assert tiles_m * tiles_n <= mv3.SEM_COUNT, name


# --------------------------------------------------------------------------------------------- kernels

def _case(fam, bm, bn, bk, split, m, n, k, lo, dev, w_=4, s_=3, reps=3):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(m, k, generator=g).half().to(dev)
    wf = torch.randn(n, k, generator=g)
    hi = wf.half()
    lov = (wf - hi.float()).half() if lo else None
    ref = x.double().cpu() @ (wf.double() if lo else hi.double()).T
    odt = torch.float32 if lo else torch.float16
    ws = torch.empty(1 << 20, dtype=torch.float32, device=dev)
    sem = torch.zeros(256, dtype=torch.int32, device=dev)
    run = mv3._make_run(fam, bm, bn, bk, split, w_, s_, hi.to(dev), None if lov is None else lov.to(dev), n, odt, ws, sem)
    outs = [run(x).clone() for _ in range(reps)]
    err = float((outs[0].double().cpu() - ref).norm() / ref.norm())
    assert err < (1e-5 if lo else 2e-3), (fam, err)
    for o in outs[1:]:  # replays: fixup is deterministic and leaves its counters at zero
        if fam in "AFBG":
            assert torch.equal(o, outs[0])
        else:
            assert torch.allclose(o.double(), outs[0].double(), rtol=1e-2, atol=1e-2)
    assert int(sem.abs().sum()) == 0
    return run


FAMS_SMALL = [("A", 1), ("B", 1), ("F", 4), ("Z", 4), ("G", 4), ("H", 4)]


@needs_interpreter
@pytest.mark.parametrize("fam,split", FAMS_SMALL)
@pytest.mark.parametrize("lo", [False, True])
def test_kernel_families_interpreter(fam, split, lo):
    bm = 128 if fam in "GH" else 16
    _case(fam, bm, 16, 32, split, 5, 32, 256, lo, "cpu")


@needs_interpreter
@pytest.mark.parametrize("fam,split", [("A", 1), ("B", 1), ("F", 3), ("Z", 3), ("G", 3), ("H", 3)])
def test_kernel_ragged_shapes_interpreter(fam, split):
    """m, n, k not multiples of the tile: masks, EVEN_K/EVEN_N False, split with a short last chunk."""
    bm = 32 if fam in "GH" else 16
    _case(fam, bm, 16, 32, split, 20 if bm == 16 else 33, 24, 200, False, "cpu")


@needs_interpreter
def test_fixup_semaphore_logic_interpreter():
    """Several m-tiles and n-tiles share one counter array: each tile's last CTA reduces and resets its own counter."""
    run = _case("F", 16, 16, 32, 5, 40, 48, 320, False, "cpu", reps=4)
    assert run.buf is not None


@needs_interpreter
def test_fixup_counter_state_after_launch_interpreter():
    ws = torch.empty(1 << 14)
    sem = torch.zeros(64, dtype=torch.int32)
    w = torch.randn(32, 128).half()
    x = torch.randn(4, 128).half()
    run = mv3._make_run("F", 16, 16, 32, 4, 4, 3, w, None, 32, torch.float16, ws, sem)
    run(x)
    assert int(sem.sum()) == 0
    sem_big = torch.zeros(1, dtype=torch.int32)  # too few counters for 2 tiles -> _Unfit, not a crash
    run2 = mv3._make_run("F", 16, 16, 32, 4, 4, 3, w, None, 32, torch.float16, ws, sem_big)
    with pytest.raises(mv3._Unfit):
        run2(x)


@needs_cuda
@pytest.mark.parametrize("fam,split", FAMS_SMALL)
def test_kernel_families_gpu_fp16(fam, split):
    if fam == "B" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA descriptors need Hopper")
    big = fam in "GH"
    run = _case(fam, 128 if big else 16, 64 if big else 16, 64, split, 256 if big else 5, 512 if big else 256,
                2048, False, "cuda", w_=8 if big else 4)
    del run


@needs_cuda
def test_kernel_router_lo_and_graph_gpu_fp16():
    run = _case("F", 16, 16, 64, 8, 20, 64, 1024, True, "cuda")
    x = torch.randn(20, 1024, device="cuda").half()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run(x)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        y = run(x)
    first = None
    for _ in range(4):
        g.replay()
        torch.cuda.synchronize()
        first = y.clone() if first is None else first
        assert torch.equal(y, first)


# --------------------------------------------------------------------------------------------- _measure dry run

def _h(o):
    return zlib.crc32(repr(o).encode())


class StubEnv:
    """CPU stand-in for _RealEnv: timings are a hash of (shape, cfg), the clock advances per call."""

    def __init__(self, step=0.0005, tma_ok=True, bad_every=0, fail_bn=None, scale=1.0, base_us=600.0):
        self.t = 0.0
        self.step, self.tma_ok, self.bad_every, self.fail_bn, self.scale, self.base_us = step, tma_ok, bad_every, fail_bn, scale, base_us
        self.tma_msg = ""
        self.timed = 0
        self.compiled = set()

    def available(self):
        return True

    def clock(self):
        return self.t

    def banner(self):
        return "gpu=stub triton=0 torch=0"

    def setup(self):
        pass

    def tma_smoke(self):
        return (True, "err=1e-4") if self.tma_ok else (False, "RuntimeError: no tma on this device")

    def prep(self, layer, m):
        n, k = mv3.shape_nk(layer, m)
        return dict(layer=layer, m=m, n=n, k=k, tol=1e-4 if layer == "router" else 1e-2)

    def build(self, ctx, cfg):
        self.t += self.step
        self.compiled.add(mv3._key(cfg, ctx["layer"]))
        if self.fail_bn is not None and cfg[3] == self.fail_bn and cfg[1] in "AB":
            raise RuntimeError("compile error: N=8 dot unsupported")
        if cfg[1] == "H" and cfg[3] == 256 and cfg[4] == 64:
            raise mv3._Unfit("workspace")
        return (ctx, cfg)

    def check(self, ctx, run):
        self.t += self.step
        h = _h((ctx["layer"], ctx["m"], run[1][0])) % 1000
        if self.bad_every and h % self.bad_every == 0:
            return 1.0
        return 1e-6

    def _us(self, ctx, name):
        h = (_h((ctx["layer"], ctx["m"], name)) & 0xFFFF) / 65535.0
        return (5.0 + 595.0 * h) * self.scale  # up to ~600 us: the widest print

    def time(self, ctx, run, reps):
        self.t += self.step * (1 + reps / 20)
        self.timed += 1
        return self._us(ctx, run[1][0]), 1e-6

    def time_base(self, ctx, reps):
        self.t += self.step
        return self.base_us + (_h(ctx["layer"]) & 7)


def _dry(capsys, **kw):
    env = StubEnv(**kw)
    mv3._measure({}, None, env)
    return capsys.readouterr().out, env


def test_dry_run_print_size_and_structure(capsys):
    out, env = _dry(capsys)
    assert len(out.encode()) < 15000, len(out.encode())
    lines = out.splitlines()
    assert lines[0].startswith("MEAS3 start") and lines[1].startswith("CFG ") and lines[2].startswith("CFGSIG ")
    assert sum(l.startswith("T64 ") for l in lines) == 30 and sum(l.startswith("B ") for l in lines) == 30
    assert lines[-1].startswith("B ") and not any(l.startswith("T64") for l in lines[-30:])
    assert "MEAS3 done" in out and "aborted" not in out and "TMA smoke: ok" in out
    for l in lines:
        if l.startswith("B "):
            assert re.match(r"^B \w+ m=\d+ base=[\d.]+ \| top-5 (\S+:[\d.]+ ?){1,5} \| ok=\d+ fail=\d+ bad=\d+$", l), l


def test_dry_run_compiles_each_key_once_and_times_everything(capsys):
    out, env = _dry(capsys)
    total = mv3.cfg_sig()[0]
    n_keys = len({mv3._key(c, l) for l, m in mv3.all_shapes() for c in mv3.shape_cfgs(l, m)})
    assert len(env.compiled) == n_keys
    # screen timings = every config except the _Unfit ones; finals add 5 per shape
    assert env.timed >= total - 100
    m = re.search(r"CFGSIG n=(\d+)", out)
    assert int(m.group(1)) == total


def test_dry_run_failures_bad_and_tma_fail(capsys):
    out, env = _dry(capsys, tma_ok=False, bad_every=11, fail_bn=8)
    assert "TMA smoke: FAIL" in out
    assert re.search(r"^FAIL A x\d+ RuntimeError: compile error", out, re.M)
    assert len(out.encode()) < 15000
    full = parse_meas.parse_full(out, KERNEL)
    cnt = full["meta"]["counts"]["pooler.m1"]
    assert cnt["fail"] > 0 and cnt["bad"] > 0
    assert not any(k.startswith("B") for k in full["data"]["pooler.m1"] if k != "base")


def test_dry_run_budget_cut_reports_skips(capsys):
    out, env = _dry(capsys, step=0.2)
    assert "skip=" in out
    assert len(out.encode()) < 15000
    assert "MEAS3 done" in out


def test_dry_run_worst_case_widths(capsys):
    out, env = _dry(capsys, scale=1.09, base_us=640.0)
    assert len(out.encode()) < 15000
    print(f"dry-run print size: {len(out.encode())} bytes")


def test_roundtrip_with_parser(capsys):
    out, env = _dry(capsys)
    full = parse_meas.parse_full(out, KERNEL)
    data = full["data"]
    assert len(data) == 30 and full["meta"]["sig_ok"] is True
    d = data["router.m128"]
    assert "base" in d and 40 < len(d) < 130
    # B-line top-5 equals the best of the T64 values (finals override the screen value)
    best = parse_meas.best_per_shape(data)
    row = next(r for r in best if r["shape"] == "router.m128")
    assert row["us"] == min(v for k, v in d.items() if k != "base")
    assert "delta_us" in row


def test_prepare_returns_baseline_state_without_cuda(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    w = {"qkv_gate": torch.randn(8, 16).bfloat16(), "o": torch.randn(16, 8).bfloat16(), "pooler": torch.randn(8, 16).bfloat16(),
         "unpooler": torch.randn(16, 8).bfloat16(), "router": torch.randn(4, 16)}
    st = mv3.prepare(w)
    assert set(st) >= {"qkv_gate", "router", "router_hi_t", "router_lo_t"}
    assert "skipped" in capsys.readouterr().out
    x = torch.randn(2, 16).bfloat16()
    y = mv3.kernel_fn({"x": x, "config": {"m": 2, "n": 4, "k": 16, "layer": "router"}}, st)
    assert y.shape == (2, 4) and y.dtype == torch.float32


def test_device_fault_after_finals1_keeps_results(capsys):
    class Faulty(StubEnv):
        def tma_smoke(self):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    env = Faulty()
    mv3._measure({}, None, env)
    out = capsys.readouterr().out
    assert "MEAS3 aborted: RuntimeError: CUDA error" in out and "phase finals1" in out
    data = parse_meas.parse_full(out, KERNEL)["data"]
    assert len(data) == 30 and all(k[0] != "B" for d in data.values() for k in d if k != "base")
    assert len(data["pooler.m1"]) > 5
