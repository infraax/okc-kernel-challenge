"""kernels/muse_moe_w4a16/measure_grouped_v1.py (MEASUREMENT-ONLY, generated): file rules, config tables, the _measure control flow with a stub
environment (no GPU), and the whole thing on the RTX 2060 against a fake SGLang (FP16 stand-ins; Marlin replaced by a torch dequant GEMM).

No GPU:        tools/heavy.sh .venv/bin/python -m pytest -q tests/test_measure_grouped_v1.py
RTX 2060:      GROUPED_GPU=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_measure_grouped_v1.py -k gpu"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
PROBLEM = "muse_moe_w4a16"
sys.path.insert(0, str(ROOT / "problems" / PROBLEM))
sys.path.insert(0, str(ROOT))
MEASURE = ROOT / "kernels" / PROBLEM / "measure_grouped_v1.py"
GPU = os.environ.get("GROUPED_GPU") == "1" and torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs GROUPED_GPU=1 and a CUDA device")

import nvfp4  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("_measure_grouped_v1", MEASURE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture()
def mg():
    return _load()


def test_source_rules_and_in_sync_with_generator():
    src = MEASURE.read_text()
    compile(src, "measure_grouped_v1.py", "exec")
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    assert "MEASUREMENT-ONLY" in src and "def _measure" in src and "NEVER submit" in src
    for banned in ("subprocess", "urllib", "requests", "socket", "os.system", ".item(", ".cpu(", "environ", "import os", ".tolist("):
        assert banned not in src, banned
    assert "USE_GROUPED = False" in src
    from tools import gen_measure_grouped_v1 as gen
    assert gen.build() == src, "regenerate: .venv/bin/python tools/gen_measure_grouped_v1.py"
    cand = (ROOT / "kernels" / PROBLEM / "grouped_v1.py").read_text()
    assert "MEASUREMENT-ONLY" not in cand and "def _measure" not in cand  # okc_run refuses to submit files that have them


def test_config_tables(mg):
    for bs in mg.SHAPES:
        for gemm in (1, 2):
            tab = mg.g_table(bs, gemm)
            assert tab and len(tab) == len(set(tab))
            names = {mg.cfg_name(c, d) for c in tab for d in (0, 1, 2)}
            assert len(names) == 3 * len(tab)
            for bm, bnh, ch, warps, stages in tab:
                assert bm in (16, 32, 64, 128) and bnh % 16 == 0 and warps in (4, 8) and 4096 % (128 * ch) == 0
                assert 4096 % bnh == 0
    assert any(c[0] == 128 for c in mg.g_table(256, 1)) and not any(c[0] == 128 for c in mg.g_table(32, 1))
    assert (64, 64, 1, 8, 3) in mg.g_table(256, 1)  # the dequant-mode probe config


class _T(tuple):
    """A tuple that can be 'run' (the stub's closures)."""

    def __call__(self):
        return None


class _Stub:
    """Environment without a GPU: deterministic fake timings, one config that fails and one that is wrong."""

    def __init__(self):
        self.t = 0.0
        self.calls = []

    def available(self):
        return True

    def clock(self):
        self.t += 0.05
        return self.t

    def banner(self):
        return "gpu=stub"

    def setup(self):
        pass

    def sync(self):
        pass

    def prep(self, bs, dist="skewed"):
        return dict(bs=bs, m1=_T(("m1", bs)), m2=_T(("m2", bs)))

    def marlin_e2e(self, ctx):
        return _T(("marlin", ctx["bs"]))

    def g_run(self, ctx, gemm, c, deq, mult=1):
        return _T(("g", ctx["bs"], gemm, c, deq))

    def e2e_run(self, ctx, c1, c2, deq, mult=1):
        return _T(("e2e", ctx["bs"], c1, c2, deq, mult))

    def check(self, ctx, gemm, run):
        _, bs, g, c, deq = run
        if deq == 2 and c[0] == 128:
            return 0.9  # wrong output
        if c == (64, 256, 1, 8, 3):
            raise RuntimeError("OutOfResources: stub")
        return 1e-3

    def time(self, fn, reps):
        base = {"m1": 300.0, "m2": 150.0, "marlin": 600.0}
        if fn[0] in base:
            return base[fn[0]]
        if fn[0] == "g":
            _, bs, g, c, deq = fn
            return 100.0 + c[0] * 0.5 + c[1] * 0.1 + c[3] + 10 * (2 - deq) + (reps % 5) * 0.01
        _, bs, c1, c2, deq, mult = fn
        return 200.0 + c1[0] * 0.2 - 5 * mult + 10 * (2 - deq)


def test_measure_flow_with_stub_env(mg, capsys):
    mg._measure(None, None, env=_Stub())
    out = capsys.readouterr().out
    assert len(out.encode()) < 12 * 1024
    lines = out.splitlines()
    b_lines = [l for l in lines if l.startswith("B moeg ")]
    assert [re.search(r"bs=(\d+)", l).group(1) for l in b_lines] == ["32", "128", "256"]
    for l in b_lines:
        assert "marlin g1=300.0 g2=150.0 e2e=600.0" in l and "g1 top5 d2" in l and "e2e d2" in l and "vs marlin" in l
        assert "bad=" in l and "fail=" in l
    assert lines[-3].startswith("B moeg") and any(l.startswith("MEASG done") for l in lines)
    assert any(l.startswith("S moeg g1 bs=256") for l in lines) and any(l.startswith("FAIL ") for l in lines)
    assert "modes d0=" in out and "d2=" in out
    # the stub's wrong config is reported as B, the failing one as F, never timed
    s256 = next(l for l in lines if l.startswith("S moeg g2 bs=256"))
    assert " d2b64n256c1w8s3:F" in s256 or "n256c1w8s3:F" in s256


def test_measure_deadline_stops_the_sweep(mg, capsys):
    env = _Stub()
    env.clock = lambda: 1e6  # past every phase deadline
    mg._measure(None, None, env=env)
    out = capsys.readouterr().out
    assert "MEASG done" in out and out.count("B moeg") == 3  # still reports (setup + probe ran)


# ---------------------------------------------------------------------------------------------- the RTX 2060, fake SGLang
def _install_fake_sglang(m, w):
    """Replace the SGLang entry points of the loaded module by torch versions over the raw (un-repacked) weights."""
    from types import SimpleNamespace
    h = w["w13"].shape[2] * 2
    inter = w["w13"].shape[1] // 2
    deq13 = nvfp4.dequantize(w["w13"][:32], w["w13_sf"][:32], 1.0) * w["w13_gs"][:32, None, None]
    deq2 = nvfp4.dequantize(w["w2"][:32], w["w2_sf"][:32], 1.0) * w["w2_gs"][:32, None, None]

    def marlin(a, c, wq, b, ws_, gs_, *rest, **kw):
        """Dense and sync-free (CUDA-graph capturable): every expert multiplies every slot, the slot -> expert map comes from the block lists."""
        sorted_ids, expert_ids = rest[4], rest[5]
        size_n, top_k, bm = kw["size_n"], kw["top_k"], kw["moe_block_size"]
        wd = deq13 if size_n == 2 * inter else deq2
        numel = kw["size_m"] * top_k  # slots (GEMM1: size_m = tokens, top_k 8; GEMM2: size_m = slots, top_k 1)
        e_rep = expert_ids.repeat_interleave(bm)[: sorted_ids.numel()]
        slot_expert = torch.full((numel + 1,), -1, dtype=torch.int32, device=a.device)
        slot_expert.scatter_(0, sorted_ids.long().clamp(max=numel), e_rep)
        slot_expert = slot_expert[:numel]
        rows = a[torch.arange(numel, device=a.device) // top_k].float()
        out = torch.zeros((numel, size_n), dtype=torch.float32, device=a.device)
        for e in range(32):
            out += (slot_expert == e).float()[:, None] * (rows @ wd[e].T)
        return out.to(a.dtype)

    m.HAVE_SGLANG = True
    m.scalar_types = SimpleNamespace(float4_e2m1f=None)
    m.moe_wna16_marlin_gemm = marlin
    m.marlin_make_workspace = lambda device, max_blocks_per_sm=4: torch.zeros(8, dtype=torch.int32, device=device)
    m.prepare_moe_nvfp4_layer_for_marlin = lambda layer: setattr(layer, "w13_weight", SimpleNamespace(data=layer.w13_weight)) or \
        setattr(layer, "w2_weight", SimpleNamespace(data=layer.w2_weight)) or \
        setattr(layer, "w13_weight_scale", SimpleNamespace(data=layer.w13_weight_scale)) or \
        setattr(layer, "w2_weight_scale", SimpleNamespace(data=layer.w2_weight_scale)) or \
        setattr(layer, "w13_weight_scale_2", SimpleNamespace(data=layer.w13_weight_scale_2)) or \
        setattr(layer, "w2_weight_scale_2", SimpleNamespace(data=layer.w2_weight_scale_2))

    def silu_mul(x, out, expert_ids=None, expert_step=1):
        i = x.shape[1] // 2
        res = (torch.nn.functional.silu(x[:, :i].float()) * x[:, i:].float()).to(out.dtype)
        out.copy_(torch.where((expert_ids >= 0)[:, None], res, out))  # sync-free: capturable

    m.silu_and_mul = silu_mul
    m.moe_align_block_size = lambda ids, bm, ne: m.moe_align_torch(ids, bm, ne)
    m.fused_post_expert_norm_weighted_sum = lambda y, s, norm_weight, eps, weight_first, slot_expert_ids: m.epilogue_torch(
        y, s, slot_expert_ids, norm_weight, eps)
    return h, inter


@needs_gpu
def test_gpu_measure_end_to_end_with_fake_sglang(capsys):
    m = _load()
    g = torch.Generator(device="cuda").manual_seed(3)
    h = inter = 256
    n_e = 256
    w13, w13_sf, gs13 = nvfp4.random_weights(2 * inter, h, generator=g, device="cuda", leading=(n_e,))
    w2, w2_sf, gs2 = nvfp4.random_weights(h, inter, generator=g, device="cuda", leading=(n_e,))
    weights = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(n_e).contiguous(), "w2": w2, "w2_sf": w2_sf,
               "w2_gs": gs2.expand(n_e).contiguous(), "post_norm_w": (0.1 * torch.randn(h, generator=g, device="cuda")).to(torch.float16)}
    _install_fake_sglang(m, weights)
    m._BF = torch.float16
    m.SCREEN_REPS, m.FINAL_REPS = 3, 5
    m.G1_BIG = m.G1_BIG[:4] + ((64, 64, 2, 8, 2),)
    m.G2_BIG = m.G2_BIG[:3]
    m.G1_SMALL, m.G2_SMALL = m.G1_SMALL[:2], m.G2_SMALL[:2]
    m.PHASE_END_S = {"A": 1e9, "B": 1e9, "C": 1e9, "D": 1e9}
    m.BUDGET_S = 1e9
    m.SHAPES = (32, 128)
    m.prepare(weights)
    out = capsys.readouterr().out
    print(out)
    assert len(out.encode()) < 12 * 1024
    b_lines = [l for l in out.splitlines() if l.startswith("B moeg ")]
    assert len(b_lines) == 2
    for l in b_lines:
        assert "marlin g1=" in l and "g1 top5 d" in l and "g2 top5 d" in l and "e2e " in l
        assert "bad=0" in l, l  # every grouped GEMM matches the (fake) Marlin outputs
    assert "FAIL " not in out or "OutOfResources" in out
