"""kernels/muse_decode_layer/sonnet_v5.py: sonnet_v4 plus C19, attention sonnet_v3's full-layer decode kernel (_full_cat_kernel + row-major reducer) for q_len 1.

No GPU:   file rules, v5 differs from v4 only by the C19 additions, the copied kernels are attention v3's verbatim, the gate and the split plan.
Dev GPU (RTX 2060, sm_75; FP8 is decoded in software there): v5's `_attention_decode` against v4's on reference.generate_input data (full q_len 1 within 2e-3, everything
              else bitwise), CUDA-graph capture of the new path. The whole layer against the organisers' reference is `tools/local_check.py muse_decode_layer ... --cases small --graph`.
Run:      tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v5.py"""
import difflib
import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_decode_layer"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v5.py"
V4 = ROOT / "kernels" / PROBLEM / "sonnet_v4.py"
ATT3 = ROOT / "kernels" / "muse_attention_decode" / "sonnet_v3.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v5_decode_layer", KERNEL)


@pytest.fixture(scope="module")
def v4():
    return _load("_sonnet_v4_for_v5", V4)


@pytest.fixture(scope="module")
def att3():
    return _load("_attention_v3_for_layer_v5", ATT3)


@pytest.fixture(scope="module")
def ref():
    return lc.Loaded(PROBLEM, KERNEL).ref


# ------------------------------------------------------------------------------------------------ file rules
def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_decode_layer"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket",
                r"requests", r"environ", r"import reference", r"from reference", r"\.tolist\(", r"urllib"):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert 'print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")' in src
    header = "\n".join(line for line in src.splitlines() if line.startswith("#"))
    for needle in ("#1 (decode 1, full) 137 -> 132.6", "#3 (decode 16, full) 340 -> 280", "#5 (decode 32, full) 540 -> 410", "1.227 -> 1.266"):
        assert needle in header, needle
    compile(src, "sonnet_v5.py", "exec")


def test_v5_is_v4_plus_the_c19_additions():
    """Every line of v4 is still in v5 (in order) except the two reducer lines that gain the ROWMAJOR layout and the _attention_decode lines that moved below the new branch."""
    a, b = V4.read_text().splitlines(), KERNEL.read_text().splitlines()
    removed = [d[1:] for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] == "-" and d[:3] != "---"]
    allowed = {
        "    M: tl.constexpr, BS: tl.constexpr, PDL: tl.constexpr = False,",
        "    base = (b * HKV + h) * (NSPLIT + 1) * M + row",
        "        pidx = base + tl.where(si == n_active, NSPLIT, si) * M",
        "    occ = 4 if rows <= 16 else 3",
        "    split_len, nsplit = _plan(batch, hkv, span, rows, BLOCK_N, occ)",
        "",
        "    part_o = torch.empty((batch, hkv, nsplit + 1, rows, 64), dtype=torch.float32, device=dev)",
        "    part_m = torch.empty((batch, hkv, nsplit + 1, rows), dtype=torch.float32, device=dev)",
        "    part_l = torch.empty_like(part_m)",
        '    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=dev)',
        '    k_cache = data["k_cache"].view(torch.uint8)',
        '    v_cache = data["v_cache"].view(torch.uint8)',
    }
    extra = [line for line in removed if line not in allowed]
    assert not extra, extra
    src = KERNEL.read_text()
    assert "S1_MOE_KNOBS = True" in src and "S2_FUSED_ROUTING = False" in src and "S3_PDL = False" in src and "S5_FULL_CAT = True" in src


@pytest.mark.parametrize("name", ["_pages", "_gather_cat", "_step_cat", "_full_cat_kernel", "_plan_full"])
def test_copied_functions_are_attention_v3s(name, mod, att3):
    got, want = getattr(mod, name), getattr(att3, name)
    gs = inspect.getsource(got.fn if hasattr(got, "fn") else got)
    ws = inspect.getsource(want.fn if hasattr(want, "fn") else want)
    assert gs == ws


def test_reducer_row_major_branch_is_attention_v3s(mod, att3):
    """The ROWMAJOR layout lines (base and pidx) are the same text as attention v3's."""
    for line in ("        base = ((b * HKV + h) * M + row) * (NSPLIT + 1)", "            pidx = base + tl.where(si == n_active, NSPLIT, si)"):
        assert line in KERNEL.read_text() and line in ATT3.read_text()


# ------------------------------------------------------------------------------------------------ gate / plan (no GPU)
def _cfg(**kw):
    c = dict(num_q_heads=16, num_kv_heads=2, head_dim=64, q_len=1, window=0, rope_theta=0.0, batch=1)
    c.update(kw)
    return c


@pytest.mark.parametrize("kw,ok", [({}, True), ({"q_len": 8}, False), ({"window": 2048, "rope_theta": 500000.0}, False), ({"window": 2048}, False),
                                   ({"rope_theta": 5e5}, False), ({"num_kv_heads": 1, "num_q_heads": 8}, False), ({"num_kv_heads": 4, "num_q_heads": 32}, True),
                                   ({"num_kv_heads": 2, "num_q_heads": 8}, False), ({"head_dim": 128}, False)])
def test_gate_is_config_only(mod, kw, ok):
    assert mod._full_cat_ok(_cfg(**kw)) is ok


def test_gate_switch(mod, monkeypatch):
    monkeypatch.setattr(mod, "S5_FULL_CAT", False)
    assert mod._full_cat_ok(_cfg()) is False


@pytest.mark.parametrize("batch,span", [(1, 80001), (16, 80001), (32, 80001), (1, 3001), (33, 20001), (3, 6001), (1, 1048577)])
def test_plan_full_matches_attention_v3_and_limits(mod, att3, batch, span):
    cfg = mod.FULL_CAT_CFG
    got = mod._plan_full(batch, 1, span, cfg["bn"], cfg["programs"])
    assert got == att3._plan_full(batch, 1, span, att3.FULL_CFG["decode"]["bn"], att3.FULL_CFG["decode"]["programs"])
    split_len, nsplit = got
    assert 1 <= nsplit <= mod.MAX_SPLITS and split_len * nsplit >= span and split_len % cfg["bn"] == 0
    assert cfg == att3.FULL_CFG["decode"] | {"kernel": "cat"} or {k: cfg[k] for k in cfg} == {k: att3.FULL_CFG["decode"][k] for k in cfg}


# ------------------------------------------------------------------------------------------------ GPU: v5 attention against v4's
def _data(ref, batch, ctx, layer, q_len, seed, **kw):
    d = ref.generate_input(batch=batch, ctx=ctx, layer=layer, seed=seed, q_len=q_len, device="cuda", **kw)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    tokens = batch * q_len
    d["qkv"] = (torch.randn((tokens, 2304), generator=g, device="cuda") * 2.0).to(torch.bfloat16)
    return d


def _clone(d):
    out = dict(d)
    out["k_cache"], out["v_cache"] = d["k_cache"].clone(), d["v_cache"].clone()
    return out


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


FULL_DECODE = [dict(batch=1, ctx=3000), dict(batch=1, ctx=80000, dist="soft", page=1), dict(batch=4, ctx=2700, spread="mixed", page=1),
               dict(batch=16, ctx=20000, spread="mixed", dist="soft", page=1), dict(batch=3, ctx=6000, dist="outliers"),
               dict(batch=33, ctx=20000, spread="mixed", page=1), dict(batch=2, ctx=130, page=1)]


@needs_cuda
@pytest.mark.parametrize("kw", FULL_DECODE, ids=lambda k: "-".join(f"{a}{b}" for a, b in k.items()))
def test_full_decode_matches_v4(mod, v4, ref, kw):
    kw = dict(kw)
    d = _data(ref, kw.pop("batch"), kw.pop("ctx"), "full", 1, 7, **kw)
    assert mod._full_cat_ok(d["config"])
    a, b = _clone(d), _clone(d)
    got = mod._attention_decode(a)
    want = v4._attention_decode(b)
    torch.cuda.synchronize()
    assert got.shape == want.shape and got.dtype == torch.bfloat16 and not torch.isnan(got.float()).any()
    assert _rel(got, want) < 2e-3, _rel(got, want)
    # the block's own key / value are written to the cache identically
    assert torch.equal(a["k_cache"].view(torch.uint8), b["k_cache"].view(torch.uint8))
    assert torch.equal(a["v_cache"].view(torch.uint8), b["v_cache"].view(torch.uint8))
    again = mod._attention_decode(_clone(d))
    assert torch.equal(again.view(torch.int16), got.view(torch.int16)), "not deterministic"


OTHERS = [dict(batch=1, ctx=3000, layer="full", q_len=8), dict(batch=3, ctx=2700, layer="full", q_len=8, dist="outliers"),
          dict(batch=4, ctx=2400, layer="sliding", page=1, spread="mixed"), dict(batch=2, ctx=2300, layer="sliding", q_len=8),
          dict(batch=1, ctx=10, layer="sliding")]


@needs_cuda
@pytest.mark.parametrize("kw", OTHERS, ids=lambda k: "-".join(f"{a}{b}" for a, b in k.items()))
def test_verify_and_sliding_are_v4_bitwise(mod, v4, ref, kw):
    kw = dict(kw)
    d = _data(ref, kw.pop("batch"), kw.pop("ctx"), kw.pop("layer"), kw.pop("q_len", 1), 11, **kw)
    assert not mod._full_cat_ok(d["config"])
    got = mod._attention_decode(_clone(d))
    want = v4._attention_decode(_clone(d))
    torch.cuda.synchronize()
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@needs_cuda
def test_full_decode_graph_capture_and_replay(mod, ref):
    d = _data(ref, 4, 2700, "full", 1, 5, spread="mixed", page=1)
    eager = mod._attention_decode(_clone(d))
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        mod._attention_decode(d)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod._attention_decode(d)
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
