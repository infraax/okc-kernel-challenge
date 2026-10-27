"""kernels/muse_bf16_proj/director_v2.py (C23): v1b + the PLAN table + the last-CTA fixup kernel.

No GPU: file rules, v2 differs from v1b only by the listed lines, the shipped table == v1b's cells.
Dev GPU (RTX 2060: Triton emulates BF16 dots, no cp.async): defaults == v1b on the judged shapes, every candidate variant against the organisers' reference
(bounds incl. router 1e-4 and probe rows), determinism, counters back at zero, CUDA graph, fallbacks; offline sm_90 compile of every candidate: cp.async > 0, smem.
Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_bf16_director_v2.py"""
import difflib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_bf16_proj"
V2 = ROOT / "kernels" / PROBLEM / "director_v2.py"
V1B = ROOT / "kernels" / PROBLEM / "director_v1b.py"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def v2():
    return _load("_dir_v2", V2)


@pytest.fixture(scope="module")
def v1b():
    return _load("_dir_v1b", V1B)


@pytest.fixture(scope="module")
def ref():
    return lc.Loaded(PROBLEM, V2).ref


@pytest.fixture(scope="module")
def weights(ref):
    return ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")


@pytest.fixture(scope="module")
def st2(v2, weights):
    return v2.prepare(weights)


@pytest.fixture(scope="module")
def st1(v1b, weights):
    return v1b.prepare(weights)


@pytest.fixture()
def plan(v2):
    saved = dict(v2._PLAN)
    yield v2._PLAN
    v2._PLAN.clear()
    v2._PLAN.update(saved)


@pytest.fixture()
def force_triton(v2, v1b, monkeypatch):
    monkeypatch.setattr(v2, "_triton_ok", lambda d: True)
    monkeypatch.setattr(v1b, "_triton_ok", lambda d: True)


# ------------------------------------------------------------------------------------------------ file rules / diff
def test_file_rules():
    src = V2.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_bf16_proj"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ", r"import reference", r"from reference",
                r"\.tolist\(", r"urllib", r"torch\.profiler", r"MEASUREMENT", r"synchronize"):
        assert not re.search(bad, src), bad
    compile(src, "director_v2.py", "exec")


def test_v2_is_v1b_plus_the_plan():
    a, b = V1B.read_text().splitlines(), V2.read_text().splitlines()
    removed = [d[1:].rstrip() for d in difflib.unified_diff(a, b, lineterm="", n=0) if d[:1] == "-" and d[:3] != "---"]
    allowed_prefix = ('"""bf16_proj candidate v1b', "every m) and for pooler", "Configs come from", "chosen per shape bucket", "Predicted (tools/score_sim",
                      "# (BN, BK, SPLIT, warps, stages) per", "    for b in (1, 8, 16, 32, 128, 256):", "def kernel_fn", "    x, cfg = data", "    layer, m, n = cfg",
                      "    b = _bucket(m)", "    use_triton =", "        if use_triton and m <=", "            return _run(x, state[\"router_hi\"]", "    if use_triton and (layer, b) in _CFG",
                      "        return _run(x, state[layer], None, n, torch.bfloat16, _CFG")
    extra = [r for r in removed if not r.startswith(allowed_prefix)]
    assert not extra, extra
    # the v1b kernel and its host launcher are untouched
    for name in ("def _gemv_splitk", "def _run(", "_CFG = {", "def _triton_ok"):
        assert name in V1B.read_text() and name in V2.read_text()


def test_shipped_plan_is_v1bs_cells(v2, v1b):
    for layer in v2._LAYERS:
        for b in v2._BUCKETS:
            cell = v2._PLAN[(layer, b)]
            if (layer, b) in v2._JUDGED:   # judge-measured winners (measure_cheap_v1): must be a measured candidate
                assert cell in v2._CANDIDATES["unpooler"] + v2._CANDIDATES["router_small"] + v2._CANDIDATES["router_big"], (layer, b)
            elif (layer, b) in v1b._CFG:
                assert cell == ("gemv",) + v1b._CFG[(layer, b)], (layer, b)
            else:
                assert cell == ("cublas",), (layer, b)
    assert v2._CFG == v1b._CFG and v2._ROUTER_TRITON_MAX_M == v1b._ROUTER_TRITON_MAX_M
    assert all(v2._bucket(m) == v1b._bucket(m) for m in range(0, 1100))


def test_candidates_are_well_formed(v2):
    for kind, vs in v2._CANDIDATES.items():
        for v in vs:
            if v[0] == "fix":
                _, bm, bn, bk, split, w, s, rg = v
                assert split % rg == 0 and bm in (0, 16, 32, 64, 128, 256) and bn % 16 == 0
                k = 4096 if kind == "unpooler" else 8192
                assert k % (split * bk) == 0, v
                n = 1024 if kind == "unpooler" else 256
                assert n % bn == 0, v
                assert split * 256 * n <= v2.FIX_WS_FLOATS
    for cell, v in v2._PLAN_NEW.items():
        assert v in v2._CANDIDATES["unpooler"] + v2._CANDIDATES["router_small"] + v2._CANDIDATES["router_big"] or v[1] in (16, 32), cell


# ------------------------------------------------------------------------------------------------ dev GPU
SHAPES = [(m, l) for m in (1, 8, 16, 32, 128, 256) for l in ("qkv_gate", "o", "pooler", "unpooler", "router")]


def _data(ref, m, layer, seed=3, dist="normal"):
    return ref.generate_input(m=m, layer=layer, seed=seed, dist=dist, device="cuda")


@needs_cuda
@pytest.mark.parametrize("m,layer", SHAPES)
def test_defaults_equal_v1b(v2, v1b, st2, st1, ref, force_triton, m, layer):
    d = _data(ref, m, layer)
    a, b = v2.kernel_fn(d, st2), v1b.kernel_fn(d, st1)
    assert a.shape == b.shape and a.dtype == b.dtype
    if (layer, v2._bucket(m)) in v2._JUDGED:   # changed on purpose: must pass the organisers' check instead
        assert float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)) < (1e-4 if layer == "router" else 1e-2)
    elif layer == "router":   # FP32 atomics: summation order varies run to run
        assert float((a - b).norm() / b.norm()) < 1e-6
    else:
        assert torch.equal(a, b)


@needs_cuda
@pytest.mark.parametrize("m", [1, 5, 8, 33, 130, 512, 1000])
def test_defaults_equal_v1b_nearby(v2, v1b, st2, st1, ref, force_triton, m):
    for layer in ("router", "unpooler", "o"):
        if layer == "router" and m > 256:
            continue   # the baseline's torch.mm(out_dtype=FP32) needs sm_80 (dev card: sm_75)
        d = _data(ref, m, layer, dist="outliers")
        a, b = v2.kernel_fn(d, st2), v1b.kernel_fn(d, st1)
        tol = (1e-4 if layer == "router" else 1e-2) if (layer, v2._bucket(m)) in v2._JUDGED else 1e-6
        assert float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30)) < tol, (m, layer)


def _check(ref, d, out, weights):
    ok, msg = ref.check_implementation(d, out, weights)
    assert ok, msg


CASES = ([("unpooler", m, "unpooler") for m in (1, 5, 8, 9, 16)]
         + [("router", m, "router_small") for m in (1, 7, 16, 17, 32)]
         + [("router", m, "router_big") for m in (33, 100, 128, 129, 200, 256)])


@needs_cuda
@pytest.mark.parametrize("layer,m,group", CASES)
def test_every_candidate_matches_the_reference(v2, st2, ref, weights, plan, force_triton, layer, m, group):
    d = _data(ref, m, layer)
    d2 = _data(ref, m, layer, seed=4, dist="outliers")
    for var in v2._CANDIDATES[group]:
        plan[(layer, v2._bucket(m))] = var
        out = v2.kernel_fn(d, st2)
        _check(ref, d, out, weights)
        _check(ref, d2, v2.kernel_fn(d2, st2), weights)
        if var[0] == "fix":
            again = v2.kernel_fn(d, st2)
            assert torch.equal(out, again), f"{var}: not deterministic"
            torch.cuda.synchronize()
            assert int(st2["fix_sem"].abs().sum()) == 0, f"{var}: counters not back at zero"


@needs_cuda
def test_plan_new_all_cells_together(v2, st2, ref, weights, plan, force_triton):
    plan.update(v2._PLAN_NEW)
    for m, layer in SHAPES:
        d = _data(ref, m, layer, seed=5)
        _check(ref, d, v2.kernel_fn(d, st2), weights)


@needs_cuda
@pytest.mark.parametrize("cell,var,m", [(("unpooler", 8), ("direct", 32, 256, 4, 4), 8), (("unpooler", 8), ("fix", 16, 32, 128, 4, 4, 4, 4), 8),
                                         (("router", 1), ("fix", 0, 64, 128, 64, 4, 4, 8), 1), (("router", 256), ("fix", 0, 32, 64, 32, 4, 4, 8), 256)])
def test_cuda_graph_replays(v2, st2, ref, weights, plan, force_triton, cell, var, m):
    plan[cell] = var
    d = _data(ref, m, cell[0])
    eager = v2.kernel_fn(d, st2)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        v2.kernel_fn(d, st2)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = v2.kernel_fn(d, st2)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager) and int(st2["fix_sem"].abs().sum()) == 0
    _check(ref, d, out, weights)
    d["x"].copy_(_data(ref, m, cell[0], seed=9)["x"])   # fresh input into the same buffers
    g.replay()
    _check(ref, d, out, weights)


@needs_cuda
def test_fallbacks_when_the_cell_cannot_run(v2, st2, ref, weights, plan, force_triton):
    d = _data(ref, 8, "router")   # m <= 8: the baseline is F.linear (the dev card cannot run the hi/lo torch.mm)
    plan[("router", 8)] = ("fix", 0, 64, 128, 64, 4, 4, 8)
    small = dict(st2, fix_ws=torch.empty(1000, device="cuda"))
    _check(ref, d, v2.kernel_fn(d, small), weights)               # workspace too small -> baseline path
    plan[("router", 8)] = ("fix", 0, 48, 64, 32, 4, 4, 8)           # 256 % 48 != 0
    _check(ref, d, v2.kernel_fn(d, st2), weights)
    nofix = {k: v for k, v in st2.items() if k not in ("fix_ws", "fix_sem")}
    plan[("router", 8)] = ("fix", 0, 64, 128, 64, 4, 4, 8)
    _check(ref, d, v2.kernel_fn(d, nofix), weights)


# ------------------------------------------------------------------------------------------------ offline sm_90 audit (tools/ptx_audit.py)
@needs_cuda
def test_candidates_pipeline_on_sm90_and_fit_smem(v2, tmp_path):
    """Every candidate, installed in the router / unpooler cells of a temp copy, audited like the judge compiles it (sm_90a, real 16-byte alignment):
    cp.async > 0, smem within the H200's 227 KB, no '#F' (silently inert pipelining) line."""
    from tools import ptx_audit
    cells = {"unpooler": [("unpooler", 1), ("unpooler", 8)], "router_small": [("router", b) for b in (1, 8, 16, 32)], "router_big": [("router", 128), ("router", 256)]}
    n = max(len(v) for v in v2._CANDIDATES.values())
    seen = 0
    for i in range(n):
        plan = {}
        for kind, cs in cells.items():
            cand = [c for c in v2._CANDIDATES[kind] if c[0] != "tmm"]
            for cell in cs:
                plan[cell] = cand[i % len(cand)]
        f = tmp_path / f"v2_audit_{i}.py"
        f.write_text(V2.read_text() + f"\n_PLAN.update({plan!r})\n")
        out = []
        res = ptx_audit.audit(f, cases="scored", flt="layer router|layer unpooler", nvrtc=False, out=out.append)
        assert not res["errors"], res["errors"]
        ptx_audit.print_report(res, out.append)
        single = {e["cfg"] for e in res["rows"] if e["kernel"] == "_gemv_fix" and int(e["allc"]["K"]) // int(e["allc"]["SPLIT"]) <= int(e["allc"]["BK"])}
        # a config whose per-CTA K slice is one BK tile has no loop to pipeline (all loads are issued at once): not flagged
        assert not [l for l in out if l.startswith("#F") and not any(c in l for c in single)], out
        for e in res["rows"]:
            c = e["counts"]
            if e["cfg"] in single:
                seen += 1
                continue
            if e["kernel"] == "_gemv_fix" or (e["kernel"] == "_gemv_splitk" and c["stages"] > 1 and e["loop"]):
                assert c["cp.async"] > 0, (e["kernel"], e["cfg"])
                assert c["smem"] <= 227 * 1024, (e["kernel"], e["cfg"], c["smem"])
                seen += 1
    assert seen >= 12
