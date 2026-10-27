"""kernels/muse_attention_decode/sonnet_v4.py and sonnet_v4b.py (single-launch sliding attention: the reducer is folded into the split
kernel behind an acq_rel arrival counter) against the organisers' reference through tools/local_check.py: the 16 task tests (CPU
interpreter, small), the task + scored shapes on a CUDA graph, boundary cases of the sliding path (split edges, empty splits, short
contexts, verify blocks longer than the context, page 64, dp heads), and CUDA-graph replays (counters are fresh on every replay).

CPU (Triton interpreter):  TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v4.py
GPU (any CUDA device, incl. the sm_75 dev card; FP8 is converted in software there):
                           tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v4.py
Without TRITON_INTERPRET=1 the CPU tests are skipped; with it the CUDA tests are."""
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from tools import local_check as lc  # noqa: E402
from test_attention_sonnet_v3 import EXTRA as EXTRA_V3  # noqa: E402

PROBLEM = "muse_attention_decode"
KERNELS = {name: ROOT / "kernels" / PROBLEM / f"{name}.py" for name in ("sonnet_v4", "sonnet_v4b")}
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="CPU tests need TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")
by_kernel = pytest.mark.parametrize("name", list(KERNELS))

# Sliding cases around the split edges (v4: 256 / 512 keys, v4b: 128 / 256 keys; window 2048 -> 2047 cached keys), a request shorter than
# one split next to a long one (`mixed`: some splits empty, so participants differ per request), a window that just starts (ctx 2048 / 2049),
# a block longer than the context, page 64, dp heads (16 KV heads).
EXTRA = EXTRA_V3 + [
    dict(batch=3, ctx=127, heads="tp8", layer="sliding", page=1),
    dict(batch=3, ctx=129, heads="tp8", layer="sliding", page=1),
    dict(batch=3, ctx=257, heads="tp8", layer="sliding", page=1),
    dict(batch=3, ctx=513, heads="tp8", layer="sliding", q_len=8, page=1),
    dict(batch=3, ctx=1025, heads="tp8", layer="sliding", q_len=8, page=1),
    dict(batch=3, ctx=2048, heads="tp8", layer="sliding", page=1),
    dict(batch=3, ctx=2049, heads="tp8", layer="sliding", page=1),
    dict(batch=5, ctx=3000, heads="tp8", layer="sliding", spread="mixed", page=1),
    dict(batch=5, ctx=3000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1, dist="soft"),
    dict(batch=2, ctx=1500, heads="dp", layer="sliding", page=1),
    dict(batch=2, ctx=1500, heads="dp", layer="sliding", q_len=8, page=64),
    dict(batch=1, ctx=2, heads="tp8", layer="sliding", page=1),
    dict(batch=3, ctx=3000, heads="tp8", layer="sliding", page=1, kv_scale=0.015625),
    dict(batch=2, ctx=2600, heads="tp8", layer="sliding", dist="edges", page=1),
    dict(batch=16, ctx=5000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
]
SLIDING_EXTRA = [k for k in EXTRA if k.get("layer") == "sliding"]


def _loaded(name):
    return lc.Loaded(PROBLEM, KERNELS[name])


def _case(loaded, kwargs, device, seed=0, graph=False):
    ok, msg = lc.run_case(loaded, kwargs, None, None, device, seed, graph)
    assert ok, f"{lc.format_case(kwargs)}: {msg}"


@by_kernel
def test_sliding_plan_in_range(name):
    # Split size comes from the window (shapes only): whole tiles, covers the window span, bounded partial buffers.
    mod = _loaded(name).cand
    for kind in ("decode", "verify"):
        sc = mod.SLIDE_CFG[kind]
        assert sc["split"] % sc["bn"] == 0
        assert -(-2047 // sc["split"]) <= 16  # the last arriver merges at most 16 + 1 partials at the scored window


@needs_cuda
@by_kernel
def test_cuda_sliding_is_one_launch(name):
    # Sliding: exactly ONE Triton launch (plus the counter memset); no reducer. Full layers: v3's two launches.
    from torch.profiler import ProfilerActivity, profile
    loaded = _loaded(name)
    mod = loaded.cand
    cases = [(dict(batch=2, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1), ["_attn_fused_kernel"]),
             (dict(batch=2, ctx=3000, heads="tp8", layer="sliding", page=1), ["_attn_fused_kernel"]),
             (dict(batch=2, ctx=3000, heads="tp8", layer="full", page=1), ["_full_cat_kernel", "_reduce_kernel"]),
             (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1), ["_full_kernel", "_reduce_kernel"])]
    for kwargs, expect in cases:
        data = loaded.ref.generate_input(**kwargs, seed=0, device="cuda")
        mod.kernel_fn(data)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            mod.kernel_fn(data)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        kernels = [n for n in names if "emset" not in n]          # the zeros() of the counter shows up as a memset or a fill kernel
        fills = [n for n in kernels if "fill" in n.lower() or "zero" in n.lower()]
        kernels = [n for n in kernels if n not in fills]
        assert sorted(kernels) == sorted(expect), (kwargs, names)
        assert len(names) - len(kernels) <= (1 if expect == ["_attn_fused_kernel"] else 0), (kwargs, names)


@by_kernel
def test_kernel_file_rules(name):
    src = KERNELS[name].read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_attention_decode"
    for bad in (r"\.item\(", r"\.cpu\(", r"synchronize", r"subprocess", r"import os", r"import socket", r"requests",
                r"environ", r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 10  # header: speedup + reasoning
    assert 'sem="acq_rel"' in src and "torch.zeros" in src
    compile(src, str(KERNELS[name]), "exec")


@needs_interpreter
@by_kernel
@pytest.mark.parametrize("kwargs", [k for k in EXTRA if k["ctx"] <= 1500 or k.get("layer") == "sliding" and k["ctx"] <= 3000 and k.get("batch", 1) <= 5],
                         ids=lc.format_case)
def test_cpu_extra_cases(name, kwargs):
    _case(_loaded(name), kwargs, "cpu")


@needs_interpreter
@by_kernel
def test_cpu_fresh_inputs_same_shapes(name):
    # Same shapes, new data: nothing may be carried over from the previous call (counters are allocated per call).
    loaded = _loaded(name)
    for kwargs in (dict(batch=2, ctx=300, heads="tp8", layer="sliding", page=1), dict(batch=2, ctx=2500, heads="tp8", layer="sliding", q_len=8, page=1)):
        for seed in (1, 2, 3):
            _case(loaded, kwargs, "cpu", seed=seed)


@needs_interpreter
@by_kernel
def test_cpu_task_tests_small(name):
    lines = []
    passed, total = lc.run(PROBLEM, KERNELS[name], "cpu", "small", out=lines.append)
    assert passed == total == 16, "\n".join(lines)


@needs_cuda
@by_kernel
@pytest.mark.parametrize("which", ["tests", "scored"])
def test_cuda_task_cases_graph(name, which):
    lines = []
    passed, total = lc.run(PROBLEM, KERNELS[name], "cuda", which, graph=True, out=lines.append)
    assert passed == total, "\n".join(lines)


@needs_cuda
@by_kernel
@pytest.mark.parametrize("kwargs", EXTRA, ids=lc.format_case)
def test_cuda_extra_cases(name, kwargs):
    loaded = _loaded(name)
    _case(loaded, kwargs, "cuda")
    _case(loaded, kwargs, "cuda", seed=7, graph=True)


REPLAY_CASES = [
    dict(batch=3, ctx=5000, heads="tp8", layer="sliding", page=1),
    dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"),
    dict(batch=5, ctx=3000, heads="tp8", layer="sliding", spread="mixed", page=1),       # participants differ per request
    dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
    dict(batch=8, ctx=8192, heads="dp", layer="sliding", q_len=8, spread="mixed"),
    dict(batch=3, ctx=4000, heads="tp8", layer="full", page=1),                          # the unchanged v3 path replays too
]


@needs_cuda
@by_kernel
@pytest.mark.parametrize("kwargs", REPLAY_CASES, ids=lc.format_case)
def test_cuda_graph_replay_three_times_identical(name, kwargs):
    # One captured graph, replayed 3 times on the same inputs: counters are re-zeroed by the graph's memset node each time, so every
    # replay merges correctly and the output is identical (the merge order is fixed by split index, not by arrival order) and equals
    # the eager call; the replay output also passes the reference's checks.
    loaded = _loaded(name)
    mod = loaded.cand
    data = loaded.ref.generate_input(**kwargs, seed=3, device="cuda")
    eager = mod.kernel_fn(data).clone()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = mod.kernel_fn(data)
    outs = []
    for _ in range(3):
        out.zero_()                                    # a stale result from the previous replay must not be able to pass
        g.replay()
        torch.cuda.synchronize()
        outs.append(out.clone())
    for o in outs:
        assert torch.isfinite(o.float()).all()
        assert torch.equal(o, outs[0]), "replays differ"
        assert torch.equal(o, eager), "graph replay differs from the eager call"
    raw = loaded.ref.collect(data, outs[-1]) if hasattr(loaded.ref, "collect") else outs[-1]
    ok, msg = loaded.ref.check_implementation(data, raw)
    assert ok, msg
