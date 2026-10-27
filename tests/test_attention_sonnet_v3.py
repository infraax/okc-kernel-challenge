"""kernels/muse_attention_decode/sonnet_v3.py and sonnet_v3b.py (v2 + the full-layer kernels) against the organisers' reference through
tools/local_check.py: the 16 task tests (CPU interpreter, small), the task + scored shapes on a CUDA graph, and boundary cases of the
full-layer kernels (BN 64 / 128 tile edges, empty splits, page 64, dp heads, short contexts, a block longer than the context).

CPU (Triton interpreter):  TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v3.py
GPU (any CUDA device, incl. the sm_75 dev card; FP8 is converted in software there):
                           tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v3.py
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
from test_attention_sonnet_v2 import EXTRA as EXTRA_V2  # noqa: E402

PROBLEM = "muse_attention_decode"
KERNELS = {name: ROOT / "kernels" / PROBLEM / f"{name}.py" for name in ("sonnet_v3", "sonnet_v3b")}
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="CPU tests need TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")
by_kernel = pytest.mark.parametrize("name", list(KERNELS))

# Full-layer cases around the tile edges (BN 64 and 128, split edges of the plan), empty splits (a short request next to a long one:
# `mixed` spread), page 64, dp heads (16 KV heads, row = 1 KB), a verify block longer than the context, 2 and odd tile counts.
EXTRA = EXTRA_V2 + [
    dict(batch=2, ctx=63, heads="tp8", layer="full", page=1),
    dict(batch=2, ctx=64, heads="tp8", layer="full", page=1),
    dict(batch=2, ctx=65, heads="tp8", layer="full", page=1),
    dict(batch=2, ctx=191, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=2, ctx=192, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=2, ctx=193, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=3, ctx=1000, heads="tp8", layer="full", spread="mixed", page=1),
    dict(batch=3, ctx=1000, heads="tp8", layer="full", q_len=8, spread="mixed", page=1),
    dict(batch=2, ctx=700, heads="dp", layer="full", page=1, dist="soft"),
    dict(batch=2, ctx=700, heads="dp", layer="full", q_len=8, page=64),
    dict(batch=2, ctx=4100, heads="tp8", layer="full", page=64, dist="coherent"),
    dict(batch=1, ctx=2, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=3, ctx=1500, heads="tp8", layer="full", page=1, kv_scale=0.015625),
    dict(batch=2, ctx=3000, heads="tp8", layer="full", dist="edges", page=1),
]


def _loaded(name):
    return lc.Loaded(PROBLEM, KERNELS[name])


def _case(loaded, kwargs, device, seed=0, graph=False):
    ok, msg = lc.run_case(loaded, kwargs, None, None, device, seed, graph)
    assert ok, f"{lc.format_case(kwargs)}: {msg}"


@by_kernel
def test_plan_full_covers_span(name):
    # The full-layer split plan is a pure function of shapes: whole tiles, covers the span, <= 255 splits, >= 4 tiles per split
    # unless one split covers everything, bounded partial buffers.
    mod = _loaded(name).cand
    for kind, rows in (("decode", 16), ("verify", 64)):
        fc = mod.FULL_CFG[kind]
        for batch in (1, 2, 16, 32):
            for pairs in (1, 8):
                for span in (1, 63, 64, 65, 2047, 80001, 86008, 1048584):
                    split_len, nsplit = mod._plan_full(batch, pairs, span, fc["bn"], fc["programs"])
                    assert split_len % fc["bn"] == 0 and split_len * nsplit >= span
                    assert (nsplit - 1) * split_len < span
                    assert nsplit <= mod.MAX_SPLITS
                    assert split_len >= mod.FULL_MIN_TILES * fc["bn"]
                    assert batch * pairs * 2 * (nsplit + 1) * rows * 64 * 4 < 400 * 2**20


@by_kernel
def test_plan_full_programs_in_range(name):
    # The scored batches get about the intended number of programs (the plan is the whole point of item 3).
    mod = _loaded(name).cand
    for kind in ("decode", "verify"):
        fc = mod.FULL_CFG[kind]
        for batch in (16, 32):
            split_len, nsplit = mod._plan_full(batch, 1, 86008, fc["bn"], fc["programs"])
            assert 0.8 * fc["programs"] <= batch * nsplit <= 1.1 * fc["programs"], (kind, batch, nsplit)


@needs_cuda
@by_kernel
def test_cuda_launches_nothing_else(name):
    # kernel_fn = exactly two Triton launches (attention + reducer); no torch op (zeros / fill / cat / copy / index) becomes a graph node.
    from torch.profiler import ProfilerActivity, profile
    loaded = _loaded(name)
    mod = loaded.cand
    cases = [(dict(batch=2, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1), "_attn_kernel"),
             (dict(batch=2, ctx=3000, heads="tp8", layer="full", page=1), "_full_cat_kernel" if mod.FULL_CFG["decode"]["kernel"] == "cat" else "_full_kernel"),
             (dict(batch=2, ctx=3000, heads="tp8", layer="full", q_len=8, page=1), "_full_kernel")]
    for kwargs, first in cases:
        data = loaded.ref.generate_input(**kwargs, seed=0, device="cuda")
        mod.kernel_fn(data)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            mod.kernel_fn(data)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        assert sorted(names) == sorted([first, "_reduce_kernel"]), (kwargs, names)


@by_kernel
def test_kernel_file_rules(name):
    src = KERNELS[name].read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_attention_decode"
    for bad in (r"\.item\(", r"\.cpu\(", r"synchronize", r"subprocess", r"import os", r"import socket", r"requests",
                r"environ", r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 10  # header: speedup + reasoning
    compile(src, str(KERNELS[name]), "exec")


@needs_interpreter
@by_kernel
@pytest.mark.parametrize("kwargs", [k for k in EXTRA if k["ctx"] <= 1500], ids=lc.format_case)
def test_cpu_extra_cases(name, kwargs):
    _case(_loaded(name), kwargs, "cpu")


@needs_interpreter
@by_kernel
def test_cpu_fresh_inputs_same_shapes(name):
    # Same shapes, new data: nothing may be carried over from the previous call.
    loaded = _loaded(name)
    for kwargs in (dict(batch=2, ctx=300, heads="tp8", layer="full", page=1), dict(batch=2, ctx=300, heads="tp8", layer="full", q_len=8, page=1)):
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
