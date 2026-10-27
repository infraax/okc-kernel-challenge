"""kernels/muse_attention_decode/sonnet_v2.py against the organisers' reference (problems/muse_attention_decode)
through tools/local_check.py.

CPU (Triton interpreter):  TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v2.py
GPU (any CUDA device, incl. the sm_75 dev card; FP8 is converted in software there):
                           tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v2.py
Without TRITON_INTERPRET=1 the CPU tests are skipped; with it the CUDA tests are."""
import os
import re
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_attention_decode"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v2.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="CPU tests need TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")

# Small CPU cases: sliding + full, decode + verify, page 1 + 64, kv_scale 1 and 1/64, the window's edges,
# a context shorter than the verify block, 16 KV heads (dp), mixed lengths.
EXTRA = [
    # v2-specific: tile / split boundaries (BLOCK_N 128), the window-edge tile with a verify block, the own-keys program
    dict(batch=2, ctx=127, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=2, ctx=128, heads="tp8", layer="full", page=1),
    dict(batch=2, ctx=129, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=3, ctx=2049, heads="tp8", layer="sliding", q_len=8, page=1),
    dict(batch=3, ctx=2055, heads="tp8", layer="sliding", q_len=8, page=64),
    dict(batch=2, ctx=2, heads="tp8", layer="sliding", q_len=8, page=1),
    dict(batch=2, ctx=1, heads="dp", layer="sliding", q_len=8, page=1),
    dict(batch=5, ctx=1500, heads="tp8", layer="full", q_len=8, spread="mixed", page=1, dist="coherent"),
    dict(batch=2, ctx=1, heads="tp8", layer="sliding", q_len=8),
    dict(batch=2, ctx=2048, heads="tp8", layer="sliding", q_len=8, page=1),
    dict(batch=2, ctx=2049, heads="tp8", layer="sliding", page=64),
    dict(batch=3, ctx=130, heads="tp8", layer="full", q_len=8, page=64, kv_scale=0.015625),
    dict(batch=1, ctx=5, heads="dp", layer="full"),
    dict(batch=3, ctx=700, heads="tp8", layer="full", spread="mixed", page=1, dist="soft"),
    dict(batch=2, ctx=2600, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1),
]


def _case(L, kwargs, device, seed=0, graph=False):
    ok, msg = lc.run_case(L, kwargs, None, None, device, seed, graph)
    assert ok, f"{lc.format_case(kwargs)}: {msg}"


@pytest.fixture(scope="module")
def loaded():
    return lc.Loaded(PROBLEM, KERNEL)


_sincos_fn = None


@triton.jit
def probe(x_ptr, s_ptr, c_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    s, c = _sincos_fn(tl.load(x_ptr + i))
    tl.store(s_ptr + i, s)
    tl.store(c_ptr + i, c)


def test_plan_covers_span():
    # The split plan is a pure function of shapes: whole tiles, covers the span, bounded partial buffers.
    for batch in (1, 2, 16, 32):
        for span in (1, 127, 2047, 80001, 86008, 1048584):
            for rows in (16, 32, 64):
                split_len, nsplit = loaded_module()._plan(batch, 2, span, rows, 128, 3)
                assert split_len % 128 == 0 and split_len * nsplit >= span
                assert (nsplit - 1) * split_len < span
                assert batch * 2 * (nsplit + 1) * rows * 64 * 4 < 400 * 2**20


def loaded_module():
    return lc.Loaded(PROBLEM, KERNEL).cand


def test_sincos_matches_torch():
    # The in-kernel sin / cos (no libdevice slow path) over the RoPE angle range of a 1M-token context.
    global _sincos_fn
    _sincos_fn = loaded_module()._sincos
    device = "cpu" if INTERPRET else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1")

    gen = torch.Generator().manual_seed(0)
    top = 2e4 if INTERPRET else 2e6        # the interpreter's FMA is not fused: exact reduction only up to ~1e4 rad
    x = ((torch.rand(2048, generator=gen) * 2 - 1) * top).to(device)
    s, c = torch.empty_like(x), torch.empty_like(x)
    probe[(1,)](x, s, c, N=2048)
    assert (s.cpu().double() - torch.sin(x.cpu().double())).abs().max() < 2e-6
    assert (c.cpu().double() - torch.cos(x.cpu().double())).abs().max() < 2e-6


@needs_cuda
def test_cuda_two_kernels_nothing_else():
    # kernel_fn = exactly two Triton launches; no torch op (zeros / fill / cat / copy / index) becomes a graph node.
    from torch.profiler import ProfilerActivity, profile
    mod = loaded_module()
    for kwargs in (dict(batch=2, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1),
                   dict(batch=2, ctx=3000, heads="tp8", layer="full", page=1)):
        data = lc.Loaded(PROBLEM, KERNEL).ref.generate_input(**kwargs, seed=0, device="cuda")
        mod.kernel_fn(data)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            mod.kernel_fn(data)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        assert sorted(names) == ["_attn_kernel", "_reduce_kernel"], names


def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_attention_decode"
    for bad in (r"\.item\(", r"\.cpu\(", r"synchronize", r"subprocess", r"import os", r"import socket", r"requests",
                r"environ", r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 10  # header: speedup + reasoning


@needs_interpreter
@pytest.mark.parametrize("kwargs", EXTRA, ids=lc.format_case)
def test_cpu_extra_cases(loaded, kwargs):
    _case(loaded, kwargs, "cpu")


@needs_interpreter
def test_cpu_fresh_inputs_same_shapes(loaded):
    # Same shapes, new data: nothing may be carried over from the previous call.
    kwargs = dict(batch=2, ctx=300, heads="tp8", layer="sliding", q_len=8, page=1)
    for seed in (1, 2, 3):
        _case(loaded, kwargs, "cpu", seed=seed)


@needs_interpreter
def test_cpu_task_tests_small():
    lines = []
    passed, total = lc.run(PROBLEM, KERNEL, "cpu", "small", out=lines.append)
    assert passed == total == 16, "\n".join(lines)


@needs_cuda
@pytest.mark.parametrize("which", ["tests", "scored"])
def test_cuda_task_cases_graph(which):
    lines = []
    passed, total = lc.run(PROBLEM, KERNEL, "cuda", which, graph=True, out=lines.append)
    assert passed == total, "\n".join(lines)


@needs_cuda
@pytest.mark.parametrize("kwargs", EXTRA, ids=lc.format_case)
def test_cuda_extra_cases(loaded, kwargs):
    _case(loaded, kwargs, "cuda")
    _case(loaded, kwargs, "cuda", seed=7, graph=True)
