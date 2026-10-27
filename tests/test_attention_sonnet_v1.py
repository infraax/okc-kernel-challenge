"""kernels/muse_attention_decode/sonnet_v1.py against the organisers' reference (problems/muse_attention_decode)
through tools/local_check.py.

CPU (Triton interpreter):  TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v1.py
GPU (any CUDA device, incl. the sm_75 dev card; FP8 is converted in software there):
                           tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v1.py
Without TRITON_INTERPRET=1 the CPU tests are skipped; with it the CUDA tests are."""
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_attention_decode"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v1.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="CPU tests need TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")

# Small CPU cases: sliding + full, decode + verify, page 1 + 64, kv_scale 1 and 1/64, the window's edges,
# a context shorter than the verify block, 16 KV heads (dp), mixed lengths.
EXTRA = [
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
