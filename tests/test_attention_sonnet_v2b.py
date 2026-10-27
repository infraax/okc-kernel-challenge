"""kernels/muse_attention_decode/sonnet_v2b.py (v2 with the swap-AB decode loop) through tools/local_check.py: the 16
task tests (CPU interpreter, small) and the task + scored shapes on a CUDA graph, plus v2's boundary cases.

CPU: TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v2b.py
GPU: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v2b.py"""
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
from test_attention_sonnet_v2 import EXTRA  # noqa: E402

PROBLEM = "muse_attention_decode"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v2b.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="CPU tests need TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")


@pytest.fixture(scope="module")
def loaded():
    return lc.Loaded(PROBLEM, KERNEL)


def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_attention_decode"
    for bad in (r"\.item\(", r"\.cpu\(", r"synchronize", r"subprocess", r"import os", r"import socket", r"requests",
                r"environ", r"import reference", r"from reference"):
        assert not re.search(bad, src), bad


@needs_interpreter
@pytest.mark.parametrize("kwargs", EXTRA, ids=lc.format_case)
def test_cpu_extra_cases(loaded, kwargs):
    ok, msg = lc.run_case(loaded, kwargs, None, None, "cpu", 0, False)
    assert ok, f"{lc.format_case(kwargs)}: {msg}"


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
    for seed, graph in ((0, False), (7, True)):
        ok, msg = lc.run_case(loaded, kwargs, None, None, "cuda", seed, graph)
        assert ok, f"{lc.format_case(kwargs)}: {msg}"
