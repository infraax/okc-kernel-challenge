"""kernels/muse_decode_layer/sonnet_v1.py: which implementation runs which shape (dispatch), the Triton router / pooler
against torch, and the layer against the organisers' reference through tools/local_check.py.

Dispatch + file rules (no GPU):   tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v1.py
Triton projections on CPU:        TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v1.py
Whole layer on the dev GPU (sm_75: torch layer, Triton attention with software FP8 on decode):
                                  tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v1.py -k cuda"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_decode_layer"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v1.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="needs TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_sonnet_v1_decode_layer", KERNEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cfg(q_len=1, batch=1, window=2048):
    return {"q_len": q_len, "batch": batch, "window": window, "tokens": q_len * batch}


def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_decode_layer"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket",
                r"requests", r"environ", r"import reference", r"from reference"):
        assert not re.search(bad, src), bad
    body = src.split("def _prepare_sglang")[1].split("def _experts")[0]  # the untimed prepare may sync, as baseline.py does
    assert src.count("synchronize") == body.count("synchronize") == 1
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15  # header: speedup + reasoning
    assert "kernels.muse_" not in src and "from kernels" not in src  # one file, nothing imported across kernels


@pytest.mark.parametrize("tokens,path", [(1, "triton"), (8, "triton"), (9, "triton"), (32, "triton"), (128, "triton"),
                                         (256, "triton"), (257, "split")])
def test_router_path_with_triton(mod, tokens, path):
    assert mod._router_path(tokens, True) == path


@pytest.mark.parametrize("tokens,path", [(1, "fp32"), (8, "fp32"), (9, "split"), (256, "split")])
def test_router_path_without_triton(mod, tokens, path):
    assert mod._router_path(tokens, False) == path


@pytest.mark.parametrize("tokens,path", [(1, "triton"), (8, "triton"), (16, "triton"), (32, "triton"), (33, "cublas"),
                                         (128, "cublas"), (256, "cublas")])
def test_pooler_path(mod, tokens, path):
    assert mod._pooler_path(tokens, True) == path
    assert mod._pooler_path(tokens, False) == "cublas"


def test_attention_path_decode_vs_verify(mod):
    for batch in (1, 16, 32, 33):
        for window in (0, 2048):
            assert mod._attention_path(_cfg(1, batch, window), True) == "triton"      # decode: Triton kernels
            assert mod._attention_path(_cfg(8, batch, window), True) == "reference"   # verify: SGLang FA3
            assert mod._attention_path(_cfg(1, batch, window), False) == "reference"  # CPU: torch
            assert mod._attention_path(_cfg(8, batch, window), False) == "reference"


def test_verify_full_knob(mod, monkeypatch):
    monkeypatch.setattr(mod, "SONNET_VERIFY_FULL_MAX_BATCH", 1)
    assert mod._attention_path(_cfg(8, 1, 0), True) == "triton"
    assert mod._attention_path(_cfg(8, 2, 0), True) == "reference"
    assert mod._attention_path(_cfg(8, 1, 2048), True) == "reference"  # sliding verify never


def test_buckets_have_configs(mod):
    for tokens in range(1, 257):
        assert ("router", mod._bucket(tokens)) in mod._CFG
    for tokens in range(1, 33):
        assert ("pooler", mod._bucket(tokens)) in mod._CFG


# The CPU interpreter mis-computes BF16 tl.dot (garbage), so the interpreter tests run the same kernels with FP16 operands:
# they check the split-K / hi+lo / bucket logic and the output dtypes, not BF16 rounding (the H200 judge does that).
@needs_interpreter
@pytest.mark.parametrize("tokens", [1, 8, 20])
def test_triton_router_matches_fp32(mod, tokens):
    g = torch.Generator().manual_seed(tokens)
    router = torch.randn(256, 8192, generator=g) * 0.02
    hi = router.half()
    lo = (router - hi.float()).half()
    state = {"router_hi": hi, "router_lo": lo}
    x = torch.randn(tokens, 8192, generator=g).half()
    got = mod._router_logits(x, None, state, "triton")
    want = x.float() @ router.T
    assert got.dtype == torch.float32 and got.shape == (tokens, 256)
    assert (got - want).norm() / want.norm() < 1e-5  # hi+lo split: FP32-class (hi alone would be ~3e-4 in FP16)


@needs_interpreter
@pytest.mark.parametrize("tokens", [1, 8, 20])
def test_triton_pooler_matches_reference(mod, tokens):
    g = torch.Generator().manual_seed(100 + tokens)
    pooler = (torch.randn(4096, 8192, generator=g) * 0.02).half()
    x = torch.randn(tokens, 8192, generator=g).half()
    cfg = mod._CFG[("pooler", mod._bucket(tokens))]
    got = mod._run(x, pooler, None, 4096, torch.float16, cfg)
    want = x.float() @ pooler.float().T
    assert got.dtype == torch.float16 and got.shape == (tokens, 4096)
    assert (got.float() - want).norm() / want.norm() < 2e-3


@needs_cuda
def test_cuda_small_layer_cases():
    lines = []
    passed, total = lc.run(PROBLEM, KERNEL, "cuda", "small", out=lines.append)
    assert passed == total == 13, "\n".join(lines)
