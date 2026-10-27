"""kernels/muse_moe_w4a16/sonnet_v1.py: the fused Triton epilogue and the local-id kernel against the eager math, the dispatch
and file rules, and the whole path against the organisers' reference through tools/local_check.py.

File rules (no GPU):          tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v1.py
Triton kernels on CPU:        TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v1.py
(The interpreter garbles BF16, so the kernel tests use FP32 and FP16 stand-ins for the expert outputs.)
Whole path on CPU (about 3 min): tools/heavy.sh .venv/bin/python tools/local_check.py muse_moe_w4a16 kernels/muse_moe_w4a16/sonnet_v1.py --device cpu --cases small"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROBLEM = "muse_moe_w4a16"
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v1.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="needs TRITON_INTERPRET=1")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_sonnet_v1_moe", KERNEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\("):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15  # header: speedup + reasoning
    assert "kernels.muse_" not in src and "from kernels" not in src
    compile(src, "sonnet_v1.py", "exec")


def _eager_reference(y, scores, local_ids, norm_w, eps):
    """The textbook per-slot math in FP32 (reference.post_expert_norm with weight (1 + w) already folded into norm_w)."""
    y = y.float()
    out = torch.zeros(y.shape[0], y.shape[2], dtype=torch.float32)
    for t in range(y.shape[0]):
        for k in range(y.shape[1]):
            if local_ids[t, k] < 0:
                continue
            v = y[t, k] * norm_w.float()
            out[t] += scores[t, k] * v * torch.rsqrt(v.pow(2).mean() + eps)
    return out


def _case(dtype, m, h, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(m, k, h, generator=g).to(dtype)
    # poison the rows of non-local slots: they must never be read (Marlin leaves them undefined)
    ids = torch.randint(-1, 32, (m, k), generator=g, dtype=torch.int32)
    if m > 1:
        ids[0, :] = -1  # a token with no local slot -> zero row
        ids[1, :] = torch.arange(k, dtype=torch.int32)  # all slots local
    ids[-1, :4] = torch.arange(4, dtype=torch.int32)  # at least some live slots
    y[ids < 0] = float("nan")
    scores = torch.rand(m, k, generator=g)
    scores = scores / scores.sum(-1, keepdim=True)
    norm_w = (1.0 + 0.1 * torch.randn(h, generator=g)).to(dtype)
    return y, scores, ids, norm_w


@needs_interpreter
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.float16, 2e-3)])
@pytest.mark.parametrize("m,h", [(1, 4096), (8, 4096), (31, 4096), (3, 1536), (2, 100)])
def test_epilogue_triton_matches_eager(mod, dtype, tol, m, h):
    y, scores, ids, norm_w = _case(dtype, m, h)
    ref = _eager_reference(y.nan_to_num(), scores, ids, norm_w, mod.POST_NORM_EPS)
    out = mod.epilogue_triton(y, scores, ids, norm_w).float()
    assert out.shape == (m, h)
    assert not torch.isnan(out).any()
    if m > 1:
        assert (out[0] == 0).all()
    err = (out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)
    assert err[(1 if m > 1 else 0):].max().item() < tol, err


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m", [1, 8, 31])
def test_epilogue_torch_twin_matches_eager(mod, dtype, m):
    y, scores, ids, norm_w = _case(dtype, m, 4096, seed=1)
    ref = _eager_reference(y.nan_to_num(), scores, ids, norm_w, mod.POST_NORM_EPS)
    out = mod.epilogue_torch(y.nan_to_num(), scores, ids, norm_w).float()
    first = 1 if m > 1 else 0
    if m > 1:
        assert (out[0] == 0).all()
    assert ((out - ref).norm(dim=-1)[first:] / ref.norm(dim=-1)[first:]).max().item() < (2e-5 if dtype == torch.float32 else 2e-3)


@needs_interpreter
@pytest.mark.parametrize("start,count", [(0, 32), (32, 32), (224, 32), (0, 256)])
def test_local_ids_kernel_matches_expert_map(mod, start, count):
    ids = torch.randint(0, 256, (13, 8), dtype=torch.int32)
    expert_map = torch.full((256,), -1, dtype=torch.int32)
    expert_map[start:start + count] = torch.arange(count, dtype=torch.int32)
    want = expert_map[ids.long()]
    assert torch.equal(mod.local_ids_triton(ids, start, count), want)
    assert torch.equal(mod.local_ids_torch(ids, start, count), want)


def test_dispatch_bounds(mod):
    assert mod.EAGER_BELOW_TOKENS == 32  # SGLang's _MIN_TOKENS_FOR_FUSED
    assert mod.BLOCK_SIZE_M_CANDIDATES == (8, 16, 32, 48, 64)
