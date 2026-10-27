"""kernels/muse_decode_layer/sonnet_v2.py: which implementation runs which shape (dispatch), the router's FP32-input hi+lo
split, the local-id and epilogue kernels against torch, and the layer against the organisers' reference through
tools/local_check.py.

Dispatch + file rules (no GPU):   tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v2.py
Triton kernels on CPU:            TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v2.py
Whole layer on the dev GPU (sm_75: torch layer, Triton attention with software FP8, decode and verify):
                                  tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v2.py -k cuda
(The interpreter garbles BF16 tl.dot, so its tests use FP32 / FP16 stand-ins for the BF16 operands.)"""
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
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v2.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="needs TRITON_INTERPRET=1")
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_sonnet_v2_decode_layer", KERNEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cfg(q_len=1, batch=1, window=2048, hq=16, hkv=2, head_dim=64):
    return {"q_len": q_len, "batch": batch, "window": window, "tokens": q_len * batch, "num_q_heads": hq,
            "num_kv_heads": hkv, "head_dim": head_dim}


def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_decode_layer"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket",
                r"requests", r"environ", r"import reference", r"from reference", r"\.tolist\("):
        assert not re.search(bad, src), bad
    body = src.split("def _prepare_sglang")[1].split("def _experts")[0]  # the untimed prepare may sync, as baseline.py does
    assert src.count("synchronize") == body.count("synchronize") == 1
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15  # header: speedup + reasoning
    assert "kernels.muse_" not in src and "from kernels" not in src  # one file, nothing imported across kernels
    assert 'print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")' in src
    compile(src, "sonnet_v2.py", "exec")


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


@pytest.mark.parametrize("tokens,path", [(1, "triton"), (8, "triton"), (16, "triton"), (31, "triton"), (32, "sglang"),
                                         (128, "sglang"), (256, "sglang")])
def test_epilogue_path(mod, tokens, path):
    assert mod._epilogue_path(tokens) == path


def test_attention_path_decode_and_verify_both_triton(mod):
    for batch in (1, 3, 16, 32, 33):
        for window in (0, 2048):
            for q_len in (1, 8):
                assert mod._attention_path(_cfg(q_len, batch, window), True) == "triton"
                assert mod._attention_path(_cfg(q_len, batch, window), False) == "reference"  # CPU: torch


def test_attention_path_unsupported_geometry_falls_back_to_fa3(mod):
    assert mod._attention_path(_cfg(head_dim=128), True) == "reference"
    assert mod._attention_path(_cfg(hq=15, hkv=2), True) == "reference"
    assert mod._attention_path(_cfg(q_len=200), True) == "reference"   # more rows / keys than the kernel's tile
    assert mod._attention_path(_cfg(q_len=16, batch=2), True) == "triton"


def test_buckets_have_configs(mod):
    for tokens in range(1, 257):
        assert ("router", mod._bucket(tokens)) in mod._CFG
    for tokens in range(1, 33):
        assert ("pooler", mod._bucket(tokens)) in mod._CFG


# ------------------------------------------------------------------------------------------------ router precision
# At <= 8 tokens the baseline's router input is an FP32 hidden (its JIT boundary kernel stores the unrounded pre-norm result,
# not the BF16 output upcast). The Triton router splits it into BF16 hi + lo in the kernel; here with FP16 stand-ins.
def _router_case(tokens, seed):
    g = torch.Generator().manual_seed(seed)
    router = torch.randn(256, 8192, generator=g) * 0.02
    hi = router.half()
    lo = (router - hi.float()).half()
    x32 = torch.randn(tokens, 8192, generator=g)           # FP32 hidden with bits below the BF16 / FP16 rounding
    return router, {"router_hi": hi, "router_lo": lo}, x32


@needs_interpreter
@pytest.mark.parametrize("tokens", [1, 8])
def test_router_fp32_input_is_split_hi_lo(mod, tokens):
    router, state, x32 = _router_case(tokens, tokens)
    want = x32 @ router.T
    got = mod._router_logits(x32.half(), x32, state, "triton")
    assert got.dtype == torch.float32 and got.shape == (tokens, 256)
    assert (got - want).norm() / want.norm() < 1e-5                          # FP32-class
    hi_only = mod._router_logits(x32.half(), None, state, "triton")           # what v1 did: the rounded input alone
    assert (hi_only - want).norm() / want.norm() > 20 * ((got - want).norm() / want.norm())


@needs_interpreter
@pytest.mark.parametrize("tokens", [1, 8, 20])
def test_router_without_fp32_input_matches_rounded_input(mod, tokens):
    router, state, x32 = _router_case(tokens, 10 + tokens)
    x = x32.half()
    got = mod._router_logits(x, None, state, "triton")
    want = x.float() @ router.T
    assert (got - want).norm() / want.norm() < 1e-5


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


# ------------------------------------------------------------------------------------------------ experts
@needs_interpreter
@pytest.mark.parametrize("start", [0, 32, 224, 100])
def test_local_ids_kernel_matches_baseline_modulo_and_expert_map(mod, start):
    ids = torch.randint(0, 256, (13, 8), dtype=torch.int32)
    local_start = torch.tensor([start], dtype=torch.int32)
    expert_map = torch.full((256,), -1, dtype=torch.int32)
    expert_map[:32] = torch.arange(32, dtype=torch.int32)
    want = expert_map[((ids - local_start) % 256).long()]       # sonnet_v1: modulo, then expert_map[...]
    got = mod.local_ids_triton(ids, local_start)
    assert got.dtype == torch.int32 and torch.equal(got, want)


def _epilogue_case(dtype, m, h=4096, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(m, k, h, generator=g).to(dtype)
    ids = torch.randint(-1, 32, (m, k), generator=g, dtype=torch.int32)
    if m > 1:
        ids[0, :] = -1
    ids[-1, :4] = torch.arange(4, dtype=torch.int32)
    y[ids < 0] = float("nan")                                   # non-local slots must never be read
    scores = torch.rand(m, k, generator=g)
    norm_w = (1.0 + 0.1 * torch.randn(h, generator=g)).to(dtype)
    return y, scores / scores.sum(-1, keepdim=True), ids, norm_w


@needs_interpreter
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 2e-5), (torch.float16, 2e-3)])
@pytest.mark.parametrize("m", [1, 8, 31])
def test_epilogue_matches_eager_branch(mod, dtype, tol, m):
    y, scores, ids, norm_w = _epilogue_case(dtype, m)
    live = (ids >= 0).unsqueeze(-1)
    f = torch.where(live, y, torch.zeros((), dtype=dtype)).float() * norm_w.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + mod.POST_NORM_EPS)
    ref = (f * scores.unsqueeze(-1)).sum(1)                     # SGLang's eager branch below 32 tokens
    out = mod.epilogue_triton(y, scores, ids, norm_w).float()
    assert out.shape == (m, 4096) and not torch.isnan(out).any()
    if m > 1:
        assert (out[0] == 0).all()
    first = 1 if m > 1 else 0
    assert ((out - ref).norm(dim=-1)[first:] / ref.norm(dim=-1)[first:]).max().item() < tol


@needs_interpreter
@pytest.mark.parametrize("tokens,fused_by_sglang", [(1, False), (8, False), (31, False), (32, True), (256, True)])
def test_experts_dispatch_epilogue(mod, monkeypatch, tokens, fused_by_sglang):
    calls = []
    monkeypatch.setattr(mod, "scalar_types", type("S", (), {"float4_e2m1f": None}), raising=False)
    monkeypatch.setattr(mod, "moe_align_block_size", lambda ids, bs, ne: (None, None, None), raising=False)
    monkeypatch.setattr(mod, "moe_wna16_marlin_gemm", lambda a, c, *args, **kw: c, raising=False)
    monkeypatch.setattr(mod, "silu_and_mul", lambda *a, **kw: None, raising=False)

    def sglang_fused(y, w, norm_weight, eps, weight_first, slot_expert_ids):
        calls.append(tokens)
        return torch.zeros(y.shape[0], y.shape[2], dtype=y.dtype)

    monkeypatch.setattr(mod, "fused_post_expert_norm_weighted_sum", sglang_fused, raising=False)
    state = {"bank": {"size_n": 8, "w13": 0, "w13_s": 0, "w13_gs": 0, "w2": 0, "w2_s": 0, "w2_gs": 0, "workspace": 0},
             "post_expert_w": torch.ones(16)}
    x = torch.zeros(tokens, 16)
    ids = torch.zeros(tokens, 8, dtype=torch.int32)
    out = mod._experts(x, torch.ones(tokens, 8), ids, torch.tensor([0], dtype=torch.int32), state)
    assert out.shape == (tokens, 16)
    assert bool(calls) == fused_by_sglang


@needs_cuda
def test_cuda_small_layer_cases():
    lines = []
    passed, total = lc.run(PROBLEM, KERNEL, "cuda", "small", out=lines.append)
    assert passed == total == 13, "\n".join(lines)
