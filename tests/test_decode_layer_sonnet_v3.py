"""kernels/muse_decode_layer/sonnet_v3.py (S1 MoE knobs, S2 fused routing, S3 PDL) and sonnet_v3_s1.py (S1 only).

File rules + step switches (no GPU):  tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v3.py
Routing kernel on CPU:                TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v3.py
Routing kernel on the dev GPU:        same command without the variable (sm_75: FP16 hi / lo operands stand in for BF16; the
                                      kernel casts its operands to the weights' dtype, so the code path is the Hopper one)
Whole layer on the dev GPU (7 min, 3.3 GB):  LAYER_CHECK=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_decode_layer_sonnet_v3.py -k layer
The routing tests derive their inputs from the organisers' reference (generate_input's residual stream, the reference's own
router rows and bias) and apply TASK.md's routing rules to the kernel's picks."""
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
KERNEL = ROOT / "kernels" / PROBLEM / "sonnet_v3.py"
KERNEL_S1 = ROOT / "kernels" / PROBLEM / "sonnet_v3_s1.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
HAVE_GPU = torch.cuda.is_available() and not INTERPRET
DEVICE = "cuda" if HAVE_GPU else "cpu"
needs_kernel_device = pytest.mark.skipif(not (INTERPRET or HAVE_GPU), reason="needs TRITON_INTERPRET=1 or a CUDA GPU")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mod():
    return _load("_sonnet_v3_decode_layer", KERNEL)


@pytest.fixture(scope="module")
def mod_s1():
    return _load("_sonnet_v3_s1_decode_layer", KERNEL_S1)


@pytest.fixture(scope="module")
def ref():
    return lc.Loaded(PROBLEM, KERNEL).ref


# ------------------------------------------------------------------------------------------------ file rules
@pytest.mark.parametrize("path", [KERNEL, KERNEL_S1])
def test_kernel_file_rules(path):
    src = path.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_decode_layer"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket",
                r"requests", r"environ", r"import reference", r"from reference", r"\.tolist\("):
        assert not re.search(bad, src), bad
    body = src.split("def _prepare_sglang")[1].split("def _marlin_knobs")[0]  # the untimed prepare may sync, as baseline.py does
    assert src.count("synchronize") == body.count("synchronize") == 1
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15  # header: what changes + prediction
    assert "kernels.muse_" not in src and "from kernels" not in src  # one file, nothing imported across kernels
    assert 'print(f"okc-diag HAVE_SGLANG={HAVE_SGLANG}")' in src
    compile(src, path.name, "exec")


_SWITCH = re.compile(r"^S[123]_[A-Z_]+ = (True|False) .*$", re.M)


def _body(path):
    """The file without its leading comment header and the three switch lines."""
    lines = path.read_text().splitlines()
    i = next(n for n, line in enumerate(lines) if not line.startswith("#"))
    return _SWITCH.sub("", "\n".join(lines[i:]))


def test_s1_file_is_v3_with_only_s1_on(mod, mod_s1):
    assert _body(KERNEL) == _body(KERNEL_S1)
    assert (mod_s1.S1_MOE_KNOBS, mod_s1.S2_FUSED_ROUTING, mod_s1.S3_PDL) == (True, False, False)
    assert (mod.S1_MOE_KNOBS, mod.S2_FUSED_ROUTING, mod.S3_PDL) == (True, True, True)


# ------------------------------------------------------------------------------------------------ S1: MoE knobs
@pytest.mark.parametrize("tokens,want", [(1, (True, False, 1024)), (2, (False, True, 2048)), (8, (False, True, 2048)),
                                         (9, (False, True, 1024)), (16, (False, True, 1024)), (32, (False, True, 1024)),
                                         (128, (False, True, 1024)), (256, (False, True, 1024))])
def test_s1_knobs_by_tokens(mod, tokens, want):
    assert mod._marlin_knobs(tokens) == want


@pytest.mark.parametrize("tokens", [1, 2, 8, 9, 16, 32, 128, 256])
def test_s1_off_is_sonnet_v2(mod, monkeypatch, tokens):
    monkeypatch.setattr(mod, "S1_MOE_KNOBS", False)
    assert mod._marlin_knobs(tokens) == (False, True, 1024)


@pytest.mark.parametrize("tokens", [1, 8, 9, 32])
def test_experts_pass_s1_knobs_to_marlin_and_epilogue(mod, monkeypatch, tokens):
    calls, epi = [], []
    monkeypatch.setattr(mod, "scalar_types", type("S", (), {"float4_e2m1f": None}), raising=False)
    monkeypatch.setattr(mod, "moe_align_block_size", lambda ids, bs, ne: (None, None, None), raising=False)
    monkeypatch.setattr(mod, "moe_wna16_marlin_gemm", lambda a, c, *args, **kw: calls.append(kw) or c, raising=False)
    monkeypatch.setattr(mod, "silu_and_mul", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(mod, "fused_post_expert_norm_weighted_sum", lambda *a, **kw: torch.zeros(tokens, 16), raising=False)
    monkeypatch.setattr(mod, "epilogue_triton", lambda y, s, ids, w, eps=0, block_h=0, warps=0: epi.append(block_h) or torch.zeros(tokens, 16))
    state = {"bank": {"size_n": 8, "w13": 0, "w13_s": 0, "w13_gs": 0, "w2": 0, "w2_s": 0, "w2_gs": 0, "workspace": 0},
             "post_expert_w": torch.ones(16)}
    ids = torch.zeros(tokens, 8, dtype=torch.int32)
    local = torch.zeros(tokens, 8, dtype=torch.int32)
    mod._experts(torch.zeros(tokens, 16), torch.ones(tokens, 8), ids, torch.tensor([0], dtype=torch.int32), state, local_ids=local)
    atomic, fp32, block_h = mod._marlin_knobs(tokens)
    assert len(calls) == 2
    assert all(kw["use_atomic_add"] == atomic and kw["use_fp32_reduce"] == fp32 for kw in calls)
    assert epi == ([block_h] if tokens < 32 else [])


# ------------------------------------------------------------------------------------------------ S2: dispatch
@pytest.mark.parametrize("tokens,live", [(1, True), (8, True), (9, True), (32, True), (33, False), (128, False), (256, False)])
def test_s2_range(mod, monkeypatch, tokens, live):
    monkeypatch.setattr(mod, "_triton_ok", lambda device: True)
    state = {"router_hi": "hi", "router_lo": "lo"}
    assert (mod._f2_weights(tokens, torch.device("cuda"), state) == ("hi", "lo")) is live
    assert mod._f2_weights(tokens, torch.device("cpu"), state) is None   # no GPU, no kernel
    monkeypatch.setattr(mod, "S2_FUSED_ROUTING", False)
    assert mod._f2_weights(tokens, torch.device("cuda"), state) is None


def test_s2_fp16_stand_in_only_when_prepared(mod, monkeypatch):
    monkeypatch.setattr(mod, "_triton_ok", lambda device: False)
    assert mod._f2_weights(4, torch.device("cuda"), {"router_hi": 1, "router_lo": 2}) is None
    assert mod._f2_weights(4, torch.device("cuda"), {"router_hi16": 3, "router_lo16": 4}) == (3, 4)


@pytest.mark.parametrize("tokens", range(1, 33))
def test_f2_config_for_every_token_count(mod, tokens):
    bn, bk, split, stages = mod._f2_cfg(tokens)
    assert 256 % bn == 0 and 8192 % (split * bk) == 0 and bn >= 16 and stages >= 1
    assert mod._f2_cfg(33) is None


# ------------------------------------------------------------------------------------------------ S3: PDL
def test_s3_off_by_switch_and_by_hardware(mod, monkeypatch):
    cpu = torch.device("cpu")
    assert mod._pdl_kwargs(cpu) == {}
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
        assert mod._pdl_kwargs(torch.device("cuda")) == {}                       # griddepcontrol is sm_90+
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (9, 0))
    monkeypatch.setattr(mod, "HAVE_PDL_API", True)
    assert mod._pdl_kwargs(torch.device("cuda")) == {"PDL": True, "launch_pdl": True}
    monkeypatch.setattr(mod, "S3_PDL", False)
    assert mod._pdl_kwargs(torch.device("cuda")) == {}
    monkeypatch.setattr(mod, "S3_PDL", True)
    monkeypatch.setattr(mod, "HAVE_PDL_API", False)                              # a Triton without gdc_wait / launch_pdl
    assert mod._pdl_kwargs(torch.device("cuda")) == {}


def test_s3_api_present_in_this_triton(mod):
    assert mod.HAVE_PDL_API  # Triton 3.8 here: tl.extra.cuda.gdc_wait / gdc_launch_dependents and CUDAOptions.launch_pdl


def test_s3_every_kernel_waits_first_and_triggers_late():
    """Source rule: each of our Triton kernels has a PDL constexpr, gdc_wait before any other statement that touches memory."""
    src = KERNEL.read_text()
    for name in ("_gemv_splitk", "_attn_kernel", "_reduce_kernel", "_local_ids_kernel", "_epilogue_kernel", "_router_f2_kernel"):
        body = src.split(f"def {name}(")[1].split("\n@triton.jit")[0].split("\n\n\n")[0]
        assert "PDL: tl.constexpr = False" in body, name
        assert body.index("tl.extra.cuda.gdc_wait()") < min(
            i for i in (body.find("tl.load"), body.find("tl.store"), body.find("tl.atomic")) if i >= 0), name
        assert "gdc_launch_dependents" in body, name


# ------------------------------------------------------------------------------------------------ S2: the routing kernel
def _operands(mod, router, bias):
    st = mod._split_router_weights(router, bias)
    if "router_hi16" in st:                       # dev GPU without BF16 tl.dot (sm_75), or the CPU interpreter
        return st, st["router_hi16"], st["router_lo16"]
    if router.device.type == "cpu":
        hi = router.half()
        return st, hi, (router - hi.float()).half()
    return st, st["router_hi"], st["router_lo"]


def _route(mod, x, hi, lo, bias, start, st):
    s = torch.tensor([start], dtype=torch.int32, device=x.device)
    return mod.fused_router_f2(x.contiguous(), hi, lo, bias.float().contiguous(), s, st["f2_done"])


def _oracle(x, router, bias, ref):
    scores = torch.sigmoid(x.double() @ router.double().T * ref.ROUTER_MULTIPLIER)
    return scores, scores + bias.double()


def _check_rules(ids, weights, local, x, router, bias, start, ref):
    """TASK.md's routing rules, against the reference's own scores (double), for every token."""
    scores, biased = _oracle(x, router, bias, ref)
    t = x.shape[0]
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32 and local.dtype == torch.int32
    assert ids.shape == weights.shape == local.shape == (t, 8)
    srt = ids.sort(-1).values
    assert bool((srt[:, 1:] != srt[:, :-1]).all()), "an expert twice in one row"
    top = biased.topk(9, dim=-1)
    eighth, ninth = top.values[:, 7], top.values[:, 8]
    ids_l = ids.long()
    assert float((eighth[:, None] - biased.gather(1, ids_l)).max()) <= ref.ROUTE_TOL      # each pick within 0.05 of the 8th best
    kept = (top.indices[:, :8, None] == ids_l[:, None, :]).any(-1)
    must = (top.values[:, :8] - ninth[:, None]) > 0.03                                    # clear winners
    assert bool((kept | ~must).all()), "a clear top-8 expert was dropped"
    picked = scores.gather(1, ids_l)
    want_w = picked / (picked.sum(-1, keepdim=True) + ref.ROUTE_EPS)
    assert torch.allclose(weights.double(), want_w, rtol=2e-4, atol=1e-6)
    rel = (ids - start) % 256
    assert torch.equal(local, torch.where(rel < 32, rel, torch.full_like(rel, -1)).to(torch.int32))
    # descending biased score (torch.topk's order), up to float noise
    got_b = biased.gather(1, ids_l)
    assert bool((got_b[:, :-1] >= got_b[:, 1:] - 1e-6).all())
    return biased


def _reference_inputs(ref, batch, q_len, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    r = ref._router(gen, device, None)
    data = ref.generate_input(batch=batch, ctx=64, layer="sliding", seed=seed, q_len=q_len, dist="soft", device=device)
    stream = data["stream"].float()
    pre_ffn = 0.1 * torch.randn(8192, generator=gen, device=device)
    h = stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + 1e-5) * (1.0 + pre_ffn)
    return r["router"], r["router_bias"], h


@needs_kernel_device
@pytest.mark.parametrize("batch,q_len,start", [(1, 1, 0), (4, 1, 32), (8, 1, 224), (1, 8, 100), (2, 8, 64), (3, 8, 0), (4, 8, 192)])
def test_routing_kernel_on_reference_inputs(mod, ref, batch, q_len, start):
    """generate_input's residual stream through the reference's router (skewed: a few experts take most tokens). T <= 8 takes the
    FP32 hidden (split hi + lo in the kernel), above that the BF16-rounded one, as sonnet_v2 feeds the router."""
    router, bias, h = _reference_inputs(ref, batch, q_len, 11 + batch * 7 + q_len, DEVICE)
    st, hi, lo = _operands(mod, router, bias)
    x = h if batch * q_len <= 8 else h.to(torch.bfloat16)
    ids, weights, local = _route(mod, x, hi, lo, bias, start, st)
    _check_rules(ids, weights, local, x, router, bias, start, ref)
    assert st["f2_done"].tolist() == [0, 0]       # counters are back at zero for the next call
    # the kernel's logits are FP32-class (hi + lo, FP32 accumulate): picks equal the reference's unless two scores nearly tie
    _, biased = _oracle(x, router, bias, ref)
    ref_ids = biased.topk(8, dim=-1).indices
    top = biased.topk(9, dim=-1).values
    gap = (top[:, 7] - top[:, 8]).abs()
    same = (ids.long() == ref_ids).all(-1)
    assert bool((same | (gap < 1e-4)).all())


@needs_kernel_device
def test_routing_kernel_every_token_count_matches_reference(mod, ref):
    router, bias, h = _reference_inputs(ref, 4, 8, 3, DEVICE)            # 32 tokens of a verify block
    st, hi, lo = _operands(mod, router, bias)
    for t in (1, 2, 5, 8, 9, 15, 16, 17, 24, 31, 32):
        x = h[:t] if t <= 8 else h[:t].to(torch.bfloat16)
        ids, weights, local = _route(mod, x, hi, lo, bias, 96, st)
        _check_rules(ids, weights, local, x, router, bias, 96, ref)


@needs_kernel_device
def test_exact_ties_take_the_lower_expert_first(mod, ref):
    """x = 0 makes every logit 0: the ranking is the bias alone. Nine experts share the top bias (the lower eight win, in index
    order, as a stable descending sort / torch.topk's convention); one sits 1e-5 below and must lose."""
    router = torch.randn(256, 8192, device=DEVICE) * 0.045
    bias = torch.zeros(256, device=DEVICE)
    bias[60:69] = 1.0
    bias[80] = 1.0 - 1e-5
    st, hi, lo = _operands(mod, router, bias)
    for t, dtype in ((4, torch.float32), (8, torch.float32), (16, torch.bfloat16), (32, torch.bfloat16)):
        x = torch.zeros(t, 8192, dtype=dtype, device=DEVICE)
        ids, weights, local = _route(mod, x, hi, lo, bias, 64, st)
        assert torch.equal(ids, torch.arange(60, 68, device=DEVICE, dtype=torch.int32).expand(t, 8))
        assert torch.equal(local[:, :4], torch.full((t, 4), -1, device=DEVICE, dtype=torch.int32))   # 60..63 are not in [64, 96)
        assert torch.equal(local[:, 4:], torch.arange(4, device=DEVICE, dtype=torch.int32).expand(t, 4))
        assert torch.allclose(weights.sum(-1), torch.ones(t, device=DEVICE), atol=1e-5)
        assert torch.allclose(weights, weights[:, :1].expand(t, 8))                                    # equal logits: equal weights


@pytest.mark.skipif(not HAVE_GPU, reason="the CPU interpreter's matmul does not give bit-identical sums for identical rows")
def test_identical_router_rows_tie_exactly_and_resolve_by_index(mod, ref):
    """Rows 4, 11, 19 are one row with equal bias: bit-identical logits, an exact tie inside a real GEMV. Index order wins."""
    g = torch.Generator(device=DEVICE).manual_seed(5)
    router = torch.randn(256, 8192, device=DEVICE, generator=g) * 1e-3
    router[11] = router[4]
    router[19] = router[4]
    bias = torch.randn(256, device=DEVICE, generator=g) * 0.05
    bias[[4, 11, 19]] = 2.0
    st, hi, lo = _operands(mod, router, bias)
    for t, dtype in ((1, torch.float32), (8, torch.float32), (13, torch.bfloat16), (32, torch.bfloat16)):
        x = torch.randn(t, 8192, device=DEVICE, generator=g)
        x = x if dtype == torch.float32 else x.to(dtype)
        ids, weights, local = _route(mod, x, hi, lo, bias, 0, st)
        assert torch.equal(ids[:, :3], torch.tensor([4, 11, 19], dtype=torch.int32, device=DEVICE).expand(t, 3))
        _check_rules(ids, weights, local, x, router, bias, 0, ref)


@needs_kernel_device
def test_near_ties_at_the_eighth_slot_stay_within_the_judge_tolerance(mod, ref):
    """Bias is set so experts 8th/9th/10th differ by 0, 1e-7 and 1e-6 in biased score: whichever the kernel picks, TASK.md's
    rules hold (within 0.05 of the 8th best; every clear winner kept)."""
    router, bias, h = _reference_inputs(ref, 4, 8, 21, DEVICE)
    bias = bias.clone()
    st, hi, lo = _operands(mod, router, bias)
    t = 8
    x = h[:t]
    scores, biased = _oracle(x, router, bias, ref)
    # force near-ties for every token at once: shift the bias of experts 200..202 so their biased scores equal the 8th best of token 0
    target = float(biased[0].topk(8).values[-1])
    for e, d in ((200, 0.0), (201, 1e-7), (202, -1e-6)):
        bias[e] = target - float(scores[0, e]) + d
    ids, weights, local = _route(mod, x, hi, lo, bias, 32, st)
    biased_after = _check_rules(ids, weights, local, x, router, bias, 32, ref)
    assert biased_after.shape == (t, 256)


@needs_kernel_device
def test_back_to_back_calls_and_tokens_past_one_block(mod, ref):
    """The arrival counters reset themselves: the same call twice gives bit-identical picks; 32 tokens span two 16-token blocks."""
    router, bias, h = _reference_inputs(ref, 4, 8, 9, DEVICE)
    st, hi, lo = _operands(mod, router, bias)
    x = h.to(torch.bfloat16)
    a = _route(mod, x, hi, lo, bias, 0, st)
    b = _route(mod, x[:20], hi, lo, bias, 0, st)
    c = _route(mod, x, hi, lo, bias, 0, st)
    for u, v in zip(a, c):
        assert torch.equal(u, v)
    for u, v in zip(a, b):
        assert torch.equal(u[:20], v)
    assert st["f2_done"].tolist() == [0, 0]


# ------------------------------------------------------------------------------------------------ the whole layer
@pytest.mark.skipif(not HAVE_GPU or os.environ.get("LAYER_CHECK") != "1", reason="LAYER_CHECK=1 on a CUDA GPU (7 min, 3.3 GB)")
@pytest.mark.parametrize("path", [KERNEL, KERNEL_S1])
def test_cuda_small_layer_cases(path):
    lines = []
    passed, total = lc.run(PROBLEM, path, "cuda", "small", graph=True, out=lines.append)
    assert passed == total == 13, "\n".join(lines)
