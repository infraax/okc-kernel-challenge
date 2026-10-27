"""kernels/muse_moe_w4a16/sonnet_v2.py: the per-M config table (bucket by M only, exactly the judge-sweep winners) and kernel_fn with SGLang stubbed
calling the ops with those arguments. No GPU, no SGLang.

Run: tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_sonnet_v2.py
Whole path on CPU (about 3 min): tools/heavy.sh .venv/bin/python tools/local_check.py muse_moe_w4a16 kernels/muse_moe_w4a16/sonnet_v2.py --device cpu --cases small"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KERNEL = ROOT / "kernels" / "muse_moe_w4a16" / "sonnet_v2.py"

# (M, name) straight from the judge log muse_moe_w4a16.8d094868ac244fcd.log (best config per scored M).
TABLE = {1: "b8.a0.rtt.t1024w8", 8: "b8.a0.rff.t2048w8", 16: "b16.a2.rft.t1024w8",
         32: "b32.a2.rff.sg", 128: "b64.a3.rff.sg", 256: "b64.a2.rff.sg"}
BUCKETS = [(0, 1), (1, 1), (2, 8), (7, 8), (8, 8), (9, 16), (15, 16), (16, 16), (17, 32), (31, 32), (32, 32), (33, 128), (100, 128),
           (128, 128), (129, 256), (256, 256), (257, 256), (1000, 256)]


@pytest.fixture()
def mod():
    spec = importlib.util.spec_from_file_location("_sonnet_v2_moe", KERNEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\("):
        assert not re.search(bad, src), bad
    assert "kernels.muse_" not in src and "from kernels" not in src
    assert "okc-diag HAVE_SGLANG" in src
    assert "1.20x" in src and "1.152x" in src and "8d094868ac244fcd" in src
    compile(src, "sonnet_v2.py", "exec")


def _expected(mod, m, bucket_m):
    c = mod.parse_config(TABLE[bucket_m])
    if c["epilogue"] == "sg" and m < 32:  # director rule: below 32 tokens always our fused epilogue
        c = dict(c, epilogue="t", block_h=1024, warps=8)
    return c


@pytest.mark.parametrize("m,bucket_m", BUCKETS)
def test_config_for_bucket_equals_table(mod, m, bucket_m):
    assert mod.config_for(m) == _expected(mod, m, bucket_m)


def test_table_names_exact(mod):
    assert [n for _, n in mod.CONFIG_TABLE] == list(TABLE.values())
    assert [mx for mx, _ in mod.CONFIG_TABLE] == [1, 8, 16, 32, 128, None]


def test_parse_config_fields(mod):
    c = mod.parse_config("b16.a2.rft.t1024w8")
    assert (c["bm"], c["align"], c["atomic1"], c["fp32_1"], c["atomic2"], c["fp32_2"], c["epilogue"], c["block_h"], c["warps"]) == \
        (16, "a2", False, True, True, False, "t1024w8", 1024, 8)
    c = mod.parse_config("b64.a3.rff.sg")
    assert (c["bm"], c["align"], c["atomic1"], c["fp32_1"], c["atomic2"], c["fp32_2"], c["epilogue"]) == \
        (64, "a3", False, True, False, True, "sg")
    c = mod.parse_config("b8.a0.rtt.t1024w8")
    assert (c["atomic1"], c["fp32_1"], c["atomic2"], c["fp32_2"], c["block_h"], c["warps"]) == (True, False, True, False, 1024, 8)
    with pytest.raises(ValueError):
        mod.parse_config("b9.a0.rff.sg")


def _install_stubs(mod, calls):
    mod.HAVE_SGLANG = True
    mod.scalar_types = type("ST", (), {"float4_e2m1f": "fp4"})
    mod.local_ids_triton = mod.local_ids_torch

    def align(ids, bm, n_exp, **kw):
        calls.append(("align", bm, n_exp, kw))
        return "sorted", "bexp", "npad"

    def marlin(a, c, *args, **kw):
        calls.append(("marlin", kw))
        return c

    def silu(x, out, **kw):
        calls.append(("silu", kw))

    def sg(y, tw, **kw):
        calls.append(("sg", kw))
        return torch.zeros(y.shape[0], y.shape[2], dtype=y.dtype)

    def tri(y, tw, ids, nw, **kw):
        calls.append(("tri", kw))
        return torch.zeros(y.shape[0], y.shape[2], dtype=y.dtype)

    mod.moe_align_block_size = align
    mod.moe_wna16_marlin_gemm = marlin
    mod.silu_and_mul = silu
    mod.fused_post_expert_norm_weighted_sum = sg
    mod.epilogue_triton = tri


def _run(mod, m, h=64, i=32, is_ep=True):
    calls = []
    _install_stubs(mod, calls)
    bank = {"w13": 0, "w13_s": 0, "w13_gs": 0, "w2": 0, "w2_s": 0, "w2_gs": 0, "count": 32, "size_n": i,
            "expert_map": object() if is_ep else None}
    state = {"banks": {(0, 32): bank}, "workspace": "ws", "norm_w": torch.ones(h, dtype=torch.bfloat16)}
    data = {"hidden_states": torch.zeros(m, h, dtype=torch.bfloat16), "topk_ids": torch.randint(0, 256, (m, 8), dtype=torch.int32),
            "topk_weights": torch.rand(m, 8), "config": {"local_start": 0, "local_count": 32}}
    out = mod.kernel_fn(data, state)
    assert out.shape == (m, h)
    return calls


@pytest.mark.parametrize("m,bucket_m", BUCKETS)
def test_kernel_fn_calls_ops_with_table_args(mod, m, bucket_m):
    c = _expected(mod, m, bucket_m)
    calls = _run(mod, m)
    kinds = [x[0] for x in calls]
    assert kinds[:4] == ["align", "marlin", "silu", "marlin"]
    assert kinds[4] == ("sg" if c["epilogue"] == "sg" else "tri") and len(kinds) == 5
    _, bm, n_exp, kw = calls[0]
    assert bm == c["bm"]
    assert n_exp == (256 if c["align"] in ("a0", "a2") else 32)
    assert kw == ({"ignore_invalid_expert": True} if c["align"] in ("a2", "a3") else {})
    g1, g2 = calls[1][1], calls[3][1]
    assert (g1["use_atomic_add"], g1["use_fp32_reduce"]) == (c["atomic1"], c["fp32_1"])
    assert (g2["use_atomic_add"], g2["use_fp32_reduce"]) == (c["atomic2"], c["fp32_2"])
    for g in (g1, g2):
        assert g["moe_block_size"] == c["bm"] and g["is_ep"] is True and g["b_q_type"] == "fp4"
        assert g["mul_topk_weights"] is False and g["is_k_full"] is True and g["is_zp_float"] is False
    assert (g1["size_m"], g1["top_k"]) == (m, 8) and (g2["size_m"], g2["top_k"]) == (m * 8, 1)
    if c["epilogue"] == "sg":
        assert calls[4][1]["weight_first"] is True and calls[4][1]["eps"] == mod.POST_NORM_EPS
    else:
        assert calls[4][1] == {"block_h": c["block_h"], "warps": c["warps"]}


def test_no_expert_map_uses_local_count_for_a0(mod):
    calls = _run(mod, 1, is_ep=False)
    assert calls[0][1:3] == (8, 32) and calls[1][1]["is_ep"] is False


def test_epilogue_triton_takes_block_and_warps(mod):
    import inspect
    sig = inspect.signature(mod.epilogue_triton)
    assert sig.parameters["block_h"].default == 1024 and sig.parameters["warps"].default == 8
