"""kernels/muse_attention_decode/sonnet_v5.py (C27) = sonnet_v3 + BN 128 / 3 stages for verify-full at batch >= 16.

No GPU: the v3 -> v5 diff is only the header comment and the config selection; the selection per batch / q_len (launch recorder on CPU tensors).
Dev GPU (any CUDA card, FP8 in software on the sm_75 card): the reference check of both BN paths (batch >= 16 and < 16, tile / split edges, dp heads, page 64,
kv_scale, a CUDA graph), v5 == v3 bitwise wherever the config is the same. Last: sm_90a PTX audit of the BN128 instantiation (regs <= 255, 0 spills, smem 98304).
Run: set -o pipefail; tools/heavy.sh .venv/bin/python -m pytest -q tests/test_attention_sonnet_v5.py"""
import difflib
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import local_check as lc  # noqa: E402

PROBLEM = "muse_attention_decode"
V3 = ROOT / "kernels" / PROBLEM / "sonnet_v3.py"
V5 = ROOT / "kernels" / PROBLEM / "sonnet_v5.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
needs_cuda = pytest.mark.skipif(INTERPRET or not torch.cuda.is_available(), reason="needs a CUDA GPU (no interpreter)")

BIG = dict(batch=16, ctx=700, heads="tp8", layer="full", q_len=8, page=1, dist="soft")
CASES = [   # BN128 path (batch >= 16, q_len 8) and BN64 path (batch < 16, decode, sliding)
    BIG,
    dict(batch=16, ctx=127, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=16, ctx=129, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=17, ctx=1500, heads="tp8", layer="full", q_len=8, spread="mixed", page=1, dist="coherent"),
    dict(batch=16, ctx=900, heads="tp8", layer="full", q_len=8, page=64, kv_scale=0.015625),
    dict(batch=16, ctx=500, heads="dp", layer="full", q_len=8, page=1),
    dict(batch=16, ctx=3000, heads="tp8", layer="full", q_len=8, page=1, dist="edges"),
    dict(batch=15, ctx=1500, heads="tp8", layer="full", q_len=8, page=1),
    dict(batch=2, ctx=2000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"),
    dict(batch=16, ctx=1500, heads="tp8", layer="full", page=1),                       # decode: cat kernel, untouched
    dict(batch=16, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"),   # sliding: untouched
]


def _mod(path=V5):
    return lc.Loaded(PROBLEM, path).cand


def test_diff_v3_v5_is_header_and_config_selection_only():
    a, b = V3.read_text().splitlines(), V5.read_text().splitlines()
    removed, added = [], []
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        (added if line[0] == "+" else removed).append(line[1:])
    assert removed == [], removed
    assert b[0] == a[0] == "# okc: problem=muse_attention_decode"
    header = [line for line in added if line.startswith("# ") or line.startswith("#   ")]
    code = [line for line in added if line not in header]
    assert len(header) == 6 and all(line.startswith("# sonnet_v5") or line.startswith("#   ") for line in header)
    assert code == ['FULL_BIG = dict(kernel="sep", bn=128, stages=3, warps=4, programs=256)   # verify-full at batch >= FULL_BIG_MIN_BATCH (C27: BN128 twin of C26\')',
                    "FULL_BIG_MIN_BATCH = 16",
                    "        if length > 1 and batch >= FULL_BIG_MIN_BATCH:",
                    "            fc = dict(FULL_BIG)"]
    assert b[1:7] == header                                           # the header block sits right after line 1; nothing else in the board code moved


class _Rec:
    """Stands in for a Triton kernel: records the launch instead of running it."""

    def __init__(self, log, name):
        self.log, self.name = log, name

    def __getitem__(self, grid):
        return lambda *a, **kw: self.log.append((self.name, tuple(grid), kw))


@pytest.mark.parametrize("batch,q_len,layer,bn,stages", [
    (1, 8, "full", 64, 4), (2, 8, "full", 64, 4), (15, 8, "full", 64, 4), (16, 8, "full", 128, 3), (32, 8, "full", 128, 3), (64, 8, "full", 128, 3),
    (16, 1, "full", None, 3), (32, 1, "full", None, 3), (16, 8, "sliding", None, 3)])
def test_config_selection(batch, q_len, layer, bn, stages):
    mod = _mod()
    ref = lc.Loaded(PROBLEM, V5).ref
    data = ref.generate_input(batch=batch, ctx=300, heads="tp8", layer=layer, seed=1, q_len=q_len, page=1, device="cpu")
    log = []
    for n in ("_full_kernel", "_full_cat_kernel", "_attn_kernel", "_reduce_kernel"):
        setattr(mod, n, _Rec(log, n))
    mod.kernel_fn(data)
    main = [r for r in log if r[0] != "_reduce_kernel"]
    assert len(main) == 1
    name, grid, kw = main[0]
    if bn is None:
        if layer == "full":
            assert name == "_full_cat_kernel" and kw["BN"] == 64 and kw["num_stages"] == 3     # decode-full: v3's
        else:
            assert name == "_attn_kernel" and kw["BN"] == 128
    else:
        assert name == "_full_kernel" and kw["BN"] == bn and kw["num_stages"] == stages and kw["num_warps"] == 4


@needs_cuda
@pytest.mark.parametrize("case", CASES, ids=lambda c: lc.format_case(c))
def test_reference_dev_shapes(case):
    ok, msg = lc.run_case(lc.Loaded(PROBLEM, V5), case, None, None, "cuda", 0, False)
    assert ok, f"{lc.format_case(case)}: {msg}"


@needs_cuda
def test_reference_in_a_cuda_graph():
    ok, msg = lc.run_case(lc.Loaded(PROBLEM, V5), BIG, None, None, "cuda", 3, True)
    assert ok, msg


@needs_cuda
def test_v5_equals_v3_where_the_config_is_unchanged_and_close_where_not():
    m3, m5 = _mod(V3), _mod(V5)
    ref = lc.Loaded(PROBLEM, V5).ref
    for case, same in ((dict(batch=2, ctx=2000, heads="tp8", layer="full", q_len=8, page=1), True),
                       (dict(batch=16, ctx=1500, heads="tp8", layer="full", page=1), True),
                       (dict(batch=16, ctx=3000, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft"), True),
                       (BIG, False)):
        d = ref.generate_input(**case, seed=4, device="cuda")
        o3, o5 = m3.kernel_fn(d).clone(), m5.kernel_fn(d).clone()
        if same:
            assert torch.equal(o3, o5), lc.format_case(case)
        else:
            err = float((o3.float() - o5.float()).norm() / o3.float().norm())
            assert err < 1e-3, err


@needs_cuda
def test_sm90_ptx_audit_of_the_bn128_instantiation():    # LAST: ptx_audit.Capture patches get_device_capability, which Triton's driver caches for the process
    from tools import ptx_audit as pa
    cases = [dict(batch=b, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, spread="agentic", dist="coherent") for b in (16, 32)]
    res = pa.audit(V5, case_list=cases, nvrtc=False)
    assert not res["errors"], res["errors"]
    big = [r for r in res["rows"] if r["kernel"] == "_full_kernel"]
    assert len(big) == 1 and big[0]["cvals"]["BN"] == "128"
    c = big[0]["counts"]
    assert (c["regs"], c["spills"], c["smem"], c["warps"], c["stages"]) == (255, 0, 98304, 4, 3), c
    assert c["regs"] <= 255 and c["cp.async"] == 48 and c["wgmma"] == 26
    small = pa.audit(V5, case_list=[dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent")], nvrtc=False)
    k = [r for r in small["rows"] if r["kernel"] == "_full_kernel"][0]
    assert k["cvals"]["BN"] == "64" and k["counts"]["regs"] == 185          # batch 1 keeps v3's code
