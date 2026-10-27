"""tools/ptx_audit.py (C29): the PTX / resource audit of Triton kernels, AOT-compiled for sm_90a.

Fast (a few seconds of Triton compiles); needs any CUDA device for Triton's driver, nothing is launched.
Run through tools/heavy.sh: .venv/bin/python -m pytest -q tests/test_ptx_audit.py"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import ptx_audit as pa  # noqa: E402

PROJ = ROOT / "kernels/muse_bf16_proj/director_v1b.py"
ATTN = ROOT / "kernels/muse_attention_decode/sonnet_v3.py"

try:
    import torch
    HAVE_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    HAVE_CUDA = False
needs_cuda = pytest.mark.skipif(not HAVE_CUDA, reason="Triton's driver needs a CUDA device (nothing is launched)")


def test_discover_finds_jit_kernels_and_launch_sites():
    kernels, launches = pa.discover(PROJ)
    assert set(kernels) == {"_gemv_splitk"} and "_gemv_splitk" in launches
    kernels, launches = pa.discover(ATTN)
    for k in ("_attn_kernel", "_full_kernel", "_full_cat_kernel", "_reduce_kernel"):
        assert k in launches, k
    for helper in ("_step", "_gather", "_softmax_step"):   # @triton.jit helpers are kernels' callees, never launched
        assert helper in kernels and helper not in launches


def test_count_ptx_ignores_comments_and_matches_variants():
    ptx = """
    // cp.async.ca.shared.global in a comment
    ld.global.nc.v4.b32 {%r1,%r2,%r3,%r4}, [%rd1];
    ld.global.b32 %r5, [%rd2];
    cp.async.cg.shared.global [%r6], [%rd3], 16;
    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r7], [%rd4, {%r8, %r9}], [%r10];
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r7], [%rd4], 16, [%r10];
    cp.async.wait_group 2;
    mbarrier.try_wait.parity.shared::cta.b64 p, [%r10], %r11;
    wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%f1}, %rd1, %rd2, 1, 1, 1, 0, 0;
    wgmma.fence.sync.aligned;
    red.global.add.f32 [%rd5], %f2;
    atom.global.add.u32 %r12, [%rd6], 1;
    bar.sync 0;
    """
    c = pa.count_ptx(ptx)
    assert (c["ld.global"], c["ld.global.nc"], c["cp.async"], c["cp.async.bulk"], c["cp.async.bulk.tensor"]) == (2, 1, 1, 1, 1)
    assert (c["mbarrier"], c["wgmma"], c["atom"], c["bar.sync"], c["wait_groups"]) == (1, 1, 2, 1, "2x1")


def test_report_line_format():
    counts = {f: 1 for f in pa.FIELDS}
    counts["regs"] = None
    text = pa.line("_k", "w4s3", counts)
    assert re.fullmatch(r"#C _k w4s3 regs=n/a spills=1 smem=1 ld\.global=1 ld\.global\.nc=1 cp\.async=1 cp\.async\.bulk=1 "
                        r"cp\.async\.bulk\.tensor=1 mbarrier=1 ldmatrix=1 mma\.sync=1 wgmma=1 cvt=1 ex2=1 bar\.sync=1 atom=1 st\.global=1", text)


def test_ptxas_log_parse():
    log = "ptxas info    : Used 128 registers, used 0 barriers, 36 bytes spill stores, 4 bytes spill loads, 110592 bytes smem"
    assert pa.parse_ptxas_log(log) == (128, 40, 110592)


@needs_cuda
def test_gemv_splitk_without_alignment_info_has_no_cp_async_the_old_probe_fact():
    """What tools/ptx_probe_gemv.py compiled (no divisibility attributes): plain ld.global, 0 cp.async, 255 regs / spills at the router m1 config."""
    res = pa.audit(PROJ, case_list=[{"m": 1, "layer": "router"}], align="none", nvrtc=False)
    (row,) = [r for r in res["rows"] if r["kernel"] == "_gemv_splitk"]
    c = row["counts"]
    assert c["cp.async"] == 0 and c["ld.global"] > 0
    assert c["stages"] == 4 and c["mma.sync"] > 0


@needs_cuda
def test_gemv_splitk_with_the_real_alignment_is_cp_async_pipelined():
    """The same config with the specialisation Triton derives from the real (16-byte aligned) arguments, i.e. what the judge compiles."""
    res = pa.audit(PROJ, case_list=[{"m": 1, "layer": "router"}], align="real", nvrtc=False)
    (row,) = [r for r in res["rows"] if r["kernel"] == "_gemv_splitk"]
    c = row["counts"]
    assert c["cp.async"] > 0 and c["ld.global"] == 0 and c["spills"] == 0
    assert not any(e.startswith("compile") for e in res["errors"])


@needs_cuda
def test_attention_full_kernel_issues_cp_async():
    """C29's key question: verify-full _full_kernel has a cp.async pipeline (wait_group > 0 keeps copies in flight)."""
    res = pa.audit(ATTN, flt=r"^batch 1, .*layer full.*q_len 8", nvrtc=False)
    rows = {r["kernel"]: r for r in res["rows"]}
    full = rows["_full_kernel"]["counts"]
    assert full["cp.async"] > 0 and full["stages"] > 1
    assert re.search(r"(?:^|,)[1-9]\d*x", full["wait_groups"])
    assert full["spills"] == 0
