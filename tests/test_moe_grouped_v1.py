"""kernels/muse_moe_w4a16/grouped_v1.py: the grouped W4A16 GEMM (Triton), its weight repack, the dequant exactness, the tile list, the
dispatch and file rules.

File rules (no GPU):     tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_grouped_v1.py
Triton on CPU:           TRITON_INTERPRET=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_grouped_v1.py
Triton on the RTX 2060:  GROUPED_GPU=1 tools/heavy.sh .venv/bin/python -m pytest -q tests/test_moe_grouped_v1.py -k gpu
(sm_75 has no BF16 tl.dot and no FP8: the kernels run with FP16 / FP32 operands and the integer E4M3 decode there.)"""
import importlib.util
import itertools
import math
import os
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
PROBLEM = "muse_moe_w4a16"
sys.path.insert(0, str(ROOT / "problems" / PROBLEM))
KERNEL = ROOT / "kernels" / PROBLEM / "grouped_v1.py"
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
GPU = os.environ.get("GROUPED_GPU") == "1" and torch.cuda.is_available()
needs_interpreter = pytest.mark.skipif(not INTERPRET, reason="needs TRITON_INTERPRET=1")
needs_gpu = pytest.mark.skipif(not GPU, reason="needs GROUPED_GPU=1 and a CUDA device")

import nvfp4  # noqa: E402  (the organisers' helper)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_grouped_v1_moe", KERNEL)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kernel_file_rules():
    src = KERNEL.read_text()
    assert src.splitlines()[0] == "# okc: problem=muse_moe_w4a16"
    for bad in (r"\.item\(", r"\.cpu\(", r"subprocess", r"import os", r"import socket", r"requests", r"environ",
                r"import reference", r"from reference", r"\.tolist\(", r"MEASUREMENT-ONLY", r"def _measure"):
        assert not re.search(bad, src), bad
    assert len([line for line in src.splitlines() if line.startswith("#")]) >= 15
    assert "kernels.muse_" not in src and "from kernels" not in src
    compile(src, "grouped_v1.py", "exec")


# ------------------------------------------------------------------------------------------------ dequant exactness (torch only)
def test_product_fits_bf16_and_fp16_exactly():
    """E2M1 value x E4M3 block scale is exactly representable in BF16 (and FP16), for every code and every finite scale byte."""
    codes = torch.arange(16, dtype=torch.uint8)
    grid = nvfp4.unpack_e2m1(torch.stack((codes[0::2] | (codes[1::2] << 4),), 0)).flatten().double()  # 16 values, code order
    assert grid.numel() == 16
    sc = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    sc = sc[~torch.isnan(sc)]
    assert sc.numel() == 254
    prod = grid[:, None] * sc[None, :]
    for dt in (torch.bfloat16, torch.float16):
        assert torch.equal(prod.to(dt).double(), prod), dt
    # and the FP32 product of the two FP32 values is that same exact number
    assert torch.equal((grid.float()[:, None] * sc.float()[None, :]).double(), prod)
    # the per-expert FP32 global scale is not folded in (it is applied to the FP32 accumulator), so BF16 never sees it
    assert prod.abs().max() == 6 * 448


def test_repack_roundtrip_is_lossless(mod):
    g = torch.Generator().manual_seed(1)
    for gated, (e, n, k) in itertools.product((False, True), [(2, 64, 256), (1, 32, 128), (3, 96, 384)]):
        w, sf, _ = nvfp4.random_weights(n, k, generator=g, device="cpu", leading=(e,), wide_scales=True)
        wp, sp = mod.repack_nvfp4(w, sf, gated)
        assert wp.shape == (e, n // 16, k // 128, 16, 64) and sp.shape == (e, n // 16, k // 128, 16, 8)
        kb = k // 128
        if gated:  # undo the gate/up interleave
            half = n // 2 // 16
            wp = wp.reshape(e, half, 2, kb, 16, 64).transpose(1, 2).reshape(e, n // 16, kb, 16, 64)
            sp = sp.reshape(e, half, 2, kb, 16, 8).transpose(1, 2).reshape(e, n // 16, kb, 16, 8)
        back_s = sp.permute(0, 1, 3, 2, 4).reshape(e, n, k // 16)
        assert torch.equal(back_s, sf.view(torch.uint8))
        assert torch.equal(wp.permute(0, 1, 3, 2, 4).reshape(e, n, k // 2), w)


# ------------------------------------------------------------------------------------------------ tile list
def _check_align(sorted_ids, expert_ids, n_post, local_ids, bm):
    flat = local_ids.reshape(-1)
    numel = flat.numel()
    npost = int(n_post)
    assert npost % bm == 0
    seen = []
    for b in range(npost // bm):
        blk = sorted_ids[b * bm:(b + 1) * bm]
        e = int(expert_ids[b])
        live = blk[blk < numel]
        assert (flat[live.long()] == e).all()
        seen += live.tolist()
        assert (blk[blk >= numel] == numel).all()
    assert sorted(seen) == list(range(numel))  # every slot, local or not, in exactly one block
    eb = expert_ids[:npost // bm].tolist()
    assert eb == sorted(eb)  # -1 (non-local) first, then experts ascending: the kernel counts the leading -1 blocks


@pytest.mark.parametrize("bm", [16, 32, 64, 128])
def test_align_torch_skewed_and_empty(mod, bm):
    g = torch.Generator().manual_seed(3)
    m = 37
    ids = torch.randint(-1, 32, (m, 8), generator=g, dtype=torch.int32)
    ids[:20] = 5  # one hot expert with 160 slots (many M blocks), most experts empty
    ids[20:25] = -1  # tokens with no local expert
    for num_experts in (32, 256):
        s, e, n = mod.moe_align_torch(ids, bm, num_experts)
        _check_align(s, e, n, ids, bm)
    ids[:] = -1  # nothing local at all
    s, e, n = mod.moe_align_torch(ids, bm, 32)
    _check_align(s, e, n, ids, bm)


# ------------------------------------------------------------------------------------------------ the grouped GEMM
def _weights(e, n, k, seed, dtype_scales_wide=True):
    g = torch.Generator().manual_seed(seed)
    w, sf, gs = nvfp4.random_weights(n, k, generator=g, device="cpu", leading=(e,), wide_scales=dtype_scales_wide)
    gsv = (gs * (1 + 0.25 * torch.rand(e, generator=g))).float()
    return w, sf, gsv


def _deq(w, sf, gsv):
    """float32 [E, N, K] = e2m1 * block scale (the organisers' dequantize without the global scale)."""
    return nvfp4.dequantize(w, sf, 1.0)


def _gemm_case(mod, dt, gated, m=24, a_div=8, n_out=64, k=256, num_local=6, hot=True, seed=0, bm=16, bnh=32, ch=1, nprog=5, warps=4,
               native=False, deq=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    n_rows = (2 if gated else 1) * n_out
    w, sf, gsv = _weights(num_local, n_rows, k, seed + 1)
    wp, sp = mod.repack_nvfp4(w, sf, gated, deq == 2)
    n_slots = m * a_div if a_div == 8 else m
    ids = torch.randint(-1, num_local, (n_slots,), generator=g, dtype=torch.int32)
    if hot:
        ids[: n_slots // 2] = 2  # a hot expert spanning several M blocks; experts 0 and the last stay (nearly) empty
        ids[ids == 0] = 1
    nrows_a = m if a_div == 8 else n_slots
    a = (torch.randn(nrows_a, k, generator=g)).to(dt)
    s_ids, e_ids, n_post = mod.moe_align_torch(ids.reshape(-1, 1), bm, num_local)
    out = torch.full((n_slots, n_out), float("nan"), dtype=dt)
    ctr = torch.zeros(1, dtype=torch.int32)
    args = [t.to(device) for t in (a, wp, sp, gsv)]
    out = out.to(device)
    ctr = ctr.to(device)
    mod.grouped_gemm(args[0], args[1], args[2], args[3], out, s_ids.to(device), e_ids.to(device), n_post.to(device), ctr, n_slots,
                     k=k, n_out=n_out, a_div=a_div, gated=gated, bm=bm, bnh=bnh, ch=ch, warps=warps, stages=2, nprog=nprog,
                     native=native, deq=deq)
    # reference: FP32 math on the same (dt-rounded) activations, FP32 dequantized weights
    wd = _deq(w, sf, gsv)
    ref = torch.full((n_slots, n_out), float("nan"))
    af = a.float()
    for s in range(n_slots):
        e = int(ids[s])
        if e < 0:
            continue
        row = af[s // a_div]
        if gated:
            gt = (wd[e, :n_out] @ row) * gsv[e]
            up = (wd[e, n_out:] @ row) * gsv[e]
            ref[s] = torch.nn.functional.silu(gt) * up
        else:
            ref[s] = (wd[e] @ row) * gsv[e]
    n_tiles = (int(n_post) // bm - int((e_ids[: int(n_post) // bm] < 0).sum())) * (n_out // bnh)
    return out.cpu().float(), ref, ids, int(ctr.cpu()), n_tiles


def _assert_gemm(out, ref, ids, tol, dt_name):
    live = (ids >= 0).reshape(-1)
    assert not torch.isnan(out[live]).any(), "a live slot was not written"
    assert torch.isnan(out[~live]).all(), "a non-local slot was written"
    err = (out[live] - ref[live]).norm() / ref[live].norm()
    assert err < tol, (dt_name, float(err))


@needs_interpreter
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("dt,tol", [(torch.float32, 2e-5), (torch.float16, 3e-3)])
def test_grouped_gemm_matches_torch(mod, gated, dt, tol):
    out, ref, ids, fetched, n_tiles = _gemm_case(mod, dt, gated)
    _assert_gemm(out, ref, ids, tol, str(dt))
    assert fetched == n_tiles  # every tile was taken exactly once (one counter fetch per processed tile)


@needs_interpreter
@pytest.mark.parametrize("nprog", [1, 3, 64])
@pytest.mark.parametrize("bm,bnh,ch", [(16, 16, 1), (32, 32, 2), (64, 64, 1)])
def test_grouped_gemm_schedules(mod, nprog, bm, bnh, ch):
    out, ref, ids, fetched, n_tiles = _gemm_case(mod, torch.float32, True, bm=bm, bnh=bnh, ch=ch, nprog=nprog, k=256, n_out=64, seed=7)
    _assert_gemm(out, ref, ids, 2e-5, f"{bm}.{bnh}.{ch}.{nprog}")
    assert fetched == n_tiles


@needs_interpreter
def test_grouped_gemm_second_gemm_rows_and_native_modes(mod):
    """GEMM2 layout (A rows = slots, a_div 1), the denormal-multiply dequant and the native FP8 scale decode."""
    for deq, native in ((0, False), (1, False), (0, True)):
        try:
            out, ref, ids, fetched, n_tiles = _gemm_case(mod, torch.float32, False, m=40, a_div=1, n_out=96, k=128, bnh=32, bm=16,
                                                         deq=deq, native=native, seed=11)
        except Exception as e:  # the interpreter may not know float8e4nv; the integer decode is what the RTX 2060 runs
            if native:
                pytest.skip(f"interpreter without native FP8: {type(e).__name__}")
            raise
        _assert_gemm(out, ref, ids, 2e-5, f"deq{deq} native{native}")
        assert fetched == n_tiles


@needs_interpreter
def test_grouped_gemm_nothing_local_and_one_expert(mod):
    # nothing local: no tile at all, no write
    g = torch.Generator().manual_seed(0)
    k, n_out, nl = 128, 32, 4
    w, sf, gsv = _weights(nl, n_out, k, 5)
    wp, sp = mod.repack_nvfp4(w, sf, False)
    ids = torch.full((16, 1), -1, dtype=torch.int32)
    s, e, n = mod.moe_align_torch(ids, 16, nl)
    a = torch.randn(16, k, generator=g)
    out = torch.full((16, n_out), 7.0)
    ctr = torch.zeros(1, dtype=torch.int32)
    mod.grouped_gemm(a, wp, sp, gsv, out, s, e, n, ctr, 16, k=k, n_out=n_out, a_div=1, gated=False, bm=16, bnh=32, ch=1, warps=4,
                     stages=2, nprog=4, native=False)
    assert (out == 7.0).all() and int(ctr) == 0


# ------------------------------------------------------------------------------------------------ dequant exactness through the kernel
@needs_interpreter
@pytest.mark.parametrize("deq", [0, 1])
def test_kernel_dequant_is_exact_for_all_code_scale_pairs(mod, deq):
    """One-hot activations pick out single weights: the GEMM's output is the dequantized weight itself, for every one of the 16 x 254
    (code, scale) pairs (codes = (n + k) mod 16 inside each 16-group, scale bytes cycling through all finite E4M3 values)."""
    n, k = 256, 256
    codes = ((torch.arange(n)[:, None] + torch.arange(k)[None, :]) % 16).to(torch.uint8)
    w = codes[:, 0::2] | (codes[:, 1::2] << 4)
    finite = [b for b in range(256) if not torch.tensor([b], dtype=torch.uint8).view(torch.float8_e4m3fn).isnan().item()]
    cell = torch.arange(n * (k // 16)).reshape(n, k // 16)
    sfb = torch.tensor(finite, dtype=torch.uint8)[cell % len(finite)]
    sf = sfb.view(torch.float8_e4m3fn)
    wp, sp = mod.repack_nvfp4(w.unsqueeze(0), sf.unsqueeze(0), False)
    a = torch.eye(k)
    ids = torch.zeros((k, 1), dtype=torch.int32)
    s, e, np_ = mod.moe_align_torch(ids, 16, 1)
    out = torch.zeros(k, n)
    mod.grouped_gemm(a, wp, sp, torch.ones(1), out, s, e, np_, torch.zeros(1, dtype=torch.int32), k, k=k, n_out=n, a_div=1, gated=False,
                     bm=16, bnh=64, ch=1, warps=4, stages=2, nprog=6, native=False, deq=deq)
    want = nvfp4.dequantize(w, sf, 1.0).T  # [k, n]
    assert torch.equal(out, want)
    assert torch.equal(out.to(torch.bfloat16).float(), want)  # and BF16 rounding of the dequantized value changes nothing


# ------------------------------------------------------------------------------------------------ whole large-M path vs FP32 reference
def _reference(x, ids, tw, wts, h, inter, eps=1e-8):
    """reference.py's math at small width, FP32 (global ids == local ids here)."""
    out = torch.zeros(x.shape[0], h)
    xf = x.float()
    for e in range(wts["w13"].shape[0]):
        tok, slot = (ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        hh = xf[tok] @ nvfp4.dequantize(wts["w13"][e], wts["w13_sf"][e], wts["w13_gs"][e]).T
        mid = torch.nn.functional.silu(hh[:, :inter]) * hh[:, inter:]
        y = mid @ nvfp4.dequantize(wts["w2"][e], wts["w2_sf"][e], wts["w2_gs"][e]).T
        y = y * (1.0 + wts["post_norm_w"].float())
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        out.index_add_(0, tok, y * tw[tok, slot].unsqueeze(1))
    return out


def _whole_path(mod, monkeypatch, m, skew, device, deq=1):
    h, inter, ne = 256, 128, 8
    dt = torch.float16
    g = torch.Generator().manual_seed(m)
    w13, w13_sf, gs13 = nvfp4.random_weights(2 * inter, h, generator=g, device="cpu", leading=(ne,))
    w2, w2_sf, gs2 = nvfp4.random_weights(h, inter, generator=g, device="cpu", leading=(ne,))
    wts = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(ne).contiguous(), "w2": w2, "w2_sf": w2_sf, "w2_gs": gs2.expand(ne).contiguous(),
           "post_norm_w": (0.1 * torch.randn(h, generator=g))}
    x = torch.randn(m, h, generator=g).half()
    scores = torch.rand(m, ne, generator=g) + (torch.rand(ne, generator=g).pow(6) * 4 if skew else 0)
    top = scores.topk(8, dim=-1)
    ids = top.indices.to(torch.int32)
    ids[ids >= ne - 2] = -1  # experts 6 and 7 behave as non-local
    tw = (top.values / top.values.sum(-1, keepdim=True)).float()
    sl = ne - 2
    ref = _reference_masked(x, ids, tw, wts, h, inter, sl)
    dev = lambda v: v.to(device) if isinstance(v, torch.Tensor) else v  # noqa: E731
    data = {"hidden_states": dev(x), "topk_ids": dev(ids), "topk_weights": dev(tw), "config": {"local_start": 0, "local_count": sl, "bs": m}}
    state = {"grouped": {(0, sl): mod._prepare_grouped({k: dev(v) for k, v in wts.items()}, 0, sl, 3, deq)},
             "norm_w": dev((wts["post_norm_w"].float() + 1.0).to(dt))}
    monkeypatch.setattr(mod, "grouped_cfg", lambda mm: {"bm": 16, "g1": dict(bnh=32, ch=1, warps=4, stages=2),
                                                        "g2": dict(bnh=64, ch=1, warps=4, stages=2), "deq": deq})
    post = mod.epilogue_triton if device == "cuda" else mod.epilogue_torch
    out = mod._kernel_grouped(data, state, lambda li, bm: mod.moe_align_torch(li, bm, sl), post, False)
    err = (out.cpu().float() - ref).norm() / ref.norm()
    assert out.dtype == dt and err < 5e-3, float(err)


@needs_interpreter
@pytest.mark.parametrize("m,skew", [(33, False), (40, True)])
def test_whole_path_small_widths(mod, monkeypatch, m, skew):
    _whole_path(mod, monkeypatch, m, skew, "cpu")


@needs_gpu
@pytest.mark.parametrize("m,skew,deq", [(33, False, 1), (47, True, 2), (64, True, 0)])
def test_gpu_whole_path_small_widths(mod, monkeypatch, m, skew, deq):
    _whole_path(mod, monkeypatch, m, skew, "cuda", deq)


def _reference_masked(x, ids, tw, wts, h, inter, nlocal):
    ids2 = ids.clone()
    ids2[ids2 >= nlocal] = -1
    sub = {k: (v[:nlocal] if k != "post_norm_w" else v) for k, v in wts.items()}
    return _reference(x, ids2, tw, sub, h, inter)


# ------------------------------------------------------------------------------------------------ dispatch
def test_dispatch_is_shape_only(mod, monkeypatch):
    """kernel_fn takes the grouped path from the number of tokens alone (>= GROUPED_MIN_M), never from values."""
    assert mod.GROUPED_MIN_M == 64
    assert mod.grouped_cfg(256)["bm"] in (16, 32, 64, 128)
    calls = []
    monkeypatch.setattr(mod, "HAVE_SGLANG", True)
    monkeypatch.setattr(mod, "_kernel_grouped", lambda *a: calls.append("grouped"))
    monkeypatch.setattr(mod, "_kernel_sglang", lambda *a: calls.append("marlin"))
    for m in (1, 8, 16, 31, 32, 33, 63, 64, 128, 257):
        mod.kernel_fn({"hidden_states": torch.zeros(m, 4)}, {"grouped": {}})
    assert calls == ["marlin"] * 7 + ["grouped"] * 3
    calls.clear()
    mod.kernel_fn({"hidden_states": torch.zeros(256, 4)}, {"grouped": None})  # not Hopper / USE_GROUPED off
    assert calls == ["marlin"]


# ------------------------------------------------------------------------------------------------ the RTX 2060 (real Triton, FP16 operands)
@needs_gpu
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("deq", [0, 1, 2])
def test_gpu_fp16_gemm_matches_torch(mod, gated, deq):
    out, ref, ids, fetched, n_tiles = _gemm_case(mod, torch.float16, gated, device="cuda", deq=deq, bm=32, bnh=32, nprog=7, warps=4)
    _assert_gemm(out, ref, ids, 3e-3, f"gpu fp16 deq{deq}")
    assert fetched == n_tiles
