# test_moe_smallm_cuda.py
import math
import pytest
import torch
import torch.nn.functional as F

from moe_smallm_cuda import (
    epilogue_triton,
    gemm1_smallm,
    gemm2_smallm,
    local_ids_triton,
    make_smallm_workspace,
)

_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequantize_nvfp4(packed: torch.Tensor, sf: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_E2M1_GRID, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8)
    codes = torch.stack((p & 0x0F, (p >> 4) & 0x0F), dim=-1).flatten(-2)
    mag = grid[(codes & 0x7).long()]
    vals = torch.where((codes & 0x8) != 0, -mag, mag)
    blocks = vals.view(*vals.shape[:-1], vals.shape[-1] // 16, 16)
    return (blocks * (sf.float() * gs).unsqueeze(-1)).view_as(vals)


def make_random_nvfp4(E: int, N: int, K: int, gen: torch.Generator, device: str = "cuda"):
    packed = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=gen, device=device)
    z = torch.randn((E, N, K // 16), generator=gen, device=device)
    sf = torch.exp2(7.0 + 0.4 * z).clamp(8.0, 448.0).to(torch.float8_e4m3fn)
    mean_sq = (4.0 ** 7.0) * math.exp(2 * (math.log(2) * 0.4) ** 2)
    gs = torch.full((E,), 1.0 / (8.6 * mean_sq * K) ** 0.5, dtype=torch.float32, device=device)
    return packed, sf, gs


def rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("M", [1, 4, 8, 16])
@pytest.mark.parametrize("k_slices", [1, 2, 4])
def test_gemm1_and_gemm2_vs_fp32_ref(M: int, k_slices: int):
    device = "cuda"
    E, H, I, top_k = 8, 256, 256, 8
    gen = torch.Generator(device=device).manual_seed(100 + M * 10 + k_slices)

    x = torch.randn((M, H), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((M, 32), generator=gen, device=device)
    topk_ids = scores.topk(top_k, dim=-1).indices.to(torch.int32)
    # Guarantee some local hits in [0, E)
    topk_ids[:, 0] = torch.arange(M, device=device, dtype=torch.int32) % E
    local_ids = local_ids_triton(topk_ids, 0, E)

    w13, w13_sf, w13_gs = make_random_nvfp4(E, 2 * I, H, gen, device)
    w2, w2_sf, w2_gs = make_random_nvfp4(E, H, I, gen, device)
    ws = make_smallm_workspace(device, max_m=16, top_k=top_k, n_local=E, max_n_tiles=8, max_k_slices=8)

    # Run twice without re-zeroing counters to verify last-CTA counter reset
    for _ in range(2):
        inter = gemm1_smallm(x, w13, w13_sf, w13_gs, local_ids, ws["partials"], ws["counters"], k_slices=k_slices)
        y = gemm2_smallm(inter, w2, w2_sf, w2_gs, local_ids, ws["partials"], ws["counters"], k_slices=k_slices)

    assert int(ws["counters"].abs().max().item()) == 0, "Fixup CTA did not reset counters to 0"

    ref_inter = torch.zeros((M, top_k, I), dtype=torch.float32, device=device)
    ref_y = torch.zeros((M, top_k, H), dtype=torch.float32, device=device)
    mask = local_ids >= 0
    for e in range(E):
        tok, slot = (local_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w13_dq = dequantize_nvfp4(w13[e], w13_sf[e], w13_gs[e])
        h = x[tok].float() @ w13_dq.T
        inter_e = F.silu(h[:, :I]) * h[:, I:]
        ref_inter[tok, slot] = inter_e
        w2_dq = dequantize_nvfp4(w2[e], w2_sf[e], w2_gs[e])
        ref_y[tok, slot] = inter_e @ w2_dq.T

    assert rel_l2(inter[mask], ref_inter[mask]) < 1.0e-2
    assert rel_l2(y[mask], ref_y[mask]) < 1.2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("M", [1, 4, 8, 16])
def test_end_to_end_epilogue_and_cudagraph(M: int):
    device = "cuda"
    E, n_experts, H, I, top_k = 32, 256, 512, 512, 8
    gen = torch.Generator(device=device).manual_seed(42 + M)

    x = torch.randn((M, H), generator=gen, device=device).to(torch.bfloat16)
    scores = torch.rand((M, n_experts), generator=gen, device=device)
    routed = scores.topk(top_k, dim=-1)
    topk_ids = (routed.indices % 48).to(torch.int32)  # ~2/3 local (0..31), ~1/3 non-local (32..47)
    topk_weights = (routed.values / routed.values.sum(-1, keepdim=True)).float()
    post_norm_w = (0.1 * torch.randn((H,), generator=gen, device=device)).to(torch.bfloat16)
    norm_w = (post_norm_w.float() + 1.0).to(torch.bfloat16)

    w13, w13_sf, w13_gs = make_random_nvfp4(E, 2 * I, H, gen, device)
    w2, w2_sf, w2_gs = make_random_nvfp4(E, H, I, gen, device)
    ws = make_smallm_workspace(device, max_m=16, top_k=top_k, n_local=E, max_n_tiles=16, max_k_slices=8)

    def run_step(inp_x, ids, weights):
        lids = local_ids_triton(ids, 0, E)
        inter = gemm1_smallm(inp_x, w13, w13_sf, w13_gs, lids, ws["partials"], ws["counters"])
        y = gemm2_smallm(inter, w2, w2_sf, w2_gs, lids, ws["partials"], ws["counters"])
        return epilogue_triton(y, weights, lids, norm_w)

    eager_out = run_step(x, topk_ids, topk_weights)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_out = run_step(x, topk_ids, topk_weights)
    g.replay()

    assert torch.equal(eager_out, graph_out), "Non-deterministic or graph-unsafe execution"

    ref_out = torch.zeros((M, H), dtype=torch.float32, device=device)
    for e in range(E):
        tok, slot = (topk_ids == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        h = x[tok].float() @ dequantize_nvfp4(w13[e], w13_sf[e], w13_gs[e]).T
        inter_e = F.silu(h[:, :I]) * h[:, I:]
        y_e = inter_e @ dequantize_nvfp4(w2[e], w2_sf[e], w2_gs[e]).T
        y_norm = y_e * (1.0 + post_norm_w.float())
        y_norm = y_norm * torch.rsqrt(y_norm.pow(2).mean(-1, keepdim=True) + 1e-8)
        ref_out.index_add_(0, tok, y_norm * topk_weights[tok, slot].unsqueeze(1))

    assert rel_l2(graph_out, ref_out.to(torch.bfloat16)) < 1.5e-2
