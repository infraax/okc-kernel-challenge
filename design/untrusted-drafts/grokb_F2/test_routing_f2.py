# pytest for fused_router_f2. Needs sm_80+ (bf16 tl.dot), or TRITON_INTERPRET=1.
import pytest
import torch

from f2_router import fused_router_f2

MULT, EPS = 0.1767766952966369, 1e-15
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="bf16 tl.dot",
)


def _split(router):
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    return hi.contiguous(), lo.contiguous()


def _oracle_logits(x, hi, lo):
    """Same three products as the kernel, in FP32 (tie groups stay exact)."""
    if x.dtype == torch.float32:
        xh = x.to(torch.bfloat16)
        xl = (x - xh.float()).to(torch.bfloat16)
        return xh.float() @ hi.float().T + xl.float() @ hi.float().T + xh.float() @ lo.float().T
    return x.float() @ hi.float().T + x.float() @ lo.float().T


def _reference_route(logits, bias, local_start):
    """reference.py route + topk + renorm + sonnet_v2 local ids."""
    scores = torch.sigmoid(logits.float() * MULT)
    biased = scores + bias.float()
    ids = biased.topk(8, dim=-1).indices.to(torch.int32)
    picked = scores.gather(1, ids.long())
    weights = picked / (picked.sum(-1, keepdim=True) + EPS)
    rel = (ids - int(local_start)) % 256
    local = torch.where(rel < 32, rel, torch.full_like(rel, -1)).to(torch.int32)
    return ids, weights, local


def _run(x, router, bias, local_start):
    hi, lo = _split(router)
    start = torch.tensor([local_start], dtype=torch.int32, device=x.device)
    return fused_router_f2(x, hi, lo, bias, start)


def test_exact_ties_and_near_ties():
    """x = 0 => logits are 0, so ranking is pure bias and matches topk bit-for-bit.

    Nine experts share a bias (the lower eight indices win). One more sits 1e-5
    below that bias and must lose. Covers FP32-x (T <= 8) and BF16-x, and a
    local_start that splits the chosen set across this GPU's 32 experts.
    """
    device = "cuda"
    for T, dtype in ((4, torch.float32), (8, torch.float32), (16, torch.bfloat16), (32, torch.bfloat16)):
        x = torch.zeros(T, 8192, dtype=dtype, device=device)
        router = torch.randn(256, 8192, device=device)
        bias = torch.zeros(256, device=device)
        bias[60:69] = 1.0
        bias[80] = 1.0 - 1e-5
        ids, weights, local = _run(x, router, bias, 64)
        ref_ids, ref_w, ref_loc = _reference_route(torch.zeros(T, 256, device=device), bias, 64)
        assert torch.equal(ids, ref_ids)
        assert torch.equal(ids, torch.arange(60, 68, device=device, dtype=torch.int32).expand(T, 8))
        assert torch.equal(local, ref_loc)
        assert torch.allclose(weights, ref_w, rtol=1e-5, atol=1e-6)
        # 60..63 are outside [64, 96); 64..67 are local 0..3.
        assert torch.equal(local[:, :4], torch.full((T, 4), -1, device=device, dtype=torch.int32))
        assert torch.equal(local[:, 4:], torch.arange(4, device=device, dtype=torch.int32).expand(T, 4))


def test_random_within_reference_routing_rule():
    """Nonzero GEMV. Ids must satisfy the judge's near-tie rule against the
    reference routing of the same hi+lo logits, and local ids must match the
    modulo map. Planted equal rows force a true tie inside the random scores.
    """
    device = "cuda"
    g = torch.Generator(device=device)
    g.manual_seed(7)
    for T, dtype, start in ((1, torch.float32, 0), (8, torch.float32, 32),
                            (16, torch.bfloat16, 96), (32, torch.bfloat16, 224)):
        x = torch.randn(T, 8192, dtype=torch.float32, device=device, generator=g)
        x = x if dtype == torch.float32 else x.to(torch.bfloat16)
        router = torch.randn(256, 8192, device=device, generator=g) * 1e-4
        router[4] = router[11] = router[19]  # exact tie, index order 4, 11, 19
        bias = torch.randn(256, device=device, generator=g) * 0.05
        bias[4] = bias[11] = bias[19] = 2.0
        hi, lo = _split(router)
        ids, weights, local = fused_router_f2(
            x, hi, lo, bias, torch.tensor([start], dtype=torch.int32, device=device))
        ref_ids, ref_w, ref_loc = _reference_route(_oracle_logits(x, hi, lo), bias, start)
        assert ids.dtype == torch.int32 and weights.dtype == torch.float32
        assert ids.shape == (T, 8) and local.shape == (T, 8)
        assert torch.equal(ids.sort(-1).values.unique_consecutive().numel() and
                           (ids.sort(-1).values[:, 1:] != ids.sort(-1).values[:, :-1]).all(),
                           torch.tensor(True, device=device))
        scores = torch.sigmoid(_oracle_logits(x, hi, lo) * MULT)
        biased = scores + bias
        eighth = biased.topk(8, dim=-1).values[:, -1]
        gap = eighth[:, None] - biased.gather(1, ids.long())
        assert float(gap.max()) <= 0.05
        ninth = biased.topk(9, dim=-1).values[:, -1]
        best = biased.topk(8, dim=-1)
        kept = (best.indices[:, :, None] == ids.long()[:, None, :]).any(-1)
        lost = (~kept) & ((best.values - ninth[:, None]) > 0.03)
        assert not bool(lost.any())
        rel = (ids - start) % 256
        assert torch.equal(local, torch.where(rel < 32, rel, torch.full_like(rel, -1)).to(torch.int32))
        # The planted tie is exact (identical rows, equal bias, split-order sum).
        assert torch.equal(ids, ref_ids)
        assert torch.allclose(weights, ref_w, rtol=1e-4, atol=1e-5)
        assert torch.equal(local, ref_loc)
        assert torch.allclose(weights.sum(-1), torch.ones(T, device=device), atol=1e-5)
