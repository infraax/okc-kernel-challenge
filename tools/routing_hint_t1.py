#!/usr/bin/env python3
"""T1 CPU routing-hint analysis for muse_decode_layer (design/test-plan-moe-routing-hint.md).

Measures fraction of exact local routed expert-bytes covered by three pre-attention guesses:
  (a) pre-attention router on incoming normalized stream (attn_input * (1 + pre_ffn_norm))
  (b) static popularity prior from router_bias alone (top-N local experts by router_bias)
  (c) priority union of (a) top-8 local experts + (b) bias ranking under a byte budget
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch

PROBLEM_DIR = Path(__file__).resolve().parents[1] / "problems" / "muse_decode_layer"
if str(PROBLEM_DIR) not in sys.path:
    sys.path.insert(0, str(PROBLEM_DIR))

import nvfp4  # noqa: E402
import reference as ref  # noqa: E402

L2_BYTES = 50 * 1000 * 1000  # 50 MB H200 L2 cache
MB = 1e6

# Scored decode_layer shapes (TASK.md) + nearby T in {1, 2, 8, 16, 32}.
# ctx is shrunk to <= 4096 so CPU cache generation & attention stay well under 4 GB RAM.
CASES = [
    # Scored decode shapes (q_len=1, T in {1, 16, 32})
    ("decode-1*", 1, 1, "sliding", "fixed", "soft", True),
    ("decode-1*", 1, 1, "full", "fixed", "soft", True),
    ("decode-16*", 16, 1, "sliding", "agentic", "soft", True),
    ("decode-16*", 16, 1, "full", "agentic", "soft", True),
    ("decode-32*", 32, 1, "sliding", "agentic", "soft", True),
    ("decode-32*", 32, 1, "full", "agentic", "soft", True),
    # Scored verify shapes (q_len=8, T in {8, 128, 256})
    ("verify-1*", 1, 8, "sliding", "fixed", "soft", True),
    ("verify-1*", 1, 8, "full", "fixed", "soft", True),
    ("verify-16*", 16, 8, "sliding", "agentic", "soft", True),
    ("verify-16*", 16, 8, "full", "agentic", "soft", True),
    ("verify-32*", 32, 8, "sliding", "agentic", "soft", True),
    ("verify-32*", 32, 8, "full", "agentic", "soft", True),
    # Nearby shapes covering T in {2, 8, 16, 32} across decode (q_len=1) and verify (q_len=8)
    ("near-d2", 2, 1, "sliding", "agentic", "soft", False),
    ("near-d8", 8, 1, "sliding", "agentic", "soft", False),
    ("near-v2", 2, 8, "sliding", "agentic", "soft", False),
    ("near-v4", 4, 8, "sliding", "agentic", "soft", False),
]


def expert_nvfp4_bytes(weights: dict) -> int:
    """Exact NVFP4 bytes per local expert (w13 + w13_sf + w13_gs + w2 + w2_sf + w2_gs)."""
    analytic = int(3 * ref.D_EXPERT * ref.D_EXPERT * (0.5 + 1.0 / nvfp4.BLOCK)) + 8
    if weights.get("w13") is None:
        return analytic
    keys = ("w13", "w13_sf", "w13_gs", "w2", "w2_sf", "w2_gs")
    measured = sum(weights[k][0].numel() * weights[k][0].element_size() for k in keys)
    assert measured == analytic, f"mismatch: {measured} != {analytic}"
    return measured


def local_set_and_ranking(biased: torch.Tensor, local_start: int) -> tuple[list[int], list[int]]:
    """Returns (active local experts in top-8, all 32 local experts ranked by top-8 freq then max score)."""
    top_ids = biased.topk(ref.TOP_K, dim=-1).indices
    rel = top_ids - local_start
    local_hits = rel[(rel >= 0) & (rel < ref.LOCAL_EXPERTS)]
    counts = torch.bincount(local_hits, minlength=ref.LOCAL_EXPERTS).float()
    max_scores = biased[:, local_start : local_start + ref.LOCAL_EXPERTS].max(dim=0).values
    # Composite key: prioritize experts actually chosen in top-8 by count, tie-break by max score.
    order = sorted(range(ref.LOCAL_EXPERTS), key=lambda e: (int(counts[e].item()), float(max_scores[e].item())),
                   reverse=True)
    active = [e for e in order if counts[e] > 0]
    return active, order


def cover_stats(exact_local: set[int], ranked_guess: list[int], budget_bytes: int,
                exp_bytes: int) -> tuple[float, float]:
    """(hit_rate_over_exact_routed_bytes, prefetched_bytes) under a byte cap."""
    if not exact_local:
        return 1.0, 0.0
    rem = budget_bytes
    covered = 0
    prefetched = 0
    for e in ranked_guess:
        if rem <= 0:
            break
        take = min(exp_bytes, rem)
        prefetched += take
        rem -= take
        if e in exact_local:
            covered += take
    return covered / (len(exact_local) * exp_bytes), float(prefetched)


@torch.no_grad()
def evaluate_shape(batch: int, q_len: int, layer: str, spread: str, dist: str, ctx: int,
                   trials: int, weights: dict, exp_bytes: int, raw_attn_input: bool) -> dict:
    n_grid = (1, 2, 4, 8)
    b_grid = (1 * exp_bytes, int(40 * MB), L2_BYTES, 2 * exp_bytes, 4 * exp_bytes)
    acc = {"routes": 0.0, "exact_n": 0.0, "a_raw_hit": 0.0, "a_raw_mb": 0.0}
    for n in n_grid:
        acc[f"a@{n}_hit"] = acc[f"b@{n}_hit"] = 0.0
    for b in b_grid:
        acc[f"c@{b}_hit"] = acc[f"c@{b}_mb"] = 0.0

    for t in range(trials):
        seed = 1000 + 97 * t + 13 * batch + 7 * q_len + (0 if layer == "sliding" else 1)
        data = ref.generate_input(batch=batch, ctx=ctx, layer=layer, seed=seed, q_len=q_len,
                                  spread=spread, dist=dist, page=1, device="cpu")
        front = ref.ref_front(data, weights)
        exact_ids = front["biased"].topk(ref.TOP_K, dim=-1).indices
        # Pick busiest 32-expert local range, matching reference.case_fields().
        pairs = torch.bincount((exact_ids // ref.LOCAL_EXPERTS).flatten(),
                               minlength=ref.N_EXPERTS // ref.LOCAL_EXPERTS)
        local_start = ref.LOCAL_EXPERTS * int(pairs.argmax())
        exact_active, _ = local_set_and_ranking(front["biased"], local_start)
        exact_set = set(exact_active)
        local_routes = int(((exact_ids - local_start) % ref.N_EXPERTS < ref.LOCAL_EXPERTS).sum())

        # (a) Pre-attention router applied to incoming normalized stream.
        h_pre = data["attn_input"].float()
        if not raw_attn_input:
            h_pre = h_pre * (1.0 + weights["pre_ffn_norm"].float())
        _, biased_a = ref.route(h_pre, weights)
        a_active, a_order = local_set_and_ranking(biased_a, local_start)

        # (b) Static popularity prior from router_bias alone on this local range.
        bias_local = weights["router_bias"][local_start : local_start + ref.LOCAL_EXPERTS]
        b_order = bias_local.argsort(descending=True).tolist()

        # (c) Priority union: (a) active top-8 local experts first, then (b) static bias order.
        c_union = a_active + [e for e in b_order if e not in set(a_active)]

        acc["routes"] += local_routes / trials
        acc["exact_n"] += len(exact_set) / trials
        hit_a, mb_a = cover_stats(exact_set, a_active, len(a_active) * exp_bytes, exp_bytes)
        acc["a_raw_hit"] += hit_a / trials
        acc["a_raw_mb"] += (mb_a / MB) / trials

        for n in n_grid:
            acc[f"a@{n}_hit"] += cover_stats(exact_set, a_order[:n], n * exp_bytes, exp_bytes)[0] / trials
            acc[f"b@{n}_hit"] += cover_stats(exact_set, b_order[:n], n * exp_bytes, exp_bytes)[0] / trials
        for b in b_grid:
            h_c, pref_c = cover_stats(exact_set, c_union, b, exp_bytes)
            acc[f"c@{b}_hit"] += h_c / trials
            acc[f"c@{b}_mb"] += (pref_c / MB) / trials
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=2048, help="Context length <= 4096 for CPU memory")
    ap.add_argument("--trials", type=int, default=3, help="Seeds averaged per shape")
    ap.add_argument("--raw-attn-input", action="store_true", help="Omit (1 + pre_ffn_norm) in guess (a)")
    args = ap.parse_args()
    assert args.ctx <= 4096, "Keep --ctx <= 4096 to stay within < 4 GB RAM on CPU"

    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cpu")
    exp_bytes = expert_nvfp4_bytes(weights)
    # Free unused NVFP4 weight buffers immediately so peak RSS stays small.
    for k in ("w13", "w13_sf", "w2", "w2_sf"):
        weights[k] = None

    b1, b40, b50, b2, b4 = exp_bytes, int(40 * MB), L2_BYTES, 2 * exp_bytes, 4 * exp_bytes
    print(f"muse_decode_layer T1 routing-hint analysis (CPU, ctx={args.ctx} <= 4096, trials={args.trials})")
    print("Note: shrinking ctx to <= 4096 does not affect routing statistics much: sliding layers only")
    print("attend to the last 2048 tokens anyway, and stream1 is dominated by alpha1*stream + soft-key mean.")
    print(f"NVFP4 expert size: {exp_bytes / MB:.2f} MB ({exp_bytes / (1 << 20):.2f} MiB) per expert "
          f"[w13: 18.87 MB, w2: 9.44 MB] | H200 L2 = {L2_BYTES / MB:.1f} MB (~{L2_BYTES / exp_bytes:.2f} experts)\n")

    hdr = (f"{'Shape':<11} {'Lyr':<4} {'T':>3} | {'ExExp':>5} {'ExMB':>6} | "
           f"{'(a)all Hit/MB':>13} {'a@1':>4} {'a@2':>4} {'a@4':>4} | "
           f"{'b@1':>4} {'b@2':>4} {'b@4':>4} {'b@8':>4} | "
           f"{'c@28M':>5} {'c@40M':>5} {'c@50M':>5} {'c@57M':>5} {'c@113M':>6}")
    print(hdr)
    print("-" * len(hdr))
    for name, b, q, lyr, spread, dist, _ in CASES:
        r = evaluate_shape(b, q, lyr, spread, dist, args.ctx, args.trials, weights, exp_bytes, args.raw_attn_input)
        t = b * q
        ex_mb = r["exact_n"] * exp_bytes / MB
        print(f"{name:<11} {lyr[:4]:<4} {t:>3} | {r['exact_n']:>5.1f} {ex_mb:>6.1f} | "
              f"{100*r['a_raw_hit']:>5.1f}%/{r['a_raw_mb']:>5.1f}M "
              f"{100*r['a@1_hit']:>3.0f}% {100*r['a@2_hit']:>3.0f}% {100*r['a@4_hit']:>3.0f}% | "
              f"{100*r['b@1_hit']:>3.0f}% {100*r['b@2_hit']:>3.0f}% "
              f"{100*r['b@4_hit']:>3.0f}% {100*r['b@8_hit']:>3.0f}% | "
              f"{100*r[f'c@{b1}_hit']:>4.0f}% {100*r[f'c@{b40}_hit']:>4.0f}% "
              f"{100*r[f'c@{b50}_hit']:>4.0f}% {100*r[f'c@{b2}_hit']:>4.0f}% {100*r[f'c@{b4}_hit']:>5.0f}%")
    print("-" * len(hdr))
    print("Prefetched MB for fixed N: N=1 -> 28.3 MB, N=2 -> 56.6 MB (>L2), N=4 -> 113.2 MB, N=8 -> 226.5 MB.")


if __name__ == "__main__":
    main()
