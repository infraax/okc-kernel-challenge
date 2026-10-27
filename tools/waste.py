#!/usr/bin/env python3
"""Waste ledger (floor-first): per problem, time-weighted ours/floor and the shapes with the most µs above the physical
floor (sol) × weight, from a judge profile. Usage: tools/waste.py  (edit `best` to the jobs to compare)."""
import sys, re, glob
sys.path.insert(0, "tools")
import score_sim as s
best = {"muse_bf16_proj": "c3b7c70b5edd43c5", "muse_attention_decode": "05c8baf70e8f4afb", "muse_decode_layer": "bc2ad26361ec4a0f"}
for prob, job in best.items():
    prof = f"experiments/profiles/{prob}.{job}.txt"
    rows = []
    for ln in open(prof):
        m = re.match(r'(\S.*?)\s{2,}([\d,.]+)\s+([\d,.]+)\s+[\d.]+x\s+([\d,.]+)', ln)
        if m and ('load=' in m.group(1)):
            case = m.group(1).strip(); sg, yo, sol = (float(m.group(i).replace(',', '')) for i in (2, 3, 4))
            rows.append((case, sg, yo, sol))
    weights = s.parse_task_md(prob)
    tot_y = tot_f = 0; waste = []
    for case, sg, yo, sol in rows:
        k = s.get_case_match(case, weights)
        info = weights.get(k) if isinstance(k, str) else (k if isinstance(k, dict) else None)
        w = float(info['weight']) if info else 1.0
        waste.append(((yo - sol) * w, case, yo, sol, w))
        tot_y += yo * w; tot_f += sol * w
    waste.sort(reverse=True)
    print(f"\n== {prob} (job {job}): ours/floor = {tot_y/tot_f:.2f}x above the physical floor (time-weighted)")
    for wv, case, yo, sol, w in waste[:5]:
        print(f"  waste {wv:9.0f} us/step  ours {yo:8.2f} floor {sol:7.2f} (x{w})  {case[:70]}")
