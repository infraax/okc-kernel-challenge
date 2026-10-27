#!/usr/bin/env python3
"""
Parses the log of kernels/muse_moe_w4a16/measure_altA_v1.py (MoE GEMM measurement check).

Input: a text log (path argument, or stdin with `-`). Relevant lines:
  MEAS-ALTA gpu=<..> ... bw_ach=<f>TB/s cal=<f>us(64MB) inst=12
  I moe m=<M> seeds=11/12/13 pairs=a/b/c distinct=a/b/c chunks=a/b/c n_act=a/b/c uniq1=<MB>/<MB>/<MB>MB uniq2=<MB>/<MB>/<MB>MB
  B moe m=<M> v=<variant> | g1=<us>/<us>/<us> g2=<us>/<us>/<us> | tb=<f>/<f> ach=<i>/<i>% x=<f>/<f> bw=<OK|e=1.2e-03|-> [pair=<us>/<us>/<us>] [!INVALID]
  S moe m=<M> g1 cta=<us>us item=<us>us span=<us> stage=<i>% stream=<i>% wait=<i>% fix=<i>% epi=<i>% bw/cta=<f>GB/s occ=<i>/<f> | g2 cta=... | pdl g2t0=<f> g2t1=<f> g2end=<f>
  MEAS-ALTA done <s>s [skipped: ...]
"""
TWINS = {'wstream', 'nocomp', 'nostage', 'stamps', 'cal'}  # diagnostic twins / ceilings, never shippable
import argparse
import json
import re
import sys
import statistics

def parse_floats(v_str):
    if not v_str or v_str == '-': return []
    v_str = re.sub(r'[a-zA-Z%]+$', '', v_str)
    return [float(x) for x in v_str.split('/') if x and x != '-']

def get_median(v_str):
    v = parse_floats(v_str)
    return statistics.median(v) if v else None

def get_first(v_str):
    v = parse_floats(v_str)
    return v[0] if v else None

def parse_lines(stream):
    b_data = {}
    s_data = {}
    has_b = False
    log_type = 'ALTA'

    for line in stream:
        line = line.strip()
        if line.startswith('MEAS-RING'):
            log_type = 'RING'
            continue
        parts = line.split()
        if len(parts) < 3: continue

        if parts[0] == 'B' and parts[1] == 'moe':
            has_b = True
            m_val = None
            v_val = None
            res = {}
            for p in parts[2:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    if k == 'm': m_val = v
                    elif k == 'v': v_val = v
                    else: res[k] = v
                elif p == '!INVALID':
                    res['INVALID'] = True
            if m_val and v_val:
                b_data.setdefault(m_val, {})[v_val] = res

        elif parts[0] == 'S' and parts[1] == 'moe':
            m_val = None
            res = {}
            prefix = ''
            for p in parts[2:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    if k == 'm': m_val = v
                    else: res[prefix + k] = v
                elif p in ('g1', 'g2', 'pdl'):
                    prefix = p + '_'
            if m_val:
                s_data[m_val] = res

    return b_data, s_data, has_b, log_type

def process_m(m_val, variants, s_data):
    # compute medians
    res = {}
    for v, data in variants.items():
        v_res = {}
        g1 = get_median(data.get('g1'))
        g2 = get_median(data.get('g2'))
        if g1 is not None: v_res['g1'] = g1
        if g2 is not None: v_res['g2'] = g2

        x_str = data.get('x', '')
        if '/' in x_str:
            xp = x_str.split('/')
            v_res['g1_x'] = get_median(xp[0])
            v_res['g2_x'] = get_median(xp[1]) if len(xp) > 1 else None
        else:
            v_res['g1_x'] = get_median(x_str)


        tb_parts = data.get('tb', '').split('/')
        if len(tb_parts) > 0 and tb_parts[0] and tb_parts[0] != '-': v_res['tb_g1'] = float(re.sub(r'[a-zA-Z%]+$', '', tb_parts[0]))
        if len(tb_parts) > 1 and tb_parts[1] and tb_parts[1] != '-': v_res['tb_g2'] = float(re.sub(r'[a-zA-Z%]+$', '', tb_parts[1]))

        v_res['bw'] = data.get('bw', '-')
        v_res['INVALID'] = data.get('INVALID', False)
        res[v] = v_res
    return res

def calculate_decisions(b_res_by_m, s_data):
    if not b_res_by_m: return None

    large_ms = [m for m in b_res_by_m.keys() if int(m) >= 32]

    decisions_per_m = {}

    for m_str in large_ms:
        variants = b_res_by_m[m_str]
        s_metrics = s_data.get(m_str, {})

        base = variants.get('base', {})
        nocomp = variants.get('nocomp', {})
        wstream = variants.get('wstream', {})
        v4 = variants.get('v4', {})

        base_g1 = base.get('g1')
        nocomp_ratio = None
        if base_g1 and nocomp.get('g1'):
            nocomp_ratio = nocomp['g1'] / base_g1

        best_gain = 0.0
        best_gain_var = None
        if base_g1:
            for v in ['lb3', 'kc', 'nb2', 'nb4', 'alt', 'altkc']:
                v_g1 = variants.get(v, {}).get('g1')
                if v_g1:
                    gain = (base_g1 / v_g1 - 1.0) * 100.0
                    if gain > best_gain:
                        best_gain = gain
                        best_gain_var = v

        wstream_tb = wstream.get('tb_g1')

        staging_share = None
        if s_metrics.get('g1_stage'):
            try: staging_share = float(s_metrics['g1_stage'].replace('%', ''))
            except ValueError: pass

        fixup_tail = 0.0
        for k in ['g1_wait', 'g1_fix', 'g1_epi']:
            if s_metrics.get(k):
                try: fixup_tail += float(s_metrics[k].replace('%', ''))
                except ValueError: pass

        pdl_g2t0 = None
        if s_metrics.get('pdl_g2t0'):
            try: pdl_g2t0 = float(s_metrics['pdl_g2t0'].replace('us', ''))
            except ValueError: pass

        pdl_g2end = None
        if s_metrics.get('pdl_g2end'):
            try: pdl_g2end = float(s_metrics['pdl_g2end'].replace('us', ''))
            except ValueError: pass

        verdict_parts = []

        if best_gain >= 25.0:
            verdict_parts.append(f"(i) Alt A = best combination ({best_gain:.1f}%)")
        elif nocomp_ratio is not None and nocomp_ratio >= 0.85 and best_gain < 15.0 and wstream_tb is not None and wstream_tb >= 3.2:
            verdict_parts.append(f"(ii) ring (C21b) (nocomp {nocomp_ratio:.2f}, gain {best_gain:.1f}%, wstream {wstream_tb:.1f})")
        elif nocomp_ratio is not None and nocomp_ratio <= 0.55:
            verdict_parts.append(f"(iii) Alt C scoping (nocomp {nocomp_ratio:.2f})")
        elif wstream_tb is not None and wstream_tb < 2.8:
            verdict_parts.append(f"(iv) repack first (wstream {wstream_tb:.1f})")

        if not verdict_parts:
            if best_gain > 0: verdict_parts.append(f"fallback: some gain ({best_gain:.1f}%)")
            else: verdict_parts.append("fallback: none")

        if staging_share is not None and staging_share > 25.0:
            verdict_parts.append(f"(v) NB into the build (staging {staging_share:.1f}%)")

        decisions_per_m[m_str] = {
            'nocomp_ratio': nocomp_ratio,
            'best_gain_pct': best_gain,
            'best_gain_var': best_gain_var,
            'wstream_tb': wstream_tb,
            'staging_share_pct': staging_share,
            'fixup_tail_pct': fixup_tail,
            'pdl_g2t0': pdl_g2t0,
            'pdl_g2end': pdl_g2end,
            'verdict': " ".join(verdict_parts),
        }

    # Best per M globally
    best_per_m = {}
    sum_best_g1 = 0
    sum_best_g2 = 0
    sum_v4_g1 = 0
    sum_v4_g2 = 0

    for m, m_vars in b_res_by_m.items():
        best_g1_v = None
        best_g1_val = 1e9
        best_g2_v = None
        best_g2_val = 1e9

        for v, d in m_vars.items():
            if not d.get('INVALID', False) and v not in TWINS:
                if d.get('g1') and d['g1'] < best_g1_val:
                    best_g1_val = d['g1']
                    best_g1_v = v
                if d.get('g2') and d['g2'] < best_g2_val:
                    best_g2_val = d['g2']
                    best_g2_v = v

        best_per_m[m] = {
            'g1_var': best_g1_v, 'g1_val': best_g1_val if best_g1_val < 1e9 else None,
            'g2_var': best_g2_v, 'g2_val': best_g2_val if best_g2_val < 1e9 else None
        }
        if best_g1_val < 1e9: sum_best_g1 += best_g1_val
        if best_g2_val < 1e9: sum_best_g2 += best_g2_val

        v4_d = m_vars.get('v4', {})
        if v4_d.get('g1'): sum_v4_g1 += v4_d['g1']
        if v4_d.get('g2'): sum_v4_g2 += v4_d['g2']

    shape_time_pred = {
        'best_sum': sum_best_g1 + sum_best_g2,
        'v4_sum': sum_v4_g1 + sum_v4_g2
    }

    return {
        'per_large_m': decisions_per_m,
        'best_per_m': best_per_m,
        'shape_time_pred': shape_time_pred
    }

def print_text_output(b_res_by_m, decisions):
    for m in sorted(b_res_by_m.keys(), key=lambda x: int(x), reverse=True):
        print(f"M={m}")
        for v, d in b_res_by_m[m].items():
            g1 = f"{d['g1']:.1f}" if d.get('g1') is not None else "-"
            g2 = f"{d['g2']:.1f}" if d.get('g2') is not None else "-"
            x = f"{d.get('g1_x', '-')}/{d.get('g2_x', '-')}"
            tbg1 = f"{d.get('tb_g1', '-')}"
            tbg2 = f"{d.get('tb_g2', '-')}"
            bw = d.get('bw', '-')
            inv = "INVALID" if d.get('INVALID') else ""
            print(f"  {v}: g1={g1} g2={g2} x={x} tb={tbg1}/{tbg2} bw={bw} {inv}")

    if decisions:
        for m_str in sorted(decisions.get('per_large_m', {}).keys(), key=lambda x: int(x), reverse=True):
            d = decisions['per_large_m'][m_str]
            print(f"\nDecisions (Large M={m_str}):")
            print(f"  NOCOMPUTE ratio: {d['nocomp_ratio']:.2f}" if d.get('nocomp_ratio') is not None else "  NOCOMPUTE ratio: -")
            print(f"  Best occupancy/stage-once gain: {d['best_gain_pct']:.1f}% ({d['best_gain_var']})")
            print(f"  WSTREAM TB/s: {d['wstream_tb']:.2f}" if d.get('wstream_tb') is not None else "  WSTREAM TB/s: -")
            print(f"  Staging share: {d['staging_share_pct']}%" if d.get('staging_share_pct') is not None else "  Staging share: -")
            print(f"  Fixup tail: {d['fixup_tail_pct']:.1f}%")
            print(f"  PDL g2t0: {d['pdl_g2t0']}" if d.get('pdl_g2t0') is not None else "  PDL g2t0: -")
            print(f"  PDL g2end: {d['pdl_g2end']}" if d.get('pdl_g2end') is not None else "  PDL g2end: -")
            print(f"  Verdict: {d['verdict']}")

        print("\nOverall Best variants per M:")
        for m, best in decisions['best_per_m'].items():
            print(f"    M={m}: g1={best['g1_var']} ({best['g1_val']}), g2={best['g2_var']} ({best['g2_val']})")
        print(f"  Predicted shape time: best={decisions['shape_time_pred']['best_sum']:.1f} vs v4={decisions['shape_time_pred']['v4_sum']:.1f}")


def calculate_decisions_ring(b_res_by_m, s_data):
    if not b_res_by_m: return None

    large_ms = [int(m) for m in b_res_by_m.keys() if int(m) >= 32]
    if not large_ms: return None
    m_max = str(max(large_ms))

    variants = b_res_by_m[m_max]
    s_metrics = s_data.get(m_max, {})

    best_v6_var = None
    best_v6_x = -1.0
    for v, d in variants.items():
        if v in ('v5', 'wstream') or d.get('INVALID', False): continue
        if d.get('g1_x') is not None and d['g1_x'] > best_v6_x:
            best_v6_x = d['g1_x']
            best_v6_var = v

    gain = (best_v6_x - 1.0) * 100.0 if best_v6_var else 0.0
    promote = bool(best_v6_var and gain >= 12.0 and variants[best_v6_var].get('bw') == 'OK')

    ring_x, rb3_x = variants.get('ring', {}).get('g1_x'), variants.get('rb3', {}).get('g1_x')
    stream_share = float(s_metrics['g1_stream'].replace('%', '')) if s_metrics.get('g1_stream') else None
    duty_cycle_residual = False
    if ring_x and rb3_x and stream_share is not None and stream_share > 60.0 and abs(ring_x - rb3_x) / max(ring_x, rb3_x) <= 0.05 and ring_x < 1.12 and rb3_x < 1.12:
        duty_cycle_residual = True

    ring_def = "register default"
    m256_vars = b_res_by_m.get('256', {})
    r256_x = m256_vars.get('ring', {}).get('g1_x')
    rb256_x = m256_vars.get('rb3', {}).get('g1_x')
    if r256_x and rb256_x and r256_x > rb256_x * 1.05:
        ring_def = "ring default"

    harness = any(d.get('INVALID') or (d.get('bw') not in ('-', 'OK')) for v, d in variants.items() if not ("nb" in v or "zflat" in v or "pdloff" in v))

    best_per_m = {}
    sum_best_g1, sum_best_g2, sum_v5_g1, sum_v5_g2 = 0, 0, 0, 0
    for m, m_vars in b_res_by_m.items():
        best_g1_v, best_g1_val, best_g2_v, best_g2_val = None, 1e9, None, 1e9
        for v, d in m_vars.items():
            if not d.get('INVALID', False) and v not in TWINS:
                if d.get('g1') and d['g1'] < best_g1_val: best_g1_val, best_g1_v = d['g1'], v
                if d.get('g2') and d['g2'] < best_g2_val: best_g2_val, best_g2_v = d['g2'], v
        best_per_m[m] = {
            'g1_var': best_g1_v, 'g1_val': best_g1_val if best_g1_val < 1e9 else None,
            'g2_var': best_g2_v, 'g2_val': best_g2_val if best_g2_val < 1e9 else None,
            'ring_vs_reg': m_vars.get('ring', {}).get('g1_x', 0) / m_vars.get('rb3', m_vars.get('rb7', {})).get('g1_x', 1) if m_vars.get('rb3', m_vars.get('rb7', {})).get('g1_x') else None,
            'ring_vs_ring82': m_vars.get('ring', {}).get('g1_x', 0) / m_vars.get('ring82', m_vars.get('ring81', {})).get('g1_x', 1) if m_vars.get('ring82', m_vars.get('ring81', {})).get('g1_x') else None
        }
        if best_g1_val < 1e9: sum_best_g1 += best_g1_val
        if best_g2_val < 1e9: sum_best_g2 += best_g2_val
        v5_d = m_vars.get('v5', {})
        if v5_d.get('g1'): sum_v5_g1 += v5_d['g1']
        if v5_d.get('g2'): sum_v5_g2 += v5_d['g2']

    return {
        'best_v6_var': best_v6_var,
        'gain': gain,
        'promote': promote,
        'duty_cycle_residual': duty_cycle_residual,
        'ring_default': ring_def,
        'harness': harness,
        'best_per_m': best_per_m,
        'shape_time_pred': {'best_sum': sum_best_g1 + sum_best_g2, 'v5_sum': sum_v5_g1 + sum_v5_g2}
    }

def print_text_output_ring(b_res_by_m, decisions):
    for m in sorted(b_res_by_m.keys(), key=lambda x: int(x), reverse=True):
        print(f"M={m}")
        for v, d in b_res_by_m[m].items():
            g1 = f"{d['g1']:.1f}" if d.get('g1') is not None else "-"
            g2 = f"{d['g2']:.1f}" if d.get('g2') is not None else "-"
            x = f"{d.get('g1_x', '-')}/{d.get('g2_x', '-')}"
            tbg1 = f"{d.get('tb_g1', '-')}"
            tbg2 = f"{d.get('tb_g2', '-')}"
            bw = d.get('bw', '-')
            inv = "INVALID" if d.get('INVALID') else ""
            print(f"  {v}: g1={g1} g2={g2} x={x} tb={tbg1}/{tbg2} bw={bw} {inv}")

    if decisions:
        print(f"\nDecisions:\n  (a) best v6 variant: {decisions['best_v6_var']} (gain {decisions['gain']:.1f}%)" + (" -> promote" if decisions['promote'] else ""))
        if decisions['duty_cycle_residual']: print(f"  (b) duty cycle residual")
        print(f"  (c) {decisions['ring_default']}")
        if decisions['harness']: print(f"  (d) harness")
        print("\nOverall Best variants per M:")
        for m, best in decisions['best_per_m'].items():
            r1 = f"ring/reg={best['ring_vs_reg']:.2f}" if best.get('ring_vs_reg') else "ring/reg=-"
            r2 = f"ring/r82={best['ring_vs_ring82']:.2f}" if best.get('ring_vs_ring82') else "ring/r82=-"
            print(f"    M={m}: g1={best['g1_var']} ({best['g1_val']}), g2={best['g2_var']} ({best['g2_val']}) [{r1}, {r2}]")
        print(f"  Predicted shape time: best={decisions['shape_time_pred']['best_sum']:.1f} vs v5={decisions['shape_time_pred']['v5_sum']:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_file', nargs='?', default='-')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--m', dest='filter_m')
    args = parser.parse_args()

    if args.log_file == '-':
        f = sys.stdin
    else:
        try:
            f = open(args.log_file, 'r')
        except FileNotFoundError:
            print(f"File not found: {args.log_file}")
            sys.exit(1)

    b_data, s_data, has_b, log_type = parse_lines(f)
    if args.log_file != '-':
        f.close()

    if not has_b:
        sys.exit(2)

    if args.filter_m:
        b_data = {m: v for m, v in b_data.items() if m == args.filter_m}
        s_data = {m: v for m, v in s_data.items() if m == args.filter_m}

    b_res = {m: process_m(m, b_data[m], s_data) for m in b_data}
    dec = calculate_decisions_ring(b_res, s_data) if log_type == 'RING' else calculate_decisions(b_res, s_data)
    if args.json:
        print(json.dumps({'variants': b_res, 'decisions': dec}, indent=2))
    else:
        print_text_output_ring(b_res, dec) if log_type == 'RING' else print_text_output(b_res, dec)

if __name__ == '__main__':
    main()
