import pytest
import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tools')))
import parse_meas_alta

FIXTURE_RING_CASE_A = """
MEAS-RING gpu=mock cal=123us inst=12 rs=a/b/cx2
I moe m=256 seeds=11/12/13
B moe m=256 v=v5 | g1=100/100/100 g2=200/200/200 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=256 v=ring | g1=80/80/80 g2=180/180/180 | tb=1.2/1.1 ach=55/55% x=1.25/1.11 bw=OK
B moe m=256 v=rb3 | g1=85/85/85 g2=180/180/180 | tb=1.2/1.1 ach=55/55% x=1.17/1.11 bw=OK
B moe m=256 v=ring82 | g1=90/90/90 g2=180/180/180 | tb=1.1/1.1 ach=55/55% x=1.11/1.11 bw=OK
B moe m=256 v=wstream | g1=100/100/100 g2=200/200/200 | tb=3.3/3.3 ach=80/80% x=1.0/1.0 bw=OK
S moe m=256 g1 cta=1.0us item=1.0us span=1.0 stage=30% stream=65% wait=10% fix=3% epi=3% bw/cta=3.3GB/s occ=1/1.0 | g2 cta=1.0 | pdl g2t0=-1.0 g2t1=2.0 g2end=3.0
I moe m=8 seeds=11/12/13
B moe m=8 v=v5 | g1=10/10/10 g2=20/20/20 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=8 v=ring | g1=8/8/8 g2=15/15/15 | tb=1.2/1.2 ach=60/60% x=1.25/1.33 bw=OK
MEAS-RING done 1.2s [skipped: ...]
"""

FIXTURE_RING_CASE_B = """
MEAS-RING gpu=mock cal=123us inst=12 rs=a/b/cx2
I moe m=256 seeds=11/12/13
B moe m=256 v=v5 | g1=100/100/100 g2=200/200/200 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=256 v=ring | g1=90/90/90 g2=180/180/180 | tb=1.1/1.1 ach=55/55% x=1.11/1.11 bw=OK
B moe m=256 v=rb3 | g1=91/91/91 g2=180/180/180 | tb=1.1/1.1 ach=55/55% x=1.09/1.11 bw=OK
B moe m=256 v=ring82 | g1=95/95/95 g2=180/180/180 | tb=1.1/1.1 ach=55/55% x=1.05/1.11 bw=OK
B moe m=256 v=wstream | g1=100/100/100 g2=200/200/200 | tb=3.3/3.3 ach=80/80% x=1.0/1.0 bw=OK
S moe m=256 g1 cta=1.0us item=1.0us span=1.0 stage=30% stream=65% wait=10% fix=3% epi=3% bw/cta=3.3GB/s occ=1/1.0 | g2 cta=1.0 | pdl g2t0=-1.0 g2t1=2.0 g2end=3.0
I moe m=8 seeds=11/12/13
B moe m=8 v=v5 | g1=10/10/10 g2=20/20/20 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=8 v=ring | g1=9/9/9 g2=15/15/15 | tb=1.1/1.1 ach=60/60% x=1.11/1.33 bw=OK
MEAS-RING done 1.2s [skipped: ...]
"""

FIXTURE_CASE1 = """
MEAS-ALTA gpu=mock
I moe m=256 seeds=11/12/13
B moe m=256 v=v4 | g1=100/100/100 g2=200/200/200 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=256 v=base | g1=100/100/100 g2=200/200/200 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=256 v=nocomp | g1=90/90/90 g2=180/180/180 | tb=1.1/1.1 ach=55/55% x=1.1/1.1 bw=OK
B moe m=256 v=alt | g1=70/70/70 g2=150/150/150 | tb=1.5/1.5 ach=70/70% x=1.4/1.3 bw=OK
B moe m=256 v=wstream | g1=100/100/100 g2=200/200/200 | tb=3.3/3.3 ach=80/80% x=1.0/1.0 bw=OK
S moe m=256 g1 cta=1.0us item=1.0us span=1.0 stage=30% stream=10% wait=10% fix=3% epi=3% bw/cta=3.3GB/s occ=1/1.0 | g2 cta=1.0 | pdl g2t0=-1.0 g2t1=2.0 g2end=3.0
I moe m=8 seeds=11/12/13
B moe m=8 v=v4 | g1=10/10/10 g2=20/20/20 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=8 v=base | g1=10/10/10 g2=20/20/20 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=8 v=alt | g1=8/8/8 g2=15/15/15 | tb=1.2/1.2 ach=60/60% x=1.2/1.2 bw=OK
MEAS-ALTA done
"""

FIXTURE_CASE2 = """
B moe m=256 v=base | g1=100/100/100 g2=200/200/200 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=256 v=nocomp | g1=90/90/90 g2=180/180/180 | tb=1.1/1.1 ach=55/55% x=1.1/1.1 bw=OK
B moe m=256 v=alt | g1=95/95/95 g2=150/150/150 | tb=1.5/1.5 ach=70/70% x=1.0/1.3 bw=OK
B moe m=256 v=wstream | g1=100/100/100 g2=200/200/200 | tb=3.3/3.3 ach=80/80% x=1.0/1.0 bw=OK
S moe m=256 g1 cta=1.0us item=1.0us span=1.0 stage=20% stream=10% wait=10% fix=3% epi=3% bw/cta=3.3GB/s occ=1/1.0 | g2 cta=1.0 | pdl g2t0=-1.0 g2t1=2.0 g2end=3.0
"""

FIXTURE_PARTIAL = """
B moe m=256 v=base | g1=100/100/100 g2=200/200/200 | tb=1.0/1.0 ach=50/50% x=1.0/1.0 bw=OK
B moe m=256 v=nocomp | g1=90/90/90 g2=180
"""

def test_medians_and_ratios_case1():
    stream = io.StringIO(FIXTURE_CASE1)
    b_data, s_data, has_b, _ = parse_meas_alta.parse_lines(stream)
    assert has_b
    b_res = {m: parse_meas_alta.process_m(m, b_data[m], s_data) for m in b_data}

    assert b_res['256']['alt']['g1'] == 70.0
    assert b_res['256']['nocomp']['g1'] == 90.0

    dec = parse_meas_alta.calculate_decisions(b_res, s_data)

    # nocomp ratio = 90/100 = 0.90
    assert abs(dec['per_large_m']['256']['nocomp_ratio'] - 0.9) < 1e-5

    # best gain for alt = 100/70 - 1 = 0.4285 = 42.85%
    assert abs(dec['per_large_m']['256']['best_gain_pct'] - 42.857) < 1e-2
    assert dec['per_large_m']['256']['best_gain_var'] == 'alt'

    # verdict checks
    assert "(i) Alt A = best combination" in dec['per_large_m']['256']['verdict']
    assert "(v) NB into the build" in dec['per_large_m']['256']['verdict']  # stage=30% > 25%

    # fixup tail = 10 + 3 + 3 = 16%
    assert abs(dec['per_large_m']['256']['fixup_tail_pct'] - 16.0) < 1e-5

def test_medians_and_ratios_case2():
    stream = io.StringIO(FIXTURE_CASE2)
    b_data, s_data, has_b, _ = parse_meas_alta.parse_lines(stream)
    assert has_b
    b_res = {m: parse_meas_alta.process_m(m, b_data[m], s_data) for m in b_data}
    dec = parse_meas_alta.calculate_decisions(b_res, s_data)

    # nocomp ratio = 90/100 = 0.90 (>=0.85)
    # alt gain = 100/95 - 1 = 5.26% (<15%)
    # wstream tb = 3.3 (>=3.2)
    assert "(ii) ring (C21b)" in dec['per_large_m']['256']['verdict']

def test_partial_line():
    stream = io.StringIO(FIXTURE_PARTIAL)
    b_data, s_data, has_b, _ = parse_meas_alta.parse_lines(stream)
    assert has_b
    assert '256' in b_data
    assert 'base' in b_data['256']
    assert 'nocomp' in b_data['256']
    assert b_data['256']['nocomp']['g2'] == '180'

    b_res = {m: parse_meas_alta.process_m(m, b_data[m], s_data) for m in b_data}
    assert b_res['256']['nocomp']['g2'] == 180.0

def test_exit_code_empty(monkeypatch):
    import sys
    monkeypatch.setattr('sys.argv', ['parse_meas_alta.py', '-'])
    monkeypatch.setattr('sys.stdin', io.StringIO("I moe m=256 seeds=11/12/13\n"))
    with pytest.raises(SystemExit) as e:
        parse_meas_alta.main()
    assert e.value.code == 2

def test_ring_case_a():
    stream = io.StringIO(FIXTURE_RING_CASE_A)
    b_data, s_data, has_b, log_type = parse_meas_alta.parse_lines(stream)
    assert has_b
    assert log_type == 'RING'

    b_res = {m: parse_meas_alta.process_m(m, b_data[m], s_data) for m in b_data}
    dec = parse_meas_alta.calculate_decisions_ring(b_res, s_data)

    assert dec['best_v6_var'] == 'ring'
    assert dec['gain'] == 25.0
    assert dec['promote'] == True
    assert dec['duty_cycle_residual'] == False
    assert dec['ring_default'] == 'ring default'

def test_ring_case_b():
    stream = io.StringIO(FIXTURE_RING_CASE_B)
    b_data, s_data, has_b, log_type = parse_meas_alta.parse_lines(stream)
    assert has_b
    assert log_type == 'RING'

    b_res = {m: parse_meas_alta.process_m(m, b_data[m], s_data) for m in b_data}
    dec = parse_meas_alta.calculate_decisions_ring(b_res, s_data)

    assert dec['best_v6_var'] == 'ring'
    assert abs(dec['gain'] - 11.0) < 1e-5
    assert dec['promote'] == False
    assert dec['duty_cycle_residual'] == True
    assert dec['ring_default'] == 'register default'
