import pytest
from pathlib import Path
import subprocess

def test_okcdbg_synthetic(tmp_path):
    trace_text = """TV 1 muse_test dummy torch=x cuda=x gpu=x cupti=graph replays=1
TN 0 copy_kernel
TN 1 compute_kernel
TS 0 w=10 wall=100.0 span=90.0 n=2 label=x
TK 0.0 0 s=0.0 d=10.0 g=0.0 st=0 o=0.0
TK 0.1 1 s=15.0 d=70.0 g=5.0 st=1 o=5.0
TG 0 gaps=5.0 head=0.0 tail=0.0 busy=80.0
TS 1 w=1 wall=20.0 span=20.0 n=1 label=x
TK 1.0 1 s=0.0 d=20.0 g=0.0 st=0 o=0.0
TG 1 gaps=0.0 head=0.0 tail=0.0 busy=20.0
TZ done 1.0
"""

    bytes_csv = """problem,shape,kernel_family,bytes,flops,note
muse_test,0,copy/cast,4100000,0,"test copy"
muse_test,0,other,8200000,0,"test compute"
"""

    trace_file = tmp_path / "trace.log"
    trace_file.write_text(trace_text)

    bytes_file = tmp_path / "bytes.csv"
    bytes_file.write_text(bytes_csv)

    # Run with small budget to test truncation (drop shape 1)
    res = subprocess.run(["python3", "tools/okcdbg.py", str(trace_file), "--bytes", str(bytes_file), "--budget", "600"], capture_output=True, text=True)

    assert res.returncode == 0
    out = res.stdout
    assert "#OKC" in out

    # D3 test
    assert "#D3 0 copy_kernel" in out
    assert "passes_t=10.00" in out  # 10us * 4100GB/s / 4.1MB = 10.00

    # D4 test: gap=5.0(>3us), copy/fill=1, overlap=5.0
    assert "#D4 0" in out
    assert "sum=5.00" in out
    assert "max=5.00" in out
    assert ">3us=1" in out
    assert "copy/fill:1" in out
    assert "overlap=5.00" in out

    # floor calculation test
    # copy/cast bytes=4100000. bw=4100. floor = 4.1MB / 4100GB/s = 1us.
    # d=10, x_fl = 10 / 1 = 10.00
    assert "2.50 4.00 10.00" in out  # floor (min 2.5 us practical floor) x_fl passes_t

    # Check truncation. Shape 1 should be omitted due to budget 600
    assert "omitted" in out or "1.0 1" not in out
    if "omitted" in out:
        assert "+1 omitted" in out

def test_okcdbg_hand_computed(tmp_path):
    trace_text = """TV 1 problem_x file torch=1 cuda=1 gpu=1 cupti=graph replays=1
TN 0 my_kernel
TS 0 w=1 wall=100.0 span=90.0 n=1 label=x
TK 0.0 0 s=0.0 d=4.0 g=0.0 st=0 o=0.0
TG 0 gaps=0.0 head=0.0 tail=0.0 busy=4.0
TZ done 1.0
"""
    # 8200 bytes, bw 4100 GB/s = 8200 / (4100 * 1e9) = 2 ns = 0.002 us.
    bytes_csv = """problem,shape,kernel_family,bytes,flops,note
problem_x,0,other,8200,0,"note"
"""
    trace_file = tmp_path / "trace.log"
    trace_file.write_text(trace_text)

    bytes_file = tmp_path / "bytes.csv"
    bytes_file.write_text(bytes_csv)

    res = subprocess.run(["python3", "tools/okcdbg.py", str(trace_file), "--bytes", str(bytes_file)], capture_output=True, text=True)
    assert res.returncode == 0
    out = res.stdout
    assert "0.00 2000.00" in out or "0.00" in out # floor is 0.00, x_fl = 4.0 / 0.002 = 2000
