import csv
import os
import subprocess
import tempfile
from pathlib import Path

def test_trace_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        log1 = tmp_path / "log1.txt"
        bytes_csv = tmp_path / "bytes.csv"
        out_dir = tmp_path / "report"

        # Construct log 1 (Problem A): truncated line, TE error line, dropped shape (not dropped explicitly but TE means error shape)
        # 2 streams with overlap.
        with open(log1, "w") as f:
            f.write("Some random garbage that is truncated... \n") # truncated line
            f.write("TV 1 probA file torch=2 cuda=12 gpu=H200 cupti=ok replays=60\n")
            f.write("TN 0 _attn_kernel\n")
            f.write("TN 1 _reduce_kernel\n")
            f.write("TS 0 w=10.0 wall=20.0 span=15.0 n=3 label=batch=1\n")
            # Stream 1
            f.write("TK 0.0 0 s=0.0 d=5.0 g=2.0 st=1 o=0.0\n")
            # Stream 2 overlaps with Stream 1
            f.write("TK 0.1 1 s=2.0 d=4.0 g=0.0 st=2 o=2.0\n")
            # Stream 1 again
            f.write("TK 0.2 0 s=6.0 d=3.0 g=1.5 st=1 o=0.0\n")
            f.write("TG 0 gaps=3.5 head=1.0 tail=0.5 busy=11.5\n")

            # Dropped shape (e.g. no TG)
            f.write("TS 1 w=5.0 wall=10.0 span=8.0 n=1 label=batch=2\n")
            f.write("TK 1.0 1 s=0.0 d=4.0 g=0.0 st=1 o=0.0\n")
            # missing TG

            # TE shape
            f.write("TS 2 w=1.0 wall=0.0 span=0.0 n=0 label=batch=3\n")
            f.write("TE 2 Some error occurred\n")

        with open(bytes_csv, "w") as f:
            f.write("problem,shape,kernel_family,bytes,flops\n")
            f.write("probA,0,attention,4.0e12,0.0\n")

        # Run script
        script_path = Path(__file__).parent.parent / "tools" / "trace_report.py"
        subprocess.run(["python3", str(script_path), str(log1), "--bytes", str(bytes_csv), "--out", str(out_dir)], check=True)

        # 1. Assert HTML parses as XML-ish and count <rect>
        html_file = out_dir / "timeline_probA.html"
        assert html_file.exists(), "HTML file not created"

        html_content = html_file.read_text()

        svg_start = html_content.find("<svg")
        assert svg_start != -1, "No SVG found"

        # we expect kernels + gaps >= 1 us
        # Shape 0: 3 kernels, 1 gap (1.5us), gap 2.0 starts at -2.0, since x=80, gap_w=2.0*scale, gap_x = 80 - gap_w.
        # Wait, the script logic:
        # gap_w = k["gap"] * scale
        # gap_x = x - gap_w
        # if gap_x >= 80:
        #    ... add gap rect ...
        # For kernel 0.0, start=0.0, so x = 80. gap=2.0, gap_w > 0, gap_x = 80 - gap_w < 80. So this gap is NOT drawn.
        # For kernel 0.1, start=2.0, gap=0.0. No gap.
        # For kernel 0.2, start=6.0, gap=1.5. x = 80 + 6*scale. gap_w = 1.5*scale. gap_x = 80 + 4.5*scale >= 80. Drawn!
        # So we have 1 gap drawn for Shape 0.
        # Total rects:
        # Shape 0: 3 kernels + 1 gap = 4
        # Shape 1: 1 kernel = 1
        # Total: 5.
        rect_count = html_content.count("<rect")
        assert rect_count == 5, f"Expected 5 rects, found {rect_count}"

        # 2. Assert SUMMARY.md ranks largest item first
        summary_file = out_dir / "SUMMARY.md"
        assert summary_file.exists(), "SUMMARY.md not created"

        summary_content = summary_file.read_text()
        assert "12500000.0%" in summary_content, "Bytes efficiency math is incorrect"

        # 3. Assert items.csv row count
        items_csv_file = out_dir / "items.csv"
        assert items_csv_file.exists(), "items.csv not created"
        with open(items_csv_file, "r") as f:
            lines = f.readlines()

        # Rows:
        # Header (1)
        # Shape 0:
        #   Kernels: 0.0, 0.1, 0.2 (3)
        #   Gaps: 0.0 (gap=2), 0.2 (gap=1.5) (2). Wait, CSV generates all gaps > 0. (0.1 has gap=0, no gap row generated)
        #   Head: (1)
        #   Tail: (1)
        # Shape 1:
        #   Kernels: 1.0 (1)
        #   (No TG -> no head/tail)
        # Shape 2:
        #   0 kernels, 0 TG -> 0 rows
        # Total expected: 1 + 3 + 2 + 1 + 1 + 1 = 9 rows.
        assert len(lines) == 9, f"Expected 9 rows in items.csv, found {len(lines)}"

        assert "| 1 | kernel | 0 | _attn_kernel" in summary_content, "Largest item not ranked first"

if __name__ == "__main__":
    test_trace_report()
