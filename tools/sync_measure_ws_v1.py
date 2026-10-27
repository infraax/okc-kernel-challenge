#!/usr/bin/env python3
"""Re-sync kernels/muse_moe_w4a16/measure_ws_v1.py with its sources after sonnet_v5.py / sonnet_v7.py changed (C30).

measure_ws_v1.py = [its own header + docstring] + [v5's Python region, from `import re` to v5's entry points] + [sonnet_v7's MOE7 block] + [its own MEASUREMENT section
and prepare()] + [v5's kernel_fn]. Only the three copied pieces are replaced; the header, the MEASUREMENT section and prepare() are kept as they are in the file.

    .venv/bin/python tools/sync_measure_ws_v1.py          # rewrite in place (prints 'unchanged' if already in sync)
    .venv/bin/python tools/sync_measure_ws_v1.py --check  # exit 1 if out of sync
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "kernels" / "muse_moe_w4a16"
M0 = "# ============================================================================================ MOE7 BEGIN"
M1 = "# ============================================================================================ MOE7 END"
MEAS = "# ============================================================================================ MEASUREMENT"
ENTRY = "# ============================================================================================ entry points"


def build(meas_text):
    v5, v7 = (K / "sonnet_v5.py").read_text(), (K / "sonnet_v7.py").read_text()
    head = meas_text[:meas_text.index("\nimport re\n") + 1]
    region = v5[v5.index("import re\n"):v5.index(ENTRY)][:-1]
    block = v7[v7.index(M0):v7.index(M1) + len(M1)]
    tail = meas_text[meas_text.index(MEAS):meas_text.rindex("def kernel_fn")]
    return head + region + "\n\n" + block + "\n\n\n" + tail + v5[v5.rindex("def kernel_fn"):]


def main():
    path = K / "measure_ws_v1.py"
    old = path.read_text()
    new = build(old)
    if new == old:
        print("unchanged")
        return 0
    if "--check" in sys.argv:
        print("OUT OF SYNC")
        return 1
    path.write_text(new)
    print("rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
