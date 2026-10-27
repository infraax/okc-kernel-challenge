"""sonnet_v3b = sonnet_v3 with the judge-tuned SMALLM_CFG table only (ks=2 configs need ~67 KB smem: H200-only, the
2060 dev card caps at 64 KB, so v3's full-size dev tests cannot run v3b's table; the H200 sweep 7e8e3866 validated them)."""
import difflib
from pathlib import Path

D = Path(__file__).resolve().parent.parent / "kernels" / "muse_moe_w4a16"


def test_v3b_differs_only_in_cfg_table():
    a = (D / "sonnet_v3.py").read_text().splitlines()
    b = (D / "sonnet_v3b.py").read_text().splitlines()
    changed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert changed and all(l[1:].startswith(("    (", "# Defaults", "# Tuned")) for l in changed), changed


def test_v3b_cfg_buckets():
    ns = {}
    src = (D / "sonnet_v3b.py").read_text()
    start = src.index("SMALLM_CFG = (")
    exec(src[start:src.index("\n)", start) + 2], ns)
    rows = ns["SMALLM_CFG"]
    assert [r[0] for r in rows] == [1, 2, 4, 8, 12, 16]
    for _, nw1, ks1, nz1, nw2, ks2, nz2 in rows:
        assert nw1 in (2, 4, 8) and nw2 in (2, 4, 8) and ks1 >= 1 and ks2 >= 1 and nz1 >= 1 and nz2 >= 1
