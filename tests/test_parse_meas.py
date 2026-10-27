"""tools/parse_meas.py on a synthetic log (own tiny config table, independent of measure_v3's real one)."""
import base64
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import parse_meas  # noqa: E402

FAKE_MODULE = '''
import zlib
# ---- CFG BEGIN (synthetic) ----
import zlib
def all_shapes():
    return [("pooler", 1), ("router", 8)]
def shape_id(layer, m):
    return f"{layer}.m{m}"
def shape_cfgs(layer, m):
    if layer == "pooler":
        return [("A16.8.512.1.4.3", "A", 16, 8, 512, 1, 4, 3), ("B16.8.512.1.4.3", "B", 16, 8, 512, 1, 4, 3),
                ("A16.16.512.1.4.4", "A", 16, 16, 512, 1, 4, 4), ("A16.32.512.1.8.4", "A", 16, 32, 512, 1, 8, 4)]
    return [("F16.16.64.8.4.3", "F", 16, 16, 64, 8, 4, 3), ("Z16.16.64.8.4.3", "Z", 16, 16, 64, 8, 4, 3)]
def cfg_sig():
    names = [c[0] for l, m in all_shapes() for c in shape_cfgs(l, m)]
    return len(names), zlib.crc32("\\n".join(names).encode()) & 0xFFFFFFFF
# ---- CFG END ----
'''


def t64(vals):
    return base64.b64encode(struct.pack(f"<{len(vals)}H", *vals)).decode()


@pytest.fixture()
def mod(tmp_path):
    p = tmp_path / "fake_measure.py"
    p.write_text(FAKE_MODULE)
    ns = parse_meas.load_cfg(p)
    return p, ns


def make_log(ns, sig_ok=True, cut_t64=False):
    n, crc = ns["cfg_sig"]()
    if not sig_ok:
        crc ^= 1
    lines = [
        "MEAS3 start gpu=NVIDIA H200 triton=3.8.0",
        "CFG name=<fam>...",
        f"CFGSIG n={n} crc={crc:08x} shapes=2",
        "TMA smoke: ok err=1e-4",
        "FAIL B x2 RuntimeError: nope",
    ]
    if not cut_t64:
        lines += [f"T64 pooler.m1 {t64([1895, 65535, 1941, 0])}", f"T64 router.m8 {t64([1114, 65534])}"]
    lines += [
        "MEAS3 done in 150.0s",
        "B pooler m=1 base=30.62 | top-5 A16.8.512.1.4.3:18.96 A16.16.512.1.4.4:19.40 | ok=2 fail=1 bad=0 skip=1",
        "B router m=8 base=21.28 | top-5 F16.16.64.8.4.3:11.14 | ok=1 fail=0 bad=1",
    ]
    return "\n".join(lines) + "\n"


def test_parse_t64_and_b_lines(mod):
    p, ns = mod
    d = parse_meas.parse_log(make_log(ns), p)
    assert d["pooler.m1"] == {"A16.8.512.1.4.3": 18.96, "A16.16.512.1.4.4": 19.40, "base": 30.62}
    # 65535 (fail), 0 (not run) and 65534 (bad) are left out; the finals of the B line override the screen value
    assert d["router.m8"] == {"F16.16.64.8.4.3": 11.14, "base": 21.28}
    meta = parse_meas.parse_full(make_log(ns), p)["meta"]
    assert meta["counts"]["pooler.m1"] == dict(ok=2, fail=1, bad=0, skip=1)
    assert meta["counts"]["router.m8"] == dict(ok=1, fail=0, bad=1, skip=0) and meta["sig_ok"] is True


def test_screen_value_kept_where_no_final(mod):
    p, ns = mod
    log = make_log(ns).replace(" A16.16.512.1.4.4:19.40", "")  # not in the top 5: only the T64 screen value remains
    d = parse_meas.parse_log(log, p)
    assert d["pooler.m1"]["A16.16.512.1.4.4"] == 19.41 and d["pooler.m1"]["A16.8.512.1.4.3"] == 18.96


def test_best_and_delta(mod):
    p, ns = mod
    rows = parse_meas.best_per_shape(parse_meas.parse_log(make_log(ns), p), per_family=True)
    r = rows[0]
    assert r["shape"] == "pooler.m1" and r["cfg"] == "A16.8.512.1.4.3" and r["us"] == 18.96
    assert r["delta_us"] == pytest.approx(18.96 - 30.62) and r["delta_pct"] == pytest.approx(100 * (18.96 / 30.62 - 1))
    assert set(r["families"]) == {"A"}


def test_cut_tail_falls_back_to_b_lines(mod):
    p, ns = mod
    d = parse_meas.parse_log(make_log(ns, cut_t64=True), p)
    assert d["pooler.m1"] == {"A16.8.512.1.4.3": 18.96, "A16.16.512.1.4.4": 19.40, "base": 30.62}


def test_signature_mismatch_exits(mod):
    p, ns = mod
    with pytest.raises(SystemExit):
        parse_meas.parse_log(make_log(ns, sig_ok=False), p)
    assert parse_meas.parse_full(make_log(ns, sig_ok=False), p, force=True)["meta"]["sig_ok"] is False


def test_cli_json_and_best(mod, tmp_path, capsys):
    p, ns = mod
    log = tmp_path / "x.log"
    log.write_text(make_log(ns))
    assert parse_meas.main([str(log), "--module", str(p)]) == 0
    assert json.loads(capsys.readouterr().out)["router.m8"]["base"] == 21.28
    assert parse_meas.main([str(log), "--module", str(p), "--best", "--per-family"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("pooler.m1") and "best=A16.8.512.1.4.3 18.96" in out[0] and "delta=-11.66us (-38.1%)" in out[0]
    assert out[1].startswith("router.m8") and "F:F16.16.64.8.4.3" not in out[1]


def test_okc_status_json_wrapper(mod):
    p, ns = mod
    doc = json.dumps({"job": "x", "log_tail": make_log(ns)})
    assert parse_meas.parse_log(doc, p)["pooler.m1"]["base"] == 30.62


def test_length_mismatch_exits(mod):
    p, ns = mod
    bad = make_log(ns).replace(t64([1895, 65535, 1941, 0]), t64([1, 2, 3]))
    with pytest.raises(SystemExit):
        parse_meas.parse_log(bad, p)


def test_clamped_values_are_dropped(mod):
    p, ns = mod
    bad = make_log(ns).replace(t64([1895, 65535, 1941, 0]), t64([65533, 65535, 1941, 65533]))
    d = parse_meas.parse_log(bad, p)
    assert "A16.32.512.1.8.4" not in d["pooler.m1"] and d["pooler.m1"]["A16.8.512.1.4.3"] == 18.96  # B-line final wins
    d2 = parse_meas.parse_log(bad.replace(" A16.8.512.1.4.3:18.96", ""), p)
    assert "A16.8.512.1.4.3" not in d2["pooler.m1"]
