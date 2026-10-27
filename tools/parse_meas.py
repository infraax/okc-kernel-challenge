#!/usr/bin/env python3
"""Parse the log of a measure_v3 run (`B` lines + `T64` blocks) into JSON {shape: {cfg: us}}.

    tools/parse_meas.py LOG [--module kernels/muse_bf16_proj/measure_v3.py] [--best] [--per-family] [--force]

LOG is the saved .log, an `okc status --json` document (its `log_tail` is used) or `-` for stdin. A `T64` block is
the base64 of uint16 little endian values (µs x 100) in the order of `shape_cfgs(layer, m)`; that table lives in the
measure file (between the `CFG BEGIN` / `CFG END` markers) and is executed from there, `CFGSIG n= crc=` in the log must
match it (else the log came from another version of the file: exit 2, unless --force). Value 0 = not run, 65535 =
fail, 65534 = bad, 65533 = clamped (>= 655.33 us); all four are left out of the JSON. `base` is the cuBLAS/SGLang baseline measured the same way
(from the `B` line); the top-5 of a `B` line (51-rep finals) fill in if the T64 block was cut off the log tail.
--best prints, per shape, the best config and its delta vs base (negative = faster).
"""
import argparse
import base64
import json
import re
import struct
import sys
from pathlib import Path

DEFAULT_MODULE = Path(__file__).resolve().parent.parent / "kernels" / "muse_bf16_proj" / "measure_v3.py"
_B_RE = re.compile(r"^B (\S+) m=(\d+) base=(\S+) \| top-5 (.*?) \| ok=(\d+) fail=(\d+) bad=(\d+)(?: skip=(\d+))?\s*$")
_T_RE = re.compile(r"^T64 (\S+) (\S*)\s*$")
_SIG_RE = re.compile(r"^CFGSIG n=(\d+) crc=([0-9a-f]+)")
FAIL, BAD, SATURATED = 65535, 65534, 65533  # 65533 = 655.33 us or more (clamped): not usable


def load_cfg(module_path):
    """Execute the pure-python CFG block of the measure file; returns its namespace."""
    src = Path(module_path).read_text()
    a, b = src.index("# ---- CFG BEGIN"), src.index("# ---- CFG END")
    ns = {}
    exec(compile(src[a:b], str(module_path), "exec"), ns)
    return ns


def parse_log(text, module_path=DEFAULT_MODULE, force=False):
    """-> dict shape_id -> {"base": us?, cfg: us, ...}, plus meta under the key '_meta' only in parse_full."""
    return parse_full(text, module_path, force)["data"]


def parse_full(text, module_path=DEFAULT_MODULE, force=False):
    text = text.strip()
    if text.startswith("{"):
        text = json.loads(text).get("log_tail", "")
    ns = load_cfg(module_path)
    cfg_by_shape = {ns["shape_id"](l, m): ns["shape_cfgs"](l, m) for l, m in ns["all_shapes"]()}
    data, meta = {}, {"sig_ok": None, "counts": {}, "order": []}
    for line in text.splitlines():
        line = line.rstrip()
        sm = _SIG_RE.match(line)
        if sm:
            n, crc = ns["cfg_sig"]()
            ok = int(sm.group(1)) == n and int(sm.group(2), 16) == crc
            meta["sig_ok"] = ok
            if not ok and not force:
                raise SystemExit(f"CFGSIG mismatch: log n={sm.group(1)} crc={sm.group(2)} vs {Path(module_path).name} "
                                 f"n={n} crc={crc:08x} (use the file of the commit that produced the log, or --force)")
            continue
        tm = _T_RE.match(line)
        if tm:
            sid, b64 = tm.group(1), tm.group(2)
            cfgs = cfg_by_shape.get(sid)
            if cfgs is None:
                continue
            raw = base64.b64decode(b64)
            vals = struct.unpack(f"<{len(raw) // 2}H", raw)
            if len(vals) != len(cfgs):
                if not force:
                    raise SystemExit(f"T64 {sid}: {len(vals)} values but the table has {len(cfgs)} configs")
            d = data.setdefault(sid, {})
            for c, v in zip(cfgs, vals):
                if 0 < v < SATURATED:
                    d[c[0]] = v / 100.0
            continue
        bm = _B_RE.match(line)
        if bm:
            layer, m = bm.group(1), int(bm.group(2))
            sid = f"{layer}.m{m}"
            d = data.setdefault(sid, {})
            if bm.group(3) not in ("-", "nan"):
                d["base"] = float(bm.group(3))
            for tok in bm.group(4).split():
                name, _, us = tok.rpartition(":")
                if name:
                    d[name] = float(us)  # finals (51 reps) override the screen value
            meta["counts"][sid] = dict(ok=int(bm.group(5)), fail=int(bm.group(6)), bad=int(bm.group(7)),
                                       skip=int(bm.group(8) or 0))
            meta["order"].append(sid)
    for sid in data:
        if sid not in meta["order"]:
            meta["order"].append(sid)
    return {"data": data, "meta": meta}


def best_per_shape(data, per_family=False):
    rows = []
    for sid, d in data.items():
        cfgs = {k: v for k, v in d.items() if k != "base"}
        if not cfgs:
            continue
        name = min(cfgs, key=cfgs.get)
        row = dict(shape=sid, base=d.get("base"), cfg=name, us=cfgs[name])
        if d.get("base"):
            row["delta_us"] = cfgs[name] - d["base"]
            row["delta_pct"] = 100.0 * (cfgs[name] / d["base"] - 1.0)
        if per_family:
            fam = {}
            for k, v in cfgs.items():
                f = k[0]
                if f not in fam or v < fam[f][1]:
                    fam[f] = (k, v)
            row["families"] = fam
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--module", default=str(DEFAULT_MODULE))
    ap.add_argument("--best", action="store_true", help="best config per shape and its delta vs base")
    ap.add_argument("--per-family", action="store_true", help="with --best: also the best of each family")
    ap.add_argument("--force", action="store_true", help="ignore a CFGSIG / length mismatch")
    a = ap.parse_args(argv)
    text = sys.stdin.read() if a.log == "-" else Path(a.log).read_text()
    full = parse_full(text, a.module, a.force)
    if not a.best:
        json.dump(full["data"], sys.stdout, indent=None, sort_keys=False)
        print()
        return 0
    order = {sid: i for i, sid in enumerate(full["meta"]["order"])}
    for r in sorted(best_per_shape(full["data"], a.per_family), key=lambda r: order.get(r["shape"], 0)):
        base = f"{r['base']:.2f}" if r["base"] is not None else "-"
        delta = f" delta={r['delta_us']:+.2f}us ({r['delta_pct']:+.1f}%)" if "delta_us" in r else ""
        fams = ""
        if a.per_family:
            fams = " | " + " ".join(f"{n}:{v:.2f}" for n, v in r["families"].values())
        print(f"{r['shape']:<14} base={base:>7} best={r['cfg']} {r['us']:.2f}{delta}{fams}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
