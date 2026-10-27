#!/usr/bin/env python3
"""Markdown report for TRACE v1 lines (design/breakdown/TRACE-FORMAT.md) from a judge log (or any text containing them).

  parse_trace.py <log file> [--md out.md]

Per shape: an indexed table (pos, kernel, start, dur, gap, stream, % of wall), totals (sum of dur, busy, sum of gaps, head,
tail, wall - span), the 5 largest gaps with the kernel before and after, per-family totals (tools/breakdown.family).
The spec invariants are checked and every violation is printed (and listed at the top of the report); exit code 1 if any.
Lines other than T* lines are ignored. st and o are optional on TK lines (absent = 0). A shape cut by `TE <i> truncated:size`
or listed as `TE <i> dropped:...` is reported as such and not checked for completeness.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from breakdown import family  # noqa: E402

TAGS = ("TV", "TN", "TS", "TK", "TG", "TE", "TZ")


def kv(parts):
    d = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            d[k] = v
    return d


def num(d, key, default=0.0):
    try:
        return float(d.get(key, default))
    except ValueError:
        return float(default)


def parse(text):
    """-> dict(tv=[...], names={id: name}, shapes={idx: {...}}, errors={idx: text}, done=secs|None, bad=[unparsable lines])."""
    out = dict(tv=[], names={}, shapes={}, errors={}, done=None, bad=[])
    for raw in text.splitlines():
        parts = raw.strip().split()
        if not parts or parts[0] not in TAGS:
            continue
        tag = parts[0]
        try:
            if tag == "TV":
                out["tv"].append(dict(problem=parts[2], file=parts[3] if len(parts) > 3 else "", **kv(parts[4:])))
            elif tag == "TN":
                out["names"][parts[1]] = " ".join(parts[2:])
            elif tag == "TS":
                d = kv(parts[2:])
                out["shapes"][parts[1]] = dict(idx=parts[1], w=num(d, "w", 1), wall=num(d, "wall"), span=num(d, "span"),
                                               n=int(num(d, "n")), label=d.get("label", ""), rep=d.get("rep", ""),
                                               kernels=[], tg=None)
            elif tag == "TK":
                shape, pos = parts[1].split(".")
                d = kv(parts[3:])
                out["shapes"][shape]["kernels"].append(dict(
                    pos=int(pos), name_id=parts[2], s=num(d, "s"), d=num(d, "d"), g=num(d, "g"),
                    st=int(num(d, "st")), o=num(d, "o")))
            elif tag == "TG":
                d = kv(parts[2:])
                out["shapes"][parts[1]]["tg"] = {k: num(d, k) for k in ("gaps", "head", "tail", "busy")}
            elif tag == "TE":
                out["errors"][parts[1]] = " ".join(parts[2:])
            elif tag == "TZ":
                out["done"] = float(parts[2])
        except (IndexError, KeyError, ValueError):
            out["bad"].append(raw.strip()[:100])
    return out


def union(segs):
    total, cs, ce = 0.0, None, None
    for s, e in sorted(segs):
        if ce is None or s > ce:
            if ce is not None:
                total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs if ce is not None else 0.0)


def check(tr):
    """-> {shape idx: [violation text]} ('' = file level)."""
    v = {"": []}
    if not tr["tv"]:
        v[""].append("no TV line")
    if tr["done"] is None:
        v[""].append("no TZ line (run cut off or log tail lost the end)")
    for line in tr["bad"]:
        v[""].append(f"unparsable line: {line}")
    for idx, s in tr["shapes"].items():
        errs = v.setdefault(idx, [])
        ks, tg = s["kernels"], s["tg"]
        trunc = "truncated" in tr["errors"].get(idx, "")
        if tg is None and s["n"] > 0:
            errs.append("no TG line")
        if ks and not trunc:
            if [k["pos"] for k in ks] != list(range(len(ks))):
                errs.append("positions are not 0..n-1 in order")
            if s["n"] != len(ks):
                errs.append(f"TS n={s['n']} but {len(ks)} TK lines")
        if any(b["s"] + 0.011 < a["s"] for a, b in zip(ks, ks[1:])):
            errs.append("TK start times are not in start order")
        for k in ks:
            if k["name_id"] not in tr["names"]:
                errs.append(f"TK {s['idx']}.{k['pos']}: name id {k['name_id']} has no TN line")
                break
        if not ks or tg is None:
            continue
        tol = 0.03 * s["span"] + 0.5
        sum_d = sum(k["d"] for k in ks)
        if s["wall"] + max(1.0, 0.03 * s["span"]) < s["span"]:       # wall (events) and span (CUPTI) come from different replays
            errs.append(f"wall {s['wall']:.2f} < span {s['span']:.2f}")
        if not trunc:
            if sum_d + tol < tg["busy"]:
                errs.append(f"sum d {sum_d:.2f} < busy {tg['busy']:.2f}")
            if abs(s["span"] - (tg["busy"] + tg["gaps"])) > tol:
                errs.append(f"span {s['span']:.2f} != busy {tg['busy']:.2f} + gaps {tg['gaps']:.2f} (tol {tol:.2f})")
            if abs(sum(k["g"] for k in ks) - tg["gaps"]) > tol:
                errs.append(f"sum of TK g {sum(k['g'] for k in ks):.2f} != TG gaps {tg['gaps']:.2f}")
            end = max(k["s"] + k["d"] for k in ks)
            if abs(end - s["span"]) > tol:
                errs.append(f"last end {end:.2f} != span {s['span']:.2f}")
            if tg["busy"] > union([(k["s"], k["s"] + k["d"]) for k in ks]) + tol:
                errs.append("busy exceeds the union of the TK intervals")
        if tg["head"] < -0.011 or tg["tail"] < -0.011:
            errs.append(f"negative head/tail ({tg['head']}, {tg['tail']})")
        if any(k["g"] < -0.011 or k["d"] < 0 or k["o"] < 0 for k in ks):
            errs.append("negative g/d/o")
    return {k: e for k, e in v.items() if e}


def report(tr, violations):
    L = ["# TRACE report", ""]
    for t in tr["tv"]:
        L.append("- " + " ".join(f"{k}={x}" for k, x in t.items()))
    if tr["done"] is not None:
        L.append(f"- prepare time {tr['done']:.0f} s")
    L += ["", "## invariants", ""]
    if violations:
        for idx, errs in violations.items():
            for e in errs:
                L.append(f"- VIOLATION {'shape ' + idx if idx else 'file'}: {e}")
    else:
        L.append("- all checks pass")
    drops = {i: e for i, e in tr["errors"].items() if i not in tr["shapes"] or e.startswith(("dropped", "truncated"))}
    if drops:
        L += ["", "Shapes not fully reported: " + ", ".join(f"{i} ({e})" for i, e in sorted(drops.items(), key=lambda t: t[0]))]
    fam_all = {}
    for idx in sorted(tr["shapes"], key=int):
        s = tr["shapes"][idx]
        ks, tg = s["kernels"], s["tg"]
        L += ["", f"## shape {idx}: {s['label']}  (w {s['w']:g}, wall {s['wall']:.2f} us, span {s['span']:.2f} us, n {s['n']}"
              + (f", replays {s['rep']}" if s["rep"] else "") + ")", ""]
        if idx in tr["errors"] and not ks:
            L.append(f"error: {tr['errors'][idx]}")
            continue
        L += ["| pos | kernel | start | dur | gap | stream | % of wall |", "|---|---|---|---|---|---|---|"]
        for k in ks:
            nm = tr["names"].get(k["name_id"], "?")[:70]
            L.append(f"| {idx}.{k['pos']} | `{nm}` | {k['s']:.2f} | {k['d']:.2f} | {k['g']:.2f} | {k['st']} | "
                     f"{100 * k['d'] / s['wall']:.1f}% |" if s["wall"] else f"| {idx}.{k['pos']} | `{nm}` | | | | | |")
        if tg:
            sum_d = sum(k["d"] for k in ks)
            L += ["", f"Totals: sum dur {sum_d:.2f} | busy {tg['busy']:.2f} | sum gaps {tg['gaps']:.2f} | head {tg['head']:.2f} | "
                      f"tail {tg['tail']:.2f} | wall - span {s['wall'] - s['span']:.2f} us"]
        big = sorted((k for k in ks if k["g"] > 0), key=lambda k: -k["g"])[:5]
        if big:
            L += ["", "Largest gaps:", "", "| gap us | after | before the next |", "|---|---|---|"]
            for k in big:
                i = k["pos"]
                prev = tr["names"].get(ks[i - 1]["name_id"], "?")[:50] if i > 0 else "(replay start)"
                L.append(f"| {k['g']:.2f} | `{prev}` ({idx}.{i - 1}) | `{tr['names'].get(k['name_id'], '?')[:50]}` ({idx}.{i}) |")
        fam = {}
        for k in ks:
            f = family(tr["names"].get(k["name_id"], ""))
            fam[f] = fam.get(f, 0.0) + k["d"]
            fam_all[f] = fam_all.get(f, 0.0) + s["w"] * k["d"]
        L += ["", "Families (sum dur): " + ", ".join(f"{f} {v:.2f}" for f, v in sorted(fam.items(), key=lambda t: -t[1]))]
    if fam_all:
        tot = sum(fam_all.values()) or 1.0
        L += ["", "## families, weighted by shape weight (sum dur)", "", "| family | w x us | share |", "|---|---|---|"]
        for f, v in sorted(fam_all.items(), key=lambda t: -t[1]):
            L.append(f"| {f} | {v:.1f} | {100 * v / tot:.1f}% |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    tr = parse(Path(a.log).read_text(errors="replace"))
    violations = check(tr)
    md = report(tr, violations)
    if a.md:
        Path(a.md).write_text(md)
    else:
        print(md)
    for idx, errs in violations.items():
        for e in errs:
            print(f"VIOLATION {'shape ' + idx if idx else 'file'}: {e}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
