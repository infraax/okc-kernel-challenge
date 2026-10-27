#!/usr/bin/env python3
"""Parse the log of a kernels/muse_moe_w4a16/measure_v1.py run: B / T64 lines (via parse_meas), plus the S (stage split), R (routing) and K (per-axis best) lines.

    tools/parse_meas_moe.py LOG [--module kernels/muse_moe_w4a16/measure_v1.py] [--json] [--force]

LOG is the saved .log, an `okc status --json` document (its `log_tail` is used) or `-` for stdin. Default output: per M the baseline (kernel_fn as is), the
stage split, the best config of each axis and the best five configs with their delta vs the baseline; --json dumps the parsed data instead.
Config names: b<block_size_m>.<align a0..a4>.r<GEMM1 reduce><GEMM2 reduce>.<epilogue> (see the measure file's docstring); CFGSIG must match the module (else --force).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_meas  # noqa: E402

DEFAULT_MODULE = Path(__file__).resolve().parent.parent / "kernels" / "muse_moe_w4a16" / "measure_v1.py"
_S_RE = re.compile(r"^S (\S+) m=(\d+) (.*)$")
_R_RE = re.compile(r"^R (\S+) m=(\d+) (.*)$")
_K_RE = re.compile(r"^K (\S+) m=(\d+) (.*)$")
_KV_RE = re.compile(r"(\w+)=([\d.]+)")


def _text(text):
    text = text.strip()
    return json.loads(text).get("log_tail", "") if text.startswith("{") else text


def parse(text, module_path=DEFAULT_MODULE, force=False):
    text = _text(text)
    full = parse_meas.parse_full(text, module_path, force)
    stage, routing, axes = {}, {}, {}
    for line in text.splitlines():
        line = line.rstrip()
        sm = _S_RE.match(line)
        if sm:
            kv = {k: float(v) for k, v in _KV_RE.findall(sm.group(3))}
            if kv:
                stage[f"{sm.group(1)}.m{sm.group(2)}"] = kv
            continue
        rm = _R_RE.match(line)
        if rm:
            routing[f"{rm.group(1)}.m{rm.group(2)}"] = {k: int(float(v)) for k, v in _KV_RE.findall(rm.group(3))}
            continue
        km = _K_RE.match(line)
        if km:
            ax = {}
            for part in km.group(3).split(" | "):
                name, _, rest = part.partition(" ")
                ax[name] = {tok.rpartition(":")[0]: float(tok.rpartition(":")[2]) for tok in rest.split() if ":" in tok}
            axes[f"{km.group(1)}.m{km.group(2)}"] = ax
    return {"data": full["data"], "meta": full["meta"], "stage": stage, "routing": routing, "axes": axes}


def render(res, module_path=DEFAULT_MODULE):
    out = []
    for sid, d in res["data"].items():
        base = d.get("base")
        cfgs = sorted((v, k) for k, v in d.items() if k != "base")
        head = f"{sid:<10} base={base:.2f}" if base is not None else f"{sid:<10} base=-"
        r = res["routing"].get(sid)
        if r:
            head += "  routing " + " ".join(f"{k}={v}" for k, v in r.items())
        out.append(head)
        s = res["stage"].get(sid)
        if s:
            out.append("  stages " + " ".join(f"{k}={v:.2f}" for k, v in s.items()))
        for ax, vals in res["axes"].get(sid, {}).items():
            out.append(f"  best {ax:<3} " + " ".join(f"{k}:{v:.1f}" for k, v in sorted(vals.items(), key=lambda t: t[1])[:6]))
        for v, k in cfgs[:5]:
            delta = f" ({v - base:+.2f} us, {100 * (v / base - 1):+.1f}%)" if base else ""
            out.append(f"  top {k} {v:.2f}{delta}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--module", default=str(DEFAULT_MODULE))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    text = sys.stdin.read() if a.log == "-" else Path(a.log).read_text()
    res = parse(text, a.module, a.force)
    if a.json:
        json.dump({k: res[k] for k in ("data", "stage", "routing", "axes")}, sys.stdout)
        print()
    else:
        print(render(res, a.module))
    return 0


if __name__ == "__main__":
    sys.exit(main())
