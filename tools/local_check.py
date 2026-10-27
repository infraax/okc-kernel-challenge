#!/usr/bin/env python3
"""Local correctness harness: run a candidate submission against the organisers' own
reference / checks (problems/<problem>/reference.py), the way the judge does, before spending a
judge check.

  .venv/bin/python tools/local_check.py <problem> <file> [--device cpu|cuda]
                                        [--cases tests|scored|small|all] [--seed N] [--graph] [--full-weights]

Cases come from problems/<problem>/TASK.md: the "must pass these N tests" block ("tests"), the
scored-shapes table ("scored"), the tests with ctx / m / batch / q_len scaled down so a CPU run
finishes quickly ("small"), or tests + scored ("all"). Per case: generate_input -> (case_fields
for the layer problems) -> prepare(weights) -> kernel_fn -> reference.collect (if any) ->
reference.check_implementation. One line per case "PASS|FAIL <case> <message>", then
"local_check: <problem> <k>/<n> passed". Exit 1 on any failure.

--graph (cuda only): after an eager warm-up, kernel_fn is captured in a CUDA graph and replayed;
the replay's output is the one that is checked. A host sync or value-dependent allocation in
kernel_fn fails the capture.

muse_moe_w4a16 fills only its 32 local experts by default (the 256-expert weights are ~7 GB);
--full-weights restores the reference's own generation.

Never modifies problems/**. The reference is imported here, never by a submission.
"""

import argparse
import copy
import importlib.util
import inspect
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBLEMS = ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer")
_PROBLEM_MODULES = ("checks", "nvfp4", "reference", "submission", "baseline")


# --------------------------------------------------------------------------- case parsing

def _value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def parse_case(text: str) -> dict:
    """'batch 5, ctx 5000, layer sliding, kv_scale 0.015625' -> kwargs for generate_input."""
    kwargs = {}
    for part in text.split(","):
        key, _, val = part.strip().partition(" ")
        if not key or not val:
            raise ValueError(f"cannot parse case field {part!r} in {text!r}")
        kwargs[key] = _value(val.strip())
    return kwargs


def format_case(kwargs: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in kwargs.items())


def parse_tests(task_md: str) -> list[dict]:
    """The block after 'must pass these N tests'."""
    m = re.search(r"must pass these (\d+) tests.*?```\w*\n(.*?)```", task_md, re.S)
    if not m:
        raise ValueError("no 'must pass these N tests' block in TASK.md")
    cases = [parse_case(line) for line in m.group(2).splitlines() if line.strip()]
    if len(cases) != int(m.group(1)):
        raise ValueError(f"TASK.md says {m.group(1)} tests, parsed {len(cases)}")
    return cases


def parse_scored(task_md: str) -> list[dict]:
    """The shape column of the scored-shapes table (| load | shape | weight | floor |), deduplicated."""
    cases, seen = [], set()
    in_table = False
    for line in task_md.splitlines():
        if re.match(r"\|\s*load\s*\|\s*shape\s*\|", line):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or set(cells[1]) <= {"-", ":", " "}:
            continue
        if cells[1] not in seen:
            seen.add(cells[1])
            cases.append(parse_case(cells[1]))
    if not cases:
        raise ValueError("no scored-shapes table in TASK.md")
    return cases


def scale_down(kwargs: dict) -> dict:
    """CPU-sized version of a case. Keeps what the test is about: window overflow (ctx stays above
    2,048 when it was), tiny contexts, m / batch above the 8-token unpooler cut-over, q_len 8,
    mixed spreads, the page size, dist and kv_scale."""
    k = dict(kwargs)
    if isinstance(k.get("ctx"), int) and k["ctx"] > 3000:
        k["ctx"] = 2300 + k["ctx"] % 700  # 2300 .. 2999: still past the sliding window
    if isinstance(k.get("batch"), int) and k["batch"] > 4:
        k["batch"] = 4
    for key in ("m", "bs"):
        if isinstance(k.get(key), int) and k[key] > 40:
            k[key] = 40 + k[key] % 7
    if isinstance(k.get("q_len"), int) and k["q_len"] > 64:
        k["q_len"] = 64 + k["q_len"] % 61
    return k


def select_cases(problem: str, which: str) -> list[dict]:
    task_md = (ROOT / "problems" / problem / "TASK.md").read_text()
    if which == "tests":
        return parse_tests(task_md)
    if which == "small":
        return [scale_down(c) for c in parse_tests(task_md)]
    if which == "scored":
        return parse_scored(task_md)
    if which == "all":
        return parse_tests(task_md) + parse_scored(task_md)
    raise ValueError(f"unknown --cases {which!r}")


# --------------------------------------------------------------------------- loading

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Loaded:
    """Fresh reference + candidate modules for one problem (problem dir on sys.path while importing,
    so `from checks import ...` / `from nvfp4 import ...` resolve to that problem's copies)."""

    def __init__(self, problem: str, file: Path):
        pdir = ROOT / "problems" / problem
        for name in _PROBLEM_MODULES:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(pdir))
        try:
            self.ref = _load("reference", pdir / "reference.py")
            self.cand = _load(f"_candidate_{problem}", file)
        finally:
            sys.path.remove(str(pdir))
        self.problem = problem
        self.has_weights = hasattr(self.ref, "generate_weights")
        params = inspect.signature(self.cand.kernel_fn).parameters
        self.kernel_takes_weights = len(params) >= 2
        if self.kernel_takes_weights != self.has_weights:
            raise ValueError(f"kernel_fn takes {len(params)} argument(s) but the problem "
                             f"{'has' if self.has_weights else 'has no'} weights")


# --------------------------------------------------------------------------- running

def _clone(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    return copy.copy(obj)


def _local_experts(ref) -> int:
    """Experts this GPU computes (ep8) of the problem's N_EXPERTS, or 0 when it has no expert dimension."""
    return ref.LOCAL["ep8"][1] if hasattr(ref, "LOCAL") and hasattr(ref, "N_EXPERTS") else 0


def _lazy_experts(ref):
    """muse_moe_w4a16 holds 256 experts (~7 GB of NVFP4 weights), more than a 4 GB scope or the 2060 can
    hold. ep8 reads only the 32 local ones, so generate_weights is made to fill those and leave the other
    expert rows as untouched torch.empty pages (no RSS until written). Use --full-weights for the real thing."""
    n_total, n_local = ref.N_EXPERTS, _local_experts(ref)
    orig = ref.random_weights

    def random_weights(rows, k, *, generator, device, leading=(), wide_scales=False):
        if tuple(leading) != (n_total,):
            return orig(rows, k, generator=generator, device=device, leading=leading, wide_scales=wide_scales)
        packed, sf, gs = orig(rows, k, generator=generator, device=device, leading=(n_local,),
                              wide_scales=wide_scales)
        import torch
        widen = lambda t: torch.empty((n_total, *t.shape[1:]), dtype=t.dtype, device=t.device)  # noqa: E731
        out = []
        for t in (packed, sf):
            full = widen(t)
            full[:n_local] = t
            out.append(full)
        return out[0], out[1], gs

    ref.random_weights = random_weights


def _clone_weights(weights, ref, lazy: bool):
    """Copy of the weights for the candidate; with lazy experts only the local rows are copied."""
    import torch
    n_total, n_local = getattr(ref, "N_EXPERTS", 0), _local_experts(ref)
    if not lazy:
        return _clone(weights)
    out = {}
    for name, t in weights.items():
        if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[0] == n_total:
            c = torch.empty_like(t)
            c[:n_local] = t[:n_local]
            out[name] = c
        else:
            out[name] = _clone(t)
    return out


def run_case(L: Loaded, kwargs: dict, weights, cand_weights, device: str, seed: int, graph: bool) -> tuple[bool, str]:
    import torch

    case = {**kwargs, "seed": seed, "device": device}
    if hasattr(L.ref, "case_fields"):  # the judge's per-input fields (local_start)
        case.update(L.ref.case_fields(case, weights))
    data = L.ref.generate_input(**case)
    args = (data, cand_weights) if L.has_weights else (data,)
    if graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            L.cand.kernel_fn(*args)  # eager warm-up (compiles)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = L.cand.kernel_fn(*args)
        g.replay()
    else:
        out = L.cand.kernel_fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    raw = L.ref.collect(data, out) if hasattr(L.ref, "collect") else out
    check_args = (data, raw, weights) if L.has_weights else (data, raw)
    ok, msg = L.ref.check_implementation(*check_args)
    return bool(ok), str(msg)


def run(problem: str, file, device: str = "cpu", cases: str = "small", seed: int = 0, graph: bool = False,
        full_weights: bool = False, out=print) -> tuple[int, int]:
    """Returns (passed, total); prints one line per case and the summary line through `out`."""
    import torch

    if problem not in PROBLEMS:
        raise ValueError(f"unknown problem {problem}")
    if graph and device != "cuda":
        raise ValueError("--graph needs --device cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda but no CUDA device")
    case_list = select_cases(problem, cases)
    L = Loaded(problem, Path(file).resolve())
    lazy = problem == "muse_moe_w4a16" and not full_weights
    if lazy:
        _lazy_experts(L.ref)
    passed = 0
    with torch.no_grad():
        weights = cand_weights = None
        if L.has_weights:
            weights = L.ref.generate_weights(L.ref.WEIGHTS_SEED, device=device)
            cand_weights = _clone_weights(weights, L.ref, lazy)  # the candidate may repack in place; the reference keeps the originals
            if hasattr(L.cand, "prepare"):
                cand_weights = L.cand.prepare(cand_weights)
        for kwargs in case_list:
            label = format_case(kwargs)
            try:
                ok, msg = run_case(L, kwargs, weights, cand_weights, device, seed, graph)
            except Exception as e:  # a crash is a failure of that case, not of the harness
                ok, msg = False, f"{type(e).__name__}: {e}"
                if "LOCAL_CHECK_TRACE" in __import__("os").environ:
                    traceback.print_exc()
            passed += ok
            out(f"{'PASS' if ok else 'FAIL'} {label}{' [graph]' if graph else ''} {msg}")
            if device == "cuda":
                torch.cuda.empty_cache()
    out(f"local_check: {problem} {passed}/{len(case_list)} passed")
    return passed, len(case_list)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("problem", choices=PROBLEMS)
    ap.add_argument("file")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--cases", default="tests", choices=["tests", "scored", "small", "all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-weights", action="store_true",
                    help="muse_moe_w4a16: generate all 256 experts (~7 GB; default fills only the 32 local ones)")
    ap.add_argument("--graph", action="store_true", help="cuda only: check the output of a CUDA-graph replay")
    a = ap.parse_args(argv)
    try:
        passed, total = run(a.problem, a.file, a.device, a.cases, a.seed, a.graph, a.full_weights)
    except ValueError as e:
        print(f"local_check: {e}", file=sys.stderr)
        return 2
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
