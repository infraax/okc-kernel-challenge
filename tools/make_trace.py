#!/usr/bin/env python3
"""Build a MEASUREMENT-ONLY trace file (TRACE v1, design/breakdown/TRACE-FORMAT.md) from one of our kernel files.

  make_trace.py <problem> <kernel_file> [--job JOB] [--replays 20] [--budget-s 200] [--max-bytes 14000] [--out PATH]

Writes kernels/<problem>/trace_<stem>.py =
  line 1   the kernel file's own `# okc: problem=<p>`
  line 2   a `# MEASUREMENT-ONLY ... NEVER submit` marker (tools/okc_run.py refuses to submit such a file)
  rest     the kernel file VERBATIM (kernel_fn, prepare, everything unchanged)
  + generated settings (_TR_*: the judge's case dicts embedded as a literal list)
  + the problem's generate_input and every helper it needs, inlined from problems/<p>/reference.py (and local modules such as
    nvfp4.py) by `ast`, all names prefixed `_tr_` (a submission cannot import the reference)
  + tools/trace_runtime.py verbatim: wraps prepare() (or runs at import when the kernel has no prepare, as muse_attention_decode):
    per judged case it builds inputs, captures kernel_fn in a CUDA graph, flushes L2 read-only, replays R times under
    torch.profiler (CUPTI) and prints the TV/TN/TS/TK/TG/TE/TZ lines. Nothing measured feeds kernel_fn.
Cases come from experiments/profiles/<problem>.<job>.json benchmarks[].case (default job: the best submitted one in the log).
"""
import argparse
import ast
import builtins
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / "experiments" / "profiles"
RUNTIME = Path(__file__).resolve().parent / "trace_runtime.py"
PREFIX = "_tr_"
BUILTINS = set(dir(builtins))


# ============================================================================================ generator inlining
class _Module:
    """One source module: its top-level statements by bound name, and its imports."""

    def __init__(self, name, path):
        self.name, self.path = name, path
        self.tree = ast.parse(path.read_text(), str(path))
        self.defs = {}      # name -> [stmt]
        self.imports = {}   # alias -> ("import", module) | ("from", module, original)
        for st in self.tree.body:
            if isinstance(st, ast.Import):
                for a in st.names:
                    self.imports[a.asname or a.name.split(".")[0]] = ("import", a.name if a.asname else a.name.split(".")[0])
            elif isinstance(st, ast.ImportFrom):
                if st.module == "__future__":
                    continue
                for a in st.names:
                    self.imports[a.asname or a.name] = ("from", st.module, a.name)
            else:
                for n in _bound(st):
                    self.defs.setdefault(n, []).append(st)


def _targets(t):
    if isinstance(t, ast.Name):
        yield t.id
    elif isinstance(t, (ast.Tuple, ast.List)):
        for e in t.elts:
            yield from _targets(e)
    elif isinstance(t, ast.Starred):
        yield from _targets(t.value)


def _bound(st):
    """Names a module-level statement binds."""
    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [st.name]
    if isinstance(st, ast.Assign):
        return [n for t in st.targets for n in _targets(t)]
    if isinstance(st, (ast.AnnAssign, ast.AugAssign)):
        return list(_targets(st.target))
    return []


def _local_names(fn):
    """Names bound inside a function / lambda scope (conservative: comprehension targets count as bound)."""
    args = fn.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if args.vararg:
        names.add(args.vararg.arg)
    if args.kwarg:
        names.add(args.kwarg.arg)
    body = fn.body if isinstance(fn.body, list) else [fn.body]
    glob = set()

    def walk(node):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(ch.name)
                continue          # its own scope
            if isinstance(ch, ast.Lambda):
                continue
            if isinstance(ch, ast.Name) and isinstance(ch.ctx, (ast.Store, ast.Del)):
                names.add(ch.id)
            elif isinstance(ch, (ast.Import, ast.ImportFrom)):
                for a in ch.names:
                    names.add(a.asname or a.name.split(".")[0])
            elif isinstance(ch, ast.ExceptHandler) and ch.name:
                names.add(ch.name)
            elif isinstance(ch, ast.Global):
                glob.update(ch.names)
            walk(ch)

    for st in body:
        if isinstance(st, ast.AST):
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(st.name)
                continue
            if isinstance(st, ast.Name) and isinstance(st.ctx, ast.Store):
                names.add(st.id)
            walk(st)
    return names - glob


class _Scoped(ast.NodeTransformer):
    """Rewrite free references to module-level names through `resolve(name) -> new name | None` (None: leave alone).
    A name bound in an enclosing function is local and never touched."""

    def __init__(self, resolve):
        self.resolve = resolve
        self.scopes = []

    def visit_Name(self, node):
        if any(node.id in s for s in self.scopes):
            return node
        new = self.resolve(node.id)
        if new is not None:
            return ast.copy_location(ast.Name(id=new, ctx=node.ctx), node)
        return node

    def _func(self, node):
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        a = node.args
        a.defaults = [self.visit(d) for d in a.defaults]
        a.kw_defaults = [self.visit(d) if d is not None else None for d in a.kw_defaults]
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            arg.annotation = None
        if hasattr(node, "returns"):
            node.returns = None
        self.scopes.append(_local_names(node))
        node.body = [self.visit(s) for s in node.body]
        self.scopes.pop()
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _func

    def visit_Lambda(self, node):
        a = node.args
        a.defaults = [self.visit(d) for d in a.defaults]
        self.scopes.append(_local_names(node))
        node.body = self.visit(node.body)
        self.scopes.pop()
        return node


class _Inliner:
    def __init__(self, problem_dir, entry_file, entry, reserved, known=None):
        self.dir = problem_dir
        self.mods = {}
        self.taken = set(reserved)
        self.newname = dict(known or {})   # (module, name) | ("ext", module, original) -> new name
        self.stmts = []          # (module order, lineno, new stmt)
        self.exts = {}           # new name -> import statement text
        self.order = []          # module names, dependencies first
        self.seen = set()
        self.visiting = set()
        self.root = self._load(entry_file.stem, entry_file)
        self._visit_module(self.root)
        self.entry_new = self._need(self.root.name, entry)

    def _load(self, name, path):
        if name not in self.mods:
            self.mods[name] = _Module(name, path)
        return self.mods[name]

    def _visit_module(self, mod):
        if mod.name in self.order or mod.name in self.visiting:
            return
        self.visiting.add(mod.name)
        for imp in mod.imports.values():
            if imp[0] == "from" and (self.dir / f"{imp[1]}.py").exists():
                self._visit_module(self._load(imp[1], self.dir / f"{imp[1]}.py"))
        self.order.append(mod.name)

    def _fresh(self, base):
        new = PREFIX + base.lstrip("_")
        i = 2
        while new in self.taken:
            new = f"{PREFIX}{base.lstrip('_')}_{i}"
            i += 1
        self.taken.add(new)
        return new

    def _need(self, modname, name):
        """New name of module-level `name` of `modname`, queueing whatever it depends on."""
        mod = self.mods[modname]
        if name in mod.imports:
            imp = mod.imports[name]
            if imp[0] == "from" and (self.dir / f"{imp[1]}.py").exists():
                return self._need(imp[1], imp[2])
            key = ("ext",) + imp[1:]
            if key not in self.newname:
                new = self._fresh(name)
                self.newname[key] = new
                if imp[0] == "import":
                    self.exts[new] = f"import {imp[1]} as {new}"
                else:
                    self.exts[new] = f"from {imp[1]} import {imp[2]} as {new}"
            return self.newname[key]
        key = (modname, name)
        if key in self.newname:
            return self.newname[key]
        if name not in mod.defs:
            raise KeyError(f"{modname}.{name}: not defined at module level of {mod.path.name} (inside if/try?)")
        new = self._fresh(name)
        self.newname[key] = new
        for st in mod.defs[name]:
            if (modname, id(st)) in self.seen:
                continue
            self.seen.add((modname, id(st)))
            self._emit(mod, st)
        return new

    def _emit(self, mod, st):
        line = st.lineno
        st = copy.deepcopy(st)

        def resolve(n):
            if n in mod.defs or n in mod.imports:
                return self._need(mod.name, n)
            if n not in BUILTINS:
                raise KeyError(f"{mod.name}: unresolved name {n!r} in {ast.unparse(st)[:60]!r}")
            return None

        scoped = _Scoped(resolve)
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new = scoped.visit(st)
            new.name = self.newname[(mod.name, st.name)]
        elif isinstance(st, ast.ClassDef):
            raise KeyError(f"{mod.name}: class {st.name} needed by the generator (not supported)")
        else:
            if isinstance(st, ast.AnnAssign):
                if st.value is None:
                    return
                st = ast.Assign(targets=[st.target], value=st.value, lineno=st.lineno)
            if isinstance(st, ast.Assign):
                # targets are Stores of module names: rename them through the same map
                for t in st.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            n.id = self._need(mod.name, n.id)
                st.value = scoped.visit(st.value)
                new = st
            else:
                new = scoped.visit(st)
        self.stmts.append((self.order.index(mod.name), line, new))

    def source(self):
        self.stmts.sort(key=lambda t: (t[0], t[1]))
        out = [f"# --- inlined from problems/<p>/*.py by tools/make_trace.py (names prefixed {PREFIX}) ---"]
        out += [self.exts[k] for k in sorted(self.exts)]
        for _, _, st in self.stmts:
            out.append(ast.unparse(ast.fix_missing_locations(st)))
        return "\n".join(out) + "\n"


def inline_generator(problem_dir, reserved=(), known=None):
    """-> (source text, new name of generate_input, parameter names of generate_input)."""
    ref = Path(problem_dir) / "reference.py"
    inl = _Inliner(Path(problem_dir), ref, "generate_input", reserved, known)
    fn = inl.root.defs["generate_input"][0]
    params = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs]
    return inl.source(), inl.entry_new, params


# ============================================================================================ kernel file facts
def kernel_facts(src):
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "kernel_fn" not in fns:
        raise SystemExit("make_trace: the kernel file defines no kernel_fn")
    nargs = len(fns["kernel_fn"].args.args)
    if nargs not in (1, 2):
        raise SystemExit(f"make_trace: kernel_fn takes {nargs} arguments (expected 1 or 2)")
    return dict(has_prepare="prepare" in fns, kfn_args=nargs)


def default_job(problem):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import breakdown
    jobs = breakdown.best_jobs()
    if problem not in jobs:
        raise SystemExit(f"make_trace: no submitted job for {problem} in experiments/log.jsonl; pass --job")
    return jobs[problem]


def load_cases(problem, job):
    path = PROFILES / f"{problem}.{job}.json"
    if not path.exists():
        raise SystemExit(f"make_trace: {path} missing")
    d = json.loads(path.read_text())
    return [b["case"] for b in sorted(d["benchmarks"], key=lambda b: b["index"])]


# ============================================================================================ assembly
def build(problem, kernel_file, job=None, replays=20, budget_s=200, max_bytes=14000, cases=None):
    kernel_file = Path(kernel_file)
    src = kernel_file.read_text()
    first = src.splitlines()[0]
    if first.strip() != f"# okc: problem={problem}":
        raise SystemExit(f"make_trace: line 1 of {kernel_file} must be '# okc: problem={problem}'")
    if PREFIX in src:
        raise SystemExit(f"make_trace: {kernel_file} already uses the {PREFIX!r} prefix")
    facts = kernel_facts(src)
    if cases is None:
        job = job or default_job(problem)
        cases = load_cases(problem, job)
    runtime = RUNTIME.read_text()
    reserved = {n for st in ast.parse(runtime).body for n in _bound(st)} | {
        a.asname or a.name for st in ast.parse(runtime).body if isinstance(st, ast.Import) for a in st.names}
    known = {("ext", "import", a.name): a.asname for st in ast.parse(runtime).body if isinstance(st, ast.Import)
             for a in st.names if a.asname}      # the runtime block's own `import torch as _tr_torch` is reused
    gen_src, gen_name, params = inline_generator(ROOT / "problems" / problem, reserved, known)
    assert gen_name == "_tr_generate_input", gen_name
    ls = "local_start" in params
    try:
        rel = kernel_file.resolve().relative_to(ROOT)
    except ValueError:
        rel = kernel_file
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    marker = (f"# MEASUREMENT-ONLY trace of {rel} (sha256 {digest}), generated by tools/make_trace.py: NEVER submit. "
              f"Everything below is the kernel file verbatim; the trace block is appended at the end.")
    lines = src.splitlines(keepends=True)
    body = lines[0] + marker + "\n" + "".join(lines[1:])
    if not body.endswith("\n"):
        body += "\n"
    settings = [
        "", "",
        "# ===================================== generated by tools/make_trace.py (trace settings) =====================================",
        f"_TR_PROBLEM = {problem!r}",
        f"_TR_FILE = {kernel_file.name!r}",
        f"_TR_JOB = {job!r}",
        f"_TR_REPLAYS = {int(replays)}",
        f"_TR_BUDGET_S = {float(budget_s)!r}",
        f"_TR_MAX_BYTES = {int(max_bytes)}",
        f"_TR_HAS_PREPARE = {facts['has_prepare']}",
        f"_TR_KFN_ARGS = {facts['kfn_args']}",
        f"_TR_LS_FROM_IDS = {ls}",
        "_TR_CASES = [",
    ] + [f"    {c!r}," for c in cases] + ["]", "", gen_src, "", runtime]
    out = body + "\n".join(settings)
    compile(out, str(kernel_file), "exec")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("kernel_file")
    ap.add_argument("--job")
    ap.add_argument("--replays", type=int, default=20)
    ap.add_argument("--budget-s", type=float, default=200)
    ap.add_argument("--max-bytes", type=int, default=14000)
    ap.add_argument("--out")
    a = ap.parse_args()
    kf = Path(a.kernel_file)
    text = build(a.problem, kf, a.job, a.replays, a.budget_s, a.max_bytes)
    out = Path(a.out) if a.out else ROOT / "kernels" / a.problem / f"trace_{kf.stem}.py"
    out.write_text(text)
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
