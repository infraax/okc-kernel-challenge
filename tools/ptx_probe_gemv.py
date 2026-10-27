import re, sys, importlib.util, triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
spec = importlib.util.spec_from_file_location("k", "kernels/muse_bf16_proj/director_v1b.py"); k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
fn = k._gemv_splitk
sig = {"x_ptr":"*bf16","w_ptr":"*bf16","wlo_ptr":"*bf16","out_ptr":"*fp32","M":"i32","N":"i32","K":"i32","stride_xm":"i32","stride_wn":"i32","stride_om":"i32",
       "BM":"constexpr","BN":"constexpr","BK":"constexpr","SPLIT":"constexpr","HAS_LO":"constexpr","ATOMIC":"constexpr"}
for name,(BM,BN,BK,SPLIT,W,ST) in {"router m1":(16,64,128,16,4,4),"pooler m1":(16,32,512,1,4,4),"router m256":(64,32,128,8,4,3)}.items():
    consts={"BM":BM,"BN":BN,"BK":BK,"SPLIT":SPLIT,"HAS_LO":True,"ATOMIC":True}
    src = ASTSource(fn=fn, signature=sig, constexprs=consts)
    c = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"num_warps":W,"num_stages":ST})
    ptx = c.asm["ptx"]
    cnt = lambda p: len(re.findall(p, ptx))
    print(f"{name:12s} stages={ST}: cp.async={cnt(r'cp\.async\.(?:ca|cg)')} cp.async.bulk={cnt(r'cp\.async\.bulk')} wgmma={cnt(r'wgmma')} mma.sync={cnt(r'mma\.sync')} ld.global={cnt(r'ld\.global')} shared={c.metadata.shared}")
