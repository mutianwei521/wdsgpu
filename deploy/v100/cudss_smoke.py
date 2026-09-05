# -*- coding: utf-8 -*-
"""cuDSS smoke on this GPU: nvmath DirectSolver on a 2-D Laplacian (SPD, CSR int32, f64), explicitly batched like GGASolver."""
import sys, time, torch, numpy as np, scipy.sparse as sp
sys.stdout.reconfigure(encoding="utf-8")
import nvmath
from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions
from nvmath.bindings.cudss import MatrixType
print("nvmath", nvmath.__version__, "torch", torch.__version__, torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
m = 100                                   # 100x100 grid -> n=10000, nnz=49600
L = sp.diags([-1, -1, 4, -1, -1], [-m, -1, 0, 1, m], shape=(m * m, m * m)).tocsr()
L = L + 1e-3 * sp.eye(m * m)
L = sp.csr_matrix(L.astype(np.float64))
n, nnz = L.shape[0], L.nnz
B = 8
ip = torch.as_tensor(L.indptr, dtype=torch.int32).cuda()
ii = torch.as_tensor(L.indices, dtype=torch.int32).cuda()
vals = torch.as_tensor(L.data, dtype=torch.float64).cuda().repeat(B, 1).contiguous()
rng = torch.Generator().manual_seed(0)
rhs = torch.rand(B, n, dtype=torch.float64, generator=rng).cuda()
a_list = [torch.sparse_csr_tensor(ip, ii, vals[i], size=(n, n)) for i in range(B)]
b_list = [rhs[i] for i in range(B)]
opts = DirectSolverOptions(sparse_system_type=MatrixType.SPD)
t0 = time.perf_counter()
s = DirectSolver(a_list, b_list, options=opts)
s.plan(); torch.cuda.synchronize(); t1 = time.perf_counter()
s.factorize(); torch.cuda.synchronize(); t2 = time.perf_counter()
x = s.solve(); torch.cuda.synchronize(); t3 = time.perf_counter()
X = torch.stack([torch.as_tensor(xi) for xi in x])
res = max((torch.as_tensor(a_list[i] @ X[i]) - rhs[i]).abs().max().item() for i in range(B))
xd = torch.linalg.solve(torch.as_tensor(L.toarray()).cuda(), rhs.T).T   # dense f64 check
err = (X - xd).abs().max().item()
print(f"n={n} nnz={nnz} B={B} plan {1e3*(t1-t0):.1f} ms, factorize {1e3*(t2-t1):.1f} ms, solve {1e3*(t3-t2):.1f} ms")
print(f"max|Ax-b| {res:.3e}  max|x-x_dense| {err:.3e}")
s.free()
print("CUDSS_OK" if res < 1e-9 and err < 1e-7 else "CUDSS_BAD")
